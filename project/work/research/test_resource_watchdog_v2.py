"""Read-only actual Win32 sample and semantic guard CPU checks."""
import copy
from datetime import datetime,timezone,timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2];OUTPUT=ROOT/'outputs/parakeet_turkish'
sys.path.insert(0,str(OUTPUT))
import resource_watchdog_v2 as m
checks=[];rejects=[]
def good(ok,label):
    if not ok:raise AssertionError(label)
    checks.append(label)
stamp=datetime.now(timezone.utc)
def memory(fraction=.8,index=0,limit=40_000_000_000):
    return {'available':True,'commit_limit_bytes':limit,'commit_used_bytes':int(limit*fraction),
      'commit_available_bytes':limit-int(limit*fraction),'commit_fraction':fraction,'observed_at_utc':(stamp+timedelta(seconds=index*5)).isoformat()}
def observe(guard,value,index=0):
    with patch.object(m,'_timestamp',return_value=value['observed_at_utc']):return guard.observe(value,observed_monotonic=index*5)[0]
mock=memory(.5);mock.update(physical_total_bytes=32_000_000_000,physical_available_bytes=16_000_000_000)
with patch.object(m,'performance_info',return_value=mock),patch.object(m,'global_memory_status',return_value=
    {'available':True,'ullTotalPageFile_bytes':40_000_000_000,'ullAvailPageFile_bytes':1_000_000_000}):
    value=m._system_memory();process=m.calling_process_commit_memory()
good(value['commit_fraction']==.5 and value['commit_available_bytes']==20_000_000_000,'System counters are sourced only from GetPerformanceInfo')
good(process['ullAvailPageFile_bytes']==1_000_000_000 and process['pid']==os.getpid(),'Calling-process headroom is separately attributed')
good(m.worker_process_commit_reason(process,os.getpid())=='actual_worker_process_commit_headroom_below2GB','True self process limit risk retained')
good(m.worker_process_commit_reason(process,os.getpid()+1) is None,'Controller calling quota never assigned to another worker')
good(m.worker_process_commit_reason({'available':True,'pid':1,'ullAvailPageFile_bytes':2_000_000_000},1) is None,'Exact2GB process headroom allowed')
for fraction,reason in [(.95,'system_commit_at_or_above95percent'),(.9,None),(.8,None)]:
    good(observe(m.ResourceGuard(),memory(fraction))==reason,'System fraction boundary'+str(fraction))
low=memory(.5,limit=3_000_000_000)
good(observe(m.ResourceGuard(),low)=='system_commit_headroom_below2GB','System absolute headroom guard')
guard=m.ResourceGuard()
good([observe(guard,memory(.91,i),i) for i in range(3)]==[None,None,'system_commit_above90percent_three_fresh_observations'],'Three fresh sustained observations')
guard=m.ResourceGuard()
good([observe(guard,memory(f,i),i) for i,f in enumerate([.91,.9,.91,.91])]==[None]*4,'Moderate streak reset')
guard=m.ResourceGuard();initial=memory(.91)
good(observe(guard,initial)==observe(guard,initial)==None and guard.moderate_streak==1,'Duplicate telemetry cannot advance streak')
good(m.ResourceGuard().observe({'available':False},observed_monotonic=0)[0]=='memory_telemetry_unavailable','API failure explicit stop')
actual=m.sample_memory(os.getpid(),'actual_CPU_guard_validation')
good(actual['system_memory']['available'] and actual['system_memory']['API'].startswith('GetPerformanceInfo'),'Actual Win32 system API read succeeds')
good(actual['calling_process_commit_memory']['pid']==os.getpid() and actual['worker_private_memory']['pid']==os.getpid(),'Actual self and process-available identity')
good(abs(actual['system_memory']['commit_fraction']-actual['system_memory']['commit_used_bytes']/actual['system_memory']['commit_limit_bytes'])<1e-15,'Actual page arithmetic consistent')
report={'schema':1,'passed':True,'CPU_only':True,'Torch_or_CUDA_imported':False,
  'source_sha256':hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest(),'checks':checks,'actual_sample':actual,
  'paired_helper_sha256':hashlib.sha256((OUTPUT/'paired_windows_commit_telemetry_v1.py').read_bytes()).hexdigest(),
  'primary_sources':['https://learn.microsoft.com/en-us/windows/win32/api/sysinfoapi/ns-sysinfoapi-memorystatusex',
    'https://learn.microsoft.com/en-us/windows/win32/api/psapi/ns-psapi-performance_information']}
(OUTPUT/'resource_watchdog_v2_cpu_verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'passed':True,'checks':len(checks),'source_sha256':report['source_sha256']}))
