"""CPU-only policy, Win32 owned-handle and live own-process telemetry guards."""
import ctypes
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]; OUTPUT=ROOT/'outputs/parakeet_turkish'
sys.path.insert(0,str(OUTPUT))
import resource_watchdog_v1 as module


def main():
    checks=[]
    def require(value,name):
        assert value,name; checks.append(name)
    base=datetime.now(timezone.utc); current=[0.]
    def stamp(seconds):return (base+timedelta(seconds=seconds)).isoformat()
    def memory(fraction=.91,headroom=None,at=None):
        limit=100_000_000_000 if headroom is None else headroom/(1-fraction)
        used=limit*fraction
        return {'available':True,'observed_at_utc':stamp(current[0] if at is None else at),
                'commit_fraction':fraction,'commit_used_bytes':used,'commit_limit_bytes':limit,
                'commit_available_bytes':limit-used}
    def call(guard,seconds,fraction=.91,headroom=None,at=None):
        current[0]=seconds
        return guard.observe(memory(fraction,headroom,at),{'available':False,'pid':123},observed_monotonic=seconds)
    protected=[OUTPUT/name for name in ('train_lora_batched.py','run_experiment_loop_continuation_v1.py',
        'cuda_cache_reclaim_v1.py','launch_with_cache_reclaim_v1.py','launch_with_memory_guard_v2.py')]
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    before={str(path):digest(path) for path in protected}
    with patch.object(module,'_timestamp',side_effect=lambda:stamp(current[0])):
        guard=module.ResourceGuard()
        for seconds,streak in [(0,1),(4.9,1),(5,2)]:
            reason,o=call(guard,seconds)
            require(reason is None and o['moderate_commit_streak']==streak,'Moderate counting onlyfresh5sec '+str(seconds))
        reason,o=call(guard,10)
        require(reason=='system_commit_above90percent_three_fresh_observations' and o['moderate_commit_streak']==3,'Third fresh moderate poll stops')
        guard=module.ResourceGuard();call(guard,0);call(guard,5)
        reason,o=call(guard,10,.90)
        require(reason is None and o['moderate_commit_streak']==0,'Exactly90percent resets')
        require(call(guard,15)[1]['moderate_commit_streak']==1,'High streak starts again after reset')
        for fraction,expected in [(.949999,None),(.95,'system_commit_at_or_above95percent'),(.99,'system_commit_at_or_above95percent')]:
            reason,_=call(module.ResourceGuard(),0,fraction)
            require(reason==expected,'Critical percentage boundary '+str(fraction))
        for headroom,expected in [(2_000_000_000,None),(1_999_999_999,'system_commit_headroom_below2GB')]:
            reason,_=call(module.ResourceGuard(),0,.80,headroom)
            require(reason==expected,'Critical headroom boundary '+str(headroom))
        guard=module.ResourceGuard();call(guard,0,.5)
        require(call(guard,1,.95)[0]=='system_commit_at_or_above95percent','Critical stop doesnotwait five-second cadence')
        guard=module.ResourceGuard();call(guard,0)
        reason,o=call(guard,5,at=0)
        require(reason is None and not o['sample_fresh'] and o['moderate_commit_streak']==1,'Duplicate timestamp cannot advance')
        reason,o=call(guard,15,at=0)
        require(reason=='memory_telemetry_stale','Duplicate telemetry stale after15sec')
        guard=module.ResourceGuard();call(guard,0)
        reason,o=call(guard,5,at=-30)
        require(reason is None and not o['sample_fresh'],'Old absoluteUTC sample ignored')
        reason,o=call(guard,15,at=-30)
        require(reason=='memory_telemetry_stale','Persistent old sample stops')
        guard=module.ResourceGuard();call(guard,0)
        current[0]=5;bad=memory();bad['commit_fraction']=float('nan')
        reason,o=guard.observe(bad,observed_monotonic=5)
        require(reason is None and o['moderate_commit_streak']==1,'Nonfinite cannot advance')
        current[0]=15
        require(guard.observe(bad,observed_monotonic=15)[0]=='memory_telemetry_stale','Persistent invalid data stops')
        bad=memory(.91);bad['commit_available_bytes']+=1_000_000
        reason,o=module.ResourceGuard().observe(bad,observed_monotonic=15)
        require(reason is None and not o['sample_fresh'],'Inconsistent commit fields cannot advance')
        require(module.ResourceGuard().observe({'available':False},observed_monotonic=0)[0]=='memory_telemetry_unavailable','Unavailable safety telemetry stops')
        guard=module.ResourceGuard()
        for seconds,fraction in [(0,.908389488),(5,.751615748)]:reason,o=call(guard,seconds,fraction)
        require(reason is None and o['moderate_commit_streak']==0,'Recorded transient pattern remains recorded without sustained stop')
    class Function:
        def __init__(self,fn):self.fn=fn
        def __call__(self,*args):return self.fn(*args)
    calls=[]
    def info(handle,pointer,size):
        calls.append(('read',handle)); value=pointer._obj
        value.PrivateUsage=45_000_000;value.WorkingSetSize=20_000_000;value.PeakWorkingSetSize=25_000_000
        return True
    kernel=SimpleNamespace(OpenProcess=Function(lambda rights,inherit,pid:calls.append(('open',rights,inherit,pid)) or 123),
                           CloseHandle=Function(lambda handle:calls.append(('close',handle)) or True))
    api=SimpleNamespace(GetProcessMemoryInfo=Function(info))
    with patch.object(module.ctypes,'WinDLL',side_effect=lambda name,**kwargs:kernel if name=='kernel32' else api):
        own=module._worker_memory(4321)
    require(own['private_bytes']==45_000_000 and own['pid']==4321,'Only requestedownedPID andprivate/working memory returned')
    require(calls==[('open',0x410,False,4321),('read',123),('close',123)],'Read-onlyprocessrights and handleclosed exactlyonce')
    calls.clear();api.GetProcessMemoryInfo=Function(lambda *args:False)
    with patch.object(module.ctypes,'WinDLL',side_effect=lambda name,**kwargs:kernel if name=='kernel32' else api):
        own=module._worker_memory(4321)
    require(not own['available'] and calls[-1]==('close',123),'ReadAPI failure closeshandle')
    calls.clear();api.GetProcessMemoryInfo=Function(lambda *args:(_ for _ in ()).throw(RuntimeError('readfailure')))
    with patch.object(module.ctypes,'WinDLL',side_effect=lambda name,**kwargs:kernel if name=='kernel32' else api):
        try:module._worker_memory(4321)
        except RuntimeError:pass
        else:raise AssertionError('exception')
    require(calls[-1]==('close',123),'ReadAPI exception closeshandle')
    for pid in (0,-1,True,'123'):
        try:module._worker_memory(pid)
        except ValueError:checks.append('Invalid ownedPID rejected '+repr(pid))
        else:raise AssertionError('invalidpid')
    live=module.sample_memory(os.getpid(),'CPU_guard_own_process')
    require(live['system_memory']['available'] and live['worker_private_memory']['available'],'ActualnativeWindows APIs readownprocess')
    require(live['worker_private_memory']['pid']==os.getpid() and live['worker_private_memory']['private_bytes']>0,'LiveownedPID private memorypositive')
    require(live['system_memory']['observed_at_utc']==live['worker_private_memory']['observed_at_utc'],'Pairedreadings share timestamp and measuredwindow')
    require(live['paired_sampling_within_one_second'],'Livepairing elapsed under1sec')
    for phase in ('x\nraw',123,'x'*129):
        try:module.sample_memory(os.getpid(),phase)
        except ValueError:checks.append('Raw/unbounded phase label rejected '+repr(phase)[:25])
        else:raise AssertionError('phase')
    for path,old in before.items():require(digest(Path(path))==old,'Protected sourceunchanged '+Path(path).name)
    report={'passed':True,'check_count':len(checks),'checks':checks,'resource_helper_sha256':digest(Path(module.__file__)),
        'policy':module.POLICY,'native_own_process_sample':live,'protected_before_after_sha256':before,
        'scope':'stdlib CPU policy/mocked Win32 handles plus actual ownPID APIs; noTorch/GPU/model or settings operations'}
    target=OUTPUT/'resource_watchdog_v1_cpu_verification.json'
    if target.exists():raise FileExistsError(target)
    target.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'passed':True,'checks':len(checks),'helper_sha256':report['resource_helper_sha256']}))


if __name__=='__main__':main()
