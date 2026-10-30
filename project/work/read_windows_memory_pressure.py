"""Read-only Win32 process/memory telemetry. No CUDA, GPU context, or mutations."""
import ctypes as c
import json
from ctypes import wintypes as w
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/parakeet_turkish'
PID=32256
kernel=c.WinDLL('kernel32',use_last_error=True)
psapi=c.WinDLL('psapi',use_last_error=True)
report={'schema':1,'observed_at_utc':datetime.now(timezone.utc).isoformat(),'pid':PID,'scope':'Read-only Win32/system/history snapshot; no CUDA imports, compute, permission changes or process mutation','counter_attempt':{'GPU Process Memory':'Get-Counter -ListSet unavailable:0xc0000bc8','GPU Adapter Memory':'Get-Counter -ListSet unavailable:0xc0000bc8'},'psutil_pid_lookup':'NtQuerySystemInformation sandbox lookup returned NoSuchProcess; not independent evidence of host process exit'}
class MemoryStatus(c.Structure):
 _fields_=[('dwLength',w.DWORD),('dwMemoryLoad',w.DWORD)]+[(name,c.c_ulonglong) for name in ['ullTotalPhys','ullAvailPhys','ullTotalPageFile','ullAvailPageFile','ullTotalVirtual','ullAvailVirtual','ullAvailExtendedVirtual']]
kernel.GlobalMemoryStatusEx.argtypes=[c.POINTER(MemoryStatus)];kernel.GlobalMemoryStatusEx.restype=w.BOOL
mem=MemoryStatus();mem.dwLength=c.sizeof(mem)
if kernel.GlobalMemoryStatusEx(c.byref(mem)):
 report['system_memory']={name:getattr(mem,name) for name,_ in MemoryStatus._fields_}
 report['system_memory'].update(physical_used_gb=(mem.ullTotalPhys-mem.ullAvailPhys)/1e9,physical_available_gb=mem.ullAvailPhys/1e9,commit_used_gb=(mem.ullTotalPageFile-mem.ullAvailPageFile)/1e9,commit_limit_gb=mem.ullTotalPageFile/1e9,commit_available_gb=mem.ullAvailPageFile/1e9)
else:report['system_memory']={'error':c.get_last_error()}
class PerformanceInfo(c.Structure):
 _fields_=[('cb',w.DWORD)]+[(name,c.c_size_t) for name in ['CommitTotal','CommitLimit','CommitPeak','PhysicalTotal','PhysicalAvailable','SystemCache','KernelTotal','KernelPaged','KernelNonpaged','PageSize']]+[(name,w.DWORD) for name in ['HandleCount','ProcessCount','ThreadCount']]
psapi.GetPerformanceInfo.argtypes=[c.POINTER(PerformanceInfo),w.DWORD];psapi.GetPerformanceInfo.restype=w.BOOL
perf=PerformanceInfo();perf.cb=c.sizeof(perf)
if psapi.GetPerformanceInfo(c.byref(perf),perf.cb):
 report['system_performance']={name:getattr(perf,name) for name,_ in PerformanceInfo._fields_ if name not in ['HandleCount','ProcessCount','ThreadCount','cb']}
 report['system_performance'].update(commit_total_bytes=perf.CommitTotal*perf.PageSize,commit_limit_bytes=perf.CommitLimit*perf.PageSize,commit_peak_bytes=perf.CommitPeak*perf.PageSize,physical_available_bytes=perf.PhysicalAvailable*perf.PageSize)
else:report['system_performance']={'error':c.get_last_error()}
kernel.OpenProcess.argtypes=[w.DWORD,w.BOOL,w.DWORD];kernel.OpenProcess.restype=w.HANDLE
kernel.CloseHandle.argtypes=[w.HANDLE];kernel.CloseHandle.restype=w.BOOL
kernel.WaitForSingleObject.argtypes=[w.HANDLE,w.DWORD];kernel.WaitForSingleObject.restype=w.DWORD
kernel.GetExitCodeProcess.argtypes=[w.HANDLE,c.POINTER(w.DWORD)];kernel.GetExitCodeProcess.restype=w.BOOL
# QUERY_LIMITED_INFORMATION | SYNCHRONIZE: reads only.
handle=kernel.OpenProcess(0x1000|0x100000,False,PID)
if handle:
 try:
  wait=kernel.WaitForSingleObject(handle,0);exit_code=w.DWORD()
  ok=kernel.GetExitCodeProcess(handle,c.byref(exit_code))
  report['process_direct']={'opened':True,'wait_status':int(wait),'running':wait==0x102,'exit_code':int(exit_code.value) if ok else None,'exit_query_error':None if ok else c.get_last_error()}
 finally:kernel.CloseHandle(handle)
else:report['process_direct']={'opened':False,'error':c.get_last_error(),'running':None,'note':'A denied/unavailable process handle does not establish termination.'}
# Direct PDH query does not enumerate the host. Counter samples remain aggregate
# and filtered to the target PID if the protected Windows counters are exposed.
pdh=c.WinDLL('pdh',use_last_error=True)
pdh.PdhOpenQueryW.argtypes=[w.LPCWSTR,c.c_size_t,c.POINTER(w.HANDLE)];pdh.PdhOpenQueryW.restype=w.LONG
pdh.PdhAddEnglishCounterW.argtypes=[w.HANDLE,w.LPCWSTR,c.c_size_t,c.POINTER(w.HANDLE)];pdh.PdhAddEnglishCounterW.restype=w.LONG
pdh.PdhCollectQueryData.argtypes=[w.HANDLE];pdh.PdhCollectQueryData.restype=w.LONG
pdh.PdhCloseQuery.argtypes=[w.HANDLE];pdh.PdhCloseQuery.restype=w.LONG
query=w.HANDLE();status=pdh.PdhOpenQueryW(None,0,c.byref(query))
report['direct_pdh']={'query_status':hex(status&0xffffffff),'counter_status':{}}
if status==0:
 try:
  for name,path in [('dedicated',r'\GPU Process Memory(*)\Dedicated Usage'),('shared',r'\GPU Process Memory(*)\Shared Usage'),('adapter_dedicated',r'\GPU Adapter Memory(*)\Dedicated Usage'),('adapter_shared',r'\GPU Adapter Memory(*)\Shared Usage')]:
   counter=w.HANDLE();code=pdh.PdhAddEnglishCounterW(query,path,0,c.byref(counter));report['direct_pdh']['counter_status'][name]=hex(code&0xffffffff)
  report['direct_pdh']['collect_status']=hex(pdh.PdhCollectQueryData(query)&0xffffffff)
 finally:pdh.PdhCloseQuery(query)
run=OUT/'runs/adaptive_20261010_recovery_b8_01_b8_quality'
history=json.loads((run/'training_history.json').read_text(encoding='utf-8'))
row=history[-1]
report['last_published_history']={k:row.get(k) for k in ['step','turkish_epoch','train_wall_seconds','process_wall_seconds','audio_seen_seconds','audio_minutes_per_wall_minute','peak_allocated_vram_gb','current_allocated_vram_gb','current_reserved_vram_gb']}
report['history_mtime_utc']=datetime.fromtimestamp((run/'training_history.json').stat().st_mtime,timezone.utc).isoformat()
report['shared_gpu_memory_or_paging_confirmed']=False
report['conclusion']='System physical/commit-pressure telemetry is reported if Win32 exposed it. GPU dedicated/shared counter visibility is unavailable in this sandbox, so shared-GPU-memory usage or actual page-in/page-out rates cannot be confirmed from allocator reservation alone.'
report['limitations']=['PyTorch reserved-memory figure is the caching allocator managed total, not an independent Windows dedicated/shared-GPU classification.','Windows total-page-file fields describe process/system commit capacity, not disk paging traffic. Physical/commit pressure does not by itself prove GPU spill or paging.','No elevated counter/process access was requested; direct handle and PDH error codes are retained.']
destination=OUT/'windows_memory_pressure_observation.json'
destination.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:report[k] for k in ['observed_at_utc','system_memory','system_performance','process_direct','direct_pdh','last_published_history','history_mtime_utc','shared_gpu_memory_or_paging_confirmed']},indent=2))
