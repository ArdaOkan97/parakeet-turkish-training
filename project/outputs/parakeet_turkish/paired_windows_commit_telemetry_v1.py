"""Read-only paired Win32 process-available and system-wide commit counters.

GlobalMemoryStatusEx page-file fields can be process constrained; only
GetPerformanceInfo CommitTotal/CommitLimit are labeled system commit here.
No Torch/CUDA, process enumeration, command lines, settings or allocations
beyond these small metadata records. Current measurements cannot reconstruct
an unrecorded historical GetPerformanceInfo reading.
"""
from __future__ import annotations
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import time

class MEMORYSTATUSEX(ctypes.Structure):
    _fields_=[('dwLength',wintypes.DWORD),('dwMemoryLoad',wintypes.DWORD)]+[
      (name,ctypes.c_ulonglong) for name in ('ullTotalPhys','ullAvailPhys','ullTotalPageFile','ullAvailPageFile',
        'ullTotalVirtual','ullAvailVirtual','ullAvailExtendedVirtual')]

class PERFORMANCE_INFORMATION(ctypes.Structure):
    _fields_=[('cb',wintypes.DWORD)]+[(name,ctypes.c_size_t) for name in
      ('CommitTotal','CommitLimit','CommitPeak','PhysicalTotal','PhysicalAvailable','SystemCache','KernelTotal',
       'KernelPaged','KernelNonpaged','PageSize')]+[(name,wintypes.DWORD) for name in ('HandleCount','ProcessCount','ThreadCount')]

def timestamp():return datetime.now(timezone.utc).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def performance_info():
    if os.name!='nt':return {'available':False,'reason':'Windows API required'}
    value=PERFORMANCE_INFORMATION();value.cb=ctypes.sizeof(value)
    api=ctypes.WinDLL('psapi',use_last_error=True).GetPerformanceInfo
    api.argtypes=[ctypes.POINTER(PERFORMANCE_INFORMATION),wintypes.DWORD];api.restype=wintypes.BOOL
    if not api(ctypes.byref(value),value.cb):return {'available':False,'error':ctypes.get_last_error()}
    page=int(value.PageSize);used=int(value.CommitTotal)*page;limit=int(value.CommitLimit)*page
    if page<=0 or limit<=0 or not 0<=used<=limit:raise ValueError('Invalid system performance counter values')
    return {'available':True,'API':'GetPerformanceInfo(PERFORMANCE_INFORMATION)','scope':'System-wide commit',
      'page_size_bytes':page,'commit_total_pages':int(value.CommitTotal),'commit_limit_pages':int(value.CommitLimit),
      'commit_peak_pages':int(value.CommitPeak),'commit_used_bytes':used,'commit_limit_bytes':limit,
      'commit_available_bytes':limit-used,'commit_fraction':used/limit,
      'physical_total_bytes':int(value.PhysicalTotal)*page,'physical_available_bytes':int(value.PhysicalAvailable)*page}

def global_memory_status():
    if os.name!='nt':return {'available':False,'reason':'Windows API required'}
    value=MEMORYSTATUSEX();value.dwLength=ctypes.sizeof(value)
    api=ctypes.WinDLL('kernel32',use_last_error=True).GlobalMemoryStatusEx
    api.argtypes=[ctypes.POINTER(MEMORYSTATUSEX)];api.restype=wintypes.BOOL
    if not api(ctypes.byref(value)):return {'available':False,'error':ctypes.get_last_error()}
    total=int(value.ullTotalPageFile);available=int(value.ullAvailPageFile)
    return {'available':True,'API':'GlobalMemoryStatusEx(MEMORYSTATUSEX)',
      'scope':'Current-process available commit; total limit is smaller of system/current process',
      'ullTotalPageFile_bytes':total,'ullAvailPageFile_bytes':available,
      'total_minus_available_bytes':total-available,
      'total_minus_available_fraction':(total-available)/total if total else None,
      'physical_load_percent':int(value.dwMemoryLoad),'physical_total_bytes':int(value.ullTotalPhys),
      'physical_available_bytes':int(value.ullAvailPhys),'virtual_available_bytes':int(value.ullAvailVirtual)}

def paired_sample():
    began=time.monotonic();before=performance_info();process=global_memory_status();after=performance_info();ended=time.monotonic()
    value={'observed_at_utc':timestamp(),'observed_monotonic':ended,'sample_elapsed_seconds':ended-began,
      'get_performance_info_before':before,'global_memory_status_ex':process,'get_performance_info_after':after}
    if all(v.get('available') for v in (before,process,after)):
        headroom=process['ullAvailPageFile_bytes'];low=min(before['commit_available_bytes'],after['commit_available_bytes'])
        high=max(before['commit_available_bytes'],after['commit_available_bytes'])
        used=process['total_minus_available_bytes']
        value['comparison']={'system_limit_equals_process_limited_total':before['commit_limit_bytes']==after['commit_limit_bytes']==process['ullTotalPageFile_bytes'],
          'process_available_minus_latest_system_headroom_bytes':headroom-after['commit_available_bytes'],
          'system_headroom_drift_during_pair_bytes':abs(after['commit_available_bytes']-before['commit_available_bytes']),
          'process_available_inside_paired_system_headroom_bounds':low<=headroom<=high,
          'process_available_below_both_system_headroom_bytes':max(0,low-headroom),
          'legacy_derived_used_minus_system_used_bytes':used-after['commit_used_bytes'],
          'legacy_derived_fraction_minus_system_fraction':process['total_minus_available_fraction']-after['commit_fraction']}
    return value

def main(args):
    if not 1<=args.samples<=60 or not 0<=args.interval_seconds<=5 or args.samples*args.interval_seconds>30:
        raise ValueError('Bounded read-only sample interval required')
    if args.output.exists():raise FileExistsError('Existing observations must remain immutable')
    rows=[]
    for index in range(args.samples):
        if index:time.sleep(args.interval_seconds)
        rows.append(paired_sample())
    valid=[r for r in rows if 'comparison' in r]
    report={'schema':1,'kind':'paired_Windows_process_available_and_system_commit_v1','created_at_utc':timestamp(),
      'source_sha256':sha(__file__),'PID':os.getpid(),'PPID':os.getppid(),'pointer_bits':ctypes.sizeof(ctypes.c_void_p)*8,
      'struct_sizes':{'MEMORYSTATUSEX':ctypes.sizeof(MEMORYSTATUSEX),'PERFORMANCE_INFORMATION':ctypes.sizeof(PERFORMANCE_INFORMATION)},
      'CPU_only':True,'CUDA_or_Torch_imported':False,'scope':'Current calling process and system; no historical inference',
      'primary_sources':[
        'https://learn.microsoft.com/en-us/windows/win32/api/sysinfoapi/ns-sysinfoapi-memorystatusex',
        'https://learn.microsoft.com/en-us/windows/win32/api/psapi/ns-psapi-performance_information'],
      'samples':rows,'summary':{'valid_paired_samples':len(valid),
        'maximum_pair_elapsed_seconds':max(r['sample_elapsed_seconds'] for r in rows),
        'maximum_legacy_minus_system_commit_fraction':max((r['comparison']['legacy_derived_fraction_minus_system_fraction'] for r in valid),default=None),
        'maximum_process_available_below_both_system_headroom_bytes':max((r['comparison']['process_available_below_both_system_headroom_bytes'] for r in valid),default=None)}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'saved':str(args.output),'PID':report['PID'],'summary':report['summary']}))

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=int,default=8)
    p.add_argument('--interval-seconds',type=float,default=.25)
    return p

if __name__=='__main__':main(parser().parse_args())
