from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUTPUT=ROOT/'outputs/parakeet_turkish'
target=OUTPUT/'resource_watchdog_v2.py'
if target.exists():raise FileExistsError(target)
source=(OUTPUT/'resource_watchdog_v1.py').read_text(encoding='utf-8')
source=source.replace('Owned-worker Win32 telemetry and a bounded system-commit guard.',
    'System-wide GetPerformanceInfo commit guard with distinct calling-process headroom.')
source=source.replace('from ctypes import wintypes','from ctypes import wintypes\nfrom paired_windows_commit_telemetry_v1 import performance_info,global_memory_status')
start=source.index('def _system_memory():')
end=source.index('\ndef _worker_memory(',start)
source=source[:start]+'''def _system_memory():
    # PERFORMANCE_INFORMATION counts pages committed by the entire system.
    # MEMORYSTATUSEX pagefile values may be constrained to the calling process.
    result = performance_info()
    if not result.get("available"):
        return result
    result["API"] = "GetPerformanceInfo CommitTotal/CommitLimit times PageSize; system-wide"
    result["scope"] = "System-wide commit, not process or job available commitment"
    total=result["physical_total_bytes"]; available=result["physical_available_bytes"]
    result["physical_load_percent"] = 100 * (total-available)/total if total else None
    return result


def calling_process_commit_memory():
    result=global_memory_status()
    result.update(pid=os.getpid(), scope="Calling process available commitment; never substitute for system committed memory")
    return result


def worker_process_commit_reason(observation, expected_worker_pid):
    """Only a fresh self-reading from the actual owned worker can signal quota risk.

    No ratio derived from total-minus-available is treated as system usage.
    A low process limit is distinct from global commit pressure. Caller must
    validate freshness/run identity when consuming a sidecar from disk.
    """
    if not observation or not observation.get("available") or observation.get("pid") != expected_worker_pid:
        return None
    value=observation.get("ullAvailPageFile_bytes")
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
        return None
    return "actual_worker_process_commit_headroom_below2GB" if value<POLICY["critical_headroom_bytes"] else None

''' +source[end:]
source=source.replace('system = _system_memory()\n    worker =', 'system = _system_memory()\n    process_available = calling_process_commit_memory()\n    worker =')
source=source.replace('worker.update(observed_at_utc=stamp)', 'worker.update(observed_at_utc=stamp)\n    process_available.update(observed_at_utc=stamp)')
source=source.replace('return {"system_memory": system, "worker_private_memory": worker, "phase": phase,',
    'return {"system_memory": system, "worker_private_memory": worker, "calling_process_commit_memory": process_available, "phase": phase,')
target.write_text(source,encoding='utf-8')
print(target)
