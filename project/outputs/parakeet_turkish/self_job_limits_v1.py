"""Read-only Windows limits for the calling process's immediate job.

No external PID, job discovery, process enumeration, Torch/CUDA, settings,
job assignment or limit mutation. A sidecar process's job is not evidence of
another process's job. Nested ancestor limits may be stricter and are not read.
"""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


PROCESS_MEMORY_FLAG = 0x100
JOB_MEMORY_FLAG = 0x200


class BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD)]


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in
                ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                 "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t)]


def decode_limits(value):
    flags = int(value.BasicLimitInformation.LimitFlags)
    process_active = bool(flags & PROCESS_MEMORY_FLAG)
    job_active = bool(flags & JOB_MEMORY_FLAG)
    return {"limit_flags": flags, "limit_flags_hex": hex(flags),
            "process_memory_limit_active": process_active,
            "job_memory_limit_active": job_active,
            "process_memory_limit_bytes": int(value.ProcessMemoryLimit) if process_active else None,
            "job_memory_limit_bytes": int(value.JobMemoryLimit) if job_active else None,
            "peak_process_memory_used_bytes": int(value.PeakProcessMemoryUsed),
            "peak_job_memory_used_bytes": int(value.PeakJobMemoryUsed),
            "limits_scope": "Immediate current job; flags gate validity of the limit fields",
            "ancestor_nested_job_limits_measured": False,
            "current_job_commit_usage_measured": False}


def sample_current_job_limits():
    result = {"schema": 1, "kind": "owned_self_immediate_job_limits_v1",
              "pid": os.getpid(), "ppid": os.getppid(),
              "observed_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Calling process only; current immediate job; no external PID or handles",
              "read_only": True, "CUDA_or_Torch_imported_by_helper": False,
              "job_or_global_settings_changed": False,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "pointer_bits": ctypes.sizeof(ctypes.c_void_p) * 8,
              "struct_sizes": {"BASIC_LIMIT_INFORMATION": ctypes.sizeof(BASIC_LIMIT_INFORMATION),
                               "IO_COUNTERS": ctypes.sizeof(IO_COUNTERS),
                               "EXTENDED_LIMIT_INFORMATION": ctypes.sizeof(EXTENDED_LIMIT_INFORMATION)}}
    if os.name != "nt":
        return {**result, "available": False, "reason": "Windows native API required"}
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.argtypes = []; kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
    kernel.IsProcessInJob.restype = wintypes.BOOL
    kernel.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int,
        ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
    kernel.QueryInformationJobObject.restype = wintypes.BOOL
    # GetCurrentProcess is a pseudo-handle, does not require CloseHandle.
    in_job = wintypes.BOOL()
    if not kernel.IsProcessInJob(kernel.GetCurrentProcess(), None, ctypes.byref(in_job)):
        return {**result, "available": False, "stage": "IsProcessInJob",
                "error": ctypes.get_last_error()}
    result["is_in_job"] = bool(in_job.value)
    if not in_job.value:
        return {**result, "available": True, "immediate_job_limits": None}
    value = EXTENDED_LIMIT_INFORMATION(); returned = wintypes.DWORD()
    # NULL hJob queries only this calling process's immediate job, documented API.
    if not kernel.QueryInformationJobObject(None, 9, ctypes.byref(value),
            ctypes.sizeof(value), ctypes.byref(returned)):
        return {**result, "available": False, "stage": "QueryInformationJobObject",
                "error": ctypes.get_last_error()}
    if returned.value and returned.value != ctypes.sizeof(value):
        return {**result, "available": False, "stage": "ReturnedSize",
                "returned_bytes": int(returned.value)}
    return {**result, "available": True, "immediate_job_limits": decode_limits(value),
            "API": "IsProcessInJob(GetCurrentProcess,NULL); QueryInformationJobObject(NULL,9)",
            "official_sources": [
                "https://learn.microsoft.com/en-us/windows/win32/api/jobapi2/nf-jobapi2-queryinformationjobobject",
                "https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_extended_limit_information"]}


def main(args):
    result = sample_current_job_limits()
    if args.output:
        root = Path(__file__).resolve().parents[2]
        path = args.output.resolve()
        if not path.is_relative_to(root):
            raise ValueError("Output must be inside the workspace")
        if path.exists():
            raise FileExistsError("Existing evidence is immutable")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    main(parser.parse_args())
