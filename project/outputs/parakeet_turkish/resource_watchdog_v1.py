"""Owned-worker Win32 telemetry and a bounded system-commit guard.

Standard library only: no CUDA, process enumeration, command-line inspection,
settings changes, trainer hooks, or model/RNG/optimizer operations. Phase labels
are supplied by the controller and must be described as inferred when inferred.
All byte thresholds use decimal GB. A transient below the critical threshold is
recorded; three distinct five-second high readings request a cooperative stop.
"""
from __future__ import annotations
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import math
import os
import time

POLICY = {"moderate_commit_fraction": .90, "critical_commit_fraction": .95,
          "critical_headroom_bytes": 2_000_000_000, "moderate_consecutive_observations": 3,
          "minimum_fresh_interval_seconds": 5., "telemetry_stale_seconds": 15.}


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD)] + [
        (name, ctypes.c_ulonglong) for name in ("total_phys", "avail_phys", "total_commit",
        "avail_commit", "total_virtual", "avail_virtual", "avail_extended")]


class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
        (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize",
        "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
        "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage", "PrivateUsage")]


def _timestamp():
    return datetime.now(timezone.utc).isoformat()


def _system_memory():
    if os.name != "nt":
        return {"available": False, "reason": "Windows native telemetry required"}
    value = MEMORYSTATUSEX(); value.length = ctypes.sizeof(value)
    function = ctypes.WinDLL("kernel32", use_last_error=True).GlobalMemoryStatusEx
    function.argtypes = [ctypes.POINTER(MEMORYSTATUSEX)]; function.restype = wintypes.BOOL
    if not function(ctypes.byref(value)):
        return {"available": False, "error": ctypes.get_last_error()}
    used = value.total_commit - value.avail_commit
    return {"available": True, "physical_load_percent": int(value.load),
            "physical_available_bytes": int(value.avail_phys), "commit_used_bytes": int(used),
            "commit_limit_bytes": int(value.total_commit), "commit_available_bytes": int(value.avail_commit),
            "commit_fraction": used / value.total_commit if value.total_commit else None,
            "API": "GlobalMemoryStatusEx; system commit/physical status"}


def _worker_memory(owned_pid):
    """Read only the PID supplied by the controller from its owned Popen child."""
    if isinstance(owned_pid, bool) or not isinstance(owned_pid, int) or owned_pid <= 0:
        raise ValueError("A positive owned-worker PID is required")
    base = {"pid": owned_pid, "scope": "Caller-supplied owned process only; no enumeration"}
    if os.name != "nt":
        return {**base, "available": False, "reason": "Windows native telemetry required"}
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    api = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]; kernel.CloseHandle.restype = wintypes.BOOL
    api.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX), wintypes.DWORD]
    api.GetProcessMemoryInfo.restype = wintypes.BOOL
    # PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, with no mutation rights.
    handle = kernel.OpenProcess(0x0400 | 0x0010, False, owned_pid)
    if not handle:
        return {**base, "available": False, "error": ctypes.get_last_error(), "stage": "OpenProcess"}
    try:
        value = PROCESS_MEMORY_COUNTERS_EX(); value.cb = ctypes.sizeof(value)
        if not api.GetProcessMemoryInfo(handle, ctypes.byref(value), value.cb):
            return {**base, "available": False, "error": ctypes.get_last_error(), "stage": "GetProcessMemoryInfo"}
        return {**base, "available": True, "private_bytes": int(value.PrivateUsage),
                "working_set_bytes": int(value.WorkingSetSize),
                "peak_working_set_bytes": int(value.PeakWorkingSetSize),
                "API": "OpenProcess/GetProcessMemoryInfo(PROCESS_MEMORY_COUNTERS_EX)"}
    finally:
        kernel.CloseHandle(handle)


def sample_memory(pid=None, phase=None):
    """Pair system and owned-process readings in one measured sequential call."""
    phase = "controller_poll" if phase is None else phase
    if not isinstance(phase, str) or len(phase) > 128 or any(c in phase for c in ("\n", "\r", "\0")):
        raise ValueError("Use a short phase label, not raw log/command text")
    before = time.monotonic()
    system = _system_memory()
    worker = _worker_memory(os.getpid() if pid is None else pid)
    observed = time.monotonic(); stamp = _timestamp()
    elapsed = observed - before
    system.update(observed_at_utc=stamp, sampling_elapsed_seconds=elapsed)
    worker.update(observed_at_utc=stamp)
    return {"system_memory": system, "worker_private_memory": worker, "phase": phase,
            "observed_monotonic": observed, "observed_at_utc": stamp,
            "sampling_elapsed_seconds": elapsed, "paired_sampling_within_one_second": elapsed <= 1.}


class ResourceGuard:
    """Critical immediate stops plus three fresh moderate-pressure readings."""
    def __init__(self):
        self.moderate_streak = 0
        self._last_counted = None
        self._last_valid = None
        self._last_stamp = None
        self._first_observed = None

    def observe(self, system_memory, worker_private_memory=None, *, observed_monotonic=None):
        observed = time.monotonic() if observed_monotonic is None else observed_monotonic
        if not isinstance(observed, (int, float)) or not math.isfinite(observed):
            raise ValueError("Finite monotonic observation time required")
        if self._first_observed is None:
            self._first_observed = observed
        result = {"system_memory": system_memory, "worker_private_memory": worker_private_memory,
                  "policy": dict(POLICY), "sample_fresh": False, "moderate_commit_streak": self.moderate_streak,
                  "observed_monotonic": observed}
        if not system_memory.get("available"):
            return "memory_telemetry_unavailable", result
        try:
            stamp = datetime.fromisoformat(system_memory["observed_at_utc"])
            if stamp.tzinfo is None:
                raise ValueError("UTC timestamp requires timezone")
            age = (datetime.fromisoformat(_timestamp()) - stamp).total_seconds()
            if age > POLICY["telemetry_stale_seconds"] or age < -1.:
                raise ValueError("UTC telemetry is stale or from the future")
            values = [system_memory[key] for key in ("commit_fraction", "commit_available_bytes",
                                                      "commit_used_bytes", "commit_limit_bytes")]
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
                raise ValueError("Finite commit telemetry required")
            fraction, headroom, used, limit = values
            if limit <= 0 or used < 0 or headroom < 0 or not 0 <= fraction <= 1:
                raise ValueError("Commit telemetry outside physical range")
            if abs((used + headroom) - limit) > 4096 or abs(used / limit - fraction) > 1e-8:
                raise ValueError("Commit telemetry fields disagree")
            if self._last_stamp is not None and stamp <= self._last_stamp:
                raise ValueError("Duplicate or older UTC telemetry")
            if self._last_valid is not None and observed < self._last_valid:
                raise ValueError("Monotonic time moved backwards")
        except (KeyError, TypeError, ValueError) as error:
            result["telemetry_ignored_reason"] = str(error)
            last = self._last_valid if self._last_valid is not None else self._first_observed
            return ("memory_telemetry_stale" if observed - last >= POLICY["telemetry_stale_seconds"] else None), result
        self._last_stamp, self._last_valid = stamp, observed
        if fraction >= POLICY["critical_commit_fraction"]:
            result.update(sample_fresh=True, immediate_critical_stop=True)
            return "system_commit_at_or_above95percent", result
        if headroom < POLICY["critical_headroom_bytes"]:
            result.update(sample_fresh=True, immediate_critical_stop=True)
            return "system_commit_headroom_below2GB", result
        if fraction <= POLICY["moderate_commit_fraction"]:
            self.moderate_streak = 0
        interval = None if self._last_counted is None else observed - self._last_counted
        if interval is None or interval >= POLICY["minimum_fresh_interval_seconds"]:
            self._last_counted = observed
            result["sample_fresh"] = True
            if fraction > POLICY["moderate_commit_fraction"]:
                self.moderate_streak += 1
        else:
            result["telemetry_ignored_reason"] = "Distinct reading inside five-second counting interval"
        result.update(moderate_commit_streak=self.moderate_streak, fresh_interval_seconds=interval)
        reason = ("system_commit_above90percent_three_fresh_observations"
                  if self.moderate_streak >= POLICY["moderate_consecutive_observations"] else None)
        return reason, result
