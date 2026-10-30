"""Focused stdlib-only reading of closed native extension resource evidence."""
import ctypes
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/parakeet_turkish"
LOOP = OUT / "runs/adaptive_20261011_native_memory_fix_01"
RUN = "adaptive_20261011_native_memory_fix_01_public_winner_extension"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def owned_pid(pid):
    if os.name != "nt":
        return {"pid": pid, "available": False, "reason": "native Windows only"}
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    api.OpenProcess.restype = ctypes.c_void_p
    api.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    api.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = api.OpenProcess(0x1000, False, pid)
    if not handle:
        error = ctypes.get_last_error()
        return {"pid": pid, "available": True, "present": False if error == 87 else None, "OpenProcess_error": error, "scope": "Only this previously owned PID; no enumeration"}
    try:
        value = ctypes.c_ulong()
        okay = api.GetExitCodeProcess(handle, ctypes.byref(value))
        return {"pid": pid, "available": bool(okay), "running": value.value == 259 if okay else None, "exit_code": value.value if okay else None, "scope": "Only this previously owned PID; no enumeration"}
    finally:
        api.CloseHandle(handle)


def compact(sample):
    system = sample["system_memory"]
    return {"at_utc": sample.get("observed_at_utc", system.get("observed_at_utc")), "phase": sample.get("phase"), "update": sample.get("eligible_update_call"), "system_commit_used_bytes": system["commit_used_bytes"], "system_commit_limit_bytes": system["commit_limit_bytes"], "system_headroom_bytes": system["commit_available_bytes"], "system_commit_fraction": system["commit_fraction"], "worker_private_bytes": sample["worker_private_memory"]["private_bytes"], "self_GMSEx_headroom_bytes": sample.get("calling_process_commit_memory", {}).get("ullAvailPageFile_bytes")}


def main():
    memory_path = LOOP / (RUN+"_memory.json")
    observations_path = LOOP / (RUN+"_memory_observations.jsonl")
    cache_path = LOOP / (RUN+"_cache.json")
    state_path = LOOP / "loop_status.json"
    memory, cache, state = [json.loads(p.read_text(encoding="utf-8")) for p in (memory_path, cache_path, state_path)]
    observations = [json.loads(line) for line in observations_path.read_text(encoding="utf-8").splitlines()]
    trial = next(t for t in state["trials"] if t["run_id"] == RUN)
    assert state["status"] == "resource_review_required" and trial["exit_code"] == 0 and trial["completed_steps"] == 358
    updates = memory["owned_update_memory_telemetry"]["samples"]
    critical = next(s for s in updates if s.get("eligible_update_call") == 354)
    assert critical["worker_private_memory"]["pid"] == memory["worker_pid"] == 29744
    assert critical["system_memory"]["commit_available_bytes"] == critical["calling_process_commit_memory"]["ullAvailPageFile_bytes"] < 2_000_000_000
    event = next(e for e in cache["cache_reclamation"]["events"] if e["eligible_update_call"] == 354)
    assert event["before"]["allocated_bytes"] == event["after"]["allocated_bytes"]
    assert event["after"]["reserved_bytes"] < event["before"]["reserved_bytes"]
    stop = dt.datetime.fromisoformat(trial["cooperative_stop_requested_at_utc"])
    parent = min(observations, key=lambda r: abs((dt.datetime.fromisoformat(r["at_utc"]) - stop).total_seconds()))
    used_self = parent["worker_self"]["sample"]
    baseline = memory["telemetry_after_torch_import"]
    after = memory["telemetry_after_worker"]
    cleanup = memory["native_cleanup_evidence"]
    report = {"schema": 1, "kind": "native_memory_fix_01_extension_stop_diagnosis_v1", "CPU_only": True, "Torch_NeMo_CUDA_or_checkpoint_loads": False, "passed": True, "run_id": RUN, "worker_pid": 29744, "launcher_pid": 1116, "closed_state": {"controller_status": state["status"], "controller_stop_reason": trial["cooperative_stop_reason"], "requested_at_utc": trial["cooperative_stop_requested_at_utc"], "child_exit_code": trial["exit_code"], "completed_steps": 358, "selected_step": trial["selected_step"], "history_sha256": trial["candidate"]["provenance"]["history_sha256"]}, "root_owned_exit_observation": {"source": "trusted parent task message; durable receipt pending", "session_id": 40244, "exit_code": 1, "observed_chunk_id": "91771a", "not_independently_repolled": True}, "critical_exact_update_sample": compact(critical), "controller_nearest_parent_poll": {"at_utc": parent["at_utc"], "parent_system": parent["memory"]["system_memory"], "used_worker_self_sample": compact(used_self), "self_sample_age_seconds": parent["worker_self"].get("age_seconds")}, "same_update_CUDA_cleanup": event, "same_update_reserved_bytes_reclaimed": event["before"]["reserved_bytes"]-event["after"]["reserved_bytes"], "previous_update_samples": [compact(s) for s in updates[-6:-1]], "startup_sample": compact(baseline), "after_worker_sample": compact(after), "recovery_after_critical": {"private_bytes_drop": critical["worker_private_memory"]["private_bytes"]-after["worker_private_memory"]["private_bytes"], "system_commit_used_bytes_drop": critical["system_memory"]["commit_used_bytes"]-after["system_memory"]["commit_used_bytes"], "headroom_after_bytes": after["system_memory"]["commit_available_bytes"]}, "native_cleanup_summary": {k:v for k,v in cleanup.items() if k != "events"}, "owned_process_status_now": [owned_pid(29744), owned_pid(1116)], "diagnosis": ["The exact worker sample is a genuine system-wide transient commit breach: GetPerformanceInfo and same-call worker GlobalMemoryStatusEx report identical headroom; no process/job quota discrepancy is shown.", "Update354 retains2.844GB active CUDA allocation but6.363GB reserved before ordinary endcleanup. Reclamation removes3.467GB reserved without changing active allocation. Private/system commit fall during later DEV/exit, consistent with a large transient allocator footprint.", "This evidence does not attribute every host byte to CUDA, NumPy, Numba, LLVM or a Python leak; global commit includes other processes and polling is sparse.", "The750MB per-clip adapter cleanup successfully ran3901times but cannot release active official-loss buffers. Split-before-FP32 adapter3 removes a redundant full joint copy; it needs the root's actual GPU gradient/resource proof before adoption.", "Critical95%/2GB guards remain valid; reducing active allocations or increasing measured system commitment headroom is appropriate, not waiving the guard."], "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in (memory_path, observations_path, cache_path, state_path)}, "no_original_files_changed": True}
    destination = OUT / "native_memory_fix_01_extension_stop_diagnosis_v1.json"
    assert not destination.exists()
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": True, "critical": report["critical_exact_update_sample"], "reclaimed_cuda_reserved_bytes": report["same_update_reserved_bytes_reclaimed"], "recovery": report["recovery_after_critical"], "processes": report["owned_process_status_now"]}))


if __name__ == "__main__":
    main()
