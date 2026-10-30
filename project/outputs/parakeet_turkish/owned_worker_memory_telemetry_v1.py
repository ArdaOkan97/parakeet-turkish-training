"""Read-only, owned-lineage memory telemetry; no CUDA initialization or control."""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from resource_watchdog_v1 import sample_memory, _worker_memory
from run_experiment_loop_continuation_v2 import inferred_worker_phase

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parents[1]
CONTROLLER_SHA = "7c0f689c6886f8cbd221acd7e8d6b0e9ab9b9b0157ffe25150bbe1bbe39e8f58"
RESOURCE_SHA = "cc4ca1f11fdef19ac38d2aa878334451a100fd79f9a2e25a5afb4ceef89f0d7c"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def process_table():
    """Internal PID/PPID snapshot only; expose only the owned descendant subtree."""
    require(os.name == "nt", "Windows native process snapshot required")
    class Entry(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
                    ("pid", wintypes.DWORD), ("heap", ctypes.c_size_t),
                    ("module", wintypes.DWORD), ("threads", wintypes.DWORD),
                    ("parent_pid", wintypes.DWORD), ("priority", wintypes.LONG),
                    ("flags", wintypes.DWORD), ("image_name", wintypes.WCHAR * 260)]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel.Process32FirstW.argtypes = kernel.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(Entry)]
    kernel.Process32FirstW.restype = kernel.Process32NextW.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.CreateToolhelp32Snapshot(2, 0)
    require(handle not in (None, ctypes.c_void_p(-1).value), "Native process snapshot failed")
    result = {}
    try:
        entry = Entry(); entry.dwSize = ctypes.sizeof(entry)
        present = kernel.Process32FirstW(handle, ctypes.byref(entry))
        while present:
            result[int(entry.pid)] = {"pid": int(entry.pid), "parent_pid": int(entry.parent_pid),
                                      "image_name": entry.image_name}
            present = kernel.Process32NextW(handle, ctypes.byref(entry))
    finally:
        kernel.CloseHandle(handle)
    return result


def process_identity(pid):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    kernel.GetProcessTimes.restype = wintypes.BOOL
    kernel.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
    kernel.QueryFullProcessImageNameW.restype = wintypes.BOOL
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return {"available": False, "error": ctypes.get_last_error()}
    result = {}
    try:
        created, exited, kernel_time, user = (wintypes.FILETIME() for _ in range(4))
        if kernel.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited), ctypes.byref(kernel_time), ctypes.byref(user)):
            ticks = (int(created.dwHighDateTime) << 32) | int(created.dwLowDateTime)
            result["created_at_utc"] = (datetime(1601, 1, 1, tzinfo=timezone.utc) + timedelta(microseconds=ticks // 10)).isoformat()
        size = wintypes.DWORD(32768); buffer = ctypes.create_unicode_buffer(size.value)
        if kernel.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
            result["image_path"] = buffer.value
        result["available"] = "created_at_utc" in result and "image_path" in result
        return result
    finally:
        kernel.CloseHandle(handle)


def owned_subtree(table, launcher_pid):
    require(launcher_pid in table, "Owned launcher no longer exists")
    owned = {launcher_pid}; frontier = [launcher_pid]
    for _ in range(8):
        children = [pid for pid, item in table.items() if item["parent_pid"] in frontier and pid not in owned]
        if not children:
            break
        require(len(owned) + len(children) <= 32, "Owned subtree exceeds conservative telemetry bound")
        owned.update(children); frontier = children
    return {pid: table[pid] for pid in sorted(owned)}


def gpu_pids():
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=5, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    require(result.returncode == 0, "Read-only GPU PID query failed")
    return {int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()}


def observe(loop_dir):
    started = time.monotonic()
    state_path = loop_dir / "loop_status.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    result = {"observed_at_utc": datetime.now(timezone.utc).isoformat(), "loop_id": state["loop_id"],
              "loop_status": state["status"], "gpu_jobs_launched": False,
              "scope": "Diagnostic sidecar; no training guard, settings, source, or process mutations"}
    active = state.get("active_run_id")
    if not active:
        return {**result, "status": "no_active_owned_worker"}
    require(state["preflight"]["source_sha256"]["run_experiment_loop_continuation_v2.py"] == CONTROLLER_SHA,
            "Recorded controller identity differs")
    trial = next(item for item in reversed(state["trials"]) if item["run_id"] == active)
    launcher = int(trial["pid"])
    table = owned_subtree(process_table(), launcher)
    for pid, entry in table.items():
        entry.update(process_identity(pid))
    launcher_created = datetime.fromisoformat(table[launcher]["created_at_utc"])
    stage_started = datetime.fromisoformat(trial["started_at_utc"])
    require(launcher_created >= stage_started - timedelta(seconds=5), "Launcher PID has stale/reused creation time")
    owned_gpu = sorted(set(table) & gpu_pids())
    result.update(run_id=active, owned_launcher_pid=launcher,
                  owned_lineage=list(table.values()), owned_gpu_context_pids=owned_gpu)
    # A single GPU context inside the validated owned lineage identifies the real worker.
    # Ambiguous/unavailable context is recorded without substituting the tiny launcher.
    selected = owned_gpu[0] if len(owned_gpu) == 1 else None
    history_path = OUTPUT / "runs" / active / "training_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
    phase, phase_evidence = inferred_worker_phase(Path(trial["log"]), history, trial["initial_step"])
    snapshot = sample_memory(selected, phase)
    result.update(status="actual_owned_gpu_worker_identified" if selected is not None else "worker_context_unresolved",
                  actual_worker_pid=selected, launcher_private_memory=_worker_memory(launcher),
                  actual_worker_private_memory=snapshot["worker_private_memory"],
                  system_memory=snapshot["system_memory"], phase=phase_evidence,
                  latest_published_step=history[-1]["step"] if history else trial["initial_step"],
                  sample_elapsed_seconds=time.monotonic()-started,
                  current_controller_private_memory_scope="Popen launcher PID, not necessarily GPU worker; use this sidecar for resolved worker")
    return result


def main(args):
    loop = args.loop_dir.resolve(); target = args.output.resolve()
    require(loop.is_relative_to(OUTPUT / "runs") and target.is_relative_to(ROOT), "Telemetry path leaves workspace")
    require(args.interval >= 5 and 0 < args.max_wall_seconds <= 3 * 3600, "Bounded read-only sampling required")
    require(not target.exists(), "Preserve existing telemetry")
    require(sha(OUTPUT / "run_experiment_loop_continuation_v2.py") == CONTROLLER_SHA
            and sha(OUTPUT / "resource_watchdog_v1.py") == RESOURCE_SHA, "Frozen sources changed")
    identity = {"schema": 1, "kind": "owned_worker_private_memory_sidecar_v1",
                "source_sha256": sha(__file__), "controller_sha256": CONTROLLER_SHA,
                "resource_helper_sha256": RESOURCE_SHA, "gpu_initialization": False,
                "commandlines_read_or_recorded": False, "no_process_or_training_mutation": True}
    target.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.max_wall_seconds
    with target.open("x", encoding="utf-8") as stream:
        while True:
            try:
                snapshot = observe(loop)
            except Exception as error:
                snapshot = {"status": "observation_unavailable", "observed_at_utc": datetime.now(timezone.utc).isoformat(),
                            "error_type": type(error).__name__, "message": str(error)}
            stream.write(json.dumps({**identity, **snapshot}, ensure_ascii=False) + "\n"); stream.flush()
            if args.once or snapshot.get("status") == "no_active_owned_worker" or time.monotonic() + args.interval >= deadline:
                break
            time.sleep(args.interval)
    print(json.dumps({"output": str(target), "gpu_jobs_launched": False}))
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--loop-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--interval", type=float, default=5.)
    p.add_argument("--max-wall-seconds", type=float, default=3 * 3600)
    p.add_argument("--once", action="store_true")
    raise SystemExit(main(p.parse_args()))
