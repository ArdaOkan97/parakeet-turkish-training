"""Thread-local Windows sleep request while one owned run is making progress."""
import argparse
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not args.run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.run_id):
        raise ValueError("Invalid run identity")
    run_dir = ROOT / "outputs/parakeet_turkish/runs" / args.run_id
    telemetry = run_dir / "training_telemetry.jsonl"
    if os.name != "nt" or not (run_dir / "resolved_trial_config.json").is_file() or not telemetry.is_file():
        raise ValueError("An actual owned Windows training run is required")
    receipt = ROOT / "work" / ("keep_awake_" + args.run_id + ".json")
    stop = ROOT / "work" / ("keep_awake_" + args.run_id + ".stop")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.SetThreadExecutionState.argtypes = [ctypes.c_uint32]
    kernel.SetThreadExecutionState.restype = ctypes.c_uint32
    state = {"run_id": args.run_id, "pid": os.getpid(), "started_at_utc": datetime.now(timezone.utc).isoformat(),
             "system_sleep_request": True, "display_sleep_changed": False,
             "global_power_settings_changed": False, "stale_telemetry_timeout_seconds": 900}
    if not kernel.SetThreadExecutionState(0x80000001):
        raise ctypes.WinError(ctypes.get_last_error())
    receipt.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(state), flush=True)
    reason = "interrupted"
    try:
        while True:
            if stop.exists():
                reason = "owned_stop_file"
                break
            if (run_dir / "trial_outcome.json").exists():
                reason = "owned_run_outcome"
                break
            if time.time() - telemetry.stat().st_mtime > 900:
                reason = "owned_run_telemetry_stale"
                break
            time.sleep(10)
    finally:
        state.update(system_sleep_request=False, reason=reason,
                     ended_at_utc=datetime.now(timezone.utc).isoformat(),
                     request_released=bool(kernel.SetThreadExecutionState(0x80000000)))
        receipt.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(state), flush=True)


if __name__ == "__main__":
    main()
