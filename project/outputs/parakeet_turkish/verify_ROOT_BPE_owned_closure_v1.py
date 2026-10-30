"""CPU-only closure check for ROOT's explicitly authorized paired BPE screens."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from atomic_io_retry_v1 import write_json, read_json
from tsc_bpe_full_decoder_transition_v1 import ROOT, STORAGE, record, checked, require


def verify(args):
    allowed = {
        "adaptive_20261015_TR_BPE512_full_decoder_02": "bpe512",
        "adaptive_20261015_TR_original8199_full_decoder_01": "original8199",
    }
    require(args.run_id in allowed and args.observed_exit == 0,
            "Only an actual owned successful paired-screen tool closure is eligible")
    base = args.base_output_dir.resolve()
    require(base.is_relative_to(STORAGE.resolve()) and base.name == args.run_id + "_loop_01",
            "Exact E-only controller namespace required")
    candidates = [base, *base.parent.glob(base.name + "_window_*")]
    completed = []
    for folder in candidates:
        path = folder / "loop_status.json"
        if not path.is_file():
            continue
        value = read_json(path)
        if value.get("run_id") == args.run_id and value.get("completed_step") == 64250:
            completed.append((path, value))
    require(len(completed) == 1, "One actually completed full-budget controller required")
    path, state = completed[0]
    require(state["status"] == "train_window_ended" and state["child_exit_code"] == 0
            and state["stop_category"] == "configured_steps_complete"
            and state["source_step"] == 62250 and state["configured_end_step"] == 64250
            and state["decoder_arm"] == allowed[args.run_id]
            and state["GPU_running"] is False and state["actual_completion"]["passed"] is True,
            "Actual native/controller closure and declared 2000 updates must all agree")
    proof = state["actual_completion"]
    for key in ("native_completion", "worker_completion", "latest_checkpoint", "best_checkpoint",
                "selection", "run_config", "initial_source350", "final_DEV"):
        checked(proof[key])
    native = read_json(proof["native_completion"]["path"])
    worker = read_json(proof["worker_completion"]["path"])
    require(native["status"] == "complete" and native["sources_unchanged"] is True
            and worker["decoder_updates"] == 2000 and worker["final350_complete"] is True
            and worker["frozen192_unchanged"] is True and worker["sources_unchanged"] is True
            and worker["mature_Adam_clock"] == 64250
            and worker["fresh_row_Adam_clock"] == (2000 if allowed[args.run_id] == "bpe512" else None),
            "Actual final DEV, native closure, frozen encoder and optimizer clocks required")
    lease = read_json(ROOT / "work/gpu_training_owner.json")
    require(lease["owner"] == "/root" and lease["running"] is False
            and lease["actual_GPU_running"] is False and lease["actual_child_exit_code"] == 0
            and lease["target_run_id"] == args.run_id,
            "ROOT sole-owner lease must confirm this actual child is closed")
    control = read_json(STORAGE / "analysis/BPE_full_decoder_entry_v1/preflight_original8199_v4.json")
    require(control["source_sha256"] == state["source_sha256"]
            and control["source_checkpoint"] == state["source_checkpoint"],
            "Both paired arms must retain the same frozen sources and coherent starting checkpoint")
    result = {
        "kind": "ROOT_BPE_screen_actual_owned_CLOSED0_v1", "passed": True,
        "at_utc": datetime.now(timezone.utc).isoformat(), "run_id": args.run_id,
        "observed_owned_session": args.observed_session, "observed_tool_exit_code": args.observed_exit,
        "observed_tool_chunk": args.observed_tool_chunk,
        "controller_status": record(path), "completed_step": 64250, "decoder_updates": 2000,
        "actual_completion": proof, "same_frozen_paired_sources": True,
        "not_quality_goal_completion": True,
    }
    target = STORAGE / "analysis/BPE_full_decoder_entry_v1" / (args.run_id + "_actual_CLOSED64250_v1.json")
    require(not target.exists(), "Never replace an actual closure receipt")
    write_json(target, result)
    print(json.dumps({"passed": True, "run_id": args.run_id, "receipt": record(target)}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--base-output-dir", type=Path, required=True)
    p.add_argument("--observed-session", type=int, required=True)
    p.add_argument("--observed-exit", type=int, required=True)
    p.add_argument("--observed-tool-chunk", required=True)
    verify(p.parse_args())
