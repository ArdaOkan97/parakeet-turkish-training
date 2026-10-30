"""Byte-preserve the actually closed full head-LR cycle; no tensor loading."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/parakeet_turkish"
RUN_ID = "adaptive_20261013_TR_uniform_headlr3e5_cosine_01"
RUN = OUT / "runs" / RUN_ID
LOOP = OUT / "runs" / (RUN_ID + "_loop_01_window_3_journal_01")
ARCHIVE = ROOT / "work/closed-sources/TSC_uniform_headlr3e5_cosine43238_complete_v1"
parser = argparse.ArgumentParser()
parser.add_argument("--actual-owned-exit-chunk", required=True)
args = parser.parse_args()
assert args.actual_owned_exit_chunk.isalnum()

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def record(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}

def write(path, value):
    path = Path(path)
    temp = path.with_suffix(".metadata.tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)

status = read(LOOP / "loop_status.json")
selection = read(RUN / "selection.json")
assert status["status"] == "train_window_ended"
assert status["child_exit_code"] == 0 and status["resource_review_required"] is False
assert status["completed_steps"] == selection["completed_steps"] == 43238
assert status["stop_category"] in ("phase_bound", "configured_steps_complete")
completion_path = Path(status["actual_runtime_completion"]["runtime_completion"]["path"])
completion = read(completion_path)
assert completion["status"] == "complete" and completion["sources_unchanged"] is True
assert selection["actual_head_lr"] == 3e-5
cycle = selection["Turkish_domain_coverage"]["matched_cycle"]
assert cycle["local_updates"] == 18129 and cycle["cycle_unique"] == 253798
assert cycle["full_cycle_complete"] is True and cycle["tail_draws"] == 10
selected_step = selection["selected_step"]
limits = {"en": 69, "es": 52, "it": 30, "fr": 74, "de": 70, "pt": 66}
refs = {"tr": 825, "en": 1045, "es": 1289, "it": 1158, "fr": 1072, "de": 1004, "pt": 1095}
summary_path = OUT / "results" / (RUN_ID + "_step_43238") / "summary.json"
best_summary_path = OUT / "results" / (RUN_ID + "_step_" + str(selected_step)) / "summary.json"
summary, best_summary = read(summary_path), read(best_summary_path)
counts = {lang: summary["validation_" + lang]["asr"] for lang in refs}
best_counts = {lang: best_summary["validation_" + lang]["asr"] for lang in refs}
assert all(counts[l]["reference_words"] == refs[l] and counts[l]["utterances"] == 50 for l in refs)
assert all(best_counts[l]["reference_words"] == refs[l] and best_counts[l]["word_errors"] <= limits[l] for l in limits)
assert selection["turkish_validation_wer"] == best_counts["tr"]["wer"]
assert not ARCHIVE.exists()
ARCHIVE.mkdir(parents=True)
paths = {
    "latest_checkpoint": Path(selection["latest_weights"]), "best_checkpoint": Path(selection["weights"]),
    "run_config": RUN / "run_config.json", "training_history": RUN / "training_history.json",
    "selection": RUN / "selection.json", "controller_status": LOOP / "loop_status.json",
    "initial_source_equivalence": RUN / "majority_initial_source_equivalence.json",
    "original_native_ready": Path(status["paths"]["native"]),
    "runtime_native_ready": Path(status["paths"]["fresh_runtime"]),
    "runtime_completion": completion_path, "owned_worker_identity": Path(status["paths"]["identity"]),
    "final_DEV_summary": summary_path, "best_DEV_summary": best_summary_path,
}
artifacts = {}
for key, source in paths.items():
    before = record(source)
    target = ARCHIVE / (key + source.suffix)
    shutil.copyfile(source, target)
    after = record(target)
    assert before["sha256"] == after["sha256"] and before["bytes"] == after["bytes"]
    artifacts[key] = {"original": before, "archived": after, "byte_identical": True}
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
receipt = {
    "schema": 1, "kind": "TSC_uniform_head_LR_actual_CLOSED_fullcycle_preservation_v1", "passed": True,
    "created_at_utc": now, "run_id": RUN_ID,
    "actual_owned_exit": {"session_id": 97558, "exit_code": 0, "tool_chunk": args.actual_owned_exit_chunk},
    "completed_step": 43238, "source_step": 25109, "native_child_exit_code": 0,
    "stop_category": status["stop_category"], "native_sources_unchanged": True,
    "retention_counter_preserved": selection["consecutive_retention_failures"],
    "selected_qualified_BEST_step": selected_step, "selected_qualified_BEST_TR_WER": best_counts["tr"]["wer"],
    "counts": counts, "best_counts": best_counts, "summary": record(summary_path),
    "best_summary": record(best_summary_path), "matched_cycle": cycle, "artifacts": artifacts,
    "next_action": "Compare completed matched head policies, then admit the prepared TR-only normalization control.",
    "CPU_checkpoint_tensor_audit_performed": False, "byte_archive_scope": True,
    "counter_RNG_coverage_or_LR_reset": False, "pagefile_or_guard_changes": False, "TEST_used": False,
    "objective_complete": False,
}
receipt_path = OUT / "matched_headlr3e5_cosine_actual_CLOSED43238_fullcycle_preservation_v1.json"
assert not receipt_path.exists()
write(receipt_path, receipt)
rr = record(receipt_path)
lease_path = ROOT / "work/gpu_training_owner.json"
lease = read(lease_path)
lease.update(status="actual_head_fullcycle_CLOSED0_preparing_measured_TR_loss_control", updated_at_utc=now,
             running=False, actual_GPU_running=False, actual_exit_code=0,
             actual_closed_tool_chunk=args.actual_owned_exit_chunk, actual_completed_step=43238,
             current_matched_cycle_unique=253798, current_matched_cycle_complete=True,
             actual_head_fullcycle_closure=rr, reserved=True, GPU_reserved=True)
write(lease_path, lease)
plan_path = OUT / "experiment_plan.json"
plan = read(plan_path)
passed = all(counts[l]["word_errors"] <= limits[l] for l in limits)
for key in ("current_training", "actual_current_training"):
    plan[key].update(status=lease["status"], actual_GPU_running=False, current_frontier=43238,
        last_complete_DEV_step=43238, last_DEV_counts=counts, last_complete_DEV_TR_WER=counts["tr"]["wer"],
        last_DEV_TR_WER=counts["tr"]["wer"], last_DEV_TR_word_errors=counts["tr"]["word_errors"],
        last_DEV_all_six_passed=passed, cycle2_unique_including_carried=253798, local_encoder_schedule_updates=18129,
        actual_retention_counter=selection["consecutive_retention_failures"], actual_CLOSED_receipt=rr,
        current_branch_qualified_BEST_step=selected_step, current_branch_qualified_BEST_TR_WER=best_counts["tr"]["wer"])
plan.update(actual_GPU_running=False, updated_at_utc=now,
    status_authority=f"Head LR 3e-5 with cosine completed 18,129 matched updates at 43,238, actual CLOSED0/{args.actual_owned_exit_chunk}; cycle2 covers all 253,798 unique clips plus 10 next-cycle visits. Strict-qualified BEST {selected_step:,}: TR {best_counts['tr']['word_errors']}/825. Completed head-policy comparison and the prepared TR-only normalization control are next; TEST remains frozen.")
write(plan_path, plan)
print(json.dumps({"passed": True, "closure": rr, "selected_step": selected_step,
                  "counts": {l: v["word_errors"] for l, v in counts.items()},
                  "best_counts": {l: v["word_errors"] for l, v in best_counts.items()}}))
