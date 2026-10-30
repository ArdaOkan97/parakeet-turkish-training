"""Preserve an actually closed normalization window by streaming byte copies."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/parakeet_turkish"
RUN_ID = "adaptive_20261014_TR_uniform_token_population_01"
RUN = OUT / "runs" / RUN_ID
parser = argparse.ArgumentParser()
parser.add_argument("--loop-dir", type=Path, required=True)
parser.add_argument("--actual-owned-session", type=int, required=True)
parser.add_argument("--actual-owned-exit-chunk", required=True)
parser.add_argument("--actual-owned-exit-code", type=int, choices=(0, 1), default=0)
parser.add_argument("--reviewed-closure", type=Path)
args = parser.parse_args()

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

loop = args.loop_dir.resolve()
assert loop.parent == (OUT / "runs").resolve() and loop.name.startswith(RUN_ID + "_loop_")
lease_path = ROOT / "work/gpu_training_owner.json"
lease = read(lease_path)
assert lease["target_run_id"] == RUN_ID and lease["actual_controller_session_id"] == args.actual_owned_session
status, selection = read(loop / "loop_status.json"), read(RUN / "selection.json")
if args.actual_owned_exit_code == 1:
    assert args.reviewed_closure is not None
    sys.path.insert(0, str(OUT))
    import run_TSC_uniform_TR_loss_journal_v2 as runtime
    status = runtime.load_previous(argparse.Namespace(previous_status=loop / "loop_status.json", reviewed_closure=args.reviewed_closure))
else:
    assert args.reviewed_closure is None
assert status["status"] == "train_window_ended" and status["child_exit_code"] == 0
assert status["resource_review_required"] is False and status["run_id"] == selection["run_id"] == RUN_ID
assert status["stop_category"] in ("budget", "configured_steps_complete", "phase_bound")
step = status["completed_steps"]
assert selection["completed_steps"] == step and 25109 < step <= 43238
completion_path = Path(status["actual_runtime_completion"]["runtime_completion"]["path"])
completion = read(completion_path)
assert completion["status"] == "complete" and completion["sources_unchanged"] is True
cycle = selection["Turkish_domain_coverage"]["matched_cycle"]
assert cycle["local_updates"] == step - 25109
if step == 43238:
    assert cycle["cycle_unique"] == 253798 and cycle["full_cycle_complete"] and cycle["tail_draws"] == 10
else:
    assert status["stop_category"] == "budget" and not cycle["full_cycle_complete"]
source_receipt = read(lease["actual_source_receipt"]["path"])
best_step = selection["selected_step"]
summary_path = OUT / "results" / (RUN_ID + "_step_" + str(step)) / "summary.json"
best_summary_path = (OUT / "results" / (RUN_ID + "_step_" + str(best_step)) / "summary.json"
                     if best_step > 25109 else Path(source_receipt["artifacts"]["source_best_DEV_summary"]["path"]))
summary, best_summary = read(summary_path), read(best_summary_path)
refs = {"tr": 825, "en": 1045, "es": 1289, "it": 1158, "fr": 1072, "de": 1004, "pt": 1095}
limits = {"en": 69, "es": 52, "it": 30, "fr": 74, "de": 70, "pt": 66}
counts = {l: summary["validation_" + l]["asr"] for l in refs}
assert all(counts[l]["reference_words"] == refs[l] and counts[l]["utterances"] == 50 for l in refs)
assert all(best_summary["validation_" + l]["asr"]["word_errors"] <= b for l, b in limits.items())
runtime_ready = Path(status["paths"].get("fresh_runtime", status["paths"]["native"]))
if not runtime_ready.exists():
    runtime_ready = Path(status["paths"]["native"])
archive = ROOT / "work/closed-sources" / ("TSC_uniform_TR_token_population" + str(step) + "_closed_v1")
assert not archive.exists()
paths = {
    "latest_checkpoint": Path(selection["latest_weights"]), "best_checkpoint": Path(selection["weights"]),
    "run_config": RUN / "run_config.json", "training_history": RUN / "training_history.json",
    "selection": RUN / "selection.json", "controller_status": loop / "loop_status.json",
    "initial_source_equivalence": RUN / "majority_initial_source_equivalence.json",
    "original_native_ready": Path(status["paths"]["native"]), "runtime_native_ready": runtime_ready,
    "runtime_completion": completion_path, "owned_worker_identity": Path(status["paths"]["identity"]),
    "final_DEV_summary": summary_path, "best_DEV_summary": best_summary_path,
}
archive.mkdir(parents=True)
artifacts = {}
for key, source in paths.items():
    before = record(source)
    target = archive / (key + source.suffix)
    shutil.copyfile(source, target)
    after = record(target)
    assert before["sha256"] == after["sha256"] and before["bytes"] == after["bytes"] and record(source) == before
    artifacts[key] = {"original": before, "archived": after, "byte_identical": True}
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
receipt = {
    "schema": 1, "kind": "TSC_TR_token_population_actual_CLOSED_preservation_v1", "passed": True,
    "observed_at_utc": now, "run_id": RUN_ID, "source_step": 25109, "completed_step": step,
    "actual_owned_exit": {"session_id": args.actual_owned_session, "exit_code": args.actual_owned_exit_code, "tool_chunk": args.actual_owned_exit_chunk},
    "actual_wrapper_failure_review": record(args.reviewed_closure) if args.reviewed_closure else None,
    "original_status_archived_unchanged": True,
    "native_child_exit_code": 0, "native_sources_unchanged": True, "stop_category": status["stop_category"],
    "actual_monitored_retention_counter": selection["consecutive_retention_failures"],
    "selected_qualified_BEST_step": best_step, "selected_qualified_BEST_TR_WER": selection["turkish_validation_wer"],
    "counts": counts, "summary": record(summary_path), "best_summary": record(best_summary_path),
    "matched_cycle": cycle, "artifacts": artifacts, "source350_repeated_on_exact": False,
    "all207_tensor_state_audit_deferred_to_actual_native_restore": True, "byte_only_archive": True,
    "no_scientific_counter_RNG_LR_or_sampler_reset": True, "pagefile_or_guard_changes": False, "TEST_used": False,
}
receipt_path = OUT / ("TR_token_population_actual_CLOSED" + str(step) + "_preservation_v1.json")
assert not receipt_path.exists()
write(receipt_path, receipt)
rr = record(receipt_path)
next_action = "compare_completed_matched_loss_control" if step == 43238 else "fresh_exact_same_ID_continuation"
lease.update(status="actual_TR_token_population_CLOSED" + str(args.actual_owned_exit_code) + "_native_complete_" + next_action, updated_at_utc=now,
    running=False, actual_GPU_running=False, actual_exit_code=args.actual_owned_exit_code, actual_closed_tool_chunk=args.actual_owned_exit_chunk,
    actual_completed_step=step, actual_encoder_schedule_updates=step - 25109,
    current_matched_cycle_unique=cycle["cycle_unique"], current_matched_cycle_complete=cycle["full_cycle_complete"],
    actual_retention_counter=selection["consecutive_retention_failures"], actual_latest_closed_receipt=rr,
    source_for_next_exact_window=artifacts["latest_checkpoint"]["archived"], reserved=True, GPU_reserved=True)
write(lease_path, lease)
plan_path = OUT / "experiment_plan.json"
plan = read(plan_path)
passed = all(counts[l]["word_errors"] <= b for l, b in limits.items())
for key in ("current_training", "actual_current_training"):
    plan[key].update(status=lease["status"], actual_GPU_running=False, current_frontier=step,
        local_encoder_schedule_updates=step - 25109, cycle2_unique_including_carried=cycle["cycle_unique"],
        full_matched_cycle_complete=cycle["full_cycle_complete"], last_complete_DEV_step=step,
        last_DEV_counts=counts, last_complete_DEV_TR_WER=counts["tr"]["wer"], last_DEV_TR_WER=counts["tr"]["wer"],
        last_DEV_TR_word_errors=counts["tr"]["word_errors"], last_DEV_all_six_passed=passed,
        actual_retention_counter=selection["consecutive_retention_failures"], actual_CLOSED_receipt=rr)
plan.update(actual_GPU_running=False, updated_at_utc=now,
    status_authority=f"TR normalization actual outer CLOSED{args.actual_owned_exit_code}/{args.actual_owned_exit_chunk}, native0 at {step:,}, complete DEV TR{counts['tr']['word_errors']}/825, qualified BEST {best_step:,}; cycle2 unique {cycle['cycle_unique']:,}/253,798. Any outer phase-verifier failure is separately reviewed and preserved. Next: {next_action}. All state and metrics preserved; no retention stop, reset, pagefile change or TEST use.")
write(plan_path, plan)
print(json.dumps({"passed": True, "receipt": rr, "step": step, "latest": artifacts["latest_checkpoint"]["archived"],
                  "BEST": best_step, "counter": selection["consecutive_retention_failures"]}))
