"""Bounded live scalar maintenance for the owned TR normalization control."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/parakeet_turkish"
RUN_ID = "adaptive_20261014_TR_uniform_token_population_01"
parser = argparse.ArgumentParser()
parser.add_argument("--DEV-step", type=int, required=True)
parser.add_argument("--milestone", required=True)
args = parser.parse_args()
assert args.milestone.replace("_", "").isalnum()

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}

def write(path, value):
    path = Path(path)
    temp = path.with_suffix(".metadata.tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)

lease_path = ROOT / "work/gpu_training_owner.json"
lease = read(lease_path)
assert lease["target_run_id"] == RUN_ID
status = read(Path(lease["actual_controller_loop"]) / "loop_status.json")
assert status["status"] == "running"
with Path(status["paths"]["log"]).open("rb") as stream:
    stream.seek(max(0, stream.seek(0, 2) - 300000))
    lines = stream.read().decode("utf-8", errors="replace").splitlines()
rows = [json.loads(line) for line in lines if line.startswith('{"step":')]
row = rows[-1]
assert row["step"] >= args.DEV_step > 25109
summary_path = OUT / "results" / (RUN_ID + "_step_" + str(args.DEV_step)) / "summary.json"
summary = read(summary_path)
refs = {"tr": 825, "en": 1045, "es": 1289, "it": 1158, "fr": 1072, "de": 1004, "pt": 1095}
limits = {"en": 69, "es": 52, "it": 30, "fr": 74, "de": 70, "pt": 66}
counts = {lang: summary["validation_" + lang]["asr"] for lang in refs}
assert all(counts[l]["reference_words"] == refs[l] and counts[l]["utterances"] == 50
           and counts[l]["wer"] == counts[l]["word_errors"] / refs[l] for l in refs)
failed = [l for l, bound in limits.items() if counts[l]["word_errors"] > bound]
best_log = next(line for line in reversed(lines) if re.match(r"^dev step\d+ .* retention (True|False) best\d+$", line))
best_step = int(best_log.rsplit("best", 1)[1])
if best_step > 25109:
    best_path = OUT / "results" / (RUN_ID + "_step_" + str(best_step)) / "summary.json"
else:
    assert best_step == 19750
    source = read(lease["actual_source_receipt"]["path"])
    best_path = Path(source["artifacts"]["source_best_DEV_summary"]["path"])
best_summary = read(best_path)
assert all(best_summary["validation_" + l]["asr"]["word_errors"] <= bound for l, bound in limits.items())
best = {"step": best_step, "TR_WER": best_summary["validation_tr"]["asr"]["wer"],
        "TR_word_errors": best_summary["validation_tr"]["asr"]["word_errors"],
        "scope": "Current branch" if best_step > 25109 else "Inherited source qualified BEST19750",
        "summary": record(best_path), "save_completion_stdout": best_log}
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
receipt = {"kind": "TSC_TR_token_population_actual_live_milestone_v1", "passed": True, "observed_at_utc": now,
    "run_id": RUN_ID, "actual_owned_session": lease["actual_controller_session_id"],
    "step": row["step"], "local_updates": row["step"] - 25109,
    "matched_cycle": row["Turkish_domain_coverage"]["matched_cycle"],
    "last_complete_DEV_step": args.DEV_step, "counts": counts, "failed_retention_languages": failed,
    "actual_branch_qualified_BEST": best, "summary": record(summary_path),
    "head_lr": row["head_lr"], "applied_encoder_lr": row["lora_lr"],
    "actual_TR_weights": row["actual_TR_tdt_weights"],
    "unweighted_TDT": row["tdt_loss_unweighted"], "weighted_TDT": row["tdt_loss_weighted"],
    "memory": status["system_and_owned_private_memory"], "TEST_used": False,
    "checkpoint_tensor_read": False, "full_study_complete": False}
receipt_path = OUT / ("TR_token_population_" + args.milestone + "_v1.json")
assert not receipt_path.exists()
write(receipt_path, receipt)
rr = record(receipt_path)
lease.update(updated_at_utc=now, actual_completed_step=row["step"], actual_encoder_schedule_updates=row["step"] - 25109,
    actual_current_window_positive_updates=row["step"] - lease.get("actual_resume_window_start_step", 25109),
    actual_remaining_updates=43238 - row["step"],
    current_matched_cycle_unique=receipt["matched_cycle"]["cycle_unique"], actual_applied_encoder_lr=row["lora_lr"],
    actual_last_complete_DEV_step=args.DEV_step, current_complete_DEV_all_six_passed=not failed,
    current_branch_qualified_BEST=best, actual_current_milestone=rr,
    latest_owned_worker_self_system_commit_observation=status["system_and_owned_private_memory"])
if "actual_retention_counter" in lease:
    lease["restored_source_retention_counter"] = lease.pop("actual_retention_counter")
if best["TR_WER"] < lease["qualified_champion_TR_WER"]:
    lease.update(qualified_champion_step=best_step, qualified_champion_TR_WER=best["TR_WER"],
        qualified_champion_scope="TR token-population branch, strict all-six-qualified saved checkpoint",
        qualified_champion_run_id=RUN_ID, qualified_champion_summary=best["summary"])
write(lease_path, lease)
plan_path = OUT / "experiment_plan.json"
plan = read(plan_path)
for key in ("current_training", "actual_current_training"):
    plan[key].update(current_frontier=row["step"], local_encoder_schedule_updates=row["step"] - 25109,
        actual_GPU_running=True, actual_completed_step=row["step"],
        actual_controller_session_id=lease["actual_controller_session_id"],
        actual_native_launcher_pid=lease["actual_owned_native_launcher_pid"],
        actual_owned_worker_pid=lease["actual_owned_worker_pid"],
        controller_loop=lease["actual_controller_loop"],
        actual_resume_window_start_step=lease["actual_resume_window_start_step"],
        actual_current_window_positive_updates=row["step"] - lease["actual_resume_window_start_step"],
        remaining_updates=43238 - row["step"], own_fullstate_restore_pending=False,
        cycle2_unique_including_carried=receipt["matched_cycle"]["cycle_unique"], applied_encoder_lr=row["lora_lr"],
        last_complete_DEV_step=args.DEV_step, last_DEV_counts=counts, last_complete_DEV_TR_WER=counts["tr"]["wer"],
        last_DEV_TR_WER=counts["tr"]["wer"], last_DEV_TR_word_errors=counts["tr"]["word_errors"],
        last_DEV_all_six_passed=not failed, last_DEV_scope="Actual complete fixed7x50 post-update DEV",
        actual_live_milestone=rr, current_branch_qualified_BEST=best)
    plan[key]["actual_remaining_updates"] = 43238 - row["step"]
    if "actual_retention_counter" in plan[key]:
        plan[key]["restored_source_retention_counter"] = plan[key].pop("actual_retention_counter")
    plan[key].pop("actual_CLOSED_receipt", None)
global_errors = round(lease["qualified_champion_TR_WER"] * 825)
plan.update(updated_at_utc=now, status_authority=f"TR-only token-population control actually running at {row['step']:,} (+{row['step'] - 25109:,}), owned{lease['actual_controller_session_id']}. Complete DEV {args.DEV_step:,}: TR {counts['tr']['word_errors']}/825; strict failures {failed}. Saved branch BEST {best_step:,}/TR{best['TR_word_errors']}; global qualified champion {lease['qualified_champion_step']:,}/TR{global_errors}. Full matched cycle and monitor-only retention continue unchanged; TEST frozen.")
if lease.get("qualified_champion_run_id") == RUN_ID:
    plan["global_qualified_champion"] = {"run_id": RUN_ID, **best}
write(plan_path, plan)
protocol_path = OUT / "post_epoch_experiment_protocol.json"
protocol = read(protocol_path)
for item in protocol["ordered_next_work"]:
    if item["order"] == 4:
        item.update(status="actual_matched_head_cycle_complete_selected_head1e5",
            actual_completion=record(OUT / "matched_headlr3e5_cosine_actual_CLOSED43238_fullcycle_preservation_v1.json"),
            actual_policy_choice=record(OUT / "completed_head_LR_policy_selection_v1.json"),
            measured_result="Qualified BEST TR279 with head1e-5 versus282 with head3e-5. Higher-head final TR275 fails ES/PT; preserve both. One matched pair, no significance claim.")
    elif item["order"] == 5:
        item.update(status="actual_TR_token_population_training_and_DEV_full_matched_cycle_incomplete",
            source_admission=lease["actual_source_receipt"], activation_status="Actual source350/all207/initial save/positive native updates passed",
            actual_source350_positive=record(OUT / "TR_token_population_actual_source350_positive_updates_v1.json"),
            actual_latest_milestone=rr, current_step=row["step"], last_complete_DEV_step=args.DEV_step,
            full_matched_cycle_complete=False, selected_head_lr=1e-5, selected_encoder_policy="cosine")
protocol["updated_at_utc"] = now
write(protocol_path, protocol)
print(json.dumps({"passed": True, "receipt": rr, "step": row["step"], "DEV": args.DEV_step,
                  "TR_errors": counts["tr"]["word_errors"], "failed": failed, "BEST": best_step}))
