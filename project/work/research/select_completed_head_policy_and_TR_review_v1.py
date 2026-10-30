"""Use actual completed matched head cycles to bind the next TR-loss control."""
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/parakeet_turkish"

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def record(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}

def create(path, value):
    path = Path(path)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return record(path)

high_path = OUT / "matched_headlr3e5_cosine_actual_CLOSED43238_fullcycle_preservation_v1.json"
low_path = OUT / "matched_encoder_cosine_actual_CLOSED43238_fullcycle_preservation_v1.json"
high, low = read(high_path), read(low_path)
assert high["passed"] and high["completed_step"] == 43238 and high["source_step"] == 25109
assert high["actual_owned_exit"]["exit_code"] == high["native_child_exit_code"] == 0
assert high["matched_cycle"]["full_cycle_complete"] and high["matched_cycle"]["cycle_unique"] == 253798
assert low["passed"] and low["completed_steps"] == 43238 and low["source_step"] == 25109
assert low["actual_owned_exit"]["exit_code"] == low["native_child_exit_code"] == 0
assert low["matched_cycle"]["full_cycle_complete"] and low["matched_cycle"]["cycle_unique"] == 253798
low_errors = low["BEST_DEV_counts"]["validation_tr"]["word_errors"]
high_errors = high["best_counts"]["tr"]["word_errors"]
selected_head_lr = 3e-5 if high_errors < low_errors else 1e-5
limits = {"en": 69, "es": 52, "it": 30, "fr": 74, "de": 70, "pt": 66}
assert all(low["BEST_DEV_counts"]["validation_" + l]["word_errors"] <= v for l, v in limits.items())
assert all(high["best_counts"][l]["word_errors"] <= v for l, v in limits.items())
def curve(run_id):
    points = []
    for folder in (OUT / "results").glob(run_id + "_step_*"):
        suffix = folder.name.rsplit("_", 1)[1]
        if not suffix.isdigit():
            continue
        step = int(suffix)
        path = folder / "summary.json"
        if not (25109 < step <= 43238 and path.is_file()):
            continue
        summary = read(path)
        failed = [l for l, bound in limits.items() if summary["validation_" + l]["asr"]["word_errors"] > bound]
        points.append({"step": step, "TR_word_errors": summary["validation_tr"]["asr"]["word_errors"],
                       "strict_all_six_passed": not failed, "failed_retention_languages": failed})
    points.sort(key=lambda point: point["step"])
    assert points[-1]["step"] == 43238
    return {"actual_post_source_DEV_points": len(points),
            "strict_qualified_points": sum(p["strict_all_six_passed"] for p in points),
            "minimum_raw_TR": min(points, key=lambda p: (p["TR_word_errors"], p["step"])),
            "minimum_strict_qualified_TR": min((p for p in points if p["strict_all_six_passed"]), key=lambda p: (p["TR_word_errors"], p["step"])),
            "scalar_curve": points, "checkpoint_or_history_array_read": False}
low_curve, high_curve = curve(low["run_id"]), curve(high["run_id"])
assert low_curve["minimum_strict_qualified_TR"]["TR_word_errors"] == low_errors
assert high_curve["minimum_strict_qualified_TR"]["TR_word_errors"] == high_errors
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
choice = {
    "schema": 1, "kind": "ROOT_completed_head_LR_policy_selection_v1", "passed": True,
    "created_at_utc": now, "decision_actor": "/root/benchmark_design under explicit user and ROOT controlled-quality authorization",
    "actual_curve_evidence_used": True, "full_matched_head_cycle_completed": True,
    "selected_head_lr": selected_head_lr, "selected_encoder_lr_schedule": "cosine",
    "criterion": "Lowest Turkish word-error count among strict all-six-qualified saved checkpoints; ties retain the original lower head rate.",
    "lower_head_LR_completion": record(low_path), "higher_head_LR_completion": record(high_path),
    "arms": [
        {"head_lr": 1e-5, "run_id": low["run_id"], "reused_completed_encoder_cosine_control": True,
         "BEST_step": low["selected_BEST_step"], "qualified_BEST_TR_word_errors": low_errors,
         "qualified_BEST_TR_WER": low_errors / 825,
         "final_TR_word_errors": low["final_DEV_counts"]["validation_tr"]["word_errors"]},
        {"head_lr": 3e-5, "run_id": high["run_id"], "reused_completed_encoder_cosine_control": False,
         "BEST_step": high["selected_qualified_BEST_step"], "qualified_BEST_TR_word_errors": high_errors,
         "qualified_BEST_TR_WER": high_errors / 825, "final_TR_word_errors": high["counts"]["tr"]["word_errors"]},
    ],
    "matched_original_full207_source_step": 25109,
    "matched_original_full207_source_sha256": "3205dff8db1582b2dfec7394332ea02003846ded696b2f7431183a9fc7da7a77",
    "matched_updates": 18129, "matched_cycle_unique": 253798, "fixed_DEV_utterances_per_language": 50,
    "comparison_limits": "One matched training pair with a reused completed lower-head cosine arm and small fixed DEV. Legacy publication cadence differed before 38,539 in the reused arm; no bitwise reproducibility, significance, or general causal superiority is claimed.",
    "all_completed_branch_histories_and_checkpoints_preserved": True, "TEST_used": False,
    "actual_saved_DEV_curve_review": {"lower_head": low_curve, "higher_head": high_curve},
}
choice_path = OUT / "completed_head_LR_policy_selection_v1.json"
choice_record = create(choice_path, choice)
source_review_path = OUT / "TSC_uniform_head_LR25109_USER_review_v1.json"
review = copy.deepcopy(read(source_review_path))
review.update(kind="ROOT_USER_TSC_uniform_TR_loss_control_review_v1", created_at_utc=now,
    decision_actor=choice["decision_actor"], prior_review=record(source_review_path),
    allow_matched_TR_loss_control=True, head_policy_selection_completed=True,
    selected_head_lr=selected_head_lr, head_policy_choice_review=choice_record,
    proposal_status="Actual encoder and head cycles completed; the next TR-only normalization control is authorized and not yet quality-tested.",
    comparison_limits=choice["comparison_limits"],
    matched_control_reuse={"reuse_existing_complete_arm": True,
        "run_id": low["run_id"] if selected_head_lr == 1e-5 else high["run_id"],
        "head_lr": selected_head_lr, "encoder_lr_schedule": "cosine", "tr_tdt_weighting": "clip",
        "completion": record(low_path if selected_head_lr == 1e-5 else high_path),
        "same_starting_state_and_future_examples": True,
        "new_control_is_independent_fork_from_declared25109_not_continuation_of43238_weights": True})
review.pop("allow_matched_head_LR_control", None)
change = review["declared_change"]
change.update(parameter="TR_TDT_normalization", choices=["clip", "token_population"], active_mode="token_population",
    fixed_population_mean_U=26.642061797177284, active_head_lr=selected_head_lr,
    replay_TDT_and_preservation_unchanged=True, natural_draws_unchanged=True,
    whole_pool_mean_TR_weight=1.0,
    group_mean_TR_weights={"fleurs": 2.36880937, "cv": 0.746516984, "tsc": 1.087622386},
    intent="Weight each TR sequence by target-token count divided by the fixed complete-population mean U; replay remains unchanged.",
    equal_gradients_clipping_group_weights_or_quality_claimed=False)
review["population_review"] = record(OUT / "original8199_TR_loss_length_exposure_CPU_review_v1.json")
review_path = OUT / "TSC_uniform_TR_loss25109_USER_review_v1.json"
review_record = create(review_path, review)
print(json.dumps({"passed": True, "selected_head_lr": selected_head_lr,
                  "lower_head_best_errors": low_errors, "higher_head_best_errors": high_errors,
                  "choice": choice_record, "TR_loss_review": review_record}))
