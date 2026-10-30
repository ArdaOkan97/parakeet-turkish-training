"""Purpose receipt only after the actual completed paired TRAIN study.

Uses existing immutable archived full-state checkpoints without recopying or
deserializing them. New purpose files are small metadata, not new model state.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from tsc_bpe_full_decoder_transition_v1 import (
    SOURCE, STORAGE, RECEIPT_KIND, HANDOFF_SHA, record, read, checked,
    require, permitted, validate_study_review,
)


def build(args):
    review_path = permitted(args.completed_study_review)
    review_record = record(review_path)
    review = json.loads(review_path.read_text(encoding="utf-8"))
    selected, arms = validate_study_review(review)
    handoff = record(SOURCE / "BPE_full_decoder_next_trial_handoff_v1.json")
    require(handoff["sha256"] == HANDOFF_SHA, "Direct user scientific handoff changed")
    require(args.decoder_arm in ("bpe512", "original8199") and args.target_run_id
            and Path(args.target_run_id).name == args.target_run_id, "One fresh named arm required")
    checkpoint = selected["checkpoint"]
    checked(checkpoint, checkpoint=True)
    for key in ("run_config", "training_pool"):
        checked(selected[key])
    for value in selected["DEV"].values(): checked(value)
    require(isinstance(selected.get("history_prefix_sha256"), str)
            and len(selected["history_prefix_sha256"]) == 64,
            "Exact chosen full-state history prefix digest required")
    prep_record = None
    if args.decoder_arm == "bpe512":
        require(args.BPE_label_preparation is not None, "Actual completed BPE TRAIN target preparation required")
        prep_record = record(args.BPE_label_preparation)
        prep = read(prep_record)
        require(prep.get("passed") is True and prep.get("source_arm") == selected["data_arm"]
                and prep.get("rows") == (352768 if selected["data_arm"] == "expanded" else 253798),
                "Target sidecar must describe the actual chosen source pool")
        checked(prep["target_catalog"])
    return {"kind": RECEIPT_KIND,
        "status": "actual_completed_study_qualified_BEST_for_decoder_screen",
        "target_run_id": args.target_run_id, "decoder_arm": args.decoder_arm,
        "source_run_id": selected["run_id"], "source_step": selected["step"],
        "source_checkpoint": checkpoint, "completed_study_review": review_record,
        "handoff": handoff, "BPE_label_preparation": prep_record,
        "source_role": "coherent_strict_all_six_qualified_BEST",
        "completed_parent_step": 65948,
        "updates_rewound_from_archived_completed_branch": 65948-selected["step"],
        "later_full_parent_history_and_endpoint_preserved": True,
        "actual_both_data_arms_complete": True,
        "frozen_encoder_in_both_decoder_arms": True,
        "mature_old_Adam_retained_new512_clock_zero": True,
        "checkpoint_payload_read_copied_or_modified": False,
        "source_choice_before_study_completion": False,
        "actual_tensor_restore_source350_and_native_fit_pending": True,
        "new_storage_root": str(STORAGE)}


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--completed-study-review", type=Path, required=True)
    p.add_argument("--target-run-id", required=True)
    p.add_argument("--decoder-arm", choices=("bpe512", "original8199"), required=True)
    p.add_argument("--BPE-label-preparation", dest="BPE_label_preparation", type=Path)
    p.add_argument("--receipt-output", type=Path)
    p.add_argument("--validate-only", action="store_true")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    receipt = build(args)
    if args.validate_only:
        print(json.dumps({"passed": True, "metadata_only": True, "source_step": receipt["source_step"],
                          "decoder_arm": args.decoder_arm, "no_source_admission_or_GPU": True}))
    else:
        require(args.receipt_output is not None, "Explicit fresh purpose receipt destination required")
        target = permitted(args.receipt_output)
        require(not target.exists(), "Never replace an existing admitted purpose receipt")
        from atomic_io_retry_v1 import write_json
        write_json(target, receipt)
        print(json.dumps({"passed": True, "receipt": record(target), "runtime_proof_pending": True}))
