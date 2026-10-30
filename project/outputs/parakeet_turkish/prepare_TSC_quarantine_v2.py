"""ROOT-only, metadata-only uncertainty quarantine after a CLOSED full TSC import.

Preserves every original file and label. Three previously reviewed TRAIN cases
are withheld as uncertainty cases, not declared wrong or relabelled by Whisper.
No audio/archive/model/GPU/network APIs. New retained JSONL preserves kept bytes.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import tsc_supplemental_data_policy_v1 as parent

HERE, ROOT = parent.OUTPUT, parent.ROOT
PARENT_POLICY_SHA = "a10b1cbaa94b4e6810f6a5fd749de4b7743bbaf68504c63bbcd5070e8cdd7cdb"
QUARANTINE_STEMS = ("188561", "192361", "188814")
QUARANTINE_IDS = tuple(f"{parent.TSC_REPO}:default:train:None:ISSAI_TSC_218/Train/{stem}.wav" for stem in QUARANTINE_STEMS)


def record(path):
    path = Path(path).resolve()
    parent.require(path.is_relative_to(ROOT), "Workspace-only evidence required")
    return {"path": str(path), "sha256": parent.sha256(path)}


def review_cases(review_path):
    review = parent.read_json(review_path)
    parent.require(review.get("kind") == "TSC_TRAIN_Whisper_actual_prediction_ROOT_review_v1"
                   and review.get("passed") is True and review.get("child_exit_code") == 0
                   and review.get("root_reviewed_actual_18_reference_hypothesis_pairs") is True
                   and review.get("human_audio_audition_performed") is False
                   and review.get("teacher_relabeling_performed") is False,
                   "Actual ROOT-reviewed TRAIN-only sanity required; no audition/relabel claim")
    evidence = review["source"]
    raw_path = parent.verified_path(evidence["path"], evidence["sha256"])
    raw = parent.read_json(raw_path)
    parent.require(raw.get("kind") == "TSC_TRAIN_pinned_Whisper_transcription_sanity_v1"
                   and raw.get("status") == "complete" and raw.get("passed") is True
                   and raw.get("no_heldout_audio_or_predictions_read") is True,
                   "Actual complete public TRAIN sanity evidence required")
    cases = {row["source_id"]: row for row in raw["predictions"] if row["source_id"] in QUARANTINE_IDS}
    parent.require(set(cases) == set(QUARANTINE_IDS), "All three exact reviewed uncertainty cases required")
    for sid, row in cases.items():
        parent.require(row.get("dataset_split") == "train" and row.get("language") == "tr"
                       and row.get("full_prefix_and_EOS_verified") is True and row.get("audio_cropped") is False
                       and row.get("original_audio_path") == sid.split("None:", 1)[1]
                       and parent.valid_digest(row.get("canonical_pcm_sha256")), "Whole paired TRAIN sanity identity differs")
    return cases, evidence


def keep_row(row, cases):
    """Only exact IDs are eligible for withholding; recorded PCM/text must agree."""
    if row["source_id"] not in QUARANTINE_IDS:
        return True
    evidence = cases[row["source_id"]]
    parent.require(row["canonical_pcm_sha256"] == evidence["canonical_pcm_sha256"]
                   and row["text"] == evidence["reference"]
                   and row["duration"] == evidence["duration"],
                   "Reviewed case PCM/original label/duration differs; no name-only quarantine")
    return False


def filter_manifest(source, destination, cases):
    """Stream bounded metadata, hash original bytes and retain exact kept bytes."""
    original_digest = hashlib.sha256(); kept_digest = hashlib.sha256()
    ordered_digest = hashlib.sha256(); count = kept = 0
    durations = []; kept_durations = []; removed = []
    with Path(source).open("rb") as incoming, Path(destination).open("xb") as outgoing:
        for line in incoming:
            original_digest.update(line)
            if not line.strip():
                outgoing.write(line); kept_digest.update(line)
                continue
            row = json.loads(line.decode("utf-8-sig")); count += 1
            parent.validate_tsc_row(row, source)
            durations.append(row["duration"])
            if keep_row(row, cases):
                outgoing.write(line); kept_digest.update(line); kept += 1
                kept_durations.append(row["duration"])
                ordered_digest.update(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())
                ordered_digest.update(b"\n")
            else:
                parent.require(row["source_id"] not in {item["source_id"] for item in removed}, "Duplicate uncertainty case ID")
                removed.append(row)
    return {"original_manifest_sha256": original_digest.hexdigest(), "retained_manifest_sha256": kept_digest.hexdigest(),
            "retained_ordered_rows_sha256": ordered_digest.hexdigest(), "original_clips": count, "retained_clips": kept,
            "original_actual_PCM_hours": math.fsum(durations)/3600,
            "retained_actual_PCM_hours": math.fsum(kept_durations)/3600,
            "quarantined_clips": len(removed), "quarantined_actual_PCM_hours": math.fsum(row["duration"] for row in removed)/3600,
            "quarantined_records": removed}


def prepare(args):
    parent.require(parent.sha256(HERE/"tsc_supplemental_data_policy_v1.py") == PARENT_POLICY_SHA,
                   "Frozen original policy changed")
    output = args.output_dir.resolve(); report = args.report.resolve()
    parent.require(output.is_relative_to(ROOT/"work/data") and not output.exists()
                   and report.is_relative_to(HERE) and not report.exists(), "NEW quarantine directory and report required")
    proof_path = args.full_import_proof.resolve(); proof = parent.read_json(proof_path)
    parent_path = parent.verified_path(HERE/"tr_public_v3_preparation.json", parent.PARENT_PROOF_SHA)
    parent.verified_path(HERE/"prepare_TSC_TRAIN_v1.py", parent.APPROVED_IMPORTER_SHA)
    parent.validate_import_proof(proof, parent.read_json(parent_path), require_full_train_attempt=True)
    closure = parent.read_json(args.import_closure)
    source = Path(proof["accepted_manifest_path"]).resolve()
    expected_source = {"path": str(source), "sha256": proof["accepted_manifest_sha256"]}
    parent.require(closure.get("kind") == "root_owned_TSC_full_import_closure_v2" and closure.get("passed") is True
                   and closure.get("root_observed_owned_session_completed") is True and closure.get("child_exit_code") == 0
                   and closure.get("import_proof") == record(proof_path)
                   and closure.get("accepted_manifest") == expected_source,
                   "ROOT must confirm actual CLOSED full import before any metadata derivative")
    parent.require(source.is_relative_to(ROOT/"work/data") and source.name == "accepted_train.jsonl"
                   and source.parent != output and proof["accepted_clips"] > 0, "Original separate accepted TRAIN source required")
    cases, raw_sanity = review_cases(args.whisper_review)
    before = (source.stat().st_size, source.stat().st_mtime_ns)
    output.mkdir(parents=True, exist_ok=False)
    retained = output/"quarantined_accepted_train.jsonl"
    result = filter_manifest(source, retained, cases)
    parent.require(result["original_manifest_sha256"] == proof["accepted_manifest_sha256"]
                   and result["original_clips"] == proof["accepted_clips"]
                   and abs(result["original_actual_PCM_hours"]-proof["accepted_actual_PCM_hours"]) <= 1e-10
                   and before == (source.stat().st_size, source.stat().st_mtime_ns), "Closed original manifest/hash/count/hours changed")
    parent.require(result["retained_clips"] > 0 and result["original_clips"]-result["retained_clips"] == result["quarantined_clips"],
                   "Positive retained corpus and exact exclusions required")
    removed = {row["source_id"] for row in result["quarantined_records"]}
    receipt = {"schema": 2, "kind": "TSC_TRAIN_uncertainty_quarantine_v2", "passed": True, "CPU_only": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "helper_sha256": parent.sha256(__file__),
        "references": {"full_import_proof": record(proof_path), "import_closure": record(args.import_closure),
            "ROOT_Whisper_review": record(args.whisper_review), "actual_Whisper_TRAIN_sanity": raw_sanity,
            "original_accepted_manifest": expected_source},
        "retained_manifest": {"path": str(retained), "sha256": result["retained_manifest_sha256"]}, **result,
        "approved_uncertainty_source_ids": list(QUARANTINE_IDS), "present_and_removed_source_ids": sorted(removed),
        "reviewed_cases_absent_from_full_import_accepted": sorted(set(QUARANTINE_IDS)-removed),
        "preserved_original_TR_clips": 74883, "preserved_original_replay_clips": 1595,
        "combined_TR_clips": 74883+result["retained_clips"],
        "combined_uniform_one_pass_optimizer_updates": math.ceil((74883+result["retained_clips"])/8),
        "rationale": "Observed whole-TRAIN reference/Whisper disagreements and short-clip ambiguity; alignment uncertainty, not proven wrong labels",
        "original_labels_of_retained_rows_preserved": True, "kept_manifest_rows_byte_identical": True,
        "teacher_relabeling_performed": False, "original_import_proof_and_files_modified": False,
        "audio_cropped_or_decoded": False, "archive_rehashed": False, "new_data_added_to_training": False,
        "human_audio_audition_performed": False, "corpus_noise_rate_estimated": False,
        "source_scope": "Metadata derivative of ROOT-confirmed full closed importer; no current PCM payload read",
        "limitations": ["Three observed risks do not establish full corpus label quality.",
                        "Actual importer/PCM and purpose-selected sanity evidence are reused; no perceptual or speaker guarantee."]}
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("x", encoding="utf-8") as stream: json.dump(receipt, stream, indent=2, ensure_ascii=False)
    print(json.dumps({key: receipt[key] for key in ("passed", "original_clips", "retained_clips", "quarantined_clips", "retained_actual_PCM_hours")}))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("full-import-proof", "import-closure", "whisper-review", "output-dir", "report"):
        result.add_argument("--"+name, type=Path, required=True)
    return result


if __name__ == "__main__": prepare(parser().parse_args())
