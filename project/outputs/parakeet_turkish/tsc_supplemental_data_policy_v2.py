"""Explicit uncertainty-quarantined TSC supplement; original v1 proof stays intact.

Delegates unchanged source/heldout/PCM/label eligibility to the frozen v1 policy.
Only a separate CLOSED-import derivative may remove the three reviewed risk IDs.
Original74883 prefix,1595 replay and all retained labels/order remain exact.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import math
from pathlib import Path

import tsc_supplemental_data_policy_v1 as parent
from tsc_supplemental_data_policy_v1 import (
    OUTPUT, ROOT, DATA, TSC_REPO, TSC_REVISION, ARCHIVE_SHA, ARCHIVE_BYTES,
    APPROVED_IMPORTER_SHA, V3_POLICY_SHA, PARENT_PROOF_SHA, PARENT_LABEL_SHA,
    TOKENIZER_FILE_SHA, TOKENIZER_SEMANTIC_SHA, NORMALIZER_SHA, REPLAY_SOURCE_SET_SHA,
    CV_REPO, CV_REVISION, FL_REPO, FL_REVISION, CONFIGS, require, sha256,
    valid_digest, read_json, read_rows, ids_sha, ordered_rows_sha, verified_path,
    validate_original_source, validate_tsc_row)
from prepare_TSC_quarantine_v2 import QUARANTINE_IDS, keep_row, review_cases

PARENT_POLICY_SHA = "a10b1cbaa94b4e6810f6a5fd749de4b7743bbaf68504c63bbcd5070e8cdd7cdb"
QUARANTINE_HELPER_SHA = "545840caa8d99fab158621a0b86c6559cd96005e07ddec30776f3f8f81d105fc"
POLICY = {**parent.POLICY, "kind": "public_turkish_v3_plus_explicit_quarantined_TSC_TRAIN_v2",
          "uncertainty_quarantine_source_ids": list(QUARANTINE_IDS),
          "uncertainty_does_not_establish_wrong_labels": True, "teacher_relabeling_allowed": False}


@dataclass(frozen=True)
class SupplementContext:
    original: parent.SupplementContext
    quarantine_path: Path
    quarantine_sha256: str
    quarantine: dict

    def __getattr__(self, name):
        return getattr(self.original, name)


def validate_quarantine(quarantine, context):
    require(quarantine.get("kind") == "TSC_TRAIN_uncertainty_quarantine_v2" and quarantine.get("passed") is True
            and quarantine.get("CPU_only") is True and quarantine.get("helper_sha256") == QUARANTINE_HELPER_SHA
            and sha256(OUTPUT/"prepare_TSC_quarantine_v2.py") == QUARANTINE_HELPER_SHA
            and sha256(OUTPUT/"tsc_supplemental_data_policy_v1.py") == PARENT_POLICY_SHA,
            "Exact completed CLOSED-import uncertainty derivative required")
    refs = quarantine["references"]
    for key in ("full_import_proof", "import_closure", "ROOT_Whisper_review", "actual_Whisper_TRAIN_sanity"):
        verified_path(refs[key]["path"], refs[key]["sha256"])
    require(refs["full_import_proof"] == {"path": str(context.proof_path), "sha256": context.proof_sha256}
            and refs["original_accepted_manifest"] == {"path": str(context.accepted_manifest_path),
                                                       "sha256": context.proof["accepted_manifest_sha256"]}
            and context.proof["full_Train_candidate_attempted"] is True,
            "Quarantine must derive from this exact unchanged full importer")
    closure = read_json(refs["import_closure"]["path"])
    require(closure.get("kind") == "root_owned_TSC_full_import_closure_v2" and closure.get("passed") is True
            and closure.get("root_observed_owned_session_completed") is True and closure.get("child_exit_code") == 0
            and closure.get("import_proof") == refs["full_import_proof"]
            and closure.get("accepted_manifest") == refs["original_accepted_manifest"], "CLOSED import ownership differs")
    cases, raw_sanity = review_cases(refs["ROOT_Whisper_review"]["path"])
    require(raw_sanity == refs["actual_Whisper_TRAIN_sanity"], "ROOT reviewed actual sanity identity differs")
    expected = []; removed = []
    for row in context.tsc_rows:
        (expected if keep_row(row, cases) else removed).append(row)
    path = verified_path(quarantine["retained_manifest"]["path"], quarantine["retained_manifest"]["sha256"])
    require(path.is_relative_to(DATA) and path != context.accepted_manifest_path
            and path.name == "quarantined_accepted_train.jsonl" and read_rows(path) == expected,
            "Only exact uncertainty exclusions allowed; kept rows/labels/order must be identical")
    removed_ids = sorted(row["source_id"] for row in removed)
    require(quarantine["approved_uncertainty_source_ids"] == list(QUARANTINE_IDS)
            and quarantine["present_and_removed_source_ids"] == removed_ids
            and quarantine["reviewed_cases_absent_from_full_import_accepted"] == sorted(set(QUARANTINE_IDS)-set(removed_ids))
            and quarantine["quarantined_records"] == removed
            and quarantine["original_clips"] == len(context.tsc_rows)
            and quarantine["retained_clips"] == len(expected) > 0
            and quarantine["quarantined_clips"] == len(removed)
            and quarantine["preserved_original_TR_clips"] == 74883
            and quarantine["preserved_original_replay_clips"] == 1595
            and quarantine["combined_TR_clips"] == 74883+len(expected)
            and quarantine["combined_uniform_one_pass_optimizer_updates"] == math.ceil((74883+len(expected))/8)
            and quarantine["original_manifest_sha256"] == context.proof["accepted_manifest_sha256"]
            and quarantine["retained_manifest_sha256"] == quarantine["retained_manifest"]["sha256"]
            and quarantine["retained_ordered_rows_sha256"] == ordered_rows_sha(expected),
            "Actual present/removed/absent counts and unchanged metadata differ")
    for name, rows in (("original", context.tsc_rows), ("retained", expected), ("quarantined", removed)):
        require(abs(quarantine[name+"_actual_PCM_hours"]-math.fsum(row["duration"] for row in rows)/3600) <= 1e-10,
                "Actual quarantine hours differ: "+name)
    for field in ("original_labels_of_retained_rows_preserved", "kept_manifest_rows_byte_identical"):
        require(quarantine.get(field) is True, "Unchanged label/byte provenance required")
    for field in ("teacher_relabeling_performed", "original_import_proof_and_files_modified", "audio_cropped_or_decoded",
                  "archive_rehashed", "new_data_added_to_training", "human_audio_audition_performed", "corpus_noise_rate_estimated"):
        require(quarantine.get(field) is False, "No relabel/crop/adoption/noise-rate/audition claim: "+field)
    return expected


def load_supplement_context(proof_path, *, quarantine_proof_path, replay_pool_path, replay_pool_sha256,
                            replay_config_path, replay_config_sha256, require_full_train_attempt=False):
    require(require_full_train_attempt is True, "V2 requires actual full TRAIN importer, not pilot adoption")
    original = parent.load_supplement_context(proof_path, replay_pool_path=replay_pool_path,
        replay_pool_sha256=replay_pool_sha256, replay_config_path=replay_config_path,
        replay_config_sha256=replay_config_sha256, require_full_train_attempt=True)
    quarantine_path = Path(quarantine_proof_path).resolve()
    require(quarantine_path.is_relative_to(OUTPUT), "NEW workspace quarantine proof required")
    quarantine = read_json(quarantine_path)
    kept = validate_quarantine(quarantine, original)
    return SupplementContext(replace(original, tsc_rows=kept), quarantine_path, sha256(quarantine_path), quarantine)


def audit_combined_rows(combined_tr, replay, context):
    audit = parent.audit_combined_rows(combined_tr, replay, context)
    audit["kind"] = "TSC_quarantined_supplemental_metadata_input_audit_v2"
    audit["identity"].update(policy=POLICY, policy_source_sha256=sha256(__file__),
        unchanged_TSC_v1_policy_sha256=PARENT_POLICY_SHA,
        TSC_original_import_accepted_manifest_sha256=context.proof["accepted_manifest_sha256"],
        TSC_accepted_manifest_sha256=context.quarantine["retained_manifest"]["sha256"],
        TSC_quarantine_proof={"path": str(context.quarantine_path), "sha256": context.quarantine_sha256},
        TSC_quarantined_clips=context.quarantine["quarantined_clips"],
        TSC_present_and_removed_source_ids=context.quarantine["present_and_removed_source_ids"],
        TSC_requested_uncertainty_source_ids=list(QUARANTINE_IDS),
        TSC_labels_relabelled=False, TSC_uncertainty_not_proven_wrong_labels=True)
    return audit


def audit_combined_manifest(manifest, replay, context):
    manifest = Path(manifest).resolve()
    protected = {Path(path).resolve() for path in context.proof["protected_existing_manifests_sha256"]}
    require(manifest.is_relative_to(DATA) and manifest not in protected
            and manifest not in (context.accepted_manifest_path, context.original_tr_manifest_path,
                                 Path(context.quarantine["retained_manifest"]["path"]).resolve()),
            "Explicit NEW combined manifest required")
    result = audit_combined_rows(read_rows(manifest), replay, context)
    result["identity"].update(combined_TR_manifest_path=str(manifest), combined_TR_manifest_sha256=sha256(manifest))
    return result
