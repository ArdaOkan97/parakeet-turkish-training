"""Isolated original8199 TRAIN policy and authentic length metadata for TSC.

Preparation requires an explicit ROOT approval of the completed import. The
frozen74883 Turkish prefix and1595 replay rows remain exact. New TSC rows have
real WAV header lengths and exact current-tokenizer labels, never fake features.
Existing parent PCM/frame proofs are reused honestly. No archive or model APIs.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import wave

import tsc_supplemental_data_policy_v1 as policy
from metrics import normalize_text

HERE, ROOT = policy.OUTPUT, policy.ROOT
KIND = "TR_frame_target_metadata_v1"  # Existing collator's explicit no-feature representation.
FRAME_PROOF_SHA = "5ddae4c6fa52390f6fb6d5b6190dbcc7d6be7c7d4c9f6aa8e9da42776fb5e021"
FRAME_METADATA_SHA = "95eeb218ef88faaec4b2f04b438527f9442dd4ab8d9de165880aaf38fd764837"
SUPPLEMENT_POLICY_SHA = "a10b1cbaa94b4e6810f6a5fd749de4b7743bbaf68504c63bbcd5070e8cdd7cdb"


def require(value, message):
    if not value:
        raise ValueError(message)


def file_record(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT), "Workspace evidence required")
    return {"path": str(path), "sha256": policy.sha256(path)}


def read_record(item):
    return policy.verified_path(item["path"], item["sha256"])


def approve(approval_path, import_proof_path, *, training=False):
    approval_path = Path(approval_path).resolve()
    approval = policy.read_json(approval_path)
    require(approval.get("kind") == "root_owned_TSC_TRAIN_candidate_approval_v1"
            and approval.get("passed") is True
            and approval.get("allow_separate_TRAIN_candidate_preparation") is True
            and approval.get("import_proof") == file_record(import_proof_path),
            "Explicit ROOT approval of this exact completed import is required")
    if training:
        require(approval.get("allow_isolated_training_arm") is True,
                "ROOT has not approved this corpus candidate for an isolated GPU arm")
        require(approval.get("human_audio_audition_performed") is False,
                "Do not claim ROOT listened when the available audio input was unsupported")
        require(approval.get("corpus_license") == "CC-BY-4.0"
                and approval.get("publisher_license_primary_url") == "https://issai.nu.edu.kz/turkic-asr/"
                and approval.get("HF_card_license_label_recorded_separately") is True,
                "Record the publisher's corpus license separately from the HF card label")
        evidence = approval.get("training_adoption_evidence", {})
        require(set(evidence) == {"source_license_review", "pilot_PCM_text_audit", "independent_Whisper_TRAIN_sanity"},
                "Source/license, actual pilot PCM/text and independent Whisper TRAIN sanity evidence are required")
        for item in evidence.values():
            read_record(item)
        pcm = policy.read_json(evidence["pilot_PCM_text_audit"]["path"])
        whisper = policy.read_json(evidence["independent_Whisper_TRAIN_sanity"]["path"])
        require(pcm.get("kind") == "TSC_TRAIN_pilot_independent_PCM_metadata_audit_v1" and pcm.get("passed") is True
                and pcm.get("fresh_PCM_content_reads") is True and pcm.get("all_zero_clips") == 0,
                "An actual passed nonzero pilot PCM/text audit is required")
        raw_source = policy.read_json(read_record({"path": whisper["provenance"]["import_proof_path"],
                                                  "sha256": whisper["provenance"]["import_proof_sha256"]}))
        predictions = whisper.get("predictions", [])
        require(whisper.get("kind") == "TSC_TRAIN_pinned_Whisper_transcription_sanity_v1"
                and whisper.get("status") == "complete" and whisper.get("passed") is True
                and raw_source.get("archive_sha256") == policy.ARCHIVE_SHA
                and raw_source.get("archive_revision") == policy.TSC_REVISION
                and whisper.get("completed_examples", 0) == len(predictions) >= 18
                and len({row["source_id"] for row in predictions}) == len(predictions)
                and all(row.get("dataset_split") == "train" and row.get("language") == "tr"
                        and row.get("full_prefix_and_EOS_verified") is True and row.get("audio_cropped") is False
                        and row["original_audio_path"].startswith("ISSAI_TSC_218/Train/") for row in predictions)
                and whisper.get("no_heldout_audio_or_predictions_read") is True
                and approval.get("independent_Whisper_TRAIN_sanity_review_accepted") is True,
                "ROOT-approved actual independent Whisper sanity on at least18 TSC TRAIN recordings is required")
    return approval


def context_from_preparation(proof):
    require(proof.get("kind") == "TSC_supplemental_TR_frame_target_inputs_v1"
            and proof.get("passed") is True and proof.get("CPU_only") is True
            and proof.get("helper_sha256") == policy.sha256(__file__)
            and policy.sha256(HERE / "tsc_supplemental_data_policy_v1.py") == SUPPLEMENT_POLICY_SHA,
            "Exact completed supplemental input preparation required")
    refs = proof["references"]
    for item in refs.values():
        read_record(item)
    context = policy.load_supplement_context(refs["import_proof"]["path"],
        replay_pool_path=refs["replay_pool"]["path"], replay_pool_sha256=refs["replay_pool"]["sha256"],
        replay_config_path=refs["replay_config"]["path"], replay_config_sha256=refs["replay_config"]["sha256"],
        require_full_train_attempt=proof["require_full_train_attempt"])
    require(refs["original_metadata"]["sha256"] == FRAME_METADATA_SHA
            and refs["original_frame_proof"]["sha256"] == FRAME_PROOF_SHA
            and proof["clips"] == len(context.original_tr_rows)+len(context.tsc_rows)
            and proof["original_TR_source_ids_sha256"] == policy.ids_sha(context.original_tr_rows),
            "Prepared original-prefix/frame/count identity differs")
    return context


def wav_header(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT), "Workspace canonical WAV required")
    before = (path.stat().st_size, path.stat().st_mtime_ns)
    with wave.open(str(path), "rb") as stream:
        require((stream.getframerate(), stream.getnchannels(), stream.getsampwidth(), stream.getcomptype())
                == (16000, 1, 2, "NONE"), "Actual canonical16k mono PCM16 header required")
        frames = stream.getnframes()
    require(before == (path.stat().st_size, path.stat().st_mtime_ns)
            and 8000 <= frames <= 480000, "Stable whole0.5–30s WAV header required")
    return frames, {"size_bytes": before[0], "mtime_ns": before[1]}


def metadata_item(row, labels):
    pcm, stat = wav_header(row["audio_filepath"])
    require(pcm / 16000 == row["duration"], "Accepted duration/actual WAV header differs")
    require(1 <= len(labels) <= 180 and min(labels) >= 0 and max(labels) < 8199 and 8192 not in labels,
            "Whole representable current8199 target required")
    digest = hashlib.sha256(json.dumps(labels, sort_keys=True).encode()).hexdigest()
    require(row["target_tokens"] == len(labels) and row["label_ids_sha256"] == digest,
            "Approved import target IDs differ")
    mel = pcm // 160
    return {"source_id": row["source_id"], "actual_PCM_frames": pcm, "valid_mel_frames": mel,
            "encoder_frames": (mel+7)//8, "target_tokens": len(labels), "label_ids_sha256": digest,
            "teacher_feature_values_required_for_TR_preservation": False,
            "canonical_WAV_stat_at_preparation": stat}


def prepare(args):
    output = args.output_dir.resolve(); report = args.report.resolve()
    require(output.is_relative_to(ROOT / "work/data") and not output.exists()
            and report.is_relative_to(HERE) and not report.exists(), "NEW isolated input directory/report required")
    approve(args.root_approval, args.tsc_import_proof)
    require(policy.sha256(HERE / "tsc_supplemental_data_policy_v1.py") == SUPPLEMENT_POLICY_SHA,
            "Reviewed supplemental policy changed")
    context = policy.load_supplement_context(args.tsc_import_proof,
        replay_pool_path=args.replay_pool, replay_pool_sha256=policy.sha256(args.replay_pool),
        replay_config_path=args.replay_config, replay_config_sha256=policy.sha256(args.replay_config),
        require_full_train_attempt=args.require_full_train_attempt)
    combined = context.original_tr_rows + context.tsc_rows
    audit = policy.audit_combined_rows(combined, context.original_replay_rows, context)
    original_proof = policy.verified_path(HERE / "TR_frame_metadata_cpu_verification_v1.json", FRAME_PROOF_SHA)
    old_proof = policy.read_json(original_proof)
    original_metadata = policy.verified_path(old_proof["metadata_path"], FRAME_METADATA_SHA)
    old = policy.read_rows(original_metadata)
    require(old_proof.get("passed") is True and old_proof.get("CPU_only") is True
            and old_proof.get("clips") == 74883 and len(old) == len(context.original_tr_rows),
            "Exact completed original frame proof required")
    for row, item in zip(context.original_tr_rows, old):
        require(row["source_id"] == item["source_id"]
                and int(round(row["duration"]*16000)) == item["actual_PCM_frames"]
                and item["valid_mel_frames"] == item["actual_PCM_frames"]//160
                and item["encoder_frames"] == (item["valid_mel_frames"]+7)//8,
                "Original ordered prefix/frame identity differs")
    # Rust tokenizers only; no Torch/Transformers/encoder inference.
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(HERE / "checkpoint/tokenizer.json"))
    require(tokenizer.get_vocab_size() == 8199, "Exact original8199 tokenizer required")
    supplemental = []
    for row in context.tsc_rows:
        ids = tokenizer.encode(row["text"], add_special_tokens=False).ids
        require(normalize_text(tokenizer.decode(ids, skip_special_tokens=True), "tr")
                == normalize_text(row["text"], "tr"), "Ungrouped TSC lexical roundtrip changed")
        supplemental.append(metadata_item(row, ids))
    metadata = old + supplemental
    max_bytes = max(item["encoder_frames"] * (item["target_tokens"]+1) * 8204 * 4 for item in metadata)
    over_bound = [{"source_id": item["source_id"], "required_joint_bytes": item["encoder_frames"]*(item["target_tokens"]+1)*8204*4}
                  for item in supplemental if item["encoder_frames"]*(item["target_tokens"]+1)*8204*4 > 1_700_000_000]
    output.mkdir(parents=True, exist_ok=False)
    files = {"combined_TR": output / "combined_TR_train.jsonl", "original_replay": output / "original_replay_train.jsonl",
             "TR_metadata": output / "combined_TR_frame_metadata.jsonl"}
    for key, rows in (("combined_TR", combined), ("original_replay", context.original_replay_rows), ("TR_metadata", metadata)):
        with files[key].open("x", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    refs = {"import_proof": file_record(args.tsc_import_proof), "ROOT_approval": file_record(args.root_approval),
            "replay_pool": file_record(args.replay_pool), "replay_config": file_record(args.replay_config),
            "original_metadata": file_record(original_metadata), "original_frame_proof": file_record(original_proof)}
    result = {"schema": 1, "kind": "TSC_supplemental_TR_frame_target_inputs_v1", "passed": True, "CPU_only": True,
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "helper_sha256": policy.sha256(__file__),
              "references": refs, "files": {key: file_record(path) for key, path in files.items()},
              "identity": audit["identity"], "clips": len(combined), "original_TR_clips": 74883,
              "original_TR_source_ids_sha256": policy.ids_sha(context.original_tr_rows),
              "TSC_clips": len(supplemental), "replay_clips": 1595,
              "metadata_sha256": policy.sha256(files["TR_metadata"]),
              "tokenizer_semantic_sha256": policy.TOKENIZER_SEMANTIC_SHA,
              "actual_TSC_WAV_headers_checked": len(supplemental), "original_metadata_rows_reused_exactly": len(old),
              "frame_formula": "mel=floor(actualPCMframes/160); encoder=ceil(mel/8)",
              "original_GPU_frame_mask_proof": file_record(HERE / "TR_metadata_actual_GPU_masks_v1.json"),
              "max_full_joint_bytes": max_bytes, "TSC_over_1_7GB_joint_bound": over_bound,
              "require_full_train_attempt": args.require_full_train_attempt,
              "feature_cache_files_written": 0, "TR_teacher_feature_values_required": False,
              "WAV_payload_rehashed_in_preparation": False, "current_WAV_payload_mutation_independently_excluded": False,
              "PCM_content_evidence_reused_from_importer_and_parent": True,
              "heldout_predictions_or_model_GPU_used": False, "existing_TRAIN_manifests_modified": False,
              "new_data_added_to_training": False, "training_launch_eligible": not over_bound,
              "limitations": ["Length formula reuses actual parent GPU mask proof; new corpus GPU masks remain a bounded launch check.",
                              "Current WAV payload checks reuse importer/parent evidence; preparation reads new WAV headers only.",
                              "No speaker/video/perceptual near-duplicate guarantee."]}
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({key: result[key] for key in ("passed", "clips", "TSC_clips", "training_launch_eligible", "max_full_joint_bytes")}))


def audit_tsc_training_inputs(args):
    proof = policy.read_json(args.tsc_input_proof)
    require(proof.get("training_launch_eligible") is True
            and proof.get("max_full_joint_bytes", math.inf) <= args.native_max_joint_bytes,
            "Whole approved combined pool must fit the unchanged native joint bound; no silent dropping/cropping")
    context = context_from_preparation(proof)
    approve(args.tsc_training_approval, context.proof_path, training=True)
    for item in proof["files"].values():
        read_record(item)
    require(Path(args.tr_manifest).resolve() == Path(proof["files"]["combined_TR"]["path"])
            and Path(args.replay_manifest).resolve() == Path(proof["files"]["original_replay"]["path"])
            and Path(args.tr_frame_metadata).resolve() == Path(proof["files"]["TR_metadata"]["path"])
            and Path(args.tr_frame_metadata_proof).resolve() == Path(args.tsc_input_proof).resolve(),
            "Use exactly the prepared combined/replay/metadata inputs and proof")
    replay = policy.read_rows(args.replay_manifest)
    audit = policy.audit_combined_manifest(args.tr_manifest, replay, context)
    require(audit["identity"] == {**proof["identity"], "combined_TR_manifest_path": str(Path(args.tr_manifest).resolve()),
                                 "combined_TR_manifest_sha256": proof["files"]["combined_TR"]["sha256"]},
            "Prepared supplemental identity changed")
    audit["identity"].update(accepted_clips=proof["clips"], accepted_actual_PCM_hours=audit["identity"]["combined_TR_actual_PCM_hours"],
        frame_target_preparation_sha256=policy.sha256(args.tsc_input_proof),
        ROOT_preparation_approval_sha256=proof["references"]["ROOT_approval"]["sha256"],
        ROOT_training_approval=file_record(args.tsc_training_approval),
        replay_sampling_stream_bitwise_identity_claimed=False,
        replay_sampling_nuisance="Original shared RNG is retained; changed TR pool length/shuffle consumes RNG differently, so replay order may differ despite unchanged replay pool/teacher/labels.")
    audit["accepted_source_ids_sha256"] = policy.ids_sha(context.original_tr_rows + context.tsc_rows)
    audit["heldout_source_ids_sha256"] = policy.ids_sha(context.heldout_rows)
    audit["replay_original_ordered_rows_sha256"] = policy.ordered_rows_sha(replay)
    return audit


def source_isolation_tsc(tr, replay, heldout, audit):
    require(policy.ids_sha(tr) == audit["accepted_source_ids_sha256"]
            and policy.ordered_rows_sha(tr) == audit["identity"]["combined_TR_ordered_rows_sha256"]
            and policy.ordered_rows_sha(replay) == audit["replay_original_ordered_rows_sha256"],
            "Whole audited TR prefix/suffix and original replay must be unchanged")
    # The old preparation includes some extra fixed heldout source metadata.
    require(not {row["source_id"] for row in heldout} & {row["source_id"] for row in tr+replay},
            "Fixed source overlap after approved combined audit")
    return audit


def annotate_tr_frame_metadata(processor, rows, metadata_path, proof_path, max_tokens):
    proof = policy.read_json(proof_path)
    require(proof.get("kind") == "TSC_supplemental_TR_frame_target_inputs_v1" and proof.get("passed") is True
            and proof.get("helper_sha256") == policy.sha256(__file__) and max_tokens == 180
            and proof["files"]["TR_metadata"] == file_record(metadata_path), "Prepared exact supplemental frame metadata required")
    tokenizer_sha = hashlib.sha256(json.dumps(json.loads(processor.tokenizer._tokenizer.to_str()),
                                    ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    require(tokenizer_sha == policy.TOKENIZER_SEMANTIC_SHA, "Keep exact original8199 tokenizer")
    metadata = policy.read_rows(metadata_path)
    require(len(rows) == len(metadata) == proof["clips"], "Whole approved combined pool required")
    result = []
    for row, item in zip(rows, metadata):
        mel = item["actual_PCM_frames"] // 160; frames = (mel+7)//8
        require(row["language"] == "tr" and row["source_id"] == item["source_id"]
                and int(round(row["duration"]*16000)) == item["actual_PCM_frames"]
                and item["valid_mel_frames"] == mel and item["encoder_frames"] == frames,
                "Approved ordered actual PCM/frame metadata changed")
        labels = processor.tokenizer(row["text"])["input_ids"]
        require(1 <= len(labels) <= max_tokens and len(labels) == item["target_tokens"]
                and min(labels) >= 0 and max(labels) < 8199 and 8192 not in labels
                and hashlib.sha256(json.dumps(labels, sort_keys=True).encode()).hexdigest() == item["label_ids_sha256"],
                "Exact ungrouped supplemental target IDs changed")
        if "canonical_WAV_stat_at_preparation" in item:
            stat = Path(row["audio_filepath"]).stat()
            require(item["canonical_WAV_stat_at_preparation"] == {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns},
                    "TSC canonical WAV stat changed after preparation; fresh verification required")
        result.append({**row, "encoder_frames": frames, "target_tokens": len(labels), "training_input_type": KIND,
                       "tr_metadata_labels": labels, "tr_frame_metadata_sha256": proof["metadata_sha256"]})
    return result


def confirm_tr_metadata_acceptance(rows, audit):
    require(policy.ids_sha(rows) == audit["accepted_source_ids_sha256"] and len(rows) == audit["identity"]["accepted_clips"],
            "Representability filtering must not silently change the approved combined pool")
    audit.update(turkish_training_representation=KIND, TR_frozen_encoder_feature_cache_required=False,
        actual_TR_metadata_accepted_clips=len(rows), actual_TR_metadata_accepted_hours=math.fsum(row["duration"] for row in rows)/3600,
        TR_metadata_acceptance_pending=False, new_TR_frame_target_metadata_acceptance_pending=False,
        acoustic_cache_acceptance_pending="not_applicable_to_TR_metadata")


def confirm_replay_after_cache(rows, audit):
    require(len(rows) == 1595 and policy.ids_sha(rows) == policy.REPLAY_SOURCE_SET_SHA
            and policy.ordered_rows_sha(rows) == audit["replay_original_ordered_rows_sha256"],
            "Original1595 replay rows/features/cache/lengths/targets must remain exact")


def closed_source_identity(args):
    closure = policy.read_json(args.source_closure)
    require(closure.get("kind") == "root_owned_TSC_closed_source_v1" and closure.get("passed") is True
            and closure.get("root_observed_owned_session_completed") is True and closure.get("child_exit_code") == 0,
            "Actual CLOSED qualifying uniform source is required")
    for key in ("source_checkpoint", "source_selection", "source_run_config", "source_training_history", "source_development_summary"):
        read_record(closure[key])
    source = policy.read_json(closure["source_run_config"]["path"])
    selection = policy.read_json(closure["source_selection"]["path"])
    require(source["run_id"] == selection["run_id"] == closure["source_run_id"]
            and selection.get("accepted") is True and selection.get("heldout_tests_evaluated") is False
            and selection.get("selection_split") == "FLEURS validation only",
            "Closed source selection/config/retention differ")
    dev = policy.read_json(closure["source_development_summary"]["path"])
    require(Path(closure["source_development_summary"]["path"]).resolve()
            == HERE / "results" / (source["run_id"]+"_step_"+str(selection["selected_step"])) / "summary.json",
            "Bind the actual selected source DEV result folder")
    baseline = policy.read_json(HERE / "results/base_v3/summary.json")
    require(dev["validation_tr"]["asr"]["wer"] == selection["turkish_validation_wer"]
            and all(dev["validation_"+lang]["asr"]["utterances"] == 50 for lang in policy.CONFIGS)
            and all(dev["validation_"+lang]["asr"]["wer"] <= baseline["validation_"+lang]["asr"]["wer"]
                    + max(.005, .1*baseline["validation_"+lang]["asr"]["wer"])
                    for lang in ("en", "es", "it", "fr", "de")),
            "Selected source's complete fixed DEV summary must pass all five original retention gates")
    history = policy.read_json(closure["source_training_history"]["path"])
    require(len(history) == selection["completed_steps"]
            and all(row["step"] == index for index, row in enumerate(history, 1)),
            "Closed source history/completed frontier differs")
    if selection["selected_step"] > 0:
        gate = history[selection["selected_step"]-1].get("validation", {})
        require(gate.get("retention_gate_passed") is True and gate.get("turkish_wer") == selection["turkish_validation_wer"],
                "Selected BEST is not the recorded qualifying DEV point")
    require(Path(selection["weights"]).resolve() == Path(closure["source_checkpoint"]["path"]).resolve(),
            "Use only the CLOSED selected BEST, never a nonqualifying latest")
    if args.resume_exact is None:
        require(args.resume_weights is not None and file_record(args.resume_weights) == closure["source_checkpoint"],
                "Initial data-only arm requires approved BEST weights and a fresh optimizer/sampler")
    require(args.resume_runtime_migration is None and args.train_limit is None and args.supplement is None
            and args.reuse_input_audit is None, "No legacy exact-state migration/subset/supplement/reuse policy for changed TSC pool")
    source_science = source["config_identity"]["training_arguments"]
    for key, value in source_science.items():
        if key not in ("epochs", "steps"):
            require(getattr(args, key) == value, "TSC initial arm changes only data/horizon; scientific argument differs: "+key)
    combined = policy.read_json(args.tsc_input_proof)
    require(source.get("turkish_clips_used") == 74883 and source.get("replay_clips_used") == 1595
            and source["config_identity"]["extended_tokenizer_sha256"] == policy.TOKENIZER_SEMANTIC_SHA
            and source["config_identity"]["tr_pool_sha256"] == combined["original_TR_source_ids_sha256"]
            and source["config_identity"]["replay_pool_sha256"] == policy.REPLAY_SOURCE_SET_SHA,
            "Source must be the original8199/74883TR/1595replay lineage, not a same-shaped alternate pool")
    require(args.epochs == 1. and args.steps == math.ceil(combined["clips"] / 8),
            "Declare one full uniform combined-pool pass; horizon must be ceil(actualCombinedN/8), not the old28082 cap")
    require(args.rank == 8 and args.alpha == 16 and args.last_n_layers is None and args.batch_size == 8
            and args.accumulation == 2 and args.replay_fraction == .5 and args.replay_kl == 10
            and args.head_lr == 5e-5 and args.lora_lr == 3e-4 and args.new_token_lr_multiplier == 6
            and args.tdt_implementation == "nemo", "Keep the agreed original8199/rank8/KD10/B8x2 recipe")
    require(args.native_max_joint_bytes == source["config_identity"]["native_loss"]["max_joint_bytes"],
            "Do not silently change the measured native joint memory bound")
    return {"closure": file_record(args.source_closure), "source_run_id": closure["source_run_id"],
            "source_selected_step": selection["selected_step"], "source_checkpoint": closure["source_checkpoint"],
            "source_scientific_arguments": source_science,
            "declared_scientific_changes": ["Append approved TSC TRAIN data to exact74883 prefix; original1595 replay unchanged",
                                             "Set one uniform combined-pool pass and ceil(actualCombinedN/8) update horizon"],
            "initial_optimizer_sampler_RNG_history_counters": "fresh", "exact_old_pool_continuation_claimed": False}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("tsc-import-proof", "root-approval", "replay-pool", "replay-config", "output-dir", "report"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--require-full-train-attempt", action="store_true")
    return p


if __name__ == "__main__":
    prepare(parser().parse_args())
