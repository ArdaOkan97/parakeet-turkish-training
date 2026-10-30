"""Build NEW native warmstart v10 after the separate cleaned TRAIN proofs pass."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/parakeet_turkish"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise AssertionError("Unexpected source shape: " + old[:80])
    return source.replace(old, new, 1)


def main():
    source_path = OUT / "train_full_nemo_turkish_v9.py"
    config_path = OUT / "full_nemo_turkish_v9.json"
    if sha(source_path) != "c398dff97bb4b255e7b45637d21adcc5424732be1064b16b872fa5ce508eb7b2" or sha(config_path) != "dc79c9548e1e493f3fefa121b644e9a95d5a8523782c3d9bd13aedf7aac0885d":
        raise ValueError("Frozen v9 changed")
    target = OUT / "train_full_nemo_turkish_v10.py"
    target_config = OUT / "full_nemo_turkish_v10.json"
    if target.exists() or target_config.exists():
        raise FileExistsError("Use NEW paths; no frozen source overwrite")
    clean_pilot = OUT / "full_nemo_clean_TRAIN_manifests_v1.json"
    clean_full = OUT / "full_nemo_clean_TRAIN_continuation_v1.json"
    pilot_proof = json.loads(clean_pilot.read_text(encoding="utf-8"))
    full_proof = json.loads(clean_full.read_text(encoding="utf-8"))
    if pilot_proof.get("passed") is not True or full_proof.get("passed") is not True:
        raise ValueError("Clean TRAIN proof is not complete")
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    cfg["kind"] = "full_nemo_lightning_turkish_clean_warmstart_v10"
    cfg["source_data_proof"] = cfg["data_proof"]
    cfg["source_data_proof_sha256"] = sha(ROOT / cfg["data_proof"])
    cfg["source_pilot_training_manifest"] = cfg["pilot_training_manifest"]
    cfg["source_continuation_data_proof"] = cfg["continuation_data_proof"]
    cfg["source_continuation_data_proof_sha256"] = cfg["continuation_data_proof_sha256"]
    cfg["data_proof"] = "outputs/parakeet_turkish/full_nemo_clean_TRAIN_manifests_v1.json"
    cfg["data_proof_sha256"] = sha(clean_pilot)
    cfg["pilot_training_manifest"] = "work/nemo_direct_trial_v1/manifests_clean_v1/train_balanced_250.jsonl"
    cfg["continuation_data_proof"] = "outputs/parakeet_turkish/full_nemo_clean_TRAIN_continuation_v1.json"
    cfg["continuation_data_proof_sha256"] = sha(clean_full)
    cfg["model"]["train_ds"]["manifest_filepath"] = "work/nemo_direct_trial_v1/manifests_clean_v1/train_balanced_full_pass.jsonl"
    cfg["training_label_change"] = {
        "policy": "delete_only_frozen_SP_unrepresentable_punctuation_spans_v1",
        "only_TRAIN_labels_changed": True,
        "source_audio_order_nonlabel_metadata_preserved": True,
        "all_canonical_clean_SP_decodes_equal_original_ASR_reference": True,
        "DEV_TEST_labels_changed": False,
        "old_source100_labels_retroactively_changed": False,
        "exact_original_label_or_gradient_continuation_claimed": False,
        "source100_remains_original_native06_checkpoint_not_v9_smoke101": True,
        "fresh_local_loop_and_original_recording_cursor": 1600,
    }
    source = source_path.read_text(encoding="utf-8")
    source = source.replace("warmstart_v9", "clean_warmstart_v10").replace("full_nemo_turkish_v9.json", "full_nemo_turkish_v10.json")
    source = replace_once(source, 'full_proof.get("kind") == "native_sequential_full_TR_pass_after_pilot_v1"',
                          'full_proof.get("kind") == "native_clean_TRAIN_full_TR_pass_after_pilot_v1"')
    source = replace_once(source, 'prepared["data_proof_sha256"] == sha(scoped(cfg["data_proof"]))',
                          'prepared["data_proof_sha256"] == sha(scoped(cfg["source_data_proof"]))')
    source = replace_once(source, 'scoped(cfg["pilot_training_manifest"]), "Source pilot stream changed"',
                          'scoped(cfg["source_pilot_training_manifest"]), "Source pilot stream changed"')
    source = replace_once(source, '    train_count, stream_audit = validate_full_stream(train, proof, cfg)',
                          '    label_audit = validate_clean_training_transition(proof, cfg)\n    train_count, stream_audit = validate_full_stream(train, proof, cfg)')
    source = replace_once(source, '        "full_stream_audit": stream_audit, "source_native_warmstart": inherited,',
                          '        "full_stream_audit": stream_audit, "source_native_warmstart": inherited,\n        "TRAIN_label_transition": label_audit, "exact_original_label_continuation_claimed": False,')
    source = replace_once(source, '        provenance["native_Lightning_warmstart"] = resume',
                          '        provenance["native_Lightning_warmstart"] = resume\n        provenance["TRAIN_label_transition"] = validation["TRAIN_label_transition"]')
    marker = '\ndef validate_source_warmstart(checkpoint_value, cfg):'
    validator = '''
def validate_clean_training_transition(proof, cfg):
    """Bind clean labels and preserve the original source100 recording cursor."""
    require(sha(scoped(cfg["data_proof"])) == cfg["data_proof_sha256"]
            and proof.get("kind") == "official_NeMo_pilot_clean_TRAIN_DEV_manifest_preparation_v1"
            and proof.get("passed") is True, "Pinned cleaned TRAIN pilot proof required")
    raw_path = scoped(cfg["source_data_proof"])
    full_raw_path = scoped(cfg["source_continuation_data_proof"])
    require(sha(raw_path) == cfg["source_data_proof_sha256"]
            and sha(full_raw_path) == cfg["source_continuation_data_proof_sha256"], "Original native source proofs changed")
    raw_proof = json.loads(raw_path.read_text(encoding="utf-8"))
    raw_full = json.loads(full_raw_path.read_text(encoding="utf-8"))
    full = json.loads(scoped(cfg["continuation_data_proof"]).read_text(encoding="utf-8"))
    require(scoped(proof["parent_proof_path"]) == raw_path and proof["parent_proof_sha256"] == sha(raw_path)
            and scoped(full["parent_proof_path"]) == full_raw_path and full["parent_proof_sha256"] == sha(full_raw_path),
            "Cleaned target/source proof chain changed")
    evidence = proof["clean_TRAIN"]
    require(evidence == full["clean_TRAIN"], "Clean full/pilot evidence differs")
    require(evidence["label_cleaning_policy"] == cfg["training_label_change"]["policy"]
            == "delete_only_frozen_SP_unrepresentable_punctuation_spans_v1"
            and evidence["cleaning_only_unknown_punctuation_surfaces"] is True
            and evidence["all_76478_canonical_sources_clean_SP_normalized_decode_equals_original_reference"] is True
            and evidence["cleaning"]["only_unknown_punctuation_removed"] is True
            and evidence["cleaning"]["all_canonical_normalized_roundtrip_passed"] is True
            and evidence["cleaning"]["counts_unchanged"] is True
            and evidence["all_original_refs_preserved_on_changed_rows"] is True
            and evidence["rows_dropped_or_cropped"] == 0,
            "Every accepted canonical TRAIN label must retain its scored original reference")
    for field in ("source_audio_order_nonlabel_metadata_preserved", "only_TRAIN_labels_changed"):
        require(cfg["training_label_change"][field] is True, "Declared TRAIN label transition changed")
    require(cfg["training_label_change"]["DEV_TEST_labels_changed"] is False
            and cfg["training_label_change"]["exact_original_label_or_gradient_continuation_claimed"] is False,
            "Label cleanup cannot claim exact old-label continuation")
    require(evidence["first1600_original_source_ids_audio_order_nonlabel_metadata_preserved"] is True
            and evidence["first1600_nonlabel_metadata_original_sha256"] == evidence["first1600_nonlabel_metadata_clean_sha256"],
            "Warmstart recording cursor changed")
    counts = evidence["canonical_pool_counts"]
    require(counts["tr_pool_clips"] == 74883 and sum(counts.get(lang + "_pool_clips", 0) for lang in LANGUAGES if lang != "tr") == 1595
            and counts["normalized_roundtrip_passed"] == 76478, "Clean canonical TRAIN acceptance count changed")
    require(sha(scoped(evidence["tokenizer_path"])) == evidence["tokenizer_sha256"], "Clean label tokenizer changed")
    require(evidence["DEV_unchanged_sha256"] == {str(scoped(p)): raw_proof["manifest_sha256"][str(scoped(p))]
            for p in cfg["model"]["validation_ds"]["manifest_filepath"]}, "DEV must remain byte-identical")
    raw_manifests = dict(raw_proof["manifest_sha256"])
    raw_manifests.update(raw_full["manifest_sha256"])
    cleaned_manifests = dict(proof["manifest_sha256"])
    cleaned_manifests.update(full["manifest_sha256"])
    for file in evidence["files"].values():
        require(raw_manifests[file["parent_path"]] == file["parent_sha256"]
                and sha(scoped(file["parent_path"])) == file["parent_sha256"]
                and cleaned_manifests[file["path"]] == file["sha256"]
                and file["ordered_nonlabel_metadata_parent_sha256"] == file["ordered_nonlabel_metadata_clean_sha256"]
                and file["source_audio_order_duration_split_language_exact"] is True, "Clean TRAIN file identity differs")
    import unicodedata
    originals = scoped(cfg["source_pilot_training_manifest"])
    clean = scoped(cfg["pilot_training_manifest"])
    prefix_changed = 0
    digest = hashlib.sha256()
    with originals.open(encoding="utf-8") as before, clean.open(encoding="utf-8") as after:
        for index in range(1600):
            old, new = json.loads(next(before)), json.loads(next(after))
            old_identity = {k: v for k, v in old.items() if k != "text"}
            new_identity = {k: v for k, v in new.items() if k not in ("text", "original_reference", "native_TF_label_policy")}
            require(old_identity == new_identity, "Source100 consumed recording metadata/order changed")
            digest.update(json.dumps(old_identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode() + b"\\n")
            if old["text"] != new["text"]:
                require(new.get("original_reference") == old["text"] and new.get("native_TF_label_policy") == evidence["label_cleaning_policy"],
                        "Changed label lost its original reference or policy")
                target_index = 0
                for character in old["text"]:
                    if target_index < len(new["text"]) and character == new["text"][target_index]:
                        target_index += 1
                    else:
                        require(unicodedata.category(character).startswith("P"), "Cleanup may delete only punctuation")
                require(target_index == len(new["text"]), "Cleanup inserted or reordered label characters")
                prefix_changed += 1
            else:
                require(new == old, "Unchanged label acquired undeclared metadata")
    require(digest.hexdigest() == evidence["first1600_nonlabel_metadata_original_sha256"], "Actual consumed prefix identity digest differs")
    return {"policy": evidence["label_cleaning_policy"], "cleaned_pilot_proof_sha256": sha(scoped(cfg["data_proof"])),
            "cleaned_full_proof_sha256": sha(scoped(cfg["continuation_data_proof"])),
            "original_source_pilot_proof_sha256": sha(raw_path), "original_source_full_proof_sha256": sha(full_raw_path),
            "canonical_sources_verified": 76478, "consumed_prefix_rows_compared": 1600,
            "all_targets_UNK_free": evidence["all_clean_canonical_targets_have_no_UNK"],
            "retained_nonpunctuation_unknown_rows": evidence["retained_nonpunctuation_unknown_rows"],
            "consumed_prefix_labels_changed": prefix_changed, "ordered_nonlabel_prefix_sha256": digest.hexdigest(),
            "source_audio_order_nonlabel_metadata_preserved": True, "DEV_TEST_changed": False,
            "cleaning_verified_by_frozen_SentencePiece_proof": True,
            "SentencePiece_or_Torch_imported_during_preflight": False,
            "exact_original_label_or_gradient_continuation_claimed": False}

'''
    source = replace_once(source, marker, validator + marker)
    source = source.replace('"""Native B1 Parakeet/NeMo/Lightning warmstart arm;', '"""Native B1 Parakeet/NeMo/Lightning clean-TRAIN warmstart arm;')
    target.write_text(source, encoding="utf-8", newline="\n")
    target_config.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"entry": str(target), "entry_sha256": sha(target), "config": str(target_config), "config_sha256": sha(target_config)}))


if __name__ == "__main__":
    main()
