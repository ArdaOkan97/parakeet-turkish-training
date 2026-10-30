"""Tiny genuine-row/schema checks only; no full-pool scan or ML imports."""
import copy
import hashlib
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/parakeet_turkish"
sys.path.insert(0, str(OUT))
import hf_bpe512_tr_metadata_v1 as helper


def expect_rejection(call):
    try:
        call()
    except (ValueError, KeyError):
        return
    raise AssertionError("Corruption was accepted")


def main():
    context = helper.read_candidate_context(OUT / "HF_BPE_512_TR_target_metadata_v1.json")
    with context.manifest_path.open(encoding="utf-8") as manifests, context.metadata_path.open(encoding="utf-8") as metadata, context.parent_metadata_path.open(encoding="utf-8") as parents:
        triples = [tuple(json.loads(line) for line in triple)
                   for triple in itertools.islice(zip(manifests, metadata, parents), 3)]
    annotated = [helper.validate_candidate_row(*triple, index) for index, triple in enumerate(triples)]
    collated = helper.collate_candidate_tr_targets(annotated)
    assert collated["target_lengths"] == [6, 7, 5] and collated["frame_lengths"] == [13, 18, 13]
    assert [len(row) for row in collated["labels"]] == [7, 7, 7]
    assert all(ids[0] == 8192 and ids[1:] == labels for ids, labels in zip(collated["decoder_input_ids"], collated["labels"]))
    assert all(labels[:length] == original[1]["label_ids"] and labels[length:] == [8192] * (7 - length)
               for labels, length, original in zip(collated["labels"], collated["target_lengths"], triples))
    assert collated["six_character_rows"] == (8193, 8199) and collated["new_BPE_rows"] == (8199, 8711)
    assert collated["duration_rows"] == (8711, 8716) and collated["old_teacher_vocab_size"] == 8193
    tests = {
        "source": lambda r, i, p: r.update(source_id="wrong"),
        "manifest_order": lambda r, i, p: i.update(manifest_row_index=1),
        "text": lambda r, i, p: r.update(text=r["text"] + "x"),
        "PCM_hash": lambda r, i, p: i.update(audio_sha256="0" * 64),
        "PCM_frames": lambda r, i, p: i.update(actual_PCM_frames=i["actual_PCM_frames"] + 160),
        "T": lambda r, i, p: i.update(encoder_frames=i["encoder_frames"] + 1),
        "U": lambda r, i, p: i.update(target_tokens=i["target_tokens"] + 1),
        "label_hash": lambda r, i, p: i.update(label_ids_sha256="0" * 64),
        "old_row_kind": lambda r, i, p: i.update(kind="TR_frame_target_metadata_v1"),
        "schema_extra": lambda r, i, p: i.update(schema=2),
        "blank_ID": lambda r, i, p: i["label_ids"].__setitem__(0, 8192),
        "duration_ID": lambda r, i, p: i["label_ids"].__setitem__(0, 8711),
        "negative_ID": lambda r, i, p: i["label_ids"].__setitem__(0, -1),
        "boolean_ID": lambda r, i, p: i["label_ids"].__setitem__(0, True),
    }
    for name, mutate in tests.items():
        row, item, parent = copy.deepcopy(triples[0])
        mutate(row, item, parent)
        expect_rejection(lambda: helper.validate_candidate_row(row, item, parent, 0))
    expect_rejection(lambda: helper.read_candidate_context(OUT / "TR_frame_metadata_cpu_verification_v1.json"))
    old = copy.deepcopy(annotated[0]); old["training_input_type"] = "TR_frame_target_metadata_v1"
    expect_rejection(lambda: helper.load_candidate_tr_item(old))
    replay = copy.deepcopy(annotated[0]); replay.update(language="es", cache_path="unused_original_replay.pt")
    expect_rejection(lambda: helper.load_candidate_tr_item(replay))
    assert "torch" not in sys.modules and "nemo" not in sys.modules and "transformers" not in sys.modules
    report = {
        "schema": 1, "kind": "inert_HF_BPE512_TR_metadata_reader_collator_tiny_verification_v1", "passed": True,
        "helper_path": str(Path(helper.__file__)), "helper_sha256": helper.sha_file(helper.__file__),
        "check_script_sha256": helper.sha_file(__file__), "candidate_proof_sha256": helper.PROOF_SHA,
        "candidate_tokenizer_file_sha256": helper.TOKENIZER_FILE_SHA,
        "candidate_tokenizer_semantic_sha256": helper.TOKENIZER_SEMANTIC_SHA,
        "genuine_prefix_rows_read": 3, "verified_target_lengths": collated["target_lengths"],
        "verified_encoder_frames": collated["frame_lengths"],
        "blank_padded_labels_and_decoder_prefix_verified": True,
        "candidate_ranges_verified": {"six_characters": [8193, 8199], "BPE": [8199, 8711], "durations": [8711, 8716]},
        "row_corruptions_rejected": list(tests), "old8199_proof_rejected": True,
        "old_row_representation_rejected": True, "replay_row_or_cache_request_rejected": True,
        "full74883_pool_scanned": False, "pool_file_hashes_recomputed": False,
        "audio_or_cache_files_read": False, "Torch_Transformers_NeMo_imported": False,
        "model_checkpoint_GPU_DEV_TEST_or_fit_performed": False,
        "candidate_activated": False, "active_files_modified": False,
        "pending_integration": [
            "Root must review this inert helper and consume the full pinned reader before a candidate arm",
            "Build resized8711 embedding/8716 joint, move generation duration suppression, and load weights-only warm start with a new optimizer",
            "Future mixed collator must explicitly accept this KIND and preserve1595 ORIGINAL replay labels/cache/teacher8193",
            "Preserve per-target TDT normalization, KD full8711 student denominator, and distinct six-character/BPE optimizer policies",
            "GPU numerical decoding, whole-update memory/throughput and DEV retention/quality remain unmeasured for this candidate",
        ],
    }
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT / "HF_BPE512_TR_reader_collator_tiny_verification_v1.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"passed": True, "helper_sha256": report["helper_sha256"], "rows_checked": 3,
                      "corruptions_rejected": len(tests) + 3, "Torch_imported": False,
                      "full_pool_scanned": False, "report": str(path)}))


if __name__ == "__main__":
    main()
