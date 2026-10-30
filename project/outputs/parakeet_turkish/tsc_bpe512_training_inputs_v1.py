"""Inactive whole combined-TRAIN BPE512 admission; stdlib only at import.

The caller must first perform the existing approved corpus/heldout/replay audit.
Pass its original accepted, unannotated TR dictionaries and both tokenizer
objects. This reader streams each of the three pinned text metadata files once
and annotates those same dictionaries in place. It never reads audio or replay.
Any exception invalidates the entire partially annotated pool: discard it and
do not start updates. The final admission marker is set only after every check.

TR_frame_target_metadata_v1 remains the physical no-teacher-feature carrier;
the separate candidate provenance declares its8711 lexical representation.
The frozen original loader/collator can therefore read tr_metadata_labels.
No existing worker/source admission or optimizer policy is changed here.
"""
from __future__ import annotations

from collections import Counter
import hashlib
from itertools import zip_longest
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/parakeet_turkish"
DATA = ROOT / "work/data"
KIND = "TR_frame_target_metadata_v1"
CANDIDATE_KIND = "TSC_TR_frame_target_metadata_HF_BPE_512_v1"
REPORT_KIND = "TSC_BPE512_whole_TRAIN_target_metadata_preparation_v1"
CLIPS = 253798
PREFIX_CLIPS = 74883
GROUP_COUNTS = {"fleurs": 2037, "cv": 72846, "tsc": 178915}
VOCAB_SIZE, BASELINE_VOCAB, BLANK_ID, TEACHER_VOCAB = 8711, 8199, 8192, 8193
MAX_TOKENS = 180
DURATIONS = (0, 1, 2, 3, 4)
REPORT_PATH = OUT / "TSC_BPE512_TRAIN_target_metadata_preparation_v1.json"
REPORT_SHA = "91e0d70894b55c552e3cbb2046c5e22afde97509b163b2496cf1b237067ddb06"
METADATA_PATH = DATA / "TSC_BPE512_TRAIN_metadata_v1/candidate_TR_frame_target_metadata.jsonl"
METADATA_SHA = "c72e41f96173def9217b09e3de6c06727de98b38768baddf938b68da84d1fe06"
METADATA_BYTES = 447655463
MANIFEST_PATH = DATA / "TSC_quality_inputs_v2/combined_TR_train.jsonl"
MANIFEST_SHA = "53b7d5517c056caf5743f559b0a6f0deb6ac3333d7e9ac08a426aea4ad52ba87"
MANIFEST_BYTES = 403658532
PARENT_PATH = DATA / "TSC_quality_inputs_v2/combined_TR_frame_metadata.jsonl"
PARENT_SHA = "5144ee50f1dfbb19e295bd00407956f8a40a8d45d375588cda3a72dd6fc5a587"
PARENT_BYTES = 104943945
SOURCE_PROOF_PATH = OUT / "TSC_quality_inputs_preparation_v2.json"
SOURCE_PROOF_SHA = "99fabf16bc5ec54bb85083ded1ddb8d5c46501e0b25f90cd834eccf6da613f8a"
BASELINE_PATH = ROOT / "work/models/lora_20261010_01_merged/tokenizer.json"
BASELINE_FILE_SHA = "9c9f537bf7190ae2f0e30448a6915c4f607fa4d011b4f62b6f1849064ef8afcd"
BASELINE_SEMANTIC_SHA = "b91d500326aca290ed709d629dab2bee56f298ea7323b94855b9f5ef0caeac75"
CANDIDATE_PATH = ROOT / "work/hf_turkish_bpe_candidate_512_v1/tokenizer.json"
CANDIDATE_FILE_SHA = "6c656b9dc5f85e3dd2c2abe16983908e5d2293da1b2ae096e92e1dc5551df327"
CANDIDATE_SEMANTIC_SHA = "10b747471a6613f818445f4c5697d04bb64ca45c48eb1bde7184d33e28e68607"
PRODUCER_SHA = "1ec210dd629739f9860dd4b1598fbb18717d938b6ff1c9dbbba537c369680145"
ADAPTER_SHA = "839194dddf83c3b08bf608d01cb540b2f160feee426a91373d14cfc426ab7494"
SOURCE_KEYS = ("audio_filepath", "duration", "sample_rate", "audio_sha256", "wav_sha256",
               "language", "dataset", "dataset_revision", "dataset_config", "dataset_split",
               "dataset_file", "text", "reference", "reference_normalized", "training_role",
               "sentence_id", "original_audio_path")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def label_sha(ids):
    # Producer and original metadata deliberately use DEFAULT separators here.
    return hashlib.sha256(json.dumps(ids, sort_keys=True).encode("utf-8")).hexdigest()


def semantic(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def read_pinned_json(path, expected):
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, "Pinned JSON changed: " + str(path))
    return json.loads(raw)


def _record(record, path, digest):
    require(Path(record["path"]).resolve() == Path(path).resolve() and record["sha256"] == digest,
            "Pinned asset record differs: " + str(path))


def validate_tokenizer_geometry(old, new):
    require(semantic(old) == BASELINE_SEMANTIC_SHA and semantic(new) == CANDIDATE_SEMANTIC_SHA,
            "Original/candidate tokenizer semantic identity changed")
    old_model, new_model = old["model"], new["model"]
    require(old_model["type"] == new_model["type"] == "BPE"
            and len(old_model["vocab"]) == BASELINE_VOCAB and len(new_model["vocab"]) == VOCAB_SIZE
            and set(old_model["vocab"].values()) == set(range(BASELINE_VOCAB))
            and set(new_model["vocab"].values()) == set(range(VOCAB_SIZE))
            and all(new_model["vocab"].get(piece) == value for piece, value in old_model["vocab"].items())
            and old_model["vocab"].get("<blank>") == BLANK_ID,
            "Original8199 prefix/interior blank/new512 geometry changed")
    require(new_model["merges"][:len(old_model["merges"])] == old_model["merges"]
            and {k:v for k,v in old_model.items() if k not in ("vocab", "merges")}
            == {k:v for k,v in new_model.items() if k not in ("vocab", "merges")}
            and {k:v for k,v in old.items() if k != "model"} == {k:v for k,v in new.items() if k != "model"},
            "Original merge prefix/model options/normalizer/pretokenizer/decoder/specials changed")


def preflight(metadata_path=METADATA_PATH, report_path=REPORT_PATH):
    """Validate small proofs/assets and file sizes, WITHOUT a whole metadata scan.

    The final raw file digests are checked during the single admission stream.
    This result alone is not complete runtime pool admission or GPU readiness.
    """
    require(Path(metadata_path).resolve() == METADATA_PATH and Path(report_path).resolve() == REPORT_PATH,
            "Only the actual whole253798 preparation is admitted")
    report = read_pinned_json(report_path, REPORT_SHA)
    require(report.get("schema") == 1 and report.get("kind") == REPORT_KIND and report.get("passed") is True
            and report.get("CPU_only") is True and report.get("candidate_activated") is False
            and report.get("clips") == CLIPS and report.get("metadata_kind") == CANDIDATE_KIND
            and Path(report["metadata_path"]).resolve() == METADATA_PATH
            and report.get("metadata_sha256") == METADATA_SHA and report.get("metadata_size_bytes") == METADATA_BYTES
            and report.get("clips_dropped_or_truncated") == 0
            and report.get("source_ID_order_text_reference_audio_paths_PCM_mel_T_and_other_parent_fields_preserved") is True
            and {key:value["counts"]["clips"] for key,value in report["per_source"].items()} == GROUP_COUNTS,
            "Exact complete whole combined TRAIN preparation required")
    _record(report["source_proof"], SOURCE_PROOF_PATH, SOURCE_PROOF_SHA)
    _record(report["manifest"], MANIFEST_PATH, MANIFEST_SHA)
    _record(report["parent_metadata"], PARENT_PATH, PARENT_SHA)
    _record(report["baseline_tokenizer"], BASELINE_PATH, BASELINE_FILE_SHA)
    _record(report["candidate_tokenizer"], CANDIDATE_PATH, CANDIDATE_FILE_SHA)
    _record(report["script"], OUT / "prepare_TSC_BPE512_TRAIN_metadata_v1.py", PRODUCER_SHA)
    _record(report["native_joint_guard"]["adapter"], OUT / "native_tdt_checkpointed_v3.py", ADAPTER_SHA)
    proof = read_pinned_json(SOURCE_PROOF_PATH, SOURCE_PROOF_SHA)
    require(proof.get("kind") == "TSC_supplemental_TR_frame_target_inputs_v2" and proof.get("passed") is True
            and proof.get("clips") == CLIPS and proof["files"]["combined_TR"]["sha256"] == MANIFEST_SHA
            and proof["files"]["TR_metadata"]["sha256"] == PARENT_SHA,
            "Original approved corpus/quarantine proof changed")
    for key,path in (("combined_TR", MANIFEST_PATH), ("TR_metadata", PARENT_PATH)):
        _record(proof["files"][key], path, MANIFEST_SHA if key == "combined_TR" else PARENT_SHA)
        require(path.stat().st_size == (MANIFEST_BYTES if key == "combined_TR" else PARENT_BYTES), "Pinned input size changed")
    _record(proof["files"]["original_replay"], DATA / "TSC_quality_inputs_v2/original_replay_train.jsonl",
            "bee39b029daeeb9ae51d02675e52e7d89f04c14bb252601f8b5267f4c7d57ec8")
    require(METADATA_PATH.stat().st_size == METADATA_BYTES, "Candidate metadata size changed")
    for key in ("script", "normalizer"):
        read_pinned_json_file = Path(report[key]["path"])
        require(hashlib.sha256(read_pinned_json_file.read_bytes()).hexdigest() == report[key]["sha256"], "Pinned preparation source changed")
    adapter = report["native_joint_guard"]["adapter"]
    require(hashlib.sha256(Path(adapter["path"]).read_bytes()).hexdigest() == ADAPTER_SHA, "Actual native guard source changed")
    old = read_pinned_json(BASELINE_PATH, BASELINE_FILE_SHA)
    new = read_pinned_json(CANDIDATE_PATH, CANDIDATE_FILE_SHA)
    validate_tokenizer_geometry(old, new)
    return report


def _backend(value):
    # Accept processor, HF tokenizer, or raw Rust Tokenizer; no imports here.
    tokenizer = getattr(value, "tokenizer", value)
    return getattr(tokenizer, "_tokenizer", tokenizer)


def _labels(backend, text):
    result = backend.encode(text, add_special_tokens=False)
    ids = result.ids if hasattr(result, "ids") else result
    require(type(ids) is list, "Tokenizer must yield explicit list IDs")
    return ids


def _validate_ids(ids, vocab):
    require(type(ids) is list and 1 <= len(ids) <= MAX_TOKENS
            and all(type(value) is int and 0 <= value < vocab and value != BLANK_ID for value in ids),
            "Target IDs are empty/blank/out of vocabulary or exceed180; no truncation permitted")


def source_group(row):
    require(row["language"] == "tr" and row["dataset_split"] in ("train", "validated"), "Only genuine TR TRAIN rows")
    if row["dataset_split"] == "validated":
        require(row.get("training_role") == "train", "Validated source lacks explicit TRAIN role")
    if row["dataset"] == "google/fleurs":
        require(row["dataset_split"] == "train" and row["dataset_config"] == "tr_tr"
                and row["dataset_revision"] == "70bb2e84b976b7e960aa89f1c648e09c59f894dd", "FLEURS source changed")
        return "fleurs"
    if row["dataset"] == "ysdede/commonvoice_17_tr_fixed":
        require(row["dataset_revision"] == "9d8a025f1ae7c5a2f5c43298c7c4910b03995a4b", "CV source changed")
        return "cv"
    require(row["dataset"] == "issai/Turkish_Speech_Corpus" and row["dataset_split"] == "train"
            and row.get("training_role") == "train"
            and row["dataset_revision"] == "d79ea01392882322e8f46657cb635236af25b6d4"
            and "ISSAI_TSC_218/Train/" in row["source_id"] and "/Dev/" not in row["source_id"]
            and "/Test/" not in row["source_id"], "TSC original Train-only role changed")
    return "tsc"


def validate_row(row, manifest, parent, item, index, baseline, candidate, max_joint_bytes):
    """One-row pure validation; returns candidate label list, without row mutation."""
    require(type(row) is dict and canonical(row) == canonical(manifest),
            "Accepted row is cropped/reordered/altered/already annotated versus raw approved manifest")
    require("cache_path" not in row and "features" not in row and "_token_ids" not in row,
            "TR carrier cannot inherit feature/cache labels")
    require(item.get("kind") == CANDIDATE_KIND and item.get("manifest_row_index") == index
            and row["source_id"] == parent["source_id"] == item["source_id"], "Candidate ordered source/index/kind changed")
    require(hashlib.sha256(canonical(row)).hexdigest() == item["training_manifest_row_sha256"]
            and hashlib.sha256(row["text"].encode("utf-8")).hexdigest() == item["text_sha256"],
            "Original raw text/reference/whole manifest-row identity changed")
    extra = {"kind", "manifest_row_index", "label_ids", "baseline_target_tokens", "baseline_label_ids_sha256",
             "training_manifest_row_sha256", "text_sha256"} | {key for key in SOURCE_KEYS if key in row}
    require(set(item) == set(parent) | extra, "Unexpected/missing candidate metadata fields")
    for key,value in parent.items():
        if key not in ("target_tokens", "label_ids_sha256"):
            require(key in item and type(item[key]) is type(value) and item[key] == value, "Copied parent frame/stat/source field changed: " + key)
    for key in SOURCE_KEYS:
        if key in row:
            require(key in item and type(item[key]) is type(row[key]) and item[key] == row[key], "Copied source/text/audio field changed: " + key)
    pcm, mel, frames = parent["actual_PCM_frames"], parent["valid_mel_frames"], parent["encoder_frames"]
    require(row["sample_rate"] == 16000 and type(row["duration"]) in (int, float) and math.isfinite(row["duration"])
            and .5 <= row["duration"] <= 30 and type(pcm) is int and pcm > 0
            and type(mel) is int and type(frames) is int and mel == pcm // 160 and frames == (mel+7)//8
            and int(round(row["duration"]*16000)) == pcm
            and parent["teacher_feature_values_required_for_TR_preservation"] is False,
            "Actual original PCM/mel/T geometry changed")
    old = _labels(baseline, row["text"]); labels = _labels(candidate, row["text"])
    _validate_ids(old, BASELINE_VOCAB); _validate_ids(labels, VOCAB_SIZE)
    _validate_ids(item["label_ids"], VOCAB_SIZE)
    require(type(parent["target_tokens"]) is int and type(item["baseline_target_tokens"]) is int
            and len(old) == parent["target_tokens"] == item["baseline_target_tokens"]
            and label_sha(old) == parent["label_ids_sha256"] == item["baseline_label_ids_sha256"], "Original8199 label length/hash changed")
    require(type(item["target_tokens"]) is int and labels == item["label_ids"] and len(labels) == item["target_tokens"]
            and label_sha(labels) == item["label_ids_sha256"], "Actual8711 candidate labels/U/hash changed")
    # Exact native v3 guard: full unpadded FP32 joint width, not total GPU-fit evidence.
    require(frames * (len(labels)+1) * (VOCAB_SIZE+len(DURATIONS)) * 4 <= max_joint_bytes,
            "Candidate target exceeds declared native per-clip joint bound; no crop/drop")
    return labels


def _admit_stream(rows, baseline, candidate, report, manifest_path, parent_path, metadata_path, max_joint_bytes):
    """Internal stream core; public entry binds all paths/report/tokenizers first."""
    require(len(rows) == report["clips"], "Whole ordered accepted combined pool required, never74883/subset")
    paths = tuple(map(Path, (manifest_path, parent_path, metadata_path)))
    stats = [(path.stat().st_size, path.stat().st_mtime_ns) for path in paths]
    digests = [hashlib.sha256() for _ in paths]
    ordered_ids, canonical_rows = hashlib.sha256(), hashlib.sha256()
    counts = Counter(); maximum_joint = 0
    with paths[0].open("rb") as source, paths[1].open("rb") as parent_stream, paths[2].open("rb") as candidate_stream:
        for index, raw in enumerate(zip_longest(source, parent_stream, candidate_stream)):
            require(index < len(rows) and all(line is not None and line.strip() for line in raw), "Extra/missing/blank paired metadata row")
            for digest,line in zip(digests,raw): digest.update(line)
            manifest, parent, item = (json.loads(line) for line in raw)
            row = rows[index]; group = source_group(row)
            require((index < report["prefix_clips"] and group in ("fleurs", "cv"))
                    or (index >= report["prefix_clips"] and group == "tsc"), "Original74883 prefix/approved TSC suffix changed")
            labels = validate_row(row, manifest, parent, item, index, baseline, candidate, max_joint_bytes)
            counts[group] += 1; ordered_ids.update((row["source_id"]+"\n").encode("utf-8")); canonical_rows.update(canonical(row)+b"\n")
            maximum_joint = max(maximum_joint, item["encoder_frames"]*(len(labels)+1)*8716*4)
            row.update(encoder_frames=item["encoder_frames"], target_tokens=len(labels),
                actual_PCM_frames=item["actual_PCM_frames"], valid_mel_frames=item["valid_mel_frames"],
                training_input_type=KIND, tr_metadata_labels=labels,
                tr_frame_metadata_sha256=report["metadata_sha256"],
                bpe512_TR_metadata_kind=CANDIDATE_KIND, bpe512_TR_report_sha256=REPORT_SHA,
                bpe512_candidate_semantic_sha256=CANDIDATE_SEMANTIC_SHA, bpe512_vocab_size=VOCAB_SIZE,
                bpe512_baseline_target_tokens=item["baseline_target_tokens"],
                bpe512_baseline_label_ids_sha256=item["baseline_label_ids_sha256"],
                bpe512_label_ids_sha256=item["label_ids_sha256"],
                bpe512_manifest_row_index=index, bpe512_manifest_row_sha256=item["training_manifest_row_sha256"],
                bpe512_text_sha256=item["text_sha256"], bpe512_TR_admission_complete=False)
    require(counts == report["group_counts"] and sum(counts.values()) == len(rows), "Full pool/source counts changed")
    expected = (report["manifest"]["sha256"], report["parent_metadata"]["sha256"], report["metadata_sha256"])
    require(tuple(value.hexdigest() for value in digests) == expected, "Actual streamed manifest/parent/candidate file hash changed")
    require(ordered_ids.hexdigest() == report["ordered_source_ids_newline_sha256"]
            and canonical_rows.hexdigest() == report["canonical_manifest_rows_sha256"], "Full source order/raw-row identity changed")
    require(stats == [(path.stat().st_size, path.stat().st_mtime_ns) for path in paths], "Input metadata changed while streamed")
    require(maximum_joint == report["native_joint_guard"]["maximum_candidate_bytes"], "Measured whole-pool arithmetic differs from pinned preparation")
    # No metadata reread or duplicate pool. The marker makes partial failure explicit.
    for row in rows: row["bpe512_TR_admission_complete"] = True
    return rows


def annotate_tr_frame_metadata(baseline_processor, candidate_processor, rows, metadata_path=METADATA_PATH,
                              proof_path=REPORT_PATH, max_tokens=MAX_TOKENS, *, max_joint_bytes=1_700_000_000,
                              return_evidence=False):
    """Return the SAME accepted-row array/dictionaries with compatible candidate TR labels.

    Encoders are caller-injected; this stdlib module imports no tokenizer/model.
    The caller's approved replay1595 is not an argument and is never retokenized.
    return_evidence=True returns (same_rows, acceptance) for a later explicit
    target-dependent sampler transition. Acceptance exists only after full success.
    """
    require(type(rows) is list and max_tokens == MAX_TOKENS and type(max_joint_bytes) is int and max_joint_bytes > 0
            and type(return_evidence) is bool,
            "Existing accepted row list/exact180/no-crop positive native joint bound required")
    report = preflight(metadata_path, proof_path)
    require(len(rows) == CLIPS, "Whole253798 accepted pool required; no74883-only/cropped pool")
    baseline, candidate = _backend(baseline_processor), _backend(candidate_processor)
    require(semantic(json.loads(baseline.to_str())) == BASELINE_SEMANTIC_SHA
            and semantic(json.loads(candidate.to_str())) == CANDIDATE_SEMANTIC_SHA,
            "Actual runtime original/candidate tokenizer differs from frozen assets")
    report = dict(report, prefix_clips=PREFIX_CLIPS, group_counts=GROUP_COUNTS)
    result = _admit_stream(rows, baseline, candidate, report, MANIFEST_PATH, PARENT_PATH, METADATA_PATH, max_joint_bytes)
    if not return_evidence:
        return result
    return result, {"kind": "whole_TSC_BPE512_TR_runtime_admission_v1", "all_rows_verified": True,
        "verified_clips": CLIPS, "metadata_sha256": METADATA_SHA, "preparation_sha256": REPORT_SHA,
        "manifest_sha256": MANIFEST_SHA, "parent_metadata_sha256": PARENT_SHA,
        "ordered_source_ids_newline_sha256": report["ordered_source_ids_newline_sha256"],
        "canonical_manifest_rows_sha256": report["canonical_manifest_rows_sha256"],
        "candidate_semantic_sha256": CANDIDATE_SEMANTIC_SHA, "baseline_semantic_sha256": BASELINE_SEMANTIC_SHA,
        "physical_carrier_kind": KIND, "candidate_metadata_kind": CANDIDATE_KIND,
        "candidate_vocab_size": VOCAB_SIZE, "blank_id": BLANK_ID, "teacher_vocab_size": TEACHER_VOCAB,
        "max_tokens": MAX_TOKENS, "max_joint_bytes": max_joint_bytes,
        "per_source_clips": GROUP_COUNTS.copy(), "rows_list_and_dictionaries_retained_in_place": True,
        "candidate_metadata_streams": 1, "replay_rows_read_or_modified": 0,
        "audio_read_or_fresh_heldout_audit_performed": False,
        "scope": "Whole admitted TRAIN labels/geometry only; not optimizer/sampler conversion or ASR/GPU-fit proof."}


def validate_training_item(row):
    """Small stdlib check returning the unchanged physical carrier's label-list shape.

    Existing tr_frame_metadata_v1.load_training_item performs actual tensorization
    in a future worker; this helper deliberately does not import it or Torch.
    """
    require(row.get("language") == "tr" and row.get("training_input_type") == KIND
            and row.get("bpe512_TR_admission_complete") is True
            and row.get("bpe512_TR_metadata_kind") == CANDIDATE_KIND
            and row.get("tr_frame_metadata_sha256") == METADATA_SHA
            and row.get("bpe512_TR_report_sha256") == REPORT_SHA
            and row.get("bpe512_candidate_semantic_sha256") == CANDIDATE_SEMANTIC_SHA
            and row.get("bpe512_vocab_size") == VOCAB_SIZE
            and "cache_path" not in row and "features" not in row, "Complete candidate-only TR admission required; replay is external")
    labels = row["tr_metadata_labels"]; _validate_ids(labels, VOCAB_SIZE)
    require(len(labels) == row["target_tokens"] and label_sha(labels) == row["bpe512_label_ids_sha256"]
            and hashlib.sha256(row["text"].encode("utf-8")).hexdigest() == row["bpe512_text_sha256"],
            "Candidate labels/text changed after admission")
    require(type(row["encoder_frames"]) is int and row["encoder_frames"] > 0
            and row["valid_mel_frames"] == row["actual_PCM_frames"]//160
            and row["encoder_frames"] == (row["valid_mel_frames"]+7)//8, "Candidate carrier frame geometry changed")
    return {"kind": KIND, "source_id": row["source_id"], "labels": labels, "encoder_frames": row["encoder_frames"]}
