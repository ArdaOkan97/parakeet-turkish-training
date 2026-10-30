"""Isolated TRAIN Turkish target reader/collator for the inert HF BPE512 arm.

Import and row/collator validation use only the standard library. This does not
read audio or load models/caches. Full-pool iteration verifies pinned files;
optional tensor creation requires a caller-supplied Torch module. The frozen
mixed replay collator does NOT accept this new KIND and is not changed here.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import zip_longest
import json
import math
from pathlib import Path
import re

KIND = "TR_frame_target_metadata_HF_BPE_512_v1"
PROOF_KIND = "inert_HF_BPE_512_TR_target_metadata_proof_v1"
PROOF_SHA = "0292b3de68b3ed81e50e11bca6fbf71ed977d575269efa91b081541242127d1c"
MANIFEST_SHA = "d2d0f6df539b1e5cfe9488960a5859bc6ab1870a1ffc634b5b707bc2cc2f5704"
PARENT_SHA = "95eeb218ef88faaec4b2f04b438527f9442dd4ab8d9de165880aaf38fd764837"
METADATA_SHA = "18fcac32c96c14fb66f4a379b066763ca37f621ed0b9cac2ed476d18ffc4c14a"
TOKENIZER_FILE_SHA = "6c656b9dc5f85e3dd2c2abe16983908e5d2293da1b2ae096e92e1dc5551df327"
TOKENIZER_SEMANTIC_SHA = "10b747471a6613f818445f4c5697d04bb64ca45c48eb1bde7184d33e28e68607"
VOCAB_SIZE, BLANK_ID, OLD_TEACHER_SIZE, MAX_TOKENS, CLIPS = 8711, 8192, 8193, 180, 74883
SIX_CHARACTER_ROWS, BPE_ROWS, DURATION_ROWS = (8193, 8199), (8199, 8711), (8711, 8716)
COPIED_KEYS = ("audio_filepath", "duration", "sample_rate", "audio_sha256", "wav_sha256",
               "language", "dataset", "dataset_revision", "dataset_config", "dataset_split", "dataset_file")
PARENT_KEYS = {"source_id", "actual_PCM_frames", "valid_mel_frames", "encoder_frames", "target_tokens",
               "label_ids_sha256", "teacher_feature_values_required_for_TR_preservation"}
METADATA_KEYS = PARENT_KEYS | set(COPIED_KEYS) | {
    "kind", "manifest_row_index", "label_ids", "training_manifest_row_sha256", "text_sha256"}
ANNOTATION_KEYS = {"encoder_frames", "target_tokens", "training_input_type", "tr_metadata_labels",
                   "tr_frame_metadata_sha256", "tr_metadata_candidate_semantic_sha256", "actual_PCM_frames",
                   "valid_mel_frames", "manifest_row_index", "tr_metadata_label_ids_sha256",
                   "tr_metadata_manifest_row_sha256", "tr_metadata_text_sha256"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(row):
    return json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def label_sha(labels):
    # Exactly the frozen label/time convention; do not change separators.
    return hashlib.sha256(json.dumps(labels, sort_keys=True).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CandidateTRContext:
    proof_path: Path
    manifest_path: Path
    metadata_path: Path
    parent_metadata_path: Path
    ordered_source_ids_sha256: str
    canonical_manifest_rows_sha256: str


def read_candidate_context(proof_path, candidate_tokenizer_path=None):
    """Read only the small proof/tokenizer, NOT the 74k metadata/manifest pool."""
    proof_path = Path(proof_path).resolve()
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    require(proof.get("schema") == 1 and proof.get("kind") == PROOF_KIND,
            "Candidate8711 proof schema/kind required; old8199 proof is incompatible")
    require(sha_file(proof_path) == PROOF_SHA, "Exact completed candidate target proof required")
    require(proof.get("passed") is True and proof.get("CPU_only") is True
            and proof.get("candidate_activated") is False and proof.get("fit_performed") is False,
            "Completed inert TRAIN-only candidate proof required")
    require(proof.get("metadata_kind") == KIND and proof.get("clips") == CLIPS
            and proof.get("metadata_sha256") == METADATA_SHA and proof.get("manifest_sha256") == MANIFEST_SHA
            and proof.get("parent_metadata_sha256") == PARENT_SHA
            and proof.get("candidate_tokenizer_semantic_sha256") == TOKENIZER_SEMANTIC_SHA
            and proof.get("candidate_tokenizer_file_sha256") == TOKENIZER_FILE_SHA
            and proof.get("candidate_vocab_size") == VOCAB_SIZE and proof.get("blank_id_unchanged") == BLANK_ID
            and proof.get("teacher_original_vocab_size_unchanged") == OLD_TEACHER_SIZE
            and proof.get("max_target_tokens_policy_unchanged") == MAX_TOKENS,
            "Candidate geometry/identity differs")
    inputs = proof["inputs"]
    candidate_path = Path(inputs["inert_candidate_tokenizer"]["path"]).resolve()
    if candidate_tokenizer_path is not None:
        require(Path(candidate_tokenizer_path).resolve() == candidate_path, "Candidate tokenizer path differs from proof")
    require(sha_file(candidate_path) == TOKENIZER_FILE_SHA, "Candidate tokenizer file changed")
    tokenizer = json.loads(candidate_path.read_text(encoding="utf-8"))
    semantic = hashlib.sha256(json.dumps(tokenizer, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    require(semantic == TOKENIZER_SEMANTIC_SHA and tokenizer["model"]["type"] == "BPE", "Candidate semantic tokenizer differs")
    vocab = tokenizer["model"]["vocab"]
    require(len(vocab) == VOCAB_SIZE and set(vocab.values()) == set(range(VOCAB_SIZE))
            and vocab.get("<blank>") == BLANK_ID, "Candidate vocabulary/blank IDs differ")
    for index, letter in enumerate(("ğ", "Ğ", "ı", "İ", "ş", "Ş"), SIX_CHARACTER_ROWS[0]):
        require(vocab.get(letter) == index, "Existing six-character IDs changed")
    for role, expected in (("accepted_TR_manifest", MANIFEST_SHA), ("audited_baseline_frame_target_metadata", PARENT_SHA)):
        require(inputs[role]["sha256"] == expected, "Dataset pin differs in proof")
    return CandidateTRContext(proof_path, Path(inputs["accepted_TR_manifest"]["path"]).resolve(),
                              Path(proof["metadata_path"]).resolve(),
                              Path(inputs["audited_baseline_frame_target_metadata"]["path"]).resolve(),
                              proof["ordered_source_ids_sha256"], proof["canonical_manifest_rows_sha256"])


def validate_labels(labels, target_tokens, expected_hash):
    require(type(labels) is list and type(target_tokens) is int and 1 <= target_tokens <= MAX_TOKENS
            and len(labels) == target_tokens, "Positive exact candidate target length required")
    require(all(type(identifier) is int and 0 <= identifier < VOCAB_SIZE and identifier != BLANK_ID for identifier in labels),
            "Unsafe target ID: blank/duration/out-of-range/noninteger")
    require(label_sha(labels) == expected_hash, "Candidate label IDs/hash differ")


def validate_candidate_row(row, item, parent, index):
    """Validate one genuine candidate row against original manifest/frame row."""
    require(set(item) == METADATA_KEYS and item.get("kind") == KIND, "Candidate row schema/kind differs")
    require(set(parent) == PARENT_KEYS, "Original frame row schema differs")
    require(type(index) is int and index >= 0 and type(item["manifest_row_index"]) is int
            and item["manifest_row_index"] == index, "Candidate manifest order/index differs")
    require(eligible_training_provenance(row)
            and item["source_id"] == parent["source_id"] == row.get("source_id"), "Ordered TRAIN Turkish source identity required")
    require("cache_path" not in row and "features" not in row, "TR metadata cannot use teacher feature caches")
    require(not (set(row) & ANNOTATION_KEYS), "Original manifest must be unannotated")
    require(type(row.get("text")) is str and hashlib.sha256(canonical(row)).hexdigest() == item["training_manifest_row_sha256"],
            "Full original manifest row/text/provenance differs")
    require(hashlib.sha256(row["text"].encode("utf-8")).hexdigest() == item["text_sha256"], "TR text hash differs")
    require(all(type(item[key]) is type(row[key]) and item[key] == row[key] for key in COPIED_KEYS),
            "Candidate copied PCM/path/duration/dataset fields differ")
    for key in ("audio_sha256", "wav_sha256", "label_ids_sha256"):
        require(type(item[key]) is str and re.fullmatch("[0-9a-f]{64}", item[key]) is not None, "Invalid PCM/WAV/label hash")
    require(type(row["sample_rate"]) is int and row["sample_rate"] == 16000
            and type(row["duration"]) in (int, float) and math.isfinite(row["duration"]) and row["duration"] > 0,
            "Positive16kHz duration required")
    require(all(type(item[key]) is int and item[key] > 0 for key in ("actual_PCM_frames", "valid_mel_frames", "encoder_frames")),
            "Positive integer PCM/mel/encoder dimensions required")
    mel = item["actual_PCM_frames"] // 160
    frames = (mel + 7) // 8
    require(round(row["duration"] * 16000) == item["actual_PCM_frames"]
            and item["valid_mel_frames"] == mel and item["encoder_frames"] == frames
            and all(item[key] == parent[key] for key in ("actual_PCM_frames", "valid_mel_frames", "encoder_frames")),
            "Audited PCM/mel/T convention differs")
    require(item["teacher_feature_values_required_for_TR_preservation"] is False
            and parent["teacher_feature_values_required_for_TR_preservation"] is False,
            "TR preservation teacher features must be absent")
    validate_labels(item["label_ids"], item["target_tokens"], item["label_ids_sha256"])
    return {**row, "encoder_frames": frames, "target_tokens": item["target_tokens"],
            "actual_PCM_frames": item["actual_PCM_frames"], "valid_mel_frames": mel,
            "training_input_type": KIND, "tr_metadata_labels": list(item["label_ids"]),
            "manifest_row_index": index, "tr_frame_metadata_sha256": METADATA_SHA,
            "tr_metadata_candidate_semantic_sha256": TOKENIZER_SEMANTIC_SHA,
            "tr_metadata_label_ids_sha256": item["label_ids_sha256"],
            "tr_metadata_manifest_row_sha256": item["training_manifest_row_sha256"],
            "tr_metadata_text_sha256": item["text_sha256"]}


def eligible_training_provenance(row):
    """Mirror frozen public-v3 raw provenance; validated CV keeps its split.

    The mandatory pinned manifest still proves heldout filtering. This is the
    per-row source-role check from public_turkish_data_policy_v3; no audio or ML
    import is needed, and FLEURS validation/test remain forbidden.
    """
    repo, revision, split = row.get("dataset"), row.get("dataset_revision"), row.get("dataset_split")
    if row.get("language") != "tr":
        return False
    if repo == "ysdede/commonvoice_17_tr_fixed":
        valid = (revision == "9d8a025f1ae7c5a2f5c43298c7c4910b03995a4b" and row.get("dataset_config") == "default"
                 and ((split == "train" and row.get("dataset_file") == "data/train-00000-of-00001.parquet")
                      or (split == "validated" and row.get("training_role") == "train"
                          and row.get("dataset_file") == "data/validated-00000-of-00001.parquet")))
    elif repo == "google/fleurs":
        valid = (revision == "70bb2e84b976b7e960aa89f1c648e09c59f894dd" and split == "train"
                 and row.get("dataset_config") == "tr_tr"
                 and row.get("dataset_file") == "parquet-data/tr_tr/train-00000-of-00001.parquet")
    else:
        valid = False
    return valid and str(row.get("source_id", "")).startswith(f"{repo}:{row.get('dataset_config')}:{split}:")


def jsonl_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            require(bool(line.strip()), "Blank JSONL row changes ordered schema")
            yield json.loads(line)


def iter_candidate_training_rows(context):
    """Future runtime reader: ALWAYS hash all files before yielding any row.

    Consume the iterator completely before training to enforce whole-pool count
    and order digests. This full reader is intentionally not run in tiny guards.
    """
    require(type(context) is CandidateTRContext, "Checked candidate context required")
    require(sha_file(context.proof_path) == PROOF_SHA, "Candidate proof changed")
    for path, expected in ((context.manifest_path, MANIFEST_SHA), (context.metadata_path, METADATA_SHA),
                           (context.parent_metadata_path, PARENT_SHA)):
        require(sha_file(path) == expected, "Pinned manifest/metadata file changed: " + str(path))
    ordered, manifests = hashlib.sha256(), hashlib.sha256()
    sources, count = set(), 0
    for index, triple in enumerate(zip_longest(jsonl_rows(context.manifest_path), jsonl_rows(context.metadata_path),
                                               jsonl_rows(context.parent_metadata_path))):
        require(all(value is not None for value in triple), "Manifest/metadata row counts differ")
        row, item, parent = triple
        require(row["source_id"] not in sources, "Duplicate TRAIN source ID")
        sources.add(row["source_id"])
        result = validate_candidate_row(row, item, parent, index)
        ordered.update((row["source_id"] + "\n").encode("utf-8"))
        manifests.update(canonical(row) + b"\n")
        count += 1
        yield result
    require(count == CLIPS and ordered.hexdigest() == context.ordered_source_ids_sha256
            and manifests.hexdigest() == context.canonical_manifest_rows_sha256, "Whole-pool count/order/manifest digest differs")


def load_candidate_tr_item(row, *, torch_module=None):
    """TR ONLY; replay must continue through the ORIGINAL frozen cache loader."""
    require(row.get("language") == "tr" and row.get("training_input_type") == KIND
            and row.get("tr_frame_metadata_sha256") == METADATA_SHA
            and row.get("tr_metadata_candidate_semantic_sha256") == TOKENIZER_SEMANTIC_SHA,
            "Candidate TR-only training identity required; old/replay items are incompatible")
    require("cache_path" not in row and "features" not in row, "Candidate TR cannot have teacher feature/cache values")
    original = {key: value for key, value in row.items() if key not in ANNOTATION_KEYS}
    require(hashlib.sha256(canonical(original)).hexdigest() == row["tr_metadata_manifest_row_sha256"]
            and hashlib.sha256(row["text"].encode("utf-8")).hexdigest() == row["tr_metadata_text_sha256"],
            "Annotated TRAIN source/text fields changed")
    labels = list(row["tr_metadata_labels"])
    validate_labels(labels, row["target_tokens"], row["tr_metadata_label_ids_sha256"])
    mel = row["actual_PCM_frames"] // 160
    require(type(row["actual_PCM_frames"]) is int and type(row["valid_mel_frames"]) is int
            and type(row["encoder_frames"]) is int and type(row["manifest_row_index"]) is int
            and row["manifest_row_index"] >= 0 and round(row["duration"] * 16000) == row["actual_PCM_frames"]
            and row["valid_mel_frames"] == mel and row["encoder_frames"] == (mel + 7) // 8 > 0,
            "Annotated exact T/PCM/mel values changed")
    if torch_module is not None:
        labels = torch_module.tensor(labels, dtype=torch_module.long)
    return {"kind": KIND, "source_id": row["source_id"], "labels": labels,
            "encoder_frames": row["encoder_frames"]}


def collate_candidate_tr_targets(rows, *, torch_module=None):
    """Only label metadata: blank-pad labels, blank-prefix decoder inputs.

    No audio processor, TrainingBatch, replay cache, or objective is changed.
    A future mixed collator must explicitly combine these with old replay items.
    """
    rows = list(rows)
    require(bool(rows), "Positive candidate TR batch required")
    items = [load_candidate_tr_item(row) for row in rows]
    require(len({row["source_id"] for row in rows}) == len(rows), "Duplicate TR batch source ID")
    maximum = max(len(item["labels"]) for item in items)
    labels = [item["labels"] + [BLANK_ID] * (maximum - len(item["labels"])) for item in items]
    result = {"kind": KIND, "vocab_size": VOCAB_SIZE, "blank_id": BLANK_ID,
              "old_teacher_vocab_size": OLD_TEACHER_SIZE, "six_character_rows": SIX_CHARACTER_ROWS,
              "new_BPE_rows": BPE_ROWS, "duration_rows": DURATION_ROWS,
              "labels": labels, "decoder_input_ids": [[BLANK_ID] + ids for ids in labels],
              "target_lengths": [len(item["labels"]) for item in items],
              "frame_lengths": [item["encoder_frames"] for item in items],
              "valid_mel_lengths": [row["valid_mel_frames"] for row in rows]}
    if torch_module is not None:
        for key in ("labels", "decoder_input_ids", "target_lengths", "frame_lengths", "valid_mel_lengths"):
            result[key] = torch_module.tensor(result[key], dtype=torch_module.long)
    return result
