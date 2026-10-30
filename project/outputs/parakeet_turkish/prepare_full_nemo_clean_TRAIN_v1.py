"""NEW native TRAIN-only labels: delete only frozen-SP unknown punctuation spans.

Every recording/source/order/nonlabel field is retained. DEV, TEST, PCM and
active manifests/tokenizer are untouched. Single-thread SentencePiece CPU only.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import unicodedata

from metrics import normalize_text

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/parakeet_turkish"
SOURCE = ROOT / "work/nemo_direct_trial_v1/manifests"
DEST = ROOT / "work/nemo_direct_trial_v1/manifests_clean_v1"
PILOT_PROOF = OUT / "full_nemo_clean_TRAIN_manifests_v1.json"
FULL_PROOF = OUT / "full_nemo_clean_TRAIN_continuation_v1.json"
POLICY = "delete_only_frozen_SP_unrepresentable_punctuation_spans_v1"


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def text_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def clean(text, tokenizer):
    ids = tokenizer.encode(text)
    if tokenizer.unk_id() not in ids:
        return text, []
    proto = tokenizer.encode(text, return_type="proto")
    original = text.encode("utf-8")
    spans = []
    for piece in proto.pieces:
        if piece.id != tokenizer.unk_id():
            continue
        if original[piece.begin:piece.end].decode("utf-8") != piece.surface:
            raise ValueError("SentencePiece UTF8 surface offsets do not match original text.")
        # Leave any nonpunctuation unknown character intact. In particular,
        # control characters already normalize to spaces and are outside this
        # label correction. Only Unicode P-category bytes may be removed.
        offset = piece.begin
        for char in piece.surface:
            end = offset + len(char.encode("utf-8"))
            if unicodedata.category(char)[0] == "P":
                spans.append({"surface": char, "begin_UTF8_byte": offset, "end_UTF8_byte": end})
            offset = end
    cursor, chunks = 0, []
    for span in spans:
        if span["begin_UTF8_byte"] < cursor:
            raise ValueError("Overlapping unknown spans.")
        chunks.append(original[cursor:span["begin_UTF8_byte"]])
        cursor = span["end_UTF8_byte"]
    chunks.append(original[cursor:])
    return b"".join(chunks).decode("utf-8"), spans


def main():
    if DEST.exists() or PILOT_PROOF.exists() or FULL_PROOF.exists():
        raise FileExistsError("Use fresh clean artifact paths; existing evidence is immutable.")
    import sentencepiece as spm
    started = time.perf_counter()
    model = ROOT / "work/nemo_direct_trial_v1/tokenizer/tokenizer.model"
    model_sha = sha(model)
    tokenizer = spm.SentencePieceProcessor(model_file=str(model))
    if tokenizer.get_piece_size() != 8198:
        raise ValueError("Frozen character-baseline8198 tokenizer is required.")
    parent_pilot_path = OUT / "full_nemo_training_manifests_v1.json"
    parent_full_path = OUT / "full_nemo_training_continuation_v1.json"
    parent_pilot = json.loads(parent_pilot_path.read_text(encoding="utf-8"))
    parent_full = json.loads(parent_full_path.read_text(encoding="utf-8"))
    original_paths = dict(parent_full["manifest_sha256"])
    original_paths.update(parent_pilot["manifest_sha256"])
    for path, expected in original_paths.items():
        if sha(path) != expected:
            raise ValueError(f"Parent source changed: {path}")
    canonical = {}
    canonical_counts = Counter()
    changed = {}
    examples = []
    removed_chars = Counter()
    retained_unknown_rows = 0
    retained_unknown_surfaces = Counter()
    pool_inputs = (SOURCE / "train_turkish_pool.jsonl", SOURCE / "train_replay_pool.jsonl")
    for path in pool_inputs:
        for row in rows(path):
            if row["dataset_split"] not in {"train", "validated"} or row["training_role"] != "train":
                raise ValueError("Only TRAIN/validated label changes are allowed.")
            sid, text, lang = row["source_id"], row["text"], row["language"]
            if sid in canonical:
                raise ValueError("Canonical pool source IDs must be unique.")
            cleaned, spans = clean(text, tokenizer)
            new_ids = tokenizer.encode(cleaned)
            decoded = tokenizer.decode(new_ids)
            if normalize_text(decoded, lang) != normalize_text(text, lang):
                raise ValueError(f"Clean native label is not exact under shared scoring: {sid}")
            if tokenizer.unk_id() in new_ids:
                retained_unknown_rows += 1
                retained_unknown_surfaces.update(p.surface for p in tokenizer.encode(cleaned, return_type="proto").pieces if p.id == tokenizer.unk_id())
            canonical[sid] = text_sha(text)
            canonical_counts[lang + "_pool_clips"] += 1
            canonical_counts["normalized_roundtrip_passed"] += 1
            if cleaned != text:
                changed[sid] = cleaned
                canonical_counts[lang + "_changed_labels"] += 1
                removed_chars.update(c for span in spans for c in span["surface"])
                if len(examples) < 8 or (lang != "tr" and sum(e["language"] == lang for e in examples) < 2):
                    examples.append({"source_id": sid, "language": lang, "original_reference": text,
                                     "clean_TRAIN_label": cleaned, "clean_SP_decoded": decoded,
                                     "ASR_normalized_original": normalize_text(text, lang),
                                     "removed_unknown_punctuation_spans": spans})
    if len(canonical) != 74883 + 1595:
        raise ValueError("Expected every accepted Turkish and replay source.")
    DEST.mkdir(parents=True)
    file_records = {}
    prefix_metadata_digests = {"original": hashlib.sha256(), "clean": hashlib.sha256()}
    first1600_source_audio_digest = hashlib.sha256()
    names = ("train_balanced_250.jsonl", "train_balanced_full_pass.jsonl", "train_balanced_after_pilot.jsonl",
             "train_turkish_pool.jsonl", "train_replay_pool.jsonl")
    for name in names:
        source, target = SOURCE / name, DEST / name
        counts = Counter()
        sequence_parent, sequence_new = hashlib.sha256(), hashlib.sha256()
        with target.open("w", encoding="utf-8", newline="\n") as handle:
            for i, row in enumerate(rows(source)):
                sid = row["source_id"]
                if sid not in canonical or canonical[sid] != text_sha(row["text"]):
                    raise ValueError("Source text differs from the verified canonical pools.")
                new = dict(row)
                if sid in changed:
                    new["text"] = changed[sid]
                    new["original_reference"] = row["text"]
                    new["native_TF_label_policy"] = POLICY
                    counts["changed_label_presentations"] += 1
                    counts[row["language"] + "_changed_label_presentations"] += 1
                compare = {k: v for k, v in new.items() if k not in {"original_reference", "native_TF_label_policy"}}
                compare["text"] = row["text"]
                if compare != row:
                    raise AssertionError("Nonlabel metadata changed.")
                identity = dict(row)
                del identity["text"]
                identity_bytes = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode() + b"\n"
                sequence_parent.update(identity_bytes)
                new_identity = {k: v for k, v in new.items() if k not in {"text", "original_reference", "native_TF_label_policy"}}
                sequence_new.update(json.dumps(new_identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode() + b"\n")
                if name == "train_balanced_250.jsonl" and i < 1600:
                    prefix_metadata_digests["original"].update(identity_bytes)
                    prefix_metadata_digests["clean"].update(json.dumps(new_identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode() + b"\n")
                    first1600_source_audio_digest.update(json.dumps([sid, row["audio_filepath"], row["wav_sha256"], row["audio_sha256"], row["duration"]], separators=(",", ":"), ensure_ascii=False).encode() + b"\n")
                handle.write(json.dumps(new, ensure_ascii=False, separators=(",", ":")) + "\n")
                counts["rows"] += 1
                counts[row["language"] + "_presentations"] += 1
        if sequence_parent.hexdigest() != sequence_new.hexdigest():
            raise AssertionError("Ordered nonlabel/source/audio metadata identity changed.")
        file_records[name] = {"parent_path": str(source), "parent_sha256": sha(source), "path": str(target), "sha256": sha(target),
            "counts": dict(counts), "ordered_nonlabel_metadata_parent_sha256": sequence_parent.hexdigest(),
            "ordered_nonlabel_metadata_clean_sha256": sequence_new.hexdigest(), "source_audio_order_duration_split_language_exact": True}
    if file_records["train_balanced_full_pass.jsonl"]["counts"]["rows"] != 149776 or file_records["train_balanced_250.jsonl"]["counts"]["rows"] != 4000:
        raise AssertionError("Full/pilot counts changed.")
    # Exact cleaned pilot byte prefix and all original DEV files are preserved.
    pilot_bytes = (DEST / "train_balanced_250.jsonl").read_bytes()
    with (DEST / "train_balanced_full_pass.jsonl").open("rb") as handle:
        if handle.read(len(pilot_bytes)) != pilot_bytes:
            raise AssertionError("Cleaned full/pilot prefixes differ.")
    dev_paths = {str(SOURCE / f"dev_{lang}.jsonl"): sha(SOURCE / f"dev_{lang}.jsonl") for lang in ("en", "es", "it", "fr", "de", "tr")}
    for path, expected in dev_paths.items():
        if expected != parent_pilot["manifest_sha256"][path]:
            raise AssertionError("Fixed DEV changed.")
    common = {"label_cleaning_policy": POLICY, "tokenizer_path": str(model), "tokenizer_sha256": model_sha,
        "cleaning_only_unknown_punctuation_surfaces": True, "canonical_pool_counts": dict(canonical_counts),
        "all_76478_canonical_sources_clean_SP_normalized_decode_equals_original_reference": True,
        "all_clean_canonical_targets_have_no_UNK": retained_unknown_rows == 0,
        "retained_nonpunctuation_unknown_rows": retained_unknown_rows,
        "retained_nonpunctuation_unknown_surfaces": dict(retained_unknown_surfaces),
        "cleaning": {"policy": POLICY, "only_unknown_punctuation_removed": True,
                     "source_audio_order_nonlabel_metadata_equivalence": True,
                     "all_canonical_normalized_roundtrip_passed": True, "counts_unchanged": True},
        "removed_unknown_punctuation_characters": [
            {"character": c, "unicode": f"U+{ord(c):04X}", "category": unicodedata.category(c), "occurrences_in_unique_pools": n}
            for c, n in removed_chars.most_common()], "finite_examples": examples,
        "files": file_records, "first1600_nonlabel_metadata_original_sha256": prefix_metadata_digests["original"].hexdigest(),
        "first1600_nonlabel_metadata_clean_sha256": prefix_metadata_digests["clean"].hexdigest(),
        "first1600_source_audio_duration_sequence_sha256": first1600_source_audio_digest.hexdigest(),
        "first1600_original_source_ids_audio_order_nonlabel_metadata_preserved": True,
        "TF_labels_are_changed_recipe_not_exact_original_label_equality": True,
        "all_original_refs_preserved_on_changed_rows": True, "PCM_read_or_rehashed": False, "rows_dropped_or_cropped": 0,
        "DEV_unchanged_sha256": dev_paths, "TEST_refs_read": False, "active_assets_or_training_files_changed": False,
        "Torch_NeMo_imported": "torch" in sys.modules or "nemo" in sys.modules, "single_thread_CPU_only": True,
        "native_SP_semantics": "Default NeMo legacyFalse direct encode_as_ids/decode_ids; deletion prevents UNK display spaces from splitting words under shared punctuation-delete scoring.",
        "parent_diagnostic_path": str(OUT / "native_sentencepiece_roundtrip_diagnosis_v1.json"),
        "parent_diagnostic_sha256": sha(OUT / "native_sentencepiece_roundtrip_diagnosis_v1.json"),
        "script_sha256": sha(__file__), "created_at_utc": datetime.now(timezone.utc).isoformat()}
    pilot = dict(parent_pilot)
    pilot.update({"kind": "official_NeMo_pilot_clean_TRAIN_DEV_manifest_preparation_v1", "passed": True,
        "parent_proof_path": str(parent_pilot_path), "parent_proof_sha256": sha(parent_pilot_path),
        "manifest_sha256": {str(DEST / "train_balanced_250.jsonl"): sha(DEST / "train_balanced_250.jsonl"), **dev_paths},
        "native_SentencePiece_label_acceptance": "CPU verified every74883TR+1595replay clean label: normalized decode exact original reference; nonpunctuation UNK controls retained and reported; actual loader/GPU still separate",
        "full_continuation_proof_path": str(FULL_PROOF), "clean_TRAIN": common})
    PILOT_PROOF.write_text(json.dumps(pilot, ensure_ascii=False, indent=2), encoding="utf-8")
    full = dict(parent_full)
    full.update({"kind": "native_clean_TRAIN_full_TR_pass_after_pilot_v1", "passed": True,
        "parent_proof_path": str(parent_full_path), "parent_proof_sha256": sha(parent_full_path),
        "pilot_proof_path": str(PILOT_PROOF), "pilot_proof_sha256": sha(PILOT_PROOF),
        "manifest_sha256": {str(DEST / name): sha(DEST / name) for name in names if name != "train_balanced_250.jsonl"},
        "clean_TRAIN": common, "elapsed_CPU_seconds": time.perf_counter() - started})
    FULL_PROOF.write_text(json.dumps(full, ensure_ascii=False, indent=2), encoding="utf-8")
    if sha(model) != model_sha:
        raise AssertionError("Frozen tokenizer changed.")
    print(json.dumps({"pilot_proof": str(PILOT_PROOF), "pilot_proof_sha256": sha(PILOT_PROOF),
        "full_proof": str(FULL_PROOF), "full_proof_sha256": sha(FULL_PROOF), "canonical": dict(canonical_counts),
        "file_counts": {k: v["counts"] for k, v in file_records.items()}, "elapsed_CPU_seconds": time.perf_counter() - started}, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
