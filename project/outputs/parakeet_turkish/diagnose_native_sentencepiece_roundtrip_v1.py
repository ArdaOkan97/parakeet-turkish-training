"""CPU-only exact native SP unknown/roundtrip diagnostic; no label/data edits."""
import collections
import hashlib
import json
from pathlib import Path
import sys
import time
import unicodedata

from metrics import normalize_text, edit_distance

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/parakeet_turkish"
DEST = OUT / "native_sentencepiece_roundtrip_diagnosis_v1.json"


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def character_record(char, count):
    return {"character": char, "unicode": f"U+{ord(char):04X}", "name": unicodedata.name(char, "UNNAMED"),
            "category": unicodedata.category(char), "occurrences": count}


def audit(values, tokenizer, include_all_raw_examples=False):
    counts = collections.Counter()
    unknown_chars = collections.Counter()
    meaningful_unknown_chars = collections.Counter()
    meaningful_examples, raw_examples = [], []
    for row in values:
        text, lang = row["text"], row["language"]
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        ref, hyp = normalize_text(text, lang), normalize_text(decoded, lang)
        has_unknown, raw_wrong, meaningful_wrong = tokenizer.unk_id() in ids, decoded != text, ref != hyp
        counts["clips"] += 1
        counts["unknown_token_clips"] += has_unknown
        counts["raw_roundtrip_mismatch_clips"] += raw_wrong
        counts["ASR_normalized_roundtrip_mismatch_clips"] += meaningful_wrong
        if not raw_wrong and not has_unknown:
            continue
        unknowns = []
        if has_unknown:
            encoded_proto = tokenizer.encode(text, return_type="proto")
            unknowns = [{"piece": p.piece, "surface": p.surface, "begin": p.begin, "end": p.end}
                        for p in encoded_proto.pieces if p.id == tokenizer.unk_id()]
            for unknown in unknowns:
                unknown_chars.update(c for c in unknown["surface"] if not c.isspace())
                if meaningful_wrong:
                    meaningful_unknown_chars.update(c for c in unknown["surface"] if not c.isspace())
        if meaningful_wrong:
            retained = [c for x in unknowns for c in x["surface"] if unicodedata.category(c)[0] not in {"P", "Z", "C"}]
            lexical_chars = [c for c in retained if unicodedata.category(c)[0] in {"L", "M", "N"}]
            if lexical_chars:
                cause = "unknown_letter_mark_or_number"
            elif retained:
                cause = "unknown_scored_symbol"
            elif has_unknown:
                cause = "unknown_punctuation_or_control_changes_word_boundary"
            else:
                cause = "known_pieces_normalization_or_spacing_changes_scored_text"
            counts[cause] += 1
            counts["normalized_reference_words_on_mismatches"] += len(ref.split())
            counts["normalized_word_edits_on_mismatches"] += edit_distance(ref.split(), hyp.split())
            counts["normalized_character_edits_on_mismatches"] += edit_distance(ref, hyp)
        else:
            cause = "raw_only_punctuation_whitespace_case_or_compatible_normalization"
            counts["raw_mismatch_but_ASR_normalized_equal"] += raw_wrong
        item = {"source_id": row["source_id"], "language": lang, "dataset_split": row.get("dataset_split"),
                "reference": text, "decoded": decoded, "reference_normalized": ref, "decoded_normalized": hyp,
                "ASR_normalized_equal": not meaningful_wrong, "classification": cause, "unknown_pieces": unknowns}
        if meaningful_wrong:
            meaningful_examples.append(item)
        if raw_wrong and (include_all_raw_examples or len(raw_examples) < 12):
            raw_examples.append(item)
    return {"counts": dict(counts), "unknown_surface_characters": [character_record(c, n) for c, n in unknown_chars.most_common()],
            "unknown_surface_characters_on_ASR_normalized_mismatches": [character_record(c, n) for c, n in meaningful_unknown_chars.most_common()],
            "all_ASR_normalized_mismatch_examples": meaningful_examples,
            "raw_mismatch_examples": raw_examples}


def main():
    if DEST.exists():
        raise FileExistsError("Preserve previous diagnostic.")
    import sentencepiece as spm
    started = time.perf_counter()
    model = ROOT / "work/nemo_direct_trial_v1/tokenizer/tokenizer.model"
    tokenizer = spm.SentencePieceProcessor(model_file=str(model))
    train = ROOT / "work/data/tr_public_v3_train.jsonl"
    result = {"schema": 1, "kind": "native_SP_current8198_unknown_and_ASR_roundtrip_CPU_diagnostic_v1",
              "tokenizer_path": str(model), "tokenizer_sha256": sha(model), "vocabulary_size": tokenizer.get_piece_size(),
              "CPU_only": True, "scalar_single_thread_encoding": True, "no_labels_or_data_changed": True,
              "no_new_pieces_learned": True, "TEST_refs_read": False,
              "TRAIN": audit(rows(train), tokenizer), "DEV_by_language": {}, "replay_TRAIN_by_language": {}}
    inputs = {str(train): sha(train), str(model): sha(model)}
    for lang in ("en", "es", "it", "fr", "de", "tr"):
        path = ROOT / f"work/nemo_direct_trial_v1/manifests/dev_{lang}.jsonl"
        result["DEV_by_language"][lang] = audit(rows(path), tokenizer, include_all_raw_examples=True)
        inputs[str(path)] = sha(path)
    pool = OUT / "runs/adaptive_20261011_native_memory_fix_01_native_resume/training_pool.json"
    with pool.open("rb") as handle:
        handle.seek(max(0, pool.stat().st_size - 8 * 1024 * 1024))
        tail = handle.read()
    marker = b'\n  "replay": '
    begin = tail.index(marker) + len(marker)
    replay, _ = json.JSONDecoder().raw_decode(tail[begin:].decode("utf-8"))
    if len(replay) != 1595:
        raise ValueError("Actual accepted replay pool changed.")
    for lang in ("en", "es", "it", "fr", "de"):
        result["replay_TRAIN_by_language"][lang] = audit((r for r in replay if r["language"] == lang), tokenizer)
    result["replay_pool_clips"] = len(replay)
    inputs[str(pool)] = sha(pool)
    wrapper = ROOT / "work/nemo-native-env-v1/Lib/site-packages/nemo/collections/common/tokenizers/sentencepiece_tokenizer.py"
    mixin = ROOT / "work/nemo-native-env-v1/Lib/site-packages/nemo/collections/asr/parts/mixins/mixins.py"
    result["native_wrapper_source_review"] = {"wrapper_path": str(wrapper), "wrapper_sha256": sha(wrapper),
        "ASR_mixin_path": str(mixin), "ASR_mixin_sha256": sha(mixin), "legacy_default": False,
        "ASR_default_constructor": "SentencePieceTokenizer(model_path=model_path), without legacy override or extra wrapper tokens",
        "ignore_extra_whitespaces_default": True,
        "text_to_ids_default": "tokenizer.encode_as_ids(text), no CTC grouping or special character transliteration",
        "ids_to_text_default": "tokenizer.decode_ids(ids), no CTC grouping; unknown display follows model unk_surface",
        "legacy_true_is_not_a_Turkish_coverage_fix": "Legacy manually recognizes separately supplied special tokens. It does not teach missing ordinary character pieces, and ASR mixin rejects non-None special_tokens.",
        "primary_source": "https://docs.nvidia.com/nemo/speech/nightly/_modules/nemo/collections/common/tokenizers/sentencepiece_tokenizer.html",
        "source_review_without_importing_wrapper": True}
    result["explanation"] = ["HF TRAIN acceptance used its converted tokenizer and ungrouped TDT decode; it does not prove native SentencePiece identical segmentation/normalization/unknown rendering.",
        "Most raw unknowns may be punctuation stripped by ASR scoring; unknown display inserts spaces, so punctuation unknowns inside a word can nevertheless change scored word boundaries.",
        "The same current SentencePiece semantics are used by stock NeMo (legacyFalse defaults). This report compares references through that semantics without changing any target.",
        "BPE512/1024 candidates preserved this decoded text exactly; their compression does not resolve pre-existing native unknown/normalization discrepancies.",
        "DEV diagnostics are not vocabulary learning input; no TEST text or GPU inference was used."]
    result["inputs_sha256"] = inputs
    result["Torch_NeMo_imported"] = "torch" in sys.modules or "nemo" in sys.modules
    result["elapsed_CPU_seconds"] = time.perf_counter() - started
    result["script_sha256"] = sha(__file__)
    DEST.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(DEST), "TRAIN": result["TRAIN"]["counts"],
        "TRAIN_meaningful_unknown_chars": result["TRAIN"]["unknown_surface_characters_on_ASR_normalized_mismatches"],
        "DEV": {k: v["counts"] for k, v in result["DEV_by_language"].items()},
        "replay": {k: v["counts"] for k, v in result["replay_TRAIN_by_language"].items()},
        "Torch_NeMo_imported": result["Torch_NeMo_imported"], "elapsed_CPU_seconds": result["elapsed_CPU_seconds"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
