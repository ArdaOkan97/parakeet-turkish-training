"""Bounded TRAIN text/decoder diagnostic; no Torch, model, audio, or CUDA."""
from pathlib import Path
import collections
import hashlib
import json
import sys

from metrics import normalize_text
from tokenizers import Tokenizer
import sentencepiece as spm

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/parakeet_turkish"
extended_path = ROOT / "work/research/lora_20261010_01/tokenizer.json"
original_path = ROOT / "work/hf-cache/hub/models--nvidia--parakeet-tdt-0.6b-v3/snapshots/541d1f99c6b0c3cd0b11a95167540bb8edefd82b/tokenizer.json"
sp_path = ROOT / "work/research/nemo_reference/tokenizer_original_v1/902810505b78491d95378fa3000d1bc6_tokenizer.model"
extended, original = Tokenizer.from_file(str(extended_path)), Tokenizer.from_file(str(original_path))
sp = spm.SentencePieceProcessor(model_file=str(sp_path))
samples, groups = {}, collections.Counter()
with (ROOT / "work/data/tr_public_v3_train.jsonl").open(encoding="utf-8") as stream:
    for line in stream:
        row = json.loads(line)
        text = row["text"]
        categories = ["first"]
        if any(c.isdigit() for c in text):
            categories.append("digits")
        if any(c in "ğĞıİşŞ" for c in text):
            categories.append("new_letters")
        if row["dataset"] == "google/fleurs":
            categories.append("FLEURS_TRAIN")
        chosen = [g for g in categories if groups[g] < 32]
        if chosen:
            samples[row["source_id"]] = row
            groups.update(chosen)
        if all(groups[g] == 32 for g in ("first", "digits", "new_letters", "FLEURS_TRAIN")):
            break

unknown_id = extended.token_to_id("<unk>")
results, differences = [], []
unknowns = collections.Counter()
total_words = total_tokens = total_new_tokens = 0
old_id_parity = old_decode_parity = 0
for row in samples.values():
    text = row["text"]
    encoded = extended.encode(text, add_special_tokens=False)
    decoded = extended.decode(encoded.ids, skip_special_tokens=True)
    matched = normalize_text(text, "tr") == normalize_text(decoded, "tr")
    total_words += len(normalize_text(text, "tr").split())
    total_tokens += len(encoded.ids)
    total_new_tokens += sum(8193 <= value <= 8198 for value in encoded.ids)
    for token_id, span in zip(encoded.ids, encoded.offsets):
        if token_id == unknown_id:
            unknowns[text[span[0]:span[1]]] += 1
    old_ids = original.encode(text, add_special_tokens=False).ids
    sp_ids = sp.encode(text, out_type=int)
    hf_text = original.decode(old_ids, skip_special_tokens=True)
    sp_text = sp.decode(sp_ids)
    old_id_parity += old_ids == sp_ids
    old_decode_parity += normalize_text(hf_text, "tr") == normalize_text(sp_text, "tr")
    if normalize_text(hf_text, "tr") != normalize_text(sp_text, "tr") and len(differences) < 6:
        differences.append({"source_id": row["source_id"], "TRAIN_reference": text, "original_HF_decode": hf_text, "original_SentencePiece_decode": sp_text, "ids_equal": old_ids == sp_ids})
    results.append({"source_id": row["source_id"], "normalized_roundtrip": matched, "tokens": len(encoded.ids), "new_character_tokens": sum(8193 <= value <= 8198 for value in encoded.ids)})

assert all(r["normalized_roundtrip"] for r in results), "Accepted TRAIN sample changed lexical text"
dev_path = OUT / "results/adaptive_20261011_native_memory_fix_01_native_resume_step_250/validation_tr.jsonl"
dev = [json.loads(line) for line in dev_path.read_text(encoding="utf-8").splitlines()]
dev_tokens = sum(len(extended.encode(row["reference"], add_special_tokens=False).ids) for row in dev)
dev_words = sum(len(normalize_text(row["reference"], "tr").split()) for row in dev)
report = {"schema": 1, "CPU_only": True, "CUDA_or_Torch_imported": False, "scope": "bounded accepted TRAIN text sample; original native SentencePiece versus original HF conversion; fixed DEV reference token lengths only", "sample_selection": dict(groups), "unique_accepted_TRAIN_rows": len(samples), "normalized_TRAIN_roundtrips_passed": len(results), "TRAIN_tokens": total_tokens, "TRAIN_words": total_words, "TRAIN_tokens_per_normalized_word": total_tokens/total_words, "new_character_tokens": total_new_tokens, "unknown_source_spans": dict(unknowns), "original_HF_vs_SentencePiece_token_ids_equal_rows": old_id_parity, "original_HF_vs_SentencePiece_normalized_decode_equal_rows": old_decode_parity, "original_decoder_differences": differences, "DEV_reference_tokens": dev_tokens, "DEV_reference_words": dev_words, "DEV_reference_tokens_per_word": dev_tokens/dev_words, "rows": results, "source_hashes": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in (extended_path, original_path, sp_path)}, "limits": ["This is a small text-only sample, not a full corpus revalidation or speech inference.", "Original SentencePiece lacks the six new letters; original-decoder differences do not invalidate the checked extended-tokenizer accepted roundtrips.", "No token IDs, tokenizer, audio, targets, benchmark normalization, or training inputs were changed."]}
destination = OUT / "turkish_training_tokenization_small_verification_v1.json"
assert not destination.exists()
destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")
print(json.dumps({k:v for k,v in report.items() if k not in ("rows", "source_hashes", "limits")}, ensure_ascii=False))
