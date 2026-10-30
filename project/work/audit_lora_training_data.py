"""Independently verify the active LoRA run's exact CPU tokenizer/cache inputs."""
import copy
import hashlib
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
from tokenizers import Tokenizer
from transformers import AutoProcessor

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs/parakeet_turkish"
sys.path.insert(0, str(OUTPUT))
from extend_vocab import CHARACTER_DONORS, _hash_json, _make_tokenizer
from metrics import normalize_text


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


torch.set_num_threads(1)
config_path = OUTPUT / "run_config.json"
audit_path = OUTPUT / "vocabulary_audit.json"
config = json.loads(config_path.read_text(encoding="utf-8"))
vocabulary = json.loads(audit_path.read_text(encoding="utf-8"))
assert config["run_id"] == "lora_20261010_01"
base_path = (ROOT / "work/base_model_path.txt").read_text().strip()
processor = AutoProcessor.from_pretrained(base_path, local_files_only=True)
original_tokenizer = processor.tokenizer
original = json.loads(original_tokenizer._tokenizer.to_str())
assert _hash_json(original) == vocabulary["original_tokenizer_sha256"]
old_vocab = original_tokenizer._tokenizer.get_vocab()
old_size = len(original_tokenizer)
assert old_size == vocabulary["original_vocab_size"] == 8193
updated = copy.deepcopy(original)
core = updated["model"]["vocab"]
for token, token_id in sorted(old_vocab.items(), key=lambda value: value[1]):
    core.setdefault(token, token_id)
additions = [character for character in CHARACTER_DONORS if character not in old_vocab]
for token_id, character in enumerate(additions, start=old_size):
    core[character] = token_id
tokenizer = _make_tokenizer(original_tokenizer, Tokenizer.from_str(json.dumps(updated, ensure_ascii=False)))
backend_text = tokenizer._tokenizer.to_str()
assert _hash_json(json.loads(backend_text)) == vocabulary["extended_tokenizer_sha256"]
namespace = hashlib.sha256((config["base_revision"] + backend_text).encode()).hexdigest()[:20]
cache = ROOT / "work/encoder-cache" / namespace
assert cache.is_dir()
tokenizer_path = ROOT / "work/research" / config["run_id"] / "tokenizer.json"
tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
tokenizer_path.write_text(backend_text, encoding="utf-8")

cv_path = ROOT / "work/data/tr_train.jsonl"
supplement_path = ROOT / config["supplement"]
replay_path = ROOT / "work/data/replay_train.jsonl"
cv_rows = rows(cv_path)
random.Random(config["seed"]).shuffle(cv_rows)
cv_rows = cv_rows[:config["train_limit"]]
groups = {"turkish_commonvoice": cv_rows, "turkish_fleurs_train": rows(supplement_path), "retention_fleurs_train": rows(replay_path)}
summaries, excluded, accepted_rows = {}, {}, []
for group, candidates in groups.items():
    accepted = []
    dropped = []
    reasons = Counter()
    for row in candidates:
        ids = tokenizer(row["text"])["input_ids"]
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        failures = []
        if not ids:
            failures.append("empty_labels")
        if len(ids) > config["max_tokens"]:
            failures.append("above_max_tokens")
        if normalize_text(decoded, row["language"]) != normalize_text(row["text"], row["language"]):
            failures.append("lexical_roundtrip")
        if group == "retention_fleurs_train" and ids and max(ids) >= old_size:
            failures.append("new_teacher_token_ids")
        if failures:
            reasons.update(failures)
            dropped.append({"source_id": row["source_id"], "duration": row["duration"], "target_tokens": len(ids), "reasons": failures})
            continue
        cache_path = cache / (hashlib.sha256(row["source_id"].encode()).hexdigest() + ".pt")
        assert cache_path.is_file(), row["source_id"]
        item = torch.load(cache_path, map_location="cpu", weights_only=True)
        assert item["source_id"] == row["source_id"]
        assert item["labels"].tolist() == ids, row["source_id"]
        assert item["features"].ndim == 2 and item["features"].shape[1] == 1024
        assert item["features"].shape[0] > 0 and item["labels"].ndim == 1
        assert torch.isfinite(item["features"]).all()
        accepted.append({"source_id": row["source_id"], "language": row["language"], "duration": row["duration"], "target_tokens": len(ids), "encoder_frames": int(item["features"].shape[0]), "labels_sha256": hashlib.sha256(item["labels"].numpy().tobytes()).hexdigest()})
        accepted_rows.append(row)
    summaries[group] = {"candidate_clips": len(candidates), "candidate_hours": sum(row["duration"] for row in candidates)/3600,
                        "accepted_clips": len(accepted), "accepted_hours": sum(row["duration"] for row in accepted)/3600,
                        "excluded_clips": len(dropped), "exclusion_reason_counts_nonexclusive": dict(reasons),
                        "target_tokens_maximum": max(row["target_tokens"] for row in accepted),
                        "encoder_frames_maximum": max(row["encoder_frames"] for row in accepted),
                        "accepted_by_language": {language: {"clips": sum(row["language"] == language for row in accepted), "hours": sum(row["duration"] for row in accepted if row["language"] == language)/3600} for language in sorted({row["language"] for row in accepted})},
                        "accepted_source_ids_and_labels": accepted}
    excluded[group] = dropped

turkish_count = sum(summaries[key]["accepted_clips"] for key in ("turkish_commonvoice", "turkish_fleurs_train"))
replay_count = summaries["retention_fleurs_train"]["accepted_clips"]
assert turkish_count == config["turkish_clips_used"] == 2166
assert replay_count == config["replay_clips_used"] == 1553
assert sum(summaries[key]["excluded_clips"] for key in ("turkish_commonvoice", "turkish_fleurs_train")) == config["excluded_training_counts"]["tr"]
assert summaries["retention_fleurs_train"]["excluded_clips"] == config["excluded_training_counts"]["replay"]
heldout_paths = [ROOT / "work/data" / (split+"_"+language+".jsonl") for split in ("validation", "test") for language in ("en", "es", "it", "fr", "de", "tr")]
heldout = [row for path in heldout_paths for row in rows(path)]
assert len({row["source_id"] for row in accepted_rows}) == len(accepted_rows)
heldout_ids = {row["source_id"] for row in heldout}
heldout_sentences = {row["sentence_id"] for row in heldout if row.get("sentence_id") is not None}
heldout_audio = {row[key] for row in heldout for key in ("audio_sha256", "wav_sha256")}
assert not any(row["source_id"] in heldout_ids or row.get("sentence_id") in heldout_sentences or any(row[key] in heldout_audio for key in ("audio_sha256", "wav_sha256")) for row in accepted_rows)
paths = [config_path, audit_path, cv_path, supplement_path, replay_path, *heldout_paths]
report = {"verified_at_utc": datetime.now(timezone.utc).isoformat(), "passed": True, "device": "CPU only; no model/GPU loaded",
          "run_id": config["run_id"], "cache_namespace": namespace, "base_revision": config["base_revision"],
          "tokenizer_path": str(tokenizer_path), "tokenizer_sha256": sha256(tokenizer_path),
          "extended_tokenizer_semantic_sha256": vocabulary["extended_tokenizer_sha256"], "input_file_sha256": {str(path): sha256(path) for path in paths},
          "totals": {"turkish_clips": turkish_count, "turkish_hours": sum(summaries[key]["accepted_hours"] for key in ("turkish_commonvoice", "turkish_fleurs_train")), "replay_clips": replay_count, "replay_hours": summaries["retention_fleurs_train"]["accepted_hours"], "cached_labels_verified": len(accepted_rows)},
          "streams": summaries, "excluded_rows": excluded,
          "notes": ["Exact Hugging Face tokenizer class/settings reconstructed from the pinned processor and the same native BPE extension; semantic SHA256 matches the actual run vocabulary audit.", "The earlier bare-tokenizers research diagnostic is superseded by the active processor's max-180-token and lexical-roundtrip filters.", "These are unique usable training-pool hours, not elapsed training audio exposure; the optimizer samples repeated shuffled cycles with 50:50 Turkish/replay.", "Cached base-encoder features are teacher data only in this run; adapted student features must be computed from raw audio."]}
destination = OUTPUT / "lora_training_data_verification.json"
destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"passed": report["passed"], "run_id": report["run_id"], "cache_namespace": namespace, "totals": report["totals"], "streams": {key: {name: value for name, value in summary.items() if name != "accepted_source_ids_and_labels"} for key, summary in summaries.items()}}, indent=2))
