"""CPU-only verification of every exported manifest and audio file."""
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs/parakeet_turkish"
summary = json.loads((OUTPUT / "data_summary.json").read_text(encoding="utf-8"))
DATA = Path(summary["data_dir"])
manifests = {}
for name, expected in summary["manifests"].items():
    path = Path(expected["path"])
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == expected["sha256"], name
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    assert len(rows) == expected["utterances"], name
    assert len({r["source_id"] for r in rows}) == len(rows), name
    for lang, count in Counter(r["language"] for r in rows).items():
        assert count == expected["languages"][lang]["utterances"], (name, lang)
    assert abs(sum(r["duration"] for r in rows) / 3600 - expected["hours"]) < 1e-10
    manifests[name] = rows

def by_id(rows):
    return {r["source_id"]: r for r in rows}

assert by_id(manifests["tr_train"] + manifests["replay_train"]) == by_id(manifests["train"])
for split, size in [("validation", 50), ("test", 100)]:
    combined = []
    for lang in ("en", "es", "it", "fr", "de", "tr"):
        rows = manifests[f"{split}_{lang}"]
        assert len(rows) == size
        assert all(r["dataset"] == "google/fleurs" and r["dataset_split"] == split and r["language"] == lang for r in rows)
        combined += rows
    assert by_id(combined) == by_id(manifests[split])
assert all(r["dataset"] == "ysdede/commonvoice_17_tr_fixed" and r["dataset_split"] == "train" and 1 <= r["duration"] <= 15 for r in manifests["tr_train"])
assert Counter(r["language"] for r in manifests["replay_train"]) == {"en": 400, "es": 400, "it": 400, "fr": 400, "de": 400}
assert all(r["dataset"] == "google/fleurs" and r["dataset_split"] == "train" for r in manifests["replay_train"])

parts = [manifests[name] for name in ("train", "validation", "test")]
for i in range(len(parts)):
    for j in range(i + 1, len(parts)):
        for key in ("source_id", "audio_sha256", "wav_sha256"):
            assert not {r[key] for r in parts[i]} & {r[key] for r in parts[j]}, (i, j, key)
        sentences_i = {r["sentence_id"] for r in parts[i] if r["dataset"] == "google/fleurs"}
        sentences_j = {r["sentence_id"] for r in parts[j] if r["dataset"] == "google/fleurs"}
        assert not sentences_i & sentences_j, (i, j, "sentence_id")

all_rows = manifests["train"] + manifests["validation"] + manifests["test"]
total_bytes = 0
for row in all_rows:
    path = Path(row["audio_filepath"])
    assert path.is_absolute() and path.is_file(), path
    info = sf.info(path)
    assert info.samplerate == 16000 and info.channels == 1 and info.subtype == "PCM_16", path
    assert abs(info.duration - row["duration"]) < 1e-7, path
    audio = path.read_bytes()
    assert hashlib.sha256(audio).hexdigest() == row["wav_sha256"], path
    total_bytes += len(audio)
    assert row["dataset_revision"] == summary["sources"][row["dataset"]]["revision"]

original = Tokenizer.from_file(str(ROOT / "work/research/tokenizer.json"))
extended = Tokenizer.from_file(str(ROOT / "work/research/extension_roundtrip/tokenizer.json"))
added_in_replay = [r["source_id"] for r in manifests["replay_train"] if any(t >= original.get_vocab_size() for t in extended.encode(r["text"]).ids)]
exact_ref_overlap = []
training_refs = {r["reference_normalized"] for r in manifests["train"]}
for row in manifests["validation"] + manifests["test"]:
    if row["reference_normalized"] in training_refs:
        exact_ref_overlap.append(row["source_id"])

report = {
    "verified_at_utc": datetime.now(timezone.utc).isoformat(),
    "device": "CPU only",
    "verified_manifest_files": len(manifests),
    "verified_unique_audio_files": len(all_rows),
    "verified_audio_bytes": total_bytes,
    "checks": ["All manifest byte hashes and summary counts/hours match", "Combined and per-language manifests match exactly", "All actual audio hashes, headers, durations, and dataset revisions match", "CommonVoice uses train only within 1-15 seconds", "Replay uses only official FLEURS train and exactly 400 examples per language", "50 development and 100 test examples for each of six languages", "Training/development/test source IDs, exact source/output audio hashes and global FLEURS sentence IDs are disjoint"],
    "turkish_clips": len(manifests["tr_train"]),
    "replay_clips": len(manifests["replay_train"]),
    "validation_clips": len(manifests["validation"]),
    "test_clips": len(manifests["test"]),
    "replay_duration_max_seconds": max(r["duration"] for r in manifests["replay_train"]),
    "replay_using_appended_tokens": len(added_in_replay),
    "replay_using_appended_tokens_source_ids": added_in_replay,
    "exact_normalized_reference_overlap_train_vs_heldout": exact_ref_overlap,
}
(OUTPUT / "data_verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
