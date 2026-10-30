"""Prepare an isolated, longer Turkish FLEURS train supplement from local cache.

Prefetch google/fleurs@70bb2e84b976b7e960aa89f1c648e09c59f894dd:
parquet-data/tr_tr/train-00000-of-00001.parquet into work/data/hf_cache.
This script never downloads or rewrites existing data/benchmark manifests.
Tokenizer checks are diagnostic; the next trainer applies its own label filter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf
from tokenizers import Tokenizer

from prepare_data import (FLEURS_REPO, WORKSPACE_DIR, collect_metadata, stable_select,
                          export_audio, sentence_ids, assert_split_isolation, summarize,
                          write_json, write_jsonl)
from metrics import normalize_text

REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
OUTPUT = Path(__file__).resolve().parent
DATA = WORKSPACE_DIR / "work/data"


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-hours", type=float, default=1.5)
    parser.add_argument("--min-seconds", type=float, default=10)
    parser.add_argument("--max-seconds", type=float, default=25)
    parser.add_argument("--min-clips", type=int, default=300)
    parser.add_argument("--max-clips", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261010)
    args = parser.parse_args()
    assert 0 < args.min_seconds <= args.max_seconds and 0 < args.min_clips <= args.max_clips
    assert args.target_hours > 0
    existing = sorted(DATA.glob("*.jsonl"))
    supplement_path = DATA / "tr_fleurs_train.jsonl"
    existing = [path for path in existing if path != supplement_path]
    existing += [DATA / "sources.lock.json", OUTPUT / "data_summary.json"]
    existing += sorted((OUTPUT / "results/base_v3").glob("*.jsonl"))
    existing += [OUTPUT / "results/base_v3/summary.json", OUTPUT / "results/base_v3/metadata.json"]
    immutable_before = {str(path.resolve()): sha(path) for path in existing if path.is_file()}
    lock = json.loads((DATA / "sources.lock.json").read_text(encoding="utf-8"))
    source = lock[FLEURS_REPO]
    assert source["revision"] == REVISION
    validation, test = read_jsonl(DATA / "validation.jsonl"), read_jsonl(DATA / "test.jsonl")
    heldout = validation + test
    heldout_ids = sentence_ids(heldout)
    heldout_refs = {normalize_text(row["text"], "tr") for row in heldout}
    heldout_source_hashes = {row["audio_sha256"] for row in heldout}
    heldout_wav_hashes = {row["wav_sha256"] for row in heldout}
    rows, provenance = collect_metadata(FLEURS_REPO, "tr_tr", "train", "tr", source, DATA, offline=True)
    exclusions = {"heldout_global_sentence_ids": 0, "heldout_exact_normalized_reference": 0,
                  "metadata_duration": 0, "decoded_duration": 0, "heldout_exact_audio": 0,
                  "duplicate_selected_audio": 0}
    candidates = []
    for row in rows:
        if row["sentence_id"] in heldout_ids:
            exclusions["heldout_global_sentence_ids"] += 1
            continue
        text = unicodedata.normalize("NFC", row["text"])
        if normalize_text(text, "tr") in heldout_refs:
            exclusions["heldout_exact_normalized_reference"] += 1
            continue
        if row["source_duration"] is None or not args.min_seconds <= row["source_duration"] <= args.max_seconds:
            exclusions["metadata_duration"] += 1
            continue
        candidates.append({**row, "text": text, "reference": text, "reference_normalized": normalize_text(text, "tr")})
    assert len(candidates) >= args.min_clips, len(candidates)
    ordered = stable_select(candidates, 0, args.seed, "tr_fleurs_train_longer_supplement")
    selected, offset = [], 0
    seen_source_hashes, seen_wav_hashes = set(), set()
    target_seconds = args.target_hours * 3600
    while len(selected) < args.max_clips and (len(selected) < args.min_clips or sum(row["duration"] for row in selected) < target_seconds):
        pending, nominal_seconds = [], sum(row["duration"] for row in selected)
        while offset < len(ordered) and len(selected) + len(pending) < args.max_clips:
            row = ordered[offset]
            offset += 1
            pending.append(row)
            nominal_seconds += row["source_duration"]
            if len(selected) + len(pending) >= args.min_clips and nominal_seconds >= target_seconds:
                break
        if not pending:
            break
        for row in export_audio(pending, DATA, args.min_seconds, args.max_seconds):
            if not args.min_seconds <= row["duration"] <= args.max_seconds:
                exclusions["decoded_duration"] += 1
                continue
            if row["audio_sha256"] in heldout_source_hashes or row["wav_sha256"] in heldout_wav_hashes:
                exclusions["heldout_exact_audio"] += 1
                continue
            if row["audio_sha256"] in seen_source_hashes or row["wav_sha256"] in seen_wav_hashes:
                exclusions["duplicate_selected_audio"] += 1
                continue
            selected.append(row)
            seen_source_hashes.add(row["audio_sha256"])
            seen_wav_hashes.add(row["wav_sha256"])
    assert args.min_clips <= len(selected) <= args.max_clips
    assert_split_isolation(selected, validation, test)
    assert not {row["source_id"] for row in selected} & {row["source_id"] for row in heldout}
    assert not {row["reference_normalized"] for row in selected} & heldout_refs
    for row in selected:
        info = sf.info(row["audio_filepath"])
        assert info.samplerate == 16000 and info.channels == 1 and info.subtype == "PCM_16"
        assert abs(info.duration - row["duration"]) < 1e-7
        assert sha(row["audio_filepath"]) == row["wav_sha256"]
        assert row["text"] == unicodedata.normalize("NFC", row["text"])
    write_jsonl(supplement_path, selected)
    immutable_after = {path: sha(path) for path in immutable_before}
    assert immutable_after == immutable_before, "An original manifest/baseline changed during preparation"
    tokenizer_path = WORKSPACE_DIR / "work/research/extension_roundtrip/tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    lexical_mismatch = over_token_limit = 0
    usable = []
    for row in selected:
        ids = tokenizer.encode(row["text"]).ids
        lexical_ok = normalize_text(tokenizer.decode(ids), "tr") == row["reference_normalized"]
        lexical_mismatch += not lexical_ok
        over_token_limit += not 0 < len(ids) <= 180
        if lexical_ok and 0 < len(ids) <= 180:
            usable.append(row)
    replay_ids = sentence_ids(read_jsonl(DATA / "replay_train.jsonl"))
    audit = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(), "device": "CPU only",
        "scope": "Optional separate official Turkish FLEURS TRAIN supplement; no optimizer run or quality claim",
        "manifest": str(supplement_path.resolve()), "manifest_sha256": sha(supplement_path),
        "source": {"repository": FLEURS_REPO, "revision": REVISION, "config": "tr_tr", "split": "train", "files": provenance},
        "source_metadata_rows": len(rows), "eligible_longer_rows": len(candidates), "exclusions": exclusions,
        "selection": {**vars(args), "method": "Stable SHA256 ordering over all filtered official train metadata, then a bounded prefix reaching target hours; exact audio/duration filtering after decode"},
        "summary": summarize(selected),
        "durations_seconds": {"minimum": min(row["duration"] for row in selected), "maximum": max(row["duration"] for row in selected), "mean": statistics.mean(row["duration"] for row in selected)},
        "global_heldout_sentence_id_overlap": 0, "heldout_exact_reference_overlap": 0,
        "heldout_exact_source_or_wav_audio_overlap": 0,
        "overlap_with_existing_replay_train_sentence_ids": len(sentence_ids(selected) & replay_ids),
        "replay_overlap_policy": "Parallel source sentences may occur across training languages; only train/heldout overlap is excluded",
        "immutable_existing_files": immutable_before, "immutable_existing_files_verified": len(immutable_before),
        "tokenizer_diagnostic_only": {"path": str(tokenizer_path), "sha256": sha(tokenizer_path), "lexical_roundtrip_mismatches": lexical_mismatch, "empty_or_over_180_tokens": over_token_limit, "representable_within_180": summarize(usable), "policy": "No tokenizer-based exclusion was applied to the supplemental manifest. The next trainer validates/filter labels using its actual processor and configuration."},
        "checks": ["Official pinned tr_tr train provenance only", "300-500 longer mono PCM16/16 kHz clips with every WAV byte hash verified", "Global heldout source/sentence/reference/audio disjointness", "Every original manifest, source lock, data summary and baseline result file retained identical SHA256", "NFC references retained and tokenizer limitations reported without silently rewriting human words"],
        "passed": True,
    }
    write_json(OUTPUT / "turkish_fleurs_supplement_audit.json", audit)
    print(json.dumps({"clips": len(selected), "hours": audit["summary"]["hours"], "mean_duration": audit["durations_seconds"]["mean"], "tokenizer_usable_clips": len(usable), "immutable_files": len(immutable_before)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
