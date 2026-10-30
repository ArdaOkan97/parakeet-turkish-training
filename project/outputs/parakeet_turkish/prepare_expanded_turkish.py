"""Offline, pinned Turkish train expansion; original data and helpers are read-only.

Writes a separate Common Voice audio directory and manifest. The accepted pilot
FLEURS train supplement is retained after the same held-out/dedup checks. Exact
PCM duration and byte hashes are recorded; tokenizer/OOV acceptance remains the
trainer's responsibility and is not claimed by this preparation step.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import resample_poly

from metrics import normalize_text

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent.parent
CV_REPO = "ysdede/commonvoice_17_tr_fixed"
CV_REVISION = "9d8a025f1ae7c5a2f5c43298c7c4910b03995a4b"
CV_FILE = "data/train-00000-of-00001.parquet"
FLEURS_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def heldout_filters(data_dir: Path) -> tuple[dict, list[Path]]:
    paths = sorted(p for p in data_dir.glob("*.jsonl") if p.name.startswith(("validation", "test")))
    if not paths:
        raise RuntimeError("No fixed held-out manifests found; refusing unisolated expansion")
    all_rows = [r for path in paths for r in rows(path)]
    filters = {
        "source_ids": {r["source_id"] for r in all_rows},
        "source_audio_hashes": {r["audio_sha256"] for r in all_rows},
        "wav_hashes": {r["wav_sha256"] for r in all_rows},
        "sentence_ids": {int(r["sentence_id"]) for r in all_rows if r.get("dataset") == "google/fleurs" and r.get("sentence_id") is not None},
        "normalized_references": {normalize_text(r["reference"], lang) for r in all_rows for lang in (r["language"], "tr")},
    }
    for path in paths:
        for record in rows(path):
            if record.get("dataset_split", record.get("split")) not in {"validation", "test"}:
                raise RuntimeError(f"Unexpected held-out split in {path}")
    return filters, paths


def forbidden_reason(record: dict, filters: dict) -> str | None:
    if record["source_id"] in filters["source_ids"]:
        return "heldout_source_id"
    if record.get("sentence_id") is not None and record.get("dataset") == "google/fleurs" and int(record["sentence_id"]) in filters["sentence_ids"]:
        return "heldout_global_fleurs_sentence_id"
    if normalize_text(record["reference"], "tr") in filters["normalized_references"]:
        return "heldout_normalized_reference"
    if record.get("audio_sha256") in filters["source_audio_hashes"]:
        return "heldout_source_audio_hash"
    if record.get("wav_sha256") in filters["wav_hashes"]:
        return "heldout_wav_hash"
    return None


def prepare(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    data_dir = args.data_dir.resolve()
    target = data_dir / "tr_expanded_train.jsonl"
    audio_dir = data_dir / "expanded_audio" / "tr"
    audio_dir.mkdir(parents=True, exist_ok=True)
    lock_path = data_dir / "sources.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock[CV_REPO]["revision"] != CV_REVISION or lock["google/fleurs"]["revision"] != FLEURS_REVISION:
        raise RuntimeError("Pinned source lock differs from the audited pilot")
    parquet_path = data_dir / "hf_cache" / "datasets--ysdede--commonvoice_17_tr_fixed" / "snapshots" / CV_REVISION / CV_FILE
    if not parquet_path.is_file():
        raise RuntimeError("Pinned cached CV train parquet is absent; no implicit network/download fallback")
    filters, heldout_paths = heldout_filters(data_dir)
    protected = sorted(set(data_dir.glob("*.jsonl")) - {target}) + [lock_path]
    protected += [ROOT / name for name in ("prepare_data.py", "metrics.py", "runtime.py", "inference.py", "benchmark.py", "train_lora.py", "efficient_tdt.py", "extend_vocab.py", "encoder_lora.py") if (ROOT / name).is_file()]
    original_sha = {str(p.relative_to(WORKSPACE)).replace("\\", "/"): sha256(p) for p in protected}
    original_files = {str(p.relative_to(WORKSPACE)).replace("\\", "/"): p for p in protected}
    source_sha = sha256(parquet_path)
    parquet = pq.ParquetFile(parquet_path)
    excludes = Counter()
    diagnostics = []
    candidates = []
    metadata = parquet.read(columns=[name for name in parquet.schema_arrow.names if name != "audio"]).to_pylist()
    for index, row in enumerate(metadata):
        text = row.get("transcription") or row.get("sentence") or row.get("text")
        normalized = normalize_text(text, "tr") if isinstance(text, str) else ""
        if not normalized:
            excludes["empty_reference"] += 1
            continue
        source_id = f"{CV_REPO}:default:train:None:{CV_FILE}#row={index}"
        record = {"dataset": CV_REPO, "dataset_revision": CV_REVISION, "dataset_config": "default", "dataset_split": "train", "dataset_file": CV_FILE, "source_id": source_id, "sentence_id": None, "original_audio_path": f"{CV_FILE}#row={index}", "language": "tr", "text": text, "reference": text, "reference_normalized": normalized, "fleurs_transcription": None, "gender": row.get("gender"), "speaker_id": row.get("client_id") or row.get("speaker_id"), "source_duration": row.get("duration"), "_row_index": index}
        reason = forbidden_reason(record, filters)
        if reason:
            excludes[reason] += 1
            continue
        candidates.append(record)
    candidates.sort(key=lambda r: hashlib.sha256(f"{args.seed}:expanded_cv_train:{CV_REVISION}:{r['source_id']}".encode()).digest())
    order = {r["_row_index"]: i for i, r in enumerate(candidates)}
    # A deterministic rank decides duplicate winners independent of parquet batches.
    seen_refs = set()
    selected = {}
    for record in candidates:
        normalized = record["reference_normalized"]
        if normalized in seen_refs:
            excludes["duplicate_training_normalized_reference"] += 1
            continue
        seen_refs.add(normalized)
        selected[record["_row_index"]] = record
    exported = []
    offset = 0
    seen_source_hashes = set()
    seen_wav_hashes = set()
    # Hash duplicates are resolved again after conversion in deterministic rank.
    converted = []
    for batch in parquet.iter_batches(batch_size=64, columns=["audio"]):
        for position in range(batch.num_rows):
            index = offset + position
            record = selected.get(index)
            if record is None:
                continue
            try:
                audio = batch.column(0)[position].as_py()
                payload = audio.get("bytes") if isinstance(audio, dict) else None
                if not payload:
                    raise ValueError("Missing embedded audio")
                source_hash = hashlib.sha256(payload).hexdigest()
                if source_hash in filters["source_audio_hashes"]:
                    excludes["heldout_source_audio_hash"] += 1
                    continue
                signal, sample_rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=True)
                if not signal.size or not np.isfinite(signal).all():
                    raise ValueError("Empty/nonfinite decoded signal")
                signal = signal.mean(axis=1)
                if sample_rate != 16000:
                    divisor = math.gcd(int(sample_rate), 16000)
                    signal = resample_poly(signal, 16000 // divisor, int(sample_rate) // divisor)
                duration = len(signal) / 16000
                if not args.min_duration <= duration <= args.max_duration:
                    excludes["actual_pcm_duration_outside_bounds"] += 1
                    continue
                identifier = hashlib.sha256(f"{CV_REVISION}:{record['source_id']}".encode()).hexdigest()[:24]
                path = (audio_dir / f"{identifier}.wav").resolve()
                # Always build expected bytes, so a stale/corrupt cached file cannot silently pass.
                buffer = io.BytesIO()
                sf.write(buffer, signal, 16000, format="WAV", subtype="PCM_16")
                wav_bytes = buffer.getvalue()
                wav_hash = hashlib.sha256(wav_bytes).hexdigest()
                if wav_hash in filters["wav_hashes"]:
                    excludes["heldout_wav_hash"] += 1
                    continue
                if not path.is_file() or sha256(path) != wav_hash:
                    temporary = path.with_suffix(".tmp.wav")
                    temporary.write_bytes(wav_bytes)
                    temporary.replace(path)
                info = sf.info(path)
                if info.samplerate != 16000 or info.channels != 1 or info.subtype != "PCM_16" or info.frames != len(signal):
                    raise ValueError("Exported PCM format/length mismatch")
                converted.append({**record, "audio_filepath": str(path), "duration": info.frames / info.samplerate, "sample_rate": 16000, "audio_sha256": source_hash, "wav_sha256": wav_hash})
            except (ValueError, RuntimeError, sf.LibsndfileError) as error:
                excludes["invalid_audio"] += 1
                if len(diagnostics) < 20:
                    diagnostics.append({"source_id": record["source_id"], "error": str(error)})
        offset += batch.num_rows
        if offset % 2048 == 0:
            print(f"Expanded CV decoded {offset}/{parquet.metadata.num_rows}, exported candidates {len(converted)}", flush=True)
    converted.sort(key=lambda r: order[r["_row_index"]])
    for record in converted:
        if record["audio_sha256"] in seen_source_hashes or record["wav_sha256"] in seen_wav_hashes:
            excludes["duplicate_training_audio"] += 1
            continue
        seen_source_hashes.add(record["audio_sha256"])
        seen_wav_hashes.add(record["wav_sha256"])
        exported.append(record)
    cv_count = len(exported)
    cv_hours = sum(r["duration"] for r in exported) / 3600
    fleurs_included = []
    audit_path = ROOT / "lora_training_data_verification.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("passed") or audit["base_revision"] != "541d1f99c6b0c3cd0b11a95167540bb8edefd82b":
        raise RuntimeError("Pilot training data audit is absent or does not match")
    supplement = {r["source_id"]: r for r in rows(data_dir / "tr_fleurs_train.jsonl")}
    accepted = audit["streams"]["turkish_fleurs_train"]["accepted_source_ids_and_labels"]
    for cached in accepted:
        record = supplement[cached["source_id"]]
        if record["dataset_split"] != "train" or record["dataset_revision"] != FLEURS_REVISION:
            raise RuntimeError("Unexpected FLEURS supplement source")
        reason = forbidden_reason(record, filters)
        if reason:
            excludes[reason] += 1
            continue
        if record["reference_normalized"] in {r["reference_normalized"] for r in exported}:
            excludes["duplicate_training_normalized_reference"] += 1
            continue
        if record["audio_sha256"] in seen_source_hashes or record["wav_sha256"] in seen_wav_hashes:
            excludes["duplicate_training_audio"] += 1
            continue
        path = Path(record["audio_filepath"])
        info = sf.info(path)
        if sha256(path) != record["wav_sha256"] or info.samplerate != 16000 or info.channels != 1 or info.subtype != "PCM_16":
            raise RuntimeError("Previously audited FLEURS WAV differs")
        duration = info.frames / info.samplerate
        if not args.min_duration <= duration <= args.max_duration:
            excludes["actual_pcm_duration_outside_bounds"] += 1
            continue
        record = {**record, "duration": duration}
        fleurs_included.append(record)
        exported.append(record)
        seen_source_hashes.add(record["audio_sha256"])
        seen_wav_hashes.add(record["wav_sha256"])
    if any(forbidden_reason(record, filters) for record in exported):
        raise RuntimeError("Final expanded training/held-out exclusion failed")
    if len({r["source_id"] for r in exported}) != len(exported) or len({r["reference_normalized"] for r in exported}) != len(exported):
        raise RuntimeError("Final source/reference uniqueness failed")
    unchanged = {name: sha256(path) == original_sha[name] for name, path in original_files.items()}
    if not all(unchanged.values()) or sha256(parquet_path) != source_sha:
        raise RuntimeError("A protected original manifest/helper/source changed during preparation")
    temporary = target.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for record in exported:
            stream.write(json.dumps({k: v for k, v in record.items() if not k.startswith("_")}, ensure_ascii=False) + "\n")
    temporary.replace(target)
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "prepared_audio_and_manifest; trainer_tokenizer_acceptance_pending", "device": "CPU only; cached source; no network or GPU",
        "seed": args.seed, "source_repository": CV_REPO, "source_revision": CV_REVISION, "source_filename": CV_FILE, "source_parquet_sha256": source_sha, "source_parquet_bytes": parquet_path.stat().st_size, "source_rows": parquet.metadata.num_rows,
        "selection": "Complete pinned CV train split in seeded SHA256 rank; one normalized-reference/audio winner; existing audited pilot FLEURS train supplement follows",
        "duration_bounds_seconds": [args.min_duration, args.max_duration], "duration_measurement": "Actual decoded, downmixed/resampled 16kHz PCM frame count, never metadata duration",
        "manifest_path": str(target.resolve()), "manifest_sha256": sha256(target), "prepared_clips": len(exported), "prepared_hours": sum(r["duration"] for r in exported) / 3600,
        "commonvoice_clips": cv_count, "commonvoice_hours": cv_hours, "fleurs_supplement_clips": len(fleurs_included), "fleurs_supplement_hours": sum(r["duration"] for r in fleurs_included) / 3600,
        "exclusions": dict(excludes), "invalid_audio_examples": diagnostics,
        "heldout_manifest_paths": [str(p.relative_to(WORKSPACE)).replace("\\", "/") for p in heldout_paths], "heldout_unique_source_ids": len(filters["source_ids"]), "heldout_global_fleurs_sentence_ids": len(filters["sentence_ids"]), "heldout_normalized_reference_keys": len(filters["normalized_references"]),
        "isolation": "Zero overlap with all fixed validation/test source IDs, source audio hashes, WAV hashes, global FLEURS sentence IDs, and normalized references; no held-out predictions read",
        "protected_original_files_sha256": original_sha, "protected_original_files_unchanged": unchanged, "source_parquet_unchanged": True, "preparation_script_sha256": sha256(Path(__file__)), "metrics_sha256": sha256(ROOT / "metrics.py"),
        "lexical_filtering": "NOT performed here: retain raw Turkish characters; trainer must apply the exact extended HF tokenizer roundtrip/blank/duration/OOV and sequence-length checks and record accepted counts/hours",
        "limitations": ["Prepared hours are not optimizer-accepted hours.", "This processed public Common Voice mirror has no client/speaker IDs; speaker diversity cannot be quantified from the exported schema.", "Text/audio label quality inherits the pinned third-party source; no manual relabeling is claimed."], "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(args.summary_path.resolve(), result)
    print(json.dumps({k: result[k] for k in ("status", "prepared_clips", "prepared_hours", "commonvoice_clips", "fleurs_supplement_clips", "manifest_path", "manifest_sha256", "elapsed_seconds")}, ensure_ascii=False), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=WORKSPACE / "work/data")
    parser.add_argument("--summary-path", type=Path, default=ROOT / "expanded_turkish_preparation.json")
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--min-duration", type=float, default=0.5)
    parser.add_argument("--max-duration", type=float, default=30.0)
    args = parser.parse_args()
    if not 0 < args.min_duration <= args.max_duration <= 30:
        parser.error("Duration bounds must satisfy 0 < min <= max <=30 seconds")
    prepare(args)
