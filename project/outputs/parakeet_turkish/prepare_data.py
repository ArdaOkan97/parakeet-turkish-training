"""Prepare pinned Turkish training and multilingual FLEURS evaluation data.

Downloads only explicitly selected parquet shards, reads embedded audio with
PyArrow/SoundFile, and writes absolute-path NeMo JSONL manifests. No datasets
Audio decoder or torchcodec dependency is used. Re-running reuses pinned source
revisions and cached WAVs. All training data stays outside validation/test.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import HfApi, hf_hub_download
from scipy.signal import resample_poly

from metrics import normalize_text

FLEURS_REPO = "google/fleurs"
CV_REPO = "ysdede/commonvoice_17_tr_fixed"
CONFIGS = {"en": "en_us", "es": "es_419", "it": "it_it", "fr": "fr_fr", "de": "de_de", "tr": "tr_tr"}
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR.parent.parent


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps({key: value for key, value in record.items() if not key.startswith("_")}, ensure_ascii=False) + "\n")
    temporary.replace(path)


def resolve_sources(args: argparse.Namespace, data_dir: Path) -> dict[str, Any]:
    lock_path = data_dir / "sources.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.exists() else {}
    api = HfApi()
    for repo, requested in ((FLEURS_REPO, args.fleurs_revision), (CV_REPO, args.cv_revision)):
        prior = lock.get(repo)
        if prior and not args.refresh_sources and requested in {"main", prior["revision"]}:
            continue
        if args.offline:
            raise RuntimeError(f"Offline preparation requires an existing source lock for {repo}")
        info = api.dataset_info(repo, revision=requested)
        revision = info.sha
        if not revision:
            raise RuntimeError(f"Could not resolve a pinned revision for {repo}")
        files = api.list_repo_files(repo, repo_type="dataset", revision=revision)
        lock[repo] = {"revision": revision, "requested_revision": requested, "files": files}
        write_json(lock_path, lock)
        print(f"Pinned {repo} to {revision}", flush=True)
    return lock


def selected_parquets(source: dict[str, Any], split: str, config: str | None = None) -> list[str]:
    result = []
    for filename in source["files"]:
        parts = filename.replace("\\", "/").split("/")
        if not parts[-1].endswith(".parquet"):
            continue
        if not re.match(rf"^{re.escape(split)}(?:[-_.]|$)", parts[-1]):
            continue
        if config and config not in parts[:-1]:
            continue
        result.append(filename)
    if not result:
        raise RuntimeError(f"No explicit parquet files found for config={config!r}, split={split!r}")
    return sorted(result)


def collect_metadata(
    repo: str,
    config: str | None,
    split: str,
    language: str,
    source: dict[str, Any],
    data_dir: Path,
    offline: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records, provenance = [], []
    for filename in selected_parquets(source, split, config):
        print(f"Loading {repo}@{source['revision']} {filename}", flush=True)
        local = Path(hf_hub_download(
            repo_id=repo,
            repo_type="dataset",
            filename=filename,
            revision=source["revision"],
            cache_dir=str(data_dir / "hf_cache"),
            local_files_only=offline,
        )).resolve()
        parquet = pq.ParquetFile(local)
        names = parquet.schema_arrow.names
        if "audio" not in names:
            raise RuntimeError(f"Missing embedded audio column in {filename}")
        columns = [name for name in names if name != "audio"]
        provenance.append({"repository": repo, "revision": source["revision"], "filename": filename, "bytes": local.stat().st_size, "rows": parquet.metadata.num_rows})
        offset = 0
        for batch in parquet.iter_batches(batch_size=1024, columns=columns):
            for position, row in enumerate(batch.to_pylist()):
                source_path = str(row.get("path") or row.get("audio_id") or f"{filename}#row={offset + position}")
                sentence_id = row.get("id") if repo == FLEURS_REPO else None
                if repo == FLEURS_REPO and sentence_id is None:
                    raise RuntimeError(f"FLEURS sentence ID is absent at {filename}:{offset + position}")
                source_id = f"{repo}:{config or 'default'}:{split}:{sentence_id}:{source_path}"
                text = row.get("raw_transcription") or row.get("transcription") or row.get("sentence") or row.get("text")
                if not isinstance(text, str) or not normalize_text(text, language):
                    continue
                duration = row.get("duration")
                if duration is None and row.get("num_samples") is not None:
                    duration = row["num_samples"] / 16000
                records.append({
                    "dataset": repo,
                    "dataset_revision": source["revision"],
                    "dataset_config": config or "default",
                    "dataset_split": split,
                    "dataset_file": filename,
                    "source_id": source_id,
                    "sentence_id": sentence_id,
                    "original_audio_path": source_path,
                    "language": language,
                    "text": text,
                    "reference": text,
                    "reference_normalized": normalize_text(text, language),
                    "fleurs_transcription": row.get("transcription") if repo == FLEURS_REPO else None,
                    "gender": row.get("gender"),
                    "speaker_id": row.get("client_id") or row.get("speaker_id"),
                    "source_duration": float(duration) if duration is not None else None,
                    "_parquet": str(local),
                    "_row_index": offset + position,
                })
            offset += batch.num_rows
    identities = [record["source_id"] for record in records]
    if len(identities) != len(set(identities)):
        raise RuntimeError(f"Duplicate source IDs found in {repo}/{config}/{split}")
    return records, provenance


def stable_select(records: list[dict[str, Any]], count: int, seed: int, namespace: str) -> list[dict[str, Any]]:
    """Stable uniform hash ordering across the complete split, not a prefix."""
    def rank(record: dict[str, Any]) -> bytes:
        return hashlib.sha256(f"{seed}:{namespace}:{record['dataset_revision']}:{record['source_id']}".encode("utf-8")).digest()
    ordered = sorted(records, key=rank)
    if count < 0:
        raise ValueError("Sample counts cannot be negative")
    if count and len(ordered) < count:
        raise RuntimeError(f"Requested {count} {namespace} samples, but only {len(ordered)} remain after filtering")
    return ordered[:count] if count else ordered


def sentence_ids(records: list[dict[str, Any]]) -> set[int]:
    # Sentence IDs are parallel across FLEURS languages, so language must not
    # namespace this set. Different dataset repositories are separate namespaces.
    return {record["sentence_id"] for record in records if record["dataset"] == FLEURS_REPO}


def assert_split_isolation(train: list[dict[str, Any]], validation: list[dict[str, Any]], test: list[dict[str, Any]]) -> None:
    names = ("train", "validation", "test")
    splits = (train, validation, test)
    for i in range(3):
        for j in range(i + 1, 3):
            overlap = sentence_ids(splits[i]) & sentence_ids(splits[j])
            if overlap:
                raise RuntimeError(f"FLEURS sentence IDs overlap between {names[i]} and {names[j]}: {sorted(overlap)[:10]}")
            audio_overlap = {r["source_id"] for r in splits[i]} & {r["source_id"] for r in splits[j]}
            if audio_overlap:
                raise RuntimeError(f"Source audio IDs overlap between {names[i]} and {names[j]}")


def export_audio(records: list[dict[str, Any]], data_dir: Path, min_duration: float, max_duration: float) -> list[dict[str, Any]]:
    selected: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        selected[record["_parquet"]][record["_row_index"]] = record
    exported: dict[str, dict[str, Any]] = {}
    for parquet_path, positions in selected.items():
        parquet = pq.ParquetFile(parquet_path)
        offset = 0
        converted = 0
        for batch in parquet.iter_batches(batch_size=32, columns=["audio"]):
            # Avoid conversion of unselected row groups' scalar Python objects.
            relevant = [(position, record) for position, record in positions.items() if offset <= position < offset + batch.num_rows]
            for position, record in relevant:
                audio = batch.column(0)[position - offset].as_py()
                payload = audio.get("bytes") if isinstance(audio, dict) else None
                if payload is None:
                    path = audio.get("path") if isinstance(audio, dict) else None
                    if path and Path(path).is_file():
                        payload = Path(path).read_bytes()
                    else:
                        raise RuntimeError(f"Audio is not embedded in parquet for {record['source_id']}")
                if not payload:
                    raise RuntimeError(f"Audio bytes are empty for {record['source_id']}")
                source_hash = hashlib.sha256(payload).hexdigest()
                identifier = hashlib.sha256(f"{record['dataset_revision']}:{record['source_id']}".encode("utf-8")).hexdigest()[:24]
                wav_path = (data_dir / "audio" / record["language"] / f"{identifier}.wav").resolve()
                wav_path.parent.mkdir(parents=True, exist_ok=True)
                source_info = sf.info(io.BytesIO(payload))
                duration = source_info.frames / source_info.samplerate
                if record["dataset"] == CV_REPO and not min_duration <= duration <= max_duration:
                    print(f"Skipping CommonVoice duration {duration:.3f}s: {record['source_id']}", flush=True)
                    continue
                cached_valid = False
                if wav_path.exists():
                    try:
                        cached_info = sf.info(wav_path)
                        cached_valid = cached_info.samplerate == 16000 and cached_info.channels == 1 and cached_info.subtype == "PCM_16" and abs(cached_info.duration - duration) <= 1 / 16000 + 1e-6
                    except (RuntimeError, sf.LibsndfileError):
                        cached_valid = False
                if not cached_valid:
                    signal, sample_rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=True)
                    signal = signal.mean(axis=1)
                    if not signal.size or not np.isfinite(signal).all():
                        raise RuntimeError(f"Invalid audio signal for {record['source_id']}")
                    if sample_rate != 16000:
                        divisor = math.gcd(sample_rate, 16000)
                        signal = resample_poly(signal, 16000 // divisor, sample_rate // divisor)
                    temporary = wav_path.with_suffix(".tmp.wav")
                    sf.write(temporary, signal, 16000, subtype="PCM_16")
                    temporary.replace(wav_path)
                info = sf.info(wav_path)
                exported_record = dict(record)
                exported_record.update({
                    "audio_filepath": str(wav_path),
                    "duration": info.frames / info.samplerate,
                    "sample_rate": info.samplerate,
                    "audio_sha256": source_hash,
                    "wav_sha256": hashlib.sha256(wav_path.read_bytes()).hexdigest(),
                })
                exported[record["source_id"]] = exported_record
                converted += 1
                if converted % 500 == 0:
                    print(f"Converted/reused {converted}/{len(positions)} selected clips from {Path(parquet_path).name}", flush=True)
            offset += batch.num_rows
        print(f"Exported {converted} clips from {Path(parquet_path).name}", flush=True)
    return [exported[r["source_id"]] for r in records if r["source_id"] in exported]


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_language = {}
    for language in CONFIGS:
        subset = [r for r in records if r["language"] == language]
        if subset:
            by_language[language] = {"utterances": len(subset), "hours": sum(r["duration"] for r in subset) / 3600}
    return {"utterances": len(records), "hours": sum(r["duration"] for r in records) / 3600, "languages": by_language}


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = Path(args.data_dir).expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    sources = resolve_sources(args, data_dir)
    provenance: list[dict[str, Any]] = []
    cv_records, cv_provenance = collect_metadata(CV_REPO, None, "train", "tr", sources[CV_REPO], data_dir, args.offline)
    provenance.extend(cv_provenance)
    cv_filtered = [r for r in cv_records if r["source_duration"] is None or args.min_duration <= r["source_duration"] <= args.max_duration]
    tr_train = stable_select(cv_filtered, args.cv_limit, args.seed, "commonvoice_train")

    fleurs: dict[str, dict[str, list[dict[str, Any]]]] = {split: {} for split in ("train", "validation", "test")}
    for split in ("test", "validation", "train"):
        for language, config in CONFIGS.items():
            if split == "train" and language == "tr":
                continue
            records, split_provenance = collect_metadata(FLEURS_REPO, config, split, language, sources[FLEURS_REPO], data_dir, args.offline)
            fleurs[split][language] = records
            provenance.extend(split_provenance)

    tests = {language: stable_select(records, args.test_per_language, args.seed, f"fleurs_{language}_test") for language, records in fleurs["test"].items()}
    test = [record for language in CONFIGS for record in tests[language]]
    test_sentence_ids = sentence_ids(test)
    validation_excluded = {}
    validations = {}
    for language, records in fleurs["validation"].items():
        candidates = [r for r in records if r["sentence_id"] not in test_sentence_ids]
        validation_excluded[language] = len(records) - len(candidates)
        validations[language] = stable_select(candidates, args.validation_per_language, args.seed, f"fleurs_{language}_validation")
    validation = [record for language in CONFIGS for record in validations[language]]
    heldout_ids = test_sentence_ids | sentence_ids(validation)
    replay = []
    replay_excluded = {}
    for language, records in fleurs["train"].items():
        candidates = [r for r in records if r["sentence_id"] not in heldout_ids]
        replay_excluded[language] = len(records) - len(candidates)
        replay.extend(stable_select(candidates, args.replay_per_language, args.seed, f"fleurs_{language}_replay"))
    assert_split_isolation(tr_train + replay, validation, test)

    print(f"Exporting {len(tr_train)} Turkish training, {len(replay)} replay, {len(validation)} validation, {len(test)} test clips", flush=True)
    converted = export_audio(tr_train + replay + validation + test, data_dir, args.min_duration, args.max_duration)
    converted_by_id = {r["source_id"]: r for r in converted}
    def converted_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [converted_by_id[r["source_id"]] for r in records if r["source_id"] in converted_by_id]
    tr_train, replay, validation, test = map(converted_records, (tr_train, replay, validation, test))
    assert_split_isolation(tr_train + replay, validation, test)
    # Byte-identical audio across an optimizer/held-out boundary is also blocked.
    train_hashes = {r["audio_sha256"] for r in tr_train + replay}
    eval_hashes = {r["audio_sha256"] for r in validation + test}
    if train_hashes & eval_hashes:
        raise RuntimeError("Byte-identical training and held-out audio detected")
    if {r["audio_sha256"] for r in validation} & {r["audio_sha256"] for r in test}:
        raise RuntimeError("Byte-identical validation and test audio detected")

    manifests = {
        "tr_train": tr_train,
        "replay_train": replay,
        "train": tr_train + replay,
        "validation": validation,
        "test": test,
    }
    for split, records in (("validation", validation), ("test", test)):
        for language in CONFIGS:
            manifests[f"{split}_{language}"] = [r for r in records if r["language"] == language]
    manifest_info = {}
    for name, records in manifests.items():
        manifest_path = (data_dir / f"{name}.jsonl").resolve()
        write_jsonl(manifest_path, records)
        manifest_info[name] = {"path": str(manifest_path), "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(), **summarize(records)}

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir),
        "seed": args.seed,
        "sampling": "SHA256 ordering seeded by source ID, split, purpose, and pinned repository revision over complete selected-language split metadata",
        "sources": {repo: {key: value for key, value in source.items() if key != "files"} for repo, source in sources.items()},
        "configs": CONFIGS,
        "training_source": "CommonVoice mirror train only; FLEURS train only for five-language replay",
        "requested": {"cv_limit": args.cv_limit or "all", "replay_per_language": args.replay_per_language, "validation_per_language": args.validation_per_language, "test_per_language": args.test_per_language, "commonvoice_min_duration": args.min_duration, "commonvoice_max_duration": args.max_duration},
        "commonvoice_metadata_rows": len(cv_records),
        "commonvoice_duration_filtered_rows": len(cv_filtered),
        "fleurs_available_rows": {split: {language: len(records) for language, records in languages.items()} for split, languages in fleurs.items()},
        "fleurs_sentence_id_filtering": {"replay_rows_excluded": replay_excluded, "validation_rows_excluded": validation_excluded, "global_cross_language_sentence_id_overlap": 0},
        "split_isolation": "Optimizer, validation, and test FLEURS sentence IDs are globally disjoint across all six languages; source IDs and exact audio hashes checked",
        "speaker_metadata": "FLEURS exposes gender but does not expose speaker IDs; official test speaker separation is inherited from source splits",
        "audio_conversion": "Mono PCM16 WAV at 16 kHz; stereo downmix and polyphase resampling if needed; original source audio SHA256 and output WAV SHA256 retained",
        "normalization": "NFC; Turkish-aware lowercase; delete Unicode punctuation; collapse whitespace; preserve accents. Raw references and FLEURS normalized source references retained. This is a paired score normalizer, not a claim of official FST equivalence.",
        "manifests": manifest_info,
        "downloaded_parquet_files": provenance,
        "downloaded_parquet_bytes": sum(item["bytes"] for item in provenance),
        "limitations": ["Small fixed evaluation sets are pilot checks; use full official test splits to substantiate final retention claims.", "The five preservation languages cannot establish retention of all 25 languages.", "FLEURS is read speech and does not represent every production acoustic setting.", "CommonVoice source is a third-party processed CC0 mirror; its pinned revision is recorded."],
    }
    write_json(Path(args.summary_path).expanduser().resolve(), summary)
    print(json.dumps({"summary": str(Path(args.summary_path).resolve()), "train": manifest_info["train"], "test": manifest_info["test"]}, ensure_ascii=False, indent=2), flush=True)
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data-dir", default=str(WORKSPACE_DIR / "work" / "data"))
    result.add_argument("--summary-path", default=str(SCRIPT_DIR / "data_summary.json"))
    result.add_argument("--seed", type=int, default=20261010)
    result.add_argument("--cv-limit", type=int, default=0, help="Turkish train clips; 0 selects all train clips")
    result.add_argument("--replay-per-language", type=int, default=400)
    result.add_argument("--validation-per-language", type=int, default=50)
    result.add_argument("--test-per-language", type=int, default=100)
    result.add_argument("--min-duration", type=float, default=1.0)
    result.add_argument("--max-duration", type=float, default=20.0)
    result.add_argument("--fleurs-revision", default="main", help="Resolved and saved to an immutable HF commit on first run")
    result.add_argument("--cv-revision", default="main", help="Resolved and saved to an immutable HF commit on first run")
    result.add_argument("--refresh-sources", action="store_true", help="Resolve new revisions instead of reusing the saved lock")
    result.add_argument("--offline", action="store_true", help="Use only locked revisions and cached parquet files")
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    if arguments.min_duration <= 0 or arguments.max_duration < arguments.min_duration:
        raise SystemExit("Invalid CommonVoice duration bounds")
    if min(arguments.cv_limit, arguments.replay_per_language, arguments.validation_per_language, arguments.test_per_language) < 0:
        raise SystemExit("Sample counts must be nonnegative")
    prepare(arguments)
