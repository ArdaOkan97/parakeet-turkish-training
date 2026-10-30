"""Pinned official Whisper large-v3 on the existing Turkish DEV set only.

Greedy and beam-5 decoding are separate, fixed protocols. This script refuses
test manifests, verifies all50 fixed DEV source/reference/audio pairs against
the baseline DEV evidence, and records checkpoint/code hashes. It does not
use Whisper's normalizer: scores use the experiment's paired Turkish normalizer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, WhisperForConditionalGeneration

from metrics import normalize_text, score_transcriptions

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent.parent
REPOSITORY = "openai/whisper-large-v3"
REVISION = "06f233fe06e710322aca913c1bc4249a0d71fce1"


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


def verified_development_rows(manifest: Path) -> tuple[list[dict], dict]:
    expected = WORKSPACE / "work/data/validation_tr.jsonl"
    if manifest.resolve() != expected.resolve():
        raise RuntimeError("This benchmark is deliberately bound to the fixed Turkish validation manifest")
    examples = rows(manifest)
    baseline_path = ROOT / "results/base_v3/validation_tr.jsonl"
    original = rows(baseline_path)
    if len(examples) != 50 or len(original) != 50:
        raise RuntimeError("Expected exactly50 paired DEV recordings")
    by_id = {r["source_id"]: r for r in original}
    if len(by_id) != 50 or len({r["source_id"] for r in examples}) != 50:
        raise RuntimeError("Duplicate development source IDs")
    for record in examples:
        if record["language"] != "tr" or record["dataset"] != "google/fleurs" or record["dataset_split"] != "validation" or record["dataset_revision"] != "70bb2e84b976b7e960aa89f1c648e09c59f894dd":
            raise RuntimeError("Unexpected development dataset/language/split/revision")
        base = by_id[record["source_id"]]
        for key in ("reference", "sentence_id", "audio_sha256", "wav_sha256", "dataset_revision"):
            if record[key] != base[key]:
                raise RuntimeError(f"DEV reference/source pairing differs: {key}")
        if record["reference_normalized"] != normalize_text(record["reference"], "tr"):
            raise RuntimeError("Stored DEV reference normalization differs")
        wav = Path(record["audio_filepath"])
        info = sf.info(wav)
        if sha256(wav) != record["wav_sha256"] or info.samplerate != 16000 or info.channels != 1 or info.subtype != "PCM_16":
            raise RuntimeError("DEV WAV content/format differs")
        actual = info.frames / info.samplerate
        if abs(actual - record["duration"]) > 1 / 16000 or not 0 < actual <= 30:
            raise RuntimeError("DEV PCM duration differs or exceeds Whisper's short-form window")
    return examples, {"manifest_path": str(manifest.resolve()), "manifest_sha256": sha256(manifest), "paired_baseline_dev_path": str(baseline_path), "paired_baseline_dev_sha256": sha256(baseline_path), "checks": "50 unique TR DEV records; pinned source/references/sentence IDs/source and WAV hashes; exact existing PCM WAV durations <=30s", "heldout_test_predictions_accessed": False}


def main(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    lock_path = WORKSPACE / "work/strong_baseline_model.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock["repository"] != REPOSITORY or lock["revision"] != REVISION:
        raise RuntimeError("Strong baseline checkpoint lock differs")
    model_path = args.model_path.resolve()
    if Path(lock["local_path"]).resolve() != model_path or not (model_path / "model.safetensors").is_file():
        raise RuntimeError("Pinned FP16 official checkpoint is not completely downloaded")
    examples, input_audit = verified_development_rows(args.manifest)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists() and not args.force:
        raise RuntimeError("Completed baseline output exists; choose a separate tag or explicit --force")
    files = [model_path / "model.safetensors"] + sorted(p for p in model_path.iterdir() if p.is_file() and p.suffix in {".json", ".txt"} and not p.name.startswith(("pytorch_model", "model.safetensors.index")))
    files_sha = {p.name: sha256(p) for p in files}
    torch.manual_seed(20261010)
    np.random.seed(20261010)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(str(model_path), local_files_only=True, use_safetensors=True, dtype=dtype, attn_implementation="sdpa").to(device).eval()
    model.config._name_or_path = f"{REPOSITORY}@{REVISION}"
    protocol = {"language": "tr", "task": "transcribe", "return_timestamps": False, "return_token_timestamps": False, "return_dict_in_generate": True, "num_beams": args.num_beams, "do_sample": False, "temperature": 0.0, "condition_on_prev_tokens": False, "max_new_tokens": 440, "compression_ratio_threshold": None, "logprob_threshold": None, "no_speech_threshold": None}
    write_json(output / "status.json", {"stage": "running", "repository": REPOSITORY, "revision": REVISION, "started_at_utc": datetime.now(timezone.utc).isoformat(), "total": len(examples), "protocol": protocol})
    predictions = []
    prediction_path = output / "validation_tr.jsonl"
    with prediction_path.open("w", encoding="utf-8", newline="\n") as stream:
        for index, record in enumerate(examples, 1):
            signal, sample_rate = sf.read(record["audio_filepath"], dtype="float32")
            if sample_rate != 16000 or not np.isfinite(signal).all():
                raise RuntimeError("Unexpected/nonfinite DEV input")
            features = processor(signal, sampling_rate=sample_rate, return_tensors="pt", return_attention_mask=True)
            inputs = {key: value.to(device=device, dtype=dtype if key == "input_features" else value.dtype) for key, value in features.items()}
            if device.type == "cuda":
                torch.cuda.synchronize()
            clip_start = time.perf_counter()
            with torch.inference_mode():
                generated_output = model.generate(**inputs, **protocol)
                generated = generated_output.sequences
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - clip_start
            token_ids = generated[0].tolist()
            hypothesis = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
            eos_ids = model.generation_config.eos_token_id
            eos_ids = [eos_ids] if isinstance(eos_ids, int) else list(eos_ids)
            prediction = {key: record[key] for key in ("source_id", "sentence_id", "language", "reference", "dataset_revision", "audio_sha256", "wav_sha256", "duration")}
            prediction.update({"split": "validation", "hypothesis": hypothesis, "generated_token_ids": token_ids, "eos_emitted": any(token in eos_ids for token in token_ids), "inference_seconds": elapsed})
            predictions.append(prediction)
            stream.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            stream.flush()
            if index % 5 == 0 or index == len(examples):
                current = score_transcriptions(predictions, language="tr")
                print(json.dumps({"completed": index, "total": len(examples), "partial_dev_wer": current["wer"], "partial_dev_cer": current["cer"], "elapsed_seconds": time.perf_counter() - started}), flush=True)
                write_json(output / "status.json", {"stage": "running", "completed": index, "total": len(examples), "updated_at_utc": datetime.now(timezone.utc).isoformat()})
    metrics = score_transcriptions(predictions, language="tr")
    metrics.pop("details")
    elapsed = sum(r["inference_seconds"] for r in predictions)
    seconds = sum(r["duration"] for r in predictions)
    summary = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "complete", "repository": REPOSITORY, "revision": REVISION, "split": "validation", "language": "tr", "protocol": protocol, "metrics": metrics, "input_audit": input_audit,
        "model_files_sha256": files_sha, "model_lock_sha256": sha256(lock_path), "script_sha256": sha256(Path(__file__)), "metrics_code_sha256": sha256(ROOT / "metrics.py"), "predictions_sha256": sha256(prediction_path),
        "device": str(device), "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU", "weight_compute_dtype": str(dtype), "attention_implementation": "sdpa", "torch_version": torch.__version__, "transformers_version": __import__("transformers").__version__,
        "peak_allocated_vram_gb": torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else None, "audio_seconds": seconds, "inference_seconds": elapsed, "real_time_factor": elapsed / seconds, "elapsed_seconds_with_load_and_audit": time.perf_counter() - started,
        "competitive_dev_target_provisional": {"rule": "Adapted Turkish dev WER <= this fixed baseline dev WER +0.02, while each five-language original-v3 retention gate passes", "max_adapted_dev_wer": metrics["wer"] + 0.02, "basis": "Measured same50 development recordings; provisional small-set target, not a claim of competitive held-out quality"},
        "limitations": ["50 recordings provide a noisy development reference; baseline and adaptive experiments share only DEV for tuning.", "Greedy and beam5 are separate decoding protocols and must not be silently pooled.", "Same experiment normalizer is used for fair pairing; scores are not claimed equivalent to published Whisper/FLEURS normalizers.", "A final held-out comparison would require separately authorized evaluation and paired uncertainty reporting."]}
    write_json(output / "summary.json", summary)
    write_json(output / "status.json", {"stage": "complete", "completed": len(predictions), "summary_sha256": sha256(output / "summary.json")})
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(json.dumps({"output": str(output), "metrics": metrics, "peak_allocated_vram_gb": summary["peak_allocated_vram_gb"], "real_time_factor": summary["real_time_factor"], "gpu_released": True}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=WORKSPACE / "work/models/whisper-large-v3")
    parser.add_argument("--manifest", type=Path, default=WORKSPACE / "work/data/validation_tr.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/strong_whisper_large_v3_greedy_dev")
    parser.add_argument("--num-beams", type=int, choices=[1, 5], default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    main(parser.parse_args())
