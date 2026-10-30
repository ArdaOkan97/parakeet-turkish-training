"""Versioned fixed Turkish TEST baseline with full-audio Whisper long-form support.

Short recordings retain the verified DEV/old TEST protocol. Recordings longer
than 30 seconds use pinned Transformers' timestamp-based sequential generation.
This runner requires a frozen DEV-only winner that has finished finalization.
It never changes or uses TEST outcomes for model selection.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, WhisperForConditionalGeneration
import transformers.models.whisper.generation_whisper as generation_source
import transformers.models.whisper.feature_extraction_whisper as frontend_source

from benchmark_strong_tr_test import (
    REPOSITORY, REVISION, FLEURS_REVISION, PAIR_KEYS, EXPECTED_PREFIX,
    EXPECTED_EOS, PROTOCOL as SHORT_PROTOCOL, OUTPUT, WORKSPACE,
    read, require, inside, rows, write_json, sha256,
    verified_frozen_selection, verified_assets,
)
from metrics import normalize_text, score_transcriptions, paired_bootstrap_delta

LONG_PROTOCOL = {**SHORT_PROTOCOL, "return_timestamps": True, "return_segments": True}
LONG_PREFIX = EXPECTED_PREFIX[:3]
TIMESTAMP_BEGIN = 50365
SHORT_SAMPLES = 30 * 16000
RESULT_TAG = "strong_whisper_large_v3_greedy_test_v2"
PROTOCOL = {
    "version": 2,
    "language": "tr",
    "task": "transcribe",
    "routing": "actual mono 16k PCM samples <=480000: short_form; >480000: long_form",
    "short_form": SHORT_PROTOCOL,
    "long_form": LONG_PROTOCOL,
    "long_form_frontend": {"truncation": False, "padding": "longest", "return_attention_mask": True},
    "long_form_algorithm": "pinned Transformers Whisper sequential timestamp-based generation",
    "max_new_tokens_scope": "440 per underlying decode window, not a whole-recording cap",
    "condition_on_prev_tokens": False,
    "protocol_difference_from_dev": "DEV used short_form for every recording; TEST uses timestamps only for recordings longer than 30s",
    "test_use": "reporting only, after DEV-only selection is frozen",
}
OLD_TEST_SHA256 = "f25e3e4cf61134711389157d0f184a9bac8e1df4c86b77392ace8692af4776eb"
OFFICIAL_LONGFORM_DOCUMENTATION = "https://huggingface.co/docs/transformers/model_doc/whisper"


def fingerprint_protocol(protocol):
    return hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def protected_code():
    old = OUTPUT / "benchmark_strong_tr_test.py"
    require(sha256(old) == OLD_TEST_SHA256, "Original TEST worker changed")
    return {"original_test_script_sha256": sha256(old),
            "whisper_generation_source_sha256": sha256(Path(generation_source.__file__)),
            "whisper_frontend_source_sha256": sha256(Path(frontend_source.__file__)),
            "official_longform_documentation": OFFICIAL_LONGFORM_DOCUMENTATION}


def mode_for_samples(samples):
    require(isinstance(samples, int) and samples > 0, "Invalid PCM sample count")
    return "short_form" if samples <= SHORT_SAMPLES else "long_form"


def verified_test_rows(manifest):
    manifest = inside(manifest)
    require(manifest == (WORKSPACE / "work/data/test_tr.jsonl").resolve(), "Use the unchanged fixed Turkish TEST manifest")
    examples = rows(manifest)
    base_path = OUTPUT / "results/base_v3/test_tr.jsonl"
    original = rows(base_path)
    require(len(examples) == len(original) == 100, "Expected exactly100 fixed Turkish TEST recordings")
    require(len({r["source_id"] for r in examples}) == len({r["source_id"] for r in original}) == 100,
            "Duplicate TEST source IDs")
    require([r["source_id"] for r in examples] == [r["source_id"] for r in original], "TEST ordering differs")
    modes = {"short_form": [], "long_form": []}
    for record, base in zip(examples, original):
        require(record["language"] == "tr" and record["dataset"] == "google/fleurs"
                and record["dataset_config"] == "tr_tr" and record["dataset_split"] == "test"
                and record["dataset_revision"] == FLEURS_REVISION, "Unexpected TEST dataset/split/revision")
        require(all(record[key] == base[key] for key in PAIR_KEYS), "TEST reference/source/audio pairing differs")
        require(record["reference_normalized"] == normalize_text(record["reference"], "tr"), "TEST normalization differs")
        wav = inside(record["audio_filepath"])
        info = sf.info(wav)
        require(sha256(wav) == record["wav_sha256"] and info.samplerate == 16000
                and info.channels == 1 and info.subtype == "PCM_16", "TEST WAV content/format differs")
        duration = info.frames / info.samplerate
        require(math.isfinite(duration) and duration > 0 and abs(duration - record["duration"]) <= 1 / 16000,
                "Actual TEST duration differs")
        modes[mode_for_samples(info.frames)].append({"source_id": record["source_id"],
            "pcm_samples": info.frames, "duration": duration, "wav_sha256": record["wav_sha256"]})
    return examples, {"manifest_path": str(manifest), "manifest_sha256": sha256(manifest),
        "paired_baseline_test_path": str(base_path), "paired_baseline_test_sha256": sha256(base_path),
        "checks": "100 ordered unique pinned FLEURS TEST sources; identical references/source/WAV hashes and actual monoPCM16/16k durations; no clipping, dropping or truncation",
        "duration_routes": modes, "short_form_count": len(modes["short_form"]), "long_form_count": len(modes["long_form"])}


def prepare_features(processor, signal, sample_rate):
    require(sample_rate == 16000 and signal.ndim == 1 and np.isfinite(signal).all(), "Invalid full TEST waveform")
    mode = mode_for_samples(len(signal))
    kwargs = {} if mode == "short_form" else {"truncation": False, "padding": "longest"}
    features = processor(signal, sampling_rate=sample_rate, return_tensors="pt", return_attention_mask=True, **kwargs)
    n_frames = int(features["input_features"].shape[-1])
    mask = features["attention_mask"]
    require(features["input_features"].shape[:2] == (1, 128) and torch.isfinite(features["input_features"]).all().item(),
            "Unexpected/nonfinite Whisper mel input")
    require(mask.ndim == 2 and mask.shape[0] == 1 and bool(((mask == 0) | (mask == 1)).all()), "Invalid frontend attention mask")
    if mode == "short_form":
        require(n_frames == 3000, "Short-form frontend differs from original padded 30s protocol")
    else:
        require(n_frames == len(signal) // 160 and n_frames > 3000,
                "Long waveform was truncated or not routed to native long-form generation")
        require(int(mask.sum()) == math.ceil(len(signal) / 160), "Long waveform attention mask does not preserve its tail")
    metadata = {"inference_mode": mode, "input_samples": len(signal), "input_mel_frames": n_frames,
                "input_attention_mask_frames": int(mask.sum()),
                "generation_protocol_sha256": fingerprint_protocol(SHORT_PROTOCOL if mode == "short_form" else LONG_PROTOCOL)}
    return features, metadata


def integer_tokens(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    require(isinstance(value, list) and all(isinstance(x, int) and not isinstance(x, bool) for x in value), "Invalid generated token IDs")
    return value


def require_full_window_tokens(tokens, prefix):
    require(tokens[:len(prefix)] == prefix and tokens[-1:] == [EXPECTED_EOS]
            and tokens.count(EXPECTED_EOS) == 1, "Whisper decode window lacks actual Turkish/transcribe prefix or terminal EOS")


class LongformTrace:
    """Observe native seek progression without changing inputs, outputs or decoder settings."""
    def __init__(self, model):
        self.model = model
        self.windows = []
        self.saved = {}

    def __enter__(self):
        model = self.model
        for name in ("_get_input_segment", "_retrieve_segment"):
            self.saved[name] = (name in model.__dict__, model.__dict__.get(name), getattr(model, name))
        original_input = self.saved["_get_input_segment"][2]
        original_segment = self.saved["_retrieve_segment"][2]

        def input_segment(**kwargs):
            result = original_input(**kwargs)
            require(kwargs["cur_bsz"] == 1 and kwargs["batch_idx_map"] == [0], "Long-form worker expects one recording at a time")
            self.windows.append({"seek_start_frame": int(kwargs["seek"][0]),
                "input_frame_count": int(kwargs["seek_num_frames"][0]),
                "total_input_frames": int(kwargs["input_features"].shape[-1]),
                "padded_window_frames": int(result.shape[-1])})
            return result

        def retrieve_segment(**kwargs):
            result = original_segment(**kwargs)
            require(self.windows and "seek_advance_frames" not in self.windows[-1], "Unexpected long-form seek ordering")
            window = self.windows[-1]
            window["seek_advance_frames"] = int(result[1])
            raw = integer_tokens(kwargs["seek_outputs"][kwargs["idx"]]["sequences"])
            require_full_window_tokens(raw, LONG_PREFIX)
            window["raw_generated_token_ids"] = raw
            window["decoder_input_token_ids"] = integer_tokens(kwargs["decoder_input_ids"][kwargs["idx"]])
            require(window["decoder_input_token_ids"] == LONG_PREFIX, "Long-form previous-token conditioning changed")
            window["terminal_eos_emitted"] = True
            return result

        model._get_input_segment = input_segment
        model._retrieve_segment = retrieve_segment
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for name, (existed, value, _) in self.saved.items():
            if existed:
                setattr(self.model, name, value)
            else:
                delattr(self.model, name)


def validate_longform(longform, tokens, input_frames):
    require(isinstance(longform, dict), "Missing long-form evidence")
    windows = longform.get("windows")
    segments = longform.get("segments")
    require(isinstance(windows, list) and len(windows) >= 2 and isinstance(segments, list) and segments,
            "Long-form output needs multiple full-audio seek windows and returned segments")
    expected_seek = 0
    for window in windows:
        start = window["seek_start_frame"]
        require(isinstance(start, int) and start == expected_seek and start < input_frames, "Long-form seek has a gap or incorrect ordering")
        require(window["total_input_frames"] == input_frames and window["input_frame_count"] == min(3000, input_frames - start)
                and window["padded_window_frames"] == 3000, "Long-form window does not cover the original mel input")
        advance = window["seek_advance_frames"]
        require(isinstance(advance, int) and advance > 0 and window["terminal_eos_emitted"] is True,
                "Long-form window did not advance or complete")
        require_full_window_tokens(integer_tokens(window["raw_generated_token_ids"]), LONG_PREFIX)
        require(window["decoder_input_token_ids"] == LONG_PREFIX, "Long-form language/task/conditioning changed")
        expected_seek = start + advance
    require(expected_seek >= input_frames and longform.get("final_seek_frame") == expected_seek
            and longform.get("full_input_consumed") is True, "Long-form generation did not reach the end of the recording")
    concatenated = []
    previous_start = -1.0
    for segment in segments:
        segment_tokens = integer_tokens(segment["tokens"])
        raw = integer_tokens(segment["raw_generated_token_ids"])
        require_full_window_tokens(raw, LONG_PREFIX)
        require(any(raw == window["raw_generated_token_ids"] for window in windows), "Segment raw generation differs from native windows")
        begin, end = segment["idxs"]
        require(isinstance(begin, int) and isinstance(end, int) and len(LONG_PREFIX) <= begin < end <= len(raw)
                and raw[begin:end] == segment_tokens, "Returned segment tokens differ from their actual generation slice")
        start_time, end_time = segment["start"], segment["end"]
        require(math.isfinite(start_time) and math.isfinite(end_time) and start_time >= previous_start
                and 0 <= start_time <= end_time, "Invalid long-form timestamp segment ordering")
        previous_start = start_time
        concatenated.extend(segment_tokens)
    require(concatenated == tokens and EXPECTED_EOS not in tokens and all(t not in tokens for t in LONG_PREFIX),
            "Aggregated long-form tokens differ from returned segments or contain a fabricated prefix/EOS")


def unpack_longform(generated, trace, input_frames):
    require(isinstance(generated, dict) and "sequences" in generated and "segments" in generated
            and len(generated["segments"]) == 1, "Pinned long-form generation result shape changed")
    tokens = integer_tokens(generated["sequences"][0])
    segments = []
    for segment in generated["segments"][0]:
        segments.append({"start": float(segment["start"]), "end": float(segment["end"]),
            "tokens": integer_tokens(segment["tokens"]), "idxs": [int(v) for v in segment["idxs"]],
            "raw_generated_token_ids": integer_tokens(segment["result"]["sequences"])})
    final_seek = trace.windows[-1]["seek_start_frame"] + trace.windows[-1]["seek_advance_frames"] if trace.windows else 0
    evidence = {"windows": trace.windows, "segments": segments, "final_seek_frame": final_seek,
                "full_input_consumed": final_seek >= input_frames}
    validate_longform(evidence, tokens, input_frames)
    return tokens, evidence


def validate_predictions(predictions, examples, *, whisper=False, processor=None):
    require(len(predictions) == len(examples) == 100 and len({r["source_id"] for r in predictions}) == 100,
            "Incomplete/duplicate100 TEST predictions")
    for prediction, record in zip(predictions, examples):
        require(all(prediction[key] == record[key] for key in PAIR_KEYS) and prediction.get("split") == "test"
                and isinstance(prediction.get("hypothesis"), str), "TEST prediction pairing/order/reference changed")
        if not whisper:
            continue
        samples = sf.info(record["audio_filepath"]).frames
        mode = mode_for_samples(samples)
        tokens = integer_tokens(prediction.get("generated_token_ids"))
        require(prediction.get("inference_mode") == mode and prediction.get("input_samples") == samples,
                "TEST mode/full PCM length changed")
        require(prediction.get("generation_protocol_sha256") == fingerprint_protocol(SHORT_PROTOCOL if mode == "short_form" else LONG_PROTOCOL),
                "TEST generation protocol changed")
        require(prediction.get("eos_emitted") is True and math.isfinite(prediction["inference_seconds"])
                and prediction["inference_seconds"] >= 0, "Incomplete generation/nonfinite timing")
        if mode == "short_form":
            require_full_window_tokens(tokens, EXPECTED_PREFIX)
            require(prediction["input_mel_frames"] == 3000
                    and prediction["input_attention_mask_frames"] == math.ceil(samples / 160)
                    and "longform" not in prediction,
                    "Short-form protocol changed")
        else:
            require(prediction["input_mel_frames"] == samples // 160 > 3000
                    and prediction["input_attention_mask_frames"] == math.ceil(samples / 160), "Long-form waveform tail was truncated")
            validate_longform(prediction.get("longform"), tokens, prediction["input_mel_frames"])
        if processor is not None:
            decoded = processor.batch_decode([tokens], skip_special_tokens=True)[0].strip()
            require(decoded == prediction["hypothesis"], "Saved hypothesis differs from actual generated-token decoding")


def cpu_compare(args, frozen, freeze_audit, examples, input_audit, assets):
    output = inside(args.output_dir, OUTPUT / "results")
    summary_path = output / "summary.json"
    summary = read(summary_path)
    strong_path = output / "test_tr.jsonl"
    code = protected_code()
    require(summary.get("status") == "complete" and summary.get("repository") == REPOSITORY
            and summary.get("revision") == REVISION and summary.get("split") == "test"
            and summary.get("language") == "tr" and summary.get("protocol") == PROTOCOL,
            "Strong TEST v2 summary/protocol is incomplete or changed")
    require(summary.get("script_sha256") == sha256(Path(__file__)) and summary.get("metrics_code_sha256") == sha256(OUTPUT / "metrics.py")
            and summary.get("input_audit") == input_audit and summary.get("protected_code") == code
            and all(summary.get(key) == value for key, value in assets.items())
            and summary.get("predictions_sha256") == sha256(strong_path), "Strong TEST v2 provenance changed")
    require(summary.get("frozen_selection_before_inference") == freeze_audit
            and datetime.fromisoformat(freeze_audit["frozen_at_utc"]) <= datetime.fromisoformat(summary["inference_started_at_utc"]),
            "Strong TEST inference preceded frozen DEV selection")
    strong = rows(strong_path)
    selected_dir = Path(freeze_audit["selected_results_path"])
    selected_path = selected_dir / "test_tr.jsonl"
    adapted = rows(selected_path)
    processor = AutoProcessor.from_pretrained(str(args.model_path), local_files_only=True)
    validate_predictions(strong, examples, whisper=True, processor=processor)
    validate_predictions(adapted, examples)
    before = score_transcriptions(strong, language="tr"); before.pop("details")
    after = score_transcriptions(adapted, language="tr"); after.pop("details")
    selected_summary = read(selected_dir / "summary.json")
    require(before == summary["metrics"] and after == selected_summary["test_tr"]["asr"], "Recomputed TEST ASR counters differ")
    bootstrap = paired_bootstrap_delta([r["reference"] for r in examples], [r["hypothesis"] for r in strong],
        [r["hypothesis"] for r in adapted], language="tr", seed=20261010, resamples=2000, confidence=.95,
        groups=[r["sentence_id"] for r in examples])
    require(bootstrap["wer"]["baseline"] == before["wer"] and bootstrap["wer"]["adapted"] == after["wer"]
            and bootstrap["cer"]["baseline"] == before["cer"] and bootstrap["cer"]["adapted"] == after["cer"], "Bootstrap point estimates differ")
    report = {"schema": 2, "status": "complete", "passed": True, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cpu_comparison": True, "run_id": frozen["run_id"], "selected_step": frozen["selected_step"],
        "selection_split": "FLEURS validation only", "heldout_split": "test", "language": "tr", "utterances": 100,
        "frozen_selection_sha256": freeze_audit["sha256"], "frozen_selection": freeze_audit,
        "baseline_repository": REPOSITORY, "baseline_revision": REVISION, "baseline_protocol": PROTOCOL,
        "baseline_asr": before, "adapted_asr": after,
        "wer_delta_percentage_points": bootstrap["wer"]["delta_percentage_points"],
        "cer_delta_percentage_points": bootstrap["cer"]["delta_percentage_points"], "paired_sentence_bootstrap": bootstrap,
        "delta_direction": "selected Parakeet minus Whisper; positive means worse",
        "provenance": {"strong_summary_path": str(summary_path), "strong_summary_sha256": sha256(summary_path),
            "strong_predictions_path": str(strong_path), "strong_predictions_sha256": sha256(strong_path),
            "selected_summary_path": str(selected_dir / "summary.json"), "selected_summary_sha256": sha256(selected_dir / "summary.json"),
            "selected_predictions_path": str(selected_path), "selected_predictions_sha256": sha256(selected_path),
            "input_audit": input_audit, **assets, "protected_code": code, "test_script_sha256": sha256(Path(__file__)),
            "metrics_code_sha256": sha256(OUTPUT / "metrics.py")},
        "limitations": ["100 fixed Turkish TEST recordings are a small pilot; paired sentence-cluster uncertainty is reported.",
            "DEV used no timestamps. TEST retains that exact protocol for short clips and uses native timestamp long-form decoding for the full >30s clip(s).",
            "This TEST comparison is reporting only and must not drive DEV selection or subsequent hyperparameter changes."]}
    target = inside(args.comparison_output, OUTPUT)
    write_json(target, report)
    print(json.dumps({"comparison_path": str(target), "passed": True, "run_id": frozen["run_id"],
        "baseline_wer": before["wer"], "adapted_wer": after["wer"], "wer_delta_percentage_points": report["wer_delta_percentage_points"]}), flush=True)
    return report


def main(args):
    started = time.perf_counter()
    code = protected_code()
    frozen, freeze_audit = verified_frozen_selection(args.frozen_selection, args.selected_run_id)
    examples, input_audit = verified_test_rows(args.manifest)
    assets = verified_assets(args.model_path)
    output = inside(args.output_dir, OUTPUT / "results")
    require(output.name == RESULT_TAG, "Use the versioned TEST v2 output tag")
    if args.validate_only:
        print(json.dumps({"passed": True, "gpu_used": False, "frozen_selection": freeze_audit,
            "input_audit": input_audit, "protocol": PROTOCOL, "protected_code": code, **assets}), flush=True)
        return
    if args.compare_only or (output / "summary.json").exists():
        cpu_compare(args, frozen, freeze_audit, examples, input_audit, assets)
        return
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20261010); np.random.seed(20261010)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    require(device.type == "cuda" and torch.cuda.is_available(), "Fixed TEST protocol requires local CUDA")
    dtype = torch.float16
    torch.cuda.reset_peak_memory_stats(device)
    processor = AutoProcessor.from_pretrained(str(args.model_path), local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(str(args.model_path), local_files_only=True,
        use_safetensors=True, dtype=dtype, attn_implementation="sdpa").to(device).eval()
    model.config._name_or_path = f"{REPOSITORY}@{REVISION}"
    require(model.generation_config.eos_token_id == EXPECTED_EOS and model.generation_config.no_timestamps_token_id + 1 == TIMESTAMP_BEGIN,
            "Pinned Whisper EOS/timestamp vocabulary differs")
    inference_started = datetime.now(timezone.utc).isoformat()
    write_json(output / "status.json", {"stage": "running", "total": 100, "protocol": PROTOCOL,
        "frozen_selection": freeze_audit, "started_at_utc": inference_started})
    predictions = []; prediction_path = output / "test_tr.jsonl"
    with prediction_path.open("w", encoding="utf-8", newline="\n") as stream:
        for index, record in enumerate(examples, 1):
            signal, sample_rate = sf.read(record["audio_filepath"], dtype="float32")
            features, metadata = prepare_features(processor, signal, sample_rate)
            inputs = {key: value.to(device=device, dtype=dtype if key == "input_features" else value.dtype) for key, value in features.items()}
            torch.cuda.synchronize(); clip_start = time.perf_counter()
            with torch.inference_mode():
                if metadata["inference_mode"] == "short_form":
                    generated = model.generate(**inputs, **SHORT_PROTOCOL).sequences
                    tokens = integer_tokens(generated[0]); require_full_window_tokens(tokens, EXPECTED_PREFIX)
                    extra = {}
                else:
                    with LongformTrace(model) as trace:
                        generated = model.generate(**inputs, **LONG_PROTOCOL)
                    tokens, evidence = unpack_longform(generated, trace, metadata["input_mel_frames"])
                    extra = {"longform": evidence}
            torch.cuda.synchronize(); elapsed = time.perf_counter() - clip_start
            prediction = {key: record[key] for key in PAIR_KEYS}
            prediction.update({"split": "test", "hypothesis": processor.batch_decode([tokens], skip_special_tokens=True)[0].strip(),
                "generated_token_ids": tokens, "eos_emitted": True, "inference_seconds": elapsed, **metadata, **extra})
            predictions.append(prediction); stream.write(json.dumps(prediction, ensure_ascii=False) + "\n"); stream.flush()
            if index % 5 == 0:
                print(json.dumps({"completed": index, "total": 100, "elapsed_seconds": time.perf_counter() - started}), flush=True)
                write_json(output / "status.json", {"stage": "running", "completed": index, "total": 100})
    validate_predictions(predictions, examples, whisper=True, processor=processor)
    metrics = score_transcriptions(predictions, language="tr"); metrics.pop("details")
    elapsed = math.fsum(r["inference_seconds"] for r in predictions); audio = math.fsum(r["duration"] for r in predictions)
    summary = {"schema": 2, "status": "complete", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": REPOSITORY, "revision": REVISION, "split": "test", "language": "tr", "protocol": PROTOCOL,
        "metrics": metrics, "input_audit": input_audit, **assets, "protected_code": code,
        "frozen_selection_before_inference": freeze_audit, "inference_started_at_utc": inference_started,
        "script_sha256": sha256(Path(__file__)), "metrics_code_sha256": sha256(OUTPUT / "metrics.py"),
        "predictions_sha256": sha256(prediction_path), "device": str(device), "hardware": torch.cuda.get_device_name(device),
        "weight_compute_dtype": str(dtype), "attention_implementation": "sdpa", "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__, "all_full_prefixes_verified": True,
        "all_terminal_eos_verified": True, "all_longform_inputs_fully_consumed": True,
        "short_form_count": input_audit["short_form_count"], "long_form_count": input_audit["long_form_count"],
        "peak_allocated_vram_gb": torch.cuda.max_memory_allocated(device) / 1e9,
        "audio_seconds": audio, "inference_seconds": elapsed, "real_time_factor": elapsed / audio,
        "elapsed_seconds_with_load_and_audit": time.perf_counter() - started}
    write_json(output / "summary.json", summary)
    write_json(output / "status.json", {"stage": "complete", "completed": 100, "summary_sha256": sha256(output / "summary.json")})
    del model, processor, inputs, features, generated
    gc.collect(); torch.cuda.empty_cache()
    print(json.dumps({"gpu_released": True, "baseline_wer": metrics["wer"]}), flush=True)
    frozen_again, freeze_again = verified_frozen_selection(args.frozen_selection, args.selected_run_id)
    require(freeze_again == freeze_audit and frozen_again == frozen, "Frozen winner changed during inference")
    require(protected_code() == code, "Protected inference code changed during TEST")
    cpu_compare(args, frozen, freeze_audit, examples, input_audit, assets)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frozen-selection", type=Path, required=True)
    p.add_argument("--selected-run-id")
    p.add_argument("--model-path", type=Path, default=WORKSPACE / "work/models/whisper-large-v3")
    p.add_argument("--manifest", type=Path, default=WORKSPACE / "work/data/test_tr.jsonl")
    p.add_argument("--output-dir", type=Path, default=OUTPUT / "results" / RESULT_TAG)
    p.add_argument("--comparison-output", type=Path, default=OUTPUT / "adaptive_strong_baseline_test_comparison_v2.json")
    p.add_argument("--device", default="cuda")
    modes = p.add_mutually_exclusive_group()
    modes.add_argument("--compare-only", action="store_true")
    modes.add_argument("--validate-only", action="store_true")
    return p


if __name__ == "__main__":
    main(parser().parse_args())
