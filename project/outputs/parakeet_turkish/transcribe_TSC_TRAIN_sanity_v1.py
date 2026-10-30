"""Bounded local-only Whisper transcription of ROOT-audited TSC TRAIN examples.

No teacher labels or training manifests are written. Preparation reads small
metadata only; ROOT must explicitly authorize the exclusive GPU execution.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import time

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent.parent
REPOSITORY = "openai/whisper-large-v3"
REVISION = "06f233fe06e710322aca913c1bc4249a0d71fce1"
IMPORTER_SHA = "08785c2446229ae6716feee5e9103d77c468faf0eaf68a602e0477bc2d36063e"
DEV_SCRIPT_SHA = "46bf78a57112d1adc7c11ec019be5c99356ea6d30f68b61bb49abc3cc6bca475"
PREFIX = [50258, 50268, 50360, 50364]
EOS = 50257
PROTOCOL = {"language": "tr", "task": "transcribe", "return_timestamps": False,
    "return_token_timestamps": False, "return_dict_in_generate": True,
    "num_beams": 1, "do_sample": False, "temperature": 0.0,
    "condition_on_prev_tokens": False, "max_new_tokens": 440,
    "compression_ratio_threshold": None, "logprob_threshold": None,
    "no_speech_threshold": None}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inside(path):
    path = Path(path).resolve()
    require(path.is_relative_to(WORKSPACE), "Workspace-scoped path required")
    return path


def read(path):
    return json.loads(inside(path).read_text(encoding="utf-8-sig"))


def write_new(path, value):
    path = inside(path)
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)


def full_tokens(tokens):
    require(isinstance(tokens, list) and tokens and all(type(x) is int for x in tokens), "Integer decoder sequence required")
    require(tokens[:4] == PREFIX and tokens[-1:] == [EOS] and tokens.count(EOS) == 1,
            "Actual Turkish/transcribe/no-timestamps prefix and terminal EOS required")
    require(len(tokens) <= len(PREFIX) + PROTOCOL["max_new_tokens"], "Decoder exceeded fixed generation bound")
    require(not any(x >= 50365 for x in tokens[4:-1]), "Short-form decoder unexpectedly emitted timestamp tokens")


def prepare(args):
    require(1 <= args.max_examples <= 20, "At most20 audited TRAIN examples permitted")
    require(math.isfinite(args.max_wall_seconds) and 1 <= args.max_wall_seconds <= 300,
            "Bounded1–300s wall budget required")
    require(math.isfinite(args.cap_gb) and 4 <= args.cap_gb <= 8, "Explicit4–8GB allocator cap required")
    output = inside(args.output)
    require(not output.exists(), "New immutable sanity output path required")
    proof_path, pcm_path = inside(args.import_proof), inside(args.pcm_audit)
    proof, pcm = read(proof_path), read(pcm_path)
    require(proof.get("kind") == "TSC_TRAIN_candidate_import_v1" and proof.get("passed") is True
            and proof.get("full_archive_hash_verified") is True and proof.get("importer_sha256") == IMPORTER_SHA,
            "Approved verified TSC TRAIN candidate import proof required")
    require(pcm.get("kind") == "TSC_TRAIN_pilot_independent_PCM_metadata_audit_v1" and pcm.get("passed") is True
            and pcm.get("all_PCM_WAV_duration_identities_exact") is True
            and inside(pcm["import_proof_path"]) == proof_path and pcm["import_proof_sha256"] == sha(proof_path),
            "ROOT PCM audit does not bind this exact import proof")
    manifest = inside(proof["accepted_manifest_path"])
    require(inside(pcm["accepted_manifest_path"]) == manifest and sha(manifest) ==
            proof["accepted_manifest_sha256"] == pcm["accepted_manifest_sha256"], "TRAIN manifest hash differs")
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    by_id = {r["source_id"]: r for r in rows}
    require(len(by_id) == len(rows) == proof["accepted_clips"] == pcm["clips_audited"], "Unique accepted TRAIN pool identity differs")
    examples = pcm["TRAIN_listening_examples"][:args.max_examples]
    require(examples and len({e["source_id"] for e in examples}) == len(examples), "Unique ROOT-selected TRAIN examples required")
    selected = []
    for example in examples:
        row = by_id[example["source_id"]]
        source = PurePosixPath(row["original_audio_path"])
        transcript = PurePosixPath(row["original_transcript_path"])
        require(source.parts[:2] == ("ISSAI_TSC_218", "Train") and len(source.parts) == 3
                and source.suffix == ".wav" and transcript == source.with_suffix(".txt"), "Only original TSC Train whole WAV/TXT pairs allowed")
        require(row["dataset"] == "issai/Turkish_Speech_Corpus" and row["dataset_split"] == "train"
                and row["training_role"] == "train" and row["language"] == "tr"
                and row["data_expansion_role"] == "tsc_train" and row["archive_sha256"] == proof["archive_sha256"]
                and row["dataset_revision"] == proof["archive_revision"], "TRAIN source/revision role differs")
        signal = example["signal"]
        require(inside(row["audio_filepath"]) == inside(example["audio_filepath"])
                and signal["source_id"] == row["source_id"]
                and row["wav_sha256"] == signal["wav_sha256"]
                and row["canonical_pcm_sha256"] == signal["canonical_pcm_sha256"]
                and row["duration"] == signal["duration_seconds"]
                and .5 <= row["duration"] <= 30 and row["sample_rate"] == 16000,
                "Example source/WAV/whole PCM identity differs")
        require(row["reference"] == row["reference_raw_text"] == row["text"], "Raw TRAIN reference must remain authoritative")
        require(example["reference"] == row["reference"], "UTF-8 ROOT listening reference differs from manifest")
        selected.append({"row": row, "ROOT_signal_evidence": signal,
            "ROOT_listening_example_reference": example["reference"],
            "ROOT_example_reference_matches_manifest": example["reference"] == row["reference"]})
    dev_receipt_path = HERE / "strong_baseline_dev_verification.json"
    dev_path = HERE / "results/strong_whisper_large_v3_greedy_dev_verified/summary.json"
    verification, dev = read(dev_receipt_path), read(dev_path)
    lock_path = WORKSPACE / "work/strong_baseline_model.lock.json"
    lock = read(lock_path)
    require(verification.get("passed") is True and verification["summary_sha256"] == sha(dev_path)
            and verification["expected_prefix"] == PREFIX and verification["expected_eos_id"] == EOS,
            "Existing verified DEV protocol receipt differs")
    require(dev["status"] == "complete" and dev["repository"] == lock["repository"] == REPOSITORY
            and dev["revision"] == lock["revision"] == REVISION and dev["protocol"] == PROTOCOL
            and dev["model_lock_sha256"] == sha(lock_path)
            and dev["script_sha256"] == DEV_SCRIPT_SHA == sha(HERE / "benchmark_strong_tr.py")
            and dev["metrics_code_sha256"] == sha(HERE / "metrics.py"), "Fixed greedy/model/scorer provenance differs")
    model_path = inside(lock["local_path"])
    return {"selected": selected, "model_path": str(model_path), "model_files_sha256": dev["model_files_sha256"],
        "import_proof_path": str(proof_path), "import_proof_sha256": sha(proof_path),
        "PCM_audit_path": str(pcm_path), "PCM_audit_sha256": sha(pcm_path),
        "accepted_manifest_path": str(manifest), "accepted_manifest_sha256": sha(manifest),
        "verified_DEV_summary_path": str(dev_path), "verified_DEV_summary_sha256": sha(dev_path),
        "verified_DEV_receipt_sha256": sha(dev_receipt_path), "model_lock_sha256": sha(lock_path),
        "metrics_sha256": sha(HERE / "metrics.py"), "script_sha256": sha(__file__),
        "resource_guard_sha256": sha(HERE / "resource_watchdog_v2.py"),
        "paired_memory_API_sha256": sha(HERE / "paired_windows_commit_telemetry_v1.py")}


def execute(args, plan):
    require(args.gpu_window_authorized, "ROOT-exclusive GPU window authorization flag required")
    # Local-only assets: no hub lookup, auth, download, dataset, or TEST access.
    for key, value in {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "RAYON_NUM_THREADS": "1"}.items():
        os.environ[key] = value
    began = time.monotonic()
    for name, expected in plan["model_files_sha256"].items():
        require(sha(Path(plan["model_path"]) / name) == expected, "Pinned local Whisper asset changed: " + name)
    import numpy as np
    import soundfile as sf
    import torch
    import transformers
    from transformers import AutoProcessor, WhisperForConditionalGeneration
    from metrics import normalize_text, score_transcriptions
    from resource_watchdog_v2 import ResourceGuard, sample_memory, worker_process_commit_reason
    torch.set_num_threads(1)
    torch.manual_seed(20261010)
    require(torch.cuda.is_available(), "Actual local CUDA required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    total = torch.cuda.get_device_properties(0).total_memory
    require(args.cap_gb * 1e9 < total, "Allocator cap must be below physical memory")
    torch.cuda.set_per_process_memory_fraction(args.cap_gb * 1e9 / total, device=0)
    guard = ResourceGuard()
    telemetry = []
    def safety(phase):
        require(time.monotonic() - began < args.max_wall_seconds, "Bounded sanity wall budget elapsed")
        sample = sample_memory(os.getpid(), phase)
        reason, observation = guard.observe(sample["system_memory"], sample["worker_private_memory"],
            observed_monotonic=sample["observed_monotonic"])
        process_reason = worker_process_commit_reason(sample["calling_process_commit_memory"], os.getpid())
        telemetry.append({**sample, "guard_observation": observation, "guard_reason": reason or process_reason})
        require(not reason and not process_reason, "Resource guard: " + str(reason or process_reason))
    predictions = []
    report = {"schema": 1, "kind": "TSC_TRAIN_pinned_Whisper_transcription_sanity_v1", "status": "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "PID": os.getpid(),
        "repository": REPOSITORY, "revision": REVISION, "protocol": PROTOCOL,
        "weight_compute_dtype": "float16", "attention_implementation": "sdpa",
        "serial_batch_size": 1, "allocator_cap_bytes": int(args.cap_gb * 1e9),
        "physical_GPU_memory_bytes": total, "requested_examples": len(plan["selected"]),
        "provenance": {k:v for k,v in plan.items() if k != "selected"},
        "no_heldout_audio_or_predictions_read": True, "training_references_unchanged": True,
        "new_teacher_labels_or_training_adoption": False,
        "limitations": ["Small deliberately selected TRAIN sample, not a benchmark or corpus-wide quality estimate.",
            "Whisper disagreement is evidence to inspect, not authority to relabel.",
            "Manifest references and ROOT listening-list references are required to match as UTF-8 text."]}
    try:
        safety("before_local_model_load")
        processor = AutoProcessor.from_pretrained(plan["model_path"], local_files_only=True)
        model = WhisperForConditionalGeneration.from_pretrained(plan["model_path"], local_files_only=True,
            use_safetensors=True, dtype=torch.float16, attn_implementation="sdpa").to("cuda").eval()
        model.config._name_or_path = REPOSITORY + "@" + REVISION
        torch.cuda.reset_peak_memory_stats()
        load_seconds = time.monotonic() - began
        for item in plan["selected"]:
            row = item["row"]
            safety("before_whole_TRAIN_recording")
            wav = inside(row["audio_filepath"])
            require(sha(wav) == row["wav_sha256"], "Audited TRAIN WAV changed")
            info = sf.info(str(wav))
            require(info.samplerate == 16000 and info.channels == 1 and info.subtype == "PCM_16"
                    and info.frames == item["ROOT_signal_evidence"]["actual_PCM_frames"]
                    and info.frames / 16000 == row["duration"] and 8000 <= info.frames <= 480000,
                    "Whole canonical TRAIN WAV geometry differs")
            pcm, sr = sf.read(str(wav), dtype="int16", always_2d=False)
            require(pcm.ndim == 1 and len(pcm) == info.frames
                    and hashlib.sha256(pcm.astype("<i2", copy=False).tobytes()).hexdigest() == row["canonical_pcm_sha256"],
                    "Actual whole PCM hash differs from ROOT audit")
            signal = pcm.astype(np.float32) / 32768.0
            features = processor(signal, sampling_rate=sr, return_tensors="pt", return_attention_mask=True)
            require(tuple(features["input_features"].shape) == (1,128,3000), "Whisper fixed short-form mel geometry differs")
            inputs = {key:value.to("cuda", dtype=torch.float16 if key == "input_features" else value.dtype)
                for key,value in features.items()}
            torch.cuda.synchronize()
            started = time.monotonic()
            with torch.inference_mode():
                generated = model.generate(**inputs, **PROTOCOL).sequences
            torch.cuda.synchronize()
            elapsed = time.monotonic() - started
            tokens = generated[0].cpu().tolist()
            full_tokens(tokens)
            hypothesis = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
            prediction = {"source_id": row["source_id"], "language": "tr", "dataset_split": "train",
                "original_audio_path": row["original_audio_path"], "audio_filepath": str(wav),
                "wav_sha256": row["wav_sha256"], "canonical_pcm_sha256": row["canonical_pcm_sha256"],
                "duration": row["duration"], "reference": row["reference"], "hypothesis": hypothesis,
                "ROOT_listening_example_reference": item["ROOT_listening_example_reference"],
                "ROOT_example_reference_matches_manifest": item["ROOT_example_reference_matches_manifest"],
                "raw_generated_token_ids": tokens, "full_prefix_and_EOS_verified": True,
                "whole_recording_samples_used": len(signal), "audio_cropped": False,
                "reference_normalized_word_count": len(normalize_text(row["reference"], "tr").split()),
                "hypothesis_normalized_word_count": len(normalize_text(hypothesis, "tr").split()),
                "signal": item["ROOT_signal_evidence"], "inference_seconds": elapsed}
            prediction["metrics"] = score_transcriptions([prediction], language="tr")
            predictions.append(prediction)
            print(json.dumps({"completed":len(predictions),"total":len(plan["selected"]),
                "source_id":row["source_id"],"WER":prediction["metrics"]["wer"]}), flush=True)
            del generated, inputs, features
            torch.cuda.empty_cache()
            safety("after_TRAIN_transcription_and_unused_cache_reclamation")
        report.update(status="complete", passed=True, torch_version=torch.__version__,
            transformers_version=transformers.__version__, load_and_asset_check_seconds=load_seconds)
    except BaseException as error:
        report.update(status="failed_or_bounded_partial", passed=False, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report.update(predictions=predictions, completed_examples=len(predictions),
            system_and_actual_process_memory_observations=telemetry,
            elapsed_seconds_with_asset_hash_and_load=time.monotonic()-began,
            peak_allocated_vram_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_vram_bytes=torch.cuda.max_memory_reserved())
        if predictions:
            report["small_TRAIN_sample_metrics"] = score_transcriptions(predictions, language="tr")
        write_new(args.output, report)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--import-proof", type=Path, default=HERE/"tsc_TRAIN_candidate_pilot_v1_preparation.json")
    p.add_argument("--pcm-audit", type=Path, default=HERE/"TSC_pilot_ROOT_PCM_quality_audit_v1.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-examples", type=int, default=20)
    p.add_argument("--max-wall-seconds", type=float, default=180)
    p.add_argument("--cap-gb", type=float, default=6)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--gpu-window-authorized", action="store_true")
    return p


def main(args):
    plan = prepare(args)
    if args.prepare_only:
        write_new(args.output, {"schema":1,"kind":"TSC_TRAIN_Whisper_sanity_CPU_prepared_v1",
            "passed":True,"GPU_or_model_or_audio_reads_performed":False,"plan":plan,
            "protocol":PROTOCOL,"expected_prefix":PREFIX,"expected_eos":EOS,
            "runtime_model_asset_hash_validation_pending":True})
        print(json.dumps({"CPU_prepared":True,"examples":len(plan["selected"]),"output":str(args.output)}))
    else:
        execute(args, plan)


if __name__ == "__main__":
    main(parser().parse_args())
