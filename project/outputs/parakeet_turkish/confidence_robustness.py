"""Small paired confidence-sensitivity experiment, separate from the benchmark.

Prepare on CPU while training runs:
  python confidence_robustness.py --prepare-only
Run only when the local GPU is free:
  python confidence_robustness.py --model v2=work/models/parakeet-v2 \
      --model v3=BASE_MODEL_PATH --model pilot=SELECTED_CHECKPOINT

The first ten fixed English development recordings are selected before any
perturbation inference. This is a descriptive sensitivity check, not a new test
set or evidence of general robustness. Existing inference helpers are unchanged.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
SAMPLE_RATE = 16000
VARIANTS = ("clean", "gain_minus_0_2_db", "noise_snr_50_db", "silence_20_ms_each_side")


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))


def waveform_sha256(audio: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(audio, dtype="<f4").tobytes()).hexdigest()


def perturb(audio: np.ndarray, source_id: str, seed: int = 20261010) -> dict[str, tuple[np.ndarray, dict]]:
    """Deterministic float32 variants without clipping or peak normalization."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
        raise ValueError("Expected a nonempty finite mono waveform.")
    clean_rms = rms(audio)
    if clean_rms <= 0:
        raise ValueError("Cannot define a noise SNR for a silent recording.")
    factor = 10 ** (-0.2 / 20)
    gained = (audio * factor).astype(np.float32)
    noise_seed = int.from_bytes(hashlib.sha256(f"{seed}:{source_id}".encode()).digest()[:8], "little")
    noise = np.random.default_rng(noise_seed).standard_normal(audio.size)
    noise -= noise.mean()
    noise *= clean_rms * 10 ** (-50 / 20) / rms(noise)
    noisy = (audio.astype(np.float64) + noise).astype(np.float32)
    effective_noise_rms = rms(noisy.astype(np.float64) - audio.astype(np.float64))
    padding = round(SAMPLE_RATE * 0.020)
    padded = np.pad(audio, (padding, padding), mode="constant")
    variants = {
        "clean": (audio.copy(), {"operation": "unchanged"}),
        "gain_minus_0_2_db": (gained, {"operation": "global amplitude attenuation", "gain_db": -0.2,
            "amplitude_factor": factor, "measured_gain_db": 20 * math.log10(rms(gained) / clean_rms)}),
        "noise_snr_50_db": (noisy, {"operation": "independent Gaussian additive noise", "target_snr_db": 50,
            "noise_seed": noise_seed, "noise_rms": effective_noise_rms,
            "measured_snr_db": 20 * math.log10(clean_rms / effective_noise_rms)}),
        "silence_20_ms_each_side": (padded, {"operation": "zero padding at both ends",
            "leading_samples": padding, "trailing_samples": padding,
            "leading_seconds": 0.020, "trailing_seconds": 0.020}),
    }
    for waveform, metadata in variants.values():
        metadata.update(samples=int(waveform.size), seconds=waveform.size / SAMPLE_RATE,
            signal_rms=rms(waveform), peak_absolute=float(np.abs(waveform).max()),
            samples_over_unit_amplitude=int((np.abs(waveform) > 1).sum()),
            waveform_float32_sha256=waveform_sha256(waveform), clipping="none")
    return variants


def prepare(manifest: Path, limit: int, seed: int) -> tuple[list[dict], list[dict], dict]:
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()][:limit]
    if len(rows) != limit or limit < 1:
        raise ValueError("Not enough recordings for the requested positive limit.")
    if len({row["source_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate selected source IDs.")
    selected, measured = [], []
    for row in rows:
        if row.get("language") != "en" or row.get("dataset_split") != "validation":
            raise ValueError("Sensitivity check must use the English development split.")
        audio_path = Path(row["audio_filepath"])
        audio, rate = sf.read(audio_path, dtype="float32")
        if rate != SAMPLE_RATE or audio.ndim != 1:
            raise ValueError("Expected prepared mono 16 kHz audio.")
        file_hash = hashlib.sha256(audio_path.read_bytes()).hexdigest()
        if row.get("wav_sha256") and row["wav_sha256"] != file_hash:
            raise ValueError(f"Prepared WAV changed: {audio_path}")
        variants = perturb(audio, row["source_id"], seed)
        assert np.array_equal(variants["clean"][0], audio)
        assert np.array_equal(variants["silence_20_ms_each_side"][0][320:-320], audio)
        assert abs(variants["noise_snr_50_db"][1]["measured_snr_db"] - 50) < 0.0001
        assert abs(variants["gain_minus_0_2_db"][1]["measured_gain_db"] + 0.2) < 0.0001
        selected.append({"row": row, "audio": audio})
        measured.append({"source_id": row["source_id"], "sentence_id": row.get("sentence_id"),
            "reference": row["text"], "wav_sha256": file_hash,
            "original_signal_rms": rms(audio), "variants": {name: info for name, (_, info) in variants.items()}})
    plan = {"schema_version": 1, "status": "prepared_cpu_only", "sample_rate": SAMPLE_RATE, "seed": seed,
        "selection": f"First {limit} rows of the existing fixed English development manifest; selected before perturbation inference.",
        "manifest": str(manifest.resolve()), "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "variants": list(VARIANTS), "samples": measured,
        "scope": "Tiny amplitude attenuation, weak additive noise, and short silent padding. Audibility is not assessed by a listening test.",
        "limitations": ["Ten English development recordings are a descriptive pilot, not an independent test or proof of broad robustness.",
            "Raw confidence includes blank competition and can change with emission timing.",
            "Confidence deltas condition on words correct in both clean and perturbed audio; ASR errors and transcript changes are reported separately.",
            "Different vocabularies make cross-model token probabilities and segmentation unequal; compare word-level sensitivity descriptively.",
            "No clipping or amplitude normalization is applied; actual peak amplitudes and any samples outside [-1, 1] are reported."]}
    return selected, measured, plan


def statistics(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "mean_absolute": None, "rms": None,
                "absolute_p95": None, "absolute_max": None}
    array = np.asarray(values, dtype=np.float64)
    return {"count": len(values), "mean": float(array.mean()), "mean_absolute": float(np.abs(array).mean()),
            "rms": rms(array), "absolute_p95": float(np.quantile(np.abs(array), 0.95)),
            "absolute_max": float(np.abs(array).max())}


def correct_by_reference(record: dict) -> dict[int, dict]:
    correct = [word for word in record["alignment"]["words"] if word["correct"]]
    mapping = {word["reference_word_index"]: word for word in correct}
    if None in mapping or len(mapping) != len(correct):
        raise ValueError("Correct-word reference indices must be unique and present.")
    return mapping


def paired_word_deltas(clean: dict, changed: dict) -> list[dict]:
    original, altered = correct_by_reference(clean), correct_by_reference(changed)
    result = []
    for index in sorted(original.keys() & altered.keys()):
        before, after = original[index], altered[index]
        if before["word"] != after["word"]:
            raise ValueError("Shared correct reference words disagree.")
        result.append({"reference_word_index": index, "word": before["word"],
            "clean_confidence": before["confidence"], "perturbed_confidence": after["confidence"],
            "confidence_delta": after["confidence"] - before["confidence"],
            "clean_conditional_nonblank": before["conditional_nonblank_confidence"],
            "perturbed_conditional_nonblank": after["conditional_nonblank_confidence"],
            "conditional_nonblank_delta": after["conditional_nonblank_confidence"] - before["conditional_nonblank_confidence"],
            "clean_token_count": len(before["token_indices"]), "perturbed_token_count": len(after["token_indices"])})
    return result


def summarize(records: list[dict]) -> dict:
    from metrics import normalize_text, score_transcriptions
    clean = {row["source_id"]: row for row in records if row["variant"] == "clean"}
    summary = {"clean_asr": score_transcriptions(clean.values(), language="en"), "perturbations": {}}
    for variant in VARIANTS[1:]:
        changed = [row for row in records if row["variant"] == variant]
        if {row["source_id"] for row in changed} != set(clean):
            raise ValueError("Clean and perturbed recording sets differ.")
        pairs = []
        for row in changed:
            pairs.extend(dict(source_id=row["source_id"], **word)
                         for word in paired_word_deltas(clean[row["source_id"]], row))
        asr = score_transcriptions(changed, language="en")
        summary["perturbations"][variant] = {
            "asr": asr, "wer_delta_percentage_points": 100 * (asr["wer"] - summary["clean_asr"]["wer"]),
            "raw_transcripts_changed": sum(row["hypothesis"] != clean[row["source_id"]]["hypothesis"] for row in changed),
            "normalized_transcripts_changed": sum(normalize_text(row["hypothesis"], "en") !=
                normalize_text(clean[row["source_id"]]["hypothesis"], "en") for row in changed),
            "shared_correct_word_confidence_delta": statistics([word["confidence_delta"] for word in pairs]),
            "shared_correct_word_conditional_nonblank_delta": statistics([word["conditional_nonblank_delta"] for word in pairs]),
            "shared_correct_words_below_0_5_clean": sum(word["clean_confidence"] < 0.5 for word in pairs),
            "shared_correct_words_below_0_5_perturbed": sum(word["perturbed_confidence"] < 0.5 for word in pairs),
            "paired_words": pairs}
    return summary


def compare_sensitivity(models: dict) -> dict:
    """Same reference-word subset for each model pair and perturbation."""
    comparisons = {}
    for left, right in itertools.combinations(models, 2):
        pair = {}
        for variant in VARIANTS[1:]:
            a = {(w["source_id"], w["reference_word_index"]): w for w in models[left]["summary"]["perturbations"][variant]["paired_words"]}
            b = {(w["source_id"], w["reference_word_index"]): w for w in models[right]["summary"]["perturbations"][variant]["paired_words"]}
            keys = sorted(a.keys() & b.keys())
            pair[variant] = {"shared_correct_words_all_four_decodes": len(keys),
                "left_raw_delta": statistics([a[k]["confidence_delta"] for k in keys]),
                "right_raw_delta": statistics([b[k]["confidence_delta"] for k in keys]),
                "right_minus_left_absolute_raw_change": statistics([abs(b[k]["confidence_delta"]) - abs(a[k]["confidence_delta"]) for k in keys]),
                "right_minus_left_absolute_conditional_change": statistics([abs(b[k]["conditional_nonblank_delta"]) - abs(a[k]["conditional_nonblank_delta"]) for k in keys])}
        comparisons[f"{left}_vs_{right}"] = pair
    return comparisons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=ROOT / "work/data/validation_en.jsonl")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--prepare-only", action="store_true", help="Validate variants and write plan without importing Torch or loading models.")
    parser.add_argument("--model", action="append", default=[], metavar="NAME=LOCAL_PATH")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    selected, _, result = prepare(args.manifest, args.limit, args.seed)
    output = args.output or OUTPUT / ("confidence_robustness_plan.json" if args.prepare_only else "confidence_robustness.json")
    if args.prepare_only:
        write_json(output, result)
        print(f"CPU-only plan verified for {len(selected)} recordings: {output}", flush=True)
        return
    model_specs = {}
    for value in args.model:
        name, delimiter, path = value.partition("=")
        if not delimiter or not name or not path or name in model_specs:
            raise ValueError("Supply unique --model NAME=LOCAL_PATH arguments.")
        model_specs[name] = path
    if len(model_specs) < 2:
        raise ValueError("Supply at least two model paths; this command uses the local GPU sequentially.")
    import torch
    from benchmark import model_fingerprint
    from confidence import align_word_confidences, generate_with_confidence
    from runtime import load_model
    result.update(status="running", models={})
    for name, path in model_specs.items():
        model, processor = load_model(path)
        model.eval()
        fingerprint = model_fingerprint(model, processor)
        records = []
        for index, selected_row in enumerate(selected):
            row, audio = selected_row["row"], selected_row["audio"]
            for variant, (waveform, info) in perturb(audio, row["source_id"], args.seed).items():
                inputs = processor(audio=[waveform], sampling_rate=SAMPLE_RATE, return_tensors="pt")
                inputs = {key: value.cuda() for key, value in inputs.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    generated, confidence = generate_with_confidence(model, processor.tokenizer, inputs,
                        language="en", temperatures=(1.0,))
                hypothesis = processor.decode(generated.sequences[0], skip_special_tokens=True)
                if hypothesis.strip() != confidence["text"].strip():
                    raise ValueError("Captured confidence differs from decoded transcript.")
                records.append({"source_id": row["source_id"], "sentence_id": row.get("sentence_id"),
                    "variant": variant, "language": "en", "reference": row["text"], "hypothesis": hypothesis,
                    "audio_variant": info, "confidence": confidence,
                    "alignment": align_word_confidences(row["text"], confidence, "en")})
                del inputs, generated
            print(f"{name}: {index + 1}/{len(selected)} recordings × {len(VARIANTS)} variants", flush=True)
        result["models"][name] = {"model_path": path, "model_fingerprint": fingerprint,
            "records": records, "summary": summarize(records)}
        write_json(output, result)
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()
    result.update(status="complete", pairwise_model_sensitivity=compare_sensitivity(result["models"]))
    write_json(output, result)
    print(f"Completed paired sensitivity check: {output}", flush=True)


if __name__ == "__main__":
    main()
