"""Small waveform/reader integration contract, not ASR quality or GPU validation."""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

from fastmss_cpu_guard_v1 import Guard, output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--real-rir-registry")
    args = parser.parse_args()
    out = output_path(args.output)
    guard = Guard(wall_seconds=90)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "outputs/parakeet_turkish"))
    from fastmss_train_augmentation_v1 import AugmentationPolicy, TrainWaveformAugmenter
    import numpy as np
    import soundfile as sf
    guard.check()
    results = {}

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        results[name] = True

    def rejects(name, function):
        try:
            function()
        except (ValueError, TypeError):
            results[name] = True
        else:
            raise AssertionError(name)

    fs = 16000
    time_axis = np.arange(2 * fs, dtype=np.float32) / fs
    source = (0.22 * np.sin(2 * np.pi * 220 * time_axis) + 0.08 * np.sin(2 * np.pi * 1307 * time_axis)).astype(np.float32)
    source_bytes = source.tobytes()
    rng = np.random.default_rng(901)
    noise = rng.standard_normal(fs // 2, dtype=np.float32) * np.float32(0.03)
    rir = np.zeros(fs // 10, dtype=np.float32)
    rir[17] = 1
    rir[147] = 0.25
    rir[457] = -0.15
    assets = []
    for kind, data in (("noise", noise), ("rir", rir)):
        path = out / (kind + "_fixture.flac")
        sf.write(path, data, fs, subtype="PCM_16")
        assets.append({"id": kind + "_fixture", "kind": kind, "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "sample_rate": fs,
            "license": "local-generated-fixture-only", "provenance": "locally synthesized seed901; not real environmental recording/FastMSS room",
            "training_use_confirmed": True, "fixture_only": True})
    registry = out / "fixture_assets.json"
    registry.write_text(json.dumps({"schema": 1, "assets": assets}, indent=2), encoding="utf-8")
    policy = AugmentationPolicy(enabled=True, apply_probability=1, noise_probability=1,
        rir_probability=1, snr_db=(15, 15), registry_path=str(registry))
    rejects("production_rejects_fixture_assets", lambda: TrainWaveformAugmenter(policy, guard=guard.check))
    augmenter = TrainWaveformAugmenter(policy, guard=guard.check, fixture_mode=True)
    row = {"source_id": "fixture_TR_001", "language": "tr", "text": "Değişmeyen kaynak metin", "duration": 2.0}
    row_before = json.dumps(row, sort_keys=True)
    context = dict(split="train", phase="weak_label_continued_training", role="student_tr", epoch=0, draw_index=17)
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    transformed, receipt = augmenter.apply(source, row, **context)
    repeat, repeated_receipt = augmenter.apply(source, row, **context)
    check("same_draw_exact_waveform_and_receipt", np.array_equal(transformed, repeat) and receipt == repeated_receipt)
    check("sample_count_float32_finite_preserved", transformed.shape == source.shape and transformed.dtype == np.float32 and np.isfinite(transformed).all())
    check("noise_and_rir_actual_nonidentity", receipt["applied"] and len(receipt["operations"]) == 2 and not np.array_equal(source, transformed))
    check("source_and_transcript_unchanged", source.tobytes() == source_bytes and json.dumps(row, sort_keys=True) == row_before)
    check("Python_global_RNG_unchanged", random.getstate() == python_rng)
    after_rng = np.random.get_state()
    check("NumPy_global_RNG_unchanged", numpy_rng[0] == after_rng[0] and np.array_equal(numpy_rng[1], after_rng[1]) and numpy_rng[2:] == after_rng[2:])
    for key, value in (("split", "dev"), ("split", "test"), ("role", "teacher"),
                       ("role", "replay"), ("phase", "cleaner_data_fine_tuning")):
        clean, why = augmenter.apply(source, row, **{**context, key: value})
        check("protected_" + key + "_" + value, clean is source and not why["applied"])
    for language in ("en", "es", "it", "fr", "de", "pt"):
        clean, why = augmenter.apply(source, {**row, "language": language}, **context)
        check("protected_retention_" + language, clean is source and not why["applied"])
    disabled = TrainWaveformAugmenter(AugmentationPolicy(), guard=guard.check)
    check("disabled_identity_without_assets", disabled.apply(source, row, **context)[0] is source)
    clean_probability = TrainWaveformAugmenter(replace(policy, apply_probability=0), guard=guard.check, fixture_mode=True)
    check("probability_zero_identity", clean_probability.apply(source, row, **context)[0] is source)
    changed, changed_receipt = augmenter.apply(source, row, **{**context, "draw_index": 18})
    check("different_durable_draw_changes_noise", not np.array_equal(changed, transformed) and changed_receipt["augmentation_seed"] != receipt["augmentation_seed"])
    long_source = np.zeros(fs * 121, dtype=np.float32)
    clean, why = augmenter.apply(long_source, row, **context)
    check("oversized_augmentation_keeps_clean_source", clean is long_source and "duration_bound" in why["reason"])
    silence = np.zeros(fs, dtype=np.float32)
    check("silence_keeps_clean_source", augmenter.apply(silence, row, **context)[0] is silence)
    rejects("invalid_sample_rate_rejected", lambda: AugmentationPolicy(sample_rate=24000))
    rejects("missing_real_noise_assets_rejected", lambda: TrainWaveformAugmenter(replace(policy, registry_path=None, rir_probability=0), guard=guard.check))
    rejects("NaN_source_rejected", lambda: augmenter.apply(np.full(fs, np.nan, dtype=np.float32), row, **context))
    # Check actual RMS SNR independently, without RIR or a peak normalization ambiguity.
    noise_only = TrainWaveformAugmenter(replace(policy, rir_probability=0), guard=guard.check, fixture_mode=True)
    noise_mix, noise_receipt = noise_only.apply(source, row, **context)
    scale = noise_receipt["peak_scale"]
    residual = noise_mix.astype(np.float64) / scale - source
    measured_snr = 10 * np.log10(np.mean(source.astype(np.float64) ** 2) / np.mean(residual ** 2))
    check("requested_SNR_measured_within_0p001dB", abs(measured_snr - 15) < 0.001)
    loud, loud_receipt = noise_only.apply(source * 4, row, **context)
    check("peak_bound_rescales_without_clipping", float(np.max(np.abs(loud))) <= 0.9900001 and loud_receipt["peak_scale"] < 1)
    receipts = []
    reader = augmenter.audio_reader(lambda _: source, context_for=lambda _: context, record=receipts.append)
    check("existing_audio_reader_contract_exact", np.array_equal(reader(row), transformed) and receipts[0] == receipt)
    bad_registry = out / "bad_hash_registry.json"
    altered = {**assets[0], "sha256": "0" * 64}
    bad_registry.write_text(json.dumps({"schema": 1, "assets": [altered]}), encoding="utf-8")
    rejects("asset_hash_mismatch_rejected", lambda: TrainWaveformAugmenter(replace(policy, rir_probability=0, registry_path=str(bad_registry)), guard=guard.check, fixture_mode=True))
    # Distribution is only a deterministic policy check, not a quality or corpus claim.
    synthetic = TrainWaveformAugmenter(AugmentationPolicy(enabled=True, noise_mode="synthetic_gaussian",
        noise_probability=1, rir_probability=0, apply_probability=0.5), guard=guard.check)
    applied = sum(synthetic.apply(source[:800], row, **{**context, "draw_index": i})[1]["applied"] for i in range(100))
    check("probability_half_has_clean_and_augmented_draws", 25 <= applied <= 75)
    actual_upstream_receipt = None
    if args.real_rir_registry:
        actual_policy = AugmentationPolicy(enabled=True, apply_probability=1,
            noise_probability=1, rir_probability=1, noise_mode="synthetic_gaussian",
            registry_path=args.real_rir_registry, snr_db=(15, 15))
        actual = TrainWaveformAugmenter(actual_policy, guard=guard.check)
        actual_audio, actual_upstream_receipt = actual.apply(source, row, **context)
        restored = TrainWaveformAugmenter(actual_policy, guard=guard.check)
        check("actual_pinned_FastMSS_RIR_and_synthetic_noise_applied", actual_upstream_receipt["applied"] and
            actual_upstream_receipt["operations"][0]["id"].startswith("FastMSS_") and
            actual_audio.shape == source.shape and np.isfinite(actual_audio).all())
        check("fresh_instance_resume_reproduces_actual_RIR_and_noise_exactly",
              np.array_equal(restored.apply(source, row, **context)[0], actual_audio))
        sf.write(out / "actual_FastMSS_RIR_plus_synthetic_noise_fixture.flac", actual_audio, fs, subtype="PCM_16")
    maximum_source = np.resize(source, fs * 120)
    maximum_mix, maximum_receipt = augmenter.apply(maximum_source, row, **context)
    check("declared_120_second_bound_actually_transformed", maximum_mix.shape == maximum_source.shape and
          np.isfinite(maximum_mix).all() and maximum_receipt["applied"] and
          float(np.max(np.abs(maximum_mix))) <= 0.9900001)
    sf.write(out / "clean_fixture.flac", source, fs, subtype="PCM_16")
    sf.write(out / "augmented_fixture.flac", transformed, fs, subtype="PCM_16")
    # Small CPU cost observation; never a GPU/training throughput claim.
    started = time.perf_counter()
    for i in range(20):
        augmenter.apply(source, row, **{**context, "draw_index": i})
    seconds = time.perf_counter() - started
    guard.check()
    report = {"status": "passed", "checks": results, "checks_passed": len(results),
        "fixture_only": True, "upstream_RIR_generation_executed": False,
        "actual_sample_rate": fs, "clip_seconds": 2, "measured_noise_SNR_db": float(measured_snr),
        "observed_20_transforms_seconds": seconds, "observed_audio_seconds_per_CPU_second": 40 / seconds,
        "observed_probability_half_count_out_of_100": applied, "example_receipt": receipt,
        "actual_upstream_RIR_consumption_receipt": actual_upstream_receipt,
        "maximum_duration_bound_actually_exercised_seconds": 120,
        "policy_sha256": augmenter.policy_sha256, "resources": guard.report(),
        "native_GPU_or_full_trainer_validation": False, "production_enabled": False,
        "torch_imported": "torch" in sys.modules, "new_downloads": False}
    (out / "CPU_validation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "passed", "checks": len(results), "report": str(out / "CPU_validation.json"),
                      "resources": report["resources"]}))
    guard.close()


if __name__ == "__main__":
    main()
