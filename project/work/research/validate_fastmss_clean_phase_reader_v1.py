"""Tiny guarded CPU proof of direct clean-phase audio and full target identity."""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys

PROJECT = Path(__file__).resolve().parents[2]
ROOT = Path(r"E:\gpu_training_scratchpad")
sys.path.insert(0, str(PROJECT / "outputs/parakeet_turkish"))
from atomic_io_retry_v1 import write_json
from resource_watchdog_v2 import sample_memory


def main(output):
    out = Path(output).resolve()
    assert out.is_relative_to(ROOT.resolve()) and os.environ["CUDA_VISIBLE_DEVICES"] == "-1"
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        assert os.environ[key] == "1"
    out.mkdir(parents=True, exist_ok=False)
    checks = {}

    def guard():
        value = sample_memory(os.getpid(), "clean_phase_fixture")
        worker, system = value["worker_private_memory"], value["system_memory"]
        assert worker["available"] and system["available"] and value["paired_sampling_within_one_second"]
        assert worker["private_bytes"] < 2_000_000_000 and system["commit_available_bytes"] >= 2_000_000_000
        assert system["commit_used_bytes"] / system["commit_limit_bytes"] < .90
        assert shutil.disk_usage(ROOT).free >= 100_000_000_000

    def check(name, value):
        assert value, name
        checks[name] = True

    def rejects(name, function):
        try:
            function()
        except (ValueError, TypeError, KeyError, AssertionError):
            checks[name] = True
        else:
            raise AssertionError(name)

    guard()
    import numpy as np
    from fastmss_train_augmentation_v1 import AugmentationPolicy, TrainWaveformAugmenter, canonical
    from fastmss_ordered_draw_binding_v1 import AugmentationDrawContext as Draw, FastMSSOrderedDrawBinding, file_binding
    from fastmss_clean_phase_reader_v1 import CleanPhaseOrderedAudioReader as Reader, PHASE
    import fastmss_cpu_prepared_transport_v1 as transport
    source = PROJECT / "outputs/parakeet_turkish"
    policy = source / "fastmss_broad_optin_v1.json"
    augmenter = TrainWaveformAugmenter(AugmentationPolicy.from_json(policy), guard=guard)
    binding = FastMSSOrderedDrawBinding(augmenter, policy_path=policy, adapter_path=source / "fastmss_train_augmentation_v1.py")
    forbidden_calls = []

    def forbidden(*args, **kwargs):
        forbidden_calls.append(True)
        raise AssertionError("Augmentation/transport reached by cleaner audio")

    augmenter.apply = forbidden
    transport.prepare_TR_transport = forbidden
    transport.VerifiedPreparedAudioReader = forbidden
    transport.cleanup_owned_cache = forbidden
    wave = (np.sin(np.arange(1600, dtype=np.float32) * np.float32(.093)) * np.float32(.2)).astype(np.float32)
    wave_sha = hashlib.sha256(wave.tobytes()).hexdigest()
    full = [23 + v % 600 for v in range(419)]
    specifications = [("duplicate_TR", "tr", "student_tr", full), ("duplicate_TR", "tr", "student_tr", full)]
    specifications += [("old_" + lang, lang, "replay", [41, 102]) for lang in ("en", "es", "it", "fr", "de", "pt")]
    specifications += [("teacher_TR", "tr", "teacher", full), ("teacher_PT", "pt", "teacher", [32, 70, 90])]
    rows, contexts, targets = [], [], []
    for index, (name, language, role, ids) in enumerate(specifications):
        draw = Draw(name, language, 2, 100 + index, phase=PHASE)
        row = {"source_id": name, "language": language, "augmentation_role": role, "text": "Ham metin " + name,
               "duration": .1, "encoder_frames": 13, "target_tokens": len(ids), "audio_draw_context": asdict(draw)}
        row["tr_metadata_labels" if language == "tr" else "clean_target_ids"] = list(ids)
        rows.append(row); contexts.append(draw); targets.append(list(ids))
    audio = [{"sample_rate": 16000, "samples": len(wave), "waveform_sha256": wave_sha} for _ in rows]
    original_rows, caller_state = canonical(rows), {"optimizer_updates": 4, "cleaner_draws": 0, "journal_offset": 119}
    original_state = dict(caller_state)
    rng_before, numpy_before = random.getstate(), np.random.get_state()
    events = []

    def make(*, actual_rows=None, draws=None, ids=None, pcm=None, reader=None, callback=None):
        return Reader(binding, rows if actual_rows is None else actual_rows, contexts if draws is None else draws,
            batch_id="cleaner_fixture_uncommitted", clean_reader=(lambda row: wave) if reader is None else reader,
            expected_target_ids=targets if ids is None else ids, expected_audio=audio if pcm is None else pcm,
            guard=guard, record_prepared=events.append if callback is None else callback)

    reader = make()
    rejects("incomplete_payload_rejected", reader.prepared_payload)
    for row in rows:
        check("exact_object_and_bytes_" + str(len(events)), reader(row) is wave and hashlib.sha256(wave.tobytes()).hexdigest() == wave_sha)
    payload = reader.prepared_payload()
    check("all_six_replay_and_two_teachers_protected", len(payload["rows"]) == 10 and
          all(not v["augmentation"]["applied"] for v in payload["rows"]))
    check("full419_targets_not_truncated", payload["rows"][0]["clean_protection"]["target_tokens"] == 419)
    check("duplicate_source_distinct_draws_preserved", payload["rows"][0]["draw"]["source_id"] == payload["rows"][1]["draw"]["source_id"] and
          payload["rows"][0]["draw"]["draw_index"] != payload["rows"][1]["draw"]["draw_index"])
    check("prepared_does_not_commit", payload["state"] == "prepared_not_committed" and not payload["committed_frontier_advanced"] and caller_state == original_state)
    check("source_rows_and_wave_unchanged", canonical(rows) == original_rows and hashlib.sha256(wave.tobytes()).hexdigest() == wave_sha)
    check("no_augmentation_transport_calls", not forbidden_calls)
    check("policy_adapter_registry_helper_identity_bound", payload["identity"]["frozen_broad_augmentation"] == binding.identity)
    payload["rows"][0]["augmentation"]["applied"] = True
    check("payload_defensive_copy", reader.prepared_payload()["rows"][0]["augmentation"]["applied"] is False)
    identity = reader.identity; identity["phase"] = "wrong"
    check("identity_defensive_copy", reader.identity["phase"] == PHASE)
    rejects("consume_once", lambda: reader(rows[0]))
    rejects("empty_rows", lambda: make(actual_rows=[]))
    rejects("missing_context", lambda: make(draws=contexts[:-1]))
    rejects("missing_fulltargets", lambda: make(ids=targets[:-1]))
    rejects("missing_audio_identity", lambda: make(pcm=audio[:-1]))
    wrong = list(contexts); wrong[0] = replace(wrong[0], phase="weak_label_continued_training")
    rejects("weak_phase_rejected", lambda: make(draws=wrong))
    rejects("DEV_context_rejected", lambda: replace(contexts[0], split="dev"))
    wrong = list(contexts); wrong[1] = contexts[0]
    rejects("duplicate_draw_position_rejected", lambda: make(draws=wrong))

    def bad_constructor(name, field, value, index=0):
        changed = deepcopy(rows); changed[index][field] = value
        rejects(name, lambda: make(actual_rows=changed))

    bad_constructor("wrong_source", "source_id", "wrong")
    bad_constructor("wrong_language", "language", "en")
    bad_constructor("missing_role", "augmentation_role", None)
    bad_constructor("wrong_role", "augmentation_role", "replay")
    bad_constructor("partial_TR_labels", "tr_metadata_labels", full[:180])
    bad_constructor("wrong_target_count", "target_tokens", 418)
    bad_constructor("wrong_replay_fulltargets", "clean_target_ids", [41], index=2)
    altered = asdict(contexts[0]); altered["epoch"] = False
    bad_constructor("wrong_context_integer_type", "audio_draw_context", altered)
    for name, updates in (("wrong_rate", {"sample_rate": 24000}), ("invalid_hash", {"waveform_sha256": "x" * 64})):
        changed = deepcopy(audio); changed[0].update(updates)
        rejects(name, lambda: make(pcm=changed))

    def bad_consume(name, row=None, waveform=None):
        instance = make(reader=(lambda item: waveform) if waveform is not None else None)
        rejects(name, lambda: instance(rows[0] if row is None else row))
        rejects(name + "_poisoned", lambda: instance(rows[0]))
        rejects(name + "_no_payload", instance.prepared_payload)

    bad_consume("out_of_order", row=rows[2])
    changed = deepcopy(rows[0]); changed["text"] += " değişti"
    bad_consume("text_changed", row=changed)
    changed = deepcopy(rows[0]); changed["tr_metadata_labels"][0] += 1
    bad_consume("same_length_target_changed", row=changed)
    changed = deepcopy(rows[0]); changed["audio_draw_context"]["draw_index"] += 1
    bad_consume("draw_changed", row=changed)
    bad_consume("wrong_wave_dtype", waveform=wave.astype(np.float64))
    bad_consume("wrong_wave_samples", waveform=wave[:-1])
    bad_consume("wrong_wave_ndim", waveform=wave.reshape(1, -1))
    bad_consume("noncontiguous_wave", waveform=np.repeat(wave, 2)[::2])
    corrupted = wave.copy(); corrupted[0] = np.nan
    bad_consume("nonfinite_wave", waveform=corrupted)
    corrupted = wave.copy(); corrupted[0] += .01
    bad_consume("wrong_clean_wave_hash", waveform=corrupted)
    check("global_RNG_unchanged", random.getstate() == rng_before and np.random.get_state()[0] == numpy_before[0] and
          np.array_equal(np.random.get_state()[1], numpy_before[1]) and np.random.get_state()[2:] == numpy_before[2:])
    check("no_Torch_SciPy_imports", not any(v == "torch" or v.startswith("torch.") or v == "scipy" or v.startswith("scipy.") for v in sys.modules))
    check("caller_frontier_still_unchanged_after_failures", caller_state == original_state)
    guard()
    write_json(out / "clean_phase_validation.json", {"passed": True, "PID": os.getpid(), "meaningful_checks": len(checks), "checks": checks,
        "source": file_binding(source / "fastmss_clean_phase_reader_v1.py"), "fixture_source": file_binding(__file__),
        "binding_identity": binding.identity, "prepared_payload": reader.prepared_payload(),
        "no_augmentation_transport_or_cache_creation": True, "no_native_training_or_quality_claim": True,
        "CUDA_masked": True, "all_six_replay_and_teacher_clean": True, "sampler_journal_frontier_advanced": False})
    print(json.dumps({"passed": True, "checks": len(checks), "PID": os.getpid()}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", required=True)
    main(parser.parse_args().output)
