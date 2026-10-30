"""Tiny synthetic NumPy callback/file protocol; never actual DATA admission."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import weakref

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "outputs/parakeet_turkish"))
import fastmss_phase_aware_feature_window_v1 as sink
from resource_watchdog_v2 import sample_memory


def guard():
    value = sample_memory(os.getpid(), "synthetic_NumPy_phase_sink_protocol")
    worker, system = value["worker_private_memory"], value["system_memory"]
    assert worker["available"] and system["available"] and value["paired_sampling_within_one_second"]
    assert worker["private_bytes"] < 2_000_000_000 and system["commit_available_bytes"] >= 2_000_000_000
    assert system["commit_used_bytes"] / system["commit_limit_bytes"] < .90
    assert shutil.disk_usage(sink.ROOT).free >= 100_000_000_000


def numpy():
    sys.path.insert(0, str(PROJECT / "work/training-env/Lib/site-packages"))
    import numpy as np
    return np


def case(folder, phase, *, transform=None, frontend=None):
    np = numpy()
    folder.mkdir(exist_ok=False)
    names = ["synthetic_owner.json", "synthetic_update.json"] + [f"u00_r{i:02}_{suffix}.npy" for i in range(2) for suffix in ("features", "mask")]
    owner = {"kind": "SYNTHETIC_phase_sink_protocol_owner_NOT_actual_DATA", "owned_names": names, "capacity_bytes": sink.CAPACITY}
    sink.write_new(folder / names[0], owner)
    store = sink._Store(folder, owner, guard, lambda: sink._lock(folder.parent / (folder.name + ".synthetic.lock")))
    raw_rows, entries, contexts, references, waves = [], [], [], [], {}
    identity = {"policy_sha256": "a" * 64, "SYNTHETIC_NOT_FROZEN_POLICY": True}
    for i, language in enumerate(("tr", "en")):
        row = {"source_id": "SYNTHETIC_" + language, "language": language, "text": "Unchanged supplied fixture label",
            "target_tokens": 3 if i == 0 else 2, "duration": .02, "encoder_frames": 1,
            "augmentation_role": "student_tr" if i == 0 else "replay"}
        row["tr_metadata_labels" if i == 0 else "clean_target_ids"] = [4, 8, 8700] if i == 0 else [6, 7]
        if i == 0 and phase == sink.bridge.CLEANER:
            row["cleaner_original_ordinal"] = 1
        wave = np.linspace(.001, .12, 320, dtype=np.float32)
        waves[row["source_id"]] = wave
        ids, ids_sha = sink.bridge.full_targets(row)
        context = {"source_id": row["source_id"], "language": language, "phase": phase, "split": "train", "epoch": 0, "draw_index": i}
        reference = {"context": context, "full_target_ids": ids, "expected_audio": {"sample_rate": 16000, "samples": len(wave), "waveform_sha256": sink._wave_hash(wave, guard)}}
        raw_rows.append(row); contexts.append(context); references.append(reference)
        entries.append({"source_id": row["source_id"], "language": language, "role": row["augmentation_role"], "context": context,
            "full_target_ids": ids, "target_ids_u32_sha256": ids_sha, "raw_text_sha256": hashlib.sha256(row["text"].encode()).hexdigest(), "original_row_sha256": sink.digest(row)})
    update = {"update_index": 0, "phase": phase, "batch_id": "SYNTHETIC_update1", "coordinator_update": {"rows": raw_rows, "contexts": contexts}, "rows": entries}
    observed = {"clean_reads": 0, "weak_transforms": 0, "frontend_calls": 0, "previous_row_buffers_released": True}
    last = [None, None]

    def reader(row):
        observed["previous_row_buffers_released"] &= all(v is None or v() is None for v in last)
        result = waves[row["source_id"]].copy()
        last[0] = weakref.ref(result); observed["clean_reads"] += 1
        return result

    def weak(wave, row, context):
        observed["weak_transforms"] += 1
        if transform is not None:
            return transform(wave, row, context)
        result = (wave + np.float32(.01)).astype(np.float32)
        last[1] = weakref.ref(result)
        return result, {"policy_sha256": identity["policy_sha256"], "source_id": row["source_id"], "applied": True,
            "label_policy": "unchanged_single_target", "operations": [{"kind": "noise", "id": "SYNTHETIC_CALLBACK_NOT_REAL_AUGMENTATION"}],
            "speed_or_speaker_mixing": False, "samples_unchanged": len(wave), "sample_rate": 16000,
            "epoch": context["epoch"], "draw_index": context["draw_index"]}

    def feature(wave):
        observed["frontend_calls"] += 1
        if frontend is not None:
            return frontend(wave)
        features = np.zeros((1, 16, 128), dtype=np.float32)
        features[:, :8] = np.mean(wave, dtype=np.float32)
        mask = np.zeros((1, 16), dtype=np.bool_); mask[:, :8] = True
        return {"input_features": features, "attention_mask": mask}

    protocol = sink._RowProtocol(update, references, identity, store, clean_reader=reader, weak_transform=weak,
        transform_identity=identity, frontend=feature, guard=guard)
    return protocol, store, observed, update


def produce(folder):
    folder.mkdir(exist_ok=False)
    result = {"kind": "SYNTHETIC_phase_sink_child_NOT_actual_DATA", "producer": sink.process_identity(), "cases": []}
    for name, phase in (("weak", sink.bridge.WEAK), ("cleaner", sink.bridge.CLEANER)):
        protocol, store, observed, update = case(folder / name, phase)
        for index in range(2):
            protocol.prepare_row(index)
        rows, payload = protocol.prepared()
        item = {"kind": "SYNTHETIC_feature_update_NOT_native", "rows": rows, "prepared_payload": payload,
            "native_transaction_sha256": "b" * 64, "before_sampler_sha256": "c" * 64}
        binding = store.write_metadata("synthetic_update.json", item)
        result["cases"].append({"folder": str(store.folder), "owner": store.owner, "update": binding, "observed": observed, "phase": phase})
    result["actual_original_frontend_or_audio_or_DATA"] = False
    result["numerical_runtime"] = {"numpy": numpy().__version__, "numpy_file": numpy().__file__}
    sink.write_new(folder / "synthetic_child_result.json", result)


def main(folder):
    folder.mkdir(exist_ok=False)
    before_cache = {p.name: sink.record(p, guard) if p.is_file() else "existing_owned_directory" for p in sink.CACHE.iterdir()} if sink.CACHE.exists() else None
    child_folder, child_guard = folder / "synthetic_child", folder / "synthetic_child_guard"
    command = [sys.executable, "-B", str(PROJECT / "work/research/ASR18_bounded_CPU_phase_v3.py"), "--folder", str(child_guard), "--seconds", "40", "--script", str(Path(__file__).resolve()), "--", "--child", "--folder", str(child_folder)]
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = child.communicate(timeout=45)
    assert child.returncode == 0, (stdout, stderr)
    result = sink.read_bound(sink.record(child_folder / "synthetic_child_result.json", guard), guard)
    monitor = sink.read_bound(sink.record(child_guard / "closure.json", guard), guard)
    assert (result["producer"]["pid"] == child.pid or result["producer"]["ppid"] == child.pid) and result["producer"]["pid"] == monitor["pid"] and monitor["actual_exit_code"] == 0 and monitor["monitor_clean"] and monitor["monitor_joined"]
    assert monitor["observed"]["samples"] >= 3 and sink.producer_closed(result["producer"])
    checks = []

    def check(value, message):
        assert value, message
        checks.append(message)

    def reject(callback, message):
        try:
            callback()
        except (ValueError, KeyError, TypeError, AssertionError, OSError, IndexError):
            checks.append(message)
            return
        raise AssertionError("Expected rejection: " + message)

    check(not sink.producer_closed(sink.process_identity()), "Live parent cannot be claimed closed")
    check(sink.producer_closed(result["producer"]), "Actual owned synthetic NumPy child waited0 and closed")
    stores = []
    for item in result["cases"]:
        store = sink._Store(Path(item["folder"]), item["owner"], guard, lambda item=item: sink._lock(folder / (item["phase"] + ".read.lock")))
        stores.append(store)
        update = sink.read_bound(item["update"], guard)
        context = [r["context"] for r in update["rows"]]
        cursor = sink._ReadCursor(store, lambda index, item=item: item["update"], updates=1, guard=guard)
        reject(lambda: cursor.prepared_payload(0), "Incomplete payload rejected:" + item["phase"])
        for row_index in range(2):
            arrays = cursor.take_row(0, row_index, context=context[row_index], native_transaction_sha256="b" * 64, before_sampler_sha256="c" * 64)
            check(set(arrays) == {"input_features", "attention_mask"} and str(arrays["input_features"].dtype) == "float32" and str(arrays["attention_mask"].dtype) == "bool", "Exact native reader array protocol:" + item["phase"] + str(row_index))
        payload = cursor.prepared_payload(0)
        check(payload["committed_frontier_advanced"] is False and payload["state"] == "prepared_not_committed", "Preparation never commits sampler:" + item["phase"])
        payload["rows"][0]["augmentation"]["applied"] = "ALTERED_COPY"
        check(cursor.prepared_payload(0)["rows"][0]["augmentation"]["applied"] != "ALTERED_COPY", "Payload defensive copy:" + item["phase"])
        reject(lambda: cursor.take_row(0, 0, context=context[0], native_transaction_sha256="b" * 64, before_sampler_sha256="c" * 64), "Reused consumed row rejected:" + item["phase"])
        fresh = sink._ReadCursor(store, lambda index, item=item: item["update"], updates=1, guard=guard)
        reject(lambda: fresh.take_row(0, 1, context=context[1], native_transaction_sha256="b" * 64, before_sampler_sha256="c" * 64), "Wrong row order poisons reader:" + item["phase"])
        reject(lambda: fresh.take_row(0, 0, context=context[0], native_transaction_sha256="b" * 64, before_sampler_sha256="c" * 64), "Failed reader cannot retry:" + item["phase"])
        check(item["observed"]["previous_row_buffers_released"], "Serial row buffers released before next read:" + item["phase"])
        check(item["observed"]["weak_transforms"] == (1 if item["phase"] == sink.bridge.WEAK else 0), "Cleaner/replay never invoke weak transform:" + item["phase"])
        check(all(r["clean_PCM_sha256"] == r["prepared_PCM_sha256"] for r in update["rows"][1:]), "Replay clean PCM byteexact:" + item["phase"])
        if item["phase"] == sink.bridge.CLEANER:
            check(update["prepared_payload"]["protection_kind"] == sink.CLEAN_PROTECTION and all(r["augmentation"]["applied"] is False for r in update["prepared_payload"]["rows"]), "Frozen CleanPhaseOrderedAudioReader payload used")
        reject(lambda item=item: sink.owner_folder(sink.record(Path(item["folder"]) / "synthetic_owner.json", guard), guard), "Synthetic owner rejected by public actual sink")
    np = numpy()
    neg = []
    def test_case(name, **kw):
        protocol, store, observed, update = case(folder / name, sink.bridge.WEAK, **kw)
        neg.append(store)
        return protocol
    p = test_case("wrong_order")
    reject(lambda: p.prepare_row(1), "Producer row order rejected")
    reject(lambda: p.prepare_row(0), "Failed producer cannot retry changed input")
    p = test_case("incomplete")
    reject(p.prepared, "Incomplete producer payload rejected")
    def mutate_source(wave, row, context):
        wave[0] += 1
        return wave, {"policy_sha256": "a" * 64, "source_id": row["source_id"], "applied": False, "label_policy": "unchanged_single_target"}
    p = test_case("mutated_source", transform=mutate_source)
    reject(lambda: p.prepare_row(0), "Clean source mutation rejected")
    def silent_change(wave, row, context):
        return wave + np.float32(.01), {"policy_sha256": "a" * 64, "source_id": row["source_id"], "applied": False, "label_policy": "unchanged_single_target"}
    p = test_case("unreported_transform", transform=silent_change)
    reject(lambda: p.prepare_row(0), "Unapplied receipt cannot hide PCM change")
    def changed_label(wave, row, context):
        row["text"] = "MUTATED"
        return wave, {"policy_sha256": "a" * 64, "source_id": row["source_id"], "applied": False, "label_policy": "unchanged_single_target"}
    p = test_case("changed_label", transform=changed_label)
    reject(lambda: p.prepare_row(0), "Raw label mutation rejected")
    def bad_frontend(wave):
        wave[0] += 1
        return {"input_features": np.zeros((1, 8, 128), np.float32), "attention_mask": np.ones((1, 8), np.bool_)}
    p = test_case("frontend_mutation", frontend=bad_frontend)
    reject(lambda: p.prepare_row(0), "Frontend PCM mutation rejected")
    def bad_mask(wave):
        mask = np.ones((1, 8), np.bool_); mask[0, 2] = False
        return {"input_features": np.zeros((1, 8, 128), np.float32), "attention_mask": mask}
    p = test_case("bad_mask", frontend=bad_mask)
    reject(lambda: p.prepare_row(0), "Non-right-padded mask rejected")
    store = stores[0]
    unknown = store.folder / "UNOWNED_fixture_sentinel.txt"
    unknown.write_bytes(b"SYNTHETIC_FIXTURE_ONLY")
    reject(store.inventory, "Unknown owned-directory file blocks all cleanup")
    assert unknown.resolve().is_relative_to(folder.resolve())
    unknown.unlink()
    check(store.inventory() < 32768, "Tiny synthetic feature files bounded")
    for index, store in enumerate(stores + neg):
        archive = folder / f"durable_synthetic_metadata_{index:02}.json"
        removed, archived = sink._retire_files(store, archive={"kind": "SYNTHETIC_PROTOCOL_NOT_COHERENT_NATIVE_BOUNDARY", "actual_child_closed": sink.producer_closed(result["producer"]), "synthetic_only": True}, evidence_path=archive)
        check(not store.folder.exists() and archive.exists() and archived["bytes"] > 0 and removed, "Metadata-first exact synthetic file retirement:" + str(index))
    after_cache = {p.name: sink.record(p, guard) if p.is_file() else "existing_owned_directory" for p in sink.CACHE.iterdir()} if sink.CACHE.exists() else None
    check(before_cache == after_cache, "Actual shared feature CACHE unchanged")
    check(not any(v in sys.modules for v in ("torch", "transformers", "librosa", "scipy", "pyarrow", "av", "soundfile")), "No Torch/frontend/audio/model imports")
    report = {"kind": "FastMSS_synthetic_NumPy_phase_sink_protocol_controls_v1", "passed": True, "checks": checks, "check_count": len(checks),
        "source": sink.record(sink.__file__, guard), "fixture_source": sink.record(__file__, guard), "actual_NumPy_child": result["producer"],
        "actual_owned_child_wait_exit_code": child.returncode, "child_guard": sink.record(child_guard / "closure.json", guard),
        "actual_shared_CACHE_unchanged": True, "actual_final_DATA_positive_path_exercised": False, "actual_original_frontend_executed": False,
        "actual_source_audio_prepared": False, "actual_native_updates": 0, "synthetic_protocol_only": True, "production_enabled": False,
        "unexercised_positive_paths": ["genuine final DATA owner/create", "actual phase-aware producer/decoder/original frontend", "actual coherent checkpoint resume/final cleanup", "native-v2 GPU integration"]}
    sink.write_new(folder / "report.json", report)
    print(json.dumps({"passed": True, "checks": len(checks), "synthetic_NumPy_protocol_only": True}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    target = sink.scoped(args.folder)
    assert target.is_relative_to((sink.ROOT / "analysis/FastMSS_integration_v1").resolve())
    produce(target) if args.child else main(target)
