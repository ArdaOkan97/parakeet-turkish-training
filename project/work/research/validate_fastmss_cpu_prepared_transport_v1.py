"""Fresh bounded CPU child transport/lifecycle proof, never native training."""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from fastmss_cpu_guard_v1 import Guard, output_path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "outputs/parakeet_turkish"))
from fastmss_train_augmentation_v1 import AugmentationPolicy, TrainWaveformAugmenter, canonical
from fastmss_ordered_draw_binding_v1 import AugmentationDrawContext as Draw, FastMSSOrderedDrawBinding
from fastmss_cpu_prepared_transport_v1 import (
    CACHE_ROOT, MAX_CACHE_BYTES, prepare_TR_transport, VerifiedPreparedAudioReader,
    observe_closed_child, cleanup_owned_cache, record, capacity_usage,
)


def binding_for(policy_path, guard):
    augmenter = TrainWaveformAugmenter(AugmentationPolicy.from_json(policy_path), guard=guard.check)
    return FastMSSOrderedDrawBinding(augmenter, policy_path=policy_path,
        adapter_path=PROJECT / "outputs/parakeet_turkish/fastmss_train_augmentation_v1.py")


def child(request_path, result_path):
    guard = Guard(wall_seconds=90)
    import numpy as np
    request = json.loads(Path(request_path).read_text())
    binding = binding_for(request["policy"], guard)
    if binding.identity != request["binding_identity"]:
        raise ValueError("CPU child static policy/environment binding differs")
    rows = request["rows"]
    contexts = tuple(Draw(**value) for value in request["contexts"])
    calls = 0

    def reader(row):
        nonlocal calls
        calls += 1
        if request.get("abort_second_read") and calls == 2:
            raise ValueError("Intentional fixture producer failure, zero optimizer updates")
        return np.load(row["audio_filepath"], allow_pickle=False)

    result = {}
    exit_code = 0
    try:
        result = prepare_TR_transport(binding, rows, contexts, batch_id=request["batch_id"],
            job_id=request["job_id"], clean_reader=reader, guard=guard.check)
    except ValueError as error:
        if not request.get("abort_second_read"):
            raise
        result = {"owner": record(CACHE_ROOT / request["job_id"] / "owner.json", guard.check),
                  "manifest": None, "intentional_fixture_error": str(error)}
        exit_code = 13
    result.update(resources=guard.report(), scipy_signal_imported="scipy.signal" in sys.modules,
        Torch_imported="torch" in sys.modules, sampler_journal_frontier_advanced=False)
    Path(result_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    if request.get("wait_release"):
        deadline = time.monotonic() + 20
        while not Path(request["wait_release"]).exists():
            guard.check()
            if time.monotonic() > deadline:
                raise ValueError("Fixture release did not arrive")
            time.sleep(0.05)
    guard.close()
    return exit_code


def main(output):
    out = output_path(output)
    guard = Guard(wall_seconds=120)
    import numpy as np
    checks = {}

    def check(name, value):
        if not value:
            raise AssertionError(name)
        checks[name] = True

    def rejects(name, function):
        try:
            function()
        except (ValueError, TypeError, FileNotFoundError, subprocess.TimeoutExpired):
            checks[name] = True
        else:
            raise AssertionError(name)

    source = PROJECT / "outputs/parakeet_turkish"
    policy_json = json.loads((source / "fastmss_train_augmentation_v1.pilot.disabled.json").read_text())
    policy_json.update(enabled=True, apply_probability=1.0, noise_probability=1.0, rir_probability=1.0)
    policy_path = out / "enabled_all_on_fixture_policy.json"
    policy_path.write_text(json.dumps(policy_json, indent=2), encoding="utf-8")
    binding = binding_for(policy_path, guard)
    wave = (0.2 * np.sin(np.arange(16000, dtype=np.float32) * (2 * np.pi * 231 / 16000))).astype(np.float32)
    clean_path = out / "original_clean_fixture.npy"
    np.save(clean_path, wave, allow_pickle=False)
    clean_record = record(clean_path, guard.check)
    clean_sha = hashlib.sha256(wave.tobytes()).hexdigest()
    tr = {"source_id": "TR_duplicate_source", "language": "tr", "augmentation_role": "student_tr", "text": "Kaynak metin değişmez.",
          "duration": 1.0, "encoder_frames": 13, "target_tokens": 3,
          "tr_metadata_labels": [72, 8199, 81], "audio_filepath": str(clean_path)}
    tr_rows = [dict(tr), dict(tr)]
    tr_contexts = [Draw(tr["source_id"], "tr", 0, i) for i in (17, 18)]
    all_rows = tr_rows + [{"source_id": "replay_" + language, "language": language, "augmentation_role": "replay", "text": language,
        "duration": 1.0, "encoder_frames": 13, "target_tokens": 2} for language in ("en", "es", "it", "fr", "de", "pt")]
    all_contexts = tr_contexts + [Draw(r["source_id"], r["language"], 0, i + 19) for i, r in enumerate(all_rows[2:])]
    all_rows.append({**tr, "source_id": "clean_finetune_TR"})
    all_contexts.append(Draw("clean_finetune_TR", "tr", 0, 25, phase="cleaner_data_fine_tuning"))
    all_rows.append({**tr, "source_id": "TR_teacher", "augmentation_role": "teacher"})
    all_contexts.append(Draw("TR_teacher", "tr", 0, 26))
    batch_id = "fixture_update0_no_optimizer_commit"
    original_sampler = {"draws": 0, "optimizer_updates": 0}
    sampler = dict(original_sampler)
    jobs = []

    def launch(number, *, wait_release=False, abort=False):
        job = "fixture_" + uuid.uuid4().hex[:16]
        request_path = out / f"request_{number}.json"
        result_path = out / f"child_result_{number}.json"
        release = out / f"release_{number}.json"
        request = {"job_id": job, "batch_id": batch_id, "policy": str(policy_path),
            "binding_identity": binding.identity, "rows": tr_rows, "contexts": [asdict(v) for v in tr_contexts],
            "wait_release": str(release) if wait_release else None, "abort_second_read": abort}
        request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")
        log = (out / f"child_{number}.log").open("x", encoding="utf-8")
        process = subprocess.Popen([sys.executable, "-X", "utf8", "-B", str(Path(__file__).resolve()),
            "--child-request", str(request_path), "--child-result", str(result_path)],
            stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy(), creationflags=subprocess.CREATE_NO_WINDOW)
        deadline = time.monotonic() + 30
        while not result_path.exists() and process.poll() is None:
            guard.check()
            if time.monotonic() > deadline:
                raise RuntimeError("Bounded fixture child timeout")
            time.sleep(0.05)
        result = json.loads(result_path.read_text())
        value = {"process": process, "log": log, "result": result, "release": release,
                 "closure_path": out / f"actual_CPU_closure_{number}.json"}
        jobs.append(value)
        return value

    first = launch(1, wait_release=True)
    r1 = first["result"]
    rejects("actual_live_child_cannot_publish_closure", lambda: observe_closed_child(first["process"],
        owner_record=r1["owner"], manifest_record=r1["manifest"], proof_path=first["closure_path"], guard=guard.check, timeout=0.01))
    check("no_fake_closure_written_while_child_alive", not first["closure_path"].exists())
    first["release"].write_text("release", encoding="utf-8")
    c1 = observe_closed_child(first["process"], owner_record=r1["owner"], manifest_record=r1["manifest"],
        proof_path=first["closure_path"], guard=guard.check, timeout=30)
    first["log"].close()
    # In parent, any accidental augmentation call is now a hard test failure.
    binding.augmenter.apply = lambda *a, **k: (_ for _ in ()).throw(AssertionError("Double augmentation in parent"))
    events = []

    def consumer(result=r1, closure=c1, rows=all_rows, contexts=all_contexts, clean_hashes=None):
        return VerifiedPreparedAudioReader(binding, rows, contexts, batch_id=batch_id,
            manifest_record=result["manifest"], closure_record=closure, protected_reader=lambda _: wave,
            record_prepared=events.append, guard=guard.check,
            expected_clean_hashes=clean_hashes or [clean_sha, clean_sha])

    reader = consumer()
    rejects("partial_batch_cannot_prepare_commit", reader.prepared_payload)
    outputs = [reader(row) for row in all_rows]
    payload1 = reader.prepared_payload()
    check("TR_prepared_once_without_parent_apply", all(not np.array_equal(outputs[i], wave) for i in (0, 1)))
    check("all_six_replay_languages_exact_clean", all(np.array_equal(value, wave) for value in outputs[2:8]))
    check("clean_finetune_TR_exact_clean", outputs[8] is wave)
    check("explicit_TR_teacher_exact_clean", outputs[9] is wave)
    check("payload_is_same_prepared_not_commit_contract", payload1["kind"] == "FastMSS_prepared_batch_v1" and
          payload1["committed_frontier_advanced"] is False and len(payload1["rows"]) == 10)
    check("duplicate_source_distinct_draws_not_collapsed", not np.array_equal(outputs[0], outputs[1]))
    check("SciPy_signal_only_in_closed_child", r1["scipy_signal_imported"] and "scipy.signal" not in sys.modules)
    rejects("prepared_waveform_cannot_be_consumed_twice", lambda: reader(all_rows[0]))
    rejects("missing_manifest_rejected", lambda: consumer(result={**r1, "manifest": {**r1["manifest"], "path": str(out / "missing.json")}}))
    rejects("wrong_manifest_SHA_rejected", lambda: consumer(result={**r1, "manifest": {**r1["manifest"], "sha256": "0" * 64}}))
    rejects("stale_draw_context_rejected", lambda: consumer(contexts=[replace(all_contexts[0], draw_index=999), *all_contexts[1:]]))
    rejects("wrong_phase_rejected", lambda: consumer(contexts=[replace(all_contexts[0], phase="cleaner_data_fine_tuning"), *all_contexts[1:]]))
    rejects("duplicate_draw_context_rejected", lambda: consumer(contexts=[all_contexts[0], all_contexts[0], *all_contexts[2:]]))
    rejects("missing_prepared_row_rejected", lambda: consumer(rows=all_rows[1:], contexts=all_contexts[1:]))
    rejects("wrong_target_ID_rejected", lambda: consumer(rows=[{**tr, "tr_metadata_labels": [72, 8200, 81]}, *all_rows[1:]]))
    rejects("wrong_raw_text_rejected", lambda: consumer(rows=[{**tr, "text": "changed"}, *all_rows[1:]]))
    rejects("wrong_sample_count_rejected", lambda: consumer(rows=[{**tr, "duration": 1.0 + 1 / 16000}, *all_rows[1:]]))
    rejects("wrong_clean_waveform_SHA_rejected", lambda: consumer(clean_hashes=["0" * 64, clean_sha]))
    rejects("wrong_teacher_role_rejected", lambda: consumer(rows=[{**tr, "augmentation_role": "teacher"}, *all_rows[1:]]))
    rejects("DEV_context_rejected", lambda: replace(all_contexts[0], split="dev"))
    rejects("TEST_context_rejected", lambda: replace(all_contexts[0], split="test"))
    ordered = consumer()
    rejects("out_of_order_actual_row_rejected", lambda: ordered(all_rows[2]))
    rejects("failed_reader_cannot_prepare_commit", ordered.prepared_payload)
    manifest1 = json.loads(Path(r1["manifest"]["path"]).read_text())
    cache = Path(manifest1["entries"][0]["cache"]["path"])
    untouched = cache.read_bytes()
    damaged = bytearray(untouched);damaged[-1] ^= 1
    pending = consumer()
    cache.write_bytes(damaged)
    rejects("cache_mutation_before_consume_rejected", lambda: pending(all_rows[0]))
    cache.write_bytes(untouched)  # exact own fixture bytes restored, no source assets edited
    second = launch(2)
    r2 = second["result"]
    c2 = observe_closed_child(second["process"], owner_record=r2["owner"], manifest_record=r2["manifest"],
        proof_path=second["closure_path"], guard=guard.check, timeout=30)
    second["log"].close()
    fresh = consumer(result=r2, closure=c2)
    fresh_outputs = [fresh(row) for row in all_rows]
    check("fresh_process_reprepare_exact_semantic_and_payload",
          r1["semantic_sha256"] == r2["semantic_sha256"] and fresh.prepared_payload() == payload1 and
          all(np.array_equal(a, b) for a, b in zip(outputs, fresh_outputs)))
    rejects("stale_other_producer_closure_rejected", lambda: consumer(result=r2, closure=c1))
    rejects("fixed_cache_outstanding_capacity_blocks_third_job", lambda: prepare_TR_transport(binding, tr_rows, tr_contexts,
        batch_id=batch_id, job_id="must_not_be_created", clean_reader=lambda _: wave, guard=guard.check))
    rejects("wrong_producer_phase_cannot_create_cache", lambda: prepare_TR_transport(binding, tr_rows,
        [replace(v, phase="cleaner_data_fine_tuning") for v in tr_contexts], batch_id=batch_id,
        job_id="wrong_phase_must_not_exist", clean_reader=lambda _: wave, guard=guard.check))
    rejects("teacher_cannot_be_CPU_augmentation_target", lambda: prepare_TR_transport(binding,
        [{**r, "augmentation_role": "teacher"} for r in tr_rows], tr_contexts, batch_id=batch_id,
        job_id="teacher_must_not_exist", clean_reader=lambda _: wave, guard=guard.check))
    check("fixed_cache_capacity_enforced", capacity_usage()[0] <= MAX_CACHE_BYTES and capacity_usage()[1] == 2)
    check("capacity_reservations_include_unwritten_work", capacity_usage()[0] >= 2 * 1024 * 1024)

    def boundary(result):
        return {"kind": "discard_uncommitted_preparation", "batch_id": batch_id,
            "owner_sha256": result["owner"]["sha256"], "sampler_before": original_sampler,
            "sampler_after": sampler, "fixture_no_native_training": True}

    def cleanup(result, closure, *, verifier=lambda value: value["sampler_before"] == value["sampler_after"]):
        return cleanup_owned_cache(owner_record=result["owner"], manifest_record=result["manifest"],
            closure_record=closure, boundary=boundary(result), verify_boundary=verifier, guard=guard.check)

    rejects("cleanup_without_verified_durable_or_discard_boundary_rejected", lambda: cleanup(r1, c1, verifier=lambda _: False))
    folder = Path(r1["manifest"]["path"]).parent
    foreign = folder / "unexpected_fixture_file.txt"
    foreign.write_text("owned by fixture but not cache inventory", encoding="utf-8")
    rejects("unknown_cache_file_causes_delete_nothing", lambda: cleanup(r1, c1))
    check("all_declared_cache_files_remain_after_cleanup_rejection", all((folder / name).exists() for name in ("owner.json", "manifest.json", "audio_000.npy", "audio_001.npy")))
    foreign.unlink()  # this exact file was created by this fixture
    cleanups = [cleanup(r1, c1), cleanup(r2, c2)]
    rejects("byte_capacity_blocks_oversized_job_before_decode", lambda: prepare_TR_transport(binding,
        [{**tr, "duration": MAX_CACHE_BYTES / 4 / 16000}], tr_contexts[:1], batch_id=batch_id,
        job_id="oversize_must_not_exist", clean_reader=lambda _: wave, guard=guard.check))
    check("cleanup_only_exact_owned_paths", all(len(v["removed_exact_owned_files"]) == 4 for v in cleanups))
    check("original_clean_source_never_deleted_or_changed", record(clean_path, guard.check) == clean_record)
    failed = launch(3, abort=True)
    r3 = failed["result"]
    c3 = observe_closed_child(failed["process"], owner_record=r3["owner"], manifest_record=None,
        proof_path=failed["closure_path"], guard=guard.check, timeout=30)
    failed["log"].close()
    check("failed_partial_producer_actual_exit_observed", failed["process"].returncode == 13 and r3["manifest"] is None)
    cleanups.append(cleanup(r3, c3))
    check("failed_partial_owned_cache_safely_discarded", not Path(r3["owner"]["path"]).parent.exists())
    check("no_sampler_journal_optimizer_frontier_advanced", sampler == original_sampler and all(not c["sampler_journal_frontier_advanced"] for c in cleanups))
    check("Torch_and_GPU_never_imported", "torch" not in sys.modules and all(not v["result"]["Torch_imported"] for v in jobs))
    check("both_source_configs_still_disabled", not json.loads((source / "fastmss_train_augmentation_v1.disabled.json").read_text())["enabled"] and
          not json.loads((source / "fastmss_train_augmentation_v1.pilot.disabled.json").read_text())["enabled"])
    report = {"status": "passed", "checks_passed": len(checks), "checks": checks, "resources_parent": guard.report(),
        "resources_actual_CPU_children": [j["result"]["resources"] for j in jobs],
        "actual_child_exit_codes": [j["process"].returncode for j in jobs], "closures": [c1, c2, c3],
        "producer_semantic_sha256": r1["semantic_sha256"], "manifest1_snapshot": manifest1,
        "prepared_payload_snapshot": payload1, "cleanup_receipts": cleanups,
        "source_clean_binding_preserved": clean_record, "cache_capacity_bytes": MAX_CACHE_BYTES,
        "parent_SciPy_signal_imported": "scipy.signal" in sys.modules, "native_production_caller_validation": False,
        "production_enabled": False, "no_native_training_or_sampler_commit_claim": True}
    path = out / "CPU_prepared_transport_validation.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "checks": len(checks), "report": str(path), "parent": report["resources_parent"],
                     "actual_child_exit_codes": report["actual_child_exit_codes"], "child_peaks": [r["peak_private_bytes"] for r in report["resources_actual_CPU_children"]]}))
    guard.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    parser.add_argument("--child-request")
    parser.add_argument("--child-result")
    args = parser.parse_args()
    if args.child_request:
        raise SystemExit(child(args.child_request, args.child_result))
    main(args.output)
