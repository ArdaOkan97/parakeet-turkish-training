"""Tiny contract-only NumPy frontend stub; NEVER an original-frontend proof.

Run only after the prior feature window is retired by its actual owner. A real
guarded reference subprocess supplies exact PCM/cache bindings. Source TR is a
clearly labelled synthetic fixture; replay uses the original protected WAV and
genuine cache bindings. No Torch, ASR, actual sampler or optimizer is loaded.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT / "outputs/parakeet_turkish"
ROOT = Path(r"E:\gpu_training_scratchpad")
ORIGINAL_BINDINGS_BYTES = 17986
ORIGINAL_BINDINGS_SHA = "84aee2288cb36c35724de29d3a55489e69e371a72f95f9f36f51a96357ba6c7d"
sys.path.insert(0, str(SOURCE))
from fastmss_feature_window_v1 import CACHE, CAPACITY, FRONTEND_ARGS, record, read_bound, write_new, scoped
from fastmss_clean_feature_contract_v1 import CleanFeatureUpdateProducer, KIND, PHASE, PROOF_KIND, digest, target_hash
from fastmss_ordered_draw_binding_v1 import AugmentationDrawContext, file_binding
from resource_watchdog_v2 import sample_memory


def guard():
    item = sample_memory(os.getpid(), "clean_feature_contract_fixture")
    worker, system = item["worker_private_memory"], item["system_memory"]
    assert worker["available"] and system["available"] and item["paired_sampling_within_one_second"]
    assert worker["private_bytes"] < 2_000_000_000 and system["commit_available_bytes"] >= 2_000_000_000
    assert system["commit_used_bytes"] / system["commit_limit_bytes"] < .90
    assert shutil.disk_usage(ROOT).free >= 100_000_000_000


def source_binding(path):
    return {**record(path, guard), "mtime_ns": Path(path).stat().st_mtime_ns}


def load_clean(row):
    import numpy as np
    if row["language"] == "tr":
        return np.load(row["audio_filepath"], allow_pickle=False)
    import soundfile as sf
    wave, sr = sf.read(row["audio_filepath"], dtype="float32", always_2d=False)
    assert sr == 16000 and wave.ndim == 1
    return wave


def references(request_path, output):
    import numpy as np
    request = json.loads(scoped(request_path).read_bytes())
    references = []
    for row, context, source in zip(request["rows"], request["contexts"], request["reference_sources"]):
        guard(); wave = load_clean(row)
        assert wave.dtype == np.float32 and wave.ndim == 1 and np.isfinite(wave).all()
        ids = row["tr_metadata_labels"] if row["language"] == "tr" else row["clean_target_ids"]
        references.append({"source_id": row["source_id"], "language": row["language"], "context": context,
            "source_row_sha256": digest(row), "source_audio_file": source_binding(row["audio_filepath"]),
            "expected_audio": {"sample_rate": 16000, "samples": len(wave), "waveform_sha256": hashlib.sha256(wave.tobytes()).hexdigest()},
            "full_target_ids": ids, "target_ids_u32_sha256": target_hash(ids, row["language"]),
            "cleaner_original_ordinal": row.get("cleaner_original_ordinal"),
            "genuine_teacher_feature_cache": None if row["language"] == "tr" else source["genuine_teacher_feature_cache"],
            "full_clean_original_audio": True, "augmentation": False})
        if row["language"] != "tr":
            assert source_binding(source["genuine_teacher_feature_cache"]["path"]) == source["genuine_teacher_feature_cache"]
        del wave
    write_new(scoped(output), {"kind": PROOF_KIND, "passed": True, "phase": PHASE,
        "cleaner_catalog_identity": request["cleaner_catalog_identity"], "actual_optimizer_updates": 0,
        "sampler_journal_frontier_advanced": False, "references": references,
        "fixture_only": True, "TR_synthetic_fixture_only_not_actual_cleaner_pool_validation": True,
        "original_replay_cache_files_unchanged": True})


def main(output):
    import numpy as np
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1"
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        assert os.environ[key] == "1"
    guard(); out = scoped(output); out.mkdir(exist_ok=False)
    # No append/mutation of a retained admitted window, even for this tiny test.
    assert not CACHE.exists() or not list(CACHE.iterdir()), "Prior owner must retire its exact window first"
    CACHE.mkdir(exist_ok=True); folder = CACHE / "clean_feature_contract_fixture_01"; folder.mkdir(exist_ok=False)
    checks = {}
    def check(name, value):
        assert value, name
        checks[name] = True
    def rejects(name, function):
        try: function()
        except (ValueError, KeyError, TypeError, AssertionError, IndexError): checks[name] = True
        else: raise AssertionError(name)
    original = read_bound({"path": str(ROOT / "analysis/FastMSS_integration_v1/clean_feature_fixture_original_bindings_v1.json"),
        "bytes": ORIGINAL_BINDINGS_BYTES, "sha256": ORIGINAL_BINDINGS_SHA}, guard)
    old_owner = original["owner_contents"]
    old_replay = read_bound({"path": str(ROOT / "analysis/ASR18_memory_diagnosis_v1/window_replay_PCM_actual_01/report_plan_v2.json"),
        "bytes": 175450, "sha256": "085fe85e85b3787e9d86650881c3e8a9c21094fdb49de1bb974e41592fd2e126"}, guard)
    choices = [next(v for v in old_replay["references"] if v["language"] == language) for language in ("de", "pt")]
    wave = (np.sin(np.arange(16000, dtype=np.float32) * np.float32(.013)) * np.float32(.2)).astype(np.float32)
    pcm = out / "synthetic_TR_fixture.npy"
    with pcm.open("xb") as stream: np.save(stream, wave, allow_pickle=False)
    ids = [23 + v % 500 for v in range(418)] + [8710]
    rows, contexts, reference_sources = [], [], []
    for index in range(2):
        context = AugmentationDrawContext("fixture_clean_TR_duplicate", "tr", 2, index, PHASE)
        rows.append({"source_id": context.source_id, "language": "tr", "text": "Sentetik sözleşme denemesi",
            "duration": 1.0, "encoder_frames": 13, "target_tokens": len(ids), "tr_metadata_labels": ids,
            "cleaner_original_ordinal": 1, "augmentation_role": "student_tr", "audio_filepath": str(pcm),
            "audio_draw_context": asdict(context), "fixture_only": True})
        contexts.append(asdict(context)); reference_sources.append({})
    for source in choices:
        row = deepcopy(source["source_row"]); context = AugmentationDrawContext(row["source_id"], row["language"], 2, len(rows), PHASE)
        row.update(augmentation_role="replay", clean_target_ids=source["full_target_ids"], audio_draw_context=asdict(context))
        rows.append(row); contexts.append(asdict(context)); reference_sources.append(source)
    catalog = {"kind": "ASR18_cleaner352768_BPE_catalog_binding_v1", "rows": 352768, "full_target_tokens": 11972886,
        "population_mean_U": 11972886 / 352768, "quality_or_language_selection_added": False,
        "augmentation_phase": PHASE, "fixture_metadata_denominator_only_not_full_catalog_validation": True}
    reference_request = out / "reference_request.json"
    write_new(reference_request, {"rows": rows, "contexts": contexts, "reference_sources": reference_sources, "cleaner_catalog_identity": catalog})
    reference_report = out / "reference_report.json"; reference_guard = out / "reference_guard"
    child = subprocess.Popen([sys.executable, "-B", str(PROJECT / "work/research/ASR18_bounded_CPU_phase_v3.py"),
        "--folder", str(reference_guard), "--seconds", "90", "--script", __file__, "--",
        "--references", str(reference_request), "--output", str(reference_report)])
    code = child.wait(timeout=100); assert code == 0
    closed = json.loads((reference_guard / "closure.json").read_bytes())
    assert closed["actual_exit_code"] == 0 and closed["monitor_clean"] and closed["monitor_joined"]
    reference_closure = out / "actual_reference_closure.json"
    write_new(reference_closure, {"kind": "FastMSS_independent_clean_audio_reference_actual_closure_v1",
        "report": record(reference_report, guard), "actual_exit_code": 0, "parent_observed_wait": True,
        "native_pid": closed["pid"], "guard_closure": record(reference_guard / "closure.json", guard)})
    names = ["owner.json"] + [f"{case}_r{i}_{key}.npy" for case in ("good", "wrongbytes") for i in range(4) for key in ("input_features", "attention_mask")]
    owner = {"kind": "FastMSS_owned_clean_provisional_feature_window_v1", "phase": PHASE, "source": file_binding(__file__),
        "producer_contract": file_binding(SOURCE / "fastmss_clean_feature_contract_v1.py"), "capacity_bytes": CAPACITY,
        "updates": 1, "owned_names": names, "prepared_is_committed": False, "sampler_journal_frontier_advanced": False}
    write_new(folder / "owner.json", owner)
    request = {"kind": KIND, "phase": PHASE, "update_index": 0, "batch_id": "clean_contract_fixture",
        "rows": rows, "contexts": contexts, "ordered_draws": [{**c, **({"ordinal": 1} if c["language"] == "tr" else {})} for c in contexts],
        "independent_clean_audio_proof": record(reference_report, guard), "independent_clean_audio_actual_closure": record(reference_closure, guard),
        "cleaner_catalog_identity": catalog, "augmentation_identity": old_owner["augmentation_identity"], "frontend": old_owner["frontend"],
        "native_transaction_sha256": "a" * 64, "before_sampler_sha256": "b" * 64, "after_sampler_sha256": "c" * 64,
        "prepared_is_committed": False, "sampler_journal_frontier_advanced": False,
        "feature_sink_source": file_binding(__file__), "feature_cache_owner": record(folder / "owner.json", guard)}
    def capacity(additional):
        guard(); return sum(p.stat().st_size for d in CACHE.iterdir() for p in d.iterdir()) + additional <= CAPACITY
    def frontend_stub(original):
        valid = len(original) // 160
        mask = np.arange(valid + 8)[None, :] < valid
        features = np.full((1, valid + 8, 128), np.float32(.125), dtype=np.float32); features[~mask] = 0
        return {"input_features": features, "attention_mask": mask}
    def make(*, actual=None, case="good", admit=capacity, frontend=frontend_stub, reader=load_clean):
        def save(index, key, array):
            path = folder / f"{case}_r{index}_{key}.npy"; written = array
            if case == "wrongbytes": written = array.copy(); written.flat[0] = 0
            with path.open("xb") as stream: np.save(stream, written, allow_pickle=False)
            return {"file": record(path, guard), "shape": list(array.shape), "dtype": str(array.dtype),
                "array_sha256": hashlib.sha256(array.tobytes()).hexdigest()}
        return CleanFeatureUpdateProducer(request if actual is None else actual, clean_reader=reader,
            frontend=frontend, save_array=save, capacity_guard=admit, guard=guard)
    import fastmss_train_augmentation_v1 as adapter
    original_constructor = adapter.TrainWaveformAugmenter.__init__
    def forbidden(*args, **kwargs): raise AssertionError("Cleaner reached augmenter constructor")
    adapter.TrainWaveformAugmenter.__init__ = forbidden
    state = {"actual_Adam": 17, "sampler": "unchanged", "journal": 841}; before = dict(state)
    producer = make(); rejects("incomplete_update_rejected", producer.prepared_update)
    for index in range(4): producer.prepare_next(index)
    payload = producer.prepared_update()
    check("full419_targets_and8710_ID_preserved", payload["rows"][0]["full_target_ids"] == ids)
    check("duplicate_source_distinct_draws_preserved", payload["rows"][0]["source_id"] == payload["rows"][1]["source_id"] and payload["rows"][0]["context"] != payload["rows"][1]["context"])
    check("all_clean_protection_no_transform", all(v["augmentation"]["applied"] is False for v in payload["prepared_payload"]["rows"]))
    check("full_clean_PCM_hashes_equal", all(v["clean_PCM_sha256"] == v["prepared_PCM_sha256"] for v in payload["rows"]))
    check("prepared_not_committed", state == before and not payload["sampler_journal_frontier_advanced"] and not payload["actual_CPU_child_exit_observed"])
    rejects("consume_once", lambda: producer.prepare_next(0))
    wrong = make(); rejects("out_of_order_poison", lambda: wrong.prepare_next(1)); rejects("poison_blocks_retry", lambda: wrong.prepare_next(0))
    for field, value in (("phase", "weak_label_continued_training"), ("update_index", True), ("prepared_is_committed", True)):
        changed = deepcopy(request); changed[field] = value; rejects("reject_" + field, lambda r=changed: make(actual=r))
    changed = deepcopy(request); changed["unknown"] = True; rejects("unknown_key", lambda: make(actual=changed))
    rejects("capacity_denied_before_write", lambda: make(admit=lambda value: False).prepare_next(0))
    rejects("actual_saved_array_tamper", lambda: make(case="wrongbytes").prepare_next(0))
    rejects("PCM_tamper_before_frontend", lambda: make(reader=lambda row: wave * np.float32(.5)).prepare_next(0))
    adapter.TrainWaveformAugmenter.__init__ = original_constructor
    check("no_Torch_frontend_or_ASR_import", not any(v in sys.modules for v in ("torch", "transformers", "scipy", "pyarrow")))
    write_new(out / "validation.json", {"passed": True, "checks": checks, "source": file_binding(__file__),
        "contract": file_binding(SOURCE / "fastmss_clean_feature_contract_v1.py"), "fixture_only": True,
        "frontend_callback_is_synthetic_stub_NOT_original_frontend_equivalence": True,
        "TR_synthetic_fixture_only_NOT_actual_cleaner_pool_compatibility": True, "original_replay_source_and_cache_bound": True,
        "actual_optimizer_updates": 0, "authoritative_frontiers_advanced": False, "feature_owner": record(folder / "owner.json", guard),
        "actual_reference_closure": record(reference_closure, guard), "owned_feature_files_retained_for_parent_actual_closure_cleanup": True,
        "prepared_update": payload, "caller_state_before": before, "caller_state_after": state})
    print(json.dumps({"passed": True, "checks": len(checks), "no_Torch": True}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", required=True); parser.add_argument("--references")
    args = parser.parse_args()
    if args.references: references(args.references, args.output)
    else: main(args.output)
