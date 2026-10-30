"""Actual new routing/AST/caller test with explicitly synthetic ML/approval data.

All CUDA/Torch/native-preload/telemetry operations are mocks. The actual frozen
TSC parser, argument validator, closed-source recipe check, atomic serializer
adapter and exact caller frame execute. No real corpus/model/audio is loaded.
"""
from __future__ import annotations
import argparse
import ast
from datetime import datetime, timezone
import functools
import hashlib
import json
import math
from pathlib import Path
import sys
import types
import uuid

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/parakeet_turkish"
sys.path.insert(0, str(OUT))
import launch_native_training_tsc_v1 as outer
import launch_with_memory_guard_tsc_v1 as memory
import launch_with_cache_reclaim_tsc_v1 as cache
import launch_with_io_retry_tsc_v1 as io
import cuda_cache_caller_tsc_v1 as caller
import tsc_training_inputs_v1 as inputs

NEW_NAMES = ("cuda_cache_caller_tsc_v1.py", "launch_with_io_retry_tsc_v1.py",
             "launch_with_cache_reclaim_tsc_v1.py", "launch_with_memory_guard_tsc_v1.py",
             "launch_native_training_tsc_v1.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle)


def main(args):
    target = (args.output or OUT / ("TSC_runtime_CPU_dispatch_verification_"+uuid.uuid4().hex+".json")).resolve()
    if not target.is_relative_to(ROOT) or target.exists():
        raise ValueError("NEW workspace receipt required; never overwrite prior proof")
    checks = []
    def check(name, value):
        if not value:
            raise AssertionError(name)
        checks.append(name)
    check("no_real_ML_modules_at_start", all(name not in sys.modules for name in ("torch", "nemo", "transformers")))
    frozen = json.loads((OUT / "domain30_runtime_chain_pins_v1.json").read_text())["source_sha256"]
    check("all_original21_sources_frozen_before", all(sha(OUT / name) == digest for name, digest in frozen.items()))
    new_before = {name: sha(OUT / name) for name in NEW_NAMES}
    dependencies = {name: sha(OUT / name) for name in ("train_lora_batched_tsc_v1.py", "tsc_training_inputs_v1.py", "tsc_supplemental_data_policy_v1.py")}
    scratch = ROOT / "work/research" / ("tsc_dispatch_fixture_"+uuid.uuid4().hex)
    scratch.mkdir()
    worker_path = OUT / "train_lora_batched_tsc_v1.py"
    line = caller.caller_line(worker_path)
    check("actual_worker_SHA_and_sole_caller_AST", sha(worker_path) == caller.WORKER_SHA and line == 545)
    check("actual_capacity_default8", memory.validate_capacity(16_000_000_000)["cap_bytes"] == 8_000_000_000)
    for cap in (7., 9.):
        try:
            memory.validate_capacity(16_000_000_000, cap)
        except ValueError:
            check("unapproved_cap_rejected_"+str(cap), True)
        else:
            raise AssertionError("Changed allocator policy accepted")
    tree = ast.parse(worker_path.read_text(encoding="utf-8"))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ("parser", "validate_arguments")]
    check("actual_parser_and_validator_AST", len(selected) == 2)
    old_tree = ast.parse((OUT / "train_lora_batched_data_v11.py").read_text(encoding="utf-8"))
    old_nodes = {node.name: node for node in old_tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
    nodes = {node.name: node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
    same = ("BucketCycler", "BalancedBucketSampler", "lr_factor", "next_evaluation_epoch", "development_gate",
            "validate_trainable_checkpoint", "annotate_cache_lengths", "save_training_state", "native_loss_function")
    check("nine_original_scientific_functions_AST_equal", all(ast.dump(nodes[name], include_attributes=False) == ast.dump(old_nodes[name], include_attributes=False) for name in same))
    calls = {"closed_source": 0, "native_identity": 0, "serializer": 0, "preload": 0, "run": 0}
    worker = types.ModuleType(worker_path.stem)
    worker.__file__ = str(worker_path)
    worker.__dict__.update(argparse=argparse, math=math, Path=Path, ROOT=ROOT, OUTPUT=OUT, DATA=ROOT / "work/data")
    old_here = inputs.HERE
    inputs.HERE = scratch  # Synthetic external source/baseline lives only in fixture scope.
    def closed_identity(options):
        calls["closed_source"] += 1
        return inputs.closed_source_identity(options)
    def native_identity(options):
        calls["native_identity"] += 1
        ready = json.loads(options.native_backend_provenance.read_text())
        check("actual_nested_fresh_native_ready", ready["kind"] == "native_tdt_training_backend_v3"
              and ready["trainer_sha256"] == caller.WORKER_SHA and ready["run_id"] == options.run_id)
        return {"fixture_only": True}
    worker.__dict__.update(closed_source_identity=closed_identity, native_loss_identity=native_identity)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(worker_path), "exec"), worker.__dict__)
    def serializer(path, value):
        calls["serializer"] += 1
        Path(path).write_bytes(value)
    worker.save_training_state = serializer
    worker.write_json = None
    native = scratch / "native_ready.json"
    asset = {name: scratch / (name+".json") for name in ("tsc_proof", "training_approval", "closure", "TR", "replay", "metadata", "data_proof")}
    checkpoint = scratch / "source_best_fixture.bin"
    checkpoint.write_bytes(b"explicit-synthetic-NOT-real-model")
    save(asset["tsc_proof"], {"kind": "TSC_supplemental_TR_frame_target_inputs_v1", "clips": 74884,
                              "original_TR_source_ids_sha256": "fixture_original_pool"})
    for name in ("training_approval", "TR", "replay", "metadata", "data_proof"):
        save(asset[name], {"synthetic_fixture_only": True})
    flags = ["--run-id", "tsc_mocked_dispatch_fixture", "--resume-weights", str(checkpoint),
             "--source-closure", str(asset["closure"]), "--tsc-input-proof", str(asset["tsc_proof"]),
             "--tsc-training-approval", str(asset["training_approval"]), "--tr-manifest", str(asset["TR"]),
             "--replay-manifest", str(asset["replay"]), "--tr-frame-metadata", str(asset["metadata"]),
             "--tr-frame-metadata-proof", str(asset["tsc_proof"]), "--data-v3-proof", str(asset["data_proof"]),
             "--epochs", "1", "--steps", "9361", "--tdt-implementation", "nemo",
             "--native-backend-provenance", str(native), "--batch-size", "8", "--accumulation", "2",
             "--replay-kl", "10", "--head-lr", "5e-5", "--lora-lr", "3e-4", "--eval-every-updates", "250",
             "--max-wall-seconds", "1"]
    options = worker.parser().parse_args(flags)
    science = {name: getattr(options, name) for name in ("epochs", "steps", "batch_size", "accumulation", "rank", "alpha",
               "last_n_layers", "replay_fraction", "replay_kl", "head_lr", "lora_lr", "new_token_lr_multiplier", "tdt_implementation")}
    science.update(epochs=3., steps=28083)
    source_run = "synthetic_closed_source"
    source_config = scratch / "source_config.json"
    source_selection, source_history = scratch / "source_selection.json", scratch / "source_history.json"
    source_summary = scratch / "results" / (source_run+"_step_1") / "summary.json"
    save(source_config, {"run_id": source_run, "turkish_clips_used": 74883, "replay_clips_used": 1595,
         "config_identity": {"training_arguments": science, "extended_tokenizer_sha256": inputs.policy.TOKENIZER_SEMANTIC_SHA,
         "tr_pool_sha256": "fixture_original_pool", "replay_pool_sha256": inputs.policy.REPLAY_SOURCE_SET_SHA,
         "native_loss": {"max_joint_bytes": 1700000000}}})
    save(source_selection, {"run_id": source_run, "accepted": True, "heldout_tests_evaluated": False,
         "selection_split": "FLEURS validation only", "selected_step": 1, "completed_steps": 1,
         "turkish_validation_wer": .5, "weights": str(checkpoint)})
    save(source_history, [{"step": 1, "validation": {"retention_gate_passed": True, "turkish_wer": .5}}])
    summary = {"validation_"+language: {"asr": {"wer": .5 if language == "tr" else .03, "utterances": 50}}
               for language in inputs.policy.CONFIGS}
    save(source_summary, summary)
    save(scratch / "results/base_v3/summary.json", summary)
    proof = {"kind": "root_owned_TSC_closed_source_v1", "passed": True, "root_observed_owned_session_completed": True,
             "child_exit_code": 0, "source_run_id": source_run}
    for name, path in (("source_checkpoint", checkpoint), ("source_selection", source_selection),
                       ("source_run_config", source_config), ("source_training_history", source_history),
                       ("source_development_summary", source_summary)):
        proof[name] = {"path": str(path.resolve()), "sha256": sha(path)}
    save(asset["closure"], proof)
    captured = []
    class FakeCUDA:
        def __init__(self): self.reserved = 5000000000; self.caps = []; self.cleanups = 0
        def get_device_properties(self, device): return types.SimpleNamespace(total_memory=16000000000, name="MOCKED_NO_GPU")
        def set_per_process_memory_fraction(self, fraction, device): self.caps.append((fraction, device))
        def synchronize(self, *args, **kwargs): return "original_return"
        def memory_allocated(self): return 2000000000
        def memory_reserved(self): return self.reserved
        def max_memory_allocated(self): return 2500000000
        def empty_cache(self): self.cleanups += 1; self.reserved = 3000000000
    cuda = FakeCUDA()
    fake_torch = types.ModuleType("torch"); fake_torch.cuda = cuda
    benchmark = types.ModuleType("benchmark"); benchmark.write_json = None
    def after_run(options, returned):
        calls["run"] += 1; captured.append(options)
        check("original_synchronize_result", returned == "original_return")
        worker.save_training_state(scratch / "tiny_state.bin", b"original-serializer-exact-bytes")
        worker.write_json(scratch / "worker_observation.json", {"run_id": options.run_id, "fixture_only": True})
    worker.__dict__.update(torch=fake_torch, after_run=after_run)
    exec(compile("def run(args):\n"+"\n"*(line-2)+"    result = torch.cuda.synchronize()\n    return after_run(args, result)\n", str(worker_path), "exec"), worker.__dict__)
    def sample(pid, phase):
        return {"phase": phase, "observed_monotonic": 1., "observed_at_utc": datetime.now(timezone.utc).isoformat(),
                "system_memory": {"available": False, "fixture_only": True},
                "worker_private_memory": {"available": False, "pid": pid, "fixture_only": True}}
    original_live = memory.LiveTelemetrySampler
    class QuietSampler(original_live):
        def __init__(self, pid, sink): super().__init__(pid, sink, sample=sample)
        def start(self): self.sample_once()
    old_attrs = {name: getattr(memory, name) for name in ("safe_sample", "creation_identity", "LiveTelemetrySampler", "UpdateTelemetry")}
    memory.safe_sample = sample
    memory.creation_identity = lambda pid: {"pid": pid, "available": False, "fixture_only": True}
    memory.LiveTelemetrySampler = QuietSampler
    memory.UpdateTelemetry = functools.partial(old_attrs["UpdateTelemetry"], sample=sample)
    mocked_preload = types.ModuleType("preload_nemo_linker_v1")
    def preload():
        calls["preload"] += 1
        check("preload_before_even_mock_Torch", "torch" not in sys.modules)
        return {"kind": "coherent_native_cuda12_9_preload_v1", "actual_binding_linker": {"version": [12, 9]},
                "cuda_initialized": False, "fixture_only": True}
    mocked_preload.preload = preload
    module_names = ("torch", "benchmark", worker_path.stem, "preload_nemo_linker_v1")
    old_modules = {name: sys.modules.get(name) for name in module_names}
    old_runpy = outer.runpy.run_path
    def nested(path, run_name):
        check("actual_TSC_outer_memory_path", Path(path) == OUT / "launch_with_memory_guard_tsc_v1.py" and run_name == "__main__")
        sys.modules.update(torch=fake_torch, benchmark=benchmark)
        sys.modules[worker_path.stem] = worker
        memory.main(memory.parser().parse_args(sys.argv[1:]))
    outer.runpy.run_path = nested
    sys.modules["preload_nemo_linker_v1"] = mocked_preload
    paths = {name: scratch / (name+".json") for name in ("io", "cache", "memory", "identity")}
    memory_args = ["--provenance", str(paths["io"]), "--cache-provenance", str(paths["cache"]),
                   "--memory-provenance", str(paths["memory"]), "--worker-identity", str(paths["identity"]),
                   "--cap-gb", "8", "--threshold-gb", "4", "--", *flags]
    outer_args = types.SimpleNamespace(native_provenance=native, memory_launcher=OUT / "launch_with_memory_guard_tsc_v1.py",
        memory_launcher_sha256=new_before["launch_with_memory_guard_tsc_v1.py"], trainer_sha256=caller.WORKER_SHA,
        validate_only=False, memory_arguments=["--", *memory_args])
    try:
        incomplete = list(flags); position = incomplete.index("--tsc-training-approval"); del incomplete[position:position+2]
        try: memory.tsc_argument_records(incomplete)
        except ValueError: check("missing_training_approval_rejects_old_default_fallback", True)
        else: raise AssertionError("Missing training approval accepted")
        outer.main(outer_args)
        check("actual_nested_all_layers_complete", calls == {"closed_source": 1, "native_identity": 1, "serializer": 1, "preload": 1, "run": 1})
        parsed = captured[0]
        check("actual_parser_B8x2_KD10_and_one_new_pool_horizon", parsed.batch_size == 8 and parsed.accumulation == 2
              and parsed.replay_kl == 10 and parsed.new_token_lr_multiplier == 6 and parsed.epochs == 1 and parsed.steps == 9361)
        check("original_atomic_serializer_bytes_once", (scratch / "tiny_state.bin").read_bytes() == b"original-serializer-exact-bytes")
        check("mock_cap8_API_once", cuda.caps == [(.5, 0)])
        check("actual_caller_frame545_reclaims_once", cuda.cleanups == 1)
        check("all_IO_cache_memory_complete", all(json.loads(paths[name].read_text())["status"] == "complete" for name in ("io", "cache", "memory")))
        evidence = json.loads(paths["cache"].read_text())["cache_reclamation"]
        check("actual_cache_scope545_end4_and_one_update", evidence["caller_line"] == line and evidence["eligible_update_calls"] == 1
              and evidence["threshold_reserved_bytes"] == 4000000000 and evidence["reclamations"] == 1)
        memory_evidence = json.loads(paths["memory"].read_text())
        identity = json.loads(paths["identity"].read_text())
        ready = json.loads(native.read_text())
        check("native_memory_identity_same_explicit_TSC_assets", ready["TSC_explicit_asset_records"] == identity["TSC_explicit_asset_records"] == memory_evidence["TSC_explicit_asset_records"])
        check("actual_TSC_policy_and_helper_identity_pins", identity["TSC_inputs_helper_sha256"] == dependencies["tsc_training_inputs_v1.py"]
              and identity["TSC_supplemental_policy_sha256"] == dependencies["tsc_supplemental_data_policy_v1.py"])
        check("actual_memory_callback_and_cap8", memory_evidence["cap_bytes"] == 8000000000
              and memory_evidence["owned_update_memory_telemetry"]["eligible_update_calls"] == 1)
        check("actual_outer_completion_source_check", json.loads(native.with_suffix(".completion.json").read_text())["sources_unchanged"] is True)
        reclaim = caller.CudaCacheReclaimer(cuda, worker_path)
        before = cuda.cleanups; cuda.reserved = 5000000000; reclaim.install()
        try:
            cuda.synchronize()
            check("other_frames_cannot_reclaim", reclaim.eligible_calls == 0 and reclaim.ignored_calls == 1 and cuda.cleanups == before)
        finally: reclaim.restore()
        changed = worker.parser().parse_args(flags); changed.new_token_lr_multiplier = 7
        try: inputs.closed_source_identity(changed)
        except ValueError: check("actual_source_checker_rejects_changed_row_gain", True)
        else: raise AssertionError("Changed science accepted")
    finally:
        inputs.HERE = old_here
        outer.runpy.run_path = old_runpy
        for name, value in old_attrs.items(): setattr(memory, name, value)
        for name, value in old_modules.items():
            if value is None: sys.modules.pop(name, None)
            else: sys.modules[name] = value
    check("no_real_ML_imported_or_retained", all(name not in sys.modules for name in ("torch", "nemo", "transformers")))
    check("all_original21_sources_frozen_after", all(sha(OUT / name) == digest for name, digest in frozen.items()))
    check("new_sources_and_dependencies_unchanged", all(sha(OUT / name) == digest for name, digest in {**new_before, **dependencies}.items()))
    result = {"schema": 1, "kind": "isolated_TSC_runtime_mocked_CPU_dispatch_verification_v1", "passed": True,
              "checks": checks, "check_count": len(checks), "source_sha256": {**new_before, **dependencies},
              "original_frozen_sources_sha256": frozen, "fixture_script_sha256": sha(__file__), "fixture_directory": str(scratch),
              "scope": "Actual new nested routes, parser/validator, closed-source recipe checker, original atomic serializer adapter and actual caller frame; synthetic approval/source/baseline and all ML/CUDA/telemetry are explicit mocks",
              "actual_TSC_import_or_data_validation": False, "actual_model_or_GPU_execution": False,
              "training_eligibility": "Pending actual ROOT approved complete corpus and CLOSED selected source; this fixture authorizes no training",
              "active_sources_modified": False, "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    with target.open("x", encoding="utf-8") as stream: json.dump(result, stream, indent=2)
    print(json.dumps({"passed": True, "check_count": len(checks), "report": str(target), "report_sha256": sha(target)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    main(parser.parse_args())
