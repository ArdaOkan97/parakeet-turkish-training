"""Repeatable stdlib fixture: actual NEW wrappers with mocked ML/telemetry only.

No model, checkpoint, audio, Torch, NeMo or GPU is imported or executed. Actual
worker parser/validation AST, dispatcher, atomic serializer adapter and caller
frame binding execute; external dataset/native identities are explicit stubs.
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
OUTPUT = ROOT / "outputs/parakeet_turkish"
sys.path.insert(0, str(OUTPUT))
import launch_native_training_bpe512_v2 as outer
import launch_with_memory_guard_bpe512_v2 as memory
import launch_with_cache_reclaim_bpe512_v2 as cache
import launch_with_io_retry_bpe512_v2 as io
import cuda_cache_caller_bpe512_v2 as caller
import launch_with_memory_guard_BPE512_resume_v2 as exact_memory

NEW_NAMES = ["cuda_cache_caller_bpe512_v2.py", "launch_with_io_retry_bpe512_v2.py",
    "launch_with_cache_reclaim_bpe512_v2.py", "launch_with_memory_guard_bpe512_v2.py",
    "launch_native_training_bpe512_v2.py", "launch_with_memory_guard_BPE512_resume_v2.py"]
OLD_PINS = {
    "train_lora_batched_data_v11.py": "4b4c9e3008205c61bb5a881972187b5b132881dc7a1d9249583d1adcd52a7d20",
    "launch_with_io_retry_v9.py": "8d49db792cafa88341a64d893eeae4f09233edde36cdca3b53954f6da102cd3e",
    "launch_with_cache_reclaim_v12.py": "9febf980a1eb51dd35e5d7768afe73664774ca0445763f6cf1e2caccc5db802b",
    "launch_with_memory_guard_v15.py": "00213ae0a1af7ec5a24c9a2912bbab43a173c40a48448885cc971e7c4d77e3aa",
    "launch_native_training_v5.py": "562717de152de43976e9c7b3a72884dd0cadc93bd7910f749375259bd4fc723a",
    "native_tdt_checkpointed_v3.py": "839194dddf83c3b08bf608d01cb540b2f160feee426a91373d14cfc426ab7494",
    "cuda_cache_reclaim_v2.py": "869b9ffaac70c9b7b6e7590ba1d330363100d6b2ddb9f8f7af9b2b9865c1c7f9",
    "download_issai_tsc_v1.py": "9370adeae0ca3789801a00c5a81b7d60fda2317a4a3ec6aaba754313f44d3295",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    target = (args.output or OUTPUT / ("HF_BPE512_runtime_CPU_dispatch_verification_" + uuid.uuid4().hex + ".json")).resolve()
    if not target.is_relative_to(ROOT) or target.exists():
        raise ValueError("A new workspace output path is required; existing receipts are immutable")
    target.parent.mkdir(parents=True, exist_ok=True)
    checks = []
    def check(name, value):
        if not value:
            raise AssertionError(name)
        checks.append(name)
    check("no_real_ML_imports_at_start", all(name not in sys.modules for name in ("torch", "nemo", "transformers")))
    original_before = {name: sha(OUTPUT / name) for name in OLD_PINS}
    check("original_pins_before", original_before == OLD_PINS)
    new_before = {name: sha(OUTPUT / name) for name in NEW_NAMES}
    scratch = ROOT / "work/research" / ("bpe512_dispatch_fixture_" + uuid.uuid4().hex)
    scratch.mkdir()
    worker_path = OUTPUT / "train_lora_batched_bpe512_v2.py"
    check("worker_pinned_and_caller_AST", sha(worker_path) == caller.WORKER_SHA and caller.caller_line(worker_path) == 618)
    check("cap_default_is8", memory.validate_capacity(16_000_000_000)["cap_bytes"] == 8_000_000_000)
    for cap in (7., 9.):
        try:
            memory.validate_capacity(16_000_000_000, cap)
        except ValueError:
            check("wrong_cap_rejected_" + str(cap), True)
        else:
            raise AssertionError("Unreviewed cap accepted")

    tree = ast.parse(worker_path.read_text(encoding="utf-8"))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in ("parser", "validate_arguments")]
    check("actual_parser_and_validator_selected", {node.name for node in selected} == {"parser", "validate_arguments"})
    calls = {"closed_source": 0, "native_identity": 0, "serializer": 0, "preload": 0, "run": 0}
    worker = types.ModuleType(worker_path.stem)
    worker.__file__ = str(worker_path)
    worker.__dict__.update(argparse=argparse, math=math, Path=Path, ROOT=ROOT, OUTPUT=OUTPUT, DATA=ROOT / "work/data")
    def closed_identity(args):
        calls["closed_source"] += 1
        return {"fixture_only": True}
    def native_identity(args):
        calls["native_identity"] += 1
        ready = json.loads(args.native_backend_provenance.read_text(encoding="utf-8"))
        check("fresh_ready_actual_nested_route", ready["kind"] == "native_tdt_training_backend_v3"
            and ready["trainer_sha256"] == caller.WORKER_SHA and ready["run_id"] == args.run_id)
        return {"fixture_only": True}
    worker.__dict__.update(closed_source_identity=closed_identity, native_loss_identity=native_identity)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(worker_path), "exec"), worker.__dict__)
    def serializer(path, value):
        calls["serializer"] += 1
        Path(path).write_bytes(value)
    worker.save_training_state = serializer
    worker.write_json = None
    captured = []

    class FakeCUDA:
        def __init__(self):
            self.reserved = 5_000_000_000
            self.caps = []; self.synchronizations = 0; self.cleanups = 0
        def get_device_properties(self, device):
            return types.SimpleNamespace(total_memory=16_000_000_000, name="MOCKED_16GB_NOT_GPU")
        def set_per_process_memory_fraction(self, fraction, device):
            self.caps.append((fraction, device))
        def is_initialized(self): return False
        def synchronize(self, *args, **kwargs):
            self.synchronizations += 1
            return "original_return"
        def memory_allocated(self): return 2_000_000_000
        def memory_reserved(self): return self.reserved
        def max_memory_allocated(self): return 2_500_000_000
        def empty_cache(self):
            self.cleanups += 1; self.reserved = 3_000_000_000
    cuda = FakeCUDA()
    fake_torch = types.ModuleType("torch"); fake_torch.cuda = cuda
    benchmark = types.ModuleType("benchmark"); benchmark.write_json = None
    def after_run(args, sync_return):
        calls["run"] += 1; captured.append(args)
        check("original_synchronize_return_preserved", sync_return == "original_return")
        worker.save_training_state(scratch / "tiny_serialized_state.bin", b"original-serializer-exact-bytes")
        worker.write_json(scratch / "worker_observation.json", {"run_id": args.run_id, "mocked": True})
    worker.__dict__.update(torch=fake_torch, after_run=after_run)
    # The inherited hook observes an actual Python frame with the reviewed file,
    # function and line. The function invokes only fake CUDA, never worker math.
    source_like = "def run(args):\n" + "\n" * 616 + "    result = torch.cuda.synchronize()\n    return after_run(args, result)\n"
    exec(compile(source_like, str(worker_path), "exec"), worker.__dict__)

    def sample(pid, phase):
        return {"phase": phase, "observed_monotonic": 1., "observed_at_utc": datetime.now(timezone.utc).isoformat(),
            "system_memory": {"available": False, "fixture_only": True},
            "worker_private_memory": {"available": False, "pid": pid, "fixture_only": True}}
    original_live = memory.LiveTelemetrySampler
    class QuietSampler(original_live):
        def __init__(self, pid, sink): super().__init__(pid, sink, sample=sample)
        def start(self): self.sample_once()
    original_update = memory.UpdateTelemetry
    old_attrs = {name: getattr(memory, name) for name in ("safe_sample", "creation_identity", "LiveTelemetrySampler", "UpdateTelemetry")}
    memory.safe_sample = sample
    memory.creation_identity = lambda pid: {"pid": pid, "available": False, "fixture_only": True}
    memory.LiveTelemetrySampler = QuietSampler
    memory.UpdateTelemetry = functools.partial(original_update, sample=sample)
    preload_module = types.ModuleType("preload_nemo_linker_v1")
    def preload():
        calls["preload"] += 1
        check("preload_before_Torch_even_stub", "torch" not in sys.modules)
        return {"kind": "coherent_native_cuda12_9_preload_v1", "actual_binding_linker": {"version": [12, 9]},
            "cuda_initialized": False, "fixture_only": True}
    preload_module.preload = preload
    module_names = ("torch", "benchmark", worker_path.stem, "preload_nemo_linker_v1")
    old_modules = {name: sys.modules.get(name) for name in module_names}
    old_runpy = outer.runpy.run_path
    def nested_dispatch(path, run_name):
        check("outer_calls_actual_new_memory_route", Path(path) == OUTPUT / "launch_with_memory_guard_bpe512_v2.py" and run_name == "__main__")
        sys.modules.update(torch=fake_torch, benchmark=benchmark)
        sys.modules[worker_path.stem] = worker
        memory.main(memory.parser().parse_args(sys.argv[1:]))
    outer.runpy.run_path = nested_dispatch
    sys.modules["preload_nemo_linker_v1"] = preload_module
    native = scratch / "native_ready.json"
    flags = ["--run-id", "bpe512_mocked_dispatch_fixture", "--resume-weights", str(scratch / "converted.pt"),
        "--bpe-source-checkpoint", str(scratch / "source.pt"), "--bpe-source-selection", str(scratch / "selection.json"),
        "--bpe-source-closure", str(scratch / "closure.json"), "--bpe-conversion-report", str(scratch / "conversion.json"),
        "--tdt-implementation", "nemo", "--native-backend-provenance", str(native),
        "--batch-size", "8", "--accumulation", "2", "--replay-kl", "10", "--head-lr", "5e-5",
        "--lora-lr", "3e-4", "--sampling-mode", "domain30", "--fleurs-fraction", ".3", "--replay-seed", "20261012", "--eval-every-updates", "250", "--max-wall-seconds", "1"]
    for name in ('converted.pt','source.pt','selection.json','closure.json','conversion.json'):
        (scratch/name).write_bytes(b'opaque synthetic fixture; not checkpoint/model parsed')
    paths = {name: scratch / (name + ".json") for name in ("io", "cache", "memory", "identity")}
    memory_args = ["--provenance", str(paths["io"]), "--cache-provenance", str(paths["cache"]),
        "--memory-provenance", str(paths["memory"]), "--worker-identity", str(paths["identity"]),
        "--cap-gb", "8", "--threshold-gb", "4", "--", *flags]
    args = types.SimpleNamespace(native_provenance=native, memory_launcher=OUTPUT / "launch_with_memory_guard_bpe512_v2.py",
        memory_launcher_sha256=sha(OUTPUT / "launch_with_memory_guard_bpe512_v2.py"), trainer_sha256=caller.WORKER_SHA,
        validate_only=False, memory_arguments=["--", *memory_args])
    try:
        outer.main(args)
        check("all_dispatch_layers_completed", calls == {"closed_source": 1, "native_identity": 1, "serializer": 1, "preload": 1, "run": 1})
        parsed = captured[0]
        check("actual_parser_preserves_scientific_flags", parsed.batch_size == 8 and parsed.accumulation == 2
            and parsed.replay_kl == 10 and parsed.head_lr == 5e-5 and parsed.eval_every_updates == 250
            and parsed.BPE_delta_multiplier == 1 and parsed.new_token_lr_multiplier == 6
            and parsed.sampling_mode == "domain30" and parsed.fleurs_fraction == .3 and parsed.replay_seed == 20261012)
        check("original_serializer_once_exact_bytes", (scratch / "tiny_serialized_state.bin").read_bytes() == b"original-serializer-exact-bytes")
        check("cap_API8_only_mocked", cuda.caps == [(0.5, 0)])
        check("exact_target618_reclaims_once", cuda.cleanups == 1)
        check("io_cache_memory_complete", all(json.loads(paths[name].read_text())["status"] == "complete" for name in ("io", "cache", "memory")))
        cache_evidence = json.loads(paths["cache"].read_text())["cache_reclamation"]
        check("caller_evidence_exact_scope", cache_evidence["caller_line"] == 618 and cache_evidence["eligible_update_calls"] == 1
            and cache_evidence["threshold_reserved_bytes"] == 4_000_000_000 and cache_evidence["reclamations"] == 1)
        memory_evidence = json.loads(paths["memory"].read_text())
        check("telemetry_target_and_cap", memory_evidence["cap_bytes"] == 8_000_000_000
            and memory_evidence["owned_update_memory_telemetry"]["eligible_update_calls"] == 1)
        check("outer_completion_binds_new_ready", json.loads(native.with_suffix(".completion.json").read_text())["sources_unchanged"] is True)
        before = cuda.cleanups
        cuda.reserved = 5_000_000_000
        reclaim = caller.CudaCacheReclaimer(cuda, worker_path)
        reclaim.install()
        try:
            cuda.synchronize()  # Fixture filename/function differ; must pass through.
            check("other_synchronize_ignored", reclaim.ignored_calls == 1 and reclaim.eligible_calls == 0 and cuda.cleanups == before)
        finally:
            reclaim.restore()
        check("original_synchronize_restored", cuda.synchronize() == "original_return")
        forbidden = worker.parser().parse_args(flags)
        forbidden.new_token_lr_multiplier = 7
        try: worker.validate_arguments(forbidden)
        except ValueError: check("actual_validator_rejects_wrong_row_gain", True)
        else: raise AssertionError("Wrong row gain accepted")
    finally:
        outer.runpy.run_path = old_runpy
        for name, value in old_attrs.items(): setattr(memory, name, value)
        for name, value in old_modules.items():
            if value is None: sys.modules.pop(name, None)
            else: sys.modules[name] = value
    exact_flags=list(flags);exact_flags[exact_flags.index('--resume-weights')]='--resume-exact'
    resume_records=exact_memory.bpe_argument_records(exact_flags)
    check('exact_only_admission_binds_original_ready',resume_records['native_backend_provenance']['sha256']==sha(native))
    check('exact_only_admission_binds_closed_snapshot',resume_records['resume_exact']['sha256']==sha(scratch/'converted.pt'))
    for forbidden_flags,description in ((flags,'warmweights'),(exact_flags+['--resume-runtime-migration',str(scratch/'closure.json')],'migration')):
        try:exact_memory.bpe_argument_records(forbidden_flags)
        except ValueError:check('exact_route_rejects_'+description,True)
        else:raise AssertionError('invalid exact admission')
    try:memory.bpe_argument_records(exact_flags)
    except ValueError:check('initial_memory_rejects_exact_resume',True)
    else:raise AssertionError('initial accepted exact')
    check('exact_scope_cap_matches8',exact_memory.validate_capacity(16_000_000_000)['cap_bytes']==8_000_000_000)
    check("no_real_ML_imported_or_retained", all(name not in sys.modules for name in ("torch", "nemo", "transformers")))
    check("original_pins_after_unchanged", {name: sha(OUTPUT / name) for name in OLD_PINS} == original_before)
    check("new_sources_unchanged_during_fixture", {name: sha(OUTPUT / name) for name in NEW_NAMES} == new_before)
    report = {"schema": 1, "kind": "isolated_HF_BPE512_domain_runtime_mocked_CPU_dispatch_verification_v2", "passed": True,
        "checks": checks, "check_count": len(checks), "source_sha256": new_before, "original_frozen_sources_sha256": original_before,
        "worker_sha256": caller.WORKER_SHA, "fixture_script_sha256": sha(__file__), "fixture_directory": str(scratch),
        "scope": "Actual new wrapper routing, worker parser/validation AST, original atomic serializer adapter and exact caller frame; external identities and all ML/CUDA/Win32 sampling are explicit mocks",
        "actual_model_or_GPU_execution": False, "actual_closed_source_validation": False,
        "actual_six_language_source_candidate_equivalence": "pending root-only future activation",
        "existing_training_or_download_changed": False, "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    with target.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": True, "check_count": len(checks), "report": str(target), "source_sha256": new_before}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="NEW workspace receipt path; default is a unique UUID filename")
    main(parser.parse_args())
