"""Stdlib-only exact-worker AST/API fixture; every ML/state/PID object is fake."""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "outputs/parakeet_turkish"
WORKER = HERE / "train_lora_batched_tsc_bpe_lora_exact_v1.py"
ORIGINAL = HERE / "train_lora_batched_tsc_bpe_lora_native4_v2.py"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def compile_nodes(nodes, env):
    exec(compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes), type_ignores=[])),
                 str(WORKER), "exec"), env)


def main(args):
    tree = ast.parse(WORKER.read_text(encoding="utf-8"))
    original_tree = ast.parse(ORIGINAL.read_text(encoding="utf-8"))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    old = {node.name: node for node in original_tree.body if isinstance(node, ast.FunctionDef)}
    checks = []

    def check(ok, name):
        assert ok, name
        checks.append(name)

    def normalized(node):
        node = copy.deepcopy(node)
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
            node.body.pop(0)
        return ast.dump(node, include_attributes=False)

    for name in ("native_loss_function", "save_training_state", "_table_bytes_sha",
                 "_protected_snapshot", "_verify_smoke_phase_reload"):
        check(normalized(functions[name]) == normalized(old[name]), "unchanged " + name)
    run, old_run = functions["run"], old["run"]
    nested = next(n for n in run.body if isinstance(n, ast.FunctionDef) and n.name == "evaluate_current")
    old_nested = next(n for n in old_run.body if isinstance(n, ast.FunctionDef) and n.name == "evaluate_current")
    check(normalized(nested) == normalized(old_nested), "unchanged strict DEV/BEST/counter callback")
    loop = next(n for n in run.body if isinstance(n, ast.For) and ast.unparse(n.target) == "step")
    old_loop = next(n for n in old_run.body if isinstance(n, ast.For) and ast.unparse(n.target) == "step")
    check(ast.dump(ast.Module(body=loop.body, type_ignores=[]), include_attributes=False) ==
          ast.dump(ast.Module(body=old_loop.body, type_ignores=[]), include_attributes=False),
          "unchanged complete native195/twoAdam update and gates")
    check(ast.unparse(loop.iter) == "range(resume_step + 1, phase_end + 1)", "no repeated prefix or N reset")

    calls = []
    def initial_parser():
        p = argparse.ArgumentParser()
        p.add_argument("--run-id", required=True)
        p.add_argument("--smoke-updates", type=int)
        return p
    env = {"initial_worker": SimpleNamespace(parser=initial_parser,
             validate_arguments=lambda value: calls.append("original_arguments"),
             native_loss_identity=lambda value: calls.append("original_native") or {"immutable": True}),
           "__doc__": "fixture", "__file__": str(WORKER), "Path": Path, "require": require,
           "OUTPUT": HERE, "file_sha": sha,
           "exact_preflight": lambda value: calls.append("actual_exact_API_mock"),
           "validate_runtime_proof": lambda value, **kw: calls.append(("fresh_proof", kw["worker_path"]))}
    compile_nodes([functions[name] for name in ("native_loss_identity", "parser", "validate_arguments")], env)
    cli = ["--run-id", "fake", "--resume-phase2-exact", "latest",
           "--phase2-exact-source-receipt", "receipt", "--phase2-exact-runtime-proof", "fresh"]
    parsed = env["parser"]().parse_args(cli)
    check(env["validate_arguments"](parsed) is parsed and calls[0] == "original_arguments" and
          calls[1] == "actual_exact_API_mock" and calls[2] == "original_native" and
          calls[3] == ("fresh_proof", str(WORKER)), "real parser/validator and fresh current-worker proof API")
    parsed.smoke_updates = 1
    try:
        env["validate_arguments"](parsed)
    except ValueError:
        check(True, "exact rejects smoke/fresh screen")
    else:
        raise AssertionError("smoke accepted")

    # Execute the actual restore-to-runtime block, with all external ML and
    # source admission mocked. The disposable OUTPUT is never watched by Trackio.
    first = next(i for i, n in enumerate(run.body) if isinstance(n, ast.Delete) and
                 ast.unparse(n) == "del original, source, baseline_processor, runtime")
    last = next(i for i, n in enumerate(run.body) if isinstance(n, ast.FunctionDef) and n.name == "save")
    segment = run.body[first:last]
    with tempfile.TemporaryDirectory(prefix="tsc_exact_worker_", dir=ROOT / "work") as directory:
        temp = Path(directory)
        config_identity = {"public_turkish_data_policy": {"proof": "bound"},
            "extended_tokenizer_sha256": "tokens", "lora_layout": {"names": ["LoRA"]},
            "tr_pool_sha256": "TR", "replay_pool_sha256": "RP",
            "source_sha256": {}, "input_sha256": {}, "adaptation_optimizers": {"betas": (.9, .999)}}
        runtime = {"step": 8000, "history": [{"step": i} for i in range(1, 8001)],
            "best_score": .37, "selected_step": 7750, "patience": 2, "retention_failures": 1,
            "next_eval_epoch": .2, "initial_gate": {"turkish_wer": .44}, "train_wall_seconds": 50.,
            "evaluation_wall_seconds": 3., "audio_seconds": 500., "language_audio_seconds": {"tr": 200.}}
        model = object()
        adaptation = SimpleNamespace(inherited_runtime={"must_not_be_used": True}, phase2_updates=0,
            cumulative_step=7000, source_step=6500, acquisition_updates=500)
        sampler = SimpleNamespace(updates=0, tr_draws=0, tr_per_update=8, replay_per_update=8,
            replay={"en": SimpleNamespace(draws=0)})
        writes = []
        old_config = {"adaptation_phase_max_updates": 1000, "immutable_science": True}
        exact_context = {"resume_step": 8000, "phase2_updates_at_resume": 1000,
            "parent_identity": json.loads(json.dumps(config_identity)), "original_config": old_config,
            "approved_total_phase2_updates": 2000, "exact_receipt_record": {"sha256": "opaque"}}
        state = {"config_identity": config_identity}
        fixture_args = SimpleNamespace(resume_phase2_exact=temp / "archived_latest.pt", run_id="fake",
            native_backend_provenance=temp / "native.json", phase2_exact_runtime_proof=temp / "fresh.json")
        run_dir = temp / "run"
        run_dir.mkdir()
        base_env = {"original": object(), "source": object(), "baseline_processor": object(),
            "runtime": {"old": True}, "adaptation": adaptation, "sampler": sampler,
            "expanded": {"inherited_runtime": {"old": True}, "vocabulary": {"extended_tokenizer_sha256": "tokens"}},
            "args": fixture_args, "exact_context": exact_context, "model": model,
            "public_data_audit": {"identity": {"proof": "bound"}}, "tr": "TR", "replay": "RP",
            "encoder_lora_metadata": lambda value: {"names": ["LoRA"]}, "ids_sha": lambda value: value,
            "canonical": canonical, "require": require, "OUTPUT": temp, "Path": Path, "file_sha": sha,
            "start_step": 7000, "configured_steps": 44958, "run_dir": run_dir, "__file__": str(WORKER),
            "artifact_record": lambda path: {"path": str(path), "sha256": "explicit_mock", "bytes": 0},
            "proof_path": lambda value: temp / "original300.json",
            "write_json": lambda path, value: writes.append((Path(path), copy.deepcopy(value)))}
        def load(path, **kw):
            assert all(name not in execution_env for name in ("original", "source", "baseline_processor", "runtime"))
            assert adaptation.inherited_runtime is None and "inherited_runtime" not in execution_env["expanded"]
            return state
        def restore(ad, sa, value, context, **kw):
            assert value is state and context is exact_context
            ad.phase2_updates = 1000; ad.cumulative_step = 8000
            sa.updates = 8000; sa.tr_draws = 64000; sa.replay["en"].draws = 64000
            return runtime
        execution_env = dict(base_env, torch=SimpleNamespace(load=load), restore_runtime=restore)
        compile_nodes(segment, execution_env)
        check(execution_env["history"] is runtime["history"] and execution_env["best"] == .37 and
              execution_env["selected_step"] == 7750 and execution_env["retention_failures"] == 1 and
              execution_env["patience"] == 2 and execution_env["next_eval"] == .2,
              "returned own-phase runtime preserves history/BEST/counters/nextDEV")
        check(execution_env["config_identity"] is config_identity and execution_env["config"] is old_config and
              execution_env["phase_end"] == 9000 and execution_env["resume_step"] == 8000,
              "typed original identity and approved TOTAL N extension preserved")
        check(len(writes) == 3 and writes[1][0] == temp / "run_config.json" and
              all(path.name not in ("best.pt", "initial300.json") and path != run_dir / "run_config.json"
                  for path, _ in writes), "no initial300 or BEST/originalconfig reinitialization")
        # A wrong restored frontier must fail before emitting any fresh event.
        runtime["step"] = 7999
        execution_env = dict(base_env, original=object(), source=object(), baseline_processor=object(),
            runtime={}, expanded={"inherited_runtime": {}, "vocabulary": {"extended_tokenizer_sha256": "tokens"}},
            torch=SimpleNamespace(load=load), restore_runtime=restore)
        try:
            compile_nodes(segment, execution_env)
        except ValueError:
            check(len(writes) == 3, "wrong own-phase frontier rejects before publication")
        else:
            raise AssertionError("wrong frontier accepted")

    synchronize = [n.lineno for n in ast.walk(run) if isinstance(n, ast.Call) and
        isinstance(n.func, ast.Attribute) and ast.unparse(n.func) == "torch.cuda.synchronize"]
    check(len(synchronize) == 1, "sole reviewed optimizer-end caller target")
    report = {"schema": 1, "kind": "TSC_BPE_LoRA_exact_worker_stdlib_verification_v1", "passed": True,
        "worker_sha256": sha(WORKER), "original_worker_sha256": sha(ORIGINAL),
        "fixture_sha256": sha(__file__), "synchronize_line": synchronize[0],
        "checks": checks, "checks_passed": len(checks),
        "scope": "Actual parser/validate and restore-block AST executed; source admission, ML/state/tensors/PID/preload explicitly mocked. No real checkpoint/audio/corpus/ML/GPU/Trackio access. Actual closed restore/hardware unmeasured."}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps({"report": str(args.output), "sha256": sha(args.output),
        "worker_sha256": sha(WORKER), "checks_passed": len(checks), "synchronize_line": synchronize[0]}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
