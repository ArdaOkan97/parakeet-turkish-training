"""Tiny real sampler + extracted parser/caller/AST checks. No ML/audio imports."""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/parakeet_turkish"
sys.path.insert(0, str(OUTPUT))
from turkish_domain_sampler_v2 import DomainBalancedBucketSampler
from domain30_training_identity_v1 import validate_recipe

PARENT = OUTPUT / "train_lora_batched_data_v11.py"
WORKER = OUTPUT / "train_lora_batched_domain30_v1.py"
parent = ast.parse(PARENT.read_text(encoding="utf-8"))
worker = ast.parse(WORKER.read_text(encoding="utf-8"))
checks = []


def check(value, name):
    if not value:
        raise AssertionError(name)
    checks.append(name)


def definition(tree, name):
    return next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)


def reject(call, name):
    try:
        call()
    except ValueError:
        checks.append(name)
    else:
        raise AssertionError("Expected rejection: " + name)


def ids_sha(rows):
    return hashlib.sha256("\n".join(sorted(row["source_id"] for row in rows)).encode()).hexdigest()


env = {"hashlib": hashlib, "Path": Path, "math": math, "random": random, "ids_sha": ids_sha,
       "argparse": argparse, "DATA": ROOT / "work/data", "OUTPUT": OUTPUT,
       "DomainBalancedBucketSampler": DomainBalancedBucketSampler,
       "file_sha": lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest(),
       "validate_recipe": validate_recipe, "native_loss_identity": lambda args: {"fixture": True}}
nodes = [definition(worker, name) for name in ("BucketCycler", "parser", "validate_arguments")]
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(WORKER), "exec"), env)
BucketCycler = env["BucketCycler"]
check(ast.dump(definition(parent, "BucketCycler")) == ast.dump(definition(worker, "BucketCycler")),
      "Actual copied BucketCycler AST remains identical")


def row(name, domain, index):
    return {"source_id": name + str(index), "language": "tr" if domain in ("cv", "fleurs") else domain,
        "dataset": "google/fleurs" if domain == "fleurs" else "ysdede/commonvoice_17_tr_fixed" if domain == "cv" else "google/fleurs",
        "dataset_config": "tr_tr" if domain == "fleurs" else None,
        "dataset_revision": "70bb2e84b976b7e960aa89f1c648e09c59f894dd" if domain != "cv" else "9d8a025f1ae7c5a2f5c43298c7c4910b03995a4b",
        "dataset_split": "train", "training_role": "train", "text": "fixture",
        "encoder_frames": 12 + index % 31, "target_tokens": 3 + index % 9, "duration": 2. + index % 7,
        "audio_sha256": hashlib.sha256((name + str(index)).encode()).hexdigest()}


tr = [row("f", "fleurs", i) for i in range(7)] + [row("c", "cv", i) for i in range(43)]
rp = [row(lang, lang, i) for lang in ("en", "es", "it", "fr", "de") for i in range(13)]
args = env["parser"]().parse_args(["--run-id", "inactive_domain_fixture", "--resume-weights", "fixture.pt",
    "--source-closure", "fixture.json", "--replay-seed", "20261012"])
env["validate_arguments"](args)
check((args.batch_size, args.accumulation, args.head_lr, args.replay_kl, args.new_token_lr_multiplier,
       args.tdt_implementation) == (8, 2, 5e-5, 10., 6., "nemo"), "Actual parser retains requested scientific defaults")
reject(lambda: validate_recipe(argparse.Namespace(**{**vars(args), "head_lr": 9e-5})), "Changed head LR rejected")
reject(lambda: validate_recipe(argparse.Namespace(**{**vars(args), "resume_weights": None})), "Unqualified fresh base forbidden")
reject(lambda: validate_recipe(argparse.Namespace(**{**vars(args), "resume_runtime_migration": Path("legacy.json")})),
       "Legacy uniform state migration into domain scientific arm rejected")

# Evaluate the constructor from the ACTUAL worker run AST with only tiny rows.
run = definition(worker, "run")
sampler_statement = next(n for n in run.body if isinstance(n, ast.Assign)
                         and any(isinstance(t, ast.Name) and t.id == "sampler" for t in n.targets))
env.update(args=args, tr=tr, replay=rp)
exec(compile(ast.Module(body=[sampler_statement], type_ignores=[]), str(WORKER), "exec"), env)
domain = env["sampler"]
check(isinstance(domain, DomainBalancedBucketSampler), "Actual worker constructor binds sampler API")


def make(mode, replay_seed=20261012):
    return DomainBalancedBucketSampler(tr, rp, seed=20261010, replay_seed=replay_seed,
        sampling_mode=mode, batch_size=8, accumulation=2, replay_fraction=.5, fleurs_fraction=.3,
        bucket_size=128, cycler_factory=BucketCycler,
        cycler_source_sha256=env["file_sha"](PARENT))


uniform = make("uniform")
nf = []
for update in range(1, 101):
    d = domain.next_batches()
    u = uniform.next_batches()
    check(len(d) == 2 and all(len(group) == 8 for group in d)
          and all(r["language"] == "tr" for r in d[0])
          and all(r["language"] != "tr" for r in d[1]), "B8x2 homogeneous packing update " + str(update))
    check([r["source_id"] for r in d[1]] == [r["source_id"] for r in u[1]],
          "Replay identity across changed TR sampling update " + str(update))
    check(domain.domain_draws["fleurs"] == update * 8 * 3 // 10,
          "Exact rational domain quota update " + str(update))
    if update <= 5:
        nf.append(sum(r["dataset"] == "google/fleurs" for r in d[0]))
check(nf == [2, 2, 3, 2, 3], "First five FLEURS TR quotas are2/2/3/2/3")
check(domain.coverage()["all_Turkish_pool_sources_seen"], "Both tiny domain pools actually covered")
check(sum(domain.replay_language_counts.values()) == 800 and domain.tr_draws == 800,
      "Complete clip exposure counters")

for mode in ("domain30", "uniform"):
    uninterrupted = make(mode)
    for _ in range(17):
        uninterrupted.next_batches()
    saved = copy.deepcopy(uninterrupted.state_dict())
    restored = make(mode)
    restored.load_state_dict(saved)
    check(saved == restored.state_dict(), "Checkpoint state roundtrip exact: " + mode)
    expected, actual = [], []
    for _ in range(40):
        expected.extend(r["source_id"] for g in uninterrupted.next_batches() for r in g)
        actual.extend(r["source_id"] for g in restored.next_batches() for r in g)
    check(expected == actual and uninterrupted.state_dict() == restored.state_dict(),
          "RNG/cursors/counters produce next640 identical IDs: " + mode)
    bad = copy.deepcopy(saved); bad["domain_draws"]["cv"] += 1
    reject(lambda: make(mode).load_state_dict(bad), "Corrupt domain counter rejected: " + mode)
    reject(lambda: make(mode, 999).load_state_dict(saved), "Replay seed resume mismatch rejected: " + mode)

small = make("domain30")
full_updates = small.updates_for_epochs(1.)
for _ in range(full_updates - 1):
    small.next_batches()
check(small.turkish_epoch < 1., "Domain epoch requires last partial CV exposure")
small.next_batches()
check(small.turkish_epoch >= 1. and small.coverage()["all_Turkish_pool_sources_seen"],
      "Configured domain full epoch really visits every tiny source")

# Exact math, loss, row updates, teacher/padding and guards are retained as source,
# not declared GPU equivalence. Compare key statements in their original scope.
parent_run = definition(parent, "run")
def statement(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
for name in ("teacher", "optimizer", "new_parameters"):
    check(ast.dump(statement(parent_run, name)) == ast.dump(statement(run, name)),
          "Unchanged original statement AST: " + name)
def update_body(tree):
    loop = next(n for n in tree.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "step")
    start = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.For))
    end = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "record" for t in n.targets))
    return [ast.dump(n) for n in loop.body[start:end]]
check(update_body(parent_run) == update_body(run), "Forward/backward/KD/clip/Adam/six-row/native cleanup AST unchanged")
check(ast.dump(definition(parent, "development_gate")) == ast.dump(definition(worker, "development_gate")),
      "All-five BEST gate function AST unchanged")
check("sampler.tr." not in WORKER.read_text(encoding="utf-8"), "No incompatible aggregate .tr view remains")
check(env["file_sha"](PARENT) == "4b4c9e3008205c61bb5a881972187b5b132881dc7a1d9249583d1adcd52a7d20",
      "Frozen active trainer11 hash unchanged")
report = {"kind": "domain30_worker_tiny_sampler_integration_v1", "passed": True,
    "checks": len(checks), "check_names": checks,
    "worker_sha256": env["file_sha"](WORKER),
    "sampler_sha256": env["file_sha"](OUTPUT / "turkish_domain_sampler_v2.py"),
    "identity_helper_sha256": env["file_sha"](OUTPUT / "domain30_training_identity_v1.py"),
    "tiny_pool": {"TR": len(tr), "fleurs": 7, "cv": 43, "replay": len(rp)},
    "matched_replay_updates": 100, "first_five_fleurs_counts": nf,
    "resume_modes_checked": ["domain30", "uniform"], "next_clips_per_mode_compared": 640,
    "fixture_domain_epoch_updates": full_updates,
    "Torch_NeMo_models_audio_GPU_imported": False, "actual_model_training_or_quality_proven": False,
    "native_outer_and_actual_IO_dispatch_pending": True,
    "legacy_uniform_replay_sequence_equivalence_claim": False}
destination = OUTPUT / "domain30_worker_tiny_sampler_integration_v1.json"
destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"path": str(destination), "passed": True, "checks": len(checks),
                  "worker_sha256": report["worker_sha256"]}))
