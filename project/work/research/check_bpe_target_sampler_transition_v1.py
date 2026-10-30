"""Small actual stdlib sampler fixture; no real source state/model is loaded."""
import argparse
import ast
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "outputs/parakeet_turkish"
sys.path.insert(0, str(HERE))
import bpe_target_sampler_transition_v1 as transition
import tsc_domain_sampler_v3 as sampler_module
from turkish_domain_sampler_v1 import CV_REVISION, FLEURS_REVISION


def main(output):
    checks = []
    def check(ok, label):
        if not ok:
            raise AssertionError(label)
        checks.append(label)
    def reject(call, label):
        try:
            call()
        except (ValueError, KeyError):
            checks.append(label)
            return
        raise AssertionError("Accepted: " + label)
    worker = HERE / "train_lora_batched_tsc_headlr_v1.py"
    tree = ast.parse(worker.read_text(encoding="utf-8"))
    definition = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BucketCycler")
    scope = {"math": math, "ids_sha": lambda rows: hashlib.sha256(
        "\n".join(sorted(r["source_id"] for r in rows)).encode()).hexdigest()}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(worker), "exec"), scope)
    tr = []
    for group, count, dataset, revision in (
        ("fleurs", 80, "google/fleurs", FLEURS_REVISION),
        ("cv", 150, "ysdede/commonvoice_17_tr_fixed", CV_REVISION),
        ("tsc", 150, sampler_module.TSC_REPO, sampler_module.TSC_REVISION)):
        for i in range(count):
            tr.append({"source_id": f"toy_{group}_{i}", "dataset": dataset,
                       "dataset_revision": revision, "dataset_split": "train",
                       "dataset_config": "tr_tr" if group == "fleurs" else "default",
                       "training_role": "train", "language": "tr", "text": "toy target",
                       "duration": 1.0, "encoder_frames": 16 + i % 4,
                       "target_tokens": 9, "label_ids_sha256": "a" * 64})
    replay = [{"source_id": f"toy_{lang}_{i}", "language": lang, "duration": 1.0,
               "encoder_frames": 16, "target_tokens": 5, "text": "toy replay"}
              for lang in sampler_module.LANGUAGES for i in range(80)]
    def build(rows):
        return sampler_module.TSCDomainBalancedBucketSampler(
            rows, replay, seed=20261010, replay_seed=20261012, sampling_mode="domain30",
            batch_size=8, accumulation=2, replay_fraction=.5, fleurs_fraction=.3,
            bucket_size=128, cycler_factory=scope["BucketCycler"],
            cycler_source_sha256="4b4c9e3008205c61bb5a881972187b5b132881dc7a1d9249583d1adcd52a7d20")
    old = build(tr)
    for _ in range(7):
        old.next_batches()
    saved = old.state_dict()
    before = transition.canonical(saved)
    candidate_rows = deepcopy(tr)
    for row in candidate_rows:
        row.update(target_tokens=6, label_ids_sha256="b" * 64)
    candidate = build(candidate_rows)
    migrated, state_sha = transition._identity_only(saved, old.identity, candidate.identity)
    check(transition.canonical(saved) == before, "Input source state remains unchanged")
    candidate.load_state_dict(migrated)
    check(transition.canonical(candidate.state_dict()) == transition.canonical(migrated),
          "Actual sampler validates and restores complete migrated fixture")
    check(candidate.updates == old.updates == 7, "Original update frontier preserved")
    check(candidate.tr_rng.getstate() == old.tr_rng.getstate()
          and candidate.replay_rng.getstate() == old.replay_rng.getstate(), "Both RNG streams preserved")
    check(candidate.coverage() == old.coverage(), "All actual source/clip/audio coverage preserved")
    for i in range(8):
        a = old.next_batches()
        b = candidate.next_batches()
        check(sorted(r["source_id"] for batch in a for r in batch)
              == sorted(r["source_id"] for batch in b for r in batch),
              f"Same next update source multiset before refill {i+1}")
    check(candidate.replay_rng.getstate() == old.replay_rng.getstate(), "Replay future RNG still identical")
    reject(lambda: transition._identity_only(saved, old.identity, old.identity), "Unchanged target fingerprint rejected")
    for key, value in (("replay_seed", 1), ("batch_size", 1), ("replay_pool_fingerprint", "c"*64),
                       ("bucket_size", 64), ("fleurs_fraction_numerator", 4)):
        altered = deepcopy(candidate.identity); altered[key] = value
        reject(lambda v=altered: transition._identity_only(saved, old.identity, v), "Undeclared identity rejected: " + key)
    bad_state = deepcopy(saved); bad_state["identity"] = candidate.identity
    reject(lambda: transition._identity_only(bad_state, old.identity, candidate.identity), "Wrong source identity rejected")
    config_path = HERE / "runs" / transition.PARENT_RUN / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    report = transition.preparation_for_source(config)
    check(report["clips"] == 253798, "Actual current config binds full checked TRAIN preparation")
    wrong_config = deepcopy(config)
    wrong_config["config_identity"]["input_sha256"][str(Path(config["tr_manifest"]).resolve())] = "0"*64
    reject(lambda: transition.preparation_for_source(wrong_config), "Changed original manifest provenance rejected")
    check("torch" not in sys.modules, "No Torch import or GPU use")
    path = Path(output).resolve()
    if path.exists():
        raise ValueError("Fresh report required")
    path.write_text(json.dumps({"schema": 1, "kind": "BPE_target_sampler_stdlib_fixture_v1",
        "passed": True, "checks": len(checks), "check_names": checks,
        "module_sha256": hashlib.sha256(Path(transition.__file__).read_bytes()).hexdigest(),
        "actual_fixture_TR_clips": len(tr), "actual_fixture_replay_clips": len(replay),
        "actual_current_config_TRAIN_proof_binding": True, "actual_CLOSED_source_state_read": False,
        "actual_full_runtime_pool_or_ASR_restore_verified": False,
        "preserved_non_identity_fixture_state_sha256": state_sha,
        "limits": ["Only target-fingerprint migration mechanics and actual immutable TRAIN proof binding",
                   "Source checkpoint/CLOSED admission and full runtime consumer remain separate",
                   "Future bucket refills and physical pack order may differ with shorter targets"]},
        indent=2), encoding="utf-8")
    print(json.dumps({"passed": True, "checks": len(checks), "output": str(path)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
