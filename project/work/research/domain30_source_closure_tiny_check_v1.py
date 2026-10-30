"""Tiny synthetic closure guard checks; no real checkpoint/model/data reads."""
import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "outputs/parakeet_turkish"))
import domain30_training_identity_v1 as helper

checks = []


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def reject(call, name):
    try:
        call()
    except ValueError:
        checks.append(name)
    else:
        raise AssertionError("Expected rejection: " + name)


with tempfile.TemporaryDirectory(dir=ROOT / "work/research", prefix="domain_closure_fixture_") as directory:
    fixture = Path(directory)
    helper.ROOT = fixture
    helper.OUTPUT = fixture / "outputs/parakeet_turkish"
    source_run = "closed_fixture"
    paths = {name: fixture / filename for name, filename in (
        ("source_checkpoint", "work/closed_fixture_best.pt"),
        ("source_selection", "outputs/parakeet_turkish/runs/closed_fixture/selection.json"),
        ("source_run_config", "outputs/parakeet_turkish/runs/closed_fixture/run_config.json"),
        ("source_training_history", "outputs/parakeet_turkish/runs/closed_fixture/training_history.json"),
        ("source_development_summary", "outputs/parakeet_turkish/results/closed_fixture_step_1/summary.json"))}
    paths["source_checkpoint"].parent.mkdir(parents=True)
    paths["source_checkpoint"].write_bytes(b"synthetic byte identity only; not a real model")
    selection = {"run_id": source_run, "accepted": True, "selected_step": 1, "completed_steps": 2,
                 "weights": str(paths["source_checkpoint"]), "turkish_validation_wer": .3}
    config = {**helper.RECIPE, "run_id": source_run,
              "turkish_clips_used": 74883, "replay_clips_used": 1595,
              "config_identity": {"extended_tokenizer_sha256": helper.TOKENIZER_SHA,
                  "tr_pool_sha256": "tiny_source_pool_digest", "replay_pool_sha256": "tiny_replay_pool_digest"}}
    summary = {"validation_" + lang: {"asr": {"utterances": 50, "wer": .3 if lang == "tr" else .1}}
               for lang in (*helper.LANGUAGES, "tr")}
    baseline = copy.deepcopy(summary); baseline["validation_tr"]["asr"]["wer"] = .8
    write(paths["source_selection"], selection)
    write(paths["source_run_config"], config)
    write(paths["source_training_history"], [{"step": 1}, {"step": 2}])
    write(paths["source_development_summary"], summary)
    write(helper.OUTPUT / "results/base_v3/summary.json", baseline)
    closure = {"kind": "root_owned_domain30_closed_source_v1", "passed": True,
        "root_observed_owned_session_completed": True, "child_exit_code": 0, "source_run_id": source_run,
        **{name: {"path": str(path), "sha256": helper.sha_file(path)} for name, path in paths.items()}}
    closure_path = fixture / "closure.json"
    write(closure_path, closure)
    args = argparse.Namespace(source_closure=closure_path, resume_exact=None,
        resume_weights=paths["source_checkpoint"], run_id="new_domain_fixture")
    result = helper.closed_source_identity(args)
    assert result["selected_step"] == 1 and result["fresh_optimizer_on_first_launch"]
    checks.append("Actual closure function binds selected checkpoint/config/history/DEV + all five gates")
    for key, bad in (("root_observed_owned_session_completed", False), ("child_exit_code", 1)):
        changed = {**closure, key: bad}; write(closure_path, changed)
        reject(lambda: helper.closed_source_identity(args), "Unclosed/nonzero source rejected: " + key)
    write(closure_path, closure)
    paths["source_checkpoint"].write_bytes(b"mutated")
    reject(lambda: helper.closed_source_identity(args), "Mutated selected checkpoint SHA rejected")
    paths["source_checkpoint"].write_bytes(b"synthetic byte identity only; not a real model")
    bad_summary = copy.deepcopy(summary); bad_summary["validation_es"]["asr"]["wer"] = .12
    write(paths["source_development_summary"], bad_summary)
    updated = copy.deepcopy(closure)
    updated["source_development_summary"]["sha256"] = helper.sha_file(paths["source_development_summary"])
    write(closure_path, updated)
    reject(lambda: helper.closed_source_identity(args), "Self-consistent hash but actual failed ES gate rejected")
    write(paths["source_development_summary"], summary); write(closure_path, closure)
    bad_config = copy.deepcopy(config); bad_config["replay_kl"] = 5.
    write(paths["source_run_config"], bad_config)
    updated = copy.deepcopy(closure); updated["source_run_config"]["sha256"] = helper.sha_file(paths["source_run_config"])
    write(closure_path, updated)
    reject(lambda: helper.closed_source_identity(args), "Different source KD recipe rejected")
    write(paths["source_run_config"], config); write(closure_path, closure)
    reject(lambda: helper.closed_source_identity(argparse.Namespace(**{**vars(args), "run_id": source_run})),
           "Reusing predecessor namespace rejected")
    reject(lambda: helper.closed_source_identity(argparse.Namespace(**{**vars(args), "resume_weights": fixture / "wrong.pt"})),
           "Fresh Adam weights must equal actual selected source path")
    print(json.dumps({"passed": True, "checks": len(checks)}))
report = {"kind": "domain30_source_closure_tiny_verification_v1", "passed": True,
          "checks": len(checks), "check_names": checks, "synthetic_only": True,
          "actual_source_closed_or_GPU_run_claimed": False}
(ROOT / "outputs/parakeet_turkish/domain30_source_closure_tiny_verification_v1.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8")
