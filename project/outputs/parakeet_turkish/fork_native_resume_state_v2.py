"""CPU-only native358 exact state fork, reusing the frozen v1 tree audit.

Only checkpoint top-level run_id changes. Latest remains an immutable archive;
only qualifying BEST is preseeded in work, so trainer's fresh latest is absent.
Scientific/runtime config migration is a separate declared receipt.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil

import fork_native_resume_state_v1 as base

HERE, ROOT = base.HERE, base.ROOT
SOURCE = "adaptive_20261011_native_memory_fix_01_public_winner_extension"
RECEIPT_SHA = "1a49692e738d07621435dd617ea6bb67afc696a49cca9759f695e4ce4d2495c0"


def main(args):
    import torch
    torch.set_num_threads(2)
    base.require(not torch.cuda.is_initialized(), "CPU fork must never initialize CUDA")
    base.require(re.fullmatch(r"[A-Za-z0-9_-]+", args.target_run_id), "Safe NEW run ID")
    receipt_path = base.scoped(args.predecessor_exit_receipt)
    base.require(base.sha(receipt_path) == RECEIPT_SHA, "Actual closed extension receipt changed")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    base.require(receipt["root_owned_exec_session"] == 40244 and receipt["root_observed_owned_session_completed"]
        and receipt["root_observed_exit_code"] == 1 and receipt["child_exit_code"] == 0
        and receipt["predecessor_run_id"] == SOURCE and receipt["child_durable_step"] == 358
        and receipt["child_selected_step"] == 0 and receipt["heldout_tests_evaluated"] is False,
        "Actual root-owned closed358/best0 source required")
    for role in ("source_latest", "source_best", "child_run_config", "child_history", "child_selection", "resource_review"):
        base.require(base.sha(base.scoped(receipt[role+"_path"])) == receipt[role+"_sha256"], "Closed source changed: "+role)
    base.require(base.sha(base.scoped(receipt["predecessor_status_path"])) == receipt["predecessor_status_sha256"], "Closed controller status changed")
    for item in receipt["runtime_provenance"].values():
        base.require(base.sha(base.scoped(item["path"])) == item["sha256"], "Closed runtime evidence changed")
    folder = base.scoped(args.output_dir)
    target_run = HERE / "runs" / args.target_run_id
    base.require(not folder.exists() and not target_run.exists(), "Fresh fork/run namespaces required")
    for role in ("latest", "best"):
        base.require(not (ROOT / "work" / (args.target_run_id+"_"+role+".pt")).exists(), "Fresh work namespace required")
    folder.mkdir(parents=True)
    manifest = {"schema": 1, "kind": "native_exact_state_run_id_fork_v2", "status": "CPU_prepared", "passed": True,
        "source_run_id": SOURCE, "target_run_id": args.target_run_id, "source_step": 358, "inherited_selected_step": 0,
        "root_owned_exit_receipt": base.record(receipt_path), "only_changed_state_path": "run_id",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_config": {"path": receipt["child_run_config_path"], "sha256": receipt["child_run_config_sha256"]},
        "source_checkpoint": {}, "fork_checkpoint": {}, "development_reuse": {},
        "runtime_identity_migration_not_performed_by_fork": True, "frozen_v1_tree_auditor_sha256": base.sha(Path(base.__file__))}
    for role, step in (("latest", 358), ("best", 0)):
        source = base.scoped(receipt["source_"+role+"_path"])
        saved = torch.load(source, map_location="cpu", weights_only=True)
        base.require(saved["run_id"] == SOURCE and saved["step"] == step and saved["selected_step"] == 0
            and len(saved["weights"]) == 207 and len(saved["history"]) == step, "Actual learned state/selection differs")
        if role == "latest":
            base.require(saved["schema"] == 2 and len(saved["optimizer"]["state"]) == 207, "Full207-Adam latest required")
        original = base.identity(saved, torch)
        saved["run_id"] = args.target_run_id
        archive = folder / (args.target_run_id+"_"+role+".pt")
        torch.save(saved, archive)
        actual = torch.load(archive, map_location="cpu", weights_only=True)
        base.require(actual["run_id"] == args.target_run_id and base.identity(actual, torch) == original,
            "Fork changed weights/Adam/sampler/RNG/history/config")
        target = ROOT / "work" / (args.target_run_id+"_"+role+".pt")
        if role == "best":
            shutil.copyfile(archive, target)
            base.require(base.sha(target) == base.sha(archive), "BEST preseed bytes changed")
        else:
            base.require(not target.exists(), "Latest archive only; mutable latest must stay absent")
        base.require(base.sha(source) == receipt["source_"+role+"_sha256"], "Source checkpoint changed during fork")
        manifest["source_checkpoint"][role] = {"path": str(source), "sha256": receipt["source_"+role+"_sha256"], "step": step}
        manifest["fork_checkpoint"][role] = {**base.record(archive), "mutable_work_path": str(target),
            "mutable_work_copy_created": role == "best", "step": step, "identity_without_top_run_id": original}
        if role == "latest":
            ci = saved["config_identity"]
            manifest["source_config_identity"] = ci
            manifest["source_config_identity_sha256"] = hashlib.sha256(json.dumps(ci, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
        del saved, actual
    for step in (0, 358):
        source = HERE / "results" / (SOURCE+"_step_"+str(step))
        target = HERE / "results" / (args.target_run_id+"_step_"+str(step))
        base.require(source.is_dir() and not target.exists(), "Actual DEV source and fresh target required")
        files = {str(p.relative_to(source)): base.sha(p) for p in source.rglob("*") if p.is_file()}
        base.require(not any(Path(name).name.startswith("test_") for name in files), "No heldout TEST copying")
        shutil.copytree(source, target)
        base.require(all(base.sha(target/name) == digest for name, digest in files.items()), "Inherited DEV bytes changed")
        manifest["development_reuse"][str(step)] = {"source_path": str(source), "target_path": str(target), "files_sha256": files,
            "evaluation_reused_without_new_inference": True, "heldout_test_copied": False}
    target_run.mkdir()
    source_initial = HERE / "runs" / SOURCE / "initial_development.json"
    shutil.copyfile(source_initial, target_run / "initial_development.json")
    manifest["initial_development_reuse"] = {"source": base.record(source_initial), "target": base.record(target_run / "initial_development.json")}
    base.require(not torch.cuda.is_initialized(), "Unexpected CUDA initialization")
    manifest["CUDA_initialized"] = False
    path = folder / "fork_manifest.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"passed": True, "manifest": str(path), "manifest_sha256": base.sha(path),
        "target_run_id": args.target_run_id, "latest_identity": manifest["fork_checkpoint"]["latest"]["identity_without_top_run_id"], "CUDA_initialized": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-run-id", required=True)
    parser.add_argument("--predecessor-exit-receipt", type=Path, default=HERE / "native_memory_fix_01_extension_exit_receipt_v1.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    main(parser.parse_args())
