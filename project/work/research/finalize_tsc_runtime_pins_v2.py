"""Publish isolated v2 routing pins after the executed focused stdlib fixture."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/parakeet_turkish"
OLD_PINS_SHA = "1a8294aa034e37d85f65fb66d4725ffe39432169be1dcaee9a77307f7465c220"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}


def main(args):
    destination = OUT / "TSC_runtime_chain_pins_v2.json"
    assert not destination.exists(), "Never overwrite any pin receipt"
    old_path = OUT / "TSC_runtime_chain_pins_v1.json"
    assert sha(old_path) == OLD_PINS_SHA
    old = json.loads(old_path.read_text(encoding="utf-8"))
    for name, digest in old["source_sha256"].items():
        assert sha(OUT / name) == digest, name
    proof_path = args.dispatch_proof.resolve()
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    assert proof["kind"] == "isolated_TSC_runtime_mocked_CPU_dispatch_verification_v2"
    assert proof["passed"] is True and proof["actual_model_or_GPU_execution"] is False
    assert proof["exact_admission_checked"] is True and proof["quarantine_source_pin_checked"] is True
    replace = {"cuda_cache_caller_tsc_v1.py", "launch_with_io_retry_tsc_v1.py",
               "launch_with_cache_reclaim_tsc_v1.py", "launch_with_memory_guard_tsc_v1.py",
               "launch_native_training_tsc_v1.py", "train_lora_batched_tsc_v1.py",
               "tsc_training_inputs_v1.py"}
    pins = {name: digest for name, digest in old["source_sha256"].items() if name not in replace}
    pins.update(proof["source_sha256"])
    assert len(pins) == 23, pins
    for name, digest in pins.items():
        assert sha(OUT / name) == digest, name
    handoff = args.worker_handoff.resolve()
    assert handoff.is_relative_to(ROOT) and sha(handoff) == args.worker_handoff_sha
    worker_proof = json.loads(handoff.read_text(encoding="utf-8"))
    assert worker_proof.get("passed") is True
    exact_path = OUT / "launch_with_memory_guard_TSC_resume_v2.py"
    assert proof["exact_memory_launcher"] == record(exact_path)
    policy = dict(old["policy"])
    policy["declared_scientific_changes"] = [
        "Append approved post-quarantine TSC TRAIN to exact74883 original TR prefix; original1595 replay/cache unchanged",
        "Exclude exactly188561/192361/188814 through immutable quarantine evidence; retain original full import proof and raw labels",
        "epochs1 and stepsceil(actualCombinedN/8); all common source recipe fields unchanged",
        "Explicitly declare domain30/independent replay to combined-uniform/shared replay RNG when the chosen CLOSED source uses domain sampling"]
    result = {"schema": 1, "kind": "isolated_TSC_runtime_chain_pins_v2", "status": "CPU_routing_prepared",
        "passed_CPU_routing_checks": True, "source_sha256": pins,
        "source_paths": {name: str(OUT / name) for name in pins},
        "worker_sha256": pins["train_lora_batched_tsc_v2.py"],
        "outer_launcher": record(OUT / "launch_native_training_tsc_v2.py"),
        "memory_launcher": record(OUT / "launch_with_memory_guard_tsc_v2.py"),
        "exact_memory_launcher": record(exact_path),
        "exact_memory_kind": "memory_guard_training_launcher_TSC_resume_v2",
        "exact_runtime_scope": "Separate process-admission source; original23 science pins/native-ready remain immutable",
        "caller_target": {"file": "train_lora_batched_tsc_v2.py", "function": "run", "line": proof["caller_line"],
            "inherited_operations": "Unchanged cuda_cache_reclaim_v2.py; caller/source binding only"},
        "policy": policy, "mandatory_worker_asset_arguments": old["mandatory_worker_asset_arguments"],
        "exact_worker_resume_argument": "--resume-exact replaces --resume-weights; all eight data/closure arguments unchanged",
        "native_ready_kind": "native_tdt_training_backend_v3",
        "native_ready_additional_asset_records_field": "TSC_explicit_asset_records",
        "memory_identity_additional_fields": old["memory_identity_additional_fields"] + ["TSC_quarantine_preparer_sha256", "TSC_parent_supplemental_policy_sha256"],
        "source_closure_required_kind": "root_owned_TSC_closed_source_v2",
        "source_closure_artifacts": old["source_closure_artifacts"],
        "training_approval_required_kind": "root_owned_TSC_TRAIN_candidate_approval_v2",
        "preparation_proof_required_kind": "TSC_supplemental_TR_frame_target_inputs_v2",
        "quarantine_binding": "Mandatory references.quarantine_proof and ROOT approval quarantine_proof; no new worker flag",
        "CPU_dispatch_proof_path": str(proof_path), "CPU_dispatch_proof_sha256": sha(proof_path),
        "repeatable_fixture_path": proof["fixture_script_path"], "repeatable_fixture_sha256": proof["fixture_script_sha256"],
        "worker_scientific_handoff_path": str(handoff), "worker_scientific_handoff_sha256": sha(handoff),
        "prior_v1_pin_receipt": record(old_path),
        "actual_TSC_import_or_training_approval_validated": False,
        "actual_GPU_training_or_model_execution": False,
        "closed_source_selected_weights": "Pending actual ROOT-confirmed CLOSED qualifying BEST",
        "same_run_exact_resume": "Exact-only admission routes prepared; full state restore and real runtime evidence remain future executed gates",
        "active_runtime_sources_or_data_changed": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat()}
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({"path": str(destination), "sha256": sha(destination), "source_count": len(pins),
                      "outer": result["outer_launcher"], "memory": result["memory_launcher"],
                      "exact_memory": result["exact_memory_launcher"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dispatch-proof", type=Path, required=True)
    parser.add_argument("--worker-handoff", type=Path, required=True)
    parser.add_argument("--worker-handoff-sha", required=True)
    main(parser.parse_args())
