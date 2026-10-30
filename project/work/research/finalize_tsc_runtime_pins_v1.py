"""Publish NEW routing pins only after the focused actual-function fixture."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/parakeet_turkish"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    destination = OUT / "TSC_runtime_chain_pins_v1.json"
    assert not destination.exists(), "Preserve all prior pin receipts"
    old = json.loads((OUT / "domain30_runtime_chain_pins_v1.json").read_text())["source_sha256"]
    proof_path = OUT / "TSC_runtime_CPU_dispatch_verification_v1.json"
    proof = json.loads(proof_path.read_text())
    assert proof["kind"] == "isolated_TSC_runtime_mocked_CPU_dispatch_verification_v1" and proof["passed"] is True
    assert proof["check_count"] == 29 and proof["actual_model_or_GPU_execution"] is False
    remove = {"cuda_cache_caller_domain30_v1.py", "launch_with_io_retry_domain30_v1.py",
              "launch_with_cache_reclaim_domain30_v1.py", "launch_with_memory_guard_domain30_v1.py",
              "launch_native_training_domain30_v1.py", "train_lora_batched_domain30_v1.py",
              "turkish_domain_sampler_v1.py", "turkish_domain_sampler_v2.py", "domain30_training_identity_v1.py"}
    pins = {name: digest for name, digest in old.items() if name not in remove}
    pins.update(proof["source_sha256"])
    pins["prepare_TSC_TRAIN_v1.py"] = "08785c2446229ae6716feee5e9103d77c468faf0eaf68a602e0477bc2d36063e"
    for name, digest in pins.items():
        assert sha(OUT / name) == digest, name
    assert len(pins) == 21
    handoff = OUT / "TSC_isolated_training_CPU_verification_v2.json"
    assert sha(handoff) == "dafa9438a55ec3196db734c9ce1945d9fb9d22f58f735083f5a0dfadc2d71a29"
    result = {"schema": 1, "kind": "isolated_TSC_runtime_chain_pins_v1", "status": "CPU_routing_prepared",
              "passed_CPU_routing_checks": True, "source_sha256": pins,
              "source_paths": {name: str(OUT / name) for name in pins},
              "worker_sha256": pins["train_lora_batched_tsc_v1.py"],
              "outer_launcher": {"path": str(OUT / "launch_native_training_tsc_v1.py"), "sha256": pins["launch_native_training_tsc_v1.py"]},
              "memory_launcher": {"path": str(OUT / "launch_with_memory_guard_tsc_v1.py"), "sha256": pins["launch_with_memory_guard_tsc_v1.py"]},
              "caller_target": {"file": "train_lora_batched_tsc_v1.py", "function": "run", "line": 545,
                                "inherited_operations": "unchanged cuda_cache_reclaim_v2.py; only caller/source binding differs"},
              "policy": {"student_vocabulary_size": 8199, "teacher_vocabulary_size": 8193, "blank_id": 8192,
                         "duration_rows": [8199, 8204], "six_character_Adam_delta_rows": [8193, 8199],
                         "six_character_Adam_delta_multiplier": 6, "physical_batch": 8, "accumulation": 2,
                         "allocator_cap_bytes": 8000000000, "optimizer_end_cache_threshold_bytes": 4000000000,
                         "native_inactive_threshold_bytes": 750000000, "native_full_joint_bound_bytes": 1700000000,
                         "native_sigma": 0., "native_omega": 0., "loss_reduction": "per-clip raw NLL / target length; physical-batch mean; unchanged accumulation",
                         "initial_optimizer_sampler_RNG_history": "fresh from ROOT-confirmed CLOSED qualifying original8199 BEST",
                         "declared_scientific_changes": ["Append approved TSC TRAIN to exact74883 original TR prefix; original1595 replay/cache unchanged",
                                                         "epochs1 and stepsceil(actualCombinedN/8); other source recipe options unchanged"],
                         "original_runtime_guard_policy_changed": False},
              "mandatory_worker_asset_arguments": ["--tsc-input-proof", "--tsc-training-approval", "--source-closure",
                                                   "--tr-manifest", "--replay-manifest", "--tr-frame-metadata", "--tr-frame-metadata-proof",
                                                   "--data-v3-proof", "--resume-weights"],
              "native_ready_kind": "native_tdt_training_backend_v3",
              "native_ready_additional_asset_records_field": "TSC_explicit_asset_records",
              "memory_identity_additional_fields": ["TSC_explicit_asset_records", "TSC_inputs_helper_sha256",
                                                     "TSC_supplemental_policy_sha256", "TSC_importer_sha256"],
              "source_closure_required_kind": "root_owned_TSC_closed_source_v1",
              "source_closure_artifacts": ["source_checkpoint", "source_selection", "source_run_config", "source_training_history", "source_development_summary"],
              "training_approval_required_kind": "root_owned_TSC_TRAIN_candidate_approval_v1",
              "preparation_proof_required_kind": "TSC_supplemental_TR_frame_target_inputs_v1",
              "CPU_dispatch_proof_path": str(proof_path), "CPU_dispatch_proof_sha256": sha(proof_path),
              "repeatable_fixture_path": str(ROOT / "work/research/check_tsc_runtime_dispatch_v1.py"),
              "repeatable_fixture_sha256": sha(ROOT / "work/research/check_tsc_runtime_dispatch_v1.py"),
              "worker_scientific_handoff_path": str(handoff), "worker_scientific_handoff_sha256": sha(handoff),
              "actual_TSC_import_or_training_approval_validated": False,
              "actual_GPU_training_or_model_execution": False, "closed_source_selected_weights": "pending actual ROOT receipt",
              "same_run_exact_resume": "not implemented by these initial-arm routes; no legacy migration or old-pool fallback",
              "active_runtime_sources_or_data_changed": False, "created_at_utc": datetime.now(timezone.utc).isoformat()}
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({"path": str(destination), "sha256": sha(destination), "source_count": len(pins),
                      "worker": result["worker_sha256"], "outer": result["outer_launcher"], "memory": result["memory_launcher"]}))


if __name__ == "__main__":
    main()
