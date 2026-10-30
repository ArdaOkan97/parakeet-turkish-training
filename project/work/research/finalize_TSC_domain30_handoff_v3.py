"""Save concise immutable source-only TSC science handoff; no corpus/model actions."""
from pathlib import Path
import hashlib,json
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/"outputs/parakeet_turkish"
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
proof_path=HERE/"TSC_domain30_v3_CPU_verification_v1.json"
proof=json.loads(proof_path.read_text(encoding="utf-8"))
assert proof["passed"] and all(sha(HERE/name)==value for name,value in proof["source_sha256"].items())
result={"schema":1,"kind":"TSC_domain30_scientific_worker_handoff_v3","status":"prepared_only",
    "created_at_utc":datetime.now(timezone.utc).isoformat(),"passed_CPU_checks":True,
    "source_sha256":proof["source_sha256"],"optimizer_end_synchronize_line":proof["optimizer_end_synchronize_line"],
    "CPU_proof":{"path":str(proof_path),"sha256":sha(proof_path)},
    "repeatable_fixture":{"path":str(ROOT/"work/research/check_TSC_domain30_v3.py"),"sha256":sha(ROOT/"work/research/check_TSC_domain30_v3.py")},
    "worker":"train_lora_batched_tsc_v3.py","native_outer":"launch_native_training_tsc_v3.py","native_ready_kind":"native_tdt_training_backend_v3",
    "new_required_arguments":{"sampling_mode":"domain30","fleurs_fraction":.3,"replay_seed":"exact CLOSED domain source value"},
    "corpus_input_contract":{"unchanged_helper":"tsc_training_inputs_v2.py","kind":"TSC_supplemental_TR_frame_target_inputs_v2",
        "quarantine_in_references":True,"new_quarantine_worker_flag":False,"ROOT_training_approval_still_required":True},
    "closed_source_contract":{"kind":"root_owned_TSC_domain30_closed_source_v3","root_observed_owned_session_completed":True,"child_exit_code":0,
        "allow_source_domain30_expansion":True,"source_sampling_metadata":"exact original domain_sampling_identity",
        "records":["source_checkpoint","source_selection","source_run_config","source_training_history","source_development_summary"],
        "initial_source":"latest actual CLOSED qualifying original8199 domain BEST; no selection made here"},
    "policy":{"FLEURS_clip_fraction":.3,"other_domain":"uniform CV plus approved TSC without replacement per cycle",
        "replay_fraction":.5,"replay_RNG_independent":True,"initial_optimizer_sampler_RNG":"fresh","strict_retention_gates_changed":False,
        "model_optimizer_objective_LR_cadence_changes":False,"full_pass_claim_requires_each_source_seen":True},
    "actual_quarantined_pool_plan":{"original_TR":74883,"FLEURS":2037,"CV":72846,"TSC":178915,"combined_TR":253798,
        "other_domain":251761,"minimum_domain_horizon_updates":44958,"TR_draws":359664,"FLEURS_draws":107899,"other_draws":251765,
        "FLEURS_repeat_visits":105862,"other_boundary_repeats":4,"unique_total_if_all_seen":253798},
    "memory_lifetime_review":{"frozen_v2_context_and_independently_loaded_combined_audit_rows":"function local, released before load_model; no arrays retained in returned audit",
        "v3_row_annotation":"same frozen v2 proof/frame/target/stat checks; streamed metadata, only worker-owned separately loaded eligible rows augmented in-place",
        "protected_files_and_context_objects_mutated":False,
        "avoided_model_time_arrays":"full metadata dictionary array and second augmented dictionary array",
        "large_context_audit_before_model_still_required":True,
        "actual_peak_memory_measured_for_v3":False,"resource_guards_unchanged":True},
    "actual_TSC_model_GPU_adoption":False,"actual_closed_qualifying_source_selected":False,
    "limitations":["Initial changed pool resets sampler frontier explicitly; subsequent same-family exact windows must retain state.",
        "30% is clip quota; actual FLEURS/TSC/CV audio exposure and resource peaks must be recorded.","Fixed DEV remains a repeatedly selected small set, not final generalization evidence."]}
path=HERE/"TSC_domain30_scientific_worker_handoff_v3.json"
with path.open("x",encoding="utf-8") as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
print(json.dumps({"path":str(path),"sha256":sha(path),"proof_sha256":sha(proof_path)}))
