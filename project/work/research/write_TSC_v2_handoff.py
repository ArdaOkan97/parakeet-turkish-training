"""Write NEW source-only handoff; does not read corpus/model/checkpoint data."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/"outputs/parakeet_turkish"
def record(path):
    path=Path(path).resolve()
    return {"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
names=("prepare_TSC_quarantine_v2.py","tsc_supplemental_data_policy_v2.py","tsc_training_inputs_v2.py","train_lora_batched_tsc_v2.py")
receipt=HERE/"TSC_quarantine_and_domain_source_admission_CPU_verification_final_v2.json"
proof=json.loads(receipt.read_text(encoding="utf-8"))
assert proof["passed"] and proof["checks"]==32
source={name:record(HERE/name)["sha256"] for name in names}
assert source==proof["source_sha256"]
handoff={"schema":2,"kind":"TSC_quarantined_isolated_training_source_handoff_v2","passed":True,
 "prepared_at_utc":datetime.now(timezone.utc).isoformat(),"source_only_readiness":True,"source_sha256":source,
 "CPU_verification":record(receipt),"repeatable_fixture":record(ROOT/"work/research/check_TSC_quarantine_v2.py"),
 "repeatable_fixture_command":"python work/research/check_TSC_quarantine_v2.py --output outputs/parakeet_turkish/<NEW_fixture_receipt>.json",
 "optimizer_end_CallerTarget":{"worker":"train_lora_batched_tsc_v2.py","function":"run","line":545},
 "native_ready_kind":"native_tdt_training_backend_v3","unchanged_resource_configuration":{"cap_gb":8,"optimizer_end_cleanup_threshold_gb":4,"native_inactive_cleanup_threshold_bytes":750000000},
 "frozen_v1_runtime_and_data_modified":False,"actual_quarantine_input_preparation_GPU_training_executed_by_this_agent":False,
 "quarantine_ROOT_command":"python outputs/parakeet_turkish/prepare_TSC_quarantine_v2.py --full-import-proof outputs/parakeet_turkish/tsc_TRAIN_candidate_full_v1_preparation.json --import-closure <ROOT_closed_import_receipt> --whisper-review outputs/parakeet_turkish/TSC_TRAIN_Whisper_actual_prediction_ROOT_review_v1.json --output-dir work/data/TSC_TRAIN_quarantined_v2 --report outputs/parakeet_turkish/TSC_TRAIN_uncertainty_quarantine_v2.json",
 "import_closure_contract":{"kind":"root_owned_TSC_full_import_closure_v2","passed":True,"root_observed_owned_session_completed":True,"child_exit_code":0,
   "import_proof":{"path":"actual completed full proof","sha256":"actual SHA"},"accepted_manifest":{"path":"actual unchanged accepted_train.jsonl","sha256":"actual SHA from full proof"}},
 "quarantine_contract":{"kind":"TSC_TRAIN_uncertainty_quarantine_v2","requested_stems":["188561","192361","188814"],
   "removal_policy":"Only actual present source IDs with reviewed exact PCM/text/duration; absent reviewed cases separately recorded",
   "labels":"All retained text/reference/audio/order/metadata bytes unchanged; no teacher relabel or crop",
   "source_truth":"Actual CLOSED importer proof retained; separate derivative never overwrites original manifests",
   "horizon":"combinedN=74883+actualRetainedSuffix; one uniform pass ceil(combinedN/8), original1595 replay unchanged"},
 "input_prepare_ROOT_command":"python outputs/parakeet_turkish/tsc_training_inputs_v2.py --tsc-import-proof <actual completed full import proof> --tsc-quarantine-proof <actual quarantine proof> --root-approval <ROOT v2 approval> --replay-pool <unchanged original1595 training_pool.json> --replay-config <original1595 run_config.json> --output-dir <NEW work/data inputs directory> --report <NEW outputs input proof> --require-full-train-attempt",
 "input_contract":{"kind":"TSC_supplemental_TR_frame_target_inputs_v2","files":["combined_TR","original_replay","TR_metadata"],
   "references_required":["import_proof","quarantine_proof","ROOT_approval","replay_pool","replay_config","original_metadata","original_frame_proof"],
   "new_worker_quarantine_cli_flag":False,"runtime_binding":"Mandatory tsc-input-proof references.quarantine_proof plus ROOT approval.quarantine_proof; fresh target labels/frames must pass existing strict acceptance and full joint bound"},
 "approval_contract":{"kind":"root_owned_TSC_TRAIN_candidate_approval_v2","passed":True,"allow_separate_TRAIN_candidate_preparation":True,
   "import_proof":{"path":"actual full import proof","sha256":"actual SHA"},"quarantine_proof":{"path":"actual derivative proof","sha256":"actual SHA"},
   "training_additional_requirements":"allow_isolated_training_arm true; actual source/license, pilot PCM/text audit and independently reviewed18 TRAIN Whisper sanity; human audition false, CC-BY-4.0 primary publisher URL, separate HF license label"},
 "source_closure_contract":{"kind":"root_owned_TSC_closed_source_v2","passed":True,"root_observed_owned_session_completed":True,"child_exit_code":0,
   "source_run_id":"actual closed original8199 uniform/domain source",
   "required_path_SHA_records":["source_checkpoint","source_selection","source_run_config","source_training_history","source_development_summary"],
   "source_checkpoint":"Only selected qualifying BEST; actual fixed6x50DEV and all5retention gates required",
   "domain_additional_fields":{"allow_source_sampler_transition_to_combined_uniform":True,"source_sampling_metadata":"exact original config_identity.domain_sampling_identity"}},
 "common_source_recipe_API":"common_source_recipe(source_run_config_dict, ROOT_closure_dict) -> common training argument dict; only exact3 domain keys excluded, epochs/steps untouched until explicitly declared combinedN horizon",
 "approve_API":"approve(approval_path, import_proof_path, *, training=False, quarantine_proof_path=None); explicit quarantine path mandatory",
 "scientific_scope":"NEW data/one-pass horizon and, for a domain source, explicit domain30->combineduniform/independent->shared replay RNG policy transition. Initial weights-only BEST warm start with fresh Adam/sampler/RNG; no exact parent continuation claim. Other recipe, teacher/replay labels, tokenizer8199, rank8, KD10, B8x2 and strict BEST gates unchanged.",
 "known_limitations":["Three purpose-selected disagreements are uncertainty cases, not proof of incorrect labels or corpus-wide noise rate.","No supported human audio audition occurred.","Actual derivative and authentic frame metadata remain ROOT-executed after completed full import/approval.","No GPU memory/quality/adoption guarantee is claimed by these source/fixture checks."]}
target=HERE/"TSC_quarantined_isolated_training_handoff_v2.json"
with target.open("x",encoding="utf-8") as stream:json.dump(handoff,stream,ensure_ascii=False,indent=2)
print(json.dumps(record(target)))
