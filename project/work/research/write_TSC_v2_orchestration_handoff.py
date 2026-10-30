"""Write a NEW compact source handoff; no actual TSC adoption or GPU launch."""
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'outputs/parakeet_turkish'
def record(path):return {'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
names=('run_hf_TSC_arm_v2.py','resume_TSC_same_science_v2.py','launch_TSC_same_science_resume_v2.py','supervise_TSC_same_science_v2.py')
proof=HERE/'TSC_same_science_exact_resume_tiny_verification_v2.json';p=json.loads(proof.read_text(encoding='utf-8'));assert p['passed'] and p['check_count']==21
value={'schema':2,'kind':'TSC_guarded_quality_continuation_source_handoff_v2',
    'status':'CPU_routing_ready_actual_approved_quarantine_combined_inputs_and_ROOT_launch_pending',
    'passed_CPU_routing_checks':True,'created_at_utc':datetime.now(timezone.utc).isoformat(),
    'orchestration_sources':{name:record(HERE/name) for name in names},
    'scientific_handoff':record(HERE/'TSC_quarantined_isolated_training_handoff_v2.json'),
    'runtime23_sources_receipt':record(HERE/'TSC_runtime_chain_pins_v2.json'),
    'exact_only_memory_launcher':record(HERE/'launch_with_memory_guard_TSC_resume_v2.py'),
    'focused_integrated_fixture':{'report':record(proof),'fixture':record(ROOT/'work/research/TSC_same_science_tiny_check_v2.py'),
        'checks':21,'synthetic_only':True,'actual_checkpoint_read_or_deserialization':False,'native_preload_and_memory_dispatch_mocked':True},
    'source_recipe_fixture':record(HERE/'TSC_v2_common_recipe_tiny_verification.json'),
    'initial_source':'Explicit ROOT-chosen actual CLOSED qualifying original8199 BEST; uniform or domain source allowed, never nonqualifying latest',
    'domain_source_admission':{'requires_closure_allow_source_sampler_transition_to_combined_uniform':True,
        'requires_exact_source_domain_sampling_identity':True,'requires_source_code_pins':True,
        'only_source_custom_recipe_keys_excluded':['sampling_mode','fleurs_fraction','replay_seed'],
        'every_common_source_recipe_parameter_retained_except_initial_epochs_steps':True,
        'TR_transition':'domain30 to uniform combined pool when domain source chosen',
        'replay_nuisance':'independent to shared TR/replay RNG when domain source chosen; exact original1595 pool/labels/teacher, no replay-sequence equality claim'},
    'quarantine':{'requested_uncertainty_stems':['188561','192361','188814'],'not_relabelled_or_declared_wrong':True,
        'actual_removed_count_and_pcm_records_from_ROOT_closed_full_import_receipt':True,
        'original_import_preserved':True,'kept_labels_unchanged':True,'original_TR74883_prefix_and_RP1595_exact':True,
        'quarantine_proof_transitively_pinned_by_input_proof_and_explicit_ROOT_approval':True},
    'first_optimizer':'fresh Adam/model-only qualifying BEST warm start; new pool/sampler/local counters are explicit',
    'horizon':'epochs1; ceil(actual approved quarantined combined TR unique count /8); never reset horizon to250',
    'DEV_feedback':'Exact original copied source eval_epoch_fraction and DEV every250 updates;250 is a screenpoint, not the scientific horizon',
    'later_windows':'SAME run_id, config/native-ready/sourceclosure/quarantine/data; byte-identical closed latest/BEST archives; full Adam/sampler/RNG/history/BEST restore enforced by frozen worker',
    'fresh_runtime':'Separate actual fresh PID/CUDA12.9 preload before Torch; immutable initial native-ready remains original config identity',
    'cache_callback_accounting':'final_completed_steps minus initial inherited_steps',
    'resources':{'cap_GB':8,'optimizer_end_cache_GB':4,'native_inactive_bytes':750000000,'strict_original_guards':True},
    'automatic_supervisor':{'window_hours':3,'max_hours':12,'budget_only':True,'full_dynamic_horizon_unchanged':True,
        'rejects_retention_plateau_resource_manual_internal_failure':True,
        'initial_closure':'Pinned actual first parent child-close/completion/terminal-publish metadata, no independent OS-exit claim',
        'successor_closure':'Supervisor-owned Popen.poll/wait exit0 before a successor'},
    'CLI_templates':{
        'first':'work/nemo-native-env-v2/Scripts/python.exe outputs/parakeet_turkish/run_hf_TSC_arm_v2.py --loop-id NEW_TSC_LOOP --source-checkpoint ACTUAL_CLOSED_QUALIFIED_BEST --source-selection ACTUAL_CLOSED_SELECTION --source-closure ROOT_TSC_CLOSED_SOURCE_V2_RECEIPT --source-run-config ACTUAL_CLOSED_SOURCE_CONFIG --tsc-input-proof ACTUAL_APPROVED_QUARANTINED_COMBINED_PREPARATION --tsc-training-approval ROOT_ACTUAL_TSC_ADOPTION_APPROVAL_V2 --source-choice-rationale EXPLICIT_SOURCE_REASON --window-hours 3 --poll-seconds 5 --gpu-window-authorized',
        'exact':'work/nemo-native-env-v2/Scripts/python.exe outputs/parakeet_turkish/resume_TSC_same_science_v2.py --source-loop ACTUAL_CLOSED_TSC_WINDOW --run-id SAME_TSC_RUN --output-dir NEW_EXACT_WINDOW --source-closed-confirmed --window-hours 3 --poll-seconds 5 --gpu-window-authorized',
        'supervisor':'work/nemo-native-env-v2/Scripts/python.exe outputs/parakeet_turkish/supervise_TSC_same_science_v2.py --source-loop ACTUAL_FIRST_LIVE_TSC_WINDOW --run-id SAME_TSC_RUN --supervisor-dir NEW_SUPERVISOR_DIR --launcher-pid ACTUAL_FIRST_LAUNCHER_PID --worker-pid ACTUAL_SELF_WORKER_PID --successor-prefix NEW_EXACT_WINDOW_PREFIX --window-hours 3 --max-hours 12 --poll-seconds 5 --gpu-window-authorized'},
    'pending_actual_prerequisites':['ROOT executed post-import quarantine and approved actual derivative','Authentic combined metadata/count/horizon with no new accepted-row dropping','Actual CLOSED qualifying source receipt with explicit choice rationale and any source-sampler transition approval','ROOT exclusive GPU ownership; actual source-first preflight; frozen worker verifies TR+replay joint bounds before optimizer'],
    'global_qualified_winner_retained_until_better_actual_DEV':True,'goal_complete':False,'TEST_or_export_performed':False,
    'ML_audio_GPU_or_live_checkpoint_execution_by_preparation':False,'active_or_frozen_v1_sources_modified':False}
output=HERE/'TSC_guarded_quality_continuation_handoff_v2.json'
with output.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2)
print(json.dumps({'passed_CPU_routing':True,'handoff':record(output),'source_sha256':{name:value['orchestration_sources'][name]['sha256'] for name in names}}))
