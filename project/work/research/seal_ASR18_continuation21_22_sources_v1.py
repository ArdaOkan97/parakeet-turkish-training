"""Lightweight source/metadata sealing only; no target module import."""
import ast,copy,hashlib,json,os
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish'
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1');F=A.parent/'FastMSS_integration_v1'
def pin(path):
    path=Path(path).resolve();return {'path':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def new(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    return pin(path)
previous=read(A/'fullpopulation_firstwindow20_conv_configuration_CPU_actual_02/finite_two_window_configuration_pending_execution_review.json')
old=read(A/'fullpopulation_firstwindow20_conv_source_manifest_v1.json');run=A/'fullpopulation_firstwindow20_conv_actual_01'
pointer=read(run/'GPU_process_0/window_1/checkpoint_20_immutable_pointer.json')
ancestor={'kind':'ASR18_verified_actual_fullpopulation20_diagnostic_ancestor_v1','global_optimizer_updates':20,
    'diagnostic_only':True,'production_enabled':False,'actual_TOOL0':pin(run/'actual_owned_CLOSED0_v1.json'),
    'report':pin(run/'report.json'),'session':previous['session'],
    'checkpoint_pointer':pin(run/'GPU_process_0/window_1/checkpoint_20_immutable_pointer.json'),
    'native_parent_source':pin(P/'work/research/run_ASR18_full_DATA_multiwindow_v11.py'),
    'native_engine_source':pin(S/'asr18_full_DATA_native_window_v5.py'),
    'deferred_recovery_requests':pin(run/'all_actual_native_checkpoints_pending_recovery.json'),
    'source_manifest':pin(A/'fullpopulation_firstwindow20_conv_source_manifest_v1.json')}
protection={'kind':'ASR18_future_owned_checkpoint_protection_plan_v1','adopt_external_sources_as_owned':False,
    'required_roles':{'source':[ancestor['actual_TOOL0'],ancestor['checkpoint_pointer'],pointer['checkpoint']],
    'history':[pin(run/'GPU_process_0/original_authoritative_commits.jsonl'),ancestor['session']],
    'DEV':[pin(A/'diagnostic_optimizer_free_serial350DEV_actual_01/actual_diagnostic_serial350DEV.json'),
           pin(A/'ROOT_fixed_serial350DEV_manifest_membership_review_v1.json')],
    'qualified_winner':[pin(P/'work/closed-sources/TSC_WorldSpeech_expanded_65948_closed_v1/best_DEV_summary.json')]}}
protected=[]
for role in ('source','history','DEV','qualified_winner'):
    for value in protection['required_roles'][role]:
        if value['path'] not in [part['path'] for part in protected]:protected.append(value)
pp=new(A/'continuation21_22_checkpoint_protection_plan_v1.json',protection)
template=read(S/'asr18_full_DATA_multiwindow.disabled.v11.json')
template.update(parent_driver_source=pin(P/'work/research/run_ASR18_full_DATA_multiwindow_v12.py'),
    worker_driver_source=str(P/'work/research/run_ASR18_full_DATA_multiwindow_v12.py'),
    maximum_updates_per_window=1,maximum_total_native_updates=22,execute_window_limit=2,
    planning_arguments=previous['planning_arguments'],initial_inherited_global_updates=20,planned_actual_new_updates=2,
    continuation_ancestor=ancestor,validation_checkpoint_storage_source=pin(S/'bpe_validation_owned_checkpoint_pairs_v1.py'),
    validation_checkpoint_namespace=r'E:\gpu_training_scratchpad\runs\ASR18_continuation21_22_checkpoint_pairs_actual_01',
    validation_checkpoint_policy=None,validation_checkpoint_protection_plan=pp,validation_checkpoint_protected_external=protected,
    validation_checkpoint_metadata_increment_bytes=64_000_000,
    validation_checkpoint_run_identity={'kind':'ASR18_diagnostic_checkpoint20_continuation21_22_run_identity_v1',
        'storage_scope':'validation_only_diagnostic','production_source_eligible':False,'qualified_winner_eligible':False,
        'ancestor':ancestor,'augmentation_identity':template['augmentation_identity'],'new_global_updates':[21,22],'sampler_reset':False})
template.pop('first_window_repreparation_comparison')
template['launch_requires']=['Exact current source/config/resource/storage ROOT reviews',
    'Actual20 TOOL0 plus all inherited1/10/20 fresh recoveries; no source restart',
    'Fresh unchanged native5,624,813,440B/CPU1,934,217,728B/storage/system sole-owner gates',
    'Two exact owned diagnostic writer slots reserved before native; no external checkpoint adoption',
    'Actual CPU6 closures before NEW21/22; resident-first fresh combined CPU admission or measured exit/restore fallback',
    'Both NEW21/22 actual PRE/save/persistence plus explicit returned checkpoint-context release and archive/index',
    'Final native actual0/dead then fresh3.35GB every NEW21/22 checkpoint recovery0',
    'Retirement admission only after all recovery0; no payload unlink/winner/production source']
tp=new(S/'asr18_full_DATA_multiwindow.disabled.v12.json',template)
pol=new(A/'continuation21_22_validation_checkpoint_policy_PROPOSAL_v1.json',read(F/'validation_checkpoint_pair_policy_DRAFT_v1.json'))
paths=[P/'work/research/run_ASR18_full_DATA_multiwindow_v12.py',S/'asr18_full_DATA_native_window_v6.py',
    S/'bpe_final_DATA_schedule_context_v2.py',S/'asr18_deferred_GPU_recovery_v2.py',S/'asr18_restore_only_runtime_v2.py',
    S/'asr18_verified_pipeline_ancestor_v1.py',S/'bpe_validation_owned_checkpoint_pairs_v1.py',
    P/'work/research/prepare_ASR18_continuation21_22_configuration_v1.py']
for path in paths:ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
direct=[pin(path) for path in paths];bindings=[];seen=set()
for value in old['bindings']+direct+[tp]:
    if value['path'] not in seen:bindings.append(value);seen.add(value['path'])
manifest={'kind':'ASR18_checkpoint20_two_one_update_windows21_22_source_review_manifest_v1','owner':'/root/memory_diagnosis',
    'requested_source_scope':'source_read_review_for_future_checkpoint20_continuation21_22_diagnostic',
    'later_execution_review_scope_required':'finite_full_DATA_checkpoint20_continuation_two_one_update_windows_v1',
    'production_enabled':False,'direct_sources':direct,'bindings':bindings,'configuration_template':tp,'CPU_owner_bundle':old['CPU_owner_bundle'],
    'ancestor':ancestor,'previous_source_read_review':pin(A/'ROOT_fullpopulation_firstwindow20_conv_source_read_review_v1.json'),
    'storage_component_source_review':pin(A/'ROOT_validation_checkpoint_pair_source_read_review_v1.json'),
    'storage_component_manifest':pin(F/'validation_checkpoint_pair_source_review_manifest_v1.json'),
    'storage_policy_proposal':pol,'protection_plan':pp,'protected_external':protected,
    'numeric_change_claimed':False,'unchanged_native_private_forecast_bytes':5624813440,'unchanged_CPU_family_bytes':1934217728,
    'additional_parent_storage_phase_bytes':64000000,'checkpoint_namespace':template['validation_checkpoint_namespace'],
    'actual_NEW_CPU_or_GPU_jobs_launched':0,'old20_and350_actual0_preserved':True,'production_source_eligible':False}
mp=new(A/'continuation21_22_full_caller_source_review_manifest_v1.json',manifest)
print(json.dumps({'manifest':mp,'direct_sources':direct,'template':tp,'policy_proposal':pol,'protection_plan':pp,'binding_count':len(bindings)}),flush=True)
