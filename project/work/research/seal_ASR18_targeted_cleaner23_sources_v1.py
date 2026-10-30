"""Lightweight source AST/receipt sealing; no ML target modules imported."""
import ast,hashlib,json,os
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';R=P/'work/research'
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')
def pin(path):
    path=Path(path).resolve();return {'path':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def new(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    return pin(path)
run=A/'continuation21_22_native_actual_01';report=read(run/'report.json');launch=read(run/'launch.json');config=read(launch['config']['path'])
assert report['passed'] and report['actual_final_global_updates']==22 and report['all_deferred_recovery_obligations_discharged']
ancestor={'kind':'ASR18_verified_actual_pipeline22_diagnostic_ancestor_v1','global_optimizer_updates':22,
    'diagnostic_only':True,'production_enabled':False,'actual_TOOL0':pin(run/'actual_owned_CLOSED0_v1.json'),
    'report':pin(run/'report.json'),'configuration':launch['config'],'session':config['session'],
    'checkpoint_pointer':report['windows'][-1]['latest_pointer'],'native_parent_source':config['parent_driver_source'],
    'native_engine_source':pin(S/'asr18_full_DATA_native_window_v6.py'),
    'deferred_recovery_requests':pin(run/'all_actual_native_checkpoints_pending_recovery.json'),'source_manifest':config['source_manifest']}
template=read(S/'asr18_full_DATA_multiwindow.disabled.v11.json')
template.update(kind='ASR18_targeted_cleaner23_config_TEMPLATE_NOT_LAUNCHABLE_v1',
    parent_driver_source=pin(R/'run_ASR18_targeted_cleaner23_v1.py'),worker_driver_source=str(R/'run_ASR18_targeted_cleaner23_v1.py'),
    maximum_windows=1,maximum_total_native_updates=23,maximum_updates_per_window=1,execute_window_limit=1,
    planning_arguments=config['planning_arguments'],cleaner_ancestor=ancestor,initial_cleaner_diagnostic_sampler_sha256=None,
    provisional_cleaner_plan=None,cleaner_conditional_CPU_resource_review=pin(A/'ROOT_cleaner_CPU6_conditional_resource_component_review_v1.json'),
    actual_completed_production_broad_passes=0,production_phase_advance_claimed=False)
template.pop('first_window_repreparation_comparison')
template['launch_requires']=['Exact NEW cleaner source/resource/configuration ROOT reviews',
    'Actual22 operator TOOL0 +native0 +both NEW21/22 fresh recovery0; no old diagnostic source restart',
    'CPU6 actual headerT before allocation/full cleanPCM/fulltargets/12273heldout/independent reference and frontend closures',
    'Whole CPU1,934,217,728 and unchanged native5,624,813,440/write64M/750M/global/CUDA guards',
    'Exact actual22 model/207Adam/all4RNG/fullsource nextdraw restored before diagnostic cleaner fork',
    'Two selected real cleaner rows1/352768 +exact carried next oldfive/PT replay; zero augmentation and full cleanermean',
    'Real cleaner23 commit in explicitly diagnostic cursor; original22 fullpopulation schedule remains immutable ancestry',
    'PRE/save/persistence ->actual native0/dead ->separate3.35GB actual fullstate23 recovery0',
    'Terminal diagnostic packet forbids further native/draws or fabricated production phase advance']
tp=new(S/'asr18_targeted_cleaner23.disabled.v1.json',template)
paths=[R/'run_ASR18_targeted_cleaner23_v1.py',R/'prepare_ASR18_targeted_cleaner23_configuration_v1.py',
    S/'asr18_verified_pipeline_ancestor_v2.py',S/'asr18_targeted_cleaner_schedule_context_v1.py',
    S/'asr18_targeted_cleaner23_native_v1.py',S/'asr18_targeted_cleaner23_recovery_v1.py',S/'asr18_targeted_cleaner23_restore_runtime_v1.py']
for path in paths:ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
direct=[pin(path) for path in paths];prior=read(config['source_manifest']['path']);bindings=[];seen=set()
for part in prior['bindings']+direct+[tp]:
    if part['path'] not in seen:bindings.append(part);seen.add(part['path'])
manifest={'kind':'ASR18_targeted_cleaner23_CPU6_native_fresh_recovery_source_review_manifest_v1','owner':'/root/memory_diagnosis',
    'requested_source_scope':'source_read_review_for_future_targeted_cleaner23_diagnostic',
    'later_execution_review_scope_required':'finite_targeted_cleaner23_CPU6_native_and_fresh_recovery_diagnostic_v1',
    'production_enabled':False,'direct_sources':direct,'bindings':bindings,'configuration_template':tp,
    'CPU_owner_bundle':prior['CPU_owner_bundle'],'cleaner_ancestor':ancestor,
    'previous_source_manifest':config['source_manifest'],
    'previous_source_read_review':pin(A/'ROOT_continuation21_22_source_read_review_v1.json'),
    'cleaner_conditional_CPU_resource_review':template['cleaner_conditional_CPU_resource_review'],
    'unchanged_native_private_forecast_bytes':5624813440,'unchanged_CPU_family_private_bytes':1934217728,
    'cleaner_mean_U':11972886/352768,'real_cleaner_diagnostic_ordinals':[1,352768],
    'source_core_native_update':pin(S/'asr18_precomputed_native_v2.py'),
    'zero_augmentation_clean_protection_required':True,'actual_NEW_audio_frontend_or_GPU_jobs':0,
    'production_phase_advance_claimed':False,'qualified_checkpoint':False,'ROOT_exact_source_resource_config_review_still_required':True}
mp=new(A/'targeted_cleaner23_full_caller_source_review_manifest_v1.json',manifest)
print(json.dumps({'manifest':mp,'direct_sources':direct,'template':tp,'binding_count':len(bindings)}),flush=True)
