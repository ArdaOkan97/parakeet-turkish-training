"""Focused missing optional runtime flag/fresh namespace metadata correction."""
import copy,importlib,json,sys
from pathlib import Path
L=Path(r'E:\gpu_training_scratchpad\analysis\Turkish_labels_BPE_circumflex_v1');sys.path.insert(0,str(L));m=importlib.import_module('audit_and_fit_append_only_v5')
N=Path(r'E:\gpu_training_scratchpad\analysis\BPE9223_training_launch_v1');F=N/'full_configuration_04';O=N/'independent_ready04_metadata_fix_review_v1';O.mkdir(exist_ok=True);m.AREA=O
S=m.PROJECT/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
def rec(path):return {k:v for k,v in m.record(path).items() if k in ('path','bytes','sha256')}
def r3(pin):return {k:pin[k] for k in ('path','bytes','sha256')}
def read(pin):assert rec(pin['path'])==r3(pin);return json.loads(Path(pin['path']).read_text(encoding='utf-8-sig'))
def main():
 g=m.Guard();g.phase='focused_ready04_optional_flag_and_fresh_namespace_joins'
 ready=F/'configuration_ready_for_final_execution_review.json';assert m.sha(ready)=='8b994ef52908c1d142bf6a131acec622cac3d3dc18f6daaa46cf97731579a638'
 c=json.loads(ready.read_text());old=json.loads((N/'full_configuration_03/configuration_ready_for_final_execution_review.json').read_text());pending=F/'configuration_pending_execution_review.json';p=json.loads(pending.read_text())
 delta=sorted(k for k in c if c[k]!=p[k]);assert set(c)==set(p) and delta==['CPU_review_join_required_before_any_execution','ROOT_CPU_execution_review','full_run_launch_enabled']
 assert c['full_run_launch_enabled'] is True and c['CPU_review_join_required_before_any_execution'] is False and p['full_run_launch_enabled'] is False and p['ROOT_CPU_execution_review'] is None
 assert type(c['collect_GPU_timeline']) is bool and c['collect_GPU_timeline'] is False and 'collect_GPU_timeline' not in old
 assert c['source_manifest']==old['source_manifest'] and c['source_manifest']['sha256']=='463c220bb3279c3bffdfe7fdf6e019f0d5e6cbf973c2dfed6b8a99012cab6e84'
 manifest=read(c['source_manifest']);assert len(manifest['bindings'])==432
 expected=copy.deepcopy(old['full_run_identity']);expected['permanent_namespace_scope']='human_authorized_BPE9223_two_broad_one_cleaner_genesis02';assert c['full_run_identity']==expected
 from bpe_broad_production_state_v1 import digest
 assert digest(expected)=='3381c4153f10cd5037f0e6f0ba760414db5a95b5ae240287f044e1fc7892e619'
 for key in ('planning_arguments','vocabulary','row_migration_plan','augmentation_identity','entire_worker_forecast','final_recovery_worker_forecast','CUDA_capacity_policy','forecast_calibration','parent_driver_source','worker_driver_source','full_owner_controller_source','CPU_controller_source','GPU_python','CPU_python','raw_lifetime_review','CPU_owned_allowance_bytes','checkpoint_storage_policy','protected_external','qualified_source_winner','qualified_source_DEV_summary','DEV_execution_sources','DEV_native_boundary_sources','DEV_base_assets','DEV_source_bindings'):
  assert c[key]==old[key],key
 assert c['planned_native_updates']==old['planned_native_updates']==401072 and c['owner_validation_only'] is False and c['diagnostic_only'] is False and c['production_enabled'] is False
 assert c['run_namespace'].endswith('BPE9223_weak2_clean1_20261018_02') and Path(c['checkpoint_storage_namespace'])==Path(c['run_namespace'])/'checkpoints'
 cpu=read(c['ROOT_CPU_execution_review']);assert c['ROOT_CPU_execution_review']['sha256']=='f5b4e87ae4a6f4f7711a9108fd96230881235ee09acdc8fddd80515c7647c29d'
 assert cpu['accepted'] is True and cpu['reviewer']=='/root' and cpu['configuration']==c['CPU_template_configuration'] and cpu['outer_parent_source']==c['parent_driver_source']
 actual=read(cpu['metadata_actual0']);report=read(cpu['metadata_report']);assert actual['actual_native_exit_code']==actual['actual_exec_tool_exit_code']==0 and actual['actual_tool_chunk_id']=='8cc729' and actual['actual_identity_dead'] is True
 assert actual['report']==cpu['metadata_report'] and report['configuration']==rec(pending) and report['exact_serial_DEV_policy_and_source_bound_callback_constructor_preflight_passed'] is True and report['DEV_callbacks_invoked_or_audio_model_loaded'] is False
 assert actual['failed_genesis_preserved']['sha256']=='50dee1a01c38cdcde05adf01e9e1b4cb39582734d7480e6145502958dce32b27'
 assert rec(actual['failed_genesis_preserved']['path'])==actual['failed_genesis_preserved']
 plan=read(c['prepared_metadata_plan']);oldplan=read(old['prepared_metadata_plan']);assert plan==oldplan and c['prepared_metadata_plan']['sha256']==old['prepared_metadata_plan']['sha256']=='5d16d85c969666a9c8902870221da06742b6ae5eab37ca9a4b82709f031052b9'
 assert plan['initial_committed_updates']==0 and len(plan['updates'])==20
 session=read(c['session']);oldsession=read(old['session']);assert session['initial_sampler_sha256']==oldsession['initial_sampler_sha256'] and session['scientific_source_state']==oldsession['scientific_source_state'] and session['DATA_evidence_sha256']==oldsession['DATA_evidence_sha256']
 assert session['full_run_identity_sha256']==digest(expected) and session['production_source_eligible'] is True and session['initial_optimizer_updates']==0 and session['owner_validation_only'] is False and session['checkpoint_diagnostic_only'] is False
 template=read(c['CPU_template_configuration']);assert template['session']==c['session'] and template['full_run_identity_sha256']==digest(expected) and Path(template['output_namespace'])==Path(c['checkpoint_storage_namespace'])/'transient'
 import asr18_full_run_serial_DEV_9223_v3 as serial
 policies={k:read(c[k]) for k in ('durable_policy','ROOT_storage_review','ROOT_DEV_policy_review','ROOT_DEV_static_execution_review','ROOT_DEV_resource_review','actual_ROOT_CPU_resource_review')}
 for key,value in policies.items():assert value['accepted'] is True and value['reviewer']=='/root' and value['run_identity_sha256']==digest(expected) and value['full_run_source_contract']==c['source_manifest']
 policy=policies['ROOT_DEV_policy_review'];static=policies['ROOT_DEV_static_execution_review'];assert policy['implementation']==static['implementation']==rec(serial.__file__) and policy['dependencies']==serial.sources()
 assert static['ROOT_policy_review']==c['ROOT_DEV_policy_review'] and static['resource_review']==c['ROOT_DEV_resource_review'] and static['source_bindings']==c['DEV_source_bindings'] and static['execution_sources']==c['DEV_execution_sources']
 assert policy['scope']=='exact_selective_optimizer_free_serial_DEV_runtime_sources_and_policy' and policy['execution_sources']==c['DEV_execution_sources'] and policy['native_boundary_sources']==c['DEV_native_boundary_sources']
 assert Path(policies['durable_policy']['namespace'])==Path(c['run_namespace'])/'durable' and policies['ROOT_storage_review']['future_namespace']==c['checkpoint_storage_namespace']
 assert cpu['resource_policy']==read(old['ROOT_CPU_execution_review'])['resource_policy']
 assert c['exact_ROOT_execution_configuration_review_still_required'] is True and not Path(c['run_namespace']).exists()
 g.check()
 result={'kind':'ASR18_independent_ready04_failed_startup_metadata_fix_review_v1','passed':True,'reviewer':'/root/label_bpe_audit','configuration':rec(ready),'pending_configuration':rec(pending),'exact_pending_to_ready_changed_keys':delta,
  'source_manifest_all432_unchanged':c['source_manifest'],'explicit_missing_runtime_key_collect_GPU_timeline_false_fixed':True,'fresh_run_namespace':c['run_namespace'],'run_identity_changed_only_permanent_namespace_scope':True,'run_identity_sha256':digest(expected),
  'same_original_model_sources_source0_RNG_snapshot_sampler_plan_newlabels_means_allnumeric_resourceguards':True,'first20_plan_full_semantic_and_raw_SHA_identical_to_valid03':True,
  'genuine_ROOT_CPU_review':c['ROOT_CPU_execution_review'],'metadata05_actual0':cpu['metadata_actual0'],'failed_genesis01_actual1_preserved':actual['failed_genesis_preserved'],
  'all_six_current_ROOT_authority_identity_namespace_and_DEV_static_policy_resource_joins_exact':True,'new_concrete_metadata_blockers':[],
  'remaining':['Exact owned uncommitted failed feature-cache retirement actual0 remains CPU owner responsibility','Separate ROOT final execution2 review must bind cleanup0 and corrected ready04 before sole GPU owner launch'],
  'fresh_run_namespace_not_created':True,'actual_audio_model_GPU_payload_jobs':0,'production_admitted_here':False}
 (O/'source_findings_v1.json').write_text(json.dumps(result,indent=2)+'\n');g.publish(completed=True,findings=rec(O/'source_findings_v1.json'));print(json.dumps({'passed':True,'findings':rec(O/'source_findings_v1.json')}))
if __name__=='__main__':main()
