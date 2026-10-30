"""Materialize inactive exact config and control argument handoff; do not execute control."""
import ast,copy,hashlib,json
from pathlib import Path
F=Path(__file__).parent;N=F.parent;P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish';R=P/'work/research'
B=Path(r'E:\gpu_training_scratchpad\runs\BPE9223_resident_RAM_20261018_01')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p):
 p=Path(p);b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def write(p,v):
 p=Path(p);assert not p.exists(),p;p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
source=read(F/'owner19_CPU8_inactive_source_handoff_v1.json');config=read(F/'configuration_candidate_template.json')
review_path=N/'ROOT_sixth_original_terminal_interval_actual_CPU_review_v1.json';review=read(review_path);policy=config['original_clip_geometry_overlay']
assert pin(review_path)['sha256']=='18c534a496d54f1006936fb95f1db125769777932ce8d1762a2c049547b8ad43'
assert review['accepted']and review['reviewer']=='/root'and review['scope']=='ASR18_source_bound_sixth_original_terminal_interval_geometry_overlay_v1'
assert review['overlay_source']==policy['source']and review['certificate']==policy['certificate'];policy['ROOT_review']=pin(review_path)
base=read(source['correct_source_base']['path']);undo=copy.deepcopy(config)
for k in ('original_clip_geometry_overlay','prior_resident_RAM_original_interval_failure','resident_RAM_resource_scheduling_policy'):undo[k]=base[k]
assert undo==base and policy['prior_overlay']==base['original_clip_geometry_overlay']
for k,v in base['resident_RAM_resource_scheduling_policy'].items():assert k=='CPU_protocol_source'or config['resident_RAM_resource_scheduling_policy'][k]==v
D=F/'production_configuration_v1';assert not D.exists();D.mkdir();path=D/'configuration.json';write(path,config)
failure=read(F/'actual_failure_closure_v2.json');pointer=failure['inputs']['pointer'];latest=read(B/'invocations/owner_000014/boot.json')['latest_failure']
output=N/'resident_RAM_training_actual_16';journal=B/'journals/commits_000015.jsonl'
assert not output.exists()and not journal.exists()and not(B/'invocations/owner_000015').exists()
control=F/'validate_final_saved140200_metadata_v1.py';ast.parse(control.read_text(encoding='utf-8'))
fields={'accepted':True,'reviewer':'/root','driver_source':source['sources']['driver'],'wrapper_source':source['sources']['wrapper'],
 'RAM_native_source':pin(S/'asr18_BPE9223_grouped_RAM_native_v1.py'),'CPU_source':source['sources']['CPU'],
 'DEV_helper_source':pin(S/'asr18_resident_serial350_DEV_v2.py'),'source_configuration':pin(path),'initial_checkpoint':pointer,'latest_failure':latest,
 'bounded_RAM_queue_bytes':268435456,'periodic_DEV_interval_updates':4000,
 'private_scalar_prefetch_source':config['private_scalar_prefetch_policy']['source'],
 'private_scalar_prefetch_actual_CPU_review':config['private_scalar_prefetch_policy']['actual_CPU_control_review'],
 'additional_prefetch_host_reservation_bytes':268435456,'original_interval_resume':pin(F/'actual_failure_closure_v2.json'),
 'original_clip_geometry_overlay':policy,'resource_scheduling_policy':config['resident_RAM_resource_scheduling_policy'],
 'periodic_DEV_CPU_quiescence_policy':config['periodic_DEV_CPU_quiescence_policy'],'initial_owner_invocation':15}
args=['--review','<FINAL_ROOT_EXECUTION_BINDING_REQUIRED>','--configuration',str(path),'--pointer',pointer['path'],'--latest-failure',latest['path'],
 '--branch',str(B),'--output',str(output),'--queue-bytes','268435456','--start-invocation','15','--original-interval-resume',str(F/'actual_failure_closure_v2.json')]
assert len(args)==18
cpu=Path(r'E:\gpu_training_scratchpad\scratch\BPE_broad_CPU_preflight_env_v2\Scripts\python.exe')
metadata_args=[str(cpu),'-X','utf8','-B',str(control),'--configuration',str(path),'--output',str(D/'typed_metadata_actual_01')]
write(D/'final_source_configuration_review_handoff_v1.json',{'kind':'ASR18_INACTIVE_exact_saved140200_source_config_typed_gate_handoff_v1',
 'production_admitted':False,'typed_control_not_executed':True,'sources':source['sources'],'source_handoff':pin(F/'owner19_CPU8_inactive_source_handoff_v1.json'),
 'driver_delta':source['driver_delta'],'wrapper_delta':source['wrapper_delta'],'configuration':pin(path),'prior_source_base':source['correct_source_base'],
 'source_vs_runtime_leaf_report':source['runtime_source_distinction'],'actual_ROOT_CPU_review':pin(review_path),
 'actual_CPU_report':pin(F/'ingress_frontend_actual_01/report.json'),'actual_CPU0_closure':pin(F/'ingress_frontend_actual_01_closure.json'),
 'typed_metadata_control_source':pin(control),'exact_metadata_control_argument_array':metadata_args,
 'future_wrapper':source['sources']['wrapper'],'exact_future_wrapper_arguments':args,'ROOT_execution_review_required_fields':fields,
 'future_active_journal':str(journal),'old_whole_and36_positive_tail_preserved':True,'full401072_adminNone':True})
print(json.dumps({'configuration':pin(path),'typed_control':pin(control),'handoff':pin(D/'final_source_configuration_review_handoff_v1.json')}),flush=True)
