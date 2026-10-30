import hashlib,json,os,time
from pathlib import Path
import psutil
F=Path(__file__).parent;A=F.parent/'resident_RAM_training_actual_24';B=Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01');I=B/'invocations/owner_000023'
assert os.environ['TORCH_EXTENSIONS_DIR'].lower().startswith('e:')
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def read(p):return json.loads(Path(p).read_text())
def who(p):return {'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()}
reviewpath=F/'ROOT_source31_fresh_signature_collect_saved167400_final_execution_review_v1.json';r=read(reviewpath)
restore=read(I/'actual_parent_restore.json');branch=read(I/'actual_journal_branch.json');ready=read(I/'CPU/actual_CPU_ready.json');admission=read(I/'actual_fresh_signature_collect_observation_admission.json')
assert restore['passed']is True and restore['all207_threeAdam_all4RNG_match_actual_pre_save']is True and restore['next_draw_packet_match']is True and restore['restore_occurred_only_at_startup_before_native']is True
assert branch['predecessor']==r['fresh_signature_resume']and branch['saved_prefix']==r['saved167400_prefix_published_not_rehashed']and branch['native_updates_or_draws_in_copy']==0
assert branch['historical_unsaved_tail_updates']==list(range(167401,167437))and Path(branch['new_active_journal']['path'])==B/'journals/commits_000023.jsonl'
assert ready['source']['protocol']==r['CPU_source']and ready['packet_credit']['logical_packet_maximum']==2 and ready['packet_credit']['helper_source']==r['packet_credit_policy']['helper_source']
assert admission['policy']==r['fresh_metadata_signature_and_collect_observation_policy']and admission['signature_scope']['active']is True and admission['original_collect_installed_after_startup_freeze']is True
boot=read(I/'boot.json');child=read(A/'current_child.json');cpu=ready['producer_identity'];wr=psutil.Process(child['launcher']['ppid']);cl=psutil.Process(cpu['ppid'])
expected=[('native',boot['producer']),('native_interpreter_launcher',child['launcher']),('wrapper',who(wr)),('CPU',cpu),('CPU_interpreter_launcher',who(cl))]
family=[]
for role,v in expected:
 actual=who(psutil.Process(v['pid']));assert abs(actual['create_time']-v['create_time'])<.001;family.append({'role':role,**v,'exact_alive':True})
progress=read(B/'progress/latest.json');assert progress['producer']==boot['producer']and progress['source']==r['driver_source']and not progress['optimizer_poisoned']and not progress['schedule_poisoned']and not(I/'actual_failure.json').exists()
paths={'actual_owned_boot':A/'actual_owned_source31_boot_session.json','boot':I/'boot.json','configuration':I/'configuration.json','restore':I/'actual_parent_restore.json','branch':I/'actual_journal_branch.json',
 'scope_admission':I/'actual_fresh_signature_collect_observation_admission.json','CPU_ready':I/'CPU/actual_CPU_ready.json','current_child':A/'current_child.json','execution_review':reviewpath}
value={'kind':'ASR18_actual_source31_restore_scope_CPUready_current5_v1','created_unix':time.time(),'owned_session':2901,'passed_compact_stage_checks':True,
 'evidence':{k:pin(p)for k,p in paths.items()},'actual_five_runtime_identities':family,'actual_progress_snapshot':progress,
 'allstate_restore_before_native_and_savedprefix_copy_after_restore_with_zero_native_or_draws':True,
 'genuine36_redo_and_natural167600_acceptance_still_pending':True,'no_speed_or_complete_fix_claim':True,
 'same_225_floors_full_reservations_MAX2_LIVE1_native_GC_policy_science':True}
p=A/'actual_source31_restore_scope_CPUready_and_current5_family_v1.json'
with p.open('x',encoding='utf-8',newline='\n')as f:json.dump(value,f,indent=2);f.write('\n')
print(json.dumps({'milestone':pin(p),'actual':progress['actual_optimizer_updates'],'CPU':cpu,'current_five':family,'evidence':value['evidence']}))
