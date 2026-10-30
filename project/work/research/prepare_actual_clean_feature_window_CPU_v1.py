"""Serial actual clean-reference/frontend children; provisional bounded window."""
import argparse,hashlib,json,os,shutil,subprocess,sys,time
from copy import deepcopy
from pathlib import Path
P=Path(__file__).resolve().parents[2];R=Path(r'E:\gpu_training_scratchpad');S=P/'outputs/parakeet_turkish'
sys.path.insert(0,str(S))
from fastmss_clean_feature_window_v1 import (CACHE,CAPACITY,FRONTEND_ARGS,create_window,load_plan,observe_feature_child,
 finalize_window,read_bound,record,scoped,write_new,require,FeatureWindowReader)
from fastmss_clean_feature_contract_v1 import KIND,PHASE
from fastmss_ordered_draw_binding_v1 import file_binding,AugmentationDrawContext
from resource_watchdog_v2 import sample_memory
CPU=R/'scratch/BPE_broad_CPU_preflight_env_v2/Scripts/python.exe';LIGHT=P/'work/nemo-native-env-v2/Scripts/python.exe'
A=R/'analysis/FastMSS_integration_v1';M=R/'analysis/ASR18_memory_diagnosis_v1'
def guard():
 v=sample_memory(os.getpid(),'actual_clean_feature_coordinator');w,s=v['worker_private_memory'],v['system_memory']
 require(w['available'] and s['available'] and v['paired_sampling_within_one_second'] and w['private_bytes']<2_000_000_000,'Fresh paired2GB')
 require(s['commit_used_bytes']/s['commit_limit_bytes']<.90 and s['commit_available_bytes']>=2_000_000_000 and shutil.disk_usage(R).free>=100_000_000_000,'System90/2GB/E100GB')
 return v
def admit(part,name,budget):
 v=guard();s=v['system_memory'];require(s['commit_used_bytes']+budget<.90*s['commit_limit_bytes'] and s['commit_available_bytes']-budget>=2_000_000_000,'Fresh bounded phase forecast admission')
 write_new(part/(name+'_admission.json'),{'passed':True,'forecast_bytes':budget,'sample':v})
def actual_guard(path):
 g=read_bound(record(path,guard),guard);require(g['actual_exit_code']==0 and g['monitor_clean'] and g['monitor_joined'] and g['observed']['samples']>=3,'Actual clean joined reference producer0')
 return g
def main(output):
 began=time.monotonic();out=scoped(output);out.mkdir(exist_ok=False)
 require(os.environ['CUDA_VISIBLE_DEVICES']=='-1','CPU only')
 plan_binding=record(M/'cleaner_after20_plan_01/plan.json',guard)
 require(plan_binding['sha256']=='128ebf4ec8906534c4ab51afe83fed009006a74194bb2821fffdfe92ff5438af','Exact native-owner diagnostic clean plan')
 plan=load_plan(plan_binding,guard);require(len(plan['updates'])==1 and plan['production_phase_transition_claimed'] is False and plan['prior_full_broad_corpus_passes']==0,'One explicitly diagnostic phase-hook only')
 tr_binding=record(A/'actual_cleaner_TR_references_01/report.json',guard);tr=read_bound(tr_binding,guard);tg=actual_guard(A/'actual_cleaner_TR_references_01_guard/closure.json')
 rp_binding=record(A/'actual_post20_clean_replay_references_01/report.json',guard);rp=read_bound(rp_binding,guard);rg=actual_guard(A/'actual_post20_clean_replay_references_01_guard/closure.json')
 require(tr['passed'] and rp['passed'] and tr['cleaner_catalog_identity']==plan['cleaner_catalog_identity'],'Exact independent original catalog references')
 original=read_bound({'path':str(A/'clean_feature_fixture_original_bindings_v1.json'),'bytes':17986,'sha256':'84aee2288cb36c35724de29d3a55489e69e371a72f95f9f36f51a96357ba6c7d'},guard)['owner_contents']
 source_paths=[__file__,S/'fastmss_clean_feature_window_v1.py',S/'fastmss_clean_feature_contract_v1.py',P/'work/research/produce_actual_clean_feature_update_CPU_v1.py',P/'work/research/bind_actual_clean_feature_audio_proof_CPU_v1.py',P/'work/research/ASR18_bounded_CPU_phase_v3.py']
 sources=[record(p,guard) for p in source_paths]
 write_new(out/'launch_binding.json',{'plan':plan_binding,'source_bindings':sources,'actual_optimizer_updates':0,'authoritative_frontiers_advanced':False})
 owner=create_window(plan_binding,window_id=out.name,augmentation_identity=original['augmentation_identity'],frontend=original['frontend'],guard=guard)
 write_new(out/'window_owner.json',owner);closures=[];parts=[]
 for index,u in enumerate(plan['updates']):
  part=out/f'update_{index+1:02}';part.mkdir();rows=[deepcopy(v) for v in u['TR_rows']+u['replay_rows']]
  refs=[];trmap={v['cleaner_original_ordinal']:v for v in tr['references']};rpmap={v['source_id']:v for v in rp['references']}
  for j,(row,ctx,draw) in enumerate(zip(rows,u['contexts'],u['draws'])):
   row['audio_draw_context']=ctx;row['augmentation_role']='student_tr' if row['language']=='tr' else 'replay'
   ref=trmap[draw['ordinal']] if row['language']=='tr' else rpmap[row['source_id']]
   if row['language']!='tr':row['clean_target_ids']=ref['full_target_ids']
   require(all(row[k]==v for k,v in ref['source_row'].items()),'Original source values unchanged')
   refs.append(ref)
  require(len(rows)==len(refs)==len(u['contexts'])==len(u['draws']),'Exact diagnostic row cardinality')
  raw_request={'phase':PHASE,'rows':rows,'contexts':u['contexts'],'reference_sources':refs,'cleaner_catalog_identity':plan['cleaner_catalog_identity'],
   'actual_optimizer_updates':0,'source_reference_proofs':{'TR_report':tr_binding,'TR_guard':record(A/'actual_cleaner_TR_references_01_guard/closure.json',guard),
   'replay_report':rp_binding,'replay_guard':record(A/'actual_post20_clean_replay_references_01_guard/closure.json',guard)}}
  proof_request=part/'reference_request.json';write_new(proof_request,raw_request);proof=part/'independent_bound_clean_reference.json';pg=part/'reference_guard'
  admit(part,'bounded_reference',100_000_000)
  child=subprocess.Popen([str(LIGHT),'-B',str(P/'work/research/ASR18_bounded_CPU_phase_v3.py'),'--folder',str(pg),'--seconds','90','--script',str(P/'work/research/bind_actual_clean_feature_audio_proof_CPU_v1.py'),'--','--request',str(proof_request),'--output',str(proof)])
  code=child.wait(timeout=100);require(code==0,'Actual independent clean-reference child closed0');gc=actual_guard(pg/'closure.json')
  closed=part/'actual_bound_clean_reference_CLOSED0.json';write_new(closed,{'kind':'FastMSS_independent_clean_audio_reference_actual_closure_v1','report':record(proof,guard),'actual_exit_code':0,'parent_observed_wait':True,'native_pid':gc['pid'],'guard_closure':record(pg/'closure.json',guard)})
  clean_request={'kind':KIND,'phase':PHASE,'update_index':index,'batch_id':u['batch_id'],'rows':rows,'contexts':u['contexts'],'ordered_draws':u['draws'],
   'independent_clean_audio_proof':record(proof,guard),'independent_clean_audio_actual_closure':record(closed,guard),'cleaner_catalog_identity':plan['cleaner_catalog_identity'],
   'augmentation_identity':original['augmentation_identity'],'frontend':original['frontend'],'native_transaction_sha256':u['expected_native_transaction_sha256'],
   'before_sampler_sha256':u['expected_before_sampler_sha256'],'after_sampler_sha256':u['expected_after_sampler_sha256'],
   'prepared_is_committed':False,'sampler_journal_frontier_advanced':False,'feature_sink_source':file_binding(S/'fastmss_clean_feature_window_v1.py'),'feature_cache_owner':owner}
  cr=part/'clean_contract_request.json';write_new(cr,clean_request)
  request=part/'frontend_request.json';write_new(request,{'kind':'FastMSS_actual_CPU_clean_feature_request_v1','owner':owner,'update_index':index,'prior_closures':closures,'clean_contract_request':record(cr,guard)})
  admit(part,'original_frontend700MB',700_000_000);fg=part/'frontend_guard'
  child=subprocess.Popen([str(CPU),'-B',str(P/'work/research/ASR18_bounded_CPU_phase_v3.py'),'--folder',str(fg),'--seconds','180','--script',str(P/'work/research/produce_actual_clean_feature_update_CPU_v1.py'),'--','--request',str(request),'--output',str(part/'producer_report.json')])
  try:
   while child.poll() is None:guard();time.sleep(.25)
  finally:
   if child.poll() is None:
    write_new(fg/'stop.json',{'reason':'coordinator_guard_closed_without_retry'});child.wait(timeout=5)
  closure=observe_feature_child(child,owner_binding=owner,index=index,proof_path=part/'actual_feature_CLOSED0.json',guard=guard)
  require(child.returncode==0,'Actual original clean frontend child must close0');fc=actual_guard(fg/'closure.json');closures.append(closure)
  parts.append({'index':index,'actual_feature_closure':closure,'guard_closure':record(fg/'closure.json',guard),'clean_reference_actual_closure':record(closed,guard)})
 manifest=finalize_window(owner,closures,guard=guard)
 reader=FeatureWindowReader(owner,manifest,guard=guard)
 for index,u in enumerate(plan['updates']):
  for j,c in enumerate(u['contexts']):
   value=reader.take_row(index,j,context=AugmentationDrawContext(**c),native_transaction_sha256=u['expected_native_transaction_sha256'],before_sampler_sha256=u['expected_before_sampler_sha256']);del value
  payload=reader.prepared_payload(index);require(all(v['augmentation']['applied'] is False for v in payload['rows']),'All actual cleaner sources/replay bypass transforms')
 for s in sources:require(record(s['path'],guard)==s,'Admitted CPU source changed')
 final=read_bound(manifest,guard)
 write_new(out/'report.json',{'passed':True,'kind':'FastMSS_actual_original_cleaner_CPU_feature_window_v1','plan':plan_binding,'owner':owner,'manifest':manifest,'source_bindings':sources,
  'updates':parts,'aggregate_feature_window_bytes':final['aggregate_files_bytes'],'actual_closed_feature_children':len(closures),
  'actual_original_CPU_frontend_executed':True,'actual_original_cleaner_and_genuine_replay_sources':True,'augmentation_or_weak_PCM_transport_invoked':False,
  'actual_optimizer_updates':0,'authoritative_frontiers_advanced':False,'full_cleaner_or_broad_corpus_passes':0,'production_phase_transition_claimed':False,
  'feature_window_retained_for_native_owner':True,'native_cleaner_consumer_validated':False,'production_enabled':False,'wall_seconds':time.monotonic()-began})
 print(json.dumps({'passed':True,'manifest':manifest,'bytes':final['aggregate_files_bytes'],'actual_frontend_children':len(closures)}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
