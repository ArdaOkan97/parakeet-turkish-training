"""Preparation only; failed01 stays failed, no helper/guard or positive requirement change."""
import ast,hashlib,json,os
from pathlib import Path
F=Path(__file__).parent
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def write(p,v):
 with p.open('x',encoding='utf-8',newline='\n')as f:json.dump(v,f,indent=2);f.write('\n')
assert os.environ['TORCH_EXTENSIONS_DIR'].lower().startswith('e:')
files=[]
def successor(oldname,newname,edits):
 old=F/oldname;new=F/newname;text=old.read_bytes().decode('utf-8');records=[]
 for a,b,n in edits:
  assert text.count(a)==n,(a,text.count(a),n);text=text.replace(a,b);records.append({'old_literal':a,'new_literal':b,'expected_count':n})
 ast.parse(text);inverse=text
 for e in reversed(records):
  assert inverse.count(e['new_literal'])==e['expected_count'];inverse=inverse.replace(e['new_literal'],e['old_literal'])
 assert inverse.encode('utf-8')==old.read_bytes()and not new.exists();files.append((old,new,text.encode('utf-8'),records))
old="  q=ThreadQuery(os.getpid(),thread_data['tid']);queries.append(q);alive=q.sample();release.set();t.join(5);require(not t.is_alive(),'Real thread joined without signal')\n  exited=q.sample();again=q.sample();require(exited['exit_filetime_100ns']>0 and again==exited and exited['creation_filetime_100ns']==alive['creation_filetime_100ns'],'Same retained handle after real thread exit');q.close()"
new="""  q=ThreadQuery(os.getpid(),thread_data['tid']);queries.append(q)
  exit_path=OUT/'retained_thread_exit_observations.jsonl'
  require(not exit_path.exists(),'Fresh bounded exit observation metadata')
  def capture_exit(label):
   observed=q.sample();record={'label':label,'monotonic_ns':time.monotonic_ns(),'actual_same_handle_counter_values':observed}
   with exit_path.open('ab')as f:f.write((json.dumps(record,separators=(',',':'))+'\\n').encode('utf-8'));f.flush()
   exit_observations.append(record);return observed
  alive=capture_exit('before_thread_release');release.set();t.join(5);require(not t.is_alive(),'Real Python thread joined without signal')
  post_join_began=time.monotonic();post_join_polls=0
  while True:
   require(post_join_polls<202,'Bounded same-handle kernel-exit observations')
   exited=capture_exit('post_Python_join_actual_kernel_exit_observation');post_join_polls+=1
   if exited['exit_filetime_100ns']>0:break
   require(time.monotonic()-post_join_began<2,'Actual kernel thread exit must be observed within2s; no inferred exit')
   time.sleep(.01)
  again=capture_exit('stable_requery_after_actual_nonzero_exit_FILETIME')
  require(exited['exit_filetime_100ns']>0 and again==exited and exited['creation_filetime_100ns']==alive['creation_filetime_100ns'],'Same retained handle actual exit identity and stable raw counters');q.close()"""
successor('finite_thread_counter_mapping_fixture_v2.py','finite_thread_counter_mapping_fixture_v3.py',[
 ("OUT=F/'finite_thread_counter_mapping_actual_01'","OUT=F/'finite_thread_counter_mapping_actual_02'",1),
 ("start_handles=ThreadQuery.opened-ThreadQuery.closed_count","start_handles=ThreadQuery.opened-ThreadQuery.closed_count;exit_observations=[]",1),
 (old,new,1),
 ("report.update({'kind':'ASR18_GetThreadTimes_new_toy_finite_CPU_actual_v1'","report.update({'cases':cases,'retained_thread_exit_observations':exit_observations,'exit_observation_snapshot':pin(OUT/'retained_thread_exit_observations.jsonl')if(OUT/'retained_thread_exit_observations.jsonl').exists()else None,'kind':'ASR18_GetThreadTimes_new_toy_finite_CPU_actual_v1'",1)])
successor('launch_finite_thread_counter_mapping_CPU_v2.py','launch_finite_thread_counter_mapping_CPU_v3.py',[
 ("FIXTURE=F/'finite_thread_counter_mapping_fixture_v2.py'","FIXTURE=F/'finite_thread_counter_mapping_fixture_v3.py'",1),
 ("RESULT=F/'finite_thread_counter_mapping_external_result_v1.json'","RESULT=F/'finite_thread_counter_mapping_external_result_v2.json'",1),
 ("OUTPUT=F/'finite_thread_counter_mapping_actual_01'","OUTPUT=F/'finite_thread_counter_mapping_actual_02'",1),
 ("stdout=F/'finite_thread_counter_mapping_stdout.log';stderr=F/'finite_thread_counter_mapping_stderr.log'","stdout=F/'finite_thread_counter_mapping_v2_stdout.log';stderr=F/'finite_thread_counter_mapping_v2_stderr.log'",1)])
for old,new,raw,records in files:
 with new.open('xb')as f:f.write(raw)
ledger={'kind':'ASR18_bounded_postjoin_actual_kernel_exit_fixture_v2_to_v3_raw_inverse_v1','old_actual01_preserved_failed':True,'actual01_failed_conjunct_scalars':'UNRECORDED; no retrospective cause claim','helper_TIDmapping_and_all_guards_unchanged':True,'records':[{'old_source':pin(old),'new_source':pin(new),'replacements':records,'newline_normalization':False,'full_old_raw_byte_inverse':True}for old,new,raw,records in files]}
write(F/'bounded_kernel_exit_observation_revision_v3_exact_raw_inverse.json',ledger)
h=json.loads((F/'finite_thread_counter_mapping_source_handoff_v2.json').read_text(encoding='utf-8'));changes={str(old.resolve()):new for old,new,raw,records in files}
h['input_pins']=[pin(changes[v['path']])if v['path']in changes else v for v in h['input_pins']]
h['input_pins'].extend(pin(F/name)for name in ['bounded_kernel_exit_observation_revision_v3_exact_raw_inverse.json','finite_thread_counter_mapping_source_handoff_v2.json','finite_thread_counter_mapping_actual01_failed_completion_v1.json','finite_thread_counter_mapping_actual_TOOL_v1.json','finite_thread_counter_mapping_external_result_v1.json','finite_thread_counter_mapping_actual_01/report.json'])
h['fixture_source']=pin(F/'finite_thread_counter_mapping_fixture_v3.py');h['launcher_source']=pin(F/'launch_finite_thread_counter_mapping_CPU_v3.py')
h['output_path']=str((F/'finite_thread_counter_mapping_actual_02').resolve());h['external_result_path']=str((F/'finite_thread_counter_mapping_external_result_v2.json').resolve())
h['exact_future_external_arguments'][0]=str((F/'launch_finite_thread_counter_mapping_CPU_v3.py').resolve())
h['actual01_failure_preserved']={'actual_TOOL_exit':1,'Popen_exit':1,'guard_error':None,'cleanup':[],'all_owned_thread_query_handles_closed':True,'no_toy_or_profile_launched':True,'unknown_failed_conjunct_values':True,'no_automatic_retry':True}
h['only_new_fixture_semantics']=['After real Pythonjoin, query SAME retained handle for actual nonzeroexitFILETIME within<=2s/10ms/max202polls; persist every raw realcreation/exit/user/kernel value before assertions','Require unchanged creation and exact stable post-exit requery as before; no inferred kernel exit','Always retain accumulatedcases and rawexitvalues on failure as well as success','Fresh actual02/externalv2/logv2 only; helper/roles/guards/requirements exact']
h['resources']['post_Python_join_kernel_exit_wait_seconds']=2;h['resources']['kernel_exit_poll_seconds']=.01;h['resources']['kernel_exit_max_polls']=202
write(F/'finite_thread_counter_mapping_source_handoff_v3.json',h)
print(json.dumps({'handoff':pin(F/'finite_thread_counter_mapping_source_handoff_v3.json'),'ledger':pin(F/'bounded_kernel_exit_observation_revision_v3_exact_raw_inverse.json'),'sources':[pin(new)for old,new,raw,records in files],'no_fixture_or_helper_import_or_execution':True}))
