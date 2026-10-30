"""Inactive repair only. No import or execution of sampler/controller/live sources."""
import ast,hashlib,json,os
from pathlib import Path
F=Path(__file__).parent
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def write(p,v):
 with p.open('x',encoding='utf-8',newline='\n')as f:json.dump(v,f,indent=2);f.write('\n')
assert os.environ['TORCH_EXTENSIONS_DIR'].lower().startswith('e:')
pending=[]
def successor(oldname,newname,edits):
 old=F/oldname;new=F/newname;raw=old.read_bytes();text=raw.decode('utf-8');ledger=[]
 for a,b,n in edits:
  assert text.count(a)==n,(oldname,a,text.count(a),n);text=text.replace(a,b);ledger.append({'old_literal':a,'new_literal':b,'expected_count':n})
 ast.parse(text);inverse=text
 for e in reversed(ledger):
  assert inverse.count(e['new_literal'])==e['expected_count'],('inverse',e);inverse=inverse.replace(e['new_literal'],e['old_literal'])
 assert inverse.encode('utf-8')==raw and not new.exists();pending.append((old,new,text.encode('utf-8'),ledger))
old_block="""  sampler=ThreadSampler(TARGET['pid'],max_threads=128,max_identities=256,max_polls=700,max_records=80000)
  next_poll=time.monotonic()
  while child.poll()is None:
   guard();sampler.poll([t.id for t in psutil.Process(TARGET['pid']).threads()]);next_poll+=.1
   time.sleep(max(0,min(.1,next_poll-time.monotonic())))
  code=child.wait(timeout=10);alignment['profiler_exit_observed_monotonic_ns']=time.monotonic_ns();require(code==0,'Actual owned profiler exit0; preserve failures')
  guard();sampler.poll([t.id for t in psutil.Process(TARGET['pid']).threads()]);alignment['final_counter_poll_monotonic_ns']=time.monotonic_ns()
  after_io=io_sample();after_health=health();require(profile.stat().st_size<=MAX_PROFILE,'Bounded64MiB original profile')"""
new_block="""  sampler=ThreadSampler(TARGET['pid'],max_threads=128,max_identities=256,max_polls=700,max_records=80000)
  scalar_files=BoundedScalarFiles(OUT);controller=FixedCounterWindow(time.monotonic())
  alignment['counter_explicit_start_monotonic_ns']=time.monotonic_ns();alignment['counter_intended_deadline_seconds']=controller.deadline
  while True:
   guard();now=time.monotonic();phase=controller.phase(now,child.poll())
   if phase=='done':break
   if phase=='counter_window'and controller.due(now):
    tids=[t.id for t in psutil.Process(TARGET['pid']).threads()]
    vector={'monotonic_ns':time.monotonic_ns(),'attempted_poll':sampler.polls+1,'poll_limit':700,'current_TID_count':len(tids),'unique_TID_count':len(set(tids)),'max_current_threads':128,'total_identity_count':len(sampler.identities),'max_identities':256,'prior_counter_records':len(sampler.records),'counter_record_limit':80000}
    scalar_files.append('preconditions',[vector]) # Flush BEFORE every original helper assertion.
    before_count=len(sampler.records);query_start=time.monotonic()
    if query_start>=controller.deadline:break_counter=True
    else:
     sampler.poll(tids);controller.recorded(query_start)
     scalar_files.append('counters',sampler.records[before_count:]);scalar_files.append('brackets',[sampler.poll_brackets[-1]])
     for row in sampler.records[before_count:]:
      key=(row['native_tid'],row['creation_filetime_100ns'])
      if key in lifetime_queries or any(v['native_tid']==key[0]and v['creation_filetime_100ns']==key[1]for v in lifetime_acquire_errors):continue
      try:
       q=ThreadQuery(TARGET['pid'],key[0],key[1]);lifetime_queries[key]=q # Incremental ownership, retain to actual profiler end.
      except(OSError,ValueError)as e:lifetime_acquire_errors.append({'native_tid':key[0],'creation_filetime_100ns':key[1],'type':type(e).__name__,'message':str(e)})
      require(len(lifetime_queries)+len(lifetime_acquire_errors)<=256,'Original bounded total identities; no extra unbounded handle history')
   if phase=='counter_window':
    time.sleep(controller.sleep_seconds(time.monotonic()))
   else:time.sleep(.05) # Only guarded profiler completion wait; NO new counter acquisition.
  code=child.wait(timeout=10);alignment['profiler_exit_observed_monotonic_ns']=time.monotonic_ns();require(code==0,'Actual owned profiler exit0; preserve failures')
  for key,q in lifetime_queries.items():lifetime_at_profile_end.append({'native_tid':key[0],'creation_filetime_100ns':key[1],'actual_same_retained_handle_at_profile_end':q.sample()})
  alignment['counter_queries_stop_at_fixed_window_not_profiler_exit']=True
  after_io=io_sample();after_health=health();require(profile.stat().st_size<=MAX_PROFILE,'Bounded64MiB original profile')"""
# No further sampling is performed if the per-batch query would start at/after the deadline.
new_block=new_block.replace("if query_start>=controller.deadline:break_counter=True\n    else:","if query_start<controller.deadline:")
mapping_marker="  # Role mapping uses both the actual profile name and the exact source frame when available."
mapping_new="""  invalid_lifetimes={(v['native_tid'],v['creation_filetime_100ns'])for v in lifetime_acquire_errors}
  invalid_lifetimes.update((v['native_tid'],v['creation_filetime_100ns'])for v in lifetime_at_profile_end if v['actual_same_retained_handle_at_profile_end']['exit_filetime_100ns'])
  invalid_lifetimes.update((v['native_tid'],v['creation_filetime_100ns'])for v in summary if v['thread_exited_observed'])
  kept=[]
  for v in mapping['mapped']:
   if(v['native_tid'],v['creation_filetime_100ns'])in invalid_lifetimes:mapping['unknown'].append({'name':v['profile_name'],'reason':'Retained lifetime exited/unproven during expanded profiler interval; no role/CPU attribution'})
   else:kept.append(v)
  mapping['mapped']=kept
"""+mapping_marker
old_write="""  sampler.close();counter=OUT/'thread_CPU_counters.jsonl'
  with counter.open('xb')as f:
   total=0
   for row in sampler.records:
    b=(json.dumps(row,separators=(',',':'))+'\\n').encode('utf-8');total+=len(b);require(total<=MAX_COUNTER_BYTES,'Counter output32MiB cap');f.write(b)"""
new_write="""  sampler.close()
  for q in lifetime_queries.values():
   if not q.closed:q.close()
  scalar_files.close();counter=OUT/'thread_CPU_counters.jsonl'"""
finally_old=""" finally:
  if sampler is not None and not sampler.closed:"""
finally_new=""" finally:
  for q in lifetime_queries.values():
   if not q.closed:
    try:q.close()
    except BaseException as e:cleanup.append({'kind':'retained_lifetime_query_close_error','type':type(e).__name__,'message':str(e)})
  if scalar_files is not None and not scalar_files.closed:
   try:scalar_files.close()
   except BaseException as e:cleanup.append({'kind':'scalar_evidence_file_close_error','type':type(e).__name__,'message':str(e)})
  if sampler is not None and not sampler.closed:"""
update_old="  value.update({'kind':'ASR18_existing_source31_thread_CPU_and_nonblocking_stacks_actual_v1'"
update_new="""  partial_artifacts={}
  for key,(name,cap)in BoundedScalarFiles.NAMES.items():
   p=OUT/name;partial_artifacts[key]=pin(p,cap)if p.exists()else{'absent':True,'path':str(p.resolve())}
  value.update({'partial_scalar_artifacts':partial_artifacts,'counter_partial_summary':sampler.summary()if sampler is not None else[],'counter_partial_poll_brackets':sampler.poll_brackets if sampler is not None else[],'counter_controller_state':None if controller is None else{'explicit_start':controller.start,'intended_deadline':controller.deadline,'recorded_polls':controller.polls,'last_clock_observation':controller.last},'retained_lifetime_acquire_errors':lifetime_acquire_errors,'retained_lifetime_checks_at_profiler_end':lifetime_at_profile_end,'CPU_counter_and_profiler_windows_are_different':True,'no_exclusive_CPU_by_profile_stack_claim':True,'kind':'ASR18_existing_source31_thread_CPU_and_nonblocking_stacks_actual_v1'"""
successor('observe_existing_source31_thread_CPU_and_stacks_v2.py','observe_existing_source31_thread_CPU_and_stacks_v3.py',[
 ('from finite_thread_counter_mapping_fixture_v2 import profile_profiles','from finite_thread_counter_mapping_fixture_v2 import profile_profiles\nfrom fixed_counter_window_and_scalar_files_v1 import FixedCounterWindow,BoundedScalarFiles',1),
 ("OUT=F/'existing_source31_thread_CPU_actual_01'","OUT=F/'existing_source31_thread_CPU_actual_02'",1),
 ("error=None;value={};opened_before=ThreadQuery.opened-ThreadQuery.closed_count","error=None;value={};opened_before=ThreadQuery.opened-ThreadQuery.closed_count;scalar_files=None;controller=None;lifetime_queries={};lifetime_acquire_errors=[];lifetime_at_profile_end=[]",1),
 (old_block,new_block,1),(mapping_marker,mapping_new,1),(old_write,new_write,1),(finally_old,finally_new,1),(update_old,update_new,1)])
successor('launch_existing_source31_thread_CPU_external_v1.py','launch_existing_source31_thread_CPU_external_v2.py',[
 ("FIXTURE=F/'observe_existing_source31_thread_CPU_and_stacks_v2.py'","FIXTURE=F/'observe_existing_source31_thread_CPU_and_stacks_v3.py'",1),
 ("RESULT=F/'existing_source31_thread_CPU_external_result_v1.json'","RESULT=F/'existing_source31_thread_CPU_external_result_v2.json'",1),
 ("OUTPUT=F/'existing_source31_thread_CPU_actual_01'","OUTPUT=F/'existing_source31_thread_CPU_actual_02'",1),
 ("stdout=F/'existing_source31_thread_CPU_stdout.log';stderr=F/'existing_source31_thread_CPU_stderr.log'","stdout=F/'existing_source31_thread_CPU_v2_stdout.log';stderr=F/'existing_source31_thread_CPU_v2_stderr.log'",1)])
successor('launch_finite_thread_counter_mapping_CPU_v3.py','launch_fixed_counter_controller_METHOD_STUB_v1.py',[
 ("FIXTURE=F/'finite_thread_counter_mapping_fixture_v3.py'","FIXTURE=F/'verify_fixed_counter_controller_METHOD_STUB_v1.py'",1),
 ("RESULT=F/'finite_thread_counter_mapping_external_result_v2.json'","RESULT=F/'fixed_counter_controller_external_result_v1.json'",1),
 ("OUTPUT=F/'finite_thread_counter_mapping_actual_02'","OUTPUT=F/'fixed_counter_controller_actual_01'",1),
 ('ASR18_GetThreadTimes_new_toy_nativeTID_mapping_finite_CPU_v1','ASR18_fixed60_counter_window_persistence_METHOD_STUB_CPU_v1',1),
 ("stdout=F/'finite_thread_counter_mapping_v2_stdout.log';stderr=F/'finite_thread_counter_mapping_v2_stderr.log'","stdout=F/'fixed_counter_controller_stdout.log';stderr=F/'fixed_counter_controller_stderr.log'",1),
 ('ASR18_GetThreadTimes_new_toy_finite_CPU_external_result_v1','ASR18_fixed_counter_window_METHOD_STUB_external_result_v1',1)])
for p in [F/'fixed_counter_window_and_scalar_files_v1.py',F/'verify_fixed_counter_controller_METHOD_STUB_v1.py']:ast.parse(p.read_text(encoding='utf-8'))
for old,new,raw,ledger in pending:
 with new.open('xb')as f:f.write(raw)
records=[{'old_source':pin(old),'new_source':pin(new),'replacements':ledger,'newline_normalization':False,'entire_old_raw_byte_inverse':True}for old,new,raw,ledger in pending]
write(F/'fixed_counter_window_live_revision_v3_exact_raw_inverse.json',{'kind':'ASR18_fixed_counter_window_control_only_inactive_raw_inverse_v1','records':records,'original_Win32_helper_and_all_training_sources_unchanged':True,'actual01_failed_bound_conjunct':'UNKNOWN; no retrospective cause claim','no_source_import_or_job_execution':True})
proof=F/'ROOT_new_thread_counter_nativeTID_mapping_actual_CPU_review_v1.json';failure=F/'ROOT_existing_source31_live_thread_attribution_actual01_failed_closure_review_v1.json'
assert pin(proof)['sha256']=='8f2ee5f17afed28f420bdfa131e3e178965e0921bb92d6356d4b6a0caacffd52'and pin(failure)['sha256']=='cbc815a2303e3a52dfb2c344c86533a74348acdc431f50a6a33a86afe27dd8ef'
inputs=[F/'fixed_counter_window_and_scalar_files_v1.py',F/'verify_fixed_counter_controller_METHOD_STUB_v1.py',F/'launch_fixed_counter_controller_METHOD_STUB_v1.py',F/'fixed_counter_window_live_revision_v3_exact_raw_inverse.json',proof,failure,F/'existing_source31_thread_CPU_actual01_failed_completion_v1.json',F/'owned_thread_times_v2.py',Path('E:/gpu_training_scratchpad/storage_config.json'),Path('E:/gpu_training_scratchpad/use_storage.ps1')]
h={'kind':'ASR18_fixed_counter_window_controller_inactive_finite_METHOD_STUB_CPU_handoff_v1','execution_admitted':False,'requested_scope':'ASR18_fixed60_counter_window_persistence_METHOD_STUB_CPU_v1','fixture_source':pin(inputs[1]),'launcher_source':pin(inputs[2]),'input_pins':[pin(p)for p in inputs],'output_path':str((F/'fixed_counter_controller_actual_01').resolve()),'external_result_path':str((F/'fixed_counter_controller_external_result_v1.json').resolve()),'fixed_resource_bounds_unchanged':{'external_wall_seconds':120,'owned_fullfamily_private_cap_bytes':536870912,'full_prospective_bytes':536870912,'admission_commit_minimum':5055306368,'admission_physical_minimum':2305306368,'runtime_commit_minimum':2268435456,'runtime_physical_minimum':1768435456,'E_free_minimum':100000000000,'CPU_mask':-1,'math_caps':1,'priority':'BelowNormal'},'proof_mode':'METHOD_STUB clocks/profiler returncodes/counter values ONLY; actual owned E scalar-file persistence/closure exercised','no_Win32_helper_or_profiler_or_target_or_oldtoy_operation':True,'inactive_live_source':pin(F/'observe_existing_source31_thread_CPU_and_stacks_v3.py'),'inactive_live_external':pin(F/'launch_existing_source31_thread_CPU_external_v2.py'),'live_execution_requires_separate_actual_ROOT_scope_after_controller_proof':True,'new_controller_completed_ROOT_gate_required_schema':{'scope':'ASR18_fixed60_counter_window_persistence_METHOD_STUB_actual_CPU_v1','accepted':True,'reviewer':'/root','controller_source':pin(F/'fixed_counter_window_and_scalar_files_v1.py'),'actual_CPU_report':'actual immutable report pin only after real invocation','all_exact_metadata_identities_dead':True},'live_repair_semantics':['Fixed60s actual counter-acquisition start/deadline independent of profiler nominal6000samples, unchanged700cap','Only resourceguard+wait aftercounterwindow until profiler exits or unchanged120externalguard fires','Every beforepoll precondition vector flushed before helpercall; counters/brackets streamed and failurepreserved','Separate retained lifetime handle per bounded threadidentity held through actual profilerend; actualexit/unproven/reused/ambiguous role becomesUNKNOWN','CPUwindow and expanded profilewindow explicitly DIFFERENT; no exact exclusive CPU-by-stack claim','Freshactual02/external2; original01failedunknowncause remains preserved']}
write(F/'fixed_counter_controller_METHOD_STUB_source_handoff_v1.json',h)
print(json.dumps({'handoff':pin(F/'fixed_counter_controller_METHOD_STUB_source_handoff_v1.json'),'ledger':pin(F/'fixed_counter_window_live_revision_v3_exact_raw_inverse.json'),'sources':[pin(new)for old,new,raw,ledger in pending],'new_controller':pin(inputs[0]),'new_METHOD_STUB_fixture':pin(inputs[1]),'no_import_or_execution':True}))
