"""Inactive source assembly only; never imports helper/controller or attaches."""
import ast,hashlib,json,os
from pathlib import Path
F=Path(__file__).resolve().parent;FIX=F.parent;N=FIX.parent
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish'
FAIL=N/'resident_RAM_original_interval170469_failure_actual_v1';B=Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01')
assert Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad'))
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 with Path(p).open('x',encoding='utf-8',newline='\n')as f:json.dump(v,f,indent=2);f.write('\n')
def successor(oldname,newname,edits):
 old=F/oldname;new=F/newname;raw=old.read_bytes();value=raw.decode();ledger=[]
 for before,after,count in edits:
  before=before.replace('\n','\r\n');after=after.replace('\n','\r\n')
  actual=value.count(before);assert actual>0 and(count is None or actual==count),(before[:140],actual,count)
  value=value.replace(before,after);ledger.append({'old_literal':before,'new_literal':after,'expected_count':actual})
 ast.parse(value);inverse=value
 for e in reversed(ledger):
  assert inverse.count(e['new_literal'])==e['expected_count'],('inverse',e['new_literal'][:120])
  inverse=inverse.replace(e['new_literal'],e['old_literal'])
 assert inverse.encode()==raw and not new.exists();new.write_bytes(value.encode())
 return {'old_source':pin(old),'new_source':pin(new),'newline_normalization':'NONE; original CRLF retained',
  'replacements_in_forward_order':ledger,'entire_RAW_forward_and_inverse_exact':True}
gate=""" controller_pin=review['completed_fixed_counter_controller_CPU_review'];controller_proof=read(controller_pin['path'])
 require(pin(controller_pin['path'])==controller_pin and controller_proof['accepted']is True
  and controller_proof['reviewer']=='/root'
  and controller_proof['scope']=='ASR18_fixed60_counter_window_persistence_METHOD_STUB_actual_CPU_v1'
  and controller_proof['controller_source']==pin(F/'fixed_counter_window_and_scalar_files_v1.py')
  and pin(controller_proof['actual_CPU_report']['path'])==controller_proof['actual_CPU_report']
  and controller_proof['all_exact_metadata_identities_dead']is True,
  'Genuine completed fixed60/persistence METHOD_STUB ROOT gate; no live or timing success inferred')
 require(review['CPU_source']==pin(S/'fastmss_resident_feature_stream_v13.py')
  and review['runtime_configuration']==pin(I/'configuration.json'), 'Exact current CPU13 and immutable source33 runtime')
 require(all(type(w['pid'])is int for w in review['protected_production_identities'])
  and {TARGET['pid'],TARGET['ppid'],30732,26892}.issubset({w['pid']for w in review['protected_production_identities']}),
  'ROOT-bound current production family and reader are never owned cleanup targets')
 PROTECTED.update(w['pid']for w in review['protected_production_identities'])
"""
old_duplicate="""     for row in sampler.records[before_count:]:
      key=(row['native_tid'],row['creation_filetime_100ns'])
      if key in lifetime_queries or any(v['native_tid']==key[0]and v['creation_filetime_100ns']==key[1]for v in lifetime_acquire_errors):continue
      try:
       q=ThreadQuery(TARGET['pid'],key[0],key[1]);lifetime_queries[key]=q # Incremental ownership, retain to actual profiler end.
      except(OSError,ValueError)as e:lifetime_acquire_errors.append({'native_tid':key[0],'creation_filetime_100ns':key[1],'type':type(e).__name__,'message':str(e)})
      require(len(lifetime_queries)+len(lifetime_acquire_errors)<=256,'Original bounded total identities; no extra unbounded handle history')"""
capture="""   if phase!='counter_window'and not lifetime_captured:
    lifetime_queries={(tid,q.creation):q for tid,q in sampler.active.items()}
    lifetime_captured=True
    alignment['existing_sampler_handles_retained_at_counter_end_monotonic_ns']=time.monotonic_ns()
    require(len(lifetime_queries)<=256 and ThreadQuery.opened-ThreadQuery.closed_count<=256,
     'Only original sampler handles retained through actual profiler end; no duplicate OpenThread')
"""
tail_function="""
def capture_current_timing_tail(output):
 path=I/'private_metadata_window_timings.jsonl';size=path.stat().st_size;start=max(0,size-65536)
 with path.open('rb')as f:f.seek(start);raw=f.read(size-start)
 if start:
  first=raw.find(b'\\n');require(first>=0,'Bounded complete timing-tail row');start+=first+1;raw=raw[first+1:]
 if not raw.endswith(b'\\n'):
  last=raw.rfind(b'\\n');require(last>=0,'Preserve only complete existing timing rows');raw=raw[:last+1]
 lines=raw.splitlines(keepends=True);require(len(lines)>=5,'Five complete late existing timing rows')
 selected=b''.join(lines[-5:]);selected_start=start+len(raw)-len(selected)
 rows=[json.loads(line)for line in lines[-5:]]
 snapshot=output/'trigger_existing_last5_timing_rows.jsonl'
 with snapshot.open('xb')as f:f.write(selected)
 return {'source_path':str(path),'source_size_observed':size,'start_offset':selected_start,
  'end_offset':selected_start+len(selected),'bytes':len(selected),'sha256':hashlib.sha256(selected).hexdigest(),
  'snapshot':pin(snapshot),'rows':rows,'selection':'last5 complete existing rows from <=64KiB after actual173020',
  'no_new_timing_span_benchmark_or_cause_or_speed_claim':True}
"""
observer_edits=[
 ('source31','source33',None),('owner_000023','owner_000024',1),
 ("TARGET={'pid':2456,'create_time':1792752131.0863705,'ppid':41672}","TARGET={'pid':3652,'create_time':1792764340.2396317,'ppid':31268}",1),
 ('4d73b710332552cbbad10246f1fe07a627fa0690b9657f4c61a6cfac20160bc1','eaa9da9d88cdc7086b2d77ee2fe52a8b36aa9dfcfe8ef20ed3678298c3a030b6',1),
 ('d62979c101b8cc4e5c95d4930c6f2a27d5333b893363db8b191505148f3cb7ed','5428362f43aeb4bda2a9591524ff5f447a9e81267dc45efe5c68c16410fce5ee',1),
 ('PROTECTED={2456,38964,26892,41672,34044}','PROTECTED={3652,31268,30732,40192,24268,26892}',1),
 ('Same exact native2456 must remain alive','Same exact native3652 must remain alive',1),
 ('run_ASR18_BPE9223_resident_RAM_training_v31.py','run_ASR18_BPE9223_resident_RAM_training_v33.py',None),
 ('Exact original source33/I23/native/runtime','Exact current source33/I24/native/runtime',1),
 ("def io_sample():return dict(psutil.Process(TARGET['pid']).io_counters()._asdict())", "def io_sample():return dict(psutil.Process(TARGET['pid']).io_counters()._asdict())\n"+tail_function,1),
 (" inputs=review['input_pins'];require",gate+" inputs=review['input_pins'];require",1),
 ("lifetime_at_profile_end=[]","lifetime_at_profile_end=[];lifetime_captured=False;peak_query_handles=0",1),
 ('  nonlocal peak,last','  nonlocal peak,last,peak_query_handles',1),
 ("  peak=max(peak,total);require(total<=LIMIT,'Total owned observer/profiler family512MiB')",
  "  peak=max(peak,total);require(total<=LIMIT,'Total owned observer/profiler family512MiB')\n  peak_query_handles=max(peak_query_handles,ThreadQuery.opened-ThreadQuery.closed_count)\n  require(0<=ThreadQuery.opened-ThreadQuery.closed_count<=256,'Original total256 owned query-handle bound')",1),
 (" initial=guard(True);before_health=health();OUT.mkdir();", " initial=guard(True);before_health=health()\n require(before_health['progress']['actual_optimizer_updates']>=173020 and before_health['progress']['last_saved_updates']>=173000,\n  'Sample only after actual scheduled172k DEV plus1020 and natural173000/post173020; no startup/postDEV-warm attribution')\n OUT.mkdir();trigger_tail=capture_current_timing_tail(OUT);",1),
 ("   if phase=='done':break",capture+"   if phase=='done':break",1),
 ("'counter_record_limit':80000}","'counter_record_limit':80000,'live_query_handles':ThreadQuery.opened-ThreadQuery.closed_count,'max_live_query_handles':256,'conservative_new_TIDs':len(set(tids)-set(sampler.active))}",1),
 ("    before_count=len(sampler.records);query_start=time.monotonic()", "    require(ThreadQuery.opened-ThreadQuery.closed_count+len(set(tids)-set(sampler.active))<=256,\n     'Conservative before-poll original256 total handle budget, vector already durable')\n    before_count=len(sampler.records);query_start=time.monotonic()",1),
 (old_duplicate,"     require(ThreadQuery.opened-ThreadQuery.closed_count<=256,'Original total handle bound after original sampler poll')",1),
 ("'CPU_counter_and_profiler_windows_are_different':True", "'trigger_existing_timing_tail':trigger_tail,'retained_lifetime_handle_origin':'EXISTING_sampler.active_references_NO_duplicate_OpenThread','peak_total_owned_query_handles':peak_query_handles,'completed_fixed_counter_controller_CPU_review':controller_pin,'CPU_counter_and_profiler_windows_are_different':True",1)]
ol=successor('observe_existing_source31_thread_CPU_and_stacks_v3.py','observe_existing_source33_thread_CPU_and_stacks_v1.py',observer_edits)
external_edits=[('source31','source33',None),
 ("FIXTURE=F/'observe_existing_source33_thread_CPU_and_stacks_v3.py'","FIXTURE=F/'observe_existing_source33_thread_CPU_and_stacks_v1.py'",1),
 ("LIMIT=536870912;PREFETCH=268435456","LIMIT=536870912;PREFETCH=268435456\nPROTECTED={3652,31268,30732,40192,24268,26892}",1),
 ("    require(not RESULT.exists()and not OUTPUT.exists()", "    require(all(type(w['pid'])is int for w in review['protected_production_identities']),'Exact ROOT-bound production cleanup exclusions')\n    PROTECTED.update(w['pid']for w in review['protected_production_identities'])\n    require(not RESULT.exists()and not OUTPUT.exists()",1),
 ('(2456,38964,26892,41672,34044,40012,31044)','PROTECTED',2),
 ('existing_source33_thread_CPU_external_result_v2.json','existing_source33_thread_CPU_external_result_v1.json',1),
 ('existing_source33_thread_CPU_v2_stdout.log','existing_source33_thread_CPU_v1_stdout.log',1),
 ('existing_source33_thread_CPU_v2_stderr.log','existing_source33_thread_CPU_v1_stderr.log',1)]
el=successor('launch_existing_source31_thread_CPU_external_v2.py','launch_existing_source33_thread_CPU_external_v1.py',external_edits)
ledger=F/'current_source33_fixed60_existing_handle_live_successor_entire_RAW_inverse_v1.json';write(ledger,{'observer':ol,'external':el})
observer=Path(ol['new_source']['path']);external=Path(el['new_source']['path'])
# All changes are inactive; actual counter/helper/controller/profile code is never imported here.
tree=ast.parse(observer.read_bytes());assert not any(isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='ThreadQuery'for n in ast.walk(tree))
gatepin=pin(F/'ROOT_fixed60_counter_controller_METHOD_STUB_actual_CPU_review_v1.json')
assert gatepin['sha256']=='16e8c070b69caf3d5867ca2144a469daf0f3f4ac65e0795d702560284c3cb349'
inputs=[];seen=set()
def add(p):
 v=pin(p)
 if v['path']not in seen:seen.add(v['path']);inputs.append(v)
for p in [observer,external,ledger,F/'fixed_counter_window_and_scalar_files_v1.py',F/'owned_thread_times_v2.py',
 F/'finite_thread_counter_mapping_fixture_v2.py',F/'finite_thread_counter_mapping_fixture_v3.py',
 F/'ROOT_new_thread_counter_nativeTID_mapping_actual_CPU_review_v1.json',
 F/'ROOT_fixed60_counter_controller_METHOD_STUB_actual_CPU_review_v1.json',F/'fixed_counter_controller_actual_01/report.json',
 F/'fixed_counter_controller_actual_completion_v1.json',F/'ROOT_existing_source31_live_thread_attribution_actual01_failed_closure_review_v1.json',
 FIX/'py_spy_0_4_2_win_amd64_v1/py-spy.exe',FIX/'ROOT_pinned_py_spy_0_4_2_isolated_E_preparation_review_v1.json',
 P/'work/research/run_ASR18_BPE9223_resident_RAM_training_v33.py',S/'fastmss_resident_feature_stream_v13.py',
 S/'asr18_BPE9223_private_scalar_prefetch_v1.py',S/'resource_watchdog_v2.py',B/'invocations/owner_000024/configuration.json',
 FAIL/'ROOT_source33_eighth_original_saved170400_recovered170600_actual_verification_v1.json',
 'E:/gpu_training_scratchpad/storage_config.json','E:/gpu_training_scratchpad/use_storage.ps1']:add(p)
h={'kind':'ASR18_current_source33_fixed60_existing_query_handle_readonly_attribution_inactive_handoff_v1',
 'execution_admitted':False,'requested_scope':'ASR18_existing_source33_GetThreadTimes_nonblocking60s_attribution_v1',
 'live_observer_source':pin(observer),'external_live_launcher_source':pin(external),'input_pins':inputs,
 'target_identity':{'pid':3652,'create_time':1792764340.2396317,'ppid':31268},
 'CPU_source':pin(S/'fastmss_resident_feature_stream_v13.py'),'runtime_configuration':pin(B/'invocations/owner_000024/configuration.json'),
 'completed_new_counter_CPU_review':pin(F/'ROOT_new_thread_counter_nativeTID_mapping_actual_CPU_review_v1.json'),
 'completed_fixed_counter_controller_CPU_review':gatepin,'completed_controller_gate_explicitly_consumed_before_any_attach':True,
 'output_path':str(F/'existing_source33_thread_CPU_actual_02'),'external_result_path':str(F/'existing_source33_thread_CPU_external_result_v1.json'),
 'execution_prerequisite':'actual173020 no earlier: real172000 scheduled phase+1020, natural173000/post173020; ROOT scope binds current5production+reader after phase',
 'trigger_existing_timing_evidence':'ONE <=64KiB current file tail; preserve exactly last5 complete original rows/no new span; absence of late slowdown remains explicit/no automatic repeated sample',
 'protected_production_identities_required_from_future_ROOT_scope_not_selected_here':True,
 'exact_external_arguments':['-Xutf8','-B',str(external),'--ROOT-review','__PENDING_SEPARATE_EXACT_ROOT_SCOPE__'],
 'limits':{'duration_counter_seconds':60,'profiler_nominal_samples':6000,'profiler_rate_Hz':100,'external_wall_seconds':120,
 'sampled_fullfamily_private_bytes':536870912,'full_prospective_bytes':536870912,'admission_commit_bytes':5055306368,
 'admission_physical_bytes':2305306368,'runtime_commit_bytes':2268435456,'runtime_physical_bytes':1768435456,
 'maximum_current_threads':128,'maximum_total_identities':256,'maximum_total_owned_query_handles':256,'poll_limit':700,
 'counter_records':80000,'E_free_bytes':100000000000,'profile_bytes':67108864,'counter_bytes':33554432,
 'stderr_log_bytes':4194304,'stdout_compact_pin_bytes':8388608},
 'no_duplicate_ThreadQuery_constructor_calls_in_observer_AST':True,
 'lifetime_proof':'Retain references to SAME sampler.active handles at counter-window end; sample after actual profiler exit; sampler.close remains handle owner',
 'counter_window_vs_profile_window_distinct_no_exclusive_CPU_by_stack_claim':True,
 'only_target_operation':'NONBLOCKING pinned py-spy read-only allthread60s nominal plus retained-handle GetThreadTimes fixed60s',
 'no_native_filter_GIL_filter_locals_subprocessprofile_suspend_inject_signal_training_model_audio_extraGC_restart':True,
 'unknown_role_exit_reuse_ambiguity_or_nonatomic_errors_preserved':True,
 'no_hard_allocation_fit_speed_cause_success_claim':True,'old_source31_live01_failed_and_unexecuted_live3_preserved':True,
 'same_training_owner_session':65067,'all_training_source_guard_science_environment_reader_unchanged':True,
 'no_automatic_retry':True,'whole_RAW_inverse':pin(ledger)}
write(F/'current_source33_fixed60_existing_handle_live_inactive_handoff_v1.json',h)
print(json.dumps({'observer':pin(observer),'external':pin(external),'inverse':pin(ledger),
 'handoff':pin(F/'current_source33_fixed60_existing_handle_live_inactive_handoff_v1.json'),'inputs':len(inputs),'execution_admitted':False}),flush=True)
