"""Preparation-only successor. Preserve every v1 unexecuted byte; no imports of proof code."""
import ast,hashlib,json,os
from pathlib import Path
F=Path(__file__).parent
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def write(p,v):
 with p.open('x',encoding='utf-8',newline='\n')as f:json.dump(v,f,indent=2);f.write('\n')
assert os.environ['TORCH_EXTENSIONS_DIR'].lower().startswith('e:')
bundles=[]
def successor(oldname,newname,replacements):
 old=F/oldname;new=F/newname;raw=old.read_bytes();text=raw.decode('utf-8');ledger=[]
 for a,b,n in replacements:
  assert text.count(a)==n,(oldname,a,text.count(a),n)
  text=text.replace(a,b);ledger.append({'old_literal':a,'new_literal':b,'expected_count':n})
 ast.parse(text);inverse=text
 for e in reversed(ledger):
  assert inverse.count(e['new_literal'])==e['expected_count'],('inverse',oldname,e)
  inverse=inverse.replace(e['new_literal'],e['old_literal'])
 assert inverse.encode('utf-8')==raw,'Full old raw byte inverse'
 assert not new.exists();bundles.append((old,new,text.encode('utf-8'),ledger))
successor('owned_thread_times_v1.py','owned_thread_times_v2.py',[
 (r"m=re.fullmatch(r'Thread (\d+)(?: \((.*)\))?',name)",r'''m=re.fullmatch(r'Thread ([1-9][0-9]*)(?: "([^"]*)")?',name)''',1),
 ("if name_ok or frame_ok:roles.append(role)","if name_ok and(not spec.get('functions')or frame_ok):roles.append(role)",1),
 ("self.active={};self.records=[];self.identities={};self.polls=0;self.closed=False;self.failures=[]","self.active={};self.records=[];self.identities={};self.polls=0;self.closed=False;self.failures=[];self.poll_brackets=[]",1),
 ("  self.records.extend({'poll':self.polls,'monotonic_ns':now,**v}for v in observed)\n  return observed","  self.records.extend({'poll':self.polls,'monotonic_ns':now,**v}for v in observed)\n  self.poll_brackets.append({'poll':self.polls,'start_monotonic_ns':now,'end_monotonic_ns':time.monotonic_ns(),'thread_records':len(observed)})\n  return observed",1)])
successor('toy_native_threads_v1.py','toy_native_threads_v2.py',[
 ('from owned_thread_times_v1 import ThreadQuery','from owned_thread_times_v2 import ThreadQuery',1)])
retained_old="  retained=[ThreadQuery(native['pid'],actual[k]['native_tid'],actual[k]['creation_filetime_100ns'])for k in ('main','scalar_worker')];queries.extend(retained)"
retained_new="  retained=[]\n  for k in ('main','scalar_worker'):\n   q=ThreadQuery(native['pid'],actual[k]['native_tid'],actual[k]['creation_filetime_100ns']);queries.append(q);retained.append(q)"
old_close="  own.close();negative(cases,'closed_handle_rejected_before_query',own.sample);negative(cases,'double_close_rejected',own.close)"
new_close=old_close+"\n  baseline=ThreadQuery.opened-ThreadQuery.closed_count;constructor_owned=[];constructor_failed=False\n  try:\n   for expected in (own.creation,own.creation+1):\n    q=ThreadQuery(os.getpid(),threading.get_native_id(),expected);constructor_owned.append(q)\n  except ValueError:constructor_failed=True\n  finally:\n   for q in constructor_owned:\n    if not q.closed:q.close()\n  require(constructor_failed and ThreadQuery.opened-ThreadQuery.closed_count==baseline,'Real second-constructor failure leaves no first-query or failed-constructor handle')\n  cases.append({'case':'real_second_constructor_failure_incremental_ownership_cleanup','passed':True})"
roles_old="roles={'main':{'profile_name':'MainThread'},'scalar_worker':{'profile_name':'ToyScalarWorker','file':str(TOY.resolve()),'functions':['toy_scalar_worker']}}"
roles_new="roles={'main':{'profile_name':'MainThread','file':str(TOY.resolve()),'functions':['main']},'scalar_worker':{'profile_name':'ToyScalarWorker','file':str(TOY.resolve()),'functions':['toy_scalar_worker']}}"
unknown_old="  unknown=profile_role_mapping([{'name':'unsupported-label','sampled_frames':[]}],summary,role_specs=roles);require(unknown['unknown']and not unknown['mapped'],'Unknown profile label retained');cases.append({'case':'unknown_label_explicit','passed':True})"
unknown_new=unknown_old+"\n  for label in ('Thread 0x1234 \\\"MainThread\\\"','Thread 1234 (MainThread)'):\n   unknown=profile_role_mapping([{'name':label,'sampled_frames':[]}],summary,role_specs=roles)\n   require(unknown['unknown']and not unknown['mapped'],'Hex PythonID or old parentheses not normalized to nativeTID')\n  cases.append({'case':'hex_PythonID_and_parenthesized_labels_unknown','passed':True})"
successor('finite_thread_counter_mapping_fixture_v1.py','finite_thread_counter_mapping_fixture_v2.py',[
 ('from owned_thread_times_v1 import','from owned_thread_times_v2 import',1),
 ("TOY=F/'toy_native_threads_v1.py'","TOY=F/'toy_native_threads_v2.py'",1),
 (retained_old,retained_new,1),(old_close,new_close,1),(roles_old,roles_new,1),
 ("+' (MainThread)'","+' \\\"MainThread\\\"'",1),
 (unknown_old,unknown_new,1),
 ("'sampler_open_races':sampler.failures","'sampler_open_races':sampler.failures,'counter_poll_brackets':sampler.poll_brackets,'counter_sample_stamp_is_poll_start_not_simultaneous_thread_query':True",1)])
successor('launch_finite_thread_counter_mapping_CPU_v1.py','launch_finite_thread_counter_mapping_CPU_v2.py',[
 ("FIXTURE=F/'finite_thread_counter_mapping_fixture_v1.py'","FIXTURE=F/'finite_thread_counter_mapping_fixture_v2.py'",1),
 ("began=time.monotonic();popen=None;popen_identity=None;observed={};cleanup=[];guard_error=None;peak=0;initial=None;last=None","began=time.monotonic();popen=None;popen_identity=None;observed={};cleanup=[];guard_error=None;peak=0;initial=None;last=None\n    exact_external_interpreter_launcher=identity(me.parent())",1),
 ("family=[me]","require(same_alive(exact_external_interpreter_launcher),'Exact live external interpreter launcher included conservatively in family')\n        family=[me,psutil.Process(exact_external_interpreter_launcher['pid'])]",1),
 ("remember(proc);sample=sample_memory(proc.pid,'finite_thread_counter_mapping_external_CPU')","if proc.pid!=exact_external_interpreter_launcher['pid']:remember(proc)\n                sample=sample_memory(proc.pid,'finite_thread_counter_mapping_external_CPU')",1)])
successor('observe_existing_source31_thread_CPU_and_stacks_v1.py','observe_existing_source31_thread_CPU_and_stacks_v2.py',[
 ('from owned_thread_times_v1 import','from owned_thread_times_v2 import',1),
 ('from finite_thread_counter_mapping_fixture_v1 import','from finite_thread_counter_mapping_fixture_v2 import',1),
 ("pin(F/'owned_thread_times_v1.py')","pin(F/'owned_thread_times_v2.py')",1),
 ("me=psutil.Process();me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);began=time.monotonic()","me=psutil.Process();me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);began=time.monotonic();exact_external_interpreter_launcher=identity(me.parent())",1),
 ("family=[me]","require(exact(exact_external_interpreter_launcher),'Exact observer interpreter-launcher parent sampled in family')\n  family=[me,psutil.Process(exact_external_interpreter_launcher['pid'])]",1),
 ("who=remember(proc);require(who['pid']not in PROTECTED,'Owned observer/profiler family only')","who=identity(proc)\n    if proc.pid!=exact_external_interpreter_launcher['pid']:remember(proc)\n    require(who['pid']not in PROTECTED,'Owned observer/profiler family only')",1),
 ("'main':{'profile_name':'MainThread'}","'main':{'profile_name':'MainThread','file':str((P/'work/research/run_ASR18_BPE9223_resident_RAM_training_v31.py').resolve()),'functions':['main']}",1),
 ("'sampler_open_or_exit_races':sampler.failures","'sampler_open_or_exit_races':sampler.failures,'counter_poll_brackets':sampler.poll_brackets,'counter_sample_stamp_is_poll_start_not_simultaneous_thread_query':True",1)])
# All assertions/AST/full inverses precede every new immutable write.
for old,new,raw,ledger in bundles:
 with new.open('xb')as f:f.write(raw)
records=[{'old_source':pin(old),'new_source':pin(new),'replacements':ledger,'newline_normalization':False,'entire_old_raw_bytes_reconstructed':True}for old,new,raw,ledger in bundles]
write(F/'quoted_nativeTID_incremental_query_ownership_exact_raw_inverse_v2.json',{'kind':'ASR18_thread_CPU_unexecuted_v1_to_v2_raw_inverse_v1','old_versions_unexecuted_preserved':True,'records':records,'fixture_or_helper_imported':False})
h=json.loads((F/'finite_thread_counter_mapping_source_handoff_v1.json').read_text(encoding='utf-8'))
old_input_paths={v['path']for v in h['input_pins']};changed={str(old.resolve()):new for old,new,raw,ledger in bundles}
h['input_pins']=[pin(changed[v['path']])if v['path']in changed else v for v in h['input_pins']]
h['input_pins'].append(pin(F/'quoted_nativeTID_incremental_query_ownership_exact_raw_inverse_v2.json'))
h['input_pins'].append(pin(F/'finite_thread_counter_mapping_source_handoff_v1.json'))
h['fixture_source']=pin(F/'finite_thread_counter_mapping_fixture_v2.py');h['launcher_source']=pin(F/'launch_finite_thread_counter_mapping_CPU_v2.py')
h['exact_future_external_arguments'][0]=str((F/'launch_finite_thread_counter_mapping_CPU_v2.py').resolve())
h['unexecuted_v1_corrections']=['Pinned py-spy0.4.2 quoted decimal nativeTID parser; hex Python-ID fallback and old parentheses unknown','Role requires exact name and source function when a source spec is supplied','Incremental retained-query ownership plus actual second-constructor-failure closure negative','Exact external interpreter-launcher parent included in sampled512MiB, excluded from descendant waits/termination']
h['live_source_inactive_only']=pin(F/'observe_existing_source31_thread_CPU_and_stacks_v2.py')
h['resources']['external_interpreter_launcher_parent_included_in_family']=True
h['scope_schema_required_completed_CPU_review']={'accepted':True,'reviewer':'/root','scope':'ASR18_GetThreadTimes_new_toy_nativeTID_mapping_actual_CPU_v1','thread_counter_helper_source':pin(F/'owned_thread_times_v2.py'),'actual_CPU_control_report':'actual immutable report pin only after genuine proof','all_exact_metadata_identities_dead':True}
write(F/'finite_thread_counter_mapping_source_handoff_v2.json',h)
print(json.dumps({'handoff':pin(F/'finite_thread_counter_mapping_source_handoff_v2.json'),'ledger':pin(F/'quoted_nativeTID_incremental_query_ownership_exact_raw_inverse_v2.json'),'sources':[pin(new)for old,new,raw,ledger in bundles],'all_v1_sources_preserved_unexecuted':True}))
