import ast,copy,difflib,hashlib,json,os
from pathlib import Path
import psutil
F=Path('E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1/resident_RAM_two_logical_packet_lookahead_v1')
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def write(p,v):
 with Path(p).open('x',encoding='utf-8')as f:json.dump(v,f,indent=2);f.write('\n')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dead(w):
 try:return abs(psutil.Process(w['pid']).create_time()-w['create_time'])>=.001
 except psutil.NoSuchProcess:return True
out=F/'finite_transport_actual_CPU_01';external=read(F/'finite_transport_actual_CPU_01_external_result.json');entry=read(F/'finite_transport_actual_CPU_01_external_entry.json')
tool={'kind':'ASR18_actual_finite_two_logical_packet_CPU01_TOOL_outcome_v1','actual_TOOL_exit_code':1,'actual_chunk_id':'5f5bde',
 'actual_external_Popen_wait_code':external['actual_Popen_wait_code'],'external_result':pin(F/'finite_transport_actual_CPU_01_external_result.json'),
 'actual_control_passed':False,'no_retry_or_production_change':True,'no_numeric_exit_inferred':True}
write(F/'finite_transport_actual_CPU_01_TOOL_outcome.json',tool)
forms=[entry['external'],entry['external_launcher']]+[x['identity']for x in external['actual_owned_identities']]
for path in [out/'entry.json',out/'ordered_two_plans/synthetic_worker_entry.json',out/'ordered_two_plans/actual_CPU_ready.json']:
 if path.exists():
  v=read(path)
  if 'producer_identity'in v:forms.append(v['producer_identity'])
  elif 'parent'in v:forms.extend([v['parent'],v['launcher']])
  else:forms.append(v)
unique=[];seen=set()
for w in forms:
 key=(w['pid'],w['create_time'])
 if key not in seen:seen.add(key);unique.append({'identity':w,'exact_identity_dead':dead(w)})
assert all(x['exact_identity_dead']for x in unique)
files=[F/'finite_transport_actual_CPU_01_TOOL_outcome.json',F/'finite_transport_actual_CPU_01_external_result.json',
 F/'finite_transport_actual_CPU_01_external_entry.json',F/'finite_transport_actual_CPU_01_external_launch.json',F/'finite_transport_actual_CPU_01_stdout.log']
files.extend(x for x in out.rglob('*')if x.is_file())
closure={'kind':'ASR18_actual_failed_finite_two_logical_packet_CPU01_exact_death_closure_v1','actual_control_passed':False,
 'actual_TOOL_exit_code':1,'actual_external_Popen_wait_code':1,'external_guard_error':external['external_guard_error'],
 'external_required_targeted_cleanup':external['targeted_cleanup'],'sampled_family_peak_bytes':external['sampled_external_plus_owned_family_peak_bytes'],
 'recorded_identity_forms_and_exact_deaths':unique,'all_recorded_exact_dead':True,
 'failure':external['failure'],'report':None,'inputs_before_after_exact':external['inputs_before_and_after_exact'],
 'evidence_pins':[pin(x)for x in files],
 'closure_limit':'Abnormal fixture abort path returned without an abort-failure file; no separate successful child-handle or thread receipt was persisted, so no normal/clean transport proof is claimed.',
 'cause_limit':'Actual remaining object types/retainer graph were not recorded. Static observer reads caller.f_locals while arrays grows, which can retain that list through CPython cached locals after original del. This is a concrete observer defect to remove, not an actual recorded retainer graph.',
 'production_or_sources_changed':False,'retry_authorized':False}
write(F/'finite_transport_actual_CPU_01_exact_death_closure.json',closure)
source=F/'verify_two_logical_packet_one_receiver_CPU_v1.py';raw=source.read_bytes();text=raw.decode().replace('\r\n','\n');tree=ast.parse(text)
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='Harness');node=next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='_receive_profile')
lines=text.splitlines(True);old=''.join(lines[node.lineno-1:node.end_lineno]);new='''    def _receive_profile(self,frame,kind,arg):
        caller=frame.f_back
        if caller is None or caller.f_code is not transport.ResidentFeatureProducer._receive.__code__:return
        # _header returns descriptors/metadata ONLY. Never read caller.f_locals:
        # its cached mapping would itself retain the receiver's growing arrays list.
        if frame.f_code is transport._header.__code__ and kind=='return':
            self.observed_header=None;self.observed_payload_spec_index=0
            if isinstance(arg,dict)and arg.get('kind')=='update':
                self.observed_header={k:arg[k]for k in ('plan_sha256','update_index','token','array_bytes')}
                self.observed_header['descriptor_bytes']=[x['bytes']for x in arg['array_descriptors']]
            return
        if frame.f_code is not transport._read_exact.__code__:return
        header=self.observed_header
        require(header is not None,'Actual payload read follows observed real header metadata')
        if kind=='call':
            count=frame.f_locals['count'] # New _read_exact frame has no allocated data yet.
            require(count==header['descriptor_bytes'][self.observed_payload_spec_index]
                and self.used_bytes==header['array_bytes']and not self.live and not self.packets,
                'Actual receiver FULL byte reservation precedes first partial allocation; LIVE1 including partial')
            self.receiver_raw_active+=1;self.receiver_raw_peak=max(self.receiver_raw_peak,self.receiver_raw_active)
            require(self.receiver_raw_active==1,'No second receiver raw allocation/read overlaps')
            self.receiver_trace.append({'kind':'before_actual_raw_allocation','plan':header['plan_sha256'],
                'index':header['update_index'],'token':header['token'],'read_bytes':count,
                'full_packet_reserved_bytes':self.used_bytes,'live_packets_before_allocation':len(self.live),
                'monotonic':time.monotonic()})
        elif kind=='return':
            view=frame.f_locals.get('view') # This returning temporary frame is not retained.
            if view is not None:self.receiver_view_refs.append(weakref.ref(view))
            self.receiver_raw_active-=1;self.observed_payload_spec_index+=1
            require(self.receiver_raw_active==0,'Actual raw read unwound before subsequent allocation')'''
edits=[{'old_literal':old,'new_literal':new,'expected_count':1}]
assert text.count(old)==1;text=text.replace(old,new)
a='        self.delivered_ownership=[]';b=a+';self.observed_header=None;self.observed_payload_spec_index=0';assert text.count(a)==1;text=text.replace(a,b);edits.append({'old_literal':a,'new_literal':b,'expected_count':1})
a="            require(all(r()is None for r in self.required_dead_refs),'Parent AND receiver payload refs dead before actual kernel post')"
b="            retained_types=[type(r()).__name__for r in self.required_dead_refs if r()is not None]\n            require(not retained_types,'Parent AND receiver payload refs dead before actual kernel post: '+repr(retained_types))"
assert text.count(a)==1;text=text.replace(a,b);edits.append({'old_literal':a,'new_literal':b,'expected_count':1})
ast.parse(text);v2=F/'verify_two_logical_packet_one_receiver_CPU_v2.py';assert not v2.exists();v2.write_bytes(text.replace('\n','\r\n').encode())
# Exact fixture-only inverse includes the observer replacement and bounded failure detail.
inverse=text
for e in reversed(edits):assert inverse.count(e['new_literal'])==e['expected_count'];inverse=inverse.replace(e['new_literal'],e['old_literal'])
assert inverse.replace('\n','\r\n').encode()==raw
ledger={'old_source':pin(source),'new_source':pin(v2),'edits_LF_normalized_then_emit_CRLF':edits,'full_raw_inverse_exact':True,
 'production_helper_CPU_or_config_edits':False,'actual_v2_control_executed':False}
write(F/'fixture1_to2_receiver_observer_exact_inverse_v1.json',ledger)
diff=F/'verify_two_logical_packet_one_receiver_CPU_1_to2.diff';diff.write_text(''.join(difflib.unified_diff(raw.decode().replace('\r\n','\n').splitlines(True),text.splitlines(True),fromfile=str(source),tofile=str(v2))),encoding='utf-8')
# Freeze an external successor with ONLY fixture path and fresh-output/record prefix changes.
e1=F/'launch_one_reviewed_two_logical_packet_CPU_v1.py';e_raw=e1.read_bytes();e_text=e_raw.decode().replace('\r\n','\n');external_edits=[]
for a,b in [('verify_two_logical_packet_one_receiver_CPU_v1.py','verify_two_logical_packet_one_receiver_CPU_v2.py'),('finite_transport_actual_CPU_01','finite_transport_actual_CPU_02')]:
 count=e_text.count(a);assert count>0;e_text=e_text.replace(a,b);external_edits.append({'old_literal':a,'new_literal':b,'expected_count':count})
ast.parse(e_text);e2=F/'launch_one_reviewed_two_logical_packet_CPU_v2.py';assert not e2.exists();e2.write_bytes(e_text.replace('\n','\r\n').encode())
write(F/'external1_to2_exact_source_changes_v1.json',{'old_source':pin(e1),'new_source':pin(e2),'edits_LF_normalized_CRLF_emitted':external_edits})
handoff={'kind':'ASR18_INACTIVE_fixture2_receiver_observer_revision_after_actual_failed01_v1',
 'actual_failed01_closure':pin(F/'finite_transport_actual_CPU_01_exact_death_closure.json'),'failed01_TOOL':pin(F/'finite_transport_actual_CPU_01_TOOL_outcome.json'),
 'fixture_source':pin(v2),'external_launcher_source':pin(e2),'fixture_diff':pin(diff),'exact_raw_inverse':pin(F/'fixture1_to2_receiver_observer_exact_inverse_v1.json'),
 'external_edit_ledger':pin(F/'external1_to2_exact_source_changes_v1.json'),
 'helper_CPU_candidate_unchanged':[pin(F/'asr18_owned_packet_credit_v2.py'),pin(Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20/outputs/parakeet_turkish/fastmss_resident_feature_stream_v11.py')),pin(F/'proposed_configuration_v1.json')],
 'new_fresh_output':str(F/'finite_transport_actual_CPU_02'),'actual_v2_control_execution_authorized':False,
 'same_internal_external_family_and_all_original_reservation_guards':True,
 'old_actual_failure_preserved_and_not_promoted':True,'new_helper_CPU_or_policy_changes':False,
 'scope':'Only remove observer call-site locals retention, use real _header return descriptors, preserve actual raw allocation/weakref checks; add retained type names if a new assertion fails.'}
write(F/'inactive_fixture2_receiver_observer_revision_handoff_v1.json',handoff)
print(json.dumps({'failed01_closure':pin(F/'finite_transport_actual_CPU_01_exact_death_closure.json'),
 'fixture2':pin(v2),'external2':pin(e2),'handoff':pin(F/'inactive_fixture2_receiver_observer_revision_handoff_v1.json'),'actual_v2_executed':False}))
