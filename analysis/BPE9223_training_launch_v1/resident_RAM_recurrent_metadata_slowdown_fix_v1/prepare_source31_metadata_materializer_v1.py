"""Freeze a metadata-only successor with all output preconditions before writes."""
import ast,difflib,hashlib,json,os
from pathlib import Path
F=Path(__file__).parent
assert os.environ['TORCH_EXTENSIONS_DIR'].lower().startswith('e:')
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
old=F/'materialize_source30_actual_transition_configuration_v2.py';new=F/'materialize_source31_actual_transition_configuration_v1.py'
raw=old.read_bytes();t=raw.decode().replace('\r\n','\n');nl='\r\n'if b'\r\n'in raw else'\n';assert t.replace('\n',nl).encode()==raw;edits=[]
def v(a,b,n=1):
 global t
 assert t.count(a)==n,(a,t.count(a),n);edits.append({'old_literal':a,'new_literal':b,'expected_count':n});t=t.replace(a,b)
v("with Path(p).open('x',encoding='utf-8')as f:","with Path(p).open('x',encoding='utf-8',newline='\\n')as f:")
v('ASR18_canonical_archive_metadata_materialization_v1','ASR18_fresh_signature_collect_metadata_materialization_v1')
v('actual_source29_canonical_archive_checkpoint_transition_01','actual_source30_fresh_signature_collect_checkpoint_transition_01')
v('ASR18_actual_source29_checkpoint_canonical_archive_transition_v1','ASR18_actual_source30_checkpoint_fresh_signature_collect_transition_v1')
v("value['owned_session']==87302", "value['owned_session']==72166")
v('Genuine supported source29 intentional stop','Genuine supported source30 intentional stop')
v('candidate_source30_configuration_completed_CPU_pending_actual_stop_v1.json','candidate_source31_configuration_completed_CPU_pending_actual_stop_v1.json')
v('prior_resident_RAM_canonical_archive_checkpoint_transition','prior_resident_RAM_fresh_signature_collect_checkpoint_transition',3)
v("sourcebase=F.parent/'resident_RAM_two_logical_packet_lookahead_v1/production_configuration_v1/configuration.json'", "sourcebase=F/'production_configuration_v1/configuration.json'")
v("del inverse['prepared_metadata_archive_representation_policy']", "del inverse['fresh_metadata_signature_and_collect_observation_policy']")
v("folder=F/'production_configuration_v1'", "folder=F/'production_configuration_source31_v1'")
v('run_ASR18_BPE9223_resident_RAM_training_v30.py','run_ASR18_BPE9223_resident_RAM_training_v31.py',2)
v("validation={'start_updates':saved+20,'end_updates':max(saved+20+1600,((saved//4000)+1)*4000+220),\n  'formula':'start=restored+20; end=max(start+1600,next_scheduled4000boundary+220)',", "validation={'restored_updates':saved,'next_scheduled_DEV_updates':((saved//4000)+1)*4000,\n  'start_updates':saved+20,'end_updates':max(saved+20+1600,((saved//4000)+1)*4000+1020),\n  'formula':'start=restored+20; end=max(start+1600,nextscheduledDEV+1020)',")
v("'actual_cleanup_boundary_scalar_archives_at_first_and_post_DEV_end_checkpoint':True,", "'no_archive_reads_original_collect_per20_scalars_only':True,\n  'all200_blocks_first400_final400_and_postDEV220_620_1020_blocks':True,\n  'engineering_acceptance':'Near original76.57 inclusive (~75+), save-excluded first-final sustained decline<=5%, no guard/science/state/checkpoint/reader regression. All pauses retained and any DEV-containing first400 explicitly labelled; partial gain keeps human fix open.',")
v('final_source30_configuration_and_future7_runtime_bindings_v1.json','final_source31_configuration_and_future7_runtime_bindings_v1.json')
v("output=F.parent/'resident_RAM_training_actual_23';require(not output.exists()and not(B/'invocations/owner_000022').exists()and not(B/'journals/commits_000022.jsonl').exists(),'Fresh next owner22/output23/journal22, no reuse')", "output=F.parent/'resident_RAM_training_actual_24';require(not output.exists()and not(B/'invocations/owner_000023').exists()and not(B/'journals/commits_000023.jsonl').exists(),'Fresh next owner23/output24/journal23, no reuse')")
v('invocations/owner_000021','invocations/owner_000022',3)
v("ROOT_source30_canonical_archive_saved", "ROOT_source31_fresh_signature_collect_saved")
v("'--start-invocation','22'", "'--start-invocation','23'")
v("'--canonical-archive-resume'", "'--fresh-signature-resume'")
v('ASR18_source30_actual_canonical_archive_transition_final_metadata_handoff_v1','ASR18_source31_actual_fresh_signature_collect_transition_final_metadata_handoff_v1')
v('run_ASR18_BPE9223_resident_RAM_autocontinue_v30.py','run_ASR18_BPE9223_resident_RAM_autocontinue_v31.py')
v("'canonical_archive_transition_validator':pin(S/'asr18_canonical_archive_checkpoint_transition_v1.py')", "'fresh_signature_transition_validator':pin(S/'asr18_fresh_signature_collect_checkpoint_transition_v1.py')")
v("'canonical_archive_resume':pin(closure)", "'fresh_signature_resume':pin(closure)")
v("'initial_owner_invocation':22", "'initial_owner_invocation':23")
v("  'prepared_metadata_archive_representation_policy':configuration['prepared_metadata_archive_representation_policy'],", "  'prepared_metadata_archive_representation_policy':configuration['prepared_metadata_archive_representation_policy'],\n  'fresh_metadata_signature_and_collect_observation_policy':configuration['fresh_metadata_signature_and_collect_observation_policy'],\n  'actual_CPU_review':configuration['fresh_metadata_signature_and_collect_observation_policy']['completed_CPU_review'],\n  'long_validation_observer_source':pin(F/'observe_source31_dynamic_durable_validation_v1.py'),")
v('final_source30_actual_canonical_archive_transition_execution_handoff_v1.json','final_source31_signature_collect_transition_execution_handoff_v1.json')
ast.parse(t);inv=t
for e in reversed(edits):
 assert inv.count(e['new_literal'])==e['expected_count'],e['new_literal'];inv=inv.replace(e['new_literal'],e['old_literal'])
assert inv.replace('\n',nl).encode()==raw;assert not new.exists();new.write_bytes(t.replace('\n',nl).encode())
diff=F/(new.stem+'.diff')
with diff.open('x',encoding='utf-8',newline='\n')as f:f.write(''.join(difflib.unified_diff(old.read_text().splitlines(True),t.splitlines(True),fromfile=str(old),tofile=str(new))))
ledger=F/'source31_metadata_materializer_exact_raw_inverse_v1.json'
with ledger.open('x',encoding='utf-8',newline='\n')as f:json.dump({'old':pin(old),'new':pin(new),'replacements':edits,'newline_policy':repr(nl),'full_old_raw_inverse_exact':True,'diff':pin(diff)},f,indent=2);f.write('\n')
hp=F/'inactive_source31_actual_stop_metadata_materializer_handoff_v1.json'
inputs=[F/'candidate_source31_configuration_completed_CPU_pending_actual_stop_v1.json',F/'production_configuration_v1/configuration.json',
 F/'actual_source30_fresh_signature_collect_checkpoint_transition_01/actual_transition_closure.json',F/'ROOT_source31_fresh_signature_collect_source_and_checkpoint_stop_review_v1.json',
 F/'ROOT_fresh_signature_original_collect_actual_CPU_review_v1.json',F/'observe_source31_dynamic_durable_validation_v1.py',new,ledger]
with hp.open('x',encoding='utf-8',newline='\n')as f:json.dump({'materialization_source':pin(new),'raw_inverse':pin(ledger),'input_pins':[pin(p)for p in inputs],
 'scope':'ASR18_fresh_signature_collect_metadata_materialization_v1','actual_pre_stop_review':pin(inputs[3]),'actual_stop_ROOT_review':'PENDING separate ROOT actual-stop acceptance',
 'metadata_only_no_actual_execution_admitted':True,'all_fresh_output_and_base_runtime7_pointer_assertions_before_writes':True,
 'explicit_output_LF_prevents_prior_runtime7_prospective_pin_mismatch':True},f,indent=2);f.write('\n')
print(json.dumps({'materializer':pin(new),'raw_inverse':pin(ledger),'handoff':pin(hp)}))
