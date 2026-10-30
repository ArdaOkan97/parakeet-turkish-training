"""Source-only immutable actual-closure regression successor; no fixture execution."""
import ast,difflib,hashlib,json,os
from pathlib import Path
F=Path(__file__).parent
assert os.environ['TORCH_EXTENSIONS_DIR'].lower().startswith('e:')
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def write(p,v):
 with p.open('x',encoding='utf-8',newline='\n')as f:json.dump(v,f,indent=2);f.write('\n')
def successor(old,new,changes):
 raw=old.read_bytes();t=raw.decode().replace('\r\n','\n');nl='\r\n'if b'\r\n'in raw else'\n';assert t.replace('\n',nl).encode()==raw;ledger=[]
 for a,b in changes:
  n=t.count(a);assert n>0,a;ledger.append({'old_literal':a,'new_literal':b,'expected_count':n});t=t.replace(a,b)
 ast.parse(t);newraw=t.replace('\n',nl).encode();assert not new.exists();new.write_bytes(newraw);inv=t
 for e in reversed(ledger):
  assert inv.count(e['new_literal'])==e['expected_count'],e['new_literal'];inv=inv.replace(e['new_literal'],e['old_literal'])
 assert inv.replace('\n',nl).encode()==raw
 diff=F/(new.stem+'.diff')
 with diff.open('x',encoding='utf-8',newline='\n')as f:f.write(''.join(difflib.unified_diff(old.read_text().splitlines(True),t.splitlines(True),fromfile=str(old),tofile=str(new))))
 return {'old':pin(old),'new':pin(new),'replacements':ledger,'original_newline_restored':repr(nl),'full_raw_inverse_exact':True,'diff':pin(diff)}
fixture=successor(F/'verify_source30_actual_checkpoint_archive_entire_selectors_v1.py',F/'verify_source31_actual_checkpoint_signature_collect_entire_selectors_v2.py',[
 ('INACTIVE actual-closure/entire-selector metadata regression for source30.','INACTIVE actual-closure/entire-selector metadata regression for source31.'),
 ('actual_closed_source29_metadata_selector_control','actual_closed_source30_metadata_selector_control'),
 ("F/'typed_selector_actual_01'","F/'typed_signature_collect_selector_actual_01'"),
 ("configuration['prior_resident_RAM_canonical_archive_checkpoint_transition']", "configuration['prior_resident_RAM_fresh_signature_collect_checkpoint_transition']"),
 ("configuration['prepared_metadata_archive_representation_policy']", "configuration['fresh_metadata_signature_and_collect_observation_policy']"),
 ('Only genuine completed ROOT codec review and actual source29 closure in final immutable configuration','Only genuine completed ROOT signature/collect review and actual source30 closure in final immutable configuration'),
 ('run_ASR18_BPE9223_resident_RAM_training_v30','run_ASR18_BPE9223_resident_RAM_training_v31'),
 ('asr18_canonical_archive_checkpoint_transition_v1','asr18_fresh_signature_collect_checkpoint_transition_v1'),
 ('checked_canonical_archive_checkpoint_resume','checked_fresh_signature_collect_checkpoint_resume'),
 ('canonical_archive_resume','fresh_signature_resume'),
 ('canonical_archive_transition_validator','fresh_signature_transition_validator'),
 ("optional=['prior_process','lookahead_resume'", "optional=['prior_process','canonical_archive_resume','lookahead_resume'"),
 ("'canonical_transition'","'signature_transition'"),
 ('invocation=22,start_invocation=22','invocation=23,start_invocation=23'),
 ('wrong.invocation=21','wrong.invocation=22'),
 ('commits_000022.jsonl','commits_000023.jsonl'),
 ("'initial_owner_invocation':22", "'initial_owner_invocation':23"),
 ('wrong.start_invocation=21','wrong.start_invocation=22'),
 ("'invocation':22", "'invocation':23"),
 ('--canonical-archive-resume','--fresh-signature-resume'),
 ('Original21 STAT','Original22 STAT'),
 ('invocations/owner_000022','invocations/owner_000023'),
 ('resident_RAM_training_actual_23','resident_RAM_training_actual_24'),
 ('ASR18_actual_source30_saved_checkpoint_canonical_archive_entire_selector_metadata_CPU_control_v1','ASR18_actual_source31_saved_checkpoint_signature_collect_entire_selector_metadata_CPU_control_v1'),
 ('real_actual_closure_archive_and_unchanged_credit_policy_validators_executed','real_actual_closure_signature_collect_and_unchanged_credit_policy_validators_executed')])
launcher=successor(F/'launch_one_reviewed_source30_selector_metadata_CPU_v1.py',F/'launch_one_reviewed_source31_signature_selector_metadata_CPU_v1.py',[
 ('verify_source30_actual_checkpoint_archive_entire_selectors_v1.py','verify_source31_actual_checkpoint_signature_collect_entire_selectors_v2.py'),
 ('typed_selector_actual_01','typed_signature_collect_selector_actual_01')])
ledger=F/'source31_entire_selector_metadata_fixture_exact_raw_inverse_v1.json';write(ledger,{'fixture':fixture,'launcher':launcher,'actual_execution_admitted':False})
h=F/'inactive_source31_entire_selector_metadata_fixture_handoff_v1.json'
write(h,{'kind':'ASR18_INACTIVE_source31_entire_selector_metadata_fixture_handoff_v1','fixture_source':fixture['new'],'launcher_source':launcher['new'],'raw_inverse':pin(ledger),
 'preserved_unexecuted_preparation_only_v1_fixture':pin(F/'verify_source31_actual_checkpoint_signature_collect_entire_selectors_v1.py'),
 'preparation_v1_TOOL1_e4975e':'Raw inverse string-count collision only; no fixture or job executed. Narrow quoted-token inverse correction in v2.',
 'source_transition_handoff':pin(F/'inactive_source31_fresh_signature_collect_transition_handoff_v1.json'),
 'actual_execution_admitted':False,'actual_stop_and_final_configuration_pending':True,
 'scope_after_actual_stop':'ASR18_actual_closed_source30_signature_metadata_selector_CPU_v1',
 'coverage':'Actual closure/all policies; ENTIRE real pre-restore branch factory and post-restore dictionary; exact wrapper entry/forwarding; strict gap0/1 METHOD_STUB negatives. Explicit no-read PrefixView, journal binding and structural review doubles.',
 'limits':'180s fixture/240s external, family512MiB/full512MiB prospective+FULL256MiB prefetch/original2GBcommit1.5GBphysical/E100GB; CPU-1/caps1/BelowNormal/UTF8/-B.',
 'no_restore_copy_model_audio_GPU_native_or_authoritative_state':True})
print(json.dumps({'handoff':pin(h),'ledger':pin(ledger),'fixture':fixture['new'],'launcher':launcher['new']}))
