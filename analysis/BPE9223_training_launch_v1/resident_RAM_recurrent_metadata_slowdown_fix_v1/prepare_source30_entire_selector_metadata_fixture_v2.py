"""Freeze a metadata regression source only; no fixture/control execution."""
import ast,difflib,hashlib,json
from pathlib import Path
F=Path(__file__).parent;OLD=F.parent/'resident_RAM_two_logical_packet_lookahead_v1'
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def write(p,v):
 with p.open('x',encoding='utf-8')as f:json.dump(v,f,indent=2);f.write('\n')
def begin(p):
 raw=p.read_bytes();t=raw.decode().replace('\r\n','\n');nl='\n';assert t.encode()==raw.replace(b'\r\n',b'\n');return t,nl,[]
def change(t,l,a,b,n=1):
 assert t.count(a)==n,(a,t.count(a),n);l.append(dict(old_literal=a,new_literal=b,expected_count=n));return t.replace(a,b)
def freeze(old,new,t,nl,l):
 ast.parse(t);assert not new.exists();new.write_bytes(t.replace('\n',nl).encode());back=t
 for e in reversed(l):
  assert back.count(e['new_literal'])==e['expected_count'];back=back.replace(e['new_literal'],e['old_literal'])
 lines=back.splitlines(True);original=old.read_bytes().decode().splitlines(True);endings=['CRLF'if x.endswith('\r\n')else'LF'if x.endswith('\n')else'NONE'for x in original];assert len(lines)==len(endings)
 rebuilt=''.join(x.rstrip('\r\n')+('\r\n'if end=='CRLF'else'\n'if end=='LF'else'')for x,end in zip(lines,endings));assert rebuilt.encode()==old.read_bytes()
 diff=F/(new.stem+'.diff')
 with diff.open('x',encoding='utf-8')as f:f.write(''.join(difflib.unified_diff(old.read_text().splitlines(True),t.splitlines(True),fromfile=str(old),tofile=str(new))))
 return dict(old=pin(old),new=pin(new),newline_policy='NEW LF; restore each old line terminator after inverse',original_line_terminators=endings,replacements=l,full_old_raw_inverse=True,diff=pin(diff))
old=OLD/'verify_source29_actual_checkpoint_transition_entire_selectors_v1.py';new=F/'verify_source30_actual_checkpoint_archive_entire_selectors_v1.py';t,nl,l=begin(old)
def c(a,b,n=1):
 global t;t=change(t,l,a,b,n)
c('metadata regression for source29.','metadata regression for source30.')
c('actual_closed_source27_metadata_selector_control','actual_closed_source29_metadata_selector_control')
c("closure_pin=review['lookahead_resume']", "closure_pin=review['canonical_archive_resume']")
c("configuration['prior_resident_RAM_approved_lookahead_checkpoint_transition']", "configuration['prior_resident_RAM_canonical_archive_checkpoint_transition']")
c("configuration['resident_RAM_packet_credit_policy']['actual_CPU_control_review']==review['actual_CPU_review']", "configuration['prepared_metadata_archive_representation_policy']['completed_CPU_review']==review['actual_CPU_review']")
c('Only genuine completed ROOT transport review and actual closure','Only genuine completed ROOT codec review and actual source29 closure')
c('import run_ASR18_BPE9223_resident_RAM_training_v29 as driver','import run_ASR18_BPE9223_resident_RAM_training_v30 as driver')
c('from asr18_approved_lookahead_checkpoint_transition_v3 import checked_lookahead_checkpoint_resume,validate_intentional_optimizer_journal_gap', 'from asr18_canonical_archive_checkpoint_transition_v1 import checked_canonical_archive_checkpoint_resume,validate_intentional_optimizer_journal_gap')
c('actual=checked_lookahead_checkpoint_resume(', 'actual=checked_canonical_archive_checkpoint_resume(')
c("optional=['prior_process','cushion225_resume'", "optional=['prior_process','lookahead_resume','cushion225_resume'")
c("lookahead_resume=closure_pin['path'],invocation=21,start_invocation=21", "canonical_archive_resume=closure_pin['path'],invocation=22,start_invocation=22")
c("n.test.left.id=='lookahead_transition'", "n.test.left.id=='canonical_transition'")
c("'lookahead_transition':actual", "'canonical_transition':actual",4)
c('wrong.invocation=20','wrong.invocation=21')
c('commits_000021.jsonl','commits_000022.jsonl',2)
oldblock="""structural_review={'lookahead_resume':closure_pin,'lookahead_transition_validator':review['lookahead_transition_validator'],
            'human_approval':configuration['resident_RAM_packet_credit_policy']['human_approval'],
            'packet_credit_policy':configuration['resident_RAM_packet_credit_policy'],
            'native_update_commit_pressure_policy':configuration['native_update_commit_pressure_policy'],'initial_owner_invocation':21}"""
newblock="""structural_review={'canonical_archive_resume':closure_pin,'canonical_archive_transition_validator':review['canonical_archive_transition_validator'],
            'human_approval':pin(F/'ROOT_direct_human_then_fix_authorization_v1.json'),'initial_owner_invocation':22}"""
c(oldblock,newblock)
c("'record':driver.record,'S':S,'require':require}", "'record':driver.record,'S':S,'Path':Path,'require':require}")
c('wrong.start_invocation=20','wrong.start_invocation=21')
c('missing.lookahead_resume=None','missing.canonical_archive_resume=None')
c("'invocation':21,'command':[]", "'invocation':22,'command':[]")
c("['--lookahead-resume',closure_pin['path']]", "['--canonical-archive-resume',closure_pin['path']]")
c('Original20 STAT only remains unchanged','Original21 STAT only remains unchanged')
c('invocations/owner_000021','invocations/owner_000022')
c("F.parent/'resident_RAM_training_actual_22'", "F.parent/'resident_RAM_training_actual_23'")
c('ASR18_actual_source29_saved_checkpoint_lookahead_entire_selector_metadata_CPU_control_v1','ASR18_actual_source30_saved_checkpoint_canonical_archive_entire_selector_metadata_CPU_control_v1')
c('real_actual_closure_and_credit_policy_validators_executed','real_actual_closure_archive_and_unchanged_credit_policy_validators_executed')
fixture=freeze(old,new,t,nl,l)
old=OLD/'launch_one_reviewed_source29_selector_metadata_CPU_v2.py';new=F/'launch_one_reviewed_source30_selector_metadata_CPU_v1.py';t,nl,l=begin(old)
t=change(t,l,'verify_source29_actual_checkpoint_transition_entire_selectors_v1.py','verify_source30_actual_checkpoint_archive_entire_selectors_v1.py')
launcher=freeze(old,new,t,nl,l)
ledger=F/'source30_entire_selector_metadata_fixture_exact_raw_inverse_v1.json';write(ledger,{'fixture':fixture,'launcher':launcher,'launcher_guards_and_CLI_exact':True,'no_execution':True})
handoff={'kind':'ASR18_INACTIVE_source30_entire_archive_transition_selector_metadata_fixture_handoff_v1','fixture_source':fixture['new'],'external_launcher_source':launcher['new'],'inverse':pin(ledger),'future_actual_output':str(F/'typed_selector_actual_01'),'future_external_result':str(F/'typed_selector_actual_01_external_result.json'),
 'required_scope_fields':['accepted','reviewer','fixture_source','external_launcher_source','input_pins','source_configuration','canonical_archive_resume','initial_checkpoint','branch','actual_CPU_review','wrapper_source','canonical_archive_transition_validator'],
 'requires_actual_stop_and_final_configuration_first':True,'production_launch_admitted':False,'actual_proof_admitted':False,
 'scope':'Real source30 actual-closure/policy validators plus entire real pre/postrestore dicts and initial wrapper forwarding. Explicit PrefixView/journal binding/structural review doubles only. Gap0/1 strict trace controls labelled METHOD_STUB. Original21 STAT plus preserved suffix COPY; no original prefix/whole/model/audio/NumPy/Torch/kernel/restore/copy/state writes.',
 'limits':{'internal_seconds':180,'external_seconds':240,'family_private_bytes':536870912,'full_prospective_bytes':536870912,'full_prefetch_bytes':268435456,'commit_floor_bytes':2000000000,'physical_floor_bytes':1500000000,'E_free_bytes':100000000000,'CUDA_VISIBLE_DEVICES':'-1','maththreads':1,'priority':'BelowNormal'},
 'exact_future_launcher_arguments':['-Xutf8','-B',launcher['new']['path'],'--ROOT-review','FUTURE_EXACT_ROOT_TYPED_METADATA_SCOPE']}
hp=F/'inactive_source30_entire_selector_metadata_fixture_handoff_v1.json';write(hp,handoff)
print(json.dumps({'handoff':pin(hp),'fixture':fixture['new'],'launcher':launcher['new'],'inverse':pin(ledger)}))
