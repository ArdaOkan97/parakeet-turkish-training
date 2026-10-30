import ast,hashlib,json
from pathlib import Path
F=Path('E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1/resident_RAM_two_logical_packet_lookahead_v1')
source=F/'verify_source29_actual_checkpoint_transition_entire_selectors_v1.py';ast.parse(source.read_text(encoding='utf-8-sig'))
old=F/'launch_one_reviewed_two_logical_packet_CPU_v2.py';text=old.read_text(encoding='utf-8-sig');edits=[]
for a,b in [('verify_two_logical_packet_one_receiver_CPU_v2.py',source.name),('finite_transport_actual_CPU_02','typed_selector_actual_01')]:
 count=text.count(a);assert count>0;text=text.replace(a,b);edits.append({'old_literal':a,'new_literal':b,'expected_count':count})
ast.parse(text);new=F/'launch_one_reviewed_source29_selector_metadata_CPU_v1.py';assert not new.exists();new.write_bytes(text.replace('\n','\r\n').encode())
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def write(p,v):
 with Path(p).open('x',encoding='utf-8')as f:json.dump(v,f,indent=2);f.write('\n')
write(F/'selector_metadata_external_exact_edits_v1.json',{'old_source':pin(old),'new_source':pin(new),'edits_LF_normalized_CRLF_emitted':edits})
write(F/'inactive_source29_entire_selector_metadata_fixture_handoff_v1.json',{
 'fixture_source':pin(source),'external_launcher_source':pin(new),'external_edit_ledger':pin(F/'selector_metadata_external_exact_edits_v1.json'),
 'actual_control_execution_admitted':False,'production_stop_or_launch_admitted':False,
 'after_actual_stop_required':['genuine actual TOOL1/deaths/closure pin','final immutable configuration with323c+actualclosure','latest genuine saved checkpoint/persistence'],
 'real_paths':['actual typed closure/source/config/policy validation','ENTIRE real source29 pre-restore factory metadata dictionary with real record(compactclosure)',
 'ENTIRE real post-restore branch metadata expression with real record(compactclosure)','ENTIRE real wrapper initial entry and first-invocation flag forwarding'],
 'doubles':['prefix view constructor no-read','prospective journal.binding in memory only','review field dictionary in memory only','gap0/1 and strict rejection traces are method stubs'],
 'no_operations':['no model/audio/GPU/NumPy/Torch/restore/prefixcopy/native/authoritative sampler/journalwrites/kernelcredit'],
 'limits':{'internal_seconds':180,'external_seconds':240,'family_private_bytes':536870912,'full_prospective_private_bytes':536870912,
 'full_additional_prefetch_bytes':268435456,'commit_floor_bytes':2000000000,'physical_floor_bytes':1500000000,'E_free_bytes':100000000000},
 'original_journal_read':'STAT only; actual closure helper reads ONLY newly preserved boundedtailCOPY, no originalprefix/whole/old18or19'})
print(json.dumps({'fixture':pin(source),'external':pin(new),'handoff':pin(F/'inactive_source29_entire_selector_metadata_fixture_handoff_v1.json'),'AST_only':True}))
