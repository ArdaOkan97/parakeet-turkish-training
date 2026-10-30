"""Reuse the bounded consumed-certificate observer; no training/code activation."""
from pathlib import Path
import ast,hashlib,json
N=Path(__file__).parent;A=N.parent/'resident_RAM_training_actual_13'
old=N.parent/'resident_RAM_training_actual_11/seal_fourth_terminal_production_checkpoint_v2.py';t=old.read_text(encoding='utf-8')
for before,after in [('owner_000010','owner_000012'),('update_000088600','update_000134600'),
    ('resident_RAM_original_interval_post88426_failure_actual_v1','resident_RAM_original_interval_post134529_failure_actual_v1'),
    ('88600','134600'),('88438','134530'),('fourth_original_terminal_interval_geometry_certificate_v1.json','fifth_original_full_clip_geometry_certificate_v1.json'),
    ('[13]','[3]'),('==120048','==200064'),('==94','==157'),('==39','==82'),('[1,751,128]','[1,1251,128]'),('[1,751]','[1,1251]'),
    ('==750','==1250'),('actual producer\'','actual producer\''),('healthy source12','healthy source16'),
    ("'owned_session':80857","'owned_session':64681"),
    ('ROOT_fourth_original_terminal_interval_saved88400_execution_review_v1.json','ROOT_fifth_original_full_clip_saved134400_execution_review_v1.json'),
    ('ROOT_fourth_original_terminal_interval_actual_CPU_review_v1.json','ROOT_fifth_original_full_clip_geometry_actual_CPU_review_v1.json'),
    ('actual_corrected_original_terminal_interval_saved134600_status.json','actual_corrected_fifth_full_clip_saved134600_status.json')]:
    assert before in t,before;t=t.replace(before,after)
begin=t.index('verified_journal=');end=t.index('native_snapshot=')
replacement='''raw_segment=[];events=[]
require(saved['offset']==273582680 and journal==B/'journals/commits_000012.jsonl','Exact saved134400 prefix and new active journal')
with journal.open('rb')as source:
 source.seek(saved['offset'])
 for update in range(134401,134531):
  line=source.readline(131073);require(line.endswith(b'\\n')and len(line)<=131072,'Bounded complete new event')
  event=json.loads(line)
  require(event['optimizer_updates']==event['native']['update']==event['prepared_evidence']['actual_positive_native_update']==update,
   'Actual130 ordered new positive updates, not copied historical tail')
  raw_segment.append(line);events.append(event)
 new_segment_end_offset=source.tell()
segment=N/'actual_new130_positive_events_134401_134530.jsonl';require(not segment.exists(),'One bounded new segment');segment.write_bytes(b''.join(raw_segment))
verified_journal=record(segment)
require(events[-1]['prepared_evidence']['metadata_sha256']==digest(metadata),'Actual corrected consumed metadata exactly bound to positive journal event')
'''
t=t[:begin]+replacement+t[end:]
t=t.replace("'ROOT_verified_committed134530_fields'","'actual_committed134530_fields'").replace("'ROOT_new38_positive_journal_verification'","'actual_new130_positive_journal_segment'")
t=t.replace("'all_original_selected_interval_samples_used_once':True","'all_original_physical_samples_used_once':True")
before="persistence=read_json(CK/'persistence_verified.json');require(persistence['named_tensors_verified']==854 and persistence['Adam_states_verified']==215,'Actual854/215 persistence')"
after=before+"\nrequire(persistence['passed']and persistence['weights_verified']==207 and persistence['independent_pre_save_state_sha256']==persistence['saved_state_sha256'],'Actual207 weights and PRE/SAVED equality')\nrestore=read_json(I/'actual_parent_restore.json');require(restore['passed']and restore['all207_threeAdam_all4RNG_match_actual_pre_save']and restore['next_draw_packet_match']and restore['restore_occurred_only_at_startup_before_native'],'Fresh exact saved134400 restore before native')"
assert before in t;t=t.replace(before,after)
t=t.replace("'actual_full_clean_PCM_sha256':ref['expected_audio']['waveform_sha256']", "'actual_full_clean_PCM_sha256':ref['expected_audio']['waveform_sha256'],'certified_full_native_PCM_sha256':certificate['native_f32_sha256'],'new130_segment_source_offset':saved['offset'],'new130_segment_end_offset':new_segment_end_offset")
ast.parse(t);out=A/'seal_fifth_full_clip_production_checkpoint_v1.py';assert not out.exists();out.write_text(t,encoding='utf-8')
print(json.dumps({'path':str(out),'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest()}),flush=True)
