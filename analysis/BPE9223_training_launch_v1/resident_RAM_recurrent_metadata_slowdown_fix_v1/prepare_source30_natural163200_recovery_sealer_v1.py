"""Inactive routine metadata sealer successor; no observer/control executed."""
import ast,difflib,hashlib,json
from pathlib import Path
F=Path(__file__).parent;A=F.parent/'resident_RAM_training_actual_23';OLD=F.parent/'resident_RAM_training_actual_22/seal_source29_saved158800_recovery_v1.py'
# Source artifacts may exist in future actual23 without constituting an owned
# launch/output. Keep the actual sealer source in F until final launch admission.
new=F/'seal_source30_saved163200_recovery_v1.py';raw=OLD.read_bytes();t=raw.decode().replace('\r\n','\n');edits=[]
def c(a,b,n=1):
 global t
 assert t.count(a)==n,(a,t.count(a),n);t=t.replace(a,b);edits.append(dict(old_literal=a,new_literal=b,expected_count=n))
c('ONLY21 new journal21 lines','ONLY21 new journal22 lines')
c('158800','163200',16)
c('158820','163220',2)
c('owner_000021','owner_000022')
c('A=Path(__file__).parent',"A=N/'resident_RAM_training_actual_23'")
c("progress=read_json(B/'progress/latest.json');require(progress['producer']=={'pid':37836,'create_time':1792733651.0335317,'ppid':4344}","boot=read_json(I/'boot.json');progress=read_json(B/'progress/latest.json');require(progress['producer']==boot['producer']")
c('Real positive post163200 source29','Real positive post163200 source30')
c("review_path=N/'resident_RAM_two_logical_packet_lookahead_v1/ROOT_source29_approved_two_logical_lookahead_saved158600_final_execution_review_v1.json'", "review_path=N/'resident_RAM_recurrent_metadata_slowdown_fix_v1/ROOT_source30_canonical_archive_saved163000_final_execution_review_v1.json'")
c("review['lookahead_resume']", "review['canonical_archive_resume']",2)
c('158601','163001',3);c('158621','163021',3);c('158622','163022')
c('158600','163000',4)
c('321119606','329760922',2)
c('source27 published163000','source29 published163000')
c('commits_000021.jsonl','commits_000022.jsonl')
c('New active21 only, historical20 retained unchanged','New active22 only, historical21 retained unchanged')
old="""families=[('native',37836,1792733651.0335317),('native_interpreter_launcher',4344,1792733651.0267332),
  ('wrapper',40576,1792733650.5702133),('CPU',cpu['pid'],cpu['create_time']),
  ('CPU_interpreter_launcher',cpu_launcher.pid,cpu_launcher.create_time())]"""
newblock="""child=read_json(A/'current_child.json');launcher=child['launcher'];wrapper=psutil.Process(launcher['ppid'])
 families=[('native',boot['producer']['pid'],boot['producer']['create_time']),
  ('native_interpreter_launcher',launcher['pid'],launcher['create_time']),
  ('wrapper',wrapper.pid,wrapper.create_time()),('CPU',cpu['pid'],cpu['create_time']),
  ('CPU_interpreter_launcher',cpu_launcher.pid,cpu_launcher.create_time())]"""
c(old,newblock)
c('Original closed20 STAT unchanged; no old prefix/whole/tail reread','Original closed21 STAT unchanged; no old prefix/whole/tail reread')
c('actual21_new_positive_events163001_163021.jsonl','actual21_new_positive_events163001_163021.jsonl') if False else None
c('actual_owned_source29_boot_session.json','actual_owned_source30_boot_session.json')
c('ASR18_actual_source29_approved_lookahead_saved163000_recovered163200_candidate_v1','ASR18_actual_source30_canonical_archive_saved163000_recovered163200_candidate_v1')
c("'owned_session':87302", "'owned_session':read_json(A/'actual_owned_source30_boot_session.json')['owned_session']")
c('original20_preserved_STAT_only','original21_preserved_STAT_only')
c('old20_historical_attempts_genuinely_redone_plus_one','old21_historical_attempts_genuinely_redone_plus_one')
c('no_old_whole_prefix20tail_tensor_or_audio_read','no_old_whole_prefix21tail_tensor_or_audio_read')
c("'no_fit_speed_saving_prevention_guarantee':True", "'actual_canonical_archive_representation_policy':review['prepared_metadata_archive_representation_policy'],\n  'new_preparation_archive':record(checkpoint/'prepared_metadata.json.gz'),\n  'long_production_validation_still_pending163020_164620':True,'no_fit_speed_saving_prevention_guarantee':True")
ast.parse(t);assert not new.exists();new.write_bytes(t.replace('\n','\r\n').encode())
back=t
for e in reversed(edits):
 assert back.count(e['new_literal'])==e['expected_count'];back=back.replace(e['new_literal'],e['old_literal'])
assert back.replace('\n','\r\n').encode()==raw
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
ledger=F/'source30_recovery_metadata_sealer_exact_raw_inverse_v1.json'
with ledger.open('x',encoding='utf-8')as f:json.dump(dict(old=pin(OLD),new=pin(new),replacements=edits,full_old_raw_inverse=True,newline_policy='Original CRLF restored byte exact',no_execution=True),f,indent=2);f.write('\n')
print(json.dumps({'sealer':pin(new),'inverse':pin(ledger)}))
