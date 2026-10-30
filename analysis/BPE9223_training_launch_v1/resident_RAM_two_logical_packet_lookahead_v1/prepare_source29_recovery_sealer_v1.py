import ast,hashlib,json
from pathlib import Path
N=Path('E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1');A=N/'resident_RAM_training_actual_22'
old=N/'resident_RAM_training_actual_21/seal_source27_saved156800_recovery_v1.py';text=old.read_text(encoding='utf-8-sig');edits=[]
def edit(a,b,n=1):
 global text
 assert text.count(a)==n,(a[:100],text.count(a),n)
 text=text.replace(a,b);edits.append({'old_literal':a,'new_literal':b,'expected_count':n})
# Honest current observation frontier labels, not source/runtime historical validators.
for a,b in [('source27','source29'),('owner_000020','owner_000021'),('156800','158800'),('156820','158820'),('156600','158600'),('156601','158601'),('156770','158621')]:
 count=text.count(a)
 if count:edit(a,b,count)
edit('Reads ONLY170 new journal20 lines','Reads ONLY21 new journal21 lines')
edit("progress['producer']=={'pid':9704,'create_time':1792727133.7017913,'ppid':32860}","progress['producer']=={'pid':37836,'create_time':1792733651.0335317,'ppid':4344}")
edit("review_path=N/'ROOT_source29_cushion2250000000_saved158600_postrestore_selector_execution_review_v1.json'", "review_path=N/'resident_RAM_two_logical_packet_lookahead_v1/ROOT_source29_approved_two_logical_lookahead_saved158600_final_execution_review_v1.json'")
edit("review['future_7_runtime_bindings_NOT_actual_restore']", "review['future_seven_runtime_bindings']")
edit("branch['predecessor']==review['postrestore_selector_resume']", "branch['predecessor']==review['lookahead_resume']")
edit("range(158601,156771)","range(158601,158621)")
edit("saved=review['saved_journal_prefix'];require(branch['saved_prefix']==saved and saved['offset']==317194933,'Exact original18 published saved prefix binding')", "saved=read_json(review['initial_checkpoint']['path'])['journal_prefix'];require(branch['saved_prefix']==saved and saved['offset']==321119606,'Exact source27 published158600 prefix binding')")
edit("{'path':str(B/'journals/commits_000020.jsonl'),'offset':317194933,'sha256':saved['sha256']}", "{'path':str(B/'journals/commits_000021.jsonl'),'offset':321119606,'sha256':saved['sha256']}")
edit("'New active20 only, no orphan promotion'", "'New active21 only, historical20 retained unchanged'")
a=""" require(ready['producer_identity']=={'create_time':1792727182.5415392,'pid':4392,'ppid':39368}and ready['source']['protocol']==review['CPU_source'],'Exact admitted CPU10 producer/source')
 families=[('native',9704,1792727133.7017913),('native_interpreter_launcher',32860,1792727133.694484),('wrapper',12328,1792727133.3462756),('CPU',4392,1792727182.5415392),('CPU_interpreter_launcher',39368,1792727182.5201604)]"""
b=""" require(ready['source']['protocol']==review['CPU_source']and ready['packet_credit']['logical_packet_maximum']==2
  and ready['packet_credit']['helper_source']==review['packet_credit_policy']['helper_source'],'Actual CPU11 production credit2/source/helper identity')
 cpu=ready['producer_identity'];cpu_launcher=psutil.Process(cpu['ppid'])
 families=[('native',37836,1792733651.0335317),('native_interpreter_launcher',4344,1792733651.0267332),
  ('wrapper',40576,1792733650.5702133),('CPU',cpu['pid'],cpu['create_time']),
  ('CPU_interpreter_launcher',cpu_launcher.pid,cpu_launcher.create_time())]"""
edit(a,b)
a=""" oldstats=[]
 for item in(review['original18_closed_STAT_only'],review['orphan19_preserved_STAT_only']):
  current=Path(item['path']).stat();require(current.st_size==item['bytes']and current.st_mtime_ns==item['mtime_ns'],'Preserved original18 and inactive orphan19 STAT only');oldstats.append(item)"""
b=""" old=read_json(review['lookahead_resume']['path'])['closed_old_journal_STAT'];current=Path(old['path']).stat()
 require(current.st_size==old['bytes']and current.st_mtime_ns==old['mtime_ns'],'Original closed20 STAT unchanged; no old prefix/whole/tail reread')
 oldstats=[old]"""
edit(a,b)
edit('range(158601,156771)', 'range(158601,158622)') if 'range(158601,156771)'in text else None
# The first global range was historical20; new event loop must include one beyond.
edit('  for update in range(158601,158621):','  for update in range(158601,158622):')
edit("and math.isfinite(n['mean_objective'])and math.isfinite(n['global_grad_norm'])and n['global_grad_norm']>0", "and math.isfinite(n['mean_objective'])and n['mean_objective']>0 and math.isfinite(n['global_grad_norm'])and n['global_grad_norm']>0\n    and ev['producer_source_sha256']==review['CPU_source']['sha256']and ev['token'].split(':')[0]==str(cpu['pid'])\n    and e['broad_draws']==14*update and e['replay_updates']==update and e['cleaner_draws']==0")
edit("actual170_new_positive_events158601_158621.jsonl","actual21_new_positive_events158601_158621.jsonl")
edit("'ROOT_closed_boot_restore_bindings':record(N/'ROOT_source29_actual_boot_runtime_bindings_and_restore_v1.json'),", "'owner_boot_restore_bindings':record(A/'actual_owned_source29_boot_session.json'),")
# Rename all remaining count/proof labels truthfully, preserving closed source27 sealer.
for a,b in [('actual170','actual21'),('new170','new21'),('new170_count\':170','new21_count\':21')]:
 count=text.count(a)
 if count:edit(a,b,count)
edit("'new21_count':170", "'new21_count':21")
edit("'owned_session':56620", "'owned_session':87302")
edit("'kind':'ASR18_actual_source29_saved158600_cushion225_recovered158800_candidate_v1'", "'kind':'ASR18_actual_source29_approved_lookahead_saved158600_recovered158800_candidate_v1'")
edit("'original18_and_inactive_orphan19_preserved_STAT_only'", "'original20_preserved_STAT_only'")
edit("'old169_historical_attempts_genuinely_redone_plus_one'", "'old20_historical_attempts_genuinely_redone_plus_one'")
edit("'no_old_whole_prefix169tail_tensor_or_audio_read'", "'no_old_whole_prefix20tail_tensor_or_audio_read'")
edit("'new225_policy_unchanged':review['native_update_commit_pressure_policy']", "'approved225_policy_unchanged':review['native_update_commit_pressure_policy'],'actual_credit2_policy':review['packet_credit_policy']")
edit("if __name__=='__main__':main()", "if __name__=='__main__':main()")
# Drop the no-op from inverse ledger and require all expected current values/labels.
edits=[x for x in edits if x['old_literal']!=x['new_literal']]
assert "range(158601,158621)"in text and "range(158601,158622)"in text
assert "'new21_count':21"in text and '170'not in text and '156'not in text
ast.parse(text);new=A/'seal_source29_saved158800_recovery_v1.py';assert not new.exists();new.write_bytes(text.encode())
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
ledger={'old_source':pin(old),'new_source':pin(new),'edits_LF_normalized':edits,
 'scope':'One routine metadata observer ONLY after natural158800/publishedpost158820; new21 currentjournal range only, no oldprefix/whole/tail/payload/audio/GPU/control',
 'actual_sealer_executed':False,'fixed400_speed_observation_is_separate_actual_continued_job_interval':[158620,159020]}
with (A/'source29_recovery_metadata_sealer_source_ledger_v1.json').open('x',encoding='utf-8')as f:json.dump(ledger,f,indent=2);f.write('\n')
print(json.dumps({'source':pin(new),'ledger':pin(A/'source29_recovery_metadata_sealer_source_ledger_v1.json'),'AST_only':True}))
