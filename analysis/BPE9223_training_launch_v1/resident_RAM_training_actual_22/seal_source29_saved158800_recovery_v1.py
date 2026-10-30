"""Metadata-only natural recovery observer; no training, loader or payload reads.
Reads ONLY21 new journal21 lines from the known saved offset, never old prefixes.
Requires natural158800 persistence and positive158820+ before its ONE execution.
"""
import hashlib,json,math,os,shutil,sys,time
from pathlib import Path
import psutil
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
from resource_watchdog_v2 import sample_memory
N=Path('E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1');B=Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01');I=B/'invocations/owner_000021';A=Path(__file__).parent
began=time.monotonic();peak=0
def guard():
 global peak
 sample=sample_memory(os.getpid(),'source29_saved158800_compact_recovery_observer');m=sample['system_memory'];peak=max(peak,sample['worker_private_memory']['private_bytes'])
 require(peak<536870912 and time.monotonic()-began<180 and m['commit_available_bytes']>=2000000000+268435456
  and m['physical_available_bytes']>=1500000000+268435456 and shutil.disk_usage('E:/').free>=100000000000,'Unchanged metadata guard/fullprefetch/E100GB')
 return sample
def main():
 me=psutil.Process();me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);parent=me.parent();initial=guard();m=initial['system_memory']
 require(m['commit_available_bytes']>=2000000000+268435456+536870912 and m['physical_available_bytes']>=1500000000+268435456+536870912,'Full prospective512MiB metadata allowance')
 require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'and Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad')),'CPU-only observer in E storage environment')
 out=A/'actual_saved158800_recovery_01';require(not out.exists(),'ONE fresh natural metadata seal')
 progress=read_json(B/'progress/latest.json');require(progress['producer']=={'pid':37836,'create_time':1792733651.0335317,'ppid':4344}
  and progress['actual_optimizer_updates']>=158820 and not progress['optimizer_poisoned']and not progress['schedule_poisoned']
  and progress['last_saved_updates']>=158800 and progress['planned_native_updates']==401072 and progress['administrative_update_stop']is None,'Real positive post158800 source29, unchanged full goal')
 require(not(I/'actual_failure.json').exists(),'No new actual training failure')
 checkpoint=B/'checkpoints/update_000158800';pointer=read_json(checkpoint/'pointer.json');persist=read_json(checkpoint/'persistence_verified.json')
 require(pointer['committed_updates']==158800 and persist['passed']is True and persist['optimizer_updates']==158800
  and persist['named_tensors_verified']==854 and persist['weights_verified']==207 and persist['Adam_states_verified']==215
  and persist['saved_semantic_state_verified']['passed']is True
  and persist['independent_pre_save_state_sha256']==persist['saved_state_sha256'],'Natural158800 full854/207/215 semantic/PRE-SAVED equality publication')
 require(progress['clocks']==[progress['actual_optimizer_updates']+x for x in(64250,2000,62250)],'All three actual clocks preserved')
 review_path=N/'resident_RAM_two_logical_packet_lookahead_v1/ROOT_source29_approved_two_logical_lookahead_saved158600_final_execution_review_v1.json';review=read_json(review_path)
 config=read_json(I/'configuration.json');require(all(config[k]==v for k,v in review['future_seven_runtime_bindings'].items()),'All7 actual runtime bindings exact')
 restore=read_json(I/'actual_parent_restore.json');branch=read_json(I/'actual_journal_branch.json');ready=read_json(I/'CPU/actual_CPU_ready.json')
 require(restore['passed']is True and restore['all207_threeAdam_all4RNG_match_actual_pre_save']is True and restore['next_draw_packet_match']is True
  and restore['restore_occurred_only_at_startup_before_native']is True and branch['predecessor']==review['lookahead_resume']
  and branch['native_updates_or_draws_in_copy']==0 and branch['historical_unsaved_tail_updates']==list(range(158601,158621)), 'Actual full restore before copied branch; new startup predecessor bound')
 saved=read_json(review['initial_checkpoint']['path'])['journal_prefix'];require(branch['saved_prefix']==saved and saved['offset']==321119606,'Exact source27 published158600 prefix binding')
 expected={'path':str(B/'journals/commits_000021.jsonl'),'offset':321119606,'sha256':saved['sha256']}
 require(branch['new_active_journal']==expected and Path(progress['journal_prefix']['path'])==Path(expected['path']),'New active21 only, historical20 retained unchanged')
 require(ready['source']['protocol']==review['CPU_source']and ready['packet_credit']['logical_packet_maximum']==2
  and ready['packet_credit']['helper_source']==review['packet_credit_policy']['helper_source'],'Actual CPU11 production credit2/source/helper identity')
 cpu=ready['producer_identity'];cpu_launcher=psutil.Process(cpu['ppid'])
 families=[('native',37836,1792733651.0335317),('native_interpreter_launcher',4344,1792733651.0267332),
  ('wrapper',40576,1792733650.5702133),('CPU',cpu['pid'],cpu['create_time']),
  ('CPU_interpreter_launcher',cpu_launcher.pid,cpu_launcher.create_time())]
 observed=[]
 for role,pid,ct in families:
  q=psutil.Process(pid);require(abs(q.create_time()-ct)<.001,'Current owned identity exactalive '+role);observed.append({'role':role,'pid':pid,'create_time':ct,'ppid':q.ppid(),'exact_alive':True})
 old=read_json(review['lookahead_resume']['path'])['closed_old_journal_STAT'];current=Path(old['path']).stat()
 require(current.st_size==old['bytes']and current.st_mtime_ns==old['mtime_ns'],'Original closed20 STAT unchanged; no old prefix/whole/tail reread')
 oldstats=[old]
 out.mkdir();write_json(out/'positive_progress_observed.json',progress)
 raw=[];events=[];journal=Path(expected['path'])
 with journal.open('rb')as stream:
  stream.seek(saved['offset'])
  for update in range(158601,158622):
   guard();line=stream.readline(1048576);require(line.endswith(b'\n')and 0<len(line)<1048576,'One complete bounded NEW actual event')
   e=json.loads(line);n=e['native'];ev=e['prepared_evidence']
   require(e['optimizer_updates']==n['update']==ev['actual_positive_native_update']==update
    and math.isfinite(n['mean_objective'])and n['mean_objective']>0 and math.isfinite(n['global_grad_norm'])and n['global_grad_norm']>0
    and ev['producer_source_sha256']==review['CPU_source']['sha256']and ev['token'].split(':')[0]==str(cpu['pid'])
    and e['broad_draws']==14*update and e['replay_updates']==update and e['cleaner_draws']==0
    and len(n['source_ids'])==len(n['languages'])==16 and n['languages'][:14]==['tr']*14
    and all(x!='tr'for x in n['languages'][14:]),'Genuine ordered positive native/16 IDs14TR+2replay')
   raw.append(line);events.append(e)
  end=stream.tell()
 for before,after in zip(events,events[1:]):require(after['broad_draws']==before['broad_draws']+14 and after['replay_updates']==before['replay_updates']+1 and after['cleaner_draws']==before['cleaner_draws'],'Ordered exact broad/replay/cleaner cursors')
 snapshot=out/'actual21_new_positive_events158601_158621.jsonl';snapshot.write_bytes(b''.join(raw));require(end-saved['offset']==snapshot.stat().st_size,'Exact only-new segment bytes')
 require('numpy'not in sys.modules and 'torch'not in sys.modules,'No payload/model/NumPy/Torch/GPU imports')
 pins={'execution_review':record(review_path),'owner_boot_restore_bindings':record(A/'actual_owned_source29_boot_session.json'),
  'boot':record(I/'boot.json'),'restore':record(I/'actual_parent_restore.json'),'branch':record(I/'actual_journal_branch.json'),'configuration':record(I/'configuration.json'),
  'CPU_ready':record(I/'CPU/actual_CPU_ready.json'),'current_child':record(A/'current_child.json'),'pointer158800':record(checkpoint/'pointer.json'),'persistence158800':record(checkpoint/'persistence_verified.json'),
  'positive_progress_observed':record(out/'positive_progress_observed.json'),'actual21_new_snapshot':record(snapshot),'sealer_source':record(__file__)}
 report={'kind':'ASR18_actual_source29_approved_lookahead_saved158600_recovered158800_candidate_v1','passed':True,'owner':'/root/memory_diagnosis','owned_session':87302,
  'initial_updates':158600,'actual_progress':progress,'new21_range':[158601,158621],'new21_count':21,'new21_segment':{'path':str(journal),'start_offset':saved['offset'],'end_offset':end,'bytes':end-saved['offset'],'sha256':record(snapshot)['sha256']},
  'actual21_snapshot':record(snapshot),'first_new_event':events[0],'last_new_event':events[-1],'all_positive_ordered_and_finite':True,'actual_all5_exact_alive':observed,
  'genuine_restore_before_saved_prefix_copy_then_native':True,'all7_runtime_bindings_exact':True,'original20_preserved_STAT_only':oldstats,
  'approved225_policy_unchanged':review['native_update_commit_pressure_policy'],'actual_credit2_policy':review['packet_credit_policy'],'old20_historical_attempts_genuinely_redone_plus_one':True,'natural158800_full854_207_215_semantic_PRE_SAVED_equality':True,
  'no_old_whole_prefix20tail_tensor_or_audio_read':True,'no_training_source_environment_reader_or_resource_change':True,'no_fit_speed_saving_prevention_guarantee':True,'full401072_continues_no_finite_stop':True,
  'ROOT_actual_recovery_acceptance_pending':True,'sealer_identity':{'pid':me.pid,'create_time':me.create_time(),'ppid':me.ppid()},'sealer_launcher':{'pid':parent.pid,'create_time':parent.create_time(),'ppid':parent.ppid()},
  'guard_initial':initial,'guard_final':guard(),'peak_private_bytes':peak,'evidence':pins}
 write_json(out/'actual_recovery_status.json',report);print(json.dumps({'status':record(out/'actual_recovery_status.json'),'sealer_identity':report['sealer_identity'],'sealer_launcher':report['sealer_launcher'],'segment':report['new21_segment'],'progress':progress['actual_optimizer_updates']}),flush=True)
if __name__=='__main__':main()
