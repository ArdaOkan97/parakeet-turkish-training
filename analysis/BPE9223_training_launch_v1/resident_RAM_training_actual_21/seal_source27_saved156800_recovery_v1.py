"""Metadata-only natural recovery observer; no training, loader or payload reads.
Reads ONLY170 new journal20 lines from the known saved offset, never old prefixes.
Requires natural156800 persistence and positive156820+ before its ONE execution.
"""
import hashlib,json,math,os,shutil,sys,time
from pathlib import Path
import psutil
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
from resource_watchdog_v2 import sample_memory
N=Path('E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1');B=Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01');I=B/'invocations/owner_000020';A=Path(__file__).parent
began=time.monotonic();peak=0
def guard():
 global peak
 sample=sample_memory(os.getpid(),'source27_saved156800_compact_recovery_observer');m=sample['system_memory'];peak=max(peak,sample['worker_private_memory']['private_bytes'])
 require(peak<536870912 and time.monotonic()-began<180 and m['commit_available_bytes']>=2000000000+268435456
  and m['physical_available_bytes']>=1500000000+268435456 and shutil.disk_usage('E:/').free>=100000000000,'Unchanged metadata guard/fullprefetch/E100GB')
 return sample
def main():
 me=psutil.Process();me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);parent=me.parent();initial=guard();m=initial['system_memory']
 require(m['commit_available_bytes']>=2000000000+268435456+536870912 and m['physical_available_bytes']>=1500000000+268435456+536870912,'Full prospective512MiB metadata allowance')
 require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'and Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad')),'CPU-only observer in E storage environment')
 out=A/'actual_saved156800_recovery_01';require(not out.exists(),'ONE fresh natural metadata seal')
 progress=read_json(B/'progress/latest.json');require(progress['producer']=={'pid':9704,'create_time':1792727133.7017913,'ppid':32860}
  and progress['actual_optimizer_updates']>=156820 and not progress['optimizer_poisoned']and not progress['schedule_poisoned']
  and progress['last_saved_updates']>=156800 and progress['planned_native_updates']==401072 and progress['administrative_update_stop']is None,'Real positive post156800 source27, unchanged full goal')
 require(not(I/'actual_failure.json').exists(),'No new actual training failure')
 checkpoint=B/'checkpoints/update_000156800';pointer=read_json(checkpoint/'pointer.json');persist=read_json(checkpoint/'persistence_verified.json')
 require(pointer['committed_updates']==156800 and persist['passed']is True and persist['optimizer_updates']==156800
  and persist['named_tensors_verified']==854 and persist['weights_verified']==207 and persist['Adam_states_verified']==215
  and persist['saved_semantic_state_verified']['passed']is True
  and persist['independent_pre_save_state_sha256']==persist['saved_state_sha256'],'Natural156800 full854/207/215 semantic/PRE-SAVED equality publication')
 require(progress['clocks']==[progress['actual_optimizer_updates']+x for x in(64250,2000,62250)],'All three actual clocks preserved')
 review_path=N/'ROOT_source27_cushion2250000000_saved156600_postrestore_selector_execution_review_v1.json';review=read_json(review_path)
 config=read_json(I/'configuration.json');require(all(config[k]==v for k,v in review['future_7_runtime_bindings_NOT_actual_restore'].items()),'All7 actual runtime bindings exact')
 restore=read_json(I/'actual_parent_restore.json');branch=read_json(I/'actual_journal_branch.json');ready=read_json(I/'CPU/actual_CPU_ready.json')
 require(restore['passed']is True and restore['all207_threeAdam_all4RNG_match_actual_pre_save']is True and restore['next_draw_packet_match']is True
  and restore['restore_occurred_only_at_startup_before_native']is True and branch['predecessor']==review['postrestore_selector_resume']
  and branch['native_updates_or_draws_in_copy']==0 and branch['historical_unsaved_tail_updates']==list(range(156601,156770)), 'Actual full restore before copied branch; new startup predecessor bound')
 saved=review['saved_journal_prefix'];require(branch['saved_prefix']==saved and saved['offset']==317194933,'Exact original18 published saved prefix binding')
 expected={'path':str(B/'journals/commits_000020.jsonl'),'offset':317194933,'sha256':saved['sha256']}
 require(branch['new_active_journal']==expected and Path(progress['journal_prefix']['path'])==Path(expected['path']),'New active20 only, no orphan promotion')
 require(ready['producer_identity']=={'create_time':1792727182.5415392,'pid':4392,'ppid':39368}and ready['source']['protocol']==review['CPU_source'],'Exact admitted CPU10 producer/source')
 families=[('native',9704,1792727133.7017913),('native_interpreter_launcher',32860,1792727133.694484),('wrapper',12328,1792727133.3462756),('CPU',4392,1792727182.5415392),('CPU_interpreter_launcher',39368,1792727182.5201604)]
 observed=[]
 for role,pid,ct in families:
  q=psutil.Process(pid);require(abs(q.create_time()-ct)<.001,'Current owned identity exactalive '+role);observed.append({'role':role,'pid':pid,'create_time':ct,'ppid':q.ppid(),'exact_alive':True})
 oldstats=[]
 for item in(review['original18_closed_STAT_only'],review['orphan19_preserved_STAT_only']):
  current=Path(item['path']).stat();require(current.st_size==item['bytes']and current.st_mtime_ns==item['mtime_ns'],'Preserved original18 and inactive orphan19 STAT only');oldstats.append(item)
 out.mkdir();write_json(out/'positive_progress_observed.json',progress)
 raw=[];events=[];journal=Path(expected['path'])
 with journal.open('rb')as stream:
  stream.seek(saved['offset'])
  for update in range(156601,156771):
   guard();line=stream.readline(1048576);require(line.endswith(b'\n')and 0<len(line)<1048576,'One complete bounded NEW actual event')
   e=json.loads(line);n=e['native'];ev=e['prepared_evidence']
   require(e['optimizer_updates']==n['update']==ev['actual_positive_native_update']==update
    and math.isfinite(n['mean_objective'])and math.isfinite(n['global_grad_norm'])and n['global_grad_norm']>0
    and len(n['source_ids'])==len(n['languages'])==16 and n['languages'][:14]==['tr']*14
    and all(x!='tr'for x in n['languages'][14:]),'Genuine ordered positive native/16 IDs14TR+2replay')
   raw.append(line);events.append(e)
  end=stream.tell()
 for before,after in zip(events,events[1:]):require(after['broad_draws']==before['broad_draws']+14 and after['replay_updates']==before['replay_updates']+1 and after['cleaner_draws']==before['cleaner_draws'],'Ordered exact broad/replay/cleaner cursors')
 snapshot=out/'actual170_new_positive_events156601_156770.jsonl';snapshot.write_bytes(b''.join(raw));require(end-saved['offset']==snapshot.stat().st_size,'Exact only-new segment bytes')
 require('numpy'not in sys.modules and 'torch'not in sys.modules,'No payload/model/NumPy/Torch/GPU imports')
 pins={'execution_review':record(review_path),'ROOT_closed_boot_restore_bindings':record(N/'ROOT_source27_actual_boot_runtime_bindings_and_restore_v1.json'),
  'boot':record(I/'boot.json'),'restore':record(I/'actual_parent_restore.json'),'branch':record(I/'actual_journal_branch.json'),'configuration':record(I/'configuration.json'),
  'CPU_ready':record(I/'CPU/actual_CPU_ready.json'),'current_child':record(A/'current_child.json'),'pointer156800':record(checkpoint/'pointer.json'),'persistence156800':record(checkpoint/'persistence_verified.json'),
  'positive_progress_observed':record(out/'positive_progress_observed.json'),'actual170_new_snapshot':record(snapshot),'sealer_source':record(__file__)}
 report={'kind':'ASR18_actual_source27_saved156600_cushion225_recovered156800_candidate_v1','passed':True,'owner':'/root/memory_diagnosis','owned_session':56620,
  'initial_updates':156600,'actual_progress':progress,'new170_range':[156601,156770],'new170_count':170,'new170_segment':{'path':str(journal),'start_offset':saved['offset'],'end_offset':end,'bytes':end-saved['offset'],'sha256':record(snapshot)['sha256']},
  'actual170_snapshot':record(snapshot),'first_new_event':events[0],'last_new_event':events[-1],'all_positive_ordered_and_finite':True,'actual_all5_exact_alive':observed,
  'genuine_restore_before_saved_prefix_copy_then_native':True,'all7_runtime_bindings_exact':True,'original18_and_inactive_orphan19_preserved_STAT_only':oldstats,
  'new225_policy_unchanged':review['native_update_commit_pressure_policy'],'old169_historical_attempts_genuinely_redone_plus_one':True,'natural156800_full854_207_215_semantic_PRE_SAVED_equality':True,
  'no_old_whole_prefix169tail_tensor_or_audio_read':True,'no_training_source_environment_reader_or_resource_change':True,'no_fit_speed_saving_prevention_guarantee':True,'full401072_continues_no_finite_stop':True,
  'ROOT_actual_recovery_acceptance_pending':True,'sealer_identity':{'pid':me.pid,'create_time':me.create_time(),'ppid':me.ppid()},'sealer_launcher':{'pid':parent.pid,'create_time':parent.create_time(),'ppid':parent.ppid()},
  'guard_initial':initial,'guard_final':guard(),'peak_private_bytes':peak,'evidence':pins}
 write_json(out/'actual_recovery_status.json',report);print(json.dumps({'status':record(out/'actual_recovery_status.json'),'sealer_identity':report['sealer_identity'],'sealer_launcher':report['sealer_launcher'],'segment':report['new170_segment'],'progress':progress['actual_optimizer_updates']}),flush=True)
if __name__=='__main__':main()
