"""INACTIVE one metadata recovery observer after natural restored+200 and +220.
Only new25 saved-offset..old actual+1 events; original24 STAT only. No archive,
tensor, audio, model, native/GC operation or source/reader/environment mutation.
Requires separate final exact ROOT execution and actual boot receipt bindings.
"""
import argparse,hashlib,json,math,os,shutil,sys,time
from pathlib import Path
import psutil
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish';R=P/'work/research';sys.path.insert(0,str(S))
from resource_watchdog_v2 import sample_memory
F=Path(__file__).resolve().parent;B=Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01')
I=B/'invocations/owner_000025';A=F.parent.parent/'resident_RAM_training_actual_26'
T=Path('E:/gpu_training_scratchpad/analysis/BPE9223_Trackio_live_v1');BEGIN=time.monotonic();PEAK=0
def require(v,m):
 if not v:raise ValueError(m)
def read(p):
 p=Path(p);require(p.stat().st_size<=8*1024**2,'Bounded compact JSON only');return json.loads(p.read_text(encoding='utf-8-sig'))
def pin(p):
 p=Path(p).resolve();require(p.stat().st_size<=8*1024**2,'Bounded source/metadata pin only');b=p.read_bytes()
 return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def write(p,v):
 with Path(p).open('x',encoding='utf-8',newline='\n') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def who(p):return {'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()}
def alive(v):
 try:return abs(psutil.Process(v['pid']).create_time()-v['create_time'])<.001
 except psutil.NoSuchProcess:return False
def guard(admission=False):
 global PEAK
 s=sample_memory(os.getpid(),'source36_dynamic_natural_recovery_metadata');m=s['system_memory'];PEAK=max(PEAK,s['worker_private_memory']['private_bytes'])
 require(s['paired_sampling_within_one_second'] and PEAK<536870912 and time.monotonic()-BEGIN<180
  and m['commit_available_bytes']>=(5055306368 if admission else 2268435456)
  and m['physical_available_bytes']>=(2305306368 if admission else 1768435456)
  and shutil.disk_usage('E:/').free>=100000000000,'Metadata512MiB/180s/full225+prospective512/FULLprefetch/floors/E100GB')
 return s
def main(a):
 me=psutil.Process();me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
 require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1' and os.environ.get('PYTHONUTF8')=='1'
  and all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'))
  and Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad')),'CPU-only E observer environment')
 review=read(a.execution_review)
 require(review['accepted'] is True and review['reviewer']=='/root' and review['recovery_observer_source']==pin(__file__),
  'Separate final exact ROOT execution binds this once-only observer')
 root_boot=read(a.ROOT_boot_review)
 require(root_boot.get('accepted',root_boot.get('passed')) is True,'Reuse separately accepted actual boot/runtime7/restore-order ROOT receipt')
 closure=read(review['gc_refresh_resume']['path']);source=pin(R/'run_ASR18_BPE9223_resident_RAM_training_v36.py')
 require(review['driver_source']==source,'Exact admitted source36')
 restored=read(review['initial_checkpoint']['path'])['committed_updates'];old_actual=closure['actual_optimizer_updates']
 require(restored==closure['last_fully_saved_optimizer_updates'] and restored%200==0 and restored<=old_actual<restored+200,
  'Actual saved frontier and honest historical optimizer/journal gap')
 target=restored+200;last_new=old_actual+1;count=last_new-restored
 out=A/'actual_dynamic_natural_recovery_01';require(not out.exists(),'ONE fresh actual recovery observation')
 initial=guard(True);progress=read(B/'progress/latest.json');boot=read(I/'boot.json')
 require(progress['source']==source and progress['producer']==boot['producer'] and progress['actual_optimizer_updates']>=target+20
  and progress['last_saved_updates']>=target and not progress['optimizer_poisoned'] and not progress['schedule_poisoned']
  and progress['planned_native_updates']==401072 and progress['administrative_update_stop'] is None
  and not(I/'actual_failure.json').exists(),'Healthy same full401072 job after natural checkpoint and actual postpositive')
 require(progress['clocks']==[progress['actual_optimizer_updates']+x for x in (64250,2000,62250)],'All original three clock offsets')
 cp=B/'checkpoints'/('update_'+str(target).zfill(9));pointer=read(cp/'pointer.json');persist=read(cp/'persistence_verified.json')
 require(pointer['committed_updates']==target and persist['optimizer_updates']==target and persist['passed'] is True
  and persist['named_tensors_verified']==854 and persist['weights_verified']==207 and persist['Adam_states_verified']==215
  and persist['saved_semantic_state_verified']['passed'] is True and persist['independent_pre_save_state_sha256']==persist['saved_state_sha256']
  and pointer['journal_prefix']==persist['journal_prefix'],'Natural full854/207/215 semantic independent PRE-SAVED equality; no payload')
 configuration=read(I/'configuration.json');expected=read(review['source_configuration']['path'])
 expected.update(review['future_seven_runtime_bindings']);require(configuration==expected,'Every actual configuration value, all7 runtime bindings exact')
 restore=read(I/'actual_parent_restore.json');branch=read(I/'actual_journal_branch.json')
 require(restore['passed'] is True and restore['all207_threeAdam_all4RNG_match_actual_pre_save'] is True
  and restore['next_draw_packet_match'] is True and restore['restore_occurred_only_at_startup_before_native'] is True
  and branch['predecessor']==review['gc_refresh_resume'] and branch['native_updates_or_draws_in_copy']==0,
  'Actual all-state restore before saved-prefix-only copy; closed boot ROOT receipt reused')
 saved=read(review['initial_checkpoint']['path'])['journal_prefix']
 require(branch['saved_prefix']==saved and branch['historical_unsaved_tail_updates']==list(range(restored+1,closure['published_journal_optimizer_updates']+1)),
  'Only published old journal tail is historical; unjournaled counter advance remains honest/redo required')
 binding={'path':str(B/'journals/commits_000025.jsonl'),'offset':saved['offset'],'sha256':saved['sha256']}
 require(branch['new_active_journal']==binding and Path(progress['journal_prefix']['path'])==Path(binding['path']),'Only exact new25 active branch')
 old=closure['closed_old_journal_STAT'];st=Path(old['path']).stat()
 require(st.st_size==old['bytes'] and st.st_mtime_ns==old['mtime_ns'],'Original24 STAT-only untouched; never prefix/whole/tail reread')
 ready_candidates=[I/'CPU/actual_CPU_ready.json']
 dev=(restored//4000+1)*4000
 if dev<=target:ready_candidates.append(I/('CPU_after_periodic_DEV_'+str(dev).zfill(9))/'actual_CPU_ready.json')
 available=[(p,read(p)) for p in ready_candidates if p.exists()]
 active=[(p,r) for p,r in available if alive(r['producer_identity'])]
 require(len(active)==1,'One honest current feature CPU cohort; no stale startup CPU claimed alive')
 ready_path,ready=active[0];cpu=ready['producer_identity']
 require(ready['source']['protocol']==review['CPU_source'] and ready['packet_credit']['logical_packet_maximum']==2
  and ready['packet_credit']['helper_source']==review['packet_credit_policy']['helper_source'],'Actual unchanged CPU13/MAX2/helper identity')
 child=read(A/'current_child.json');launcher=child['launcher']
 family=[('native',boot['producer']),('native_interpreter_launcher',launcher),('wrapper',who(psutil.Process(launcher['ppid']))),
  ('CPU',cpu),('CPU_interpreter_launcher',who(psutil.Process(cpu['ppid'])))]
 require(all(alive(w) for _,w in family),'All5 exact current runtime identities alive')
 out.mkdir();write(out/'positive_progress_observed.json',progress)
 raw=[];events=[];journal=Path(binding['path']);total=0
 with journal.open('rb') as f:
  f.seek(saved['offset'])
  for u in range(restored+1,last_new+1):
   guard();line=f.readline(1048576);total+=len(line)
   require(line.endswith(b'\n') and 0<len(line)<1048576 and total<=4*1024**2,'Only bounded complete NEW event range')
   e=json.loads(line);n=e['native'];ev=e['prepared_evidence']
   require(e['optimizer_updates']==n['update']==ev['actual_positive_native_update']==u
    and math.isfinite(n['mean_objective']) and n['mean_objective']>0 and math.isfinite(n['global_grad_norm']) and n['global_grad_norm']>0
    and ev['producer_source_sha256']==review['CPU_source']['sha256']
    and e['broad_draws']==14*u and e['replay_updates']==u and e['cleaner_draws']==0
    and len(n['source_ids'])==len(n['languages'])==16 and n['languages'][:14]==['tr']*14
    and all(x!='tr' for x in n['languages'][14:]),'Genuine ordered nativepositive finiteobjective/grad/CPU13/16IDs14TR2replay/absolute cursors')
   raw.append(line);events.append(e)
  end=f.tell()
 require(len({x['prepared_evidence']['token'] for x in events})==count,'Unique actual new prepared tokens')
 snap=out/'only_new_redone_positives_through_old_actual_plus_one.jsonl'
 with snap.open('xb') as f:f.write(b''.join(raw));f.flush();os.fsync(f.fileno())
 require(end-saved['offset']==snap.stat().st_size,'Exact only-new range offset/bytes')
 reader={}
 for name in ('current_status.json','mirror_state.json','owner.json'):
  p=T/name;value=read(p);write(out/('reader_'+name),value);reader[name]={'source_pin':pin(p),'bounded_capture':pin(out/('reader_'+name))}
 require(alive({'pid':26892,'create_time':1792652904.3359847}),'Healthy unchanged reader identity, advancing catchup not restarted')
 require(not any(k in sys.modules for k in ('torch','numpy','av')),'No model/audio/scientific runtime imported')
 report={'kind':'ASR18_source36_dynamic_saved_checkpoint_recovery_candidate_v1','passed':True,
  'execution_review':pin(a.execution_review),'closed_ROOT_boot_review':pin(a.ROOT_boot_review),'initial_updates':restored,
  'natural_saved_updates':target,'old_actual_optimizer_frontier':old_actual,'new_positive_count':count,'new_positive_range':[restored+1,last_new],
  'new_positive_snapshot':pin(snap),'new_positive_segment':{'path':str(journal),'start_offset':saved['offset'],'end_offset':end,
   'bytes':end-saved['offset'],'sha256':pin(snap)['sha256']},'first_new_event':events[0],'last_new_event':events[-1],
  'actual_progress':progress,'all5_exact_alive':[{'role':role,**w,'exact_alive':True} for role,w in family],
  'boot':pin(I/'boot.json'),'restore':pin(I/'actual_parent_restore.json'),'branch':pin(I/'actual_journal_branch.json'),
  'configuration':pin(I/'configuration.json'),'CPU_ready':pin(ready_path),'current_child':pin(A/'current_child.json'),
  'pointer':pin(cp/'pointer.json'),'persistence':pin(cp/'persistence_verified.json'),'reader_bounded_raw_captures':reader,
  'old24_STAT_only':old,'full854_207_215_semantic_PRE_SAVED_and_postpositive':True,
  'full_restore_before_savedprefix_copy_then_genuine_redo':True,'all7_runtime_and_remaining_configuration_exact':True,
  'original225_MAX2_LIVE1_full_reservations_guards_science_unchanged':True,'one_postDEV_refresh_policy':configuration['one_postDEV_gc_refresh_policy'],
  'one_event_real_full854_semantic_equality_and_long_speed_validation_not_assessed_by_recovery_observer':True,'human_speed_fix_open':True,
  'sealer_identity':who(me),'sealer_launcher_identity':who(me.parent()),'initial_guard':initial,'final_guard':guard(),'peak_private_bytes':PEAK,
  'no_old_journal_prefix_whole_tail_archive_tensor_audio_GC_GPU_or_production_mutation':True}
 write(out/'actual_recovery_status.json',report);print(json.dumps({'status':pin(out/'actual_recovery_status.json'),
  'restored':restored,'natural':target,'new_range':[restored+1,last_new],'segment':report['new_positive_segment']}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--execution-review',required=True);p.add_argument('--ROOT-boot-review',required=True);main(p.parse_args())
