"""Metadata-only natural recovery observer; no training, loader or payload reads.
Reads ONLY70 new journal24 lines from the known saved offset, never old prefixes.
Requires natural170600 persistence and positive170620+ before its ONE execution.
"""
import argparse,gzip,hashlib,io,json,math,os,shutil,sys,time
from pathlib import Path
import psutil
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
from resource_watchdog_v2 import sample_memory
from bpe_broad_production_state_v1 import digest
N=Path('E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1');B=Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01');I=B/'invocations/owner_000024';A=N/'resident_RAM_training_actual_25';F=Path(__file__).resolve().parent
began=time.monotonic();peak=0
def guard():
 global peak
 sample=sample_memory(os.getpid(),'source33_saved170600_compact_recovery_observer');m=sample['system_memory'];peak=max(peak,sample['worker_private_memory']['private_bytes'])
 require(peak<536870912 and time.monotonic()-began<180 and m['commit_available_bytes']>=2000000000+268435456
  and m['physical_available_bytes']>=1500000000+268435456 and shutil.disk_usage('E:/').free>=100000000000,'Unchanged metadata guard/fullprefetch/E100GB')
 return sample
def main(a):
 me=psutil.Process();me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);parent=me.parent();initial=guard();m=initial['system_memory']
 require(m['commit_available_bytes']>=2000000000+268435456+536870912 and m['physical_available_bytes']>=1500000000+268435456+536870912,'Full prospective512MiB metadata allowance')
 require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'and Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad')),'CPU-only observer in E storage environment')
 out=A/'actual_saved170600_recovery_01';require(not out.exists(),'ONE fresh natural metadata seal')
 boot=read_json(I/'boot.json');progress=read_json(B/'progress/latest.json');require(progress['producer']==boot['producer']
  and progress['actual_optimizer_updates']>=170620 and not progress['optimizer_poisoned']and not progress['schedule_poisoned']
  and progress['last_saved_updates']>=170600 and progress['planned_native_updates']==401072 and progress['administrative_update_stop']is None,'Real positive post170600 source33, unchanged full goal')
 require(not(I/'actual_failure.json').exists(),'No new actual training failure')
 checkpoint=B/'checkpoints/update_000170600';pointer=read_json(checkpoint/'pointer.json');persist=read_json(checkpoint/'persistence_verified.json')
 require(pointer['committed_updates']==170600 and persist['passed']is True and persist['optimizer_updates']==170600
  and persist['named_tensors_verified']==854 and persist['weights_verified']==207 and persist['Adam_states_verified']==215
  and persist['saved_semantic_state_verified']['passed']is True
  and persist['independent_pre_save_state_sha256']==persist['saved_state_sha256'],'Natural170600 full854/207/215 semantic/PRE-SAVED equality publication')
 require(progress['clocks']==[progress['actual_optimizer_updates']+x for x in(64250,2000,62250)],'All three actual clocks preserved')
 review_path=Path(a.ROOT_review);review=read_json(review_path)
 config=read_json(I/'configuration.json');require(all(config[k]==v for k,v in read_json(review['runtime7']['path'])['future_runtime_values'].items()),'All7 actual runtime bindings exact')
 restore=read_json(I/'actual_parent_restore.json');branch=read_json(I/'actual_journal_branch.json');ready=read_json(I/'CPU/actual_CPU_ready.json')
 require(restore['passed']is True and restore['all207_threeAdam_all4RNG_match_actual_pre_save']is True and restore['next_draw_packet_match']is True
  and restore['restore_occurred_only_at_startup_before_native']is True and branch['predecessor']==review['eighth_original_resume']
  and branch['native_updates_or_draws_in_copy']==0 and branch['historical_unsaved_tail_updates']==list(range(170401,170470)), 'Actual full restore before copied branch; new startup predecessor bound')
 saved=read_json(review['initial_checkpoint']['path'])['journal_prefix'];require(branch['saved_prefix']==saved and saved['offset']==344297457,'Exact source30 published170400 prefix binding')
 expected={'path':str(B/'journals/commits_000024.jsonl'),'offset':344297457,'sha256':saved['sha256']}
 require(branch['new_active_journal']==expected and Path(progress['journal_prefix']['path'])==Path(expected['path']),'New active23 only, historical23 retained unchanged')
 require(ready['source']['protocol']==review['CPU_source']and ready['packet_credit']['logical_packet_maximum']==2
  and ready['packet_credit']['helper_source']==review['packet_credit_policy']['helper_source'],'Actual CPU13 production credit2/source/helper identity')
 cpu=ready['producer_identity'];cpu_launcher=psutil.Process(cpu['ppid'])
 child=read_json(A/'current_child.json');launcher=child['launcher'];wrapper=psutil.Process(launcher['ppid'])
 families=[('native',boot['producer']['pid'],boot['producer']['create_time']),
  ('native_interpreter_launcher',launcher['pid'],launcher['create_time']),
  ('wrapper',wrapper.pid,wrapper.create_time()),('CPU',cpu['pid'],cpu['create_time']),
  ('CPU_interpreter_launcher',cpu_launcher.pid,cpu_launcher.create_time())]
 observed=[]
 for role,pid,ct in families:
  q=psutil.Process(pid);require(abs(q.create_time()-ct)<.001,'Current owned identity exactalive '+role);observed.append({'role':role,'pid':pid,'create_time':ct,'ppid':q.ppid(),'exact_alive':True})
 old=read_json(review['eighth_original_resume']['path'])['journal']['original_STAT_only'];current=Path(old['path']).stat()
 require(current.st_size==old['bytes']and current.st_mtime_ns==old['mtime_ns'],'Original closed23 STAT unchanged; no old prefix/whole/tail reread')
 oldstats=[old]
 out.mkdir();write_json(out/'positive_progress_observed.json',progress)
 raw=[];events=[];journal=Path(expected['path'])
 with journal.open('rb')as stream:
  stream.seek(saved['offset'])
  for update in range(170401,170471):
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
 snapshot=out/'actual70_new_positive_events170401_170470.jsonl';snapshot.write_bytes(b''.join(raw));require(end-saved['offset']==snapshot.stat().st_size,'Exact only-new segment bytes')
 # One NEW standard170600 metadata archive, never audio or tensor data.
 archive=checkpoint/'prepared_metadata.json.gz';require(archive.stat().st_size<=8*1024**2,'Bounded new standard archive compressed extent')
 archive_raw=archive.read_bytes();archive_pin={'path':str(archive),'bytes':len(archive_raw),'sha256':hashlib.sha256(archive_raw).hexdigest()}
 with gzip.GzipFile(fileobj=io.BytesIO(archive_raw),mode='rb')as zipped:uncompressed=zipped.read(32*1024**2+1)
 require(len(uncompressed)<=32*1024**2,'Bounded new standard metadata JSON, no whole tensor/audio')
 guard();prepared=json.loads(uncompressed)
 require(len(prepared)==200 and [x['native']['update']for x in prepared]==list(range(170401,170601)),
  'All200 standard archive records retain their original ordered actual update labels')
 selected=prepared[69];metadata=selected['metadata'];native=selected['native'];row=metadata['rows'][5]
 reference=metadata['same_pass_clean_evidence'][5];certificate_pin=review['original_clip_geometry_overlay']['certificate'];certificate=read_json(certificate_pin['path'])
 require(native['update']==170470 and row['source_id']==native['source_ids'][5]==certificate['original_row']['unique_id']
  and row['full_target_ids']==reference['full_target_ids']==certificate['full_target_ids']and row['target_tokens']==99
  and row['samples']==273624 and row['encoder_frames']==214
  and row['clean_PCM_sha256']=='3a9e9240d7348caf1b2a26f53b98d362836ec6cad088a747cc31008a0cf9659b'
  and row['input_features']['shape']==[1,1711,128]and row['attention_mask']['shape']==[1,1711],
  'Actual repaired170470 row6 original full99/clean273624/hash/feature geometry reached native')
 proof=reference['before_draw_geometry_proof']
 require(reference['whole_original_clip']is True and reference['actual_geometry_and_heldout_checked']is True
  and proof['certificate']==certificate_pin and proof['source']==review['original_clip_geometry_overlay']['source']
  and proof['historical_declaration']==certificate['original_declaration']
  and proof['actual_geometry']==certificate['actual_geometry']and proof['all_original_physical_samples_used_once']is True
  and proof['original_annotations_plan_labels_and_encoder_geometry_unchanged']is True
  and proof['historical_declaration']['native_samples']==820896 and proof['historical_declaration']['clean16k_samples']==273632,
  'Actual eighth certificate/alias9/full original physical PCM and unchanged larger historical reservations')
 require(digest(metadata)==events[-1]['prepared_evidence']['metadata_sha256'],'Actual complete preparation metadata matches directnew70 commit')
 raw_native_JSON_digest=digest(native)
 routes=native['joint_work_unit_allocation_routes']
 require(type(routes)is dict and set(routes)=={str(i)for i in range(16)},'Only known original16 integer routekeys may be restored')
 integer_native=dict(native);integer_native['joint_work_unit_allocation_routes']={int(k):v for k,v in routes.items()}
 require(digest(integer_native)==events[-1]['native']['full_native_receipt_sha256'],
  'Exact full native receipt after ONLY original integer routekeys restored; raw JSON digest remains distinct')
 corrected={'global_update':170470,'archive_index':69,'consumed_row_index':6,'original_ordinal':332560,
  'row':row,'same_pass_clean_reference':reference,'full_native_receipt':native,
  'raw_native_JSON_string_routekeys_digest':raw_native_JSON_digest,
  'original_integer_routekeys_native_commit_digest':digest(integer_native),
  'metadata_digest':digest(metadata),'archive':archive_pin,'archive_uncompressed_bytes':len(uncompressed),
  'archive_uncompressed_sha256':hashlib.sha256(uncompressed).hexdigest(),'original_historical_declared_native_samples':820896,
  'original_historical_declared_clean_samples':273632,'no_freed_capacity_credit':True}
 write_json(out/'actual170470_row6_preparation_and_native.json',corrected)
 require('numpy'not in sys.modules and 'torch'not in sys.modules,'No payload/model/NumPy/Torch/GPU imports')
 pins={'execution_review':record(review_path),'owner_boot_restore_bindings':record(A/'actual_owned_source33_boot_session.json'),
  'boot':record(I/'boot.json'),'restore':record(I/'actual_parent_restore.json'),'branch':record(I/'actual_journal_branch.json'),'configuration':record(I/'configuration.json'),
  'CPU_ready':record(I/'CPU/actual_CPU_ready.json'),'current_child':record(A/'current_child.json'),'pointer170600':record(checkpoint/'pointer.json'),'persistence170600':record(checkpoint/'persistence_verified.json'),
  'positive_progress_observed':record(out/'positive_progress_observed.json'),'actual70_new_snapshot':record(snapshot),'actual170470_row6':record(out/'actual170470_row6_preparation_and_native.json'),'standard170600_preparation_archive':archive_pin,'sealer_source':record(__file__)}
 report={'kind':'ASR18_actual_source33_eighth_original_interval_saved170400_recovered170600_candidate_v1','passed':True,'owner':'/root/memory_diagnosis','owned_session':read_json(A/'actual_owned_source33_boot_session.json')['owned_session'],
  'initial_updates':170400,'actual_progress':progress,'new70_range':[170401,170470],'new70_count':70,'new70_segment':{'path':str(journal),'start_offset':saved['offset'],'end_offset':end,'bytes':end-saved['offset'],'sha256':record(snapshot)['sha256']},
  'actual70_snapshot':record(snapshot),'first_new_event':events[0],'last_new_event':events[-1],'all_positive_ordered_and_finite':True,'actual_all5_exact_alive':observed,
  'genuine_restore_before_saved_prefix_copy_then_native':True,'all7_runtime_bindings_exact':True,'original23_preserved_STAT_only':oldstats,
  'approved225_policy_unchanged':review['native_update_commit_pressure_policy'],'actual_credit2_policy':review['packet_credit_policy'],'old23_historical_attempts_genuinely_redone_plus_one':True,'natural170600_full854_207_215_semantic_PRE_SAVED_equality':True,
  'no_old_whole_prefix23tail_tensor_or_audio_read':True,'no_training_source_environment_reader_or_resource_change':True,'actual_canonical_archive_representation_policy':review['prepared_metadata_archive_representation_policy'],
  'actual_signature_collect_observation_policy':review['fresh_metadata_signature_and_collect_observation_policy'],
  'signature_collect_scope_admission':record(I/'actual_fresh_signature_collect_observation_admission.json'),
  'human_recurrent_speed_fix_still_OPEN_not_validated_by_audio_repair':True,'no_fit_speed_saving_prevention_guarantee':True,'full401072_continues_no_finite_stop':True,
  'repaired170470_row6_production_evidence':corrected,'ROOT_actual_recovery_acceptance_pending':True,'sealer_identity':{'pid':me.pid,'create_time':me.create_time(),'ppid':me.ppid()},'sealer_launcher':{'pid':parent.pid,'create_time':parent.create_time(),'ppid':parent.ppid()},
  'guard_initial':initial,'guard_final':guard(),'peak_private_bytes':peak,'evidence':pins}
 write_json(out/'actual_recovery_status.json',report);print(json.dumps({'status':record(out/'actual_recovery_status.json'),'sealer_identity':report['sealer_identity'],'sealer_launcher':report['sealer_launcher'],'segment':report['new70_segment'],'progress':progress['actual_optimizer_updates']}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--ROOT-review',required=True);main(p.parse_args())
