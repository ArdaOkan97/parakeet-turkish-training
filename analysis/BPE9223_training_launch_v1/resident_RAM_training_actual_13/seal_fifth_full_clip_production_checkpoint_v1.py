"""Compact actual consumed-certificate/checkpoint observation, no training action."""
import copy,gzip,json,sys
from pathlib import Path
import psutil
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import digest
N=Path(__file__).parent;B=Path(r'E:\gpu_training_scratchpad\runs\BPE9223_resident_RAM_20261018_01');I=B/'invocations/owner_000012';CK=B/'checkpoints/update_000134600';F=N.parent/'resident_RAM_original_interval_post134529_failure_actual_v1'
latest=read_json(B/'latest_checkpoint.json');progress_bytes=(B/'progress/latest.json').read_bytes();progress=json.loads(progress_bytes)
require(latest['committed_updates']>=134600 and latest['full_state_verified'] is True and progress['actual_optimizer_updates']>134600,'Actual134600 published fullstate then resumed positives')
require(progress['producer']==read_json(I/'boot.json')['producer'] and progress['optimizer_poisoned'] is False and progress['schedule_poisoned'] is False,'Same healthy source16 actual producer')
pointer=read_json(CK/'pointer.json');archive=record(CK/'prepared_metadata.json.gz')
require(archive==pointer['identity']['caller_identity']['preparation_archive'],'Actual fullcheckpoint binds consumed metadata archive')
with gzip.open(archive['path'],'rb')as f:parts=json.load(f)
records=[v for v in parts if v['native']['update']==134530];require(len(records)==1,'One real134530 native receipt')
part=records[0];metadata=part['metadata'];ref=metadata['same_pass_clean_evidence'][3];entry=metadata['rows'][3];proof=ref['before_draw_geometry_proof'];certificate=read_json(F/'fifth_original_full_clip_geometry_certificate_v1.json')
require(metadata['global_update']==part['native']['update']==134530 and entry['source_id']==certificate['original_row']['unique_id'] and entry['samples']==200064 and entry['encoder_frames']==157 and len(entry['full_target_ids'])==82,'Exact actual original physical source/targets/geometry consumed')
require(entry['input_features']['shape']==[1,1251,128] and entry['attention_mask']['shape']==[1,1251]
 and proof['historical_mel_frames']==1250 and proof['actual_physical_mel_frames']==1250,
 'Production physical feature shape and explicit conservative historical mel reconciliation')
require(proof['certificate']==record(F/'fifth_original_full_clip_geometry_certificate_v1.json')
 and proof['actual_geometry']==certificate['actual_geometry'] and proof['historical_declaration']==certificate['original_declaration']
 and ref['expected_audio']['waveform_sha256']==certificate['clean_f32_sha256'] and ref['full_target_ids']==certificate['full_target_ids'],'Actual production correction proof/PCM/fulltargets')
branch=read_json(I/'actual_journal_branch.json');saved=branch['saved_prefix'];journal=Path(branch['new_active_journal']['path'])
raw_segment=[];events=[]
require(saved['offset']==273582680 and journal==B/'journals/commits_000012.jsonl','Exact saved134400 prefix and new active journal')
with journal.open('rb')as source:
 source.seek(saved['offset'])
 for update in range(134401,134531):
  line=source.readline(131073);require(line.endswith(b'\n')and len(line)<=131072,'Bounded complete new event')
  event=json.loads(line)
  require(event['optimizer_updates']==event['native']['update']==event['prepared_evidence']['actual_positive_native_update']==update,
   'Actual130 ordered new positive updates, not copied historical tail')
  raw_segment.append(line);events.append(event)
 new_segment_end_offset=source.tell()
segment=N/'actual_new130_positive_events_134401_134530.jsonl';require(not segment.exists(),'One bounded new segment');segment.write_bytes(b''.join(raw_segment))
verified_journal=record(segment)
require(events[-1]['prepared_evidence']['metadata_sha256']==digest(metadata),'Actual corrected consumed metadata exactly bound to positive journal event')
native_snapshot=copy.deepcopy(part['native'])
route_map=native_snapshot['joint_work_unit_allocation_routes']
require(set(route_map)=={str(i)for i in range(len(metadata['rows']))},'Source-declared full integer-index route map serialized by JSON')
native_snapshot['joint_work_unit_allocation_routes']={int(k):v for k,v in route_map.items()}
require(events[-1]['native']['full_native_receipt_sha256']==digest(native_snapshot),'Commit-time full native digest after restoring ONLY original integer route-key types')
require(all(events[-1]['native'][k]==part['native'][k] for k in ('update','phase','source_ids','languages','mean_objective','global_grad_norm')),'All compact committed native fields equal archived receipt')
persistence=read_json(CK/'persistence_verified.json');require(persistence['named_tensors_verified']==854 and persistence['Adam_states_verified']==215,'Actual854/215 persistence')
require(persistence['passed']and persistence['weights_verified']==207 and persistence['independent_pre_save_state_sha256']==persistence['saved_state_sha256'],'Actual207 weights and PRE/SAVED equality')
restore=read_json(I/'actual_parent_restore.json');require(restore['passed']and restore['all207_threeAdam_all4RNG_match_actual_pre_save']and restore['next_draw_packet_match']and restore['restore_occurred_only_at_startup_before_native'],'Fresh exact saved134400 restore before native')
who=progress['producer'];require(abs(psutil.Process(who['pid']).create_time()-who['create_time'])<.001,'Actual same live native')
CPU=read_json(I/'CPU/actual_CPU_ready.json');cpu=CPU['producer_identity'];require(abs(psutil.Process(cpu['pid']).create_time()-cpu['create_time'])<.001,'Same actual resident CPU')
observed=N/'actual_verified134600_positive_progress.json';require(not observed.exists(),'Unique observation');observed.write_bytes(progress_bytes)
report={'kind':'ASR18_actual_original_terminal_interval_corrected_and_saved134600_continuation_v1','passed':True,'owned_session':64681,'native':who,'CPU':cpu,'driver_source':progress['source'],'configuration':read_json(I/'boot.json')['configuration'],'ROOT_execution_review':record(N.parent/'ROOT_fifth_original_full_clip_saved134400_execution_review_v1.json'),'actual_CPU_review':record(N.parent/'ROOT_fifth_original_full_clip_geometry_actual_CPU_review_v1.json'),'actual_startup_restore':record(I/'actual_parent_restore.json'),'actual_journal_branch':record(I/'actual_journal_branch.json'),'actual_corrected_update':134530,'consumed_row_index':3,'consumed_row':entry,'actual_consumed_geometry_proof':proof,'actual_full_clean_PCM_sha256':ref['expected_audio']['waveform_sha256'],'certified_full_native_PCM_sha256':certificate['native_f32_sha256'],'new130_segment_source_offset':saved['offset'],'new130_segment_end_offset':new_segment_end_offset,'actual_native_receipt_archived_JSON_sha256':digest(part['native']),'actual_native_receipt_commit_type_sha256':digest(native_snapshot),'JSON_integer_route_keys_restored_for_commit_digest':True,'actual_native_receipt_summary':{k:part['native'][k] for k in ('update','phase','source_ids','languages','mean_objective','global_grad_norm')},'actual_committed134530_fields':events[-1],'actual_new130_positive_journal_segment':verified_journal,'checkpoint_pointer':record(CK/'pointer.json'),'checkpoint_persistence':record(CK/'persistence_verified.json'),'consumed_metadata_archive':archive,'current_progress':record(observed),'actual_optimizer_updates':progress['actual_optimizer_updates'],'native_updates_this_process':progress['native_updates_this_process'],'last_verified_save':134600,'old_whole_journal_and_tail_preserved':branch['preserved_original'],'old_unsaved_tail_stays_historical':branch['historical_unsaved_tail_updates'],'all_original_physical_samples_used_once':True,'original_annotation_indexes_labels_sampler_order_and_fullpass_denominator_unchanged':True,'full_run_target':401072,'administrative_update_stop':None,'training_continues':True,'no_new_speed_study':True}
p=N/'actual_corrected_fifth_full_clip_saved134600_status.json';require(not p.exists(),'One compact receipt');write_json(p,report);print(json.dumps({'receipt':record(p),'actual_updates':progress['actual_optimizer_updates'],'checkpoint':report['checkpoint_pointer'],'native':who}),flush=True)
