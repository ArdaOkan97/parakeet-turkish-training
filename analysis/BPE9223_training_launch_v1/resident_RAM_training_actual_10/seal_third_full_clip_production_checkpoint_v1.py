"""Compact actual consumed-certificate/checkpoint observation, no training action."""
import copy,gzip,json,sys
from pathlib import Path
import psutil
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import digest
N=Path(__file__).parent;B=Path(r'E:\gpu_training_scratchpad\runs\BPE9223_resident_RAM_20261018_01');I=B/'invocations/owner_000009';CK=B/'checkpoints/update_000047800';F=N.parent/'resident_RAM_original_interval_post47727_failure_actual_v1'
latest=read_json(B/'latest_checkpoint.json');progress_bytes=(B/'progress/latest.json').read_bytes();progress=json.loads(progress_bytes)
require(latest['committed_updates']>=47800 and latest['full_state_verified'] is True and progress['actual_optimizer_updates']>47800,'Actual47800 published fullstate then resumed positives')
require(progress['producer']==read_json(I/'boot.json')['producer'] and progress['optimizer_poisoned'] is False and progress['schedule_poisoned'] is False,'Same healthy source10 actual producer')
pointer=read_json(CK/'pointer.json');archive=record(CK/'prepared_metadata.json.gz')
require(archive==pointer['identity']['caller_identity']['preparation_archive'],'Actual fullcheckpoint binds consumed metadata archive')
with gzip.open(archive['path'],'rb')as f:parts=json.load(f)
records=[v for v in parts if v['native']['update']==47731];require(len(records)==1,'One real47731 native receipt')
part=records[0];metadata=part['metadata'];ref=metadata['same_pass_clean_evidence'][2];entry=metadata['rows'][2];proof=ref['before_draw_geometry_proof'];certificate=read_json(F/'third_original_full_clip_geometry_certificate_v1.json')
require(metadata['global_update']==part['native']['update']==47731 and entry['source_id']==certificate['original_row']['unique_id'] and entry['samples']==239630 and entry['encoder_frames']==188 and len(entry['full_target_ids'])==91,'Exact actual original physical source/targets/geometry consumed')
require(entry['input_features']['shape']==[1,1498,128] and entry['attention_mask']['shape']==[1,1498]
 and proof['historical_mel_frames']==1497 and proof['actual_physical_mel_frames']==1497,
 'Production physical feature shape and explicit conservative historical mel reconciliation')
require(proof['certificate']==record(F/'third_original_full_clip_geometry_certificate_v1.json')
 and proof['actual_geometry']==certificate['actual_geometry'] and proof['historical_declaration']==certificate['original_declaration']
 and ref['expected_audio']['waveform_sha256']==certificate['clean_f32_sha256'] and ref['full_target_ids']==certificate['full_target_ids'],'Actual production correction proof/PCM/fulltargets')
branch=read_json(I/'actual_journal_branch.json');saved=branch['saved_prefix'];journal=Path(branch['new_active_journal']['path'])
with journal.open('rb')as f:
 f.seek(saved['offset']);events=[json.loads(f.readline())for _ in range(131)]
require([e['optimizer_updates']for e in events]==list(range(47601,47732)) and events[-1]['prepared_evidence']['metadata_sha256']==digest(metadata),'Actual positive Adam/transaction journal binds certified consumed metadata')
native_snapshot=copy.deepcopy(part['native'])
route_map=native_snapshot['joint_work_unit_allocation_routes']
require(set(route_map)=={str(i)for i in range(len(metadata['rows']))},'Source-declared full integer-index route map serialized by JSON')
native_snapshot['joint_work_unit_allocation_routes']={int(k):v for k,v in route_map.items()}
require(events[-1]['native']['full_native_receipt_sha256']==digest(native_snapshot),'Commit-time full native digest after restoring ONLY original integer route-key types')
require(all(events[-1]['native'][k]==part['native'][k] for k in ('update','phase','source_ids','languages','mean_objective','global_grad_norm')),'All compact committed native fields equal archived receipt')
persistence=read_json(CK/'persistence_verified.json');require(persistence['named_tensors_verified']==854 and persistence['Adam_states_verified']==215,'Actual854/215 persistence')
who=progress['producer'];require(abs(psutil.Process(who['pid']).create_time()-who['create_time'])<.001,'Actual same live native')
CPU=read_json(I/'CPU/actual_CPU_ready.json');cpu=CPU['producer_identity'];require(abs(psutil.Process(cpu['pid']).create_time()-cpu['create_time'])<.001,'Same actual resident CPU')
observed=N/'actual_verified47800_positive_progress.json';require(not observed.exists(),'Unique observation');observed.write_bytes(progress_bytes)
report={'kind':'ASR18_actual_full_original_clip_corrected_and_saved47800_continuation_v1','passed':True,'owned_session':62574,'native':who,'CPU':cpu,'driver_source':progress['source'],'configuration':read_json(I/'boot.json')['configuration'],'ROOT_execution_review':record(N.parent/'ROOT_third_original_full_clip_saved47600_execution_review_v1.json'),'actual_CPU_review':record(N.parent/'ROOT_third_original_full_clip_geometry_overlay_actual_CPU_review_v1.json'),'actual_startup_restore':record(I/'actual_parent_restore.json'),'actual_journal_branch':record(I/'actual_journal_branch.json'),'actual_corrected_update':47731,'consumed_row_index':2,'consumed_row':entry,'actual_consumed_geometry_proof':proof,'actual_full_clean_PCM_sha256':ref['expected_audio']['waveform_sha256'],'actual_native_receipt_archived_JSON_sha256':digest(part['native']),'actual_native_receipt_commit_type_sha256':digest(native_snapshot),'JSON_integer_route_keys_restored_for_commit_digest':True,'actual_native_receipt_summary':{k:part['native'][k] for k in ('update','phase','source_ids','languages','mean_objective','global_grad_norm')},'journal_event47731':events[-1],'checkpoint_pointer':record(CK/'pointer.json'),'checkpoint_persistence':record(CK/'persistence_verified.json'),'consumed_metadata_archive':archive,'current_progress':record(observed),'actual_optimizer_updates':progress['actual_optimizer_updates'],'native_updates_this_process':progress['native_updates_this_process'],'last_verified_save':47800,'old_whole_journal_and_tail_preserved':branch['preserved_original'],'old_unsaved_tail_stays_historical':branch['historical_unsaved_tail_updates'],'all_original_physical_samples_used_once':True,'original_annotation_indexes_labels_sampler_order_and_fullpass_denominator_unchanged':True,'full_run_target':401072,'administrative_update_stop':None,'training_continues':True,'no_new_speed_study':True}
p=N/'actual_corrected_original_clip_saved47800_status.json';require(not p.exists(),'One compact receipt');write_json(p,report);print(json.dumps({'receipt':record(p),'actual_updates':progress['actual_optimizer_updates'],'checkpoint':report['checkpoint_pointer'],'native':who}),flush=True)
