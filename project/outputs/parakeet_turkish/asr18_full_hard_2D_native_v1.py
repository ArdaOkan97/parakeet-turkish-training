"""Two full retained long-row native updates; diagnostic only, no corpus draw."""
import gc,json,os,time
from pathlib import Path
from functools import partial
import torch
from atomic_io_retry_v1 import write_json
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_v3 import assert_model,prohibit_model_transforms,augmentation_binding
from bpe_broad_production_state_v1 import CommitJournal
from bpe_streaming_training_state_v1 import checkpoint,restore
from asr18_real_target_TDT_validation_v9 import objective
from asr18_broad_stream_checkpoint20_probe_v1 import NativePeak
from tdt_tiled_full_native_v6 import bounded_tdt_loss
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')
S=Path(__file__).resolve().parent

def validate(model,processor,optimizer,provenance,folder,*,stage,admit,stop_check):
    from resource_watchdog_v2 import sample_memory
    oracle=A/'real_target_2D_TDT_gradients_01/real_model_equivalence.json'
    joint=A/'instance_denominator_joint_01/actual_joint_precision.json'
    require(json.loads(oracle.read_text())['passed'] and json.loads(joint.read_text())['passed'],'Unchanged all207 tolerance gates and byte-exact joint lifetime control required')
    source=A/'hard_U_rows_decode_01/decoded/report.json';decoded=json.loads(source.read_text())
    require(decoded['passed'] and decoded['heldout_PCM_check_performed'],'Complete actual decoded/heldout source proof')
    rows=[item['row'] for item in decoded['rows']]
    require([(r['broad_original']['ordinal'],r['target_tokens'],r['encoder_frames']) for r in rows]==[(1051133,574,1991),(1051316,484,1505)],'Exact two retained full hard shapes')
    for item in decoded['rows']:require(record(item['PCM_cache']['path'])==item['PCM_cache'],'Full original PCM changed')
    declaration={'kind':'ASR18_two_full_retained_hard_rows_2D_native_v1','diagnostic_only':True,'native_updates_bound':2,
        'full_source_audio_and_all_raw_targets':True,'no_quality_or_language_filters':True,'augmentation':False,'replay_clips':0,
        'physical_batch':1,'effective_batch':1,'corpus_passes':0,'consumed_production_draws':0,'production_mean_U':None,
        'new_policy':model.joint._ASR18_fp32_joint_policy,'loss':record(S/'tdt_tiled_full_native_v6.py'),
        'full_vocabulary_normalization':8711,'full_native_DP_operands_bound_bytes':1_700_000_000,'tile_bound_bytes':16_000_000,
        'source_PCM_proof':record(source),'all207_oracle':record(oracle),'joint_lifetime_oracle':record(joint),
        'rows':[{'source_id':r['source_id'],'samples':round(r['duration']*16000),'T':r['encoder_frames'],'U':r['target_tokens'],
            'dense_joint_bytes_NOT_materialized':r['encoder_frames']*(r['target_tokens']+1)*8716*4,
            'full_native_DP_workspace_bound_bytes':r['encoder_frames']*(r['target_tokens']+1)*48} for r in rows],
        'checkpoint_mode':'streamed_4MiB_CPU_slices_then_validated_binding_then_latest_pointer',
        'original_source_clocks':[64250,2000,62250],'checkpoint_qualification':'unqualified_diagnostic'}
    write_json(folder/'declaration.json',declaration)
    identity={'caller':record(__file__),'declaration':declaration,'source_records':provenance['source_records']}
    disabled=S/'fastmss_train_augmentation_v1.pilot.disabled.json'
    augmentation=augmentation_binding({'augmentation_policy':str(disabled)},stop_check)
    journal=CommitJournal(folder/'committed_rows_original.jsonl');history=[];previous_peak=None;saved=None
    prohibit_model_transforms(model,torch);model.train();model.encoder.eval();assert_model(model,optimizer,torch)
    native=partial(bounded_tdt_loss,guard=stop_check)
    for index,row in enumerate(rows,1):
        stop_check();current=sample_memory(os.getpid(),'full_hard_geometry_admission')['worker_private_memory']
        require(current['available'],'Fresh current native private memory required')
        # Row2 is strictly smaller than the measured row1 T/U/dense work.
        budget=1_800_000_000 if previous_peak is None else max(250_000_000,previous_peak['private_peak_bytes']-current['private_bytes']+250_000_000)
        write_json(folder/f'update_{index}_budget.json',{'incremental_bytes':budget,'current_private_bytes':current['private_bytes'],
            'prior_phase':previous_peak,'scope':'Exactly these two pinned full rows, first T/U dominates second; not production forecast'})
        admit(f'native_update_full_hard_row_{index}',budget)
        optimizer.zero_grad();began=time.monotonic();monitor=NativePeak();monitor.start()
        try:
            result,batch=objective(model,processor,row,native)
            require(bool(torch.isfinite(result.total)),'Finite full-T/full-U objective')
            loss=float(result.total.detach());result.total.backward()
            require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in optimizer.parameters.values()),'All207 finite full native gradients')
            norm=float(optimizer.step());optimizer.zero_grad();del result,batch
            torch.cuda.synchronize();before_gc={'allocated_bytes':torch.cuda.memory_allocated(),'private_bytes':sample_memory(os.getpid(),'full_hard_before_GC')['worker_private_memory']['private_bytes']}
            objects=gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache()
            after_gc={'allocated_bytes':torch.cuda.memory_allocated(),'private_bytes':sample_memory(os.getpid(),'full_hard_after_GC')['worker_private_memory']['private_bytes']}
            require(torch.cuda.memory_reserved()<=3_000_000_000,'Unchanged native end-cache bound');assert_model(model,optimizer,torch)
        finally:previous_peak=monitor.finish()
        item={'update':index,'source_id':row['source_id'],'full_T':row['encoder_frames'],'full_U':row['target_tokens'],'objective':loss,
            'global_grad_norm':norm,'native_seconds':time.monotonic()-began,'native_memory':previous_peak,
            'post_graph_GC_observation':{'before':before_gc,'after':after_gc,'collected_objects':objects},'diagnostic_only':True}
        state={'diagnostic_committed_rows':index,'source_ids':[r['source_id'] for r in rows[:index]],'corpus_passes':0}
        try:journal.append({'kind':'diagnostic_actual_Adam_commit','state':state,'receipt':item})
        except BaseException:
            optimizer.poisoned=True;raise
        history.append(item);write_json(folder/'native_progress.json',{'completed_updates':index,'clocks':[c+index for c in (64250,2000,62250)],'history':history,'diagnostic_only':True})
        stage('full_hard_native_row_committed',**item)
        admit(f'full_hard_streaming_checkpoint_{index}',64_000_000)
        saved=checkpoint(folder/f'after_full_hard_update{index}.state.json',torch=torch,optimizer=optimizer,state=state,provenance=provenance,
            identity=identity,augmentation=augmentation,journal=journal,diagnostic_only=True,guard=stop_check)
        # Writer returned only after metadata and payload fsync. Bind both before
        # atomically publishing a latest pointer; existence alone never commits.
        require(record(saved['path'])==saved,'Completed metadata binding changed')
        document=json.loads(Path(saved['path']).read_text());require(record(document['payload']['path'])==document['payload'],'Completed payload binding changed')
        write_json(folder/'latest_coherent_checkpoint.json',{'checkpoint':saved,'payload':document['payload'],'committed_state':state,'clocks':[c+index for c in (64250,2000,62250)],'diagnostic_only':True})
        admit(f'full_hard_streaming_restore_{index}',350_000_000)
        restored,journal,branch=restore(saved,torch=torch,model=model,optimizer=optimizer,identity=identity,augmentation=augmentation,
            new_journal_path=folder/f'committed_rows_restored_{index}.jsonl',diagnostic_only=True,guard=stop_check)
        require(restored==state and optimizer.updates==index,'Exact all207/threeAdam/RNG/full source frontier restoration')
        write_json(folder/f'restore_{index}.json',{'passed':True,'checkpoint':saved,'all207_weights_threeAdam_and_RNG_byte_exact':True,
            'state':state,'branch':branch,'owned_mutable_CPU_steps_and_closed_ACCESS_READ_mapping':True})
    report={'passed':True,'native_updates':2,'history':history,'checkpoint':saved,'declaration':declaration,'final_clocks':[64252,2002,62252],
        'full159s_U574_and120s_U484_supported':True,'both_full_audio_and_targets_retained':True,'both_checkpoint_restores_byte_exact':True,
        'production_ready':False,'diagnostic_only':True,'same_process_two_updates_only':True,'sustained20_update_stability_claimed':False,
        'native_total_seconds':sum(item['native_seconds'] for item in history),'native_audio_seconds':sum(r['duration'] for r in rows)}
    write_json(folder/'full_hard_native_validation.json',report);return report
