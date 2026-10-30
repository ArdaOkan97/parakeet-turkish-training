"""One clean-phase native hook trial from coherent20; zero production passes."""
from copy import deepcopy
from dataclasses import asdict
import json,os,time
from pathlib import Path
from atomic_io_retry_v1 import write_json
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import CarriedReplay,digest
from bpe_broad_transaction_v1 import DrawTransaction
from bpe_replay_bucket_no_runtime_v1 import BucketCycler
from bpe_broad_production_v3 import prohibit_model_transforms,assert_model
from bpe_streaming_training_state_v1 import checkpoint as save_state,restore as restore_state
from asr18_cleaner_fixture_cursor_v1 import DiagnosticCleanerCursor
from asr18_precomputed_native_v2 import native_update
from asr18_feature_window_native_probe_v1 import NativePeak
from asr18_cleaner_native_frontier_evidence_v1 import publish as publish_frontier
from asr18_calibrated_geometry_forecast_v2 import CalibratedGeometryForecast
from asr18_metadata_checkpoint_binding_v1 import MetadataCheckpointBinding
from fastmss_clean_feature_window_v1 import FeatureWindowReader,read_bound
from fastmss_ordered_draw_binding_v1 import AugmentationDrawContext
from resource_watchdog_v2 import sample_memory
S=Path(__file__).resolve().parent

def prepare_probe(model,processor,teacher,optimizer,provenance,replay_snapshot,folder,config,*,torch,stage,admit,stop_check):
    require(config['production_enabled'] is False and config['maximum_native_updates']==1,'Only one finite diagnostic clean hook')
    window_binding=config['feature_window_manifest'];owner_binding=config['feature_window_owner']
    window=read_bound(window_binding,stop_check);owner=read_bound(owner_binding,stop_check);plan=read_bound(window['plan'],stop_check)
    handoff=read_bound(config['actual_CPU_handoff'],stop_check)
    require(handoff['kind']=='FastMSS_actual_original_clean_CPU_feature_handoff_v1' and handoff['passed'] is True
        and handoff['owner']==owner_binding and handoff['manifest']==window_binding and handoff['update']==window['updates'][0]['update']
        and handoff['actual_owned_CPU_exit']['actual_exit_code']==0 and handoff['actual_owned_CPU_exit']['owned_session']==13220
        and handoff['actual_owned_CPU_exit']['actual_tool_chunk']=='0a580c' and handoff['actual_original_frontend_exit']['actual_exit_code']==0
        and handoff['actual_original_frontend_exit']['native_pid']==39324 and handoff['native_cleaner_consumer_validated'] is False,
        'Actual owned closed original clean CPU/frontend proof required, without prior native approval')
    for entry in handoff['source_bindings']:require(record(entry['path'])==entry,'Pinned clean CPU producer/lifecycle source changed')
    for section in ('actual_owned_CPU_exit','actual_original_frontend_exit'):
        proof=read_bound(handoff[section]['guard'],stop_check)
        require(proof['actual_exit_code']==0 and proof['monitor_clean'] and proof['monitor_joined']
            and proof['observed']['samples']>=3,'Actual clean joined CPU guard closure required')
    require(window['owner']==owner_binding and len(window['updates'])==len(plan['updates'])==1 and plan['source_parent_checkpoint']==config['parent_checkpoint']
        and plan['initial_source_optimizer_updates']==20 and plan['prior_full_broad_corpus_passes']==0 and plan['production_phase_transition_claimed'] is False,
        'Actual closed single clean fixture; no completed production broad phase')
    require(plan['full_cleaner_mean_U']==11972886/352768,'Exact full cleaner target denominator')
    binding=MetadataCheckpointBinding(owner['augmentation_identity'],guard=stop_check)
    prohibit_model_transforms(model,torch);model._broad_processor=processor
    admit('restore_actual20_before_cleaner_hook',350_000_000)
    restored,journal,parent_branch=restore_state(plan['source_parent_checkpoint'],torch=torch,model=model,optimizer=optimizer,
        identity=plan['source_parent_identity'],augmentation=binding,new_journal_path=folder/'from_actual20_authoritative_prefix.jsonl',diagnostic_only=True,guard=stop_check)
    initial=plan['initial_diagnostic_sampler'];require(optimizer.updates==20 and digest(restored)==digest({'broad_cursor':initial['continued_broad_diagnostic_cursor'],
        'carried_replay':initial['carried_replay']}),'Actual all207/Adam/RNG checkpoint20 source state differs')
    identity_cursor=initial['diagnostic_cleaner_cursor'];cursor=DiagnosticCleanerCursor(ordinals=identity_cursor['ordinals'],
        identity=identity_cursor['inner']['identity'],seed=identity_cursor['inner']['seed']);cursor.load_state_dict(identity_cursor)
    pool=plan['original_replay_pool'];require(record(pool['path'])==pool,'Original replay rows bound')
    replay=CarriedReplay(json.loads(Path(pool['path']).read_text())['replay'],replay_snapshot,BucketCycler);replay.load_state_dict(restored['carried_replay'])
    def state_now():return {'continued_broad_diagnostic_cursor':restored['broad_cursor'],'diagnostic_cleaner_cursor':cursor.state_dict(),'carried_replay':replay.state_dict()}
    planned=plan['updates'][0];by_ordinal={r['cleaner_original_ordinal']:r for r in planned['TR_rows']}
    def resolve(ordinal):return {'ordinal':ordinal,'source_id':by_ordinal[ordinal]['source_id'],'language':'tr'}
    tx=DrawTransaction(cursor,replay,optimizer_updates=20,resolve_TR=resolve);before=state_now()
    require(digest(before)==planned['expected_before_sampler_sha256'] and tx.identity_digest==planned['expected_native_transaction_sha256']
        and tx.TR_draws==planned['TR_draws'] and list(tx.replay_rows)==planned['replay_rows'],'Actual post20 source/RNG/cleaner diagnostic draw differs')
    rows=[deepcopy(by_ordinal[d['ordinal']]) for d in tx.TR_draws]+[{**r,'augmentation_role':'replay'} for r in tx.replay_rows]
    draws=planned['draws'];contexts=tuple(AugmentationDrawContext(**v) for v in planned['contexts']);part=window['updates'][0]
    prepared=read_bound(part['update'],stop_check)
    require(prepared['native_transaction_sha256']==tx.identity_digest and prepared['before_sampler_sha256']==digest(before)
        and prepared['after_sampler_sha256']==planned['expected_after_sampler_sha256']
        and prepared['prepared_payload']['protection_kind']=='FastMSS_explicit_clean_phase_protection_v1'
        and all(r['augmentation']['applied'] is False for r in prepared['prepared_payload']['rows']), 'Exact clean-only closed inputs and source frontier')
    reader=FeatureWindowReader(owner_binding,window_binding,guard=stop_check)
    forecast=CalibratedGeometryForecast(**config['forecast_calibration'])
    current=sample_memory(os.getpid(),'clean_hook_prospective_budget')['worker_private_memory'];require(current['available'],'Fresh current native private telemetry')
    budget=forecast.budget(rows=prepared['rows'],current_private_bytes=current['private_bytes'],update_in_process=1)
    write_json(folder/'native_forecast_before_actual21.json',budget);admit('prospective_clean_native_update21',budget['incremental_bytes'])
    declaration={'kind':'ASR18_actual_cleaner_hook_after20_diagnostic_v1','diagnostic_only':True,'production_enabled':False,
        'parent_checkpoint':plan['source_parent_checkpoint'],'initial_clocks':[64270,2020,62270],'full_broad_corpus_passes':0,
        'production_cleaner_stage_transition_claimed':False,'TR_ordinals':identity_cursor['ordinals'],'phase':'cleaner_data_fine_tuning',
        'full_cleaner_mean_U':11972886/352768,'feature_window':window_binding,'native_consumer':record(S/'asr18_precomputed_native_v2.py'),
        'parent_frontend_augmenter_transport_invoked':False,'forecast':record(folder/'native_forecast_before_actual21.json')}
    write_json(folder/'declaration.json',declaration);began=time.monotonic();peak=NativePeak();peak.start()
    try:
        native=native_update(model,teacher,optimizer,rows,draws,
            feature_for_row=lambda i:reader.take_row(0,i,context=contexts[i],native_transaction_sha256=tx.identity_digest,before_sampler_sha256=digest(before)),
            update_binding=part['update'],torch=torch,population_mean_U=11972886/352768,update=21,stage=stage,
            stop_check=stop_check,phase='cleaner_data_fine_tuning',feature_update_index=0)
        publish_frontier(folder,optimizer,native=native,committed=False)
    except BaseException as exc:
        publish_frontier(folder,optimizer,error=repr(exc))
        raise
    finally:
        memory=peak.finish();memory['wall_seconds']=time.monotonic()-began;write_json(folder/'native_phase_21.json',memory)
    publish_frontier(folder,optimizer,native=native,committed=False)
    forecast.observe(memory);covered=memory['private_peak_bytes']<=budget['current_private_bytes']+budget['incremental_bytes']
    write_json(folder/'actual_forecast_coverage_21.json',{'passed':covered,'before_private_bytes':budget['current_private_bytes'],'forecast':budget,
        'actual_native_peak_bytes':memory['private_peak_bytes'],'remaining_margin_bytes':budget['current_private_bytes']+budget['incremental_bytes']-memory['private_peak_bytes'],
        'production_forecast_approved':False})
    if not covered:optimizer.poisoned=True;raise ValueError('Prospective native private forecast missed; positive21 is uncheckpointed, restore coherent20')
    require(reader.prepared_payload(0)==native['augmentation'] and optimizer.updates==21,'Actual consumed clean payload and positive Adam proof')
    try:
        cursor.commit(tx.TR_draws);replay.load_state_dict(tx.after_replay);state=state_now()
        require(digest(state)==planned['expected_after_sampler_sha256'],'Exact diagnostic cleaner cursor/carried replay after actual commit')
        journal.append({'kind':'ASR18_committed_cleaner_hook_diagnostic_update_v1','optimizer_updates':21,'phase':'cleaner_data_fine_tuning',
            'sampler_state_sha256':digest(state),'transaction_sha256':tx.identity_digest,'prepared_update':part['update'],'native':native})
        tx.closed=True
    except BaseException:optimizer.poisoned=True;tx.poisoned=True;raise
    publish_frontier(folder,optimizer,native=native,committed=True)
    own_identity={'kind':'ASR18_cleaner_hook_checkpoint_identity_v1','config':record(config['_path']),'probe':record(__file__),
        'native_consumer':record(S/'asr18_precomputed_native_v2.py'),'declaration':declaration,'parent_checkpoint':plan['source_parent_checkpoint']}
    admit('bounded_slice_cleaner_checkpoint21',64_000_000)
    checkpoint_before=sample_memory(os.getpid(),'cleaner_checkpoint21_before')['worker_private_memory'];require(checkpoint_before['available'],'Available checkpoint private baseline')
    checkpoint_peak=NativePeak();checkpoint_peak.start()
    try:
        saved=save_state(folder/'after_update21.state.json',torch=torch,optimizer=optimizer,state=state,provenance=provenance,identity=own_identity,
            augmentation=binding,journal=journal,diagnostic_only=True,guard=stop_check)
    finally:
        checkpoint_memory=checkpoint_peak.finish()
    checkpoint_covered=checkpoint_memory['private_peak_bytes']<=checkpoint_before['private_bytes']+64_000_000
    write_json(folder/'checkpoint21_write_memory.json',{'before':checkpoint_before['private_bytes'],'phase':checkpoint_memory,'incremental_budget_bytes':64_000_000,'passed':checkpoint_covered})
    require(checkpoint_covered,'Checkpoint21 private write forecast missed; preserve returned checkpoint as review-required')
    require(record(saved['path'])==saved,'Writer returned completed metadata');metadata=json.loads(Path(saved['path']).read_text());require(record(metadata['payload']['path'])==metadata['payload'],'Completed checkpoint payload hash')
    write_json(folder/'latest_coherent_checkpoint.json',{'checkpoint':saved,'payload':metadata['payload'],'sampler':state,
        'committed_updates':21,'clocks':[64271,2021,62271],'writer_returned_after_fsync':True,'diagnostic_only':True})
    admit('bounded_slice_cleaner_checkpoint21_exact_restore',350_000_000)
    restore_before=sample_memory(os.getpid(),'cleaner_checkpoint21_restore_before')['worker_private_memory'];require(restore_before['available'],'Available restore private baseline')
    restore_peak=NativePeak();restore_peak.start()
    try:
        after,restored_journal,own_branch=restore_state(saved,torch=torch,model=model,optimizer=optimizer,identity=own_identity,
            augmentation=binding,new_journal_path=folder/'from_checkpoint21_authoritative_prefix.jsonl',diagnostic_only=True,guard=stop_check)
    finally:
        restore_memory=restore_peak.finish()
    restore_covered=restore_memory['private_peak_bytes']<=restore_before['private_bytes']+350_000_000
    write_json(folder/'checkpoint21_restore_memory.json',{'before':restore_before['private_bytes'],'phase':restore_memory,'incremental_budget_bytes':350_000_000,'passed':restore_covered})
    require(restore_covered,'Checkpoint21 restore private forecast missed; no prospective coverage approval')
    require(after==state and optimizer.updates==21 and digest(after)==planned['expected_after_sampler_sha256'],'All207/threeAdam/allRNG and both diagnostic/replay frontiers exact restored')
    cursor.load_state_dict(after['diagnostic_cleaner_cursor']);replay.load_state_dict(after['carried_replay'])
    def resume_verified(value):
        require(value['checkpoint']==saved and value['committed_window_updates']==1 and digest(state_now())==planned['expected_after_sampler_sha256'],'Actual native/source frontiers verified before reader resume')
        return {'passed':True,'phase':'cleaner_data_fine_tuning','checkpoint':saved,'feature_window_manifest':window_binding,
            'committed_window_updates':1,'committed_transaction_sha256':tx.identity_digest,'committed_after_sampler_sha256':digest(after),
            'next_before_sampler_sha256':digest(after),'augmentation_identity':owner['augmentation_identity'],'native_checkpoint_state_verified':True,
            'actual_restored_sampler_verified':True,'committed_journal_prefix_verified':True}
    resumed=FeatureWindowReader(owner_binding,window_binding,guard=stop_check,resume={'checkpoint':saved,'committed_window_updates':1},verify_resume=resume_verified)
    require(resumed.prepared_payload(0)==native['augmentation'] and cursor.remaining==0 and replay.updates==21,'Restored exact clean payload/sampler; no consumed feature reuse')
    optimizer.validate(full=True);assert_model(model,optimizer,torch);torch.cuda.synchronize();torch.cuda.empty_cache()
    result={'passed':True,'native_updates_this_child':1,'coherent_chain_updates':21,'checkpoint':saved,'final_clocks':[64271,2021,62271],
        'native_memory':memory,'checkpoint_memory':checkpoint_memory,'restore_memory':restore_memory,'all_native_write_restore_private_forecasts_covered':True,'forecast_covered_actual_clean21':True,'forecast_coverage':record(folder/'actual_forecast_coverage_21.json'),
        'all207_weights_three_Adam_Python_NumPy_Torch_CUDA_RNG_exact':True,'actual_cleaner_cursor_carried_replay_and_journal_prefix_restored':True,
        'original20_source_and_sampler_restored_before_next_exactdraw':True,'clean_feature_payload_exact_before_after_restore':True,
        'parent_checkpoint20_branch':parent_branch,'checkpoint21_branch':own_branch,'prepared_clean_payload':native['augmentation'],
        'actual_languages':[r['language'] for r in rows],'actual_applied_augmentation_count':0,'parent_frontend_augmenter_transport_invoked':False,
        'full_cleaner_mean_U':11972886/352768,'full_broad_corpus_passes':0,'production_cleaner_phase_transition_claimed':False,
        'feature_window':window_binding,'feature_window_retained':True,'diagnostic_only':True,'production_ready':False}
    write_json(folder/'production_caller_probe.json',result);return result
