"""Finite original-state native control/window; provisional inputs never commit draws."""
from dataclasses import asdict
import json,os,threading,time
from pathlib import Path
from bpe_broad_audio_v1 import record,require
from atomic_io_retry_v1 import write_json
from bpe_broad_production_state_v1 import CarriedReplay,CommitJournal,digest
from bpe_broad_transaction_v1 import DrawTransaction
from bpe_broad_production_v3 import prohibit_model_transforms,augmentation_binding
from asr18_narrow_training_primitives_v1 import BucketCycler
from asr18_repeat_fixture_cursor_v1 import RepeatedFixtureCursor
from asr18_precomputed_native_v1 import native_update
from fastmss_feature_window_v1 import FeatureWindowReader,read_bound
from fastmss_ordered_draw_binding_v1 import AugmentationDrawContext
from bpe_streaming_training_state_v1 import checkpoint as save_state,restore as restore_state

PROJECT=Path(__file__).resolve().parents[2]
SOURCE=Path(__file__).resolve().parent
AREA=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')

class NativePeak:
    """Fresh available native samples only; errors prohibit another admission."""
    def __init__(self):
        self.done=threading.Event();self.count=0;self.peak=0;self.last=None;self.error=None
        self.thread=threading.Thread(target=self._run,daemon=True)
    def _run(self):
        from resource_watchdog_v2 import sample_memory
        try:
            while not self.done.is_set():
                item=sample_memory(os.getpid(),'precomputed_native_phase')['worker_private_memory']
                require(item['available'],'Unavailable native private sample')
                self.count+=1;self.peak=max(self.peak,item['private_bytes']);self.last=time.monotonic();self.done.wait(.05)
        except BaseException as exc:self.error=repr(exc)
    def start(self):self.thread.start()
    def finish(self):
        self.done.set();self.thread.join(timeout=2)
        require(not self.thread.is_alive() and self.error is None and self.count>=3 and self.last is not None and time.monotonic()-self.last<2,
                'Joined clean >=3 fresh native-phase samples required; no guessed fallback')
        return {'private_peak_bytes':self.peak,'available_sample_count':self.count,'last_sample_monotonic':self.last,'error':self.error,'joined':True,'interval_seconds':.05}

def prepare_probe(model,processor,teacher,optimizer,provenance,replay_snapshot,folder,config,*,torch,stage,admit,stop_check):
    from resource_watchdog_v2 import sample_memory
    require(config['production_enabled'] is False and config['maximum_native_updates'] in (1,20),'Only finite native control or20 window, no production')
    maximum=config['maximum_native_updates'];window_binding=config['feature_window_manifest'];owner_binding=config['feature_window_owner']
    window=read_bound(window_binding,stop_check);plan=read_bound(window['plan'],stop_check)
    require(len(plan['updates'])==20 and window['owner']==owner_binding and plan['source_records']['bpe']==provenance['source_records']['bpe'],
            'Exact20 closed provisional inputs from original BPE source required')
    require(plan['original_clocks']==[64250,2000,62250] and optimizer.updates==0,'Never seed from a diagnostic update')
    by_ordinal={v['ordinal']:v for v in plan['TR_rows']}
    cursor=RepeatedFixtureCursor(ordinals=sorted(by_ordinal),identity={'fixture_manifest':plan['TR_fixture_manifest'],'is_final_corpus':False},seed=digest(replay_snapshot['turkish_rng']),repeats=20)
    pool=plan['source_records']['replay_pool'];require(record(pool['path'])==pool,'Original replay pool changed')
    carried=CarriedReplay(json.loads(Path(pool['path']).read_text(encoding='utf-8'))['replay'],replay_snapshot,BucketCycler)
    def state_now():return {'broad_cursor':cursor.state_dict(),'carried_replay':carried.state_dict()}
    require(digest(state_now())==digest(plan['initial_private_planning_sampler']),'Original actual source sampler differs from provisional plan')
    def resolve(ordinal):return {'ordinal':ordinal,'source_id':by_ordinal[ordinal]['unique_id'],'language':'tr'}
    def proposal():return DrawTransaction(cursor,carried,optimizer_updates=optimizer.updates,resolve_TR=resolve)
    declaration={'kind':'ASR18_precomputed_sameprocess_native_window_v1','diagnostic_only':True,'production_enabled':False,
        'maximum_native_updates':maximum,'original_clocks':[64250,2000,62250],'feature_window':window_binding,'plan':window['plan'],
        'full_corpus_complete_passes':0,'fixture_mean_U':plan['fixture_mean_U'],'physical_batch':1,'TR_per_update':14,'replay_per_update':2,
        'parent_frontend_decoder_or_augmentation_invoked':False,'native_numeric_policy':model.joint._ASR18_fp32_joint_policy,
        'raw_labels_and_quality_flags_preserved':True,'all207_FP32_Adam_preserved':True,'guards':config['resource_guards']}
    write_json(folder/'declaration.json',declaration)
    identity={'config':record(config['_path']),'probe':record(__file__),'native_consumer':record(SOURCE/'asr18_precomputed_native_v1.py'),
        'bootstrap':record(SOURCE/'bpe_broad_streamed_bootstrap_v2.py'),'feature_window':window_binding,'declaration':declaration}
    journal=CommitJournal(folder/'committed_original.jsonl');binding=augmentation_binding(config,stop_check)
    require(binding.identity==window['augmentation_identity'],'Frozen original producer policy/library identity changed')
    reader=FeatureWindowReader(owner_binding,window_binding,guard=stop_check)
    prohibit_model_transforms(model,torch);model._broad_processor=processor
    history=[];metrics=[];previous_peak=None;checkpoint_records=[];restores=[];started=time.monotonic();state=None
    def publish(saved):
        require(record(saved['path'])==saved,'Writer returned metadata binding required')
        value=json.loads(Path(saved['path']).read_text(encoding='utf-8'))
        require(record(value['payload']['path'])==value['payload'],'Completed fsynced payload binding differs')
        latest={'checkpoint':saved,'payload':value['payload'],'sampler':state,'committed_updates':optimizer.updates,
            'clocks':[c+optimizer.updates for c in (64250,2000,62250)],'writer_returned_after_fsync':True,'diagnostic_only':True}
        write_json(folder/'latest_coherent_checkpoint.json',latest);return latest
    for update in range(1,maximum+1):
        planned=plan['updates'][update-1];before=digest(state_now());transaction=proposal()
        require(transaction.identity_digest==planned['expected_native_transaction_sha256'] and before==planned['expected_before_sampler_sha256'] and
            transaction.TR_draws==planned['TR_draws'] and transaction.TR_identity==planned['TR_identity'] and list(transaction.replay_rows)==[{k:v for k,v in r.items() if k!='augmentation_role'} for r in planned['replay_rows']],
            'Actual original cursor/replay selection differs from provisional transaction')
        part=window['updates'][update-1];prepared=read_bound(part['update'],stop_check)
        require(prepared['native_transaction_sha256']==transaction.identity_digest and prepared['before_sampler_sha256']==before and prepared['after_sampler_sha256']==planned['expected_after_sampler_sha256'],
                'Feature producer does not bind exact native transaction/frontiers')
        decoded=prepared['input_evidence']['decoder_report_contents'];rows=[v['row'] for v in decoded['rows']]
        for row,draw in zip(rows,transaction.TR_draws):
            raw=by_ordinal[draw['ordinal']]
            require(row['broad_original']['ordinal']==draw['ordinal'] and row['durable_draw']==draw and row['tr_metadata_labels']==raw['BPE']['target_ids'] and
                all(raw[k]==v for k,v in row['broad_original'].items()),'Full raw row/targets differ from independently pinned catalog fixture')
        rows.extend({**r,'augmentation_role':'replay'} for r in transaction.replay_rows)
        contexts=tuple(AugmentationDrawContext(**v) for v in planned['contexts'])
        require([asdict(v) for v in contexts]==[v['context'] for v in prepared['rows']],'Exact ordered source/draw contexts differ')
        def take(i):return reader.take_row(update-1,i,context=contexts[i],native_transaction_sha256=transaction.identity_digest,before_sampler_sha256=before)
        current=sample_memory(os.getpid(),'precomputed_native_admission')['worker_private_memory'];require(current['available'],'Fresh admission private sample')
        budget=2_000_000_000 if update==1 else max(250_000_000,previous_peak['private_peak_bytes']-current['private_bytes']+250_000_000)
        write_json(folder/f'update_{update}_budget.json',{'incremental_bytes':budget,'current_private_bytes':current['private_bytes'],'prior_native_phase':previous_peak,
            'same_fixed14_geometry_scope_only':True,'fresh_replay_and_stochastic_augmentation_geometry_not_production_generalized':True,'no_CPU_child_between_updates':True})
        admit('native_integrated_update_'+str(update),budget);peak=NativePeak();peak.start();began=time.monotonic()
        try:
            item=native_update(model,teacher,optimizer,rows,planned['draws'],feature_for_row=take,update_binding=part['update'],torch=torch,
                population_mean_U=plan['fixture_mean_U'],update=update,stage=stage,stop_check=stop_check,phase=config['phase'])
            require(reader.prepared_payload(update-1)==item['augmentation'],'Actual consumed-once feature payload differs before sampler commit')
            state=transaction.commit(optimizer=optimizer,journal=journal,native_receipt=item)
        except BaseException:
            if optimizer.updates!=update-1:optimizer.poisoned=True
            elif not transaction.closed:transaction.abandon()
            raise
        finally:previous_peak=peak.finish()
        previous_peak.update(wall_seconds=time.monotonic()-began,training_audio_seconds=sum(float(r['duration']) for r in rows))
        require(digest(state)==planned['expected_after_sampler_sha256'],'Actual positive-Adam committed frontier differs from planned after-state')
        metrics.append(previous_peak);history.append(item)
        write_json(folder/f'native_phase_{update}.json',previous_peak)
        write_json(folder/'native_progress.json',{'completed_updates':update,'history':history,'clocks':[c+optimizer.updates for c in (64250,2000,62250)],'diagnostic_only':True})
        stage('precomputed_native_committed',update=update,native_memory=previous_peak,frontier_sha256=digest(state))
        del rows,contexts,prepared,decoded
        if update in (1,10,maximum):
            expected=None
            if update<20:
                pending=proposal();expected={'transaction':pending.identity_digest,'before':digest(state_now()),'prepared_payload_sha256':read_bound(window['updates'][update]['update'],stop_check)['prepared_payload']['payload_sha256']}
                require(expected['transaction']==plan['updates'][update]['expected_native_transaction_sha256'],'Expected next original draw differs before save')
                pending.abandon()
            admit('precomputed_checkpoint_'+str(update),64_000_000)
            saved=save_state(folder/f'after_update{update}.state.json',torch=torch,optimizer=optimizer,state=state,provenance=provenance,
                identity=identity,augmentation=binding,journal=journal,diagnostic_only=True,guard=stop_check)
            publish(saved);checkpoint_records.append(saved)
            old_prefix=journal.binding();journal.append({'kind':'intentional_uncheckpointed_provisional_tail','actual_Adam_commit':False})
            preserved=journal.binding();admit('precomputed_restore_'+str(update),350_000_000)
            state,journal,branch=restore_state(saved,torch=torch,model=model,optimizer=optimizer,identity=identity,augmentation=binding,
                new_journal_path=folder/f'committed_resumed_{update}.jsonl',diagnostic_only=True,guard=stop_check)
            cursor.load_state_dict(state['broad_cursor']);carried.load_state_dict(state['carried_replay'])
            require(optimizer.updates==update and digest(state_now())==planned['expected_after_sampler_sha256'] and journal.binding()['sha256']==old_prefix['sha256'],
                'Restored native Adam, both samplers and journal prefix differ')
            if expected is not None:
                pending=proposal();require(pending.identity_digest==expected['transaction'] and digest(state_now())==expected['before'] and
                    read_bound(window['updates'][update]['update'],stop_check)['prepared_payload']['payload_sha256']==expected['prepared_payload_sha256'],
                    'Actual postrestore next transaction/prepared augmentation differs from pre-save expectation');pending.abandon()
            def verify_resume(resume):
                require(resume=={'checkpoint':saved,'committed_window_updates':update} and digest(state_now())==planned['expected_after_sampler_sha256'] and optimizer.updates==update and journal.binding()['sha256']==old_prefix['sha256'],
                        'Actual restored state and journal must be verified before local feature-reader resume')
                return {'passed':True,'checkpoint':saved,'feature_window_manifest':window_binding,'committed_window_updates':update,
                    'committed_transaction_sha256':planned['expected_native_transaction_sha256'],'committed_after_sampler_sha256':planned['expected_after_sampler_sha256'],
                    'next_before_sampler_sha256':plan['updates'][update]['expected_before_sampler_sha256'] if update<20 else planned['expected_after_sampler_sha256'],
                    'augmentation_identity':window['augmentation_identity'],'native_checkpoint_state_verified':True,'actual_restored_sampler_verified':True,'committed_journal_prefix_verified':True}
            reader=FeatureWindowReader(owner_binding,window_binding,guard=stop_check,resume={'checkpoint':saved,'committed_window_updates':update},verify_resume=verify_resume)
            proof={'passed':True,'checkpoint':saved,'updates':update,'all207_weights_all3_Adam_and_Python_NumPy_Torch_CUDA_RNG_exact':True,
                'actual_both_sampler_and_journal_prefix_verified':True,'preserved_old_journal':preserved,'branch':branch,'pre_save_next':expected,
                'all_future_feature_files_retained':True};restores.append(proof);write_json(folder/f'restore_proof_{update}.json',proof)
    result={'passed':True,'native_updates':maximum,'same_owned_GPU_process':True,'sustained20_proven':maximum==20,'checkpoint':checkpoint_records[-1],
        'checkpoint_records':checkpoint_records,'restore_proofs':restores,'final_clocks':[c+optimizer.updates for c in (64250,2000,62250)],
        'history':history,'native_phase_metrics':metrics,'total_native_wall_seconds':sum(v['wall_seconds'] for v in metrics),
        'prepare_probe_wall_seconds':time.monotonic()-started,'total_training_audio_seconds':sum(v['training_audio_seconds'] for v in metrics),
        'feature_window':window_binding,'feature_cache_retained_for_separate_control_or_final_coherent_cleanup':True,'parent_frontend_or_augmentation_invoked':False,
        'full_corpus_complete_passes':0,'production_ready':False,'diagnostic_only':True,'declaration':declaration}
    write_json(folder/'production_caller_probe.json',result);return result
