"""Changed SAME-process20-update fixed14-source resource diagnostic, not corpus coverage."""
from copy import deepcopy
from dataclasses import asdict
import json,os,threading,time
from pathlib import Path
from bpe_broad_production_v5 import *
from fastmss_ordered_draw_binding_v1 import AugmentationDrawContext
from asr18_repeat_fixture_cursor_v1 import RepeatedFixtureCursor
from bpe_streaming_training_state_v1 import checkpoint as streamed_checkpoint, restore as streamed_restore

AREA=ROOT/'analysis/ASR18_memory_diagnosis_v1'

class NativePeak:
    """Only the named native phase. Missing/stale/error samples prohibit reuse."""
    def __init__(self):
        self.done=threading.Event();self.count=0;self.peak=0;self.last=None;self.error=None
        self.thread=threading.Thread(target=self._run,daemon=True)
    def _run(self):
        from resource_watchdog_v2 import sample_memory
        try:
            while not self.done.is_set():
                item=sample_memory(os.getpid(),'canonical_broad_native_phase')['worker_private_memory']
                require(item['available'],'Unavailable native private sample')
                self.count+=1;self.peak=max(self.peak,item['private_bytes']);self.last=time.monotonic();self.done.wait(.05)
        except BaseException as exc:self.error=repr(exc)
    def start(self):self.thread.start()
    def finish(self):
        self.done.set();self.thread.join(timeout=2)
        require(not self.thread.is_alive() and self.error is None and self.count>=3 and self.last is not None and time.monotonic()-self.last<2,
                'Live clean >=3 native-phase samples required; no guessed fallback')
        return {'private_peak_bytes':self.peak,'available_sample_count':self.count,'last_sample_monotonic':self.last,'error':self.error,'joined':True,'interval_seconds':.05}

def prepare_probe(model,processor,teacher,optimizer,provenance,replay_snapshot,folder,config,*,torch,stage,admit,stop_check):
    from tsc_bpe_full_decoder_transition_v1 import preflight
    from train_lora_batched_tsc_bpe_full_decoder_v1 import load_catalogs
    from train_lora_batched_data_v11 import BucketCycler
    from types import SimpleNamespace
    from resource_watchdog_v2 import sample_memory
    require(config['production_enabled'] is False,'This fixture cannot arm production')
    require(json.loads((AREA/'real_target_TDT_gradients_06/real_model_equivalence.json').read_text())['passed'] and
        json.loads((AREA/'real_target_TDT_large_02/full_large_row_validation.json').read_text())['passed'],'Actual all207 and full-large-row proofs required')
    source_config=json.loads((ROOT/'runs/adaptive_20261015_TR_BPE512_full_decoder_02/run_config.json').read_text())
    context=preflight(SimpleNamespace(**source_config['arguments']));original,tr,replay_rows=load_catalogs(context)
    carried=CarriedReplay(replay_rows,replay_snapshot,BucketCycler)
    require(carried.rng.getstate()==replay_snapshot['replay_rng'] and carried.pt_rng.getstate()==replay_snapshot['Portuguese_rng'],
        'Actual source replay/selector RNG must be carried without reseeding')
    source=Path(config['probe_manifest']);manifest=[json.loads(v) for v in source.read_text().splitlines()]
    require(len(manifest)==14 and len({r['ordinal'] for r in manifest})==14,'Fixed exact14 distinct source fixture')
    by_ordinal={r['ordinal']:r for r in manifest};ordinals=sorted(by_ordinal)

    population={'fixture_manifest':record(source),'is_final_corpus':False}
    cursor=RepeatedFixtureCursor(ordinals=ordinals,identity=population,seed=digest(replay_snapshot['turkish_rng']),repeats=20)
    def resolve(ordinal):return {'ordinal':ordinal,'source_id':by_ordinal[ordinal]['unique_id'],'language':'tr'}
    declaration={'kind':'ASR18_sameprocess_14source20update_streaming_checkpoint_diagnostic_v1','production_training':False,'native_updates_bound':20,
        'fixture_TR_population':14,'fixture_repeated_cycles':20,'full_corpus_complete_passes':0,'production_mean_U':None,
        'fixture_mean_U':sum(r['BPE']['target_tokens'] for r in manifest)/14,'physical_batch':1,'TR_per_update':14,'carried_replay_per_update':2,
        'original_source_records':provenance['source_records'],'original_source_clocks':[64250,2000,62250],
        'new_joint_policy':model.joint._ASR18_fp32_joint_policy,'source_manifest':record(source),'guards':config['resource_guards'],
        'diagnostic_checkpoint_unqualified':True,'all_raw_quality_flags_ignored':True,'source_labels_rewritten':False}
    write_json(folder/'declaration.json',declaration)
    identity={'config':record(config['_path']),'caller':record(SOURCE/'bpe_broad_production_v5.py'),'streamed_checkpoint':record(SOURCE/'bpe_streaming_training_state_v1.py'),'probe':record(__file__),'declaration':declaration}
    journal=CommitJournal(folder/'committed_updates_original.jsonl');binding=augmentation_binding(config,stop_check)
    prohibit_model_transforms(model,torch);model._broad_processor=processor;history=[];native_proof=None;expected_next=None;state=None
    original_journal=None;branch_receipt=None;last_prepared=None;phase_metrics=[];whole_begin=time.monotonic()
    def proposal():return DrawTransaction(cursor,carried,optimizer_updates=optimizer.updates,resolve_TR=resolve)
    def contexts_for(transaction):
        draws=[]
        for row,draw in zip(transaction.TR_identity,transaction.TR_draws):draws.append({**row,**draw,'phase':config['phase']})
        for i,row in enumerate(transaction.replay_rows):draws.append({'source_id':row['source_id'],'language':row['language'],
            'ordinal':None,'phase':config['phase'],'epoch':transaction.TR_draws[-1]['epoch'],
            'draw_index':transaction.TR_draws[-1]['draw_index']+1+i})
        contexts=tuple(AugmentationDrawContext(d['source_id'],d['language'],d['epoch'],d['draw_index'],phase=d['phase']) for d in draws)
        return draws,contexts
    def prepare(transaction,scope):
        draws,contexts=contexts_for(transaction);part=folder/scope
        prepared=decode_prepared(transaction.TR_draws,contexts[:14],part,config,batch_id='update_'+str(optimizer.updates+1),
            admit=admit,stop_check=stop_check,diagnostic_only=True)
        rows=[v['row'] for v in prepared['rows']]
        for item,row in zip(prepared['rows'],rows):
            old=by_ordinal[row['broad_original']['ordinal']]
            require(all(old[k]==v for k,v in row['broad_original'].items()) and row['tr_metadata_labels']==old['BPE']['target_ids'],
                'Exact independently pinned raw fields/targets differ')
        rows.extend({**r,'augmentation_role':'replay'} for r in transaction.replay_rows)
        reader=prepared_reader(prepared,rows,contexts,binding,batch_id='update_'+str(optimizer.updates+1),stop_check=stop_check)
        return prepared,rows,draws,reader,part
    try:
        for update in range(1,21):
            transaction=proposal()
            prepared,rows,draws,reader,part=prepare(transaction,folder.name+'_actual_'+str(update))
            last_prepared=(prepared,part,'update_'+str(update))
            if expected_next is not None and update==expected_next['expected_update']:
                require(expected_next is not None and transaction.identity_digest==expected_next['transaction'] and
                    prepared['augmentation_transport']['semantic_sha256']==expected_next['transport_semantic'],
                    'Actual postrestore source/draw/augmentation differs from pre-save expectation')
            current=sample_memory(os.getpid(),'canonical_repeated_geometry_admission')['worker_private_memory'];require(current['available'],'Fresh admission private sample')
            if update==1:budget=2_000_000_000
            else:
                require(native_proof and native_proof['available_sample_count']>=3 and native_proof['error'] is None and native_proof['joined'],'Measured available native peak required')
                budget=max(250_000_000,native_proof['private_peak_bytes']-current['private_bytes']+250_000_000)
            write_json(folder/('update_'+str(update)+'_budget.json'),{'incremental_bytes':budget,'current_private_bytes':current['private_bytes'],
                'prior_native_phase':native_proof,'geometry_scope':'same14 exact full sources + two carried replay rows; no production extrapolation',
                'CPU_child_closed_before_native':True,'parent_does_not_apply_augmentation':True,'global_guards_unchanged':True})
            admit('native_integrated_update_'+str(update),budget)
            native_started=time.monotonic();peak=NativePeak();peak.start()
            try:item=native_update(model,teacher,optimizer,rows,draws,reader,torch=torch,population_mean_U=declaration['fixture_mean_U'],
                update=update,stage=stage,stop_check=stop_check,phase=config['phase'])
            finally:native_proof=peak.finish()
            native_proof['wall_seconds']=time.monotonic()-native_started
            native_proof['training_audio_seconds']=sum(float(r['duration']) for r in rows)
            phase_metrics.append(native_proof)
            state=transaction.commit(optimizer=optimizer,journal=journal,native_receipt=item);history.append(item)
            write_json(folder/('native_phase_'+str(update)+'.json'),native_proof)
            write_json(folder/'native_progress.json',{'completed_updates':update,'history':history,'clocks':[c+optimizer.updates for c in (64250,2000,62250)],'diagnostic_only':True})
            stage('integrated_caller_update_committed',update=update,TR_draws=cursor.draws,coverage=cursor.coverage(),native_memory=native_proof)
            cleanup_prepared(prepared,part,batch_id='update_'+str(update),journal=journal,discard=False,stop_check=stop_check);last_prepared=None
            del rows,reader,prepared
            if update in (1,10):
                # Capture next inputs BEFORE saving/restoring checkpoint1.
                pending=proposal();expected,_,_,_,expected_folder=prepare(pending,folder.name+'_expected_next_'+str(update+1))
                expected_next={'expected_update':update+1,'transaction':pending.identity_digest,'transport_semantic':expected['augmentation_transport']['semantic_sha256']}
                cleanup_prepared(expected,expected_folder,batch_id='update_'+str(update+1),journal=journal,discard=True,stop_check=stop_check);pending.abandon();del expected,pending
                admit('integrated_checkpoint_'+str(update),64_000_000)
                checkpoint1=streamed_checkpoint(folder/('after_update'+str(update)+'.state.json'),torch=torch,optimizer=optimizer,state=state,provenance=provenance,
                    identity=identity,augmentation=binding,journal=journal,diagnostic_only=True,guard=stop_check)
                old_prefix=journal.binding();journal.append({'kind':'intentional_fixture_uncheckpointed_preparation_tail','actual_Adam_commit':False,'authoritative_frontier_advanced':False})
                original_journal=journal.binding()
                admit('integrated_exact_branched_restore_'+str(update),350_000_000)
                state,journal,branch_receipt=streamed_restore(checkpoint1,torch=torch,model=model,optimizer=optimizer,identity=identity,
                    augmentation=binding,new_journal_path=folder/('committed_updates_resumed_'+str(update)+'.jsonl'),diagnostic_only=True,guard=stop_check)
                cursor.load_state_dict(state['broad_cursor']);carried.load_state_dict(state['carried_replay'])
                require(optimizer.updates==update and cursor.draws==14*update and journal.binding()['sha256']==old_prefix['sha256'],'Actual checkpoint1 frontier/Adam/prefix restoration')
                write_json(folder/('resume_proof_'+str(update)+'.json'),{'passed':True,'checkpoint':checkpoint1,'all207_weights_and_all3_Adam_byte_exact':True,
                    'Python_Torch_CUDA_RNG_exact':True,'source_replay_queues_and_selector_RNG_carried':True,'pre_save_expected_next':expected_next,
                    'preserved_old_journal':original_journal,'branched_restore':branch_receipt,'diagnostic_only':True})
        require(cursor.draws==280 and cursor.coverage()['complete'] and carried.updates==20,'Exactly20 repeated fixed14-source diagnostic cycles and carried replay updates;0corpuspasses')
        admit('integrated_checkpoint_final',64_000_000)
        saved=streamed_checkpoint(folder/'after_update20.state.json',torch=torch,optimizer=optimizer,state=state,provenance=provenance,
            identity=identity,augmentation=binding,journal=journal,diagnostic_only=True,guard=stop_check)
        applied=sum(e['augmentation']['applied'] for item in history for e in item['augmentation']['rows'])
        require(applied>0,'At least one actual transform required')
        report={'passed':True,'native_updates':20,'actual_augmented_waveforms':applied,'history':history,'checkpoint':saved,
            'final_clocks':[64270,2020,62270],'fixture_only_coverage':cursor.coverage(),'full_corpus_complete_passes':0,
            'pre_save_actual_postrestore_draw_and_CPU_augmentation_exact':True,'journal_tail_preserved_and_branched':branch_receipt,
            'transient_caches_verified_and_evicted_at_boundaries':True,'last_native_memory':native_proof,'production_ready':False,
            'diagnostic_only':True,'declaration':declaration,'native_phase_metrics':phase_metrics,
            'prepare_probe_wall_seconds':time.monotonic()-whole_begin,'total_native_wall_seconds':sum(v['wall_seconds'] for v in phase_metrics),
            'total_training_audio_seconds':sum(v['training_audio_seconds'] for v in phase_metrics),
            'legacy_COW_mapped_CPU_steps_replaced_by_owned_clones':True,'same_owned_process_all20_updates':True}
        write_json(folder/'production_caller_probe.json',report);return report
    finally:
        if last_prepared is not None:
            # A failed native/forecast leaves no coherent new sampler commit;
            # discard only preparation, never repair/retry mutated live state.
            prepared,part,batch=last_prepared
            cleanup_prepared(prepared,part,batch_id=batch,journal=journal,discard=True,stop_check=lambda:None)
        if tr is not original:tr.close()
        original.close()
