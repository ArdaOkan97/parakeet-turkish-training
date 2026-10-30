"""One fresh-process continuation of a coherent diagnostic checkpoint1.

The parent's uncheckpointed update2 is discarded. This fork restores all clocks,
queues and RNG from its actual coherent checkpoint1, performs exactly one new
update2, writes a bounded-slice checkpoint and restores it byte exactly. It is
not evidence of sustained same-process production stability.
"""
from dataclasses import asdict
import json, os, time
from pathlib import Path
from bpe_broad_production_v5 import *
from asr18_broad_integrated_probe_v2 import NativePeak
from fastmss_ordered_draw_binding_v1 import AugmentationDrawContext
from bpe_streaming_training_state_v1 import checkpoint as streamed_checkpoint, restore as streamed_restore

AREA = ROOT/'analysis/ASR18_memory_diagnosis_v1'

def prepare_probe(model, processor, teacher, optimizer, provenance, replay_snapshot, folder, config, *, torch, stage, admit, stop_check):
    from tsc_bpe_full_decoder_transition_v1 import preflight
    from train_lora_batched_tsc_bpe_full_decoder_v1 import load_catalogs
    from train_lora_batched_data_v11 import BucketCycler
    from types import SimpleNamespace
    from resource_watchdog_v2 import sample_memory
    require(config['production_enabled'] is False, 'Finite diagnostic cannot arm production')
    parent = config['coherent_diagnostic_parent']
    require(all(record(item['path']) == item for item in parent.values()), 'Exact immutable parent bindings required')
    parent_report = json.loads(Path(parent['report']['path']).read_text())
    resume = json.loads(Path(parent['resume_proof']['path']).read_text())
    declaration = json.loads(Path(parent['declaration']['path']).read_text())
    require(parent_report['actual_exit_code'] == 1 and parent_report['stop_reason'] is None
            and parent_report['durable_native_progress']['completed_updates'] == 2
            and resume['passed'] and resume['checkpoint'] == parent['checkpoint'], 'Actual coherent checkpoint1 required; uncheckpointed later state is discarded')
    legacy_identity = {'config': parent['config'], 'caller': parent['caller'], 'probe': parent['probe'], 'declaration': declaration}
    source_config = json.loads((ROOT/'runs/adaptive_20261015_TR_BPE512_full_decoder_02/run_config.json').read_text())
    context = preflight(SimpleNamespace(**source_config['arguments']))
    original, tr, replay_rows = load_catalogs(context)
    carried = CarriedReplay(replay_rows, replay_snapshot, BucketCycler)
    manifest = [json.loads(v) for v in Path(config['probe_manifest']).read_text().splitlines()]
    require(len(manifest) == len({r['ordinal'] for r in manifest}) == 14, 'Exact14 diagnostic population')
    by_ordinal = {r['ordinal']: r for r in manifest}; ordinals = sorted(by_ordinal)
    class FixtureCursor(CompletePassCursor):
        def preview(self, count):
            return [{**v, 'ordinal': ordinals[v['ordinal']-1]} for v in super().preview(count)]
    population = {'fixture_manifest': record(config['probe_manifest']), 'is_final_corpus': False}
    cursor = FixtureCursor(population=14, identity=population, seed=digest(replay_snapshot['turkish_rng']), passes=2)
    journal = None; prepared = None; part = None
    binding = augmentation_binding(config, stop_check)
    prohibit_model_transforms(model, torch); model._broad_processor = processor
    try:
        admit('fresh_process_restore_actual_coherent_checkpoint1', 350_000_000)
        state, journal, parent_branch = restore_branched(parent['checkpoint']['path'], torch=torch, model=model,
            optimizer=optimizer, identity=legacy_identity, augmentation=binding, new_journal_path=folder/'from_parent_checkpoint1.jsonl',
            diagnostic_only=True, guard=stop_check)
        cursor.load_state_dict(state['broad_cursor']); carried.load_state_dict(state['carried_replay'])
        require(optimizer.updates == 1 and cursor.draws == 14 and carried.updates == 1, 'Exact coherent parent clocks/frontiers required')
        stage('fresh_process_checkpoint1_restored', parent=parent['checkpoint'], clocks=[64251, 2001, 62251])
        identity = {'kind': 'ASR18_fresh_process_streamed_checkpoint_continuation_v1',
                    'caller': record(SOURCE/'bpe_broad_production_v5.py'), 'probe': record(__file__),
                    'config': record(config['_path']), 'legacy_parent_identity': legacy_identity,
                    'coherent_parent': parent['checkpoint'], 'uncheckpointed_parent_update2_discarded': True,
                    'streaming_state': record(SOURCE/'bpe_streaming_training_state_v1.py')}
        def resolve(ordinal):
            return {'ordinal': ordinal, 'source_id': by_ordinal[ordinal]['unique_id'], 'language': 'tr'}
        transaction = DrawTransaction(cursor, carried, optimizer_updates=optimizer.updates, resolve_TR=resolve)
        draws = [{**row, **draw, 'phase': config['phase']} for row, draw in zip(transaction.TR_identity, transaction.TR_draws)]
        draws.extend({'source_id': row['source_id'], 'language': row['language'], 'ordinal': None, 'phase': config['phase'],
                      'epoch': transaction.TR_draws[-1]['epoch'], 'draw_index': transaction.TR_draws[-1]['draw_index']+1+i}
                     for i, row in enumerate(transaction.replay_rows))
        contexts = tuple(AugmentationDrawContext(d['source_id'], d['language'], d['epoch'], d['draw_index'], phase=d['phase']) for d in draws)
        part = folder/(folder.name+'_actual_2')
        prepared = decode_prepared(transaction.TR_draws, contexts[:14], part, config, batch_id='update_2',
            admit=admit, stop_check=stop_check, diagnostic_only=True)
        require(transaction.identity_digest == resume['pre_save_expected_next']['transaction']
                and prepared['augmentation_transport']['semantic_sha256'] == resume['pre_save_expected_next']['transport_semantic'],
                'Fresh-process actual source/draw/augmentation differs from the original pre-save expectation')
        rows = [item['row'] for item in prepared['rows']]
        for item in rows:
            old = by_ordinal[item['broad_original']['ordinal']]
            require(all(old[k] == v for k, v in item['broad_original'].items())
                    and item['tr_metadata_labels'] == old['BPE']['target_ids'], 'Exact pinned raw source fields/full labels differ')
        rows.extend({**r, 'augmentation_role': 'replay'} for r in transaction.replay_rows)
        reader = prepared_reader(prepared, rows, contexts, binding, batch_id='update_2', stop_check=stop_check)
        admit('fresh_process_cold_native_update2', 2_000_000_000)
        peak = NativePeak(); peak.start()
        try:
            native = native_update(model, teacher, optimizer, rows, draws, reader, torch=torch,
                population_mean_U=declaration['fixture_mean_U'], update=2, stage=stage, stop_check=stop_check, phase=config['phase'])
        finally:
            native_memory = peak.finish()
        state = transaction.commit(optimizer=optimizer, journal=journal, native_receipt=native)
        write_json(folder/'native_progress.json', {'completed_updates': 2, 'native_updates_this_child': 1,
            'coherent_parent_updates': 1, 'clocks': [64252, 2002, 62252], 'history': [native], 'diagnostic_only': True})
        write_json(folder/'native_phase_2.json', native_memory)
        cleanup_prepared(prepared, part, batch_id='update_2', journal=journal, discard=False, stop_check=stop_check)
        prepared = None
        del rows, reader
        # CPU fixture proves bounded4MiB slices and a5.95MB measured increment.
        # GPU transfer is still measured here, under unchanged global guards.
        proof = config['streaming_checkpoint_CPU_proof']
        require(record(proof['path']) == proof and json.loads(Path(proof['path']).read_text())['passed'], 'Actual streaming CPU fixture required')
        admit('bounded_slice_final_checkpoint2', 64_000_000)
        before = sample_memory(os.getpid(), 'stream_checkpoint_before')['worker_private_memory']['private_bytes']
        peak = NativePeak(); peak.start()
        try:
            saved = streamed_checkpoint(folder/'after_update2.state.json', torch=torch, optimizer=optimizer, state=state,
                provenance=provenance, identity=identity, augmentation=binding, journal=journal, diagnostic_only=True, guard=stop_check)
        finally:
            checkpoint_memory = peak.finish()
        write_json(folder/'stream_checkpoint_memory.json', {'before_private_bytes': before, 'phase': checkpoint_memory,
            'sampled_increment_bytes': max(0, checkpoint_memory['private_peak_bytes']-before), 'forecast_bytes': 64_000_000})
        admit('bounded_slice_checkpoint2_exact_restore', 350_000_000)
        restored, restored_journal, own_branch = streamed_restore(saved, torch=torch, model=model, optimizer=optimizer,
            identity=identity, augmentation=binding, new_journal_path=folder/'after_stream_restore.jsonl', diagnostic_only=True, guard=stop_check)
        require(restored == state and optimizer.updates == 2 and cursor.coverage()['complete'], 'Final full state/frontier restoration differs')
        for partition in optimizer.optimizers:
            for parameter in partition.param_groups[0]['params']:
                # Tiny copied mutable CPU clock remains safe after mapping close.
                actual = partition.state[parameter]['step']
                before_step = actual.item(); actual.add_(0); require(actual.item() == before_step, 'Mutable owned CPU Adam clock differs')
        optimizer.validate(full=True); torch.cuda.synchronize(); torch.cuda.empty_cache()
        report = {'passed': True, 'native_updates_this_child': 1, 'coherent_chain_updates': 2,
            'clocks': [64252, 2002, 62252], 'checkpoint': saved, 'checkpoint_memory': checkpoint_memory,
            'native_memory': native_memory, 'parent_checkpoint1_branch': parent_branch,
            'stream_checkpoint2_branch': own_branch, 'all207_weights_all3Adam_and_PythonTorchCUDANumPy_RNG_exact': True,
            'CPU_Adam_steps_owned_mutable_after_mapping_close': True,
            'pre_save_expected_actual_fresh_process_draw_and_augmentation_exact': True,
            'fixture_only_coverage': cursor.coverage(), 'full_corpus_complete_passes': 0,
            'same_process_two_update_stability_proved': False, 'parent_uncheckpointed_update2_discarded': True,
            'source_optimizer_or_RNG_reset': False, 'diagnostic_only': True, 'production_ready': False}
        write_json(folder/'production_caller_probe.json', report)
        return report
    finally:
        if prepared is not None and journal is not None:
            cleanup_prepared(prepared, part, batch_id='update_2', journal=journal, discard=True, stop_check=lambda: None)
        if tr is not original:
            tr.close()
        original.close()
