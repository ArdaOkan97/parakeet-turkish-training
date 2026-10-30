"""Persistent ROOT-authorized two-control shared-head trials; no ML imports.

Every independent shared-head trial starts from CLOSED phase1 step7000/K500. A real one-update
smoke, source300 equality and serialized-state checks precede its main run.
On monitor failure the queue requests a whole-update stop and waits for its
owned controller. An unobserved close blocks all successors. This new executor
never takes over a live v1 queue; ROOT must confirm closure and archive its state
before a fresh v2 namespace is authorized.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import re
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PYTHON = ROOT / 'work/monitor-env/Scripts/python.exe'
REFRESH_PYTHON = ROOT / 'work/training-env/Scripts/python.exe'
SOURCE_LOOP = HERE / 'runs/adaptive_20261011_hf_TSC_BPE512_acquire_01_loop_01'
STATE = HERE / 'experiment_queue_state.json'
CONTROLLER = HERE / 'run_TSC_bpe_shared_head_memory_v2.py'
CREATOR = HERE / 'record_TSC_BPE_shared_head_source_v1.py'
LANGUAGES = ('tr', 'en', 'es', 'it', 'fr', 'de')
LIMITS = {'en': 69, 'es': 52, 'it': 30, 'fr': 74, 'de': 70}
PHASE = 'HF_TSC_BPE512_new_rows_plus_shared_heads_v1'
SOURCE_CHECKPOINT_SHA = 'd228603b1394adafe55a4f7e6209983643b2e5afcaded881006436d3351943f7'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    for attempt in range(20):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.1)


def verify_pins(manifest):
    required_sources = {Path(__file__).name, CONTROLLER.name, CREATOR.name,
                        'train_lora_batched_tsc_bpe_shared_head_memory_v2.py',
                        'tsc_bpe_shared_head_memory_v2.py',
                        'bpe_newrows_shared_head_adaptation_v1.py',
                        'compact_training_row_objects_v1.py',
                        'native_tdt_checkpointed_v4.py', 'metrics.py'}
    require(set(manifest['source_sha256']) == required_sources, 'Complete queue dependency pins required')
    for name, digest in manifest['source_sha256'].items():
        require(sha(HERE / name) == digest, 'Queue dependency changed: ' + name)
    require(manifest['kind'] == 'ROOT_authorized_TSC_shared_head_memory_queue_v1'
            and manifest['source_selected_step'] == 7000
            and manifest['source_acquisition_updates'] == 500
            and manifest['original_source_step'] == 6500
            and manifest['source_checkpoint_sha256'] == SOURCE_CHECKPOINT_SHA,
            'Queue must use the qualified acquired7000 state, not a regressed phase2 state')
    require(manifest['row_lr'] == 3e-4 and manifest['shared_learning_rates'] == [1e-5,3e-6]
            and type(manifest['material_TR_WER_improvement']) in (int,float)
            and math.isfinite(manifest['material_TR_WER_improvement']) and .005 <= manifest['material_TR_WER_improvement'] <= .1
            and manifest['main_updates'] == 1000
            and manifest['ROOT_exclusive_GPU_authorized'] is True,
            'Only row3e-4/shared1e-5 then3e-6 independent1000 screens are authorized')
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{2,110}', manifest['queue_id']) is not None,
            'Safe fresh queue namespace required')
    trial_namespace = manifest.get('trial_namespace')
    require(isinstance(trial_namespace, str)
            and re.fullmatch(r'[A-Za-z0-9_-]{2,24}', trial_namespace) is not None,
            'Safe explicitly declared trial namespace required')
    verify_closed_predecessor(manifest)



CLEANUP_GRACE_SECONDS = 600.


def option(arguments, flag):
    require(arguments.count(flag) == 1, 'One owned-controller option required: ' + flag)
    index = arguments.index(flag)
    require(index + 1 < len(arguments), 'Missing owned-controller option: ' + flag)
    return arguments[index + 1]


def cooperative_stop_path(arguments):
    """Only the exact owned GPU-controller command may name a stopfile."""
    if '--gpu-window-authorized' not in arguments:
        return None
    require(str(CONTROLLER) in arguments and '--validate-only' not in arguments,
            'GPU cleanup requires the pinned controller command')
    run_id = option(arguments, '--run-id')
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{2,110}', run_id) is not None,
            'Invalid owned run namespace')
    loop = Path(option(arguments, '--output-dir')).resolve()
    require(loop.parent == (HERE / 'runs').resolve()
            and loop.name == run_id + '_loop_01', 'Cleanup stop must be in this owned controller loop')
    return loop / (run_id + '_stop.txt')


def request_owned_stop(path):
    # The controller creates its own fresh directory. Creating it here would
    # violate admission during startup; retry after it exists instead.
    if path is None or not path.parent.is_dir():
        return False
    try:
        with path.open('x', encoding='utf-8') as stream:
            stream.write('Cooperative whole-update save/DEV: queue monitor failure; ROOT review required\n')
    except FileExistsError:
        pass  # Preserve an existing ROOT/resource/budget stop byte-for-byte.
    return True


def close_owned_after_failure(child, stop_path, *, grace_seconds=CLEANUP_GRACE_SECONDS):
    """Use only our retained Popen handle; no OS enumeration or forced kill."""
    evidence = {'pid': child.pid, 'cleanup_requested': True,
                'started_at_utc': now(), 'grace_seconds': grace_seconds,
                'cooperative_stop_path': str(stop_path) if stop_path else None,
                'cooperative_stop_present_or_created': False,
                'actual_wait_observed': False, 'closure_observed': False,
                'exit_code': None, 'forced_termination_used': False, 'cleanup_errors': []}
    deadline = time.monotonic() + grace_seconds
    # wait(0), unlike a status file or missing PID, can actually observe an
    # already-closed owned child without requesting an unnecessary stop.
    timeout = 0.
    while True:
        try:
            code = child.wait(timeout=timeout)
            evidence.update(actual_wait_observed=True, closure_observed=True,
                            exit_code=code, observed_at_utc=now())
            return evidence
        except subprocess.TimeoutExpired:
            pass
        except BaseException as error:
            evidence['cleanup_errors'].append(repr(error))
        if stop_path is not None and not evidence['cooperative_stop_present_or_created']:
            try:
                evidence['cooperative_stop_present_or_created'] = request_owned_stop(stop_path)
            except BaseException as error:
                evidence['cleanup_errors'].append(repr(error))
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            evidence.update(status='owned_controller_closure_unobserved', finished_at_utc=now())
            return evidence
        timeout = min(5., remaining)
        # Some handle errors return immediately. Avoid spinning on a broken
        # wait handle while retaining the same bounded cleanup deadline.
        if evidence['cleanup_errors']:
            try:
                time.sleep(min(.1, remaining))
            except BaseException as error:
                evidence['cleanup_errors'].append(repr(error))


def execute_owned(label, arguments, queue, publish, state):
    """Observe actual closure before reporting repair or admitting a successor."""
    stop_path = cooperative_stop_path(arguments)
    log = queue / (label + '.log')
    publish(status='running', current_stage=label, current_command=arguments, current_log=str(log))
    child = None
    close = None
    error = None
    try:
        with log.open('x', encoding='utf-8') as stream:
            child = subprocess.Popen(arguments, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            # Keep the handle before this fallible state publication.
            state['owned_child_pid'] = child.pid
            publish(owned_child_pid=child.pid, child_started_at_utc=now())
            while child.poll() is None:
                publish()
                time.sleep(5)
            code = child.wait()
            close = {'pid': child.pid, 'cleanup_requested': False,
                     'actual_wait_observed': True, 'closure_observed': True,
                     'exit_code': code, 'observed_at_utc': now(), 'forced_termination_used': False}
    except BaseException as caught:
        error = repr(caught)
        raise
    finally:
        if child is not None:
            if close is None:
                close = close_owned_after_failure(child, stop_path)
            close.update(kind='TSC_row_LR_queue_owned_child_closure_v2', stage=label,
                         monitoring_error=error, objective_complete=False)
            # Even if the publication that caused the failure remains broken,
            # retain independent queue-local cleanup evidence and emit it to
            # ROOT's captured stdout. Never clear the PID without actual wait.
            state['last_owned_child_cleanup'] = close
            state['owned_child_pid'] = None if close['closure_observed'] else child.pid
            if close['actual_wait_observed']:
                state['last_actual_waitpid'] = {'pid': child.pid, 'exit_code': close['exit_code'],
                                               'observed_at_utc': close['observed_at_utc']}
            try:
                write(queue / (label + '_owned_closure.json'), close)
            except BaseException as persistence_error:
                state['cleanup_evidence_persistence_error'] = repr(persistence_error)
            print(json.dumps({'owned_child_closure': close}, allow_nan=False), flush=True)
    publish(last_actual_waitpid=state['last_actual_waitpid'], owned_child_pid=None)
    require(close['exit_code'] == 0, 'Actual child failed at ' + label + '; inspect ' + str(log))

IDENTITY_FIELDS=('source_id','sentence_id','language','split','reference','dataset_revision','audio_sha256','wav_sha256','duration')


def saved_predictions(path):
    """Keep only fixed-ID/reference/hypothesis fields; discard large confidence arrays perrow."""
    rows=[];emitted=0;unique=set();utterances=0
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            require(line.strip() and len(line)<=8000000,'Bounded complete saved prediction line required')
            value=json.loads(line);tokens=value.get('confidence',{}).get('tokens')
            require(type(tokens) is list and all(type(item.get('token_id')) is int for item in tokens),
                'Actual emitted token paths required; missing paths cannot be zero')
            hits=[token['token_id'] for token in tokens if 8199<=token['token_id']<8711]
            emitted+=len(hits);unique.update(hits);utterances+=bool(hits)
            rows.append({key:value[key] for key in (*IDENTITY_FIELDS,'hypothesis')})
    require(len(rows)==50 and len({r['source_id'] for r in rows})==50,'Complete unique fixed50 predictions required')
    return rows,{'emitted_count':emitted,'unique_ids':sorted(unique),'unique_id_count':len(unique),'utterance_count':utterances}


def validate_update_marker(row, local, row_lr, shared_lr):
    require(type(local) is int and local>0 and row.get('step')==7000+local
        and row.get('adaptation_updates_completed')==local and row.get('source_phase1_step')==7000
        and row.get('original_source_selected_step')==6500 and row.get('selected_acquisition_updates')==500
        and row.get('row_Adam_clock')==500+local and row.get('shared_Adam_clock')==6500+local
        and row.get('row_lr')==row_lr and row.get('shared_lr')==row.get('head_lr')==shared_lr
        and row.get('lora_lr')==0. and row.get('new_token_update_multiplier')==1.
        and row.get('old_six_character_update_multiplier_applied') is False,
        'Actual shared localN/rowK+N/sharedS+N/LRs/mask policy changed')


def validate_phase_frontier(config, selection, history_count, row_lr, shared_lr):
    local=selection.get('adaptation_updates_completed')
    identity=config.get('config_identity',{});phase=identity.get('adaptation_phase',{});opt=identity.get('adaptation_optimizers',{})
    require(type(local) is int and 0<local<=1000 and selection.get('phase')==config.get('phase')==identity.get('phase')==PHASE
        and selection.get('heldout_tests_evaluated') is False and selection.get('source_phase1_step')==7000
        and selection.get('original_source_selected_step')==6500 and selection.get('selected_acquisition_updates')==500
        and selection.get('row_Adam_clock')==500+local and selection.get('shared_Adam_clock')==6500+local
        and selection.get('completed_steps')==7000+local==history_count
        and config.get('row_lr')==phase.get('row_lr')==opt.get('active_row_lr')==row_lr
        and config.get('shared_lr')==phase.get('shared_lr')==opt.get('active_shared_lr')==shared_lr
        and phase.get('active_parameter_tensors')==opt.get('active_parameter_tensors')==15
        and phase.get('active_shared_head_tensors')==opt.get('active_shared_head_tensors')==12
        and phase.get('frozen_encoder_LoRA_tensors')==opt.get('frozen_encoder_LoRA_tensors')==192
        and opt.get('kind')=='HF_TSC_BPE512_new_rows_plus_shared_heads_identity_v1'
        and opt.get('utility_sha256')=='384d01abe9b7105761491b8acda5d3de4eb8c40ae679eee2701269079cf9967c'
        and opt.get('source_moments_or_clocks_reset') is False and opt.get('original12_shared_moments_active') is True
        and opt.get('original3_resized_table_moments_active') is False and opt.get('original192_encoder_LoRA_moments_active') is False
        and phase.get('further_unfreeze_allowed') is False and config.get('turkish_clips_used')==253798
        and config.get('replay_clips_used')==1595,
        'Original source7000/S6500/K500 and actual active15/frozen192 complete frontier required')
    return local


def materially_improving_latest(latest, source_wer, threshold):
    require(type(source_wer) in (int,float) and math.isfinite(source_wer) and 0<=source_wer
        and type(threshold) in (int,float) and math.isfinite(threshold) and threshold>=.005,
        'Declared finite material-improvement threshold and actual initialWER required')
    return (latest.get('retention_passed') is True and latest['TR_WER']<source_wer
        and source_wer-latest['TR_WER']>=threshold)


def review_closed(run_id, loop, smoke, row_lr, shared_lr, material_threshold):
    """Only after actual ownedwait0: rescore every complete saved fixed50x6 DEV."""
    from metrics import score_transcriptions
    status=read(loop/'loop_status.json')
    require(status['run_id']==run_id and status['controller_sha256']==sha(CONTROLLER)
        and status['status']=='train_window_ended' and status['child_exit_code']==0
        and status['resource_review_required'] is False
        and status['stop_category'] in ('phase_bound','retention','development_target_met','plateau'),
        'Actual successfully CLOSED pinned shared-head child required')
    native=Path(status['paths']['native']);completion=read(native.with_suffix('.completion.json'))
    require(completion['status']=='complete' and completion['sources_unchanged'] is True
        and not completion.get('error') and completion['run_id']==run_id
        and completion['backend_ready_sha256']==sha(native)
        and completion['actual_dispatch_source_sha256']==sha(CONTROLLER),
        'Actual unchanged native completion must bind startup and executor')
    run=HERE/'runs'/run_id;proof=read(run/'TSC_BPE_shared_head_initial_source_equivalence.json')
    require(proof['kind']=='TSC_BPE512_shared_heads_before_update_source_equivalence_v1'
        and proof['run_id']==run_id and proof['source_selected_step']==7000
        and proof['original_source_step']==6500 and proof['selected_acquisition_updates']==500
        and proof['compared_fixed_DEV_utterances']==300
        and all(proof.get(key) is True for key in ('passed','actual_GPU_decode','before_shared_head_updates',
            'unsuppressed_new512_ids','all_raw_transcripts_and_emitted_paths_exact','all_six_ASR_counts_exact',
            'all_five_retention_gates_passed','RNG_restored_after_added_evaluation',
            'module_training_modes_restored_after_added_evaluation')),
        'Actual unsuppressed source300 equality/RNG/modes required')
    release=read(run/'source_admission_release.json');compact=release['training_row_compaction']
    require(release['kind']=='shared_heads_startup_admission_release_v2' and release['passed'] is True
        and release['before_first_optimizer_update'] is True and release['initial_BEST_and_latest_committed'] is True
        and release['actual_source300_sha256']==sha(run/'TSC_BPE_shared_head_initial_source_equivalence.json')
        and release['expanded_inherited_runtime_alias_removed'] is True and release['authoritative_history_rows_retained']==7000
        and release['row_Adam_clock']==500 and release['shared_Adam_clock']==6500
        and release['cache_release']['shared_head_admission_entries_after']==0
        and release['cache_release']['original_acquisition_admission_entries_after']==0,
        'Actual startup-only host release must preserve authoritative counters/history')
    require(compact['kind']=='audited_training_row_object_compaction_v1' and compact['passed'] is True
        and compact['rows_verified']==253798 and compact['same_row_list_and_row_dict_objects'] is True
        and compact['same_label_list_objects'] is True and compact['all_rows_canonical_equal'] is True
        and compact['all_row_key_orders_equal'] is True
        and compact['ordered_row_canonical_sha256_before']==compact['ordered_row_canonical_sha256_after']
        and compact['no_RNG_drawn'] is True and compact['label_vocab_size']==8711,
        'Every admitted training row/label/object/value/order must stay exact')
    config=read(run/'run_config.json');selection=read(run/'selection.json');history=read(run/'training_history.json')
    local=validate_phase_frontier(config,selection,len(history),row_lr,shared_lr)
    require(config['run_id']==selection['run_id']==run_id and history[6999]['step']==7000
        and all(row['step']==index for index,row in enumerate(history,1)), 'Actual history source/order/run differs')
    for n,row in enumerate(history[7000:],1):validate_update_marker(row,n,row_lr,shared_lr)
    if smoke:
        mechanical=read(run/'adaptation_actual_one_update_smoke.json')
        require(local==1 and mechanical['kind']=='TSC_BPE_shared_heads_actual_one_update_smoke_v1'
            and mechanical['run_id']==run_id and mechanical['row_clock_K_plus_N']==501
            and mechanical['shared_clock_S_plus_N']==6501 and mechanical['row_lr']==row_lr
            and mechanical['shared_lr']==shared_lr,'Actual one-update two-clock smoke required')
        for key in ('passed','actual_native_ASR_backward_and_two_Adam','all3_new_row_deltas_nonzero',
            'shared12_gradients_finite_and_all_deltas_nonzero',
            'protected_old_six_blank_duration_and_frozen192_encoder_LoRA_rawbytes_exact',
            'parked_original207_and_phase1_unchanged','complete207_two_optimizer_phase_save_load_passed',
            'all207_reloaded_weight_bytes_exact','all15_Adam_moments_clocks_and_groups_exact',
            'saved_sampler_history_config_and_actual_RNG_frontier_exact'):
            require(mechanical.get(key) is True,'Actual shared smoke failed:'+key)
    results=[]
    for row in history[7000:]:
        if 'validation' not in row:continue
        step=row['step'];result=HERE/'results'/f'{run_id}_step_{step}';summary=read(result/'summary.json')
        scores={};emissions={}
        for language in LANGUAGES:
            predictions,use=saved_predictions(result/f'validation_{language}.jsonl')
            initial,_=saved_predictions(HERE/'results'/f'{run_id}_step_7000'/f'validation_{language}.jsonl')
            require(all(all(a[key]==b[key] for key in IDENTITY_FIELDS) for a,b in zip(initial,predictions)),
                'Fixed reference/ID/order/audio metadata changed since actual source300')
            score=score_transcriptions(predictions,language=language);score.pop('details')
            require(score==summary[f'validation_{language}']['asr'],'Independent saved DEV rescore differs')
            scores[language]=score;emissions[language]=use['emitted_count']
        all_five=all(scores[language]['word_errors']<=cap for language,cap in LIMITS.items())
        require(all_five==row['validation']['retention_gate_passed'] and scores['tr']['wer']==row['validation']['turkish_wer'],
            'Independent actual TR/retention gate differs')
        results.append({'step':step,'TR_WER':scores['tr']['wer'],'scores':scores,'retention_passed':all_five,'new_BPE_emissions':emissions})
    require(results and results[-1]['step']==selection['completed_steps'],'Final complete actualDEV required')
    before,after=history[6999],history[-1];train_time=after['train_wall_seconds']-before['train_wall_seconds']
    require(train_time>0 and math.isfinite(train_time),'Actual phase trainwall must increase')
    delta={key:after[key]-before[key] for key in ('audio_seen_seconds','turkish_audio_seen_seconds','replay_audio_seen_seconds')}
    require(all(math.isfinite(v) and v>0 for v in delta.values()) and math.isclose(delta['audio_seen_seconds'],
        delta['turkish_audio_seen_seconds']+delta['replay_audio_seen_seconds'],rel_tol=1e-8,abs_tol=1e-6),'Actual phaseaudio components differ')
    throughput={key:delta[counter]/train_time for key,counter in (('mixed_audio_min_per_train_min','audio_seen_seconds'),
        ('TR_audio_min_per_train_min','turkish_audio_seen_seconds'),('replay_audio_min_per_train_min','replay_audio_seen_seconds'))}
    source_wer=config['initial_adaptation_turkish_wer'];latest=results[-1]
    meaningful=materially_improving_latest(latest,source_wer,material_threshold)
    return {'run_id':run_id,'passed':True,'updates_completed':local,'row_Adam_clock':500+local,'shared_Adam_clock':6500+local,
        'row_lr':row_lr,'shared_lr':shared_lr,'actual_native_completion_sha256':sha(native.with_suffix('.completion.json')),
        'DEV_points':results,'last_retention_passed':latest['retention_passed'],'source_phase1_TR_WER':source_wer,
        'latest_TR_WER':latest['TR_WER'],'latest_materially_improving_and_all5':meaningful,
        'material_TR_WER_improvement_threshold':material_threshold,'selected_step':selection['selected_step'],
        'throughput_excludes_startup_and_DEV':throughput,'phase_audio_seconds':delta,'phase_train_wall_seconds':train_time,
        'coverage_at_source':before.get('Turkish_domain_coverage'),'coverage_at_latest':after.get('Turkish_domain_coverage'),
        'source_and_phase_unique_coverage_are_cumulative_not_draws':True,'one_update_rate_is_not_steady':smoke,
        'goal_complete':False,'no_TEST_selection':True,'compaction_actual253798_verified':True}


def backlog_note(text, reference):
    command = [str(HERE / 'tools/backlog.exe'), 'task', 'edit', 'ASR-15',
               '--append-notes', text, '--add-ref', str(reference)]
    subprocess.run(command, cwd=HERE, check=True, capture_output=True, text=True, encoding='utf-8')
    subprocess.run([str(REFRESH_PYTHON), '-X', 'utf8', str(HERE / 'refresh_backlog.py')],
                   cwd=ROOT, check=True, capture_output=True, text=True, encoding='utf-8')


def read_verified_JSON(item):
    path=Path(item['path']).resolve()
    require(path.is_relative_to(ROOT) and path.suffix=='.json' and sha(path)==item['sha256'],
        'ROOT-owned predecessor record must be immutable workspaceJSON')
    return read(path)


def validate_predecessor_metadata(receipt, prior, *, queue_id, executor_sha, controller_sha, resource_review=None):
    require(receipt.get('schema')==1 and receipt.get('kind')=='ROOT_owned_closed_queue_for_shared_heads_v1'
        and receipt.get('status')=='actual_CLOSED_reviewed' and receipt.get('allow_next_shared_head_queue') is True
        and receipt.get('next_queue_id')==queue_id and receipt.get('next_queue_script_sha256')==executor_sha
        and receipt.get('next_controller_sha256')==controller_sha and receipt.get('guards_retained') is True
        and receipt.get('allocator_cap_bytes')==8000000000 and receipt.get('completed_update_cache_threshold_bytes')==3000000000
        and prior.get('owned_child_pid') is None,'Explicit closed-predecessor/new-executor approval and guards required')
    observed=receipt.get('owned_actual_exit',{});code=observed.get('exit_code')
    require(type(observed.get('session_id')) is int and observed['session_id']>0
        and type(code) is int and code in (0,1) and re.fullmatch(r'[0-9a-f]+',observed.get('tool_chunk','')) is not None
        and datetime.fromisoformat(observed['observed_at_utc'].replace('Z','+00:00')).tzinfo is not None
        and datetime.fromisoformat(observed['observed_at_utc'].replace('Z','+00:00'))>=datetime.fromisoformat(prior['finished_at_utc'].replace('Z','+00:00')),
        'Actual ROOT-owned CLOSED exit/time must follow predecessor terminal state')
    if code==0:
        require(prior.get('status') in ('trials_complete_review_required','competitive_latest_full_pass_review_required'),
            'CLOSED0 diagnostic boundary only; no claimed running predecessor')
    else:
        review=resource_review or {}
        require(prior.get('status')=='repair_required' and prior.get('owned_controller_closure_observed') is True
            and review.get('kind')=='ROOT_closed_queue_resource_repair_for_shared_heads_v1' and review.get('passed') is True
            and review.get('resource_review_cleared') is True and review.get('unchanged_guard_thresholds') is True
            and review.get('predecessor_state_sha256')==receipt['predecessor_state']['sha256']
            and review.get('predecessor_owned_exit')==observed and review.get('next_controller_sha256')==controller_sha
            and review.get('actual_native_complete_and_sources_unchanged') is True
            and review.get('actual_owned_controller_child_exit_code')==0,
            'CLOSED1 permits only explicit hash-bound ROOT reviewed/repaired resource, never inferredretry')
    return True


def verify_closed_predecessor(manifest):
    receipt=read_verified_JSON(manifest['predecessor_closed_receipt'])
    prior=read_verified_JSON(receipt['predecessor_state']);review=None
    if receipt.get('owned_actual_exit',{}).get('exit_code')==1:
        review=read_verified_JSON(receipt['resource_review'])
        controller=read_verified_JSON(review['actual_controller_status']);native=read_verified_JSON(review['actual_native_completion'])
        require(controller.get('child_exit_code')==0 and controller.get('resource_review_required') is True
            and controller.get('status')=='train_window_ended' and native.get('status')=='complete'
            and native.get('sources_unchanged') is True and not native.get('error')
            and native.get('run_id')==controller.get('run_id')
            and native.get('actual_dispatch_source_sha256')==controller.get('controller_sha256'),
            'Actual resource child whole-state closure/native completion must corroborate ROOT review')
    validate_predecessor_metadata(receipt,prior,queue_id=manifest['queue_id'],executor_sha=sha(__file__),
        controller_sha=sha(CONTROLLER),resource_review=review)


def fresh_memory_preflight():
    from resource_watchdog_v2 import sample_memory
    sample = sample_memory(os.getpid(), 'shared_head_memory_queue_before_GPU_dispatch')
    memory = sample['system_memory']
    historical_peak = 12_176_424_960
    projected = memory['commit_used_bytes'] + historical_peak
    require(memory['available'] is True and projected / memory['commit_limit_bytes'] < .95
            and memory['commit_limit_bytes'] - projected > 2_000_000_000,
            'Fresh baseline plus historical owned peak fails unchanged critical guard; defer GPU dispatch')
    return {'sample': memory, 'historical_owned_private_peak_bytes': historical_peak,
            'projected_fraction': projected / memory['commit_limit_bytes'],
            'prediction_is_not_actual_fit_proof': True}


def run(args):
    manifest = read(args.manifest)
    verify_pins(manifest)
    if args.validate_only:
        print(json.dumps({'passed': True, 'GPU_launched': False, 'queue': manifest['queue_id']}))
        return 0
    require(args.gpu_window_authorized, 'ROOT must authorize the sole sequential GPU queue')
    trial_namespace = manifest['trial_namespace']
    queue = ROOT / 'work/experiment_queues' / manifest['queue_id']
    require(not queue.exists() and not STATE.exists(), 'Fresh queue namespace/state required')
    queue.mkdir(parents=True)
    state = {'kind': manifest['kind'], 'queue_id': manifest['queue_id'], 'pid': os.getpid(),
             'ppid': os.getppid(), 'started_at_utc': now(), 'status': 'starting',
             'manifest': str(args.manifest.resolve()), 'manifest_sha256': sha(args.manifest),
             'trials': [], 'objective_complete': False, 'next_review': 'authorized recurring chat heartbeat'}

    def publish(**values):
        state.update(values, updated_at_utc=now())
        write(STATE, state)
        write(queue / 'state.json', state)

    def execute(label, arguments):
        verify_pins(manifest)
        if '--gpu-window-authorized' in arguments:
            publish(fresh_memory_preflight=fresh_memory_preflight())
        execute_owned(label, arguments, queue, publish, state)

    try:
        for tag, shared_lr in (('shared1e5',1e-5),('shared3e6',3e-6)):
            row_lr=manifest['row_lr']
            for stage in ('smoke', 'main'):
                run_id = f'adaptive_20261011_hf_TSC_BPE512_shared_head_{tag}_{trial_namespace}_{stage}'
                loop = HERE / 'runs' / (run_id + '_loop_01')
                receipt = HERE / f'TSC_BPE_shared_head_{tag}_{trial_namespace}_{stage}_CLOSED_source_v1.json'
                archive = ROOT / 'work/closed-sources' / f'TSC_BPE_shared_head_{tag}_{trial_namespace}_{stage}_source_v1'
                execute(tag + '_' + stage + '_archive', [str(PYTHON), '-S', str(CREATOR),
                        '--source-loop', str(SOURCE_LOOP), '--chosen-checkpoint-role', 'latest',
                        '--owned-session-id', '20495', '--owned-exit-chunk', '50fde1',
                        '--owned-exit-observed-at', '2026-10-11T22:31:27Z', '--row-lr', str(row_lr), '--shared-lr', str(shared_lr),
                        '--target-run-id', run_id, '--archive-dir', str(archive), '--receipt-output', str(receipt)])
                require(read(receipt)['artifacts']['chosen_checkpoint']['sha256'] == SOURCE_CHECKPOINT_SHA,
                        'Wrong source checkpoint admitted')
                command = [str(PYTHON), '-S', str(CONTROLLER), '--source-loop', str(SOURCE_LOOP),
                           '--source-receipt', str(receipt), '--run-id', run_id, '--output-dir', str(loop),
                           '--row-lr', str(row_lr), '--shared-lr', str(shared_lr), '--adaptation-max-updates', '1000', '--window-hours', '3']
                if stage == 'smoke':
                    command.extend(('--smoke-updates', '1'))
                preflight = HERE / f'{run_id}_queue_preflight_v2.json'
                execute(tag + '_' + stage + '_preflight', command + ['--validate-only', '--preflight-output', str(preflight)])
                publish(current_run_id=run_id, current_controller_loop=str(loop))
                execute(tag + '_' + stage + '_train', command + ['--gpu-window-authorized'])
                review = review_closed(run_id, loop, stage == 'smoke', row_lr, shared_lr, manifest['material_TR_WER_improvement'])
                report = queue / (tag + '_' + stage + '_review.json')
                write(report, review)
                state['trials'].append(review)
                publish(status='reviewed', current_stage=tag + '_' + stage + '_review')
                points = '; '.join(f"step{p['step']} TR WER{100*p['TR_WER']:.4f}% all5={p['retention_passed']} newTRtokens={p['new_BPE_emissions']['tr']}" for p in review['DEV_points'])
                backlog_note(f"Persistent local queue {manifest['queue_id']}: actual CLOSED {run_id}; rowLR={row_lr}/sharedLR={shared_lr}, encoderLoRA192 and old/six/blank/duration table rows frozen; named12 sourceS and acquired rowK moments retained. {points}. Goal remains incomplete; no held-out TEST used. Actual latest quality is reviewed before any next control.", report)
                print(run_id + ': ' + points, flush=True)
                if stage=='main' and tag=='shared1e5' and review['latest_materially_improving_and_all5']:
                    publish(status='competitive_latest_full_pass_review_required',finished_at_utc=now(),
                        current_stage=None,current_run_id=None,full_pass_candidate=review,
                        second_control_not_started=True,next_action='Heartbeat reviews actual CLOSED competitive latest fullstate for a declared same-run fullpass entry; do not reset learned rows to launch secondcontrol blindly')
                    backlog_note('First shared-head main ends with materially improved qualifying latest. Second control is paused for fullpass-entry decision; objective remains incomplete, all guard/scorer/data rules retained.',STATE)
                    return 0
                if stage == 'smoke' and not review['last_retention_passed']:
                    state['trials'].append({'arm': tag, 'main_skipped': True, 'reason': 'actual_smoke_retention_failure'})
                    publish()
                    break
        publish(status='trials_complete_review_required', finished_at_utc=now(),
                current_stage=None, current_run_id=None,
                next_action='Recurring chat heartbeat reviews learning curves and launches next quality experiment/full-pass recipe; short screens do not achieve the objective')
        backlog_note('Persistent two-control shared-head queue completed its scheduled diagnostic trial. Objective remains incomplete. Recurring chat review is active to select and launch the next experiment; do not interpret queue completion as competitive Turkish WER or a completed full dataset pass.', STATE)
        return 0
    except BaseException as error:
        # execute_owned has already attempted cooperative closure. A timeout is
        # explicitly unknown, and this terminal repair path cannot retry/spawn.
        state.update(status='repair_required', error=repr(error), finished_at_utc=now(),
                owned_controller_closure_observed=(state.get('last_owned_child_cleanup', {}).get('closure_observed')
                    if state.get('last_owned_child_cleanup', {}).get('stage') == state.get('current_stage') else None),
                next_action='ROOT checks actual owned closure before any repair or new GPU dispatch')
        try:
            publish()
        except BaseException as persistence_error:
            state['repair_publication_error'] = repr(persistence_error)
            try:
                write(queue / 'repair_state.json', state)
            except BaseException as fallback_error:
                state['repair_fallback_error'] = repr(fallback_error)
        print(json.dumps({'repair_required': state}, allow_nan=False), flush=True)
        return 1


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--manifest', type=Path, required=True)
    result.add_argument('--validate-only', action='store_true')
    result.add_argument('--gpu-window-authorized', action='store_true')
    return result


if __name__ == '__main__':
    raise SystemExit(run(parser().parse_args()))
