"""CPU-only final audit of batch-8 quality recovery and full-audio TEST.

Original adaptive auditor bytes and original results remain read-only. Numerical
equivalence, DEV quality adoption, and final transcription quality are distinct.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
from pathlib import Path

import verify_adaptive_results as legacy
from verify_adaptive_results import *
from atomic_io_retry_v1 import write_json as atomic_write_json

RECOVERY_V1_AUDITOR_SHA256 = 'bec333b9fc0606da39168d7db0bf4c4f38a557b412b41fd8969b516adf41a361'
RECOVERY_V1_CONTROLLER_SHA256 = '5712954946ee3f68ebfc7e25075ecec2e718432f2393bf327c51d6c4905e406c'
ORIGINAL_AUDITOR_SHA256 = '68558524b47e079b294399e0e54ba235dff81e047cba60ef954b2b637d089443'
STRONG_V2_SHA256 = '91012f94ba80bd1c0bbee7f06855d3120d5c982e461184b30968c9fc0a4c9580'
ORIGINAL_STRONG_TEST_SHA256 = 'f25e3e4cf61134711389157d0f184a9bac8e1df4c86b77392ace8692af4776eb'
STRONG_V2_FILENAME = 'benchmark_strong_tr_test_v2.py'
LARGE_BATCH_AUDITOR_SHA256 = '72637b4f144192ec419eb9f1759b7ee599e0df7fd6519212874ffa3321ffc1a9'
TRANSITION_SHA256 = '26249c7267316710d349292410968044e325fc58ad7be00518f70554627714cd'


def stamp(value, check, label):
    value = datetime.fromisoformat(value)
    check.require(value.tzinfo is not None, label + ': timezone-aware timestamp')
    return value


def configuration(config):
    return {'batch_size':config['batch_size'], 'accumulation':config['accumulation'],
            'chunk_size':config['chunk_size'], 'tdt_implementation':config['tdt_implementation'],
            'encoder_checkpointing':config['encoder_activation_checkpointing']}


def audit_deadline(state,original,check):
    check.equal(state['deadline_at_utc'],original['deadline_at_utc'],'original absolute deadline retained')
    deadline=stamp(state['deadline_at_utc'],check,'original absolute deadline')
    initial=stamp(original['started_at_utc'],check,'original loop start')
    check.equal(deadline,initial+timedelta(hours=12),'absolute original 12-hour budget includes sleep/pause/failures')
    started=stamp(state['started_at_utc'],check,'recovery start')
    check.require(initial<=started<deadline,'recovery does not renew original budget')
    return initial,started,deadline


def audit_experimental_exposure(trial,trial_config,prior,check,previous_best=None):
    """An effective16 quality trial is deliberately not an equivalence screen."""
    check.equal((trial_config['batch_size'],trial_config['accumulation']),(8,2),'user-prioritized physicalB8/accumulation2/effective16')
    check.equal(trial['name'],'b8_quality','single explicit B8 quality trial')
    previous_best=previous_best or {'weights':prior['selection']['weights'],'parameters':prior['parameters']}
    check.equal(trial['parameters'],previous_best['parameters'],'B8 retains current best DEV winner learning parameters')
    check.equal(Path(trial['initialization_weights']).resolve(),Path(previous_best['weights']).resolve(),'B8 starts from best preceding DEV-qualified weights')
    near(check,trial['epochs_planned'],3.,'B8 requested bounded3TR epochs')
    prior_config=json_load(OUTPUT/'runs'/prior['run_id']/'run_config.json')
    for key in ['seed','replay_fraction','schedule','turkish_clips_used','replay_clips_used','lora_layout']:
        check.equal(trial_config[key],prior_config[key],'B8 preserves source/layout/objective policy '+key)
    check.equal((trial_config['tr_clips_per_update'],trial_config['replay_clips_per_update']),(8,8),'B8 per-update8TR+8replay clips')
    check.equal((trial_config['warmup_steps'],trial_config['eval_per_epoch'],trial_config['eval_epoch_fraction']),(25,4.,.25),'B8 declared warmup and DEV cadence')
    check.equal(trial_config['warmup_steps']*16,prior_config['warmup_steps']*4,'same400-clip warmup exposure; optimizer batch deliberately changed')
    check.equal(trial_config['epochs'],3.,'B8 training request3epochs under original wall budget')
    check.equal(trial_config['steps'],math.ceil(3*trial_config['turkish_clips_used']/8),'B8 configured steps from actual eligibleTRpool and per-updatecount')
    check.require(trial['batch_numerical_equivalence_proven'] is False,'B8 quality experiment never claims strict numerical equivalence')
    if 'selection_provenance' in trial:
        check.require(trial['status']=='complete' and trial['exit_code']==0 and not trial.get('selection_rejected'),
                      'B8 qualification requires successful DEV-selected evidence; early stopping is valid')
        completed=trial['selection']['completed_steps']
        check.require(type(completed) is int and 0<=completed<=trial_config['steps'],'B8 supports actual completed/early-stopped history including step0')



def audit_io(path, worker_name, run_id, check, *, expected_arguments=None):
    path = local_path(path, check, 'I/O adapter provenance')
    evidence = json_load(path)
    check.equal(evidence['kind'], 'atomic_io_retry_v1', 'versioned I/O adapter identity')
    check.equal(evidence['status'], 'complete', 'worker completed with durable adapter provenance')
    check.equal(evidence['training_semantics_changed'], False, 'adapter changes persistence only')
    check.equal(evidence['retry_winerrors'], [5,32,33], 'only Windows sharing/access errors retried')
    check.require(finite(evidence['retry_seconds']) and 0 < evidence['retry_seconds'] <= 60, 'bounded I/O retry budget')
    check.require(evidence['unique_same_directory_temporary'] is True and evidence['fsync_before_replace'] is True,
                  'private same-directory flushed atomic commits')
    worker = OUTPUT / worker_name
    check.equal(Path(evidence['worker']).resolve(), worker.resolve(), 'exact frozen worker path')
    for key, source in [('worker_sha256',worker),('launcher_sha256',OUTPUT/'launch_with_io_retry_v1.py'),
                        ('helper_sha256',OUTPUT/'atomic_io_retry_v1.py')]:
        check.equal(evidence[key], file_sha(source), 'actual adapter source before: ' + key)
        check.equal(evidence[key+'_after'], file_sha(source), 'actual adapter source after: ' + key)
    arguments = evidence['arguments']
    check.require(type(arguments) is list and all(type(x) is str for x in arguments), 'adapter exact string argument vector')
    arguments = arguments[1:] if arguments[:1] == ['--'] else arguments
    check.require(arguments.count('--run-id') == 1 and arguments[arguments.index('--run-id')+1] == run_id,
                  'adapter belongs to exact selected run')
    if expected_arguments is not None:
        check.equal(arguments, expected_arguments, 'worker arguments preserve recorded launch vector')
    started = stamp(evidence['started_at_utc'],check,'adapter start')
    finished = stamp(evidence['finished_at_utc'],check,'adapter finish')
    check.require(started <= finished, 'adapter timestamps ordered')
    for event in evidence['retry_events']:
        check.require(Path(event['path']).resolve().is_relative_to(ROOT.resolve()),'retried commit destination stays in workspace (private temp may be consumed)')
        check.require(type(event['retries']) is int and event['retries'] > 0
                      and finite(event['elapsed_seconds']) and 0 <= event['elapsed_seconds'] <= evidence['retry_seconds']+1,
                      'finite recorded bounded successful retries')
    return {'path':str(path),'sha256':file_sha(path),'worker':worker_name,
            'retry_events':len(evidence['retry_events']),'arguments':arguments,
            'started_at_utc':evidence['started_at_utc'],'finished_at_utc':evidence['finished_at_utc']}


def audit_transition(state,check):
    recovery=state['recovery']
    check.equal(recovery['version'],2,'explicit throughput-prioritized recovery version2')
    path=local_path(recovery['transition_manifest_snapshot_path'],check,'intentional transition snapshot')
    check.equal(file_sha(path),TRANSITION_SHA256,'reviewed authorized intentional B8 transition bytes')
    check.equal(recovery['transition_manifest_snapshot_sha256'],file_sha(path),'controller binds exact transition snapshot')
    original_path=local_path(recovery['transition_original_path'],check,'original intentional transition')
    check.equal(file_sha(original_path),file_sha(path),'original transition remains unchanged')
    check.equal(recovery['transition_original_sha256'],file_sha(path),'original transition exact controller binding')
    transition=json_load(path)
    check.require(transition['kind']=='user_prioritized_B8_transition' and transition['intentional_transition'] is True
                  and transition['user_authorized_throughput_priority'] is True and transition['prior_controller_exited'] is True,
                  'v1 stop was an authorized cooperative transition, not unclassified infrastructure damage')
    check.equal(transition['prior_controller_observed_exit_code'],1,'intentional owned-marker guard caused recorded v1 exit1')
    check.equal((transition['original_loop_id'],transition['prior_recovery_loop_id'],transition['target_recovery_loop_id']),
                (recovery['original_loop_id'],recovery['prior_recovery_loop_id'],state['loop_id']),'transition exact original/prior/new loop identity')
    check.equal(transition['original_deadline_at_utc'],state['deadline_at_utc'],'transition retains original absolute deadline')
    check.equal(transition['prior_controller_sha256'],RECOVERY_V1_CONTROLLER_SHA256,'intentional transition frozen prior controller')
    check.equal(transition['target_configuration'],{'batch_size':8,'accumulation':2,'effective_batch_size':16,'warmup_steps':25},'authorized B8/effective16 warmup25')
    prior_path=local_path(recovery['prior_recovery_status_snapshot_path'],check,'prior recovery status snapshot')
    check.equal(file_sha(prior_path),recovery['prior_recovery_status_snapshot_sha256'],'prior recovery status snapshot exact bytes')
    check.equal(file_sha(prior_path),transition['prior_status_snapshot_sha256'],'transition and recovery bind identical prior status')
    check.equal(file_sha(local_path(transition['prior_status_snapshot_path'],check,'original prior status snapshot')),file_sha(prior_path),'original prior transition status snapshot unchanged')
    check.equal(file_sha(OUTPUT/'runs'/recovery['prior_recovery_loop_id']/'loop_status.json'),file_sha(prior_path),'prior recovery final status remains unchanged')
    prior=json_load(prior_path)
    check.equal((prior['status'],prior['failure_message']),('failed','Trial ID/stop marker already exists.'),'observed v1 exit is the explicit owned-marker transition')
    check.require(stamp(prior['finished_at_utc'],check,'prior controller finish')
                  <=stamp(transition['created_at_utc'],check,'transition recorded')
                  <=stamp(state['started_at_utc'],check,'B8 recovery start'),'prior child/controller finish before successor begins')
    completed=transition['completed_C'];directory=OUTPUT/'runs'/completed['run_id']
    check.equal(completed['run_id'],recovery['prior_recovery_loop_id']+'_screen_C','exact cooperatively stopped C identity')
    for prefix,target in [('run_config',directory/'run_config.json'),('training_history',directory/'training_history.json'),
                          ('selection',directory/'selection.json'),('best_checkpoint',ROOT/'work'/(completed['run_id']+'_best.pt')),
                          ('latest_checkpoint',ROOT/'work'/(completed['run_id']+'_latest.pt'))]:
        check.equal(local_path(completed[prefix+'_path'],check,'preserved C '+prefix),target.resolve(),'exact prior C '+prefix+' path')
        check.equal(file_sha(target),completed[prefix+'_sha256'],'preserved C '+prefix+' bytes unchanged')
    selection=json_load(directory/'selection.json');history=json_load(directory/'training_history.json')
    check.equal((selection['selected_step'],selection['completed_steps'],selection['accepted']),
                (completed['selected_step'],completed['completed_steps'],completed['accepted']),'actual C completed DEV/checkpoint binding')
    check.equal(selection['completed_steps'],len(history),'cooperative C finished actual contiguous updates')
    check.equal(selection['stop_early_reason'],'controller_stop_file','C stopped cooperatively at authorized transition')
    expected={(ROOT/'work'/(recovery['prior_recovery_loop_id']+'_'+stage+'.stop')).resolve() for stage in ('screen_C','screen_B2','extend_1')}
    check.equal({Path(marker['path']).resolve() for marker in transition['owned_guard_markers']},expected,'exact owned guards prevent later v1 GPU children')
    for marker in transition['owned_guard_markers']:
        check.equal(file_sha(local_path(marker['path'],check,'owned transition marker')),marker['sha256'],'owned transition marker remains unchanged')
    return transition,prior


def audit_recovery(state, frozen, config, check):
    check.equal(file_sha(OUTPUT/'verify_adaptive_results.py'), ORIGINAL_AUDITOR_SHA256, 'original adaptive auditor byte-identical')
    check.equal(file_sha(OUTPUT/'verify_recovery_results_v1.py'),RECOVERY_V1_AUDITOR_SHA256,'prior recovery auditor byte-identical')
    check.equal(file_sha(OUTPUT/'run_experiment_loop_recovery.py'),RECOVERY_V1_CONTROLLER_SHA256,'prior recovery controller byte-identical')
    recovery = state['recovery']
    controller = local_path(recovery['controller_path'],check,'recovery controller')
    check.equal(controller,(OUTPUT/'run_experiment_loop_recovery_v2.py').resolve(),'versioned recovery controller path')
    check.equal(recovery['controller_sha256'],file_sha(controller),'actual frozen recovery controller bytes')
    source_snapshot = local_path(recovery['original_status_snapshot'],check,'preserved original loop snapshot')
    check.equal(recovery['original_status_snapshot_sha256'],file_sha(source_snapshot),'original failed loop snapshot bytes')
    original = json_load(source_snapshot)
    original_directory = OUTPUT/'runs'/recovery['original_loop_id']
    check.equal(file_sha(original_directory/'loop_status.json'),file_sha(source_snapshot),'original failed loop status remains unchanged')
    check.equal(original['status'],'strong_test_failed','recovery starts from infrastructure failure after B export')
    check.equal(original['loop_id'],recovery['original_loop_id'],'preserved original loop ID')
    initial,started,deadline=audit_deadline(state,original,check)
    transition,prior_recovery=audit_transition(state,check)
    old_freeze = source_snapshot.parent/'original_frozen_final_selection.json'
    check.equal(file_sha(old_freeze),recovery['original_frozen_selection_snapshot_sha256'],'preserved old selection snapshot bytes')
    check.equal(file_sha(old_freeze),file_sha(original_directory/'frozen_final_selection.json'),'original frozen selection unchanged')
    for key,name in [('io_retry_launcher_sha256','launch_with_io_retry_v1.py'),('io_retry_helper_sha256','atomic_io_retry_v1.py')]:
        check.equal(recovery[key],file_sha(OUTPUT/name),'recovery records actual versioned persistence adapter: '+name)
    check.equal(recovery['strong_test_worker'],STRONG_V2_FILENAME,'versioned full-audio strong TEST worker selected')
    check.equal(recovery['audit_worker'],Path(__file__).name,'actual versioned recovery auditor selected')
    check.equal(file_sha(OUTPUT/STRONG_V2_FILENAME),STRONG_V2_SHA256,'reviewed full-audio TEST source remains frozen')
    for name,digest in recovery['previous_output_artifacts'].items():
        check.require(Path(name).name==name,'safe prior artifact filename')
        check.equal(file_sha(source_snapshot.parent/'previous_output_artifacts'/name),digest,'prior output artifact preserved: '+name)
    carried_id = recovery['original_loop_id']+'_screen_B'
    carried = [trial for trial in state['trials'] if trial.get('carried_from_original_loop')]
    check.equal(len(carried),1,'exactly one prior completed B carried into DEV pool')
    carried = carried[0]
    check.equal(carried['run_id'],carried_id,'carried original B identity')
    prior = [trial for trial in original['trials'] if trial['run_id']==carried_id]
    check.equal(len(prior),1,'unique original B trial')
    for key in ['parameters','initialization_weights','initialization_sha256','completed_updates','status','exit_code']:
        check.equal(carried[key],prior[0][key],'carried B preserves prior '+key)
    original_frozen = json_load(old_freeze)
    check.equal(original_frozen['run_id'],carried_id,'old frozen winner is carried B')
    check.equal(carried['selection_provenance'],original_frozen['provenance'],'carried B exact frozen DEV/source/checkpoint binding')
    check.equal(carried['selected_configuration']['batch_size'],1,'carried B physical batch remains1')
    check.equal(carried['selected_configuration']['accumulation'],4,'carried B effective batch remains4')
    check.require(carried['batch_numerical_equivalence_proven'] is False
                  and carried['existing_training_configuration_verified'] is True,
                  'carried B operational verification never claims strict BF16 equivalence')
    adapter_reports=[]
    check.equal(sum(trial.get('carried_from_prior_recovery') is True for trial in state['trials']),1,'exactly one preserved cooperative C carried into DEV pool')
    check.require(sum(not trial.get('carried_from_original_loop') and not trial.get('carried_from_prior_recovery') for trial in state['trials'])<=1,'at most one new B8 quality child under the original budget')
    previous_best={'run_id':carried_id,'wer':original_frozen['wer'],'weights':original_frozen['weights'],
                   'parameters':carried['parameters'],'selected_configuration':carried['selected_configuration']}
    for trial in state['trials']:
        if trial.get('carried_from_original_loop'):continue
        trial_id=safe_run(trial['run_id'],check,'recovery trial')
        is_prior=trial.get('carried_from_prior_recovery') is True
        if is_prior:
            source=[entry for entry in prior_recovery['trials'] if entry['run_id']==trial_id]
            check.equal(len(source),1,'exactly one carried cooperative C trial')
            check.equal(trial_id,transition['completed_C']['run_id'],'only preserved C carried from prior recovery')
            for key,value in source[0].items():check.equal(trial[key],value,'carried C immutable prior trial field '+key)
            check.require(trial['prior_partial_exposure'] is True,'partial C exposure is explicit and not described as a completed quarter')
            check.require(trial['selection']['turkish_validation_wer']>=original_frozen['wer'],'observed prior C cannot outrank preserved B fallback')
        else:
            check.equal(trial['name'],'b8_quality','only new GPU training child is the authorized B8 quality run')
            check.equal(trial['batch_quality_experiment'],True,'changed effective optimizer batch is explicit quality experiment')
        begin=stamp(trial['started_at_utc'],check,'trial start')
        end=stamp(trial['finished_at_utc'],check,'trial finish')
        check.require((initial<=begin<=end<=started if is_prior else started<=begin<=end<=deadline),'carried or new trial fits its ordered original absolute budget')
        check.require(trial['optimizer_resume'] is False,'recovery fresh screen/extension optimizer policy unchanged')
        path=OUTPUT/'runs'/trial_id/'run_config.json'
        if not path.exists():
            check.require(trial['status']!='complete','missing training config cannot qualify')
            continue
        trial_config=json_load(path)
        check.equal(configuration(trial_config),trial['selected_configuration'],'candidate-specific actual batch configuration')
        check.equal(trial_config['batch_size']*trial_config['accumulation'],4 if is_prior else 16,'prior effective4 versus new effective16 quality optimizer batch')
        check.equal(trial_config['resume_exact'],None,'fresh optimizer screens/extensions do not pretend exact state resume')
        check.equal(Path(trial_config['resume_weights']).resolve(),Path(trial['initialization_weights']).resolve(),'actual warm-start weights match plan')
        check.equal(file_sha(local_path(trial['initialization_weights'],check,'trial warm start')),trial['initialization_sha256'],'warm-start trainable bytes frozen')
        for argument,key in [('--head-lr','head_lr'),('--lora-lr','lora_lr'),('--new-token-lr-multiplier','new_token_lr_multiplier')]:
            near(check,trial_config[key],trial['parameters'][key],'actual planned '+argument)
        check.equal((trial_config['replay_kl'],trial_config['duration_kl'],trial_config['feature_distillation']),(5.,.5,1.),'preservation objectives unchanged')
        check.require(trial['batch_numerical_equivalence_proven'] is False,'quality trial never claims BF16 numerical equivalence')
        check.equal(trial['existing_training_configuration_verified'],trial['selected_configuration']==carried['selected_configuration'],
                    'operational B1 verification is distinct from experimental B8 quality qualification')
        if trial['name'].startswith('extend_'):
            check.equal(Path(trial['initialization_weights']).resolve(),Path(previous_best['weights']).resolve(),'extension starts from preceding DEV-qualified winner')
            check.equal(trial['selected_configuration'],previous_best['selected_configuration'],'extension inherits preceding DEV-qualified configuration')
            check.equal(trial['parameters'],previous_best['parameters'],'extension inherits preceding DEV-qualified learning parameters')
            check.equal(trial['batch_quality_experiment'],False,'extension is not a new unequal-exposure B2 screen')
        if trial['status']=='complete' and trial['exit_code']==0:
            expected=trial['command'][trial['command'].index('--')+1:]
            adapter=audit_io(trial['io_retry_provenance_path'],'train_lora_batched.py',trial_id,check,expected_arguments=expected)
            check.equal(adapter['sha256'],trial['io_retry_provenance_sha256'],'controller exact worker adapter evidence bytes')
            check.equal(json_load(Path(adapter['path'])),trial['io_retry_provenance'],'controller exact worker adapter evidence contents')
            adapter_reports.append(adapter)
        if trial.get('batch_quality_experiment'):
            audit_experimental_exposure(trial,trial_config,prior[0],check,previous_best)
            check.equal(trial['backend_flags_documented'],state['recovery']['backend_flags_documented'],'B8 explicitly declares benchmark/training backend difference')
        if 'selection_provenance' in trial and not trial.get('selection_rejected'):
            score=trial['selection']['turkish_validation_wer']
            if score<previous_best['wer']:
                previous_best={'run_id':trial_id,'wer':score,'weights':trial['selection']['weights'],
                               'parameters':trial['parameters'],'selected_configuration':trial['selected_configuration']}
    check.equal(previous_best['run_id'],frozen['run_id'],'replayed DEV-only trial progression chooses frozen run')
    check.equal(previous_best['selected_configuration'],frozen['selected_configuration'],'replayed DEV-only progression chooses configuration')
    if state.get('reused_original_finalization'):
        reused=audit_carried_reuse(state,frozen,check)
    else:
        check.require(frozen['run_id']!=carried_id,'original B fallback must reuse its immutable finalization')
        io=audit_io(state['finalizer_io_retry_provenance_path'],'finalize_lora.py',frozen['run_id'],check,
                    expected_arguments=['--run-id',frozen['run_id']])
        check.equal(io['sha256'],state['finalizer_io_retry_provenance_sha256'],'controller finalizer adapter evidence bytes')
        check.require(stamp(io['finished_at_utc'],check,'finalizer finish')<=deadline,'fresh finalizer fits original absolute budget')
        adapter_reports.append(io)
        reused=None
    winner_config=frozen['selected_configuration']
    check.equal(configuration(config),winner_config,'frozen winner exact actual training configuration')
    check.equal(state['best_development_candidate']['selected_configuration'],winner_config,'controller winning configuration matches frozen')
    check.require((winner_config['batch_size'],winner_config['accumulation']) in ((1,4),(8,2)),'winner is preserved effective4 B1 or DEV-qualified effective16 B8')
    return {'original_deadline_at_utc':state['deadline_at_utc'],'original_status_sha256':file_sha(source_snapshot),
            'recovery_controller_sha256':file_sha(controller),'io_adapter_reports':adapter_reports,
            'reused_original_finalization':reused,
            'winner_configuration':winner_config,'prior_B_TEST_already_evaluated':True,
            'intentional_transition':{'path':str(Path(recovery['transition_manifest_snapshot_path']).resolve()),'sha256':TRANSITION_SHA256,
                                     'prior_controller_exit1_is_owned_transition':True,'cooperative_C_selected_step':transition['completed_C']['selected_step']},
            'selection_policy':'B8 changes effective optimizer batch; recovery choices use actual DEV and all retention gates only. Prior B TEST existed before recovery, so final TEST is descriptive.'}


def audit_carried_reuse(state,frozen,check):
    reused=state['reused_original_finalization']
    expected_id=state['recovery']['original_loop_id']+'_screen_B'
    check.equal(frozen['run_id'],expected_id,'only immutable original B may reuse finalization')
    check.equal((reused['run_id'],reused['selected_step']),(frozen['run_id'],frozen['selected_step']),'reuse selected run/step identity')
    directory=OUTPUT/'runs'/expected_id
    snapshot=Path(state['recovery']['original_status_snapshot']).parent/'reused_original_finalization.json'
    check.equal(json_load(snapshot),reused,'durable carried-export reuse provenance equals controller state')
    audit_path=local_path(reused['quality_verification_path'],check,'prior B independent quality report')
    check.equal(audit_path,(OUTPUT/'sleep_recovery_quality_verification.json').resolve(),'exact prior sleep quality snapshot')
    check.equal(file_sha(audit_path),reused['quality_verification_sha256'],'prior B quality verification bytes')
    quality=json_load(audit_path)
    check.require(quality['passed'] is True and quality['checks']==8244,'completed prior independent B quality verification')
    check.equal((quality['run_id'],quality['selected_step']),(frozen['run_id'],frozen['selected_step']),'prior quality verification selected state')
    for value,digest in quality['preserved_evidence_sha256'].items():
        check.equal(file_sha(local_path(value,check,'preserved B evidence')),digest,'immutable pre-recovery B evidence '+Path(value).name)
    check.equal(file_sha(directory/'selection.json'),reused['final_selection_sha256'],'old finalized B selection bytes remain unchanged')
    check.equal(file_sha(directory/'finalization_provenance.json'),reused['finalization_provenance_sha256'],'old B merge/export provenance unchanged')
    check.equal(json_load(directory/'finalization_provenance.json')['candidate_files_sha256'],reused['candidate_files_sha256'],'reuse binds original complete exported HF file dictionary')
    # audit_export below independently streams and verifies both candidate and
    # published model files. Reuse does not claim a newly executed finalizer.
    check.require('finalizer_io_retry_provenance_path' not in state,'carried reuse cannot claim a new finalizer adapter execution')
    return reused


def check_frozen_selection(frozen_path,state,baseline,check):
    # Rejected or incomplete controlled exposure never enters the winner pool.
    filtered={**state,'trials':[trial for trial in state['trials'] if not trial.get('selection_rejected')]}
    return legacy.check_frozen_selection(frozen_path,filtered,baseline,check)


def verify_whisper_prediction(row,source,processor,check):
    import soundfile as sf
    info=sf.info(source['audio_filepath']);samples=info.frames
    check.require(info.samplerate==16000 and info.channels==1 and info.subtype=='PCM_16','actual full monoPCM16/16k input')
    check.equal(file_sha(Path(source['audio_filepath'])),source['wav_sha256'],'actual complete WAV bytes retained')
    near(check,samples/16000,source['duration'],'actual full waveform duration',atol=1/16000)
    mode='short_form' if samples<=480000 else 'long_form'
    check.equal(row['inference_mode'],mode,'actual sample-count based duration route')
    check.equal(row['input_samples'],samples,'whole original waveform fed to inference')
    check.equal(row['input_attention_mask_frames'],math.ceil(samples/160),'frontend attention mask includes full tail')
    check.require(finite(row['inference_seconds']) and row['inference_seconds']>=0,'finite nonnegative actual inference timer')
    check.equal(row['eos_emitted'],True,'all underlying decoding windows complete')
    tokens=row['generated_token_ids']
    token_list(check,tokens,'saved emitted token IDs')
    short=json_load(OUTPUT/'results/strong_whisper_large_v3_greedy_dev_verified/summary.json')['protocol']
    generation=short if mode=='short_form' else {**short,'return_timestamps':True,'return_segments':True}
    expected_sha=hashlib.sha256(json.dumps(generation,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    check.equal(row['generation_protocol_sha256'],expected_sha,'exact predeclared per-recording generation protocol')
    if mode=='short_form':
        check.equal(row['input_mel_frames'],3000,'original short-form padded frontend')
        check.require('longform' not in row,'short-form metadata is not fabricated long-form')
        check_full_tokens(tokens,[50258,50268,50360,50364],check,'short-form actual prefix/EOS')
    else:
        frames=samples//160
        check.require(row['input_mel_frames']==frames>3000,'long-form frontend never truncates tail')
        verify_longform(row['longform'],tokens,frames,check)
    check.equal(processor.batch_decode([tokens],skip_special_tokens=True)[0].strip(),row['hypothesis'],'saved text decoded from actual emitted tokens')


def token_list(check,tokens,label):
    check.require(type(tokens) is list and all(type(value) is int and 0<=value<=51865 for value in tokens),label)


def check_full_tokens(tokens,prefix,check,label):
    token_list(check,tokens,label+': integer tokenizer IDs')
    check.require(tokens[:len(prefix)]==prefix and tokens[-1:]==[50257] and tokens.count(50257)==1,label)


def verify_longform(evidence,tokens,input_frames,check):
    windows=evidence['windows'];segments=evidence['segments']
    check.require(type(windows) is list and len(windows)>=2 and type(segments) is list and bool(segments),'multiple native long-form windows and returned segments')
    seek=0
    for window in windows:
        check.require(type(window['seek_start_frame']) is int and window['seek_start_frame']==seek<input_frames,'contiguous native seek progression without skipped windows')
        check.equal(window['total_input_frames'],input_frames,'all windows reference whole original mel input')
        check.equal(window['input_frame_count'],min(3000,input_frames-seek),'native window valid full-input span')
        check.equal(window['padded_window_frames'],3000,'native window original30s encoder padding')
        check.require(type(window['seek_advance_frames']) is int and 0<window['seek_advance_frames']<=3000,'native timestamp seek advances by bounded positive amount')
        check.equal(window['decoder_input_token_ids'],[50258,50268,50360],'actual Turkish/transcribe long-form prompt without previous-token conditioning')
        check_full_tokens(window['raw_generated_token_ids'],[50258,50268,50360],check,'native raw window actual prefix/EOS')
        check.equal(window['terminal_eos_emitted'],True,'native window emitted actual terminal EOS')
        seek+=window['seek_advance_frames']
    check.require(seek>=input_frames and evidence['final_seek_frame']==seek and evidence['full_input_consumed'] is True,'native generation reaches actual end of full input')
    all_tokens=[];previous=-1.
    for segment in segments:
        raw=segment['raw_generated_token_ids'];part=segment['tokens'];idxs=segment['idxs']
        token_list(check,part,'native returned segment token IDs')
        check_full_tokens(raw,[50258,50268,50360],check,'segment binds actual full window generation')
        matching=[window for window in windows if window['raw_generated_token_ids']==raw]
        check.require(bool(matching),'segment raw generation matches observed native seek window')
        check.require(type(idxs) is list and len(idxs)==2 and all(type(x) is int for x in idxs)
                      and 3<=idxs[0]<idxs[1]<=len(raw) and raw[idxs[0]:idxs[1]]==part,'segment is exact recorded native raw-generation slice')
        begin,end=segment['start'],segment['end']
        check.require(finite(begin) and finite(end) and 0<=begin<=end and begin>=previous,'finite monotone native segment timestamps')
        check.require(any(w['seek_start_frame']/100-1e-6<=begin<=end<=w['seek_start_frame']/100+30+1e-6 for w in matching),'timestamps lie within their actual native decoding window')
        previous=begin;all_tokens.extend(part)
    check.equal(all_tokens,tokens,'aggregate is exactly native returned segment concatenation')
    check.require(all(value not in tokens for value in [50258,50268,50360,50257]),'aggregate contains no fabricated prompt/EOS')


def audit_strong_extras(summary,comparison,manifest,check):
    import transformers.models.whisper.generation_whisper as generation
    import transformers.models.whisper.feature_extraction_whisper as frontend
    expected={'original_test_script_sha256':ORIGINAL_STRONG_TEST_SHA256,
              'whisper_generation_source_sha256':file_sha(Path(generation.__file__)),
              'whisper_frontend_source_sha256':file_sha(Path(frontend.__file__)),
              'official_longform_documentation':'https://huggingface.co/docs/transformers/model_doc/whisper'}
    check.equal(file_sha(OUTPUT/'benchmark_strong_tr_test.py'),ORIGINAL_STRONG_TEST_SHA256,'original failed short-form worker untouched')
    check.equal(summary['protected_code'],expected,'actual pinned native generation/frontend source bytes')
    check.equal(comparison['provenance']['protected_code'],expected,'comparison binds same native implementation')
    check.require(summary['all_longform_inputs_fully_consumed'] is True,'all full-audio long-form tails consumed')
    import soundfile as sf
    routes={'short_form':[],'long_form':[]}
    for source in manifest:
        info=sf.info(source['audio_filepath']);mode='short_form' if info.frames<=480000 else 'long_form'
        routes[mode].append({'source_id':source['source_id'],'pcm_samples':info.frames,
                             'duration':info.frames/info.samplerate,'wav_sha256':source['wav_sha256']})
    check.equal(summary['input_audit']['duration_routes'],routes,'all100 input duration routes independently reconstructed from WAVs')
    for mode in routes:
        check.equal(summary[mode+'_count'],len(routes[mode]),'actual summary '+mode+' count')
        check.equal(summary['input_audit'][mode+'_count'],len(routes[mode]),'actual input audit '+mode+' count')
    check.equal((len(routes['short_form']),len(routes['long_form'])),(99,1),'unchanged100TEST includes the36.84s recording')
    freeze=summary['frozen_selection_before_inference']
    before=stamp(freeze['frozen_at_utc'],check,'DEV selection freeze')
    inferred=stamp(summary['inference_started_at_utc'],check,'strong TEST inference start')
    created=stamp(summary['created_at_utc'],check,'strong TEST completion')
    check.require(before<=inferred<=created,'full TEST inference starts after frozen DEV winner')


def audit_runtime_configuration(state,frozen,config,profile,check):
    check.equal(state['selected_configuration'],profile['selected_configuration'],'measured B1 default retained independently of selected experimental configuration')
    check.equal(configuration(config),frozen['selected_configuration'],'selected checkpoint uses candidate-specific batch configuration')
    numerical_evidence = audit_batch_numerical_evidence(check)
    requested={**profile['selected_configuration'],'batch_size':8,'accumulation':2}
    check.equal(state['priority_quality_configuration'],requested,'requested B8 runtime uses audited chunk/checkpoint/loss configuration')
    check.equal(state['effective_batch_change'],{'prior_B':4,'quality_trial':16,'new_optimizer':True,
                'learning_rates_scaled':False,'warmup_clip_exposure_prior':400,'warmup_clip_exposure_quality':400},
                'effective optimizer batch change, unscaled LR, fresh Adam and400-clip warmup are explicit')
    check.equal(state['development_cadence'],{'initial_epoch_thresholds':[.25,.5,1.0],'after_epoch1_interval':.25,
                'bounded_final_stop_evaluation':True},'actual frozen trainer DEV cadence, including no.75 first-epoch evaluation')
    runtime_path=OUTPUT/'b8_runtime_cpu_review.json';runtime=json_load(runtime_path)
    check.require(runtime['passed'] is True and runtime['checks_passed']==26 and runtime['gpu_initialized'] is False,
                  'independent source/default B8 runtime review completed without GPU')
    for key,target in [('speed_report_sha256',OUTPUT/'large_batch_speed_report.json'),
                       ('matrix_sha256',OUTPUT/'large_batch_matrix.json'),
                       ('independent_matrix_audit_sha256',OUTPUT/'large_batch_results_verification.json')]:
        check.equal(runtime[key],file_sha(target),'runtime review binds exact completed benchmark '+key)
    for name,digest in runtime['source_sha256'].items():
        check.require(Path(name).name==name,'safe actual runtime source filename')
        check.equal(file_sha(OUTPUT/name),digest,'runtime backend-default source remains unchanged '+name)
    documented=state['recovery']['backend_flags_documented']
    check.equal(documented['trainer_fresh_defaults'],{'cudnn_allow_tf32':True,'cuda_matmul_allow_tf32':False},'declared unmodified training backend defaults')
    check.equal(documented['B8_profile'],{'cudnn_allow_tf32':False,'cuda_matmul_allow_tf32':False},'declared benchmark backend flags')
    check.require(runtime['fresh_training_environment_backend_defaults']['tf32_cudnn_allowed'] is True
                  and runtime['fresh_training_environment_backend_defaults']['tf32_matmul_allowed'] is False,
                  'actual fresh CPU defaults corroborate declared backend difference; not a GPU runtime measurement')
    for row in runtime['profiles']:
        check.equal(file_sha(local_path(row['path'],check,'audited B8 timing')),row['sha256'],'actual B8 typical/long timing evidence unchanged')
        check.require(row['runtime']['batch_size']==8 and row['runtime']['accumulation']==2
                      and row['runtime']['effective_batch']==16 and row['strict_same_shape_bf16_gate_passed'] is False
                      and row['adoption_eligible'] is False,'historical B8 throughput cannot become a numerical-adoption claim')
    return {'FP32_forced_MATH':'passed matched numerical oracle',
            'BF16_same_shape_strict':'failed for B1 and larger physical batches; recorded strict thresholds remain unchanged',
            'cross_shape_BF16':'not numerically equivalent; B8/effective16 is a different optimizer-batch quality run',
            'adoption_scope':'Best actual qualifying DEV checkpoint and all five retention gates; bounded early stops are valid. TEST never selects configuration.',
            'winner_configuration':frozen['selected_configuration'],
            'source_evidence':numerical_evidence,'B8_runtime_review_path':str(runtime_path),'B8_runtime_review_sha256':file_sha(runtime_path),
            'B8_historical_timings':runtime['profiles'],'backend_flags_documented':documented,
            'timing_transfer_claim':False,'scope_limits':runtime['scope_limits']}


def audit_batch_numerical_evidence(check):
    """Bind classification to the completed immutable CPU audit, without tensors."""
    audit_path=OUTPUT/'large_batch_results_verification.json'
    speed_path=OUTPUT/'large_batch_speed_report.json'
    matrix_path=OUTPUT/'large_batch_matrix.json'
    evidence=json_load(audit_path);speed=json_load(speed_path);matrix=json_load(matrix_path)
    check.require(evidence['passed'] is True and evidence['checks_count']==108939,
                  'completed independent physical-batch integrity audit')
    check.equal(evidence['auditor_sha256'],LARGE_BATCH_AUDITOR_SHA256,'reviewed physical-batch auditor identity')
    check.equal(file_sha(OUTPUT/'verify_large_batch_results.py'),LARGE_BATCH_AUDITOR_SHA256,'physical-batch auditor remains frozen')
    check.equal(evidence['matrix_sha256_at_audit'],file_sha(matrix_path),'actual audited physical-batch matrix remains unchanged')
    check.equal(Path(speed['verification_path']).resolve(),audit_path.resolve(),'speed report refers to exact numerical/integrity audit')
    check.equal(speed['verification_sha256'],file_sha(audit_path),'speed report exact independent verification bytes')
    check.require(speed['integrity_passed'] is True and evidence['adoption_eligible'] is False
                  and speed['adoption_eligible'] is False,'integrity success never becomes numerical adoption')
    indexed={row['case_id']:row for row in evidence['profiles']}
    check.require(len(indexed)==len(evidence['profiles'])==30,'all30 unique numerical/reference/timing cases audited')
    check.equal(set(indexed),{row['case_id'] for row in matrix['cases']},'numerical classification same complete matrix cases')
    fp32=[];bf16=[]
    for row in matrix['cases']:
        path=local_path(row['report_path'],check,'frozen physical-batch report')
        check.equal(file_sha(path),row['report_sha256'],'physical-batch report remains byte-identical: '+row['case_id'])
        report=json_load(path);summary=indexed[row['case_id']]
        if summary['mode']=='fp32-parity':
            gate=summary['numerical_comparison']['strict_gate']
            check.equal(report['parity']['passed'],gate['passed'],'reported FP32 classification agrees with independent tensor audit')
            check.equal(report['parity']['strict_gate']['passed'],gate['passed'],'reported original FP32 gate preserved')
            check.require(summary['precision']=='fp32' and summary['backend']=='math'
                          and gate['passed'] is True and gate['same_shape'] is False,
                          'strict forced-MATH FP32 oracle passed')
            check.equal(gate['limits'],{'gradient_relative_l2':.002,'gradient_cosine_min':.99999,
                        'feature_relative_l2':.0001,'component_absolute_delta':.0001,
                        'weighted_objective_absolute_delta':.0001},'original FP32 thresholds unchanged')
            fp32.append((summary['family'],summary['batch_size']))
        elif summary['mode']=='timing':
            gate=summary['numerical_comparison']['strict_gate']
            check.equal(report['parity']['passed'],gate['passed'],'reported BF16 classification agrees with independent tensor audit')
            check.equal(report['parity']['strict_gate']['passed'],gate['passed'],'reported original BF16 failure preserved')
            check.require(summary['precision']=='bf16' and gate['passed'] is False and gate['same_shape'] is True,
                          'strict same-shape BF16 failure retained, including B1')
            check.equal(gate['limits'],{'gradient_relative_l2':.0005,'gradient_cosine_min':.9999995,
                        'feature_relative_l2':.0001,'component_absolute_delta':.0001,
                        'weighted_objective_absolute_delta':None},'original BF16 thresholds unchanged')
            check.require(summary['status']=='measured_with_numerical_gate_failure' and summary['adoption_eligible'] is False,
                          'exploratory timing never proves numerical equivalence')
            bf16.append((summary['family'],summary['batch_size']))
    check.equal(set(fp32),{(family,batch) for family in ('typical','long') for batch in (2,4,8,16)},'all8 forced-MATH FP32 batch cases')
    check.equal(set(bf16),{(family,batch) for family in ('typical','long') for batch in (1,2,4,8,16)},'all10 same-shape BF16 strict failures')
    return {'verification_path':str(audit_path),'verification_sha256':file_sha(audit_path),
            'matrix_sha256':file_sha(matrix_path),'speed_report_sha256':file_sha(speed_path),
            'fp32_passed_cases':len(fp32),'bf16_strict_failed_cases':len(bf16),
            'scope':'Previously completed full-tensor audit; this recovery audit rechecks its hashes and recorded gates without repeating GPU inference.'}


# Independent strong TEST and core audit functions are appended below. They
# reuse only the previously verified read-only scoring/export/selection helpers.


def strong_test(path,frozen_path,frozen,saved,strong_dev,check):
    if path is None:return {'available':False,'reason':'No completed frozen-winner strong TEST comparison was supplied.'}
    comparison=json_load(local_path(path,check,'strong heldout comparison'))
    check.require(comparison['passed'] is True and comparison['status']=='complete','completed verified strong heldout comparison')
    check.equal(comparison['run_id'],frozen['run_id'],'strong TEST comparison frozen winner run')
    check.equal(comparison['selected_step'],frozen['selected_step'],'strong TEST comparison frozen winner step')
    check.equal(comparison['frozen_selection_sha256'],file_sha(frozen_path),'strong TEST comparison exact immutable frozen selection SHA')
    folder=OUTPUT/'results/strong_whisper_large_v3_greedy_test_v2'
    summary=json_load(folder/'summary.json');predictions=rows_load(folder/'test_tr.jsonl')
    check.require(summary['status']=='complete' and summary['split']=='test' and summary['language']=='tr','complete strong Turkish TEST run')
    check.equal((summary['repository'],summary['revision']),('openai/whisper-large-v3',STRONG_REVISION),'strong TEST pinned official model')
    from benchmark_strong_tr_test_v2 import PROTOCOL as expected_protocol
    check.equal(summary['protocol'],expected_protocol,'exact versioned full-audio TEST protocol')
    check.equal(summary['protocol']['short_form'],strong_dev['protocol'],'short TEST retains original verified DEV decoder protocol')
    check.equal(summary['protocol']['long_form'],{**strong_dev['protocol'],'return_timestamps':True,'return_segments':True},'long TEST changes only timestamp/segment routing')
    check.equal(summary['model_files_sha256'],strong_dev['model_files_sha256'],'strong DEV/TEST identical pinned model assets')
    check.equal((comparison['baseline_repository'],comparison['baseline_revision']),('openai/whisper-large-v3',STRONG_REVISION),'strong comparison model repository/revision')
    check.equal(comparison['baseline_protocol'],summary['protocol'],'strong comparison same fixed decoder protocol')
    check.equal((comparison['selection_split'],comparison['heldout_split'],comparison['language'],comparison['utterances']),
                ('FLEURS validation only','test','tr',100),'strong comparison DEV-only selection and fixed100TR TEST scope')
    check.equal(summary['metrics_code_sha256'],file_sha(OUTPUT/'metrics.py'),'strong TEST shared scorer bytes')
    check.equal(summary['predictions_sha256'],file_sha(folder/'test_tr.jsonl'),'strong TEST prediction bytes')
    check.equal(summary['script_sha256'],file_sha(OUTPUT/STRONG_V2_FILENAME),'strong TEST worker source bytes')
    provenance=comparison['provenance']
    expected_paths={'strong_summary':folder/'summary.json','strong_predictions':folder/'test_tr.jsonl',
                    'selected_summary':Path(json_load(OUTPUT/'runs'/frozen['run_id']/'adapted_results_path.json')['summary']),
                    'selected_predictions':Path(json_load(OUTPUT/'runs'/frozen['run_id']/'adapted_results_path.json')['summary']).parent/'test_tr.jsonl'}
    for key,target in expected_paths.items():
        check.equal(Path(provenance[key+'_path']).resolve(),target.resolve(),'strong comparison exact '+key+' path')
        check.equal(provenance[key+'_sha256'],file_sha(target),'strong comparison exact '+key+' bytes')
    check.equal(provenance['test_script_sha256'],file_sha(OUTPUT/STRONG_V2_FILENAME),'strong comparison worker SHA')
    check.equal(provenance['metrics_code_sha256'],file_sha(OUTPUT/'metrics.py'),'strong comparison scorer SHA')
    check.equal(provenance['verified_dev_summary_sha256'],frozen['baseline_provenance']['summary_sha256'],'strong TEST comparison same verified DEV baseline')
    check.equal(provenance['verified_dev_script_sha256'],strong_dev['script_sha256'],'strong comparison verified DEV source SHA')
    input_audit=summary['input_audit']
    check.equal(input_audit,provenance['input_audit'],'strong comparison exact input audit')
    for key,target in (('manifest',DATA/'test_tr.jsonl'),('paired_baseline_test',OUTPUT/'results/base_v3/test_tr.jsonl')):
        check.equal(Path(input_audit[key+'_path']).resolve(),target.resolve(),'strong TEST exact '+key+' path')
        check.equal(input_audit[key+'_sha256'],file_sha(target),'strong TEST exact '+key+' bytes')
    freeze=comparison['frozen_selection']
    check.equal(freeze['sha256'],file_sha(frozen_path),'strong comparison frozen selection digest')
    check.equal(Path(freeze['path']).resolve(),Path(frozen_path).resolve(),'strong comparison frozen selection path')
    check.equal((freeze['run_id'],freeze['selected_step']),(frozen['run_id'],frozen['selected_step']),'strong comparison frozen run/step')
    for key,target in (('selected_results_marker',OUTPUT/'runs'/frozen['run_id']/'adapted_results_path.json'),
                       ('selected_final_selection',OUTPUT/'runs'/frozen['run_id']/'selection.json'),
                       ('selected_finalization_provenance',OUTPUT/'runs'/frozen['run_id']/'finalization_provenance.json')):
        check.equal(freeze[key+'_sha256'],file_sha(target),'strong comparison exact '+key+' bytes')
    lock_path=ROOT/'work/strong_baseline_model.lock.json';lock=json_load(lock_path)
    check.equal((lock['repository'],lock['revision']),('openai/whisper-large-v3',STRONG_REVISION),'strong pinned model lock identity')
    check.equal(file_sha(lock_path),summary['model_lock_sha256'],'strong TEST actual model lock bytes')
    check.equal(summary['model_lock_sha256'],provenance['model_lock_sha256'],'strong comparison model lock SHA')
    check.equal(summary['model_files_sha256'],provenance['model_files_sha256'],'strong comparison complete pinned asset hashes')
    model_path=local_path(lock['local_path'],check,'strong pinned model folder')
    for name,expected in summary['model_files_sha256'].items():
        check.require(Path(name).name==name,'strong safe asset filename')
        check.equal(file_sha(model_path/name),expected,'strong actual pinned model asset: '+name)
    from transformers import AutoTokenizer
    processor=AutoTokenizer.from_pretrained(str(model_path),local_files_only=True)
    check.require(summary['all_full_prefixes_verified'] and summary['all_terminal_eos_verified'],'strong TEST full prefix/EOS summary checks')
    before=summary['frozen_selection_before_inference']
    check.equal(file_sha(local_path(before['path'],check,'strong baseline original selection freeze')),before['sha256'],'strong baseline original pre-inference freeze bytes')
    before_stamp=datetime.fromisoformat(before['frozen_at_utc']);created=datetime.fromisoformat(summary['created_at_utc'])
    check.require(before_stamp.tzinfo is not None and created.tzinfo is not None and before_stamp<=created,
                  'strong baseline inference follows its recorded timezone-aware selection freeze')
    adapted=saved['tr']['test'][1];manifest=rows_load(DATA/'test_tr.jsonl')
    audit_strong_extras(summary,comparison,manifest,check)
    check.equal(len(predictions),100,'strong TEST exactly100 fixed recordings')
    check.equal([r['source_id'] for r in predictions],[r['source_id'] for r in manifest],'strong TEST exact source IDs/order')
    for row,source in zip(predictions,manifest):
        check.require(all(row[key]==source['text' if key=='reference' else key] for key in
                          ('reference','sentence_id','dataset_revision','audio_sha256','wav_sha256','duration','language')),'strong TEST exact human reference/audio provenance')
        check.equal(row['split'],'test','strong TEST actual split')
        verify_whisper_prediction(row,source,processor,check)
    metrics=score_transcriptions(predictions,language='tr');metrics.pop('details')
    candidate=score_transcriptions(adapted,language='tr');candidate.pop('details')
    check.equal(metrics,summary['metrics'],'strong TEST exact corpus WER/CER counters')
    check.equal(metrics,comparison['baseline_asr'],'comparison strong TEST exact metrics')
    check.equal(candidate,comparison['adapted_asr'],'comparison candidate TEST exact metrics')
    paired=paired_bootstrap_delta([row['reference'] for row in predictions],[row['hypothesis'] for row in predictions],
                                  [row['hypothesis'] for row in adapted],language='tr',groups=[row['sentence_id'] for row in predictions])
    check.equal(paired,comparison['paired_sentence_bootstrap'],'strong TEST exact paired sentence-cluster bootstrap')
    near(check,comparison['wer_delta_percentage_points'],paired['wer']['delta_percentage_points'],'strong TEST WER gap arithmetic')
    near(check,comparison['cer_delta_percentage_points'],paired['cer']['delta_percentage_points'],'strong TEST CER gap arithmetic')
    return {'available':True,'comparison_path':str(Path(path).resolve()),'comparison_sha256':file_sha(path),
            'baseline_asr':metrics,'candidate_asr':candidate,'paired_sentence_bootstrap':paired,
            'completed_at_utc':summary['created_at_utc'],
            'candidate_minus_strong_wer_gap_percentage_points':paired['wer']['delta_percentage_points'],
            'small_test_within_two_percentage_points':candidate['wer']<=metrics['wer']+.02,
            'scope':'100 fixed Turkish TEST recordings after DEV recovery selection is frozen; 99 retain original short-form protocol and the36.84s recording uses native full-audio timestamp long-form; model/protocol sizes differ; small-set descriptive comparison.'}

def audit(args,check):
    frozen_path=local_path(args.frozen_selection,check,'frozen selection')
    loop_dir=frozen_path.parent;state_path=loop_dir/'loop_status.json';state=json_load(state_path)
    loop_id=safe_run(state['loop_id'],check,'adaptive loop')
    check.equal(loop_dir.name,loop_id,'frozen selection is loop-specific')
    baseline=local_path(args.baseline_results,check,'original multilingual baseline')
    check.equal(baseline,(OUTPUT/'results/base_v3').resolve(),'same unchanged original-v3 benchmark')
    base_summary=json_load(baseline/'summary.json')
    frozen,run_dir,config,training,history,history_summary=check_frozen_selection(frozen_path,state,base_summary,check)
    recovery_evidence=audit_recovery(state,frozen,config,check)
    strong_dev=strong_development(frozen,check)
    selection,pointer,provenance,checkpoint,adapted=audit_export(run_dir,config,frozen,check)
    languages,gate,english,saved,metadata=audit_predictions(run_dir,selection,provenance,checkpoint,adapted,baseline,check)
    check.equal(state['final_selection'],selection,'controller actual finalization selection matches durable winner')
    check.equal(state['merged_turkish_development_wer'],gate['turkish_wer'],'controller merged DEV score')
    competitive=gate['passed'] and gate['turkish_wer']<=frozen['competitive_threshold_wer']
    check.equal(state['competitive_on_development'],competitive,'controller competitive flag uses reloaded merged DEV gates')
    rates=[]
    for trial in state['trials']:
        path=OUTPUT/'runs'/trial['run_id']/'training_history.json'
        if path.exists():rates.append(audit_training_rate(trial['run_id'],check))
    measured=[r for r in rates if r.get('available',True) and r['training_wall_seconds']>0]
    total_audio=sum(r['audio_seconds'] for r in measured);total_wall=sum(r['training_wall_seconds'] for r in measured)
    actual={'trial_rates':rates,'audio_seconds':total_audio,'training_wall_seconds':total_wall,
            'audio_min_per_wall_min':total_audio/total_wall if total_wall else None,
            'scope':'Sum actual logged sampled-audio seconds / measured sampling/data/cache/STFT/H2D/forward/backward/Adam seconds across adaptive trials; excludes DEV/checkpoint/history serialization/startup.'}
    b8_ids={trial['run_id'] for trial in state['trials'] if trial.get('name')=='b8_quality'}
    b8_rates=[rate for rate in measured if rate['run_id'] in b8_ids]
    b8_audio=sum(rate['audio_seconds'] for rate in b8_rates);b8_wall=sum(rate['training_wall_seconds'] for rate in b8_rates)
    actual_b8={'trial_rates':b8_rates,'audio_seconds':b8_audio,'training_wall_seconds':b8_wall,
               'audio_min_per_wall_min':b8_audio/b8_wall if b8_wall else None,
               'scope':'Actual new B8/effective16 intervals only; excludes carried B/C, DEV, startup and checkpoint/history serialization. Historical profile rates are not transferred to this run.'}
    profile=json_load(OUTPUT/'batching_speed_comparison.json');profile_audit=json_load(OUTPUT/'batching_selection_independent_audit.json')
    check.require(profile_audit['passed'] is True,'independently audited selected quartet throughput')
    check.equal(profile_audit['selection_audit']['sha256'],file_sha(OUTPUT/'batching_speed_comparison.json'),'quartet throughput report remains independently verified')
    check.equal(state['profile_sha256'],file_sha(OUTPUT/'batching_speed_comparison.json'),'controller selected exact measured speed report')
    numerical=audit_runtime_configuration(state,frozen,config,profile,check)
    strong=strong_test(args.strong_test_comparison,frozen_path,frozen,saved,strong_dev,check)
    if strong['available']:
        check.require(stamp(strong['completed_at_utc'],check,'strong TEST finish')
                      <=stamp(state['deadline_at_utc'],check,'original absolute deadline'),
                      'full strong TEST inference fits original absolute recovery budget')
    if args.strong_test_comparison is not None:
        check.equal(state['strong_test_comparison_sha256'],file_sha(args.strong_test_comparison),'controller strong TEST comparison bytes')
    return {'recovery':recovery_evidence,'batch_numerical_classification':numerical,'loop_id':loop_id,'run_id':frozen['run_id'],'selected_step':frozen['selected_step'],
            'frozen_selection_sha256':file_sha(frozen_path),'frozen_selection_path':str(frozen_path),
            'controller_state_sha256_at_audit':file_sha(state_path),'controller_state_snapshot_status':state['status'],
            'winner_history':history_summary,'checkpoint':str(checkpoint),'checkpoint_state_sha256':metadata['model_fingerprint']['identity']['model_state_sha256'],
            'candidate_export_files_sha256':provenance['candidate_files_sha256'],'adapted_results_path':str(adapted),
            'languages':languages,'merged_development_gate':gate,'competitive_on_development':competitive,
            'strong_development_wer':strong_dev['metrics']['wer'],'competitive_development_threshold_wer':frozen['competitive_threshold_wer'],
            'strong_test_comparison':strong,'english_confidence':english,'actual_adaptive_training_rate':actual,'actual_B8_training_rate':actual_b8,
            'matched_quartet_speed':{'before_audio_min_per_wall_min':profile['before_audio_min_per_wall_min'],
                                    'after_audio_min_per_wall_min':profile['after_audio_min_per_wall_min'],
                                    'throughput_ratio':profile['throughput_ratio'],'scope':profile['timing_scope'],
                                    'independent_audit_sha256':file_sha(OUTPUT/'batching_selection_independent_audit.json')},
            'heldout_old_language_gates_passed':all(languages[lang]['heldout_quality_gate_passed'] for lang in LANGUAGES[:-1]),
            'turkish_heldout_improved_over_original_v3':languages['tr']['heldout_quality_gate_passed'],
            'quality_scope':'Small fixed FLEURS pilot:50 DEV +100 TEST recordings per language. Prior B TEST existed before infrastructure recovery; recovery changes and candidate selection use DEV only, with final TEST descriptive after selection freeze. Multiple trials reuse DEV. WER improvement alone does not establish useful Turkish ASR.'}


def text_report(result):
    report=legacy.text_report(result)
    if not result['passed']:return report
    report=report.replace('B1/accumulation4 uses the faster exact loss; unresolved B2 was rejected.',
        'The earlier matched quartet used B1/accumulation4; this is operational speed evidence, not strict BF16 numerical equivalence.')
    cfg=result['recovery']['winner_configuration']
    report+='\nRecovery retained original absolute deadline '+result['recovery']['original_deadline_at_utc']+'.\n'
    report+=f"Selected training configuration: physical batch{cfg['batch_size']}, accumulation{cfg['accumulation']}, effective{cfg['batch_size']*cfg['accumulation']}.\n"
    report+='B8/accumulation2 changes the effective optimizer batch from4 to16, keeps current best learning rates/objectives/layout and uses a fresh Adam optimizer. Warmup25 matches100 prior effective4 warmup updates in400 clip presentations. B8 can qualify only through actual best DEV quality and all five retention gates; bounded early stops are valid. TEST does not select it.\n'
    report+='Historical FP32 forced-MATH numerical oracle passed. Strict same-shape BF16 equivalence failed for B1 and larger batches; thresholds were not waived. Frozen timing used cudnn TF32=False, while the unmodified training default is=True (matmul TF32=False in both). Numerical equivalence and timing transfer are not claimed.\n'
    for row in result['batch_numerical_classification']['B8_historical_timings']:
        report+=f"Historical fixed effective16 {row['family']} B8 throughput {row['audio_min_per_wall_min']:.3f} audio min/wall min; gain versus matched B1 {row['gain_vs_matched_effective16_B1']:.3f}x; allocated peak {row['peak_allocated_vram_gb']:.3f}GB.\n"
    b8=result['actual_B8_training_rate']
    if b8['audio_min_per_wall_min'] is not None:
        report+=f"Actual new B8 training intervals: {b8['audio_seconds']/60:.2f} audio minutes / {b8['training_wall_seconds']/60:.2f} measured minutes = {b8['audio_min_per_wall_min']:.3f} audio min/wall min; carried trials and DEV/checkpoint/startup are excluded.\n"
    else:report+='No completed measured B8 training intervals were available.\n'
    report+='Prior C stopped cooperatively at191 updates and saved DEV/checkpoint evidence. The prior controller exit1 was the intentional owned-marker transition to B8, rather than a new infrastructure diagnosis.\n'
    report+='The36.84s Turkish TEST recording is included in full through native timestamp long-form decoding; other99 clips retain the verified short-form DEV protocol.\n'
    if result['recovery']['reused_original_finalization']:
        report+='Original B merge/export and multilingual benchmark were reused with unchanged hashes; no new finalizer execution is claimed.\n'
    return report


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frozen-selection',type=Path,required=True)
    p.add_argument('--strong-test-comparison',type=Path,required=True)
    p.add_argument('--baseline-results',type=Path,default=OUTPUT/'results/base_v3')
    p.add_argument('--output-dir',type=Path)
    return p


def main(args=None):
    p=parser();args=args or p.parse_args();check=Checks()
    directory=(args.output_dir or args.frozen_selection.parent).resolve()
    if not directory.is_relative_to((OUTPUT/'runs').resolve()):p.error('output-dir must stay within experiment runs')
    if directory!=args.frozen_selection.resolve().parent:p.error('output-dir must be the supplied frozen recovery loop directory')
    output_state=json_load(directory/'loop_status.json')
    if not isinstance(output_state.get('recovery'),dict) or output_state['recovery'].get('version')!=2 or directory.name==output_state['recovery'].get('original_loop_id'):
        p.error('only a new recovery loop may receive audit outputs; original runs remain read-only')
    result={'schema':1,'passed':False,'status':'failed','gpu_used':False,'verified_at_utc':datetime.now(timezone.utc).isoformat(),
            'auditor_sha256':file_sha(Path(__file__)),'original_auditor_sha256':file_sha(OUTPUT/'verify_adaptive_results.py'),
            'protected_helper_sha256':file_sha(OUTPUT/'verify_final_results.py')}
    try:
        result.update(audit(args,check));result.update(passed=True,status='complete')
    except Exception as error:
        result.update(failed_check=str(error),error_type=type(error).__name__)
    result['checks']=check.items;result['passed_check_count']=len(check.items)
    report=text_report(result);result['human_report_sha256']=hashlib.sha256(report.encode('utf-8')).hexdigest()
    result['human_report_path']=str(directory/'adaptive_recovery_v2_final_report.txt')
    from atomic_io_retry_v1 import commit_writer
    commit_writer(directory/'adaptive_recovery_v2_final_report.txt',lambda path:path.write_text(report,encoding='utf-8'))
    atomic_write_json(directory/'adaptive_results_verification.json',result)
    commit_writer(OUTPUT/'adaptive_recovery_v2_final_report.txt',lambda path:path.write_text(report,encoding='utf-8'))
    atomic_write_json(OUTPUT/'recovery_results_verification_v2.json',result)
    print(json.dumps({key:result.get(key) for key in ('passed','status','run_id','loop_id','selected_step','frozen_selection_sha256','passed_check_count','failed_check')},indent=2))
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
