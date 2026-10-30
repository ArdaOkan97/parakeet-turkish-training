"""Compact independent CPU audit for a DEV-selected cache-managed continuation.

Reuses the frozen score/export/confidence audit, without copying its hundreds
of thousands of successful check labels into the controller's live state.
No inference, CUDA initialization, or original artifact writes occur here.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from atomic_io_retry_v1 import write_json
import verify_adaptive_results as quality
from verify_final_results import file_sha, json_load, AuditError

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parents[1]

class CompactChecks:
    def __init__(self):
        self.count = 0
        self.examples = []
    def require(self, condition, name):
        if not condition:
            raise AuditError(name)
        self.count += 1
        if len(self.examples) < 100:
            self.examples.append(name)
    def equal(self, actual, expected, name):
        self.require(actual == expected, name)

def local(value, check, label):
    path = Path(value).resolve()
    check.require(path.is_relative_to(ROOT.resolve()) and path.exists(), label+' workspace path')
    return path

def bound_file(entry, check, label):
    path = local(entry['path'], check, label)
    check.equal(file_sha(path), entry['sha256'], label+' bytes')
    return path

def validate_cache_evidence(path, check, *, completed, initial_step):
    evidence = json_load(local(path, check, 'cache evidence'))
    check.equal(evidence['status'], 'complete', 'cache-managed child finished cleanly')
    check.equal(evidence['active_process_attachment'], False, 'new process only')
    for before, after, name in (
        ('launcher_sha256','launcher_sha256_after','launch_with_cache_reclaim_v1.py'),
        ('cache_helper_sha256','cache_helper_sha256_after','cuda_cache_reclaim_v1.py'),
        ('frozen_io_launcher_sha256','frozen_io_launcher_sha256_after','launch_with_io_retry_v1.py'),
        ('frozen_io_helper_sha256','frozen_io_helper_sha256_after','atomic_io_retry_v1.py')):
        check.equal(evidence[before], evidence[after], name+' before/after')
        check.equal(evidence[after], file_sha(OUTPUT/name), name+' current source')
    cache = evidence['cache_reclamation']
    check.equal(cache['worker_sha256'], file_sha(OUTPUT/'train_lora_batched.py'), 'frozen trainer bytes')
    check.equal(cache['worker_sha256_after'], cache['worker_sha256'], 'trainer unchanged')
    check.equal((cache['caller_function'],cache['caller_line']), ('run',539), 'optimizer-end scope')
    check.equal(cache['threshold_reserved_bytes'], 6_000_000_000, 'bounded unused cache trigger')
    check.equal(cache['eligible_update_calls'], completed-initial_step, 'actual cache hook ran once for every new update')
    check.equal(cache['failed_reclamations'], 0, 'no failed cache releases')
    for key in ('changes_training_math','changes_backend_flags','changes_rng_or_sampler','changes_optimizer_state'):
        check.equal(cache[key], False, key)
    check.require(cache['cleanup_seconds_in_training_timing'] >= 0, 'cleanup included in training timing')
    for event in cache['events']:
        check.equal(event['status'],'complete','recorded release complete')
        check.require(event['before']['reserved_bytes'] > 6_000_000_000, 'release only above trigger')
        check.equal(event['before']['allocated_bytes'],event['after']['allocated_bytes'], 'live tensor allocation unchanged by release')
        check.require(event['after']['reserved_bytes'] <= event['before']['reserved_bytes'], 'inactive reservation did not increase')
    return {key: cache[key] for key in ('eligible_update_calls','reclamations','failed_reclamations',
                                       'cleanup_seconds_in_training_timing','events_omitted')}

def validate_memory_evidence(entry, check):
    runtime = entry['cache_evidence']
    path = local(runtime['memory_guard_provenance_path'],check,'per-process memory guard')
    check.equal(file_sha(path),runtime['memory_guard_provenance_sha256'],'guard evidence immutable')
    evidence = json_load(path)
    check.equal((evidence['kind'],evidence['status']),
                ('memory_guard_training_launcher_v2','complete'),'guard completed')
    check.equal(evidence['cleanup_threshold_bytes'],6_000_000_000,'guard links actual6GB cleanup policy')
    check.equal((evidence['cap_bytes'],evidence['allocator_cap_applied']),
                (13_000_000_000,True),'actual13GB allocator cap applied')
    check.require(15e9 <= evidence['physical_cuda_device_bytes'] <= 18e9,'reviewed CUDA device capacity')
    check.equal(evidence['per_process_allocator_fraction'],
                evidence['cap_bytes']/evidence['physical_cuda_device_bytes'],'actual cap fraction')
    for key in ('global_driver_settings_changed','training_math_changed','backend_flags_changed','RNG_sampler_optimizer_changed'):
        check.equal(evidence[key],False,'guard scope '+key)
    for key,name in [('memory_launcher_sha256','launch_with_memory_guard_v2.py'),
                     ('cache_launcher_sha256','launch_with_cache_reclaim_v1.py'),
                     ('cache_helper_sha256','cuda_cache_reclaim_v1.py')]:
        check.equal(evidence[key],evidence[key+'_after'],'guard source before/after '+name)
        check.equal(evidence[key],file_sha(OUTPUT/name),'guard source current '+name)
    for key,trial_key in [('cache_provenance_path','cache_provenance_path'),('io_provenance_path','io_provenance_path')]:
        check.equal(Path(evidence[key]).resolve(),Path(entry[trial_key]).resolve(),'guard evidence linkage '+key)
    return {'path':str(path),'sha256':file_sha(path),'cap_bytes':evidence['cap_bytes']}

def validate_loop_provenance(state, directory, frozen, check):
    check.equal(state['kind'],'renewed_finite_B8_continuation_v1','new finite continuation schema')
    check.equal(directory.name,state['loop_id'],'loop directory identity')
    check.equal(state['selection_split'],'FLEURS validation only','loop DEV-only policy')
    started = datetime.fromisoformat(state['started_at_utc'])
    deadline = datetime.fromisoformat(state['deadline_at_utc'])
    check.require(started.tzinfo is not None and deadline.tzinfo is not None
                  and 0 < (deadline-started).total_seconds() <= 3*3600,'bounded renewed wall window')
    check.equal(state['runtime']['batch_size'],8,'physical B8')
    check.equal(state['runtime']['accumulation'],2,'accumulation2')
    check.equal(state['runtime']['effective_batch'],16,'effective16')
    check.equal(state['runtime']['threshold_reserved_gb'],6,'reserved-cache policy')
    policy = state['watchdog_policy']
    check.equal((policy['reserved_limit_gb'],policy['reserved_observations'],policy['commit_limit_fraction']),
                (12,3,.9),'reviewed memory watchdog limits')
    check.equal(policy['resource_stop_blocks_all_further_GPU'],True,'no blind resource-failure restart')
    check.equal(state['requires_resource_review'],False,'no unresolved memory or worker failure')
    preflight = state['preflight']
    check.equal(file_sha(__file__),preflight['final_audit_worker_sha256'],'reviewed final auditor source unchanged')
    for name, digest in preflight['source_sha256'].items():
        check.equal(file_sha(OUTPUT/name),digest,'continued runtime source '+name)
    policy_proof = preflight['cache_validation']
    proof_path = local(policy_proof['validation_path'],check,'GPU memory-policy validation')
    check.equal(file_sha(proof_path),policy_proof['validation_sha256'],'GPU proof manifest immutable')
    # The producer's status is insufficient: rerun the reviewed controller's
    # structural/source/RNG/actual-hook validator on the bound raw GPU records.
    from run_experiment_loop_continuation_v1 import validate_cache_gpu
    prior_config = json_load(OUTPUT/'runs'/(state['prior_loop_id']+'_b8_quality')/'run_config.json')
    check.equal(validate_cache_gpu(proof_path,prior_config),policy_proof,'GPU raw proof and preflight agree')
    receipt_path = local(preflight['prior_exit_receipt_path'],check,'parent exit receipt')
    check.equal(file_sha(receipt_path),preflight['prior_exit_receipt_sha256'],'owned exit receipt immutable')
    receipt = json_load(receipt_path)
    check.equal(receipt['root_observed_owned_session_completed'],True,'parent observed owned predecessor exit')
    check.equal((receipt['owned_exec_session_id'],receipt['root_observed_exit_code']),(13816,0),'predecessor owned-session exit')
    check.equal(file_sha(local(receipt['predecessor_status_path'],check,'prior status')),
                receipt['predecessor_status_sha256'],'prior loop untouched')
    check.equal(file_sha(local(receipt['predecessor_audit_path'],check,'prior audit')),
                receipt['predecessor_audit_sha256'],'prior verified results untouched')
    check.equal(state['prior_deadline_at_utc'],receipt['predecessor_deadline_unchanged'],'prior deadline unchanged')
    candidate = local(receipt['immutable_prior_candidate'],check,'prior ordinary export')
    for name, digest in receipt['prior_hf_files_sha256'].items():
        check.equal(file_sha(candidate/name),digest,'old ordinary HF export retained '+name)
    fork_path = local(state['fork_manifest_path'],check,'exact-state fork manifest')
    check.equal(file_sha(fork_path),state['fork_manifest_sha256'],'fork manifest immutable')
    fork = json_load(fork_path)
    check.require(fork['passed'] is True and fork['cuda_initialized'] is False,'full CPU fork verified')
    check.equal(fork['only_changed_path'],'run_id','fork changes namespace only')
    check.equal((fork['source_completed_step'],fork['inherited_selected_step']),(907,801),'exact inherited progress')
    for evidence in fork['inherited_development_reuse'].values():
        check.equal(evidence['evaluation_reused_without_new_inference'],True,'inherited DEV explicitly reused')
        for source, record in evidence['files'].items():
            for path in (source,record['snapshot_path']):
                check.equal(file_sha(local(path,check,'inherited DEV file')),
                            record['sha256'],'inherited DEV bytes')
    archive_path = local(state['initial_fork_archive_path'],check,'initial full-state archive')
    check.equal(file_sha(archive_path),state['initial_fork_archive_sha256'],'initial archive manifest immutable')
    archive = json_load(archive_path)
    check.equal(archive['fork_manifest_sha256'],file_sha(fork_path),'archive bound to original fork')
    for kind,record in archive['states'].items():
        check.equal(file_sha(local(record['archive_path'],check,'immutable '+kind+' state')),
                    fork['fork_checkpoints'][kind]['sha256'],'initial full '+kind+' fork bytes retained')
    prior_id = fork['source_run_id']
    prior_selection = json_load(OUTPUT/'runs'/prior_id/'training_selection.json')
    candidates = [(prior_id,prior_selection['turkish_validation_wer'],prior_selection['selected_step'])]
    cache_rates = []
    for trial in state['trials']:
        if trial.get('status') != 'complete' or trial.get('exit_code') != 0:
            continue
        trial_dir = OUTPUT/'runs'/trial['run_id']
        selection_path = trial_dir/'training_selection.json'
        if not selection_path.exists():
            selection_path = trial_dir/'selection.json'
        selected = json_load(selection_path)
        history = json_load(trial_dir/'training_history.json')
        configuration = json_load(trial_dir/'run_config.json')
        check.equal((configuration['batch_size'],configuration['accumulation']),(8,2),'every actual trial stays B8x2')
        for name,digest in configuration['code_sha256'].items():
            check.equal(file_sha(OUTPUT/name),digest,'actual trial unchanged source '+name)
        for row in history:
            check.require(all(math.isfinite(row[key]) for key in ('tdt_loss','replay_kl','replay_duration_kl',
                          'replay_feature_distillation','gradient_norm')),'actual trial finite objective/gradients')
        check.equal(selected['completed_steps'],len(history),'trial final durable completion')
        if selected.get('accepted') is True:
            quality.selected_history(trial['run_id'],selected,configuration,history,
                                     json_load(OUTPUT/'results/base_v3/summary.json'),check)
        else:
            check.equal(selected['heldout_tests_evaluated'],False,'rejected stage never used TEST')
            check.equal([r['step'] for r in history],list(range(1,len(history)+1)),'rejected stage durable contiguous history')
        initial = trial['inherited_completed_step']
        for row in history[initial:]:
            check.require(row['current_reserved_vram_gb'] <= 12,'new published update below memory stop limit')
        runtime = trial['cache_evidence']
        cache_path = local(runtime['cache_provenance_path'],check,'cache runtime')
        check.equal(file_sha(cache_path),runtime['cache_provenance_sha256'],'actual cache evidence bytes')
        io_path = local(runtime['io_provenance_path'],check,'atomic IO runtime')
        check.equal(file_sha(io_path),runtime['io_provenance_sha256'],'actual IO evidence bytes')
        io = json_load(io_path)
        check.equal((io['status'],io['training_semantics_changed']),('complete',False),'unchanged atomic IO scope')
        cache = validate_cache_evidence(cache_path,check,completed=len(history),initial_step=initial)
        cache['memory_guard'] = validate_memory_evidence(trial,check)
        cache['training_rate'] = quality.audit_training_rate(trial['run_id'],check)
        if initial and len(history)>initial:
            prefix = history[initial-1];tail=history[-1]
            check.equal(history[:initial],json_load(OUTPUT/'runs'/prior_id/'training_history.json')[:initial],
                        'inherited training prefix remains exact')
            cache['new_stage_audio_min_per_wall_min'] = ((tail['audio_seen_seconds']-prefix['audio_seen_seconds'])/
                 (tail['train_wall_seconds']-prefix['train_wall_seconds']))
        identity = trial['start_weights_provenance']
        check.equal(file_sha(local(identity['immutable_input_path'],check,'immutable stage input')),
                    identity['immutable_input_sha256'],'warm start input bytes immutable')
        check.equal(identity['immutable_input_sha256'],identity['original_source_sha256'],'start input byte exact')
        check.equal(configuration['warm_start']['sha256'],identity['immutable_input_sha256'],'actual worker loaded recorded stage input')
        check.equal(configuration['warm_start']['mode'],
                    'exact_resume' if initial else 'new_optimizer_warm_start','actual optimizer resume policy')
        cache_rates.append(cache)
        if selected.get('accepted') is True:
            candidates.append((trial['run_id'],selected['turkish_validation_wer'],selected['selected_step']))
    expected = min(candidates,key=lambda item:item[1])
    check.equal((frozen['run_id'],frozen['wer'],frozen['selected_step']),expected,'global best qualifying DEV trial wins')
    return {'prior_exit_receipt_sha256':file_sha(receipt_path),'fork_manifest_sha256':file_sha(fork_path),
            'cache_runtime_summaries':cache_rates,'renewed_window_hours':state['window_hours']}

def audit(args, check):
    directory = local(args.loop_dir, check, 'loop directory')
    state = json_load(directory/'loop_status.json')
    frozen_path = local(args.frozen_selection, check, 'frozen selection')
    check.equal(frozen_path.parent, directory, 'loop-specific freeze')
    frozen = json_load(frozen_path)
    check.equal(frozen['selection_split'], 'FLEURS validation only', 'DEV-only freeze')
    run_id = quality.safe_run(frozen['run_id'], check, 'winner')
    loop_evidence = validate_loop_provenance(state,directory,frozen,check)
    run_dir = OUTPUT/'runs'/run_id
    config = json_load(run_dir/'run_config.json')
    training = json_load(run_dir/'training_selection.json')
    history = json_load(run_dir/'training_history.json')
    baseline = OUTPUT/'results/base_v3'
    base = json_load(baseline/'summary.json')
    selected = quality.selected_history(run_id, training, config, history, base, check)
    dev = quality.selected_development_predictions(run_id, training, base, check)
    check.equal((frozen['selected_step'],frozen['wer']),
                (training['selected_step'],training['turkish_validation_wer']), 'frozen actual DEV choice')
    for key, path in [('selection_sha256',run_dir/'training_selection.json'),
                      ('run_config_sha256',run_dir/'run_config.json'),
                      ('history_sha256',run_dir/'training_history.json')]:
        check.equal(file_sha(path),frozen['provenance'][key], 'frozen '+key)
    check.equal(file_sha(local(frozen['weights'],check,'selected weights')),
                frozen['provenance']['weights_sha256'],'selected weight bytes')
    check.equal((config['batch_size'],config['accumulation'],config['effective_batch']), (8,2,16), 'physical/effective batch')
    for name, expected in config['code_sha256'].items():
        check.equal(file_sha(OUTPUT/name), expected, 'frozen training source '+name)
    for path, expected in config['input_sha256'].items():
        check.equal(file_sha(local(path,check,'training input')),expected,'unchanged input '+path)
    selection, pointer, provenance, checkpoint, adapted = quality.audit_export(run_dir,config,frozen,check)
    languages, gate, english, saved, metadata = quality.audit_predictions(
        run_dir,selection,provenance,checkpoint,adapted,baseline,check)
    strong_dev = quality.strong_development(frozen,check)
    comparison_path = local(args.strong_test_comparison,check,'reused strong TEST comparison')
    from compare_strong_test_reuse_v1 import audit_reused_comparison
    comparison = audit_reused_comparison(comparison_path,frozen_path,check)
    check.equal(comparison['run_id'],run_id,'strong comparison winner')
    check.equal(comparison['selected_step'],frozen['selected_step'],'strong comparison selected step')
    check.equal(comparison['frozen_selection_sha256'],file_sha(frozen_path),'strong comparison freeze')
    check.equal(gate['turkish_wer'] <= frozen['competitive_threshold_wer'],
                state['competitive_on_development'], 'competitive flag uses merged DEV only')
    return {'loop_id':state['loop_id'],'run_id':run_id,'selected_step':frozen['selected_step'],
            'continuation_provenance':loop_evidence,
            'winner_history':selected,'unmerged_selected_development':dev,
            'merged_development_gate':gate,'languages':languages,'english_confidence':english,
            'strong_development_wer':strong_dev['metrics']['wer'],
            'competitive_development_threshold_wer':frozen['competitive_threshold_wer'],
            'competitive_on_development':state['competitive_on_development'],
            'strong_test_comparison_path':str(comparison_path),
            'strong_test_comparison_sha256':file_sha(comparison_path),
            'checkpoint':str(checkpoint),'adapted_results_path':str(adapted),
            'frozen_selection_sha256':file_sha(frozen_path),
            'quality_scope':'Small fixed FLEURS pilot; 50 DEV and100 TEST per language; prior TEST is already seen and cannot drive selection.'}

def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--loop-dir',type=Path,required=True)
    result.add_argument('--frozen-selection',type=Path,required=True)
    result.add_argument('--strong-test-comparison',type=Path,required=True)
    result.add_argument('--output-dir',type=Path)
    return result

def main(args=None):
    args = args or parser().parse_args()
    directory = args.loop_dir.resolve()
    destination = (args.output_dir or directory).resolve()
    if (directory.parent != OUTPUT/'runs' or directory.name != 'adaptive_20261010_continuation_01'
            or destination != directory or args.frozen_selection.resolve().parent != directory):
        raise ValueError('Write audit only inside this new owned continuation directory')
    if (destination/'adaptive_results_verification.json').exists():
        raise FileExistsError('Existing final verification evidence is immutable')
    check = CompactChecks()
    result = {'schema':1,'kind':'continued_loop_results_verification_v1',
              'created_at_utc':datetime.now(timezone.utc).isoformat(),
              'audit_source_sha256':file_sha(__file__),'passed':False,'GPU_inference':False}
    try:
        result.update(audit(args,check),passed=True)
    except Exception as error:
        result.update(failed_check=str(error),error_type=type(error).__name__)
    result.update(passed_check_count=check.count,successful_check_examples=check.examples)
    destination.mkdir(parents=True,exist_ok=True)
    write_json(destination/'adaptive_results_verification.json',result)
    print(json.dumps({key:result.get(key) for key in ('passed','run_id','selected_step','passed_check_count','failed_check')}))
    return 0 if result['passed'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
