"""Independent CPU final audit of the resource-reviewed1773 continuation.

The predecessor has full-state/DEV evidence, not a new TEST/export. Existing
verified Whisper TEST is reused only after a new DEV-only freeze. The changed
system-pressure guard does not change the training objective or numerical gates.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path

import verify_continued_loop_results_v1 as previous
import fork_training_state_v2 as fork_worker
from atomic_io_retry_v1 import read_bytes_shared, write_json
from verify_final_results import AuditError, file_sha, json_load
import verify_adaptive_results as quality

OUTPUT=Path(__file__).resolve().parent
ROOT=OUTPUT.parents[1]
LOOP_ID='adaptive_20261010_continuation_02'
SOURCE_RUN='adaptive_20261010_continuation_01_resume_B8'
RECEIPT_SHA256='d2dc4fed6a4f0a105ac981ba1759a95656636f3e177ddaedac44ef55b8d278a3'
PARENT_AUDITOR_SHA256='e50669e5600eea70be8960cb8fd2075628f21b128a72fc093ba1b8e4525131e4'
FORK_HELPER_SHA256='ab57a1e3b1cf5dc01582858ea845901be02247b5650ce70cfce93bbfc14a5766'
RESOURCE_HELPER_SHA256='cc4ca1f11fdef19ac38d2aa878334451a100fd79f9a2e25a5afb4ceef89f0d7c'
local=previous.local


class CompactChecks(previous.CompactChecks):
    def __init__(self):
        super().__init__()
        self.tensor_count=0


def audit_predecessor(preflight,state,check):
    check.equal(file_sha(OUTPUT/'verify_continued_loop_results_v1.py'),PARENT_AUDITOR_SHA256,'Original audit unchanged')
    check.equal(file_sha(OUTPUT/'fork_training_state_v2.py'),FORK_HELPER_SHA256,'Completed exact fork helper unchanged')
    check.equal(file_sha(OUTPUT/'resource_watchdog_v1.py'),RESOURCE_HELPER_SHA256,'Reviewed resource helper unchanged')
    path=local(preflight['prior_exit_receipt_path'],check,'owned predecessor receipt')
    check.equal(file_sha(path),RECEIPT_SHA256,'Exact root-owned reviewed exit receipt')
    check.equal(preflight['prior_exit_receipt_sha256'],RECEIPT_SHA256,'Preflight receipt binding')
    receipt=json_load(path)
    old_loop=OUTPUT/'runs/adaptive_20261010_continuation_01';old_run=OUTPUT/'runs'/SOURCE_RUN
    fork_worker.validate_resource_receipt(receipt,old_loop,old_run,check)
    check.equal(state['prior_loop_id'],old_loop.name,'Resource-reviewed predecessor loop')
    check.equal(state['prior_deadline_at_utc'],receipt['predecessor_deadline_at_utc'],'Previous finite window untouched')
    check.equal(preflight['prior_resource_review_sha256'],receipt['resource_review_sha256'],'Resource review binding')
    review=json_load(local(receipt['resource_review_path'],check,'actual resource review'))
    check.equal((review['kind'],review['child_exit_code'],review['durable_step'],
        review['all_five_retention_gates_passed'],review['published_final_export_or_test_claim']),
        ('continuation_stopped_resource_review_observation_v1',0,1773,True,False),
        'Actual resource review scope/completion/DEV evidence')
    for source,entry in review['source_evidence'].items():
        check.equal(file_sha(local(source,check,'resource-reviewed original evidence')),entry['sha256'],
            'Resource-reviewed original source bytes')
    old_receipt=json_load(local(receipt['prior_audited_receipt_path'],check,'original completed-result receipt'))
    check.equal(file_sha(local(old_receipt['predecessor_audit_path'],check,'original completed quality audit')),
        old_receipt['predecessor_audit_sha256'],'Original completed quality audit retained')
    check.equal(file_sha(local(old_receipt['predecessor_status_path'],check,'original completed loop')),
        old_receipt['predecessor_status_sha256'],'Original completed loop retained')
    check.equal(old_receipt['predecessor_deadline_unchanged'],'2026-10-10T23:41:22.889692+00:00','Original absolute deadline untouched')
    check.equal(preflight['prior_audit_sha256'],old_receipt['predecessor_audit_sha256'],'Strong TEST reference uses audited original export')
    for name,digest in receipt['prior_hf_files_sha256'].items():
        check.equal(file_sha(Path(receipt['immutable_prior_candidate'])/name),digest,'Prior ordinary801 export retained '+name)
    from run_experiment_loop_continuation_v2 import cache_evidence
    runtime=receipt['runtime_provenance']
    actual=cache_evidence(Path(runtime['cache_provenance_path']['path']),
        Path(runtime['io_provenance_path']['path']),SOURCE_RUN,907,
        Path(runtime['memory_guard_provenance_path']['path']))
    check.equal(actual,preflight['prior_runtime_cache_evidence'],'Predecessor actual866 scoped cache updates bound')
    return receipt


def audit_full_fork(state,directory,receipt,check):
    import torch
    torch.set_num_threads(2)
    manifest_path=local(state['fork_manifest_path'],check,'fork manifest')
    check.equal(file_sha(manifest_path),state['fork_manifest_sha256'],'Immutable exact-fork manifest')
    manifest=json_load(manifest_path)
    check.equal((manifest['kind'],manifest['passed'],manifest['source_run_id'],manifest['target_run_id'],
        manifest['source_completed_step'],manifest['inherited_selected_step'],manifest['only_changed_path']),
        ('exact_training_state_run_id_fork_v2',True,SOURCE_RUN,LOOP_ID+'_resume_B8',1773,1773,'run_id'),
        'Exact1773 fork identity/namespace-only change')
    check.equal(manifest['source_review_kind'],'resource_stop_full_state_DEV_review','DEV/full-state review distinct from export/TEST')
    check.equal((manifest['fresh_TEST_evaluated'],manifest['fresh_export_published']),(False,False),'No false predecessor fresh TEST/export')
    check.equal(manifest['predecessor_exit_receipt_sha256'],RECEIPT_SHA256,'Fork reviewed-exit binding')
    check.equal(manifest['fork_helper_sha256'],FORK_HELPER_SHA256,'Actual completed helper identity')
    source_config=json_load(OUTPUT/'runs'/SOURCE_RUN/'run_config.json')
    check.equal(manifest['inherited_config_identity'],source_config['config_identity'],'Nested optimizer/data/model config unchanged')
    proof_path=directory/'fork_snapshot_independent_verification.json';proof=json_load(proof_path)
    check.require(proof['passed'] is True and proof['signed_zero_byte_identity_checked'] is True
        and proof['CUDA_initialized'] is False,'Serialized raw full-state proof passed without CUDA')
    check.equal(proof['fork_manifest_sha256'],file_sha(manifest_path),'Raw proof exact manifest')
    archive_path=local(state['initial_fork_archive_path'],check,'initial immutable archive manifest')
    check.equal(file_sha(archive_path),state['initial_fork_archive_sha256'],'Archive manifest immutable')
    archive=json_load(archive_path)
    check.equal(archive['fork_manifest_sha256'],file_sha(manifest_path),'Archive completed fork linkage')
    check.equal(archive['independent_fork_verification_sha256'],file_sha(proof_path),'Archive full-state proof linkage')
    for role,record in archive['states'].items():
        source_path=local(manifest['source_checkpoints'][role]['path'],check,'immutable parent '+role)
        archive_state=local(record['archive_path'],check,'immutable initial '+role+' state')
        check.require(archive_state.is_relative_to(directory/'fork_snapshot/initial_fork_states'),'Owned initial state archive')
        check.equal(file_sha(source_path),receipt['source_'+role+'_sha256'],'Parent '+role+' bytes preserved')
        check.equal(file_sha(archive_state),manifest['fork_checkpoints'][role]['sha256'],'Fork initial '+role+' bytes retained')
        source=torch.load(io.BytesIO(read_bytes_shared(source_path)),map_location='cpu',weights_only=True)
        copied=torch.load(io.BytesIO(read_bytes_shared(archive_state)),map_location='cpu',weights_only=True)
        check.equal((source['run_id'],copied['run_id'],copied['step'],copied['selected_step']),
            (SOURCE_RUN,LOOP_ID+'_resume_B8',1773,1773),'Actual serialized state identity')
        check.equal(list(source),list(copied),'Full-state ordered keys preserved')
        for key in source:
            if key!='run_id':fork_worker.raw_exact(source[key],copied[key],check,'archive.'+role+'.'+key)
        del source,copied
    for record in manifest['snapshot_files'].values():
        path=local(record['snapshot_path'],check,'preserved predecessor copy')
        check.require(path.is_relative_to(directory/'fork_snapshot'),'Snapshot write scope')
        check.equal(file_sha(path),record['sha256'],'Preserved source copy bytes')
    for record in manifest['inherited_development_reuse'].values():
        check.equal(record['evaluation_reused_without_new_inference'],True,'Historical DEV explicitly reused')
        for source,entry in record['files'].items():
            for path in (source,entry['snapshot_path']):
                check.equal(file_sha(local(path,check,'inherited DEV')),entry['sha256'],'Inherited DEV original/copy exact')
    check.require(not torch.cuda.is_initialized(),'Full fork audit initialized no CUDA')
    return manifest


def audit_watchdog_policy(policy,check):
    expected={'reserved_limit_gb':12,'reserved_observations':3,'commit_limit_fraction':.9,
        'commit_critical_fraction':.95,'commit_headroom_bytes':2_000_000_000,'commit_observations':3,
        'commit_min_interval_seconds':5,'commit_telemetry_max_age_seconds':15,
        'memory_observations_phase_tagged':True,'resource_stop_blocks_all_further_GPU':True}
    for key,value in expected.items():check.equal(policy[key],value,'Reviewed watchdog '+key)
    from resource_watchdog_v1 import POLICY
    check.equal(POLICY,{'moderate_commit_fraction':.90,'critical_commit_fraction':.95,
        'critical_headroom_bytes':2_000_000_000,'moderate_consecutive_observations':3,
        'minimum_fresh_interval_seconds':5.,'telemetry_stale_seconds':15.},'Exact resource helper policy')


def audit_watchdog_observations(trial,check):
    from resource_watchdog_v1 import POLICY
    rows=trial['watchdog_observations']
    check.require(rows,'Actual owned worker memory polls recorded')
    previous_monotonic=None
    for row in rows:
        check.equal(row['policy'],POLICY,'Actual pressure poll policy')
        worker=row['worker_private_memory'];system=row['system_memory']
        check.equal(worker['pid'],trial['pid'],'Private memory only owned child PID')
        check.require(system['available'] is True,'Actual global pressure telemetry available')
        check.equal(worker['observed_at_utc'],system['observed_at_utc'],'Paired owned/global sample timestamp')
        check.require(math.isfinite(row['observed_monotonic']),'Finite poll clock')
        if previous_monotonic is not None:check.require(row['observed_monotonic']>=previous_monotonic,'Poll monotonic order')
        previous_monotonic=row['observed_monotonic']
        fraction=system['commit_fraction'];headroom=system['commit_available_bytes'];limit=system['commit_limit_bytes']
        check.require(all(math.isfinite(x) for x in (fraction,headroom,limit)) and limit>0
            and 0<=fraction<=1 and headroom>=0,'Valid actual pressure bounds')
        check.require(abs(system['commit_used_bytes']+headroom-limit)<=4096
            and abs(system['commit_used_bytes']/limit-fraction)<=1e-8,'Actual pressure fields agree')
        if worker['available']:
            check.require(all(isinstance(worker[key],int) and worker[key]>=0 for key in
                ('private_bytes','working_set_bytes','peak_working_set_bytes')),'Owned process byte telemetry valid')
        else:check.require('error' in worker or 'reason' in worker,'Private read failure retained explicitly')
        check.equal(row['worker_phase']['inferred_from_bounded_log_tail'],True,'Worker phase inference labelled')
        check.require(row['moderate_commit_streak']<3 and fraction<.95 and headroom>=2_000_000_000,
            'Successful finalization has no unresolved actual critical/sustained-pressure poll')


def validate_loop_provenance(state,directory,frozen,check):
    check.equal((state['kind'],state['loop_id']),('renewed_finite_B8_continuation_v2',LOOP_ID),'Owned new continuation identity')
    check.equal(state['selection_split'],'FLEURS validation only','DEV-only adaptation')
    start=datetime.fromisoformat(state['started_at_utc']);deadline=datetime.fromisoformat(state['deadline_at_utc'])
    check.require(start.tzinfo is not None and deadline.tzinfo is not None
        and 0<(deadline-start).total_seconds()<=10800,'Bounded separately renewed window')
    check.equal((state['runtime']['batch_size'],state['runtime']['accumulation'],state['runtime']['effective_batch']),
        (8,2,16),'Physical/effective batch')
    check.equal((state['runtime']['threshold_reserved_gb'],state['runtime']['allocator_cap_gb']),(6,13),'Unchanged reviewed CUDA cache/cap')
    check.equal(state['runtime']['strict_BF16_numeric_equivalence_passed'],False,'Historical strict BF16 failure retained')
    audit_watchdog_policy(state['watchdog_policy'],check)
    check.equal(state['requires_resource_review'],False,'No unresolved resource stop')
    preflight=state['preflight']
    check.equal(file_sha(__file__),preflight['final_audit_worker_sha256'],'Frozen final audit source')
    for name,digest in preflight['source_sha256'].items():check.equal(file_sha(OUTPUT/name),digest,'Actual continued source '+name)
    receipt=audit_predecessor(preflight,state,check)
    manifest=audit_full_fork(state,directory,receipt,check)
    from run_experiment_loop_continuation_v2 import validate_cache_gpu
    proof=preflight['cache_validation'];path=local(proof['validation_path'],check,'Actual CUDA policy proof')
    check.equal(file_sha(path),proof['validation_sha256'],'Bound CUDA proof immutable')
    source_config=json_load(OUTPUT/'runs'/SOURCE_RUN/'run_config.json')
    check.equal(validate_cache_gpu(path,source_config),proof,'Raw causal/production proof and numerical gates recomputed')
    source_selection=json_load(OUTPUT/'runs'/SOURCE_RUN/'selection.json')
    candidates=[(LOOP_ID+'_resume_B8',source_selection['turkish_validation_wer'],1773)]
    rates=[];exact_seen=False
    baseline=json_load(OUTPUT/'results/base_v3/summary.json')
    for trial in state['trials']:
        check.require(trial['status']=='complete' and trial['exit_code']==0,'All recorded actual trials completed cleanly')
        run=trial['run_id'];run_dir=OUTPUT/'runs'/run
        selected=json_load(run_dir/('training_selection.json' if (run_dir/'training_selection.json').exists() else 'selection.json'))
        history=json_load(run_dir/'training_history.json');config=json_load(run_dir/'run_config.json')
        check.equal((config['batch_size'],config['accumulation']),(8,2),'Each trial keeps B8x2')
        for name,digest in config['code_sha256'].items():check.equal(file_sha(OUTPUT/name),digest,'Actual worker source '+name)
        for record in history:
            check.require(all(math.isfinite(record[key]) for key in ('tdt_loss','replay_kl','replay_duration_kl',
                'replay_feature_distillation','gradient_norm')),'Finite actual objective/gradient history')
        check.equal(selected['completed_steps'],len(history),'Actual durable completion')
        initial=trial['inherited_completed_step'];identity=trial['start_weights_provenance']
        check.equal(file_sha(local(identity['immutable_input_path'],check,'stage immutable input')),identity['immutable_input_sha256'],'Actual stage input retained')
        check.equal(identity['immutable_input_sha256'],identity['original_source_sha256'],'Stage input copy exact')
        check.equal(config['warm_start']['sha256'],identity['immutable_input_sha256'],'Frozen worker loaded recorded input')
        check.equal(config['warm_start']['mode'],'exact_resume' if initial else 'new_optimizer_warm_start','Optimizer resume/restart policy')
        if initial:
            check.equal((run,initial),(LOOP_ID+'_resume_B8',1773),'Single exact1773 continuation namespace')
            check.equal(history[:initial],json_load(OUTPUT/'runs'/SOURCE_RUN/'training_history.json'),'Inherited1773 history prefix exact')
            check.equal(identity['immutable_input_sha256'],manifest['fork_checkpoints']['latest']['sha256'],'Exact worker input is archived1773 fork')
            exact_seen=True
        if selected.get('accepted') is True:
            quality.selected_history(run,selected,config,history,baseline,check)
            candidates.append((run,selected['turkish_validation_wer'],selected['selected_step']))
        else:
            check.equal(selected['heldout_tests_evaluated'],False,'Rejected stage never selected on TEST')
            check.equal([x['step'] for x in history],list(range(1,len(history)+1)),'Rejected stage history contiguous')
        runtime=trial['cache_evidence']
        cache_path=local(runtime['cache_provenance_path'],check,'actual cache evidence')
        check.equal(file_sha(cache_path),runtime['cache_provenance_sha256'],'Actual cache proof bytes')
        io_path=local(runtime['io_provenance_path'],check,'actual IO evidence')
        check.equal(file_sha(io_path),runtime['io_provenance_sha256'],'Actual IO proof bytes')
        io=json_load(io_path);check.equal((io['status'],io['training_semantics_changed']),('complete',False),'IO scope complete/unchanged')
        cache=previous.validate_cache_evidence(cache_path,check,completed=len(history),initial_step=initial)
        cache['memory_guard']=previous.validate_memory_evidence(trial,check)
        audit_watchdog_observations(trial,check)
        cache['training_rate']=quality.audit_training_rate(run,check)
        if initial and len(history)>initial:
            base=history[initial-1];tail=history[-1]
            cache['new_stage_audio_min_per_wall_min']=(tail['audio_seen_seconds']-base['audio_seen_seconds'])/(tail['train_wall_seconds']-base['train_wall_seconds'])
        rates.append(cache)
    check.require(exact_seen,'New fork namespace has an actual completed exact-resume child')
    expected=min(candidates,key=lambda value:value[1])
    check.equal((frozen['run_id'],frozen['wer'],frozen['selected_step']),expected,'Global best qualifying DEV model with namespace-safe inherited fallback')
    return {'predecessor_exit_receipt_sha256':RECEIPT_SHA256,'fork_manifest_sha256':state['fork_manifest_sha256'],
        'cache_runtime_summaries':rates,'renewed_window_hours':state['window_hours'],'full_fork_tensor_leaves_rechecked':check.tensor_count}


def audit(args,check):
    directory=local(args.loop_dir,check,'new loop directory');state=json_load(directory/'loop_status.json')
    frozen_path=local(args.frozen_selection,check,'new DEV freeze');check.equal(frozen_path.parent,directory,'Owned loop-specific freeze')
    frozen=json_load(frozen_path);check.equal(frozen['selection_split'],'FLEURS validation only','Frozen DEV-only choice')
    run=quality.safe_run(frozen['run_id'],check,'winner')
    loop_proof=validate_loop_provenance(state,directory,frozen,check)
    run_dir=OUTPUT/'runs'/run;config=json_load(run_dir/'run_config.json');training=json_load(run_dir/'training_selection.json')
    history=json_load(run_dir/'training_history.json');baseline=OUTPUT/'results/base_v3';base=json_load(baseline/'summary.json')
    selected=quality.selected_history(run,training,config,history,base,check)
    dev=quality.selected_development_predictions(run,training,base,check)
    check.equal((frozen['selected_step'],frozen['wer']),(training['selected_step'],training['turkish_validation_wer']),'Frozen actual DEV choice')
    for key,path in [('selection_sha256',run_dir/'training_selection.json'),('run_config_sha256',run_dir/'run_config.json'),('history_sha256',run_dir/'training_history.json')]:
        check.equal(file_sha(path),frozen['provenance'][key],'Frozen '+key)
    check.equal(file_sha(local(frozen['weights'],check,'selected weight state')),frozen['provenance']['weights_sha256'],'Frozen weights')
    for path,digest in config['input_sha256'].items():check.equal(file_sha(local(path,check,'actual training input')),digest,'Immutable data identity')
    selection,pointer,provenance,checkpoint,adapted=quality.audit_export(run_dir,config,frozen,check)
    languages,gate,english,saved,metadata=quality.audit_predictions(run_dir,selection,provenance,checkpoint,adapted,baseline,check)
    strong_dev=quality.strong_development(frozen,check)
    from compare_strong_test_reuse_v1 import audit_reused_comparison
    comparison_path=local(args.strong_test_comparison,check,'reused measured strong TEST comparison')
    comparison=audit_reused_comparison(comparison_path,frozen_path,check)
    check.equal((comparison['run_id'],comparison['selected_step'],comparison['frozen_selection_sha256']),
        (run,frozen['selected_step'],file_sha(frozen_path)),'Strong comparison exact frozen winner')
    competitive=gate['turkish_wer']<=frozen['competitive_threshold_wer']
    check.equal(competitive,state['competitive_on_development'],'MergedDEV competitive flag')
    return {'loop_id':LOOP_ID,'run_id':run,'selected_step':frozen['selected_step'],'frozen_selection_sha256':file_sha(frozen_path),
        'continuation_provenance':loop_proof,'winner_history':selected,'unmerged_selected_development':dev,
        'merged_development_gate':gate,'languages':languages,'english_confidence':english,
        'competitive_on_development':competitive,'quality_objective_measurements_satisfied':competitive and gate['retention_gate_passed'],
        'strong_development_wer':strong_dev['metrics']['wer'],'competitive_development_threshold_wer':frozen['competitive_threshold_wer'],
        'strong_test_comparison_path':str(comparison_path),'strong_test_comparison_sha256':file_sha(comparison_path),
        'checkpoint':str(checkpoint),'adapted_results_path':str(adapted),
        'quality_scope':'Small fixed FLEURS pilot;50 DEV/100 TEST per language. Previously seen TEST cannot drive adaptation/selection; relative Turkish gains alone do not establish useful accuracy.'}


def parser():
    value=argparse.ArgumentParser(description=__doc__)
    for name in ('loop-dir','frozen-selection','strong-test-comparison'):value.add_argument('--'+name,type=Path,required=True)
    value.add_argument('--output-dir',type=Path)
    return value


def main(args=None):
    args=args or parser().parse_args();directory=args.loop_dir.resolve();destination=(args.output_dir or directory).resolve()
    if directory!=OUTPUT/'runs'/LOOP_ID or destination!=directory or args.frozen_selection.resolve().parent!=directory:
        raise ValueError('Write only inside the new owned continuation02 loop')
    path=destination/'adaptive_results_verification.json'
    if path.exists():raise FileExistsError('Existing final audit evidence is immutable')
    check=CompactChecks();result={'schema':1,'kind':'continued_loop_results_verification_v2',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'audit_source_sha256':file_sha(__file__),
        'passed':False,'GPU_inference':False}
    try:result.update(audit(args,check),passed=True)
    except Exception as error:result.update(failed_check=str(error),error_type=type(error).__name__)
    result.update(passed_check_count=check.count,successful_check_examples=check.examples)
    destination.mkdir(parents=True,exist_ok=True);write_json(path,result)
    print(json.dumps({key:result.get(key) for key in ('passed','run_id','selected_step','passed_check_count','failed_check')}))
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
