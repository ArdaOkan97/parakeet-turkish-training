"""INACTIVE same-ID shared-head frontier admission; no actual state read here.

Only a ROOT CLOSED latest improving over its startingchampion and passingallfive gates is admitted; the11.9394% final goal is not an entry gate. Earlier
BEST needs a separate explicit branch, rather than erasing closed history. Total
N extension is separate from immutable initial screen/scientific identity.
Torch is injected only by the future exclusive worker for utility/RNG restore.
"""
from __future__ import annotations
from datetime import datetime
import os
from pathlib import Path
import random
import sys
from tsc_bpe_shared_head_memory_v2 import (preflight as initial_preflight, HERE,
    PHASE, STATE_KIND, PROOF_KIND, PROOF_NAME, require, sha, canonical, scoped,
    read, record, checked_record)

RECEIPT_KIND='ROOT_owned_closed_TSC_BPE_shared_heads_exact_v1'
UTILITY_SHA='384d01abe9b7105761491b8acda5d3de4eb8c40ae679eee2701269079cf9967c'
INITIAL_WORKER='train_lora_batched_tsc_bpe_shared_head_memory_v2.py'
INITIAL_CONTROLLER='run_TSC_bpe_shared_head_memory_v2.py'
INITIAL_WORKER_SHA='ea74795363e0c09114e0a496ccbea55dab34172202d40c6a9aefb5f7cdfa8290'
INITIAL_CONTROLLER_SHA='c599a08848c3476da6e3c073ef7c0afd6fadd441cdf9498b99f449762395daed'

def validate_receipt(value,run_id):
    require(value.get('kind')==RECEIPT_KIND and value.get('schema')==1
        and value.get('status')=='actual_CLOSED_complete_shared_head_boundary'
        and value.get('run_id')==value.get('target_run_id')==run_id
        and value.get('allow_same_phase_exact_resume') is True
        and value.get('guards_retained') is True and value.get('further_unfreeze_allowed') is False,
        'ROOT same-ID CLOSED own-shared-head approval required')
    s,k,n,step,total=(value.get(key) for key in ('original_step','source_acquisition_updates',
        'shared_head_updates','completed_steps','approved_total_shared_head_updates'))
    require(all(type(v) is int and v>0 for v in (s,k,n,step,total)) and step==s+k+n
        and n<total<=44958-s-k,'Exact S/K/N and approved remaining horizon required')
    require(value.get('chosen_checkpoint_role') == 'latest'
        and value.get('chosen_step')==value.get('latest_DEV_step')==step and value.get('competitive_qualifying_latest_approved') is True,
        'No hidden earlier-BEST rewind; chosen own frontier must improve starting champion and passallfive')
    exits=value.get('owned_actual_exits',[])
    require(any(e.get('role')=='trainer_controller' for e in exits)
        and all(e.get('exit_code')==0 and e.get('session_id') and (e.get('chunk_id') or e.get('tool_chunk'))
            and datetime.fromisoformat(e['observed_at_utc'].replace('Z','+00:00')).tzinfo is not None for e in exits)
        and value.get('source_stop_category') in ('budget','phase_bound'),
        'Actual CLOSED0 budget/screen boundary only; no blind failure/resource restart')
    return value

def preflight(args):
    require(getattr(args,'resume_shared_head_exact',None) is not None and getattr(args,'shared_head_exact_source_receipt',None) is not None
        and getattr(args,'smoke_updates',None) is None
        and all(getattr(args,key,None) is None for key in ('resume_exact','resume_weights','resume_runtime_migration',
            'resume_optimization','resume_acquisition','resume_adaptation')),'Own exact mode only; no fallback/source reset')
    receipt_path=scoped(args.shared_head_exact_source_receipt); receipt=validate_receipt(read(receipt_path),args.run_id)
    artifacts=receipt['artifacts']
    for key in ('latest_checkpoint','best_checkpoint'):checked_record(artifacts[key],checkpoint=True)
    for key in ('run_config','training_history','selection','controller_status','original_native_ready',
        'initial_source_equivalence','runtime_native_ready','runtime_completion','fresh_runtime','memory','cache','io',
        'latest_DEV_summary','latest_DEV_metadata','startup_source_release'):
        checked_record(artifacts[key])
    require(scoped(args.resume_shared_head_exact)==scoped(artifacts['latest_checkpoint']['path']),
        'Use archived latest full state; BEST is independently preserved')
    config=read(artifacts['run_config']['path']);identity=config['config_identity']
    history=read(artifacts['training_history']['path']);selection=read(artifacts['selection']['path'])
    s,k,n,step=(receipt[key] for key in ('original_step','source_acquisition_updates','shared_head_updates','completed_steps'))
    phase=identity['adaptation_phase'];row_lr=args.row_lr;shared_lr=args.shared_lr
    require(config['run_id']==selection['run_id']==args.run_id and config['phase']==identity['phase']==selection['phase']==PHASE
        and canonical(identity)==receipt['scientific_config_identity_sha256'] and config['steps']==44958
        and config['smoke_updates'] is None and phase['source_phase1_step']==s+k
        and phase['selected_acquisition_updates']==k and phase['original_source_selected_step']==s
        and phase['row_lr']==row_lr and row_lr in (1e-3,3e-4)
        and phase['shared_lr']==shared_lr and shared_lr in (1e-5,3e-6)
        and phase['active_parameter_tensors']==15 and phase['active_shared_head_tensors']==12
        and phase['frozen_encoder_LoRA_tensors']==192 and phase['further_unfreeze_allowed'] is False,'Original declared shared15/two-Adam recipe/identity required')
    require(receipt['approved_total_shared_head_updates']<=phase['effective_max_updates']
        or receipt.get('allow_stopping_bound_extension') is True,'Explicit total-budget extension required')
    require(len(history)==step and all(row['step']==i for i,row in enumerate(history,1))
        and canonical(history)==receipt['training_history_sha256']
        and all(row['adaptation_updates_completed']==row['step']-s-k and row['row_Adam_clock']==k+row['adaptation_updates_completed']
            and row['shared_Adam_clock']==s+row['adaptation_updates_completed']
            for row in history[s+k:]),'Full original history/localN/actualAdamK+N required')
    gate=history[-1].get('validation',{})
    summary=read(artifacts['latest_DEV_summary']['path'])
    require(gate.get('retention_gate_passed') is True and gate.get('turkish_wer') is not None
        and gate['turkish_wer']==summary['validation_tr']['asr']['wer']
        and all(summary['validation_'+language]['asr']['utterances']==50 for language in ('tr','en','es','it','fr','de'))
        and gate['turkish_wer']<config['initial_adaptation_turkish_wer'],
        'Do not extend a retention failure or an all-five but regressed latest')
    baseline=read(HERE/'results/base_v3/summary.json')
    for language in ('en','es','it','fr','de'):
        base=baseline['validation_'+language]['asr']['wer']
        require(summary['validation_'+language]['asr']['wer']<=base+max(.005,.1*base),
            'Actual latest six-language summary fails original retention:'+language)
    require(selection['completed_steps']==step and selection['adaptation_updates_completed']==n
        and selection['source_phase1_step']==s+k and selection['selected_acquisition_updates']==k
        and selection['original_source_selected_step']==s and selection['row_Adam_clock']==k+n and selection['shared_Adam_clock']==s+n
        and s+k<=selection['selected_step']<=step,'Own selected/cumulative/retained Adam frontier differs')
    controller=read(artifacts['controller_status']['path']);completion=read(artifacts['runtime_completion']['path'])
    owned=read(artifacts['fresh_runtime']['path'])
    require(controller['status']=='train_window_ended' and controller['child_exit_code']==0
        and controller['resource_review_required'] is False and controller['run_id']==args.run_id
        and controller['completed_steps']==step and controller['adaptation_updates_completed']==n,
        'Actual nonresource CLOSED own controller required')
    source_executor=receipt['source_controller'];checked_record(source_executor)
    require(Path(source_executor['path']).name in (INITIAL_CONTROLLER,'run_TSC_bpe_shared_head_exact_v1.py')
        and controller['controller_sha256']==source_executor['sha256'], 'Source executor closure pin differs')
    if Path(source_executor['path']).name==INITIAL_CONTROLLER:
        require(source_executor['sha256']==INITIAL_CONTROLLER_SHA,'Original controller changed')
    else:
        require(source_executor['sha256']==sha(HERE/'run_TSC_bpe_shared_head_exact_v1.py'),
            'Closed exact executor must match frozen actual source')
    require(completion['kind']=='TSC_BPE_adaptation_native_completion_v1' and completion['status']=='complete'
        and completion['sources_unchanged'] is True and completion['run_id']==args.run_id
        and completion['actual_dispatch_source_sha256']==source_executor['sha256']
        and completion['backend_ready_sha256']==artifacts['runtime_native_ready']['sha256']
        and completion['pid']==owned['pid'],'Actual current-window owned native completion required')
    for key,kind in (('io','TSC_BPE_adaptation_atomic_IO_v1'),('cache','TSC_BPE_adaptation_cache_runtime_v1'),
        ('memory','TSC_BPE_adaptation_memory_runtime_v1')):
        value=read(artifacts[key]['path'])
        require(value['kind']==kind and value['status']=='complete' and value['sources_unchanged'] is True,
            'Complete unchanged actual runtime required:'+key)
    memory=read(artifacts['memory']['path']);cache=read(artifacts['cache']['path'])
    require(memory['cap_bytes']==8000000000 and memory['cleanup_threshold_bytes']==3000000000
        and memory['allocator_cap_applied'] is True and memory['all_phase_owned_memory_telemetry']['thread_join_completed'] is True
        and cache['cache_reclamation']['eligible_update_calls']==n-controller.get('shared_updates_at_window_start',0),
        'Preserve cap8/end3/current-window suffix hook count')
    release=read(artifacts['startup_source_release']['path'])
    compact=release.get('training_row_compaction',{})
    require(release.get('run_id')==args.run_id and release.get('passed') is True
        and compact.get('passed') is True and compact.get('rows_verified')==253798
        and compact.get('all_rows_canonical_equal') is True and compact.get('all_row_key_orders_equal') is True
        and compact.get('same_row_list_and_row_dict_objects') is True and compact.get('same_label_list_objects') is True
        and identity['source_sha256'].get('compact_training_row_objects_v1.py')=='1b3f9f0e3bdd19c285f2825a53685d7cf9fd964040c9087560f610a5ed1ec96d',
        'Keep actual complete unchanged carrier compaction proof')
    proof=read(artifacts['initial_source_equivalence']['path']);ready=read(artifacts['original_native_ready']['path'])
    require(proof['kind']==PROOF_KIND and proof['run_id']==args.run_id and proof['passed'] is True
        and proof['actual_GPU_decode'] is True and proof['before_shared_head_updates'] is True
        and proof['compared_fixed_DEV_utterances']==300 and proof['all_raw_transcripts_and_emitted_paths_exact'] is True
        and proof['all_six_ASR_counts_exact'] is True and proof['all_five_retention_gates_passed'] is True
        and identity['initial_source_equivalence_sha256']==artifacts['initial_source_equivalence']['sha256']
        and ready['run_id']==args.run_id and ready['trainer_sha256']==INITIAL_WORKER_SHA
        and ready['launcher_sha256']==INITIAL_CONTROLLER_SHA,'Keep original actual300/native scientific tokens')
    require(scoped(args.native_backend_provenance)==scoped(artifacts['original_native_ready']['path']),
        'Keep original ready path; new process proof is separate')
    ignored={'stop_file','max_wall_seconds','native_backend_provenance'}
    for key,value in identity['training_arguments'].items():
        if key in ignored:continue
        actual=getattr(args,key,None)
        require((scoped(actual)==scoped(value) if isinstance(actual,Path) else actual==value),
            'Immutable original argument changed:'+key)
    original=initial_preflight(args)
    return {**original,'original_context':original,'original_config':config,'parent_identity':identity,
        'exact_receipt':receipt,'exact_receipt_record':record(receipt_path),'archived_selection':selection,
        'resume_step':step,'shared_updates_at_resume':n,'approved_total_shared_head_updates':receipt['approved_total_shared_head_updates'],
        'source_phase1_step':s+k,'archived_history':history}

def validate_runtime_proof(args,*,worker_path):
    proof=read(scoped(args.shared_head_exact_runtime_proof));outer=HERE/'run_TSC_bpe_shared_head_exact_v1.py'
    require(proof['kind']=='TSC_BPE_shared_heads_exact_fresh_preload_v1' and proof['status']=='ready'
        and proof['phase']==PHASE and proof['run_id']==args.run_id and proof['pid']==os.getpid() and proof['ppid']==os.getppid()
        and proof['worker_executor']==record(scoped(worker_path)) and proof['controller_executor']==record(outer)
        and proof['exact_helper_sha256']==sha(__file__) and proof['original_native_ready']==record(args.native_backend_provenance)
        and proof['source_receipt']==record(args.shared_head_exact_source_receipt),'Fresh actual executor/PID proof required')
    preload=proof['preload'];cap=proof['allocator_policy']
    require(preload['kind']=='coherent_native_cuda12_9_preload_v1' and preload['actual_binding_linker']['version']==[12,9]
        and preload['cuda_initialized'] is False and preload['module_sha256']==sha(HERE/'preload_nemo_linker_v1.py')
        and preload['successful_launcher_sha256']==sha(HERE/'launch_nemo_linker_v4.py')
        and cap['cap_bytes']==8000000000 and cap['cleanup_threshold_bytes']==3000000000 and cap['allocator_cap_applied'] is True
        and proof['compactor_sha256']=='1b3f9f0e3bdd19c285f2825a53685d7cf9fd964040c9087560f610a5ed1ec96d'
        and proof['native4_adapter_sha256']=='24b9cf5b8f3e11d5b043c22d0940da3735a46f8f70b6f5810b97f17d545cd90e'
        and Path(sys.executable).resolve()==HERE.parent.parent/'work/nemo-native-env-v2/Scripts/python.exe',
        'Retain coherent12.9/native4/cap8/end3')
    return proof

def runtime_equal(left,right,torch_module):
    if torch_module.is_tensor(left) or torch_module.is_tensor(right):
        return torch_module.is_tensor(left) and torch_module.is_tensor(right) and left.shape==right.shape and left.dtype==right.dtype \
            and torch_module.equal(left.detach().cpu().contiguous().reshape(-1).view(torch_module.uint8),right.detach().cpu().contiguous().reshape(-1).view(torch_module.uint8))
    if type(left) is not type(right):return False
    if isinstance(left,dict):return left.keys()==right.keys() and all(runtime_equal(left[k],right[k],torch_module) for k in left)
    if isinstance(left,(list,tuple)):return len(left)==len(right) and all(runtime_equal(a,b,torch_module) for a,b in zip(left,right))
    return left==right

def full_TRAIN_coverage(coverage):
    """Budget44958 alone is never evidence of full unique coverage."""
    expected={'fleurs':2037,'cv':72846,'tsc':178915}
    groups=coverage.get('source_groups',{})
    counts={name:groups.get(name,{}).get('unique_clips_seen') for name in expected}
    complete=all(groups.get(name,{}).get('pool_clips')==total and counts[name]==total for name,total in expected.items())
    return {'actual_unique_TR_clips':sum(n for n in counts.values() if type(n) is int),
        'expected_unique_TR_clips':253798,'per_source_unique':counts,'expected_per_source_unique':expected,
        'full_unique_TRAIN_coverage_complete':complete,'configured_horizon_alone_proves_coverage':False}


def restore_runtime(adaptation,sampler,state,context,*,torch_module,random_module=random):
    receipt=context['exact_receipt'];runtime=state.get('runtime_state',{})
    s,k,n=(receipt[key] for key in ('original_step','source_acquisition_updates','shared_head_updates'));step=s+k+n
    require(sha(HERE/'bpe_newrows_shared_head_adaptation_v1.py')==UTILITY_SHA
        and state.get('schema')==1 and state.get('kind')==STATE_KIND
        and state.get('complete_two_optimizer_update_boundary') is True
        and state.get('run_id')==runtime.get('run_id')==context['original_config']['run_id']
        and state.get('step')==runtime.get('step')==state.get('cumulative_step')==step
        and state.get('original_source_selected_step')==runtime.get('original_source_selected_step')==s
        and state.get('selected_acquisition_updates')==runtime.get('selected_acquisition_updates')==k
        and state.get('phase2_updates')==state.get('shared_head_updates_completed')==runtime.get('adaptation_updates_completed')==n
        and state.get('row_Adam_clock')==k+n and state.get('shared_Adam_clock')==s+n
        and state.get('source_phase1_step')==runtime.get('source_phase1_step')==s+k
        and runtime.get('source_selected_step')==s,
        'One complete own207/row3+shared12/S-K-N/runtime frontier required')
    require(state['config_identity']==runtime['config_identity']
        and canonical(state['config_identity'])==canonical(context['parent_identity'])
        and canonical(state['phase_identity'])==canonical(context['parent_identity']['adaptation_optimizers'])
        and canonical(runtime['history'])==receipt['training_history_sha256'] and len(runtime['history'])==step
        and state['selected_step']==runtime['selected_step']==context['archived_selection']['selected_step']
        and runtime['best_score']==context['archived_selection']['turkish_validation_wer']
        and runtime['sampler']['updates']==step
        and canonical(runtime['sampler']['identity'])==canonical(context['parent_identity']['domain_sampling_identity']),
        'Typed state/runtime aliases plus canonical JSON science/history/BEST identity required')
    for key in ('python_rng','torch_rng','cuda_rng','patience','retention_failures','next_eval_epoch',
                'train_wall_seconds','evaluation_wall_seconds','audio_seconds','language_audio_seconds','initial_gate'):
        require(key in runtime and runtime[key] is not None,'Missing exact runtime:'+key)
    diagnostics=runtime['history'][-1]['selection_diagnostics']
    require(runtime['retention_failures']==diagnostics['consecutive_retention_failures']==0
        and runtime['patience']==diagnostics['plateau_epoch_evaluations'],'Carry actual selection/failure counters')
    # The pinned utility validates all207/protected bytes and actual15 moments,
    # groups/FP32/finite/nonnegative variance/K+N/S+N clocks before copying.
    returned=adaptation.load_phase_state(state)
    require(runtime_equal(returned,runtime,torch_module) and adaptation.optimizer_updates_completed==n
        and adaptation.cumulative_step==step,'Use CURRENT returned runtime, never the phase1 constructor clone')
    adaptation.inherited_runtime=None
    sampler.load_state_dict(returned['sampler'])
    require(runtime_equal(sampler.state_dict(),returned['sampler'],torch_module)
        and canonical(sampler.coverage())==canonical(context['archived_selection']['Turkish_domain_coverage']),
        'Exact sampler queues/RNG/draws/domain/source coverage changed')
    random_module.setstate(returned['python_rng']);torch_module.set_rng_state(returned['torch_rng'])
    torch_module.cuda.set_rng_state_all(returned['cuda_rng']);adaptation.zero_grad()
    return returned
