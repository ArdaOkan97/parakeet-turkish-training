"""INACTIVE same-ID continued-row frontier admission; no actual state read here.

Only a ROOT CLOSED, competitive qualifying latest frontier is admitted. Earlier
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
from tsc_bpe_row_lr_transition_v1 import (preflight as initial_preflight, HERE,
    PHASE, STATE_KIND, PROOF_KIND, PROOF_NAME, require, sha, canonical, scoped,
    read, record, checked_record)

RECEIPT_KIND='ROOT_owned_closed_TSC_BPE_continued_row_exact_v1'
UTILITY_SHA='ee25eb365a3bd578c48ca95aa66be4429146ef96319ac5614d12c8a6248049b7'
INITIAL_WORKER='train_lora_batched_tsc_bpe_row_lr_v1.py'
INITIAL_CONTROLLER='run_TSC_bpe_row_lr_v1.py'
INITIAL_WORKER_SHA='7f3f71c0ad6091c9345d866bebaca81d78e2de38b2e818ff6223cb8d5ee644f1'
INITIAL_CONTROLLER_SHA='c820dc8552370ac08a934a9595500e65c036f0fd223eaa38b05be0cfda0b5012'

def validate_receipt(value,run_id):
    require(value.get('kind')==RECEIPT_KIND and value.get('schema')==1
        and value.get('status')=='actual_CLOSED_complete_continued_row_boundary'
        and value.get('run_id')==value.get('target_run_id')==run_id
        and value.get('allow_same_phase_exact_resume') is True
        and value.get('guards_retained') is True and value.get('unfreeze_allowed') is False,
        'ROOT same-ID CLOSED own-row approval required')
    s,k,n,step,total=(value.get(key) for key in ('original_step','source_acquisition_updates',
        'continued_acquisition_updates','completed_steps','approved_total_continued_updates'))
    require(all(type(v) is int and v>0 for v in (s,k,n,step,total)) and step==s+k+n
        and n<total<=44958-s-k,'Exact S/K/N and approved remaining horizon required')
    require(value.get('chosen_checkpoint_role') in ('latest','best')
        and value.get('chosen_step')==step and value.get('competitive_qualifying_latest_approved') is True,
        'No hidden earlier-BEST rewind; chosen own frontier must be competitive and qualify')
    exits=value.get('owned_actual_exits',[])
    require(any(e.get('role')=='trainer_controller' for e in exits)
        and all(e.get('exit_code')==0 and e.get('session_id') and (e.get('chunk_id') or e.get('tool_chunk'))
            and datetime.fromisoformat(e['observed_at_utc'].replace('Z','+00:00')).tzinfo is not None for e in exits)
        and value.get('source_stop_category') in ('budget','phase_bound'),
        'Actual CLOSED0 budget/screen boundary only; no blind failure/resource restart')
    return value

def preflight(args):
    require(getattr(args,'resume_row_exact',None) is not None and getattr(args,'row_exact_source_receipt',None) is not None
        and getattr(args,'smoke_updates',None) is None
        and all(getattr(args,key,None) is None for key in ('resume_exact','resume_weights','resume_runtime_migration',
            'resume_optimization','resume_acquisition','resume_adaptation')),'Own exact mode only; no fallback/source reset')
    receipt_path=scoped(args.row_exact_source_receipt); receipt=validate_receipt(read(receipt_path),args.run_id)
    artifacts=receipt['artifacts']
    for key in ('latest_checkpoint','best_checkpoint'):checked_record(artifacts[key],checkpoint=True)
    for key in ('run_config','training_history','selection','controller_status','original_native_ready',
        'initial_source_equivalence','runtime_native_ready','runtime_completion','fresh_runtime','memory','cache','io'):
        checked_record(artifacts[key])
    require(scoped(args.resume_row_exact)==scoped(artifacts['latest_checkpoint']['path']),
        'Use archived latest full state; BEST is independently preserved')
    if receipt['chosen_checkpoint_role']=='best':
        require(artifacts['best_checkpoint']['sha256']==artifacts['latest_checkpoint']['sha256']
            and artifacts['best_checkpoint']['bytes']==artifacts['latest_checkpoint']['bytes'],
            'BEST label may not substitute an earlier/different optimizer frontier')
    config=read(artifacts['run_config']['path']);identity=config['config_identity']
    history=read(artifacts['training_history']['path']);selection=read(artifacts['selection']['path'])
    s,k,n,step=(receipt[key] for key in ('original_step','source_acquisition_updates','continued_acquisition_updates','completed_steps'))
    phase=identity['acquisition_phase'];lr=args.active_row_lr
    require(config['run_id']==selection['run_id']==args.run_id and config['phase']==identity['phase']==selection['phase']==PHASE
        and canonical(identity)==receipt['scientific_config_identity_sha256'] and config['steps']==44958
        and config['smoke_updates'] is None and phase['source_selected_step']==s+k
        and phase['source_acquisition_updates']==k and phase['source_row_lr']==1e-4
        and phase['acquisition_lr']==lr and lr in (1e-3,3e-4) and phase['active_optimizer_tables']==3
        and phase['unfreeze_allowed'] is False,'Original declared row-only recipe/identity required')
    require(receipt['approved_total_continued_updates']<=phase['effective_max_updates']
        or receipt.get('allow_stopping_bound_extension') is True,'Explicit total-budget extension required')
    require(len(history)==step and all(row['step']==i for i,row in enumerate(history,1))
        and canonical(history)==receipt['training_history_sha256']
        and all(row['acquisition_updates_completed']==row['step']-s-k and row['row_Adam_clock']==k+row['acquisition_updates_completed']
            for row in history[s+k:]),'Full original history/localN/actualAdamK+N required')
    gate=history[-1].get('validation',{})
    require(gate.get('retention_gate_passed') is True and gate.get('turkish_wer') is not None
        and gate['turkish_wer']<=config['initial_acquisition_turkish_wer'],
        'Do not extend a retention failure or an all-five but regressed latest')
    require(selection['completed_steps']==step and selection['acquisition_updates_completed']==n
        and selection['source_selected_step']==s+k and selection['source_acquisition_updates']==k
        and selection['original_source_selected_step']==s and selection['row_Adam_clock']==k+n
        and s+k<=selection['selected_step']<=step,'Own selected/cumulative/retained Adam frontier differs')
    controller=read(artifacts['controller_status']['path']);completion=read(artifacts['runtime_completion']['path'])
    owned=read(artifacts['fresh_runtime']['path'])
    require(controller['status']=='train_window_ended' and controller['child_exit_code']==0
        and controller['resource_review_required'] is False and controller['run_id']==args.run_id
        and controller['completed_steps']==step and controller['acquisition_updates_completed']==n,
        'Actual nonresource CLOSED own controller required')
    source_executor=receipt['source_controller'];checked_record(source_executor)
    require(Path(source_executor['path']).name in (INITIAL_CONTROLLER,'resume_TSC_bpe_row_same_phase_v1.py')
        and controller['controller_sha256']==source_executor['sha256'], 'Source executor closure pin differs')
    if Path(source_executor['path']).name==INITIAL_CONTROLLER:
        require(source_executor['sha256']==INITIAL_CONTROLLER_SHA,'Original controller changed')
    require(completion['kind']=='TSC_BPE_acquisition_native_completion_v1' and completion['status']=='complete'
        and completion['sources_unchanged'] is True and completion['run_id']==args.run_id
        and completion['actual_dispatch_source_sha256']==source_executor['sha256']
        and completion['backend_ready_sha256']==artifacts['runtime_native_ready']['sha256']
        and completion['pid']==owned['pid'],'Actual current-window owned native completion required')
    for key,kind in (('io','TSC_BPE_acquisition_atomic_IO_v1'),('cache','TSC_BPE_acquisition_cache_runtime_v1'),
        ('memory','TSC_BPE_acquisition_memory_runtime_v1')):
        value=read(artifacts[key]['path'])
        require(value['kind']==kind and value['status']=='complete' and value['sources_unchanged'] is True,
            'Complete unchanged actual runtime required:'+key)
    memory=read(artifacts['memory']['path']);cache=read(artifacts['cache']['path'])
    require(memory['cap_bytes']==8000000000 and memory['cleanup_threshold_bytes']==4000000000
        and memory['allocator_cap_applied'] is True and memory['all_phase_owned_memory_telemetry']['thread_join_completed'] is True
        and cache['cache_reclamation']['eligible_update_calls']==n-controller.get('continued_updates_at_window_start',0),
        'Preserve cap8/end4/current-window suffix hook count')
    proof=read(artifacts['initial_source_equivalence']['path']);ready=read(artifacts['original_native_ready']['path'])
    require(proof['kind']==PROOF_KIND and proof['run_id']==args.run_id and proof['passed'] is True
        and proof['actual_GPU_decode'] is True and proof['before_row_lr_updates'] is True
        and proof['compared_fixed_DEV_utterances']==300 and proof['all_raw_transcripts_and_emitted_paths_exact'] is True
        and proof['all_six_ASR_counts_exact'] is True and proof['all_five_retention_gates_passed'] is True
        and identity['initial_source_equivalence_sha256']==artifacts['initial_source_equivalence']['sha256']
        and ready['run_id']==args.run_id and ready['trainer_sha256']==INITIAL_WORKER_SHA
        and ready['launcher_sha256']==INITIAL_CONTROLLER_SHA,'Keep original actual300/native scientific tokens')
    require(scoped(args.native_backend_provenance)==scoped(artifacts['original_native_ready']['path']),
        'Keep original ready path; new process proof is separate')
    ignored={'stop_file','max_wall_seconds','log_every','native_backend_provenance'}
    for key,value in identity['training_arguments'].items():
        if key in ignored:continue
        actual=getattr(args,key,None)
        require((scoped(actual)==scoped(value) if isinstance(actual,Path) else actual==value),
            'Immutable original argument changed:'+key)
    original=initial_preflight(args)
    return {**original,'original_context':original,'original_config':config,'parent_identity':identity,
        'exact_receipt':receipt,'exact_receipt_record':record(receipt_path),'archived_selection':selection,
        'resume_step':step,'continued_updates_at_resume':n,'approved_total_continued_updates':receipt['approved_total_continued_updates'],
        'source_phase1_step':s+k,'archived_history':history}

def validate_runtime_proof(args,*,worker_path):
    proof=read(scoped(args.row_exact_runtime_proof));outer=HERE/'resume_TSC_bpe_row_same_phase_v1.py'
    require(proof['kind']=='TSC_BPE_continued_row_exact_fresh_preload_v1' and proof['status']=='ready'
        and proof['phase']==PHASE and proof['run_id']==args.run_id and proof['pid']==os.getpid() and proof['ppid']==os.getppid()
        and proof['worker_executor']==record(scoped(worker_path)) and proof['controller_executor']==record(outer)
        and proof['exact_helper_sha256']==sha(__file__) and proof['original_native_ready']==record(args.native_backend_provenance)
        and proof['source_receipt']==record(args.row_exact_source_receipt),'Fresh actual executor/PID proof required')
    preload=proof['preload'];cap=proof['allocator_policy']
    require(preload['kind']=='coherent_native_cuda12_9_preload_v1' and preload['actual_binding_linker']['version']==[12,9]
        and preload['cuda_initialized'] is False and preload['module_sha256']==sha(HERE/'preload_nemo_linker_v1.py')
        and preload['successful_launcher_sha256']==sha(HERE/'launch_nemo_linker_v4.py')
        and cap['cap_bytes']==8000000000 and cap['cleanup_threshold_bytes']==4000000000 and cap['allocator_cap_applied'] is True
        and proof['native4_adapter_sha256']=='24b9cf5b8f3e11d5b043c22d0940da3735a46f8f70b6f5810b97f17d545cd90e'
        and Path(sys.executable).resolve()==HERE.parent.parent/'work/nemo-native-env-v2/Scripts/python.exe',
        'Retain coherent12.9/native4/cap8/end4')
    return proof

def runtime_equal(left,right,torch_module):
    if torch_module.is_tensor(left) or torch_module.is_tensor(right):
        return torch_module.is_tensor(left) and torch_module.is_tensor(right) and left.shape==right.shape and left.dtype==right.dtype \
            and torch_module.equal(left.detach().cpu().contiguous().reshape(-1).view(torch_module.uint8),right.detach().cpu().contiguous().reshape(-1).view(torch_module.uint8))
    if type(left) is not type(right):return False
    if isinstance(left,dict):return left.keys()==right.keys() and all(runtime_equal(left[k],right[k],torch_module) for k in left)
    if isinstance(left,(list,tuple)):return len(left)==len(right) and all(runtime_equal(a,b,torch_module) for a,b in zip(left,right))
    return left==right

def restore_runtime(acquisition,sampler,state,context,*,torch_module,random_module=random):
    receipt=context['exact_receipt'];runtime=state.get('runtime_state',{});s=receipt['original_step'];k=receipt['source_acquisition_updates'];n=receipt['continued_acquisition_updates'];step=s+k+n
    require(sha(HERE/'bpe_newrows_continued_acquisition_v1.py')==UTILITY_SHA
        and state.get('schema')==1 and state.get('kind')==STATE_KIND and state.get('complete_optimizer_update_boundary') is True
        and state.get('run_id')==runtime.get('run_id')==context['original_config']['run_id']
        and state.get('step')==runtime.get('step')==state.get('cumulative_step')==step
        and state.get('source_step')==runtime.get('original_source_selected_step')==s
        and state.get('source_acquisition_updates')==runtime.get('source_acquisition_updates')==k
        and state.get('continued_acquisition_updates')==runtime.get('acquisition_updates_completed')==n
        and state.get('acquisition_optimizer_updates')==k+n
        and state.get('source_selected_step')==runtime.get('source_selected_step')==s+k,
        'One actual complete own207/3Adam S/K/N/runtime frontier required')
    require(state['config_identity']==runtime['config_identity']
        and canonical(state['config_identity'])==canonical(context['parent_identity'])
        and canonical(state['phase_identity'])==canonical(context['parent_identity']['acquisition_optimizer'])
        and canonical(runtime['history'])==receipt['training_history_sha256'] and len(runtime['history'])==step
        and state['selected_step']==runtime['selected_step']==context['archived_selection']['selected_step']
        and runtime['best_score']==context['archived_selection']['turkish_validation_wer']
        and runtime['sampler']['updates']==step
        and canonical(runtime['sampler']['identity'])==canonical(context['parent_identity']['domain_sampling_identity']),
        'Typed internal aliases and serialization-equivalent JSON science/history/BEST identity required')
    for key in ('python_rng','torch_rng','cuda_rng','patience','retention_failures','next_eval_epoch',
        'train_wall_seconds','evaluation_wall_seconds','audio_seconds','language_audio_seconds','initial_gate'):
        require(key in runtime and runtime[key] is not None,'Missing exact runtime:'+key)
    diagnostics=runtime['history'][-1]['selection_diagnostics']
    require(runtime['retention_failures']==diagnostics['consecutive_retention_failures']==0
        and runtime['patience']==diagnostics['plateau_epoch_evaluations'],'Selection/failure counters must carry')
    returned=acquisition.load_phase_state(state)
    require(runtime_equal(returned,runtime,torch_module) and acquisition.optimizer_updates_completed==n
        and acquisition.cumulative_step==step,'Use fully validated returned own runtime, not old phase1 clone')
    sampler.load_state_dict(returned['sampler'])
    require(runtime_equal(sampler.state_dict(),returned['sampler'],torch_module),'Exact sampler queues/RNG/coverage changed')
    random_module.setstate(returned['python_rng']);torch_module.set_rng_state(returned['torch_rng'])
    torch_module.cuda.set_rng_state_all(returned['cuda_rng']);acquisition.zero_grad()
    return returned
