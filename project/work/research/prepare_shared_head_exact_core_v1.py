"""NEW same-phase shared-head admission/restore core, source-only preparation."""
from pathlib import Path
import ast

ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'outputs/parakeet_turkish'
text=(HERE/'tsc_bpe_row_exact_transition_v1.py').read_text(encoding='utf-8')
def rep(old,new):
    global text
    assert old in text,old[:100]
    text=text.replace(old,new)
rep('from tsc_bpe_row_lr_transition_v1 import','from tsc_bpe_shared_head_memory_v2 import')
rep("ROOT_owned_closed_TSC_BPE_continued_row_exact_v1","ROOT_owned_closed_TSC_BPE_shared_heads_exact_v1")
rep("actual_CLOSED_complete_continued_row_boundary","actual_CLOSED_complete_shared_head_boundary")
rep("ee25eb365a3bd578c48ca95aa66be4429146ef96319ac5614d12c8a6248049b7","384d01abe9b7105761491b8acda5d3de4eb8c40ae679eee2701269079cf9967c")
rep("train_lora_batched_tsc_bpe_row_lr_v1.py","train_lora_batched_tsc_bpe_shared_head_memory_v2.py")
rep("run_TSC_bpe_row_lr_v1.py","run_TSC_bpe_shared_head_memory_v2.py")
rep("7f3f71c0ad6091c9345d866bebaca81d78e2de38b2e818ff6223cb8d5ee644f1","ea74795363e0c09114e0a496ccbea55dab34172202d40c6a9aefb5f7cdfa8290")
rep("c820dc8552370ac08a934a9595500e65c036f0fd223eaa38b05be0cfda0b5012","c599a08848c3476da6e3c073ef7c0afd6fadd441cdf9498b99f449762395daed")
rep('continued_acquisition_updates','shared_head_updates')
rep('approved_total_continued_updates','approved_total_shared_head_updates')
rep('continued_updates_at_resume','shared_updates_at_resume')
rep('continued_updates_at_window_start','shared_updates_at_window_start')
rep('resume_row_exact','resume_shared_head_exact')
rep('row_exact_source_receipt','shared_head_exact_source_receipt')
rep('row_exact_runtime_proof','shared_head_exact_runtime_proof')
rep('resume_TSC_bpe_row_same_phase_v1.py','run_TSC_bpe_shared_head_exact_v1.py')
rep('TSC_BPE_continued_row_exact_fresh_preload_v1','TSC_BPE_shared_heads_exact_fresh_preload_v1')
rep('before_row_lr_updates','before_shared_head_updates')
rep('TSC_BPE_acquisition_native_completion_v1','TSC_BPE_adaptation_native_completion_v1')
rep('TSC_BPE_acquisition_atomic_IO_v1','TSC_BPE_adaptation_atomic_IO_v1')
rep('TSC_BPE_acquisition_cache_runtime_v1','TSC_BPE_adaptation_cache_runtime_v1')
rep('TSC_BPE_acquisition_memory_runtime_v1','TSC_BPE_adaptation_memory_runtime_v1')
rep('4000000000','3000000000')
rep("value.get('unfreeze_allowed') is False","value.get('further_unfreeze_allowed') is False")
rep("value.get('chosen_checkpoint_role') in ('latest','best')","value.get('chosen_checkpoint_role') == 'latest'")
rep("phase=identity['acquisition_phase'];lr=args.active_row_lr","phase=identity['adaptation_phase'];row_lr=args.row_lr;shared_lr=args.shared_lr")
rep("phase['source_selected_step']==s+k\n        and phase['source_acquisition_updates']==k and phase['source_row_lr']==1e-4\n        and phase['acquisition_lr']==lr and lr in (1e-3,3e-4) and phase['active_optimizer_tables']==3\n        and phase['unfreeze_allowed'] is False",
    "phase['source_phase1_step']==s+k\n        and phase['selected_acquisition_updates']==k and phase['original_source_selected_step']==s\n        and phase['row_lr']==row_lr and row_lr in (1e-3,3e-4)\n        and phase['shared_lr']==shared_lr and shared_lr in (1e-5,3e-6)\n        and phase['active_parameter_tensors']==15 and phase['active_shared_head_tensors']==12\n        and phase['frozen_encoder_LoRA_tensors']==192 and phase['further_unfreeze_allowed'] is False")
rep("row['acquisition_updates_completed']==row['step']-s-k and row['row_Adam_clock']==k+row['acquisition_updates_completed']",
    "row['adaptation_updates_completed']==row['step']-s-k and row['row_Adam_clock']==k+row['adaptation_updates_completed']\n            and row['shared_Adam_clock']==s+row['adaptation_updates_completed']")
rep("gate['turkish_wer']<=config['initial_acquisition_turkish_wer']", "gate['turkish_wer']<config['initial_adaptation_turkish_wer']")
rep("selection['acquisition_updates_completed']==n\n        and selection['source_selected_step']==s+k and selection['source_acquisition_updates']==k\n        and selection['original_source_selected_step']==s and selection['row_Adam_clock']==k+n",
    "selection['adaptation_updates_completed']==n\n        and selection['source_phase1_step']==s+k and selection['selected_acquisition_updates']==k\n        and selection['original_source_selected_step']==s and selection['row_Adam_clock']==k+n and selection['shared_Adam_clock']==s+n")
rep("controller['acquisition_updates_completed']==n", "controller['adaptation_updates_completed']==n")
rep("ignored={'stop_file','max_wall_seconds','log_every','native_backend_provenance'}", "ignored={'stop_file','max_wall_seconds','native_backend_provenance'}")
rep("same-ID CLOSED own-row", "same-ID CLOSED own-shared-head")
rep("Original declared row-only", "Original declared shared15/two-Adam")
rep("Preserve cap8/end4", "Preserve cap8/end3")
rep("Retain coherent12.9/native4/cap8/end4", "Retain coherent12.9/native4/cap8/end3")
rep("from tsc_bpe_shared_head_memory_v2 import (preflight as initial_preflight, HERE,", "from tsc_bpe_shared_head_memory_v2 import (preflight as initial_preflight, HERE,")
start=text.index('def restore_runtime(')
text=text[:start]+'''def full_TRAIN_coverage(coverage):
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
'''
# Same source phase1 admission still uses its immutable original receipt/flags.
# No initial execution identity is rewritten or new source300 evaluation added.
text=text.replace("and cap['allocator_cap_applied'] is True", "and cap['allocator_cap_applied'] is True\n        and proof['compactor_sha256']=='1b3f9f0e3bdd19c285f2825a53685d7cf9fd964040c9087560f610a5ed1ec96d'")
text=text.replace('INACTIVE same-ID continued-row frontier admission','INACTIVE same-ID shared-head frontier admission')
text=text.replace('Only a ROOT CLOSED, competitive qualifying latest frontier is admitted.',
    'Only a ROOT CLOSED latest improving over its startingchampion and passingallfive gates is admitted; the11.9394% final goal is not an entry gate.')
path=HERE/'tsc_bpe_shared_head_exact_transition_v1.py';assert not path.exists();ast.parse(text);path.write_text(text,encoding='utf-8')
print('Saved NEW minimal shared own-phase admission+restore core; exact worker/dispatcher wiring still pending.')
