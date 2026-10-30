"""Derive one focused stdlib fixture from frozen fulluniform dispatch fixture."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'work/research'
t=(HERE/'check_TSC_uniform_full_dispatch_v1.py').read_text(encoding='utf-8')
t=t.replace('run_TSC_uniform_full_v1.py','run_TSC_uniform_majority_v1.py')
t=t.replace('train_lora_batched_tsc_headlr_v1.py','train_lora_batched_tsc_uniform_full_v1.py')
t=t.replace("'best, selected_step = (math.inf, None)' in run", "'best, selected_step = (math.inf, None)' not in run and 'next_eval = resume[\'next_eval_epoch\']' in run")
t=t.replace('source207_Adam_and_PT_recovery_all6_BEST_separate','source207_Adam_BEST_and_no_PT_grace_or_evalclock_reset')
t=t.replace("'--uniform-full-source-receipt',", "'--uniform-full-source-receipt','--majority-source-receipt',")
t=t.replace("'--resume-exact' if exact else '--resume-uniform-full'", "'--resume-exact' if exact else '--resume-majority'")
t=t.replace('TSC_uniform_full_stdlib_dispatch_verification_v1','TSC_uniform_majority_stdlib_dispatch_verification_v1')
t=t.replace("'actual_GPU_model_source_PT_admission_executed':False", "'actual_GPU_model_source_PT_admission_executed':False,'actual_parent_checkpoint_read':False")
# Add real metadata/trace/own-state function checks without importing the worker.
marker="    current={'available':True"
position=t.index(marker)
extra='''    parent_tree=ast.parse((HERE/'train_lora_batched_tsc_uniform_full_v1.py').read_text())
    parent_functions={node.name:node for node in parent_tree.body if isinstance(node,ast.FunctionDef)}
    for name in ('lr_factor','development_gate','evaluate','save_training_state','native_loss_function','annotate_cache_lengths'):
        check('unchanged_'+name,ast.dump(functions[name],include_attributes=False)==ast.dump(parent_functions[name],include_attributes=False))
    def update_body(fn):
        loop=next(node for node in fn.body if isinstance(node,ast.For) and isinstance(node.target,ast.Name) and node.target.id=='step')
        end=next(i for i,node in enumerate(loop.body) if isinstance(node,ast.Expr) and ast.unparse(node)=='torch.cuda.synchronize()')
        return ast.dump(ast.Module(body=loop.body[:end+1],type_ignores=[]),include_attributes=False)
    check('whole_207_backward_clip_Adam_sixboost_native_update_AST_identical',update_body(functions['run'])==update_body(parent_functions['run']))
    check('actual_prior_index_reopened_no_fresh_admission', 'prepare_approved_TR_rows(' not in ast.unparse(functions['load_indexed_pool'])
        and 'context["indexed_audit"]' in ast.get_source_segment(worker_path.read_text(),functions['load_indexed_pool']))
    check('parent_all6_BEST_not_saved_as_failed_H', 'selected_step == start_step' in run and 'source_best_checkpoint' in run)
    check('density40_2_4_parser_defaults',all(text in ast.unparse(functions['parser']) for text in ('default=0.125','default=40.0','default=2.0','default=4.0')))
    check('record_function_not_shadowed_and_trace_path_keyword_fixed', 'record as file_record' in worker_path.read_text()
        and 'checkpoint_path=str(path)' in worker_path.read_text() and 'stage_memory(memory_trace, "before_atomic207_state_save", step=step, path=' not in run)
'''
t=t[:position]+extra+t[position:]
dest=HERE/'check_TSC_uniform_majority_dispatch_v1.py'
if dest.exists():raise ValueError('Fixture already exists')
dest.write_text(t,encoding='utf-8',newline='\n')
print(dest)
