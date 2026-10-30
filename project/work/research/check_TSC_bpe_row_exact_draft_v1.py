"""Small inert exact receipt/source-flow fixture. No state/model/GPU reads."""
import ast
import copy
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'outputs/parakeet_turkish'
sys.path.insert(0,str(HERE))
import tsc_bpe_row_exact_transition_v1 as h

def main():
    checks=[]
    def passed(name):checks.append(name)
    def rejected(callback,name):
        try:callback()
        except (ValueError,KeyError):passed(name);return
        raise AssertionError('Accepted:'+name)
    r={'kind':h.RECEIPT_KIND,'schema':1,'status':'actual_CLOSED_complete_continued_row_boundary',
        'run_id':'synthetic_same_row','target_run_id':'synthetic_same_row','allow_same_phase_exact_resume':True,
        'guards_retained':True,'unfreeze_allowed':False,'original_step':6500,'source_acquisition_updates':500,
        'continued_acquisition_updates':1000,'completed_steps':8000,'approved_total_continued_updates':3000,
        'chosen_checkpoint_role':'latest','chosen_step':8000,'competitive_qualifying_latest_approved':True,
        'source_stop_category':'phase_bound','owned_actual_exits':[{'role':'trainer_controller','exit_code':0,
        'session_id':999,'chunk_id':'EXPLICIT_FIXTURE_ONLY','observed_at_utc':'2026-10-12T00:00:00Z'}]}
    assert h.validate_receipt(r,'synthetic_same_row') is r
    passed('actual_receipt_admits_separate3000_total_without_rewriting_initial1000')
    for key,bad in (('chosen_step',7750),('target_run_id','different_run'),('source_stop_category','resource'),
        ('competitive_qualifying_latest_approved',False),('approved_total_continued_updates',37959),
        ('unfreeze_allowed',True)):
        altered=copy.deepcopy(r);altered[key]=bad
        rejected(lambda altered=altered:h.validate_receipt(altered,'synthetic_same_row'),'reject_'+key)
    assert h.runtime_equal({'group':{'betas':(.9,.999)}},{'group':{'betas':(.9,.999)}},type('NoTorch',(),{'is_tensor':staticmethod(lambda v:False)}))
    assert not h.runtime_equal({'group':{'betas':(.9,.999)}},{'group':{'betas':[.9,.999]}},type('NoTorch',(),{'is_tensor':staticmethod(lambda v:False)}))
    passed('typed_internal_runtime_preserves_tuple_group_while_external_comparison_is_canonical')
    base=ast.parse((HERE/'train_lora_batched_tsc_bpe_row_lr_v1.py').read_text(encoding='utf-8'))
    exact=ast.parse((HERE/'train_lora_batched_tsc_bpe_row_exact_v1.py').read_text(encoding='utf-8'))
    bf={n.name:n for n in base.body if isinstance(n,ast.FunctionDef)}
    ef={n.name:n for n in exact.body if isinstance(n,ast.FunctionDef)}
    for name in ('batch_objective','save_training_state','_protected_snapshot','_verify_smoke_phase_reload','native_loss_function'):
        assert ast.dump(bf[name],include_attributes=False)==ast.dump(ef[name],include_attributes=False),name
    passed('actual_objective_207_threeAdam_serializer_native4_and_protected_byte_checker_AST_unchanged')
    run=ef['run'];text=ast.unparse(run)
    assert 'initial_source_equivalence(' not in text and 'run_config.json' not in text.split('write_json(OUTPUT')[0]
    assert 'range(resume_step + 1, phase_end + 1)' in text and 'original_config' in text
    assert text.index('del original, source, runtime, baseline_processor')<text.index('own_state = torch.load')
    assert text.index('restore_runtime(')<text.index('range(resume_step + 1, phase_end + 1)')
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='save'
        and ast.unparse(n.args[0])=='best_path' for n in run.body if isinstance(n,ast.Expr) for n in ast.walk(n))
    passed('source300_BEST_config_untouched_and_old_CPU_pair_released_before_ownlatest')
    assert not any(name in sys.modules for name in ('torch','transformers','nemo'))
    passed('no_actual_ML_state_or_hardware_execution')
    print(json.dumps({'passed':True,'checks':checks,'count':len(checks),
        'scope':'Synthetic receipts and actual source AST only; no actual own row screen closed/admitted'}))

if __name__=='__main__':main()
