"""Narrow stdlib receipt/LR/frontier compatibility checks; no real admission."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'outputs/parakeet_turkish'
sys.path.insert(0,str(HERE))
import tsc_bpe_row_lr_transition_v1 as h

def main():
    checks=[]
    def check(ok,name):
        assert ok,name
        checks.append(name)
    def rejected(call,name):
        try:call()
        except (ValueError,KeyError):checks.append(name);return
        raise AssertionError(name+' accepted')
    for lr in (1e-3,3e-4):
        r={'schema':1,'kind':h.RECEIPT_KIND,'status':'actual_CLOSED_qualified_phase1',
            'target_run_id':'synthetic_rowLR','source_run_id':'synthetic_phase1',
            'allow_continued_row_only_acquisition':True,'source_phase2_latest_used':False,
            'declared_row_lr':h.continued_acquisition_declaration(lr),'original_step':10,
            'acquisition_updates':2,'chosen_step':12,'completed_steps':12,'chosen_checkpoint_role':'latest',
            'retention_gate_passed_at_chosen_step':True,'owned_actual_exits':[{'role':'trainer_controller',
                'exit_code':0,'session_id':999,'chunk_id':'EXPLICIT_SYNTHETIC','observed_at_utc':'fixture_only'}]}
        check(h.validate_receipt(r,'synthetic_rowLR',lr) is r,'actual_receipt_'+str(lr))
        for key,bad in (('source_phase2_latest_used',True),('kind','ROOT_owned_closed_TSC_BPE_phase1_for_LoRA_v1'),
            ('acquisition_updates',0),('chosen_checkpoint_role','phase2_latest')):
            altered=copy.deepcopy(r);altered[key]=bad
            rejected(lambda altered=altered:h.validate_receipt(altered,'synthetic_rowLR',lr),'reject_'+str(lr)+'_'+key)
        altered=copy.deepcopy(r);altered['declared_row_lr']['source_row_lr']=lr
        rejected(lambda:h.validate_receipt(altered,'synthetic_rowLR',lr),'reject_source_LR_not_original1e4_'+str(lr))
    rejected(lambda:h.continued_acquisition_declaration(1e-4),'old_active_row_LR_rejected')
    rejected(lambda:h.continued_acquisition_declaration(3e-5),'LoRA_rate_cannot_be_row_rate')
    old=ast.parse((HERE/'tsc_bpe_lora_transition_v2.py').read_text(encoding='utf-8'))
    new=ast.parse(Path(h.__file__).read_text(encoding='utf-8'))
    of={n.name:n for n in old.body if isinstance(n,ast.FunctionDef)}
    nf={n.name:n for n in new.body if isinstance(n,ast.FunctionDef)}
    for name in ('validate_chosen_selection','compare_development'):
        check(ast.dump(of[name],include_attributes=False)==ast.dump(nf[name],include_attributes=False),
            name+'_actual300_and_chosen_gate_contract_unchanged')
    check(not any(n in sys.modules for n in ('torch','transformers','nemo')),'no_ML_imported')
    destination=HERE/'TSC_BPE_row_LR_boundary_CPU_verification_v1.json'
    report={'kind':'inactive_TSC_BPE_continued_row_LR_boundary_CPU_verification_v1','passed':True,
        'source_sha256':hashlib.sha256(Path(h.__file__).read_bytes()).hexdigest(),'checks':checks,'assertions':len(checks),
        'actual_source_checkpoint_read_or_admitted':False,'Torch_or_GPU_executed':False,
        'scope':'DeclaredLR/actualreceiptfunc/sourcephase1 and preserved saved300 comparator AST only'}
    with destination.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps({'passed':True,'assertions':len(checks),'sha256':hashlib.sha256(destination.read_bytes()).hexdigest()}))

if __name__=='__main__':main()
