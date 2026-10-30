"""Focused stdlib declaration/admission guards; no actual CLOSED source admitted."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'outputs/parakeet_turkish'
sys.path.insert(0,str(HERE))
import tsc_bpe_lora_low_lr_transition_v1 as h

def main():
    checks=[]
    def check(ok,name):
        assert ok,name
        checks.append(name)
    def rejected(call,name):
        try:call()
        except (ValueError,KeyError,AttributeError):checks.append(name);return
        raise AssertionError('Invalid contract accepted: '+name)
    declaration=h.adaptation_declaration()
    check(declaration['encoder_LoRA_learning_rate']==3e-5 and declaration['source_encoder_LoRA_learning_rate']==3e-4
        and declaration['new_rows_learning_rate']==1e-4 and declaration['source_phase2_latest_used'] is False,
        'one_active_LR_change_source_moments_and_row_rate_distinct')
    receipt={'schema':1,'kind':h.RECEIPT_KIND,'status':'actual_CLOSED_qualified_phase1',
        'target_run_id':'synthetic_new_low_lr','source_run_id':'synthetic_phase1',
        'allow_new_rows_plus_encoder_LoRA':True,'source_phase2_latest_used':False,
        'declared_adaptation':declaration,'original_step':10,'acquisition_updates':2,
        'chosen_step':12,'completed_steps':12,'chosen_checkpoint_role':'latest',
        'retention_gate_passed_at_chosen_step':True,'owned_actual_exits':[{'role':'trainer_controller',
        'exit_code':0,'session_id':999,'chunk_id':'EXPLICIT_FIXTURE_NO_EXECUTION',
        'observed_at_utc':'fixture_only'}]}
    check(h.validate_receipt(receipt,'synthetic_new_low_lr') is receipt,'actual_new_receipt_validator_accepts_declaration_shape')
    for key,bad in (('kind','ROOT_owned_closed_TSC_BPE_phase1_for_LoRA_v1'),('source_phase2_latest_used',True),
        ('chosen_checkpoint_role','phase2_latest'),('acquisition_updates',0)):
        altered=copy.deepcopy(receipt);altered[key]=bad
        rejected(lambda altered=altered:h.validate_receipt(altered,'synthetic_new_low_lr'),'reject_'+key)
    for key,bad in (('encoder_LoRA_learning_rate',3e-4),('source_encoder_LoRA_learning_rate',3e-5),
        ('new_rows_learning_rate',3e-4),('original192_LoRA_moments_and_clock_resumed',False)):
        altered=copy.deepcopy(receipt);altered['declared_adaptation'][key]=bad
        rejected(lambda altered=altered:h.validate_receipt(altered,'synthetic_new_low_lr'),'reject_declaration_'+key)
    with tempfile.TemporaryDirectory(dir=ROOT/'work/research',prefix='low_lr_boundary_fixture_') as directory:
        temp=Path(directory);rp=temp/'opaque_receipt.json';rp.write_text('{}',encoding='utf-8')
        chosen=temp/'opaque_unused_chosen.pt'
        values={'resume_adaptation':chosen,'adaptation_source_receipt':rp,'run_id':'synthetic_new_low_lr',
            'lora_lr':3e-4,'head_lr':1e-5,'adapt_lora_lr':3e-5,'native_max_joint_bytes':1850000000,
            'adaptation_max_updates':250,'smoke_updates':None}
        keys=('tsc_input_proof','tsc_training_approval','source_closure','tr_manifest','replay_manifest',
            'data_v3_proof','tr_frame_metadata','tr_frame_metadata_proof','reuse_input_audit','supplement',
            'candidate_dir','candidate_tr_metadata','candidate_tr_metadata_proof')
        values.update({key:None for key in keys})
        context={'receipt':{'artifacts':{'chosen_checkpoint':{'path':str(chosen)}}},
            'source_recipe':{'lora_lr':3e-4,'head_lr':1e-5},'config':{key:None for key in keys}}
        original=h._context;h._context=lambda *args:context
        try:
            check(h.preflight(SimpleNamespace(**values)) is context,'actual_preflight_function_copies_dormant_recipe_and_new_active_rate')
            for key,bad in (('lora_lr',3e-5),('adapt_lora_lr',3e-4),('adaptation_max_updates',1001),
                ('resume_adaptation',temp/'wrong_unused_chosen.pt'),('resume_exact',temp/'unsupported.pt')):
                altered={**values,key:bad}
                rejected(lambda altered=altered:h.preflight(SimpleNamespace(**altered)),'reject_preflight_'+key)
        finally:h._context=original
    old_tree=ast.parse((HERE/'tsc_bpe_lora_transition_v2.py').read_text(encoding='utf-8'))
    new_tree=ast.parse(Path(h.__file__).read_text(encoding='utf-8'))
    old={n.name:n for n in old_tree.body if isinstance(n,ast.FunctionDef)}
    new={n.name:n for n in new_tree.body if isinstance(n,ast.FunctionDef)}
    for name in ('validate_phase1_state','validate_chosen_selection','compare_development'):
        check(ast.dump(old[name],include_attributes=False)==ast.dump(new[name],include_attributes=False),
            name+'_same_typed_internal_and_JSON_canonical_frontier_or_actual300_contract')
    check(not any(name in sys.modules for name in ('torch','transformers','nemo')),'no_ML_imported')
    report={'kind':'inactive_TSC_BPE_LoRA_low_lr_boundary_CPU_verification_v1','passed':True,
        'source_sha256':hashlib.sha256(Path(h.__file__).read_bytes()).hexdigest(),'checks':checks,
        'assertions':len(checks),'actual_CLOSED_source_admission':False,'actual_Torch_state_restoration':False,
        'actual_GPU_or_ASR':False,'scope':'Synthetic declaration/actual-function preflight guards and preserved producer/schema AST only'}
    dest=HERE/'TSC_BPE_LoRA_low_lr_boundary_CPU_verification_v1.json'
    with dest.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps({'passed':True,'checks':len(checks),'path':str(dest),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}))

if __name__=='__main__':main()
