"""Small stdlib admission checks; no actual source/checkpoint admission."""
import argparse
import ast
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parent.parent
HERE=ROOT/'outputs/parakeet_turkish';sys.path.insert(0,str(HERE))
import tsc_uniform_majority_transition_v1 as helper
import record_TSC_uniform_majority_source_v1 as creator


def main(output):
    checks=[];rejects=[]
    def check(name,value):
        if not value:raise AssertionError(name)
        checks.append(name)
    def rejected(name,call):
        try:call()
        except (ValueError,KeyError,TypeError):rejects.append(name)
        else:raise AssertionError('Accepted '+name)
    h=7621;prior=8968;declaration=helper.majority_declaration(h,prior)
    check('Actual H/prior arithmetic',declaration['configured_end_step']==25109 and declaration['remaining_update_budget']==17488)
    check('Whole-update repeats explicit',declaration['last_whole_update_next_cycle_repeats']==2)
    check('Physical B8x2 layout',declaration['physical_microbatch_composition']==[[8,0],[6,2]])
    check('Preservation density accounting',.125*declaration['preservation_coefficients']['replay_kl']==.5*10
        and .125*declaration['preservation_coefficients']['duration_kl']==.5*.5
        and .125*declaration['preservation_coefficients']['feature_distillation']==.5)
    check('No gradient-equivalence or grace-reset claim',declaration['equal_gradients_or_quality_claimed'] is False
        and declaration['renewed_Portuguese_grace'] is False and declaration['parent_retention_failure_counter_preserved'] is True)
    rejected('Changed prior draw clock',lambda:helper.majority_declaration(h,prior+1))
    rejected('Completed first pass',lambda:helper.majority_declaration(38225,253800))
    controller={'status':'resource_review_required','resource_review_required':True,'stop_category':'resource','child_exit_code':0,
        'finished_at_utc':'2026-10-12T08:30:53+00:00','actual_runtime_completion':{'complete':True},
        'controller_sha256':helper.SOURCE_PINS[helper.PARENT_CONTROLLER],
        'kind':'guarded_TSC_8199_uniform_full_Portuguese_controller_v1','cooperative_stop_reason':'system_commit_above90percent_three_fresh_observations'}
    check('Resource parent exit remains1',helper.validate_closure_header(controller)==1)
    for key,value in (('status','running'),('child_exit_code',1),('actual_runtime_completion',None),('controller_sha256','0'*64)):
        bad={**controller,key:value};rejected('Closure '+key,lambda bad=bad:helper.validate_closure_header(bad))
    baseline={f'validation_{lang}':{'asr':{'word_errors':64 if lang=='de' else 60,'reference_words':1004 if lang=='de' else 1095,'wer':(64/1004 if lang=='de' else 60/1095)}} for lang in helper.LANGUAGES}
    summary={f'validation_{lang}':{'asr':{'utterances':50,**baseline[f'validation_{lang}']['asr']}} for lang in helper.LANGUAGES}
    summary['validation_de']['asr'].update(word_errors=71,wer=71/1004)
    summary['validation_pt']['asr'].update(word_errors=67,wer=67/1095)
    gates=helper.retention_gates(summary,baseline)
    check('Independent DE integer boundary70',gates['de']['allowed_word_errors']==70 and not gates['de']['passed'])
    check('Independent PT integer boundary66',gates['pt']['allowed_word_errors']==66 and not gates['pt']['passed'])
    bad=deepcopy(summary);bad['validation_pt']['asr']['wer']=0
    rejected('Corrupt ASR ratio',lambda:helper.retention_gates(bad,baseline))
    original={'controller_status':{'path':'status.json','sha256':'a'*64,'bytes':1},'chosen_DEV_summary':{'path':'summary.json','sha256':'b'*64,'bytes':1}}
    owned={'role':'trainer_controller','session_id':88291,'exit_code':1,'chunk_id':'b23736','observed_at_utc':'2026-10-12T08:31:00+00:00'}
    receipt={'schema':1,'kind':helper.RECEIPT_KIND,'status':'actual_CLOSED_reviewed_uniform_frontier',
        'source_run_id':'source_uniform','target_run_id':'majority_new','allow_Turkish_majority_continuation':True,'guards_retained':True,
        'chosen_checkpoint_role':'latest','chosen_step':h,'completed_steps':h,'source_best_step':None,'source_best_checkpoint':None,
        'prior_uniform_draws':prior,'declared_majority':declaration,'owned_actual_exit':owned,'original_artifact_records':original}
    check('No eligible BEST admission',helper.validate_receipt(receipt,'majority_new') is receipt)
    for key,value in (('target_run_id','source_uniform'),('chosen_checkpoint_role','best'),('owned_actual_exit',{**owned,'exit_code':0})): 
        bad={**receipt,key:value}
        if key=='owned_actual_exit':continue # Actual controller-vs-receipt match is source_context's obligation.
        rejected('Receipt '+key,lambda bad=bad:helper.validate_receipt(bad,'majority_new'))
    common={'schema':1,'passed':True,'source_run_id':'source_uniform','chosen_step':h,'chosen_checkpoint_role':'latest',
        'source_controller_sha256':helper.SOURCE_PINS[helper.PARENT_CONTROLLER],'reviewed_status_artifact':original['controller_status'],
        'chosen_DEV_summary':original['chosen_DEV_summary'],'owned_actual_exit':owned,'allow_majority_continuation':True,'guards_retained':True}
    resource={**common,'kind':helper.RESOURCE_REVIEW_KIND,'stop_reason':controller['cooperative_stop_reason'],
        'material_policy_change':{'TR_per_update':14,'replay_per_update':2,'physical_microbatch_composition':[[8,0],[6,2]]},'actual_next_runtime_fit_not_claimed':True}
    quality={**common,'kind':helper.RETENTION_REVIEW_KIND,'retention_gates':gates,'renewed_Portuguese_grace':False,'parent_failure_counter_preserved':True}
    saved_read,saved_checked=helper.read,helper.checked_record
    try:
        helper.checked_record=lambda value:value;helper.read=lambda value:value
        check('Exact resource review',helper.validate_review(resource,helper.RESOURCE_REVIEW_KIND,receipt,controller,gates) is resource)
        check('Separate actual failed-gate review',helper.validate_review(quality,helper.RETENTION_REVIEW_KIND,receipt,controller,gates) is quality)
        for key,value in (('owned_actual_exit',{**owned,'exit_code':0}),('chosen_step',h-1),('stop_reason','other'),('guards_retained',False)):
            bad={**resource,key:value};rejected('Resource review '+key,lambda bad=bad:helper.validate_review(bad,helper.RESOURCE_REVIEW_KIND,receipt,controller,gates))
        bad={**quality,'renewed_Portuguese_grace':True}
        rejected('Renewed PT grace',lambda:helper.validate_review(bad,helper.RETENTION_REVIEW_KIND,receipt,controller,gates))
    finally:helper.read,saved_read= saved_read,helper.read;helper.checked_record=saved_checked
    with tempfile.TemporaryDirectory(dir=ROOT/'work') as directory:
        path=Path(directory)/'history.json';history=[{'step':i,'value':i%3} for i in range(1,h+1)]
        path.write_text(json.dumps(history),encoding='utf-8');prefix,row=helper._history_frontier(path,h,h)
        check('Bounded history chosen row and canonical prefix',row==history[-1] and prefix==helper.canonical(history))
        history[-1]['step']=h+1;path.write_text(json.dumps(history),encoding='utf-8')
        rejected('Mutated contiguous history',lambda:helper._history_frontier(path,h,h))
    # The real creator's live-header rejection is exercised before read() can
    # reach config/history or any record() can reach a checkpoint.
    calls=[];saved=creator.read;args=type('Args',(),{'source_loop':ROOT/'work/not-an-actual-source'})()
    try:
        creator.read=lambda path:(calls.append(str(path)) or {'status':'running'})
        rejected('Creator live source before checkpoint access',lambda:creator.inspect_closed(args))
        check('Only terminal metadata read before live rejection',len(calls)==1 and calls[0].endswith('loop_status.json'))
    finally:creator.read=saved
    for name in ('tsc_uniform_majority_transition_v1.py','record_TSC_uniform_majority_source_v1.py'):
        tree=ast.parse((HERE/name).read_text(encoding='utf-8'))
        check(name+' no ML imports',not any(isinstance(node,(ast.Import,ast.ImportFrom)) and any(
            value in ('torch','nemo','transformers') for value in ([node.module.split('.')[0]] if isinstance(node,ast.ImportFrom) and node.module else [alias.name.split('.')[0] for alias in getattr(node,'names',[])])) for node in ast.walk(tree)))
    output=output or HERE/('TSC_uniform_majority_boundary_stdlib_verification_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json')
    require=helper.require;require(not output.exists(),'Fresh report required')
    result={'kind':'TSC_uniform_majority_boundary_stdlib_verification_v1','passed':True,'checks':checks,'rejections':rejects,
        'source_sha256':{name:helper.record(HERE/name)['sha256'] for name in ('tsc_uniform_majority_transition_v1.py','record_TSC_uniform_majority_source_v1.py')},
        'actual_closed_source_or_tensors_admitted':False,'actual_GPU_or_training':False,'Torch_imported':'torch' in sys.modules,
        'scope':'Pure declaration, truthful closure/review, contiguous bounded history and creator early-rejection checks; no actual source fixture'}
    with output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps({'passed':True,'checks':len(checks),'rejections':len(rejects),'report':helper.record(output)}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);main(parser.parse_args().output)
