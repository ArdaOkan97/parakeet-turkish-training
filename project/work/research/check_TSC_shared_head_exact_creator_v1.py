"""Tiny stdlib schema/frontier checks only; no real source, state or ML admission."""
from pathlib import Path
from datetime import datetime
import ast
import copy
import hashlib
import importlib.util
import json
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[2]; HERE=ROOT/'outputs/parakeet_turkish'
sys.path.insert(0,str(HERE))
name='record_TSC_BPE_shared_head_exact_source_v1'
spec=importlib.util.spec_from_file_location(name,HERE/(name+'.py'))
writer=importlib.util.module_from_spec(spec);spec.loader.exec_module(writer)
checks=[];rejects=[]

def ok(condition,label):
    if not condition:raise AssertionError(label)
    checks.append(label)

def bad(call,label):
    try:call()
    except (ValueError,KeyError,TypeError):rejects.append(label);checks.append(label);return
    raise AssertionError('Unexpected acceptance: '+label)

header={'status':'train_window_ended','child_exit_code':0,'resource_review_required':False,
    'phase':writer.PHASE,'stop_category':'phase_bound','finished_at_utc':'2026-10-12T06:00:00Z',
    'actual_runtime_completion':{'passed':True},'smoke_updates':None,'run_id':'fixture_shared_01'}
ok(writer.validate_closure_header(header)=='fixture_shared_01','actual header schema only accepted')
operational={**header,'phase':{'phase':'development_evaluation','inferred_from_bounded_log_tail':True}}
ok(writer.validate_closure_header(operational)=='fixture_shared_01','actual producer operational phase dict admitted; scientific config validated separately')
for source in (writer.FIRST_CONTROLLER,writer.EXACT_CONTROLLER):
    source_tree=ast.parse((HERE/source).read_text(encoding='utf-8'))
    updates=[node for node in ast.walk(source_tree) if isinstance(node,ast.Call)
        and isinstance(node.func,ast.Attribute) and ast.unparse(node.func)=='state.update'
        and any(key.arg=='phase' and isinstance(key.value,ast.Name) and key.value.id=='phase' for key in node.keywords)]
    observations=[node for node in ast.walk(source_tree) if isinstance(node,ast.Call)
        and isinstance(node.func,ast.Attribute) and node.func.attr=='log_phase']
    ok(bool(updates and observations),'actual '+source+' publishes operational phase dict')
for key,value in [('status','training'),('child_exit_code',1),('resource_review_required',True),
                  ('stop_category','retention'),('stop_category','manual'),('smoke_updates',1),('actual_runtime_completion',None)]:
    changed={**header,key:value};bad(lambda value=changed:writer.validate_closure_header(value),'reject '+key+' '+str(value))
s,k,n=1,1,2;step=s+k+n
phase={'original_source_selected_step':s,'selected_acquisition_updates':k,'source_phase1_step':s+k,
    'active_parameter_tensors':15,'active_shared_head_tensors':12,'frozen_encoder_LoRA_tensors':192,
    'row_lr':3e-4,'shared_lr':1e-5,'further_unfreeze_allowed':False,'effective_max_updates':1000}
config={'run_id':header['run_id'],'phase':writer.PHASE,'config_identity':{'phase':writer.PHASE,'adaptation_phase':phase},
    'smoke_updates':None,'steps':44958,'initial_adaptation_turkish_wer':.39}
selection={'run_id':header['run_id'],'phase':writer.PHASE,'completed_steps':step,'selected_step':step,
    'adaptation_updates_completed':n,'source_phase1_step':s+k,'original_source_selected_step':s,
    'selected_acquisition_updates':k,'row_Adam_clock':k+n,'shared_Adam_clock':s+n}
closed={**header,'completed_steps':step,'selected_step':step,'adaptation_updates_completed':n}
history=[{'step':1},{'step':2}]+[{'step':s+k+local,'adaptation_updates_completed':local,'row_Adam_clock':k+local,
    'shared_Adam_clock':s+local,'row_lr':3e-4,'shared_lr':1e-5,'lora_lr':0.} for local in range(1,n+1)]
ok(writer.validate_frontier(config,selection,history,closed)==(s,k,n,step,step),'S/K/N frontier and two Adam clocks accepted')
for key,value in [('row_Adam_clock',4),('shared_Adam_clock',4),('lora_lr',3e-4),('row_lr',1e-3)]:
    changed=copy.deepcopy(history);changed[-1][key]=value
    bad(lambda rows=changed:writer.validate_frontier(config,selection,rows,closed),'reject changed frontier '+key)
changed=copy.deepcopy(config);changed['config_identity']['adaptation_phase']['active_parameter_tensors']=195
bad(lambda:writer.validate_frontier(changed,selection,history,closed),'reject shared-versus-LoRA phase confusion')
changed=copy.deepcopy(config);changed['phase']='wrong_scientific_phase'
bad(lambda:writer.validate_frontier(changed,selection,history,closed),'reject wrong scientific config phase independently of operational phase')
bad(lambda:writer.validate_frontier(config,selection,history[:-1],closed),'reject incomplete inherited history')
summary={'validation_'+lang:{'asr':{'utterances':50,'word_errors':1,'reference_words':100,'wer':.01}} for lang in writer.LANGUAGES}
ok(writer.competitive_latest(summary,{'retention_gate_passed':True,'turkish_wer':.01},.39)['tr']['wer']==.01,'strict improved latest and original gates accepted')
bad(lambda:writer.competitive_latest(summary,{'retention_gate_passed':False,'turkish_wer':.01},.39),'reject failed latest gate')
bad(lambda:writer.competitive_latest(summary,{'retention_gate_passed':True,'turkish_wer':.01},.005),'reject regressed latest despite all-five')
changed=copy.deepcopy(summary);changed['validation_de']['asr']['word_errors']=71
bad(lambda:writer.competitive_latest(changed,{'retention_gate_passed':True,'turkish_wer':.01},.39),'reject unchanged German gate breach')
tree=ast.parse((HERE/'tsc_bpe_shared_head_exact_transition_v1.py').read_text(encoding='utf-8'))
validate_node=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='validate_receipt')
namespace={'require':writer.require,'RECEIPT_KIND':writer.RECEIPT_KIND,'datetime':datetime}
exec(compile(ast.Module(body=[validate_node],type_ignores=[]),str(HERE/'tsc_bpe_shared_head_exact_transition_v1.py'),'exec'),namespace)
receipt={'schema':1,'kind':writer.RECEIPT_KIND,'status':'actual_CLOSED_complete_shared_head_boundary',
    'run_id':'fixture_shared_01','target_run_id':'fixture_shared_01','allow_same_phase_exact_resume':True,
    'guards_retained':True,'further_unfreeze_allowed':False,'original_step':s,'source_acquisition_updates':k,
    'shared_head_updates':n,'completed_steps':step,'approved_total_shared_head_updates':10,
    'chosen_checkpoint_role':'latest','chosen_step':step,'latest_DEV_step':step,'competitive_qualifying_latest_approved':True,
    'owned_actual_exits':[{'role':'trainer_controller','session_id':1,'exit_code':0,'tool_chunk':'abc','observed_at_utc':'2026-10-12T06:01:00Z'}],
    'source_stop_category':'phase_bound'}
ok(namespace['validate_receipt'](receipt,'fixture_shared_01') is receipt,'actual consumer receipt field schema accepted')
for key,value in [('approved_total_shared_head_updates',n),('completed_steps',step+1),('chosen_checkpoint_role','best'),
                  ('source_stop_category','retention'),('competitive_qualifying_latest_approved',False)]:
    changed={**receipt,key:value};bad(lambda value=changed:namespace['validate_receipt'](value,'fixture_shared_01'),'consumer rejects '+key)
with tempfile.TemporaryDirectory(prefix='shared_exact_creator_',dir=ROOT/'work/research') as directory:
    loop=Path(directory);(loop/'loop_status.json').write_text(json.dumps({**header,'status':'training'}),encoding='utf-8')
    args=writer.parser().parse_args(['--source-loop',str(loop),'--approved-total-shared-head-updates','1000',
        '--owned-session-id','1','--owned-exit-chunk','abc','--owned-exit-observed-at','2026-10-12T06:01:00Z','--validate-only'])
    before=writer.archive_ops.CHECKPOINT_BYTE_READS
    bad(lambda:writer.make_receipt(args),'actual writer rejects synthetic running header before source bytes')
    ok(writer.archive_ops.CHECKPOINT_BYTE_READS==before==0,'checkpoint byte reads remain zero')
ok(writer.IMMUTABLE_TOKENS==('original_native_ready','initial_source_equivalence'),'original ready and source300 paths remain scientific inputs')
ok(writer.record(writer.archive_ops.__file__)['sha256']==writer.ARCHIVE_OPS_SHA,'borrowed immutable byte archive operations source matches')
ok('torch' not in sys.modules and 'trackio' not in sys.modules,'no ML/logger imported')
report={'schema':1,'kind':'inactive_shared_head_exact_creator_stdlib_verification_v1','passed':True,
    'source':writer.record(writer.__file__),'checks':checks,'assertions':len(checks),'rejections':len(rejects),
    'scope':'Synthetic small JSON metadata and actual pure admission methods only. No actual source/PT/archive/process/ML/GPU admission.',
    'checkpoint_byte_reads':writer.archive_ops.CHECKPOINT_BYTE_READS,'actual_source_admitted':False,'GPU_launched':False}
output=HERE/'TSC_BPE_shared_head_exact_creator_stdlib_verification_v2.json'
with output.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2,ensure_ascii=False);stream.write('\n')
print(json.dumps({'passed':True,'assertions':len(checks),'rejections':len(rejects),'source':report['source'],'report':writer.record(output)}))
