"""Tiny synthetic metadata/source fixture; no jobs, real state, ML or GPU."""
from __future__ import annotations
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'outputs/parakeet_turkish'

def main():
    path=HERE/'run_TSC_shared_head_memory_queue_v1.py'
    spec=importlib.util.spec_from_file_location('shared_queue_fixture',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    checks=[];rejections=0
    def check(ok,name):
        assert ok,name
        checks.append(name)
    def reject(call,name):
        nonlocal rejections
        try:call()
        except (ValueError,KeyError):rejections+=1;checks.append(name)
        else:raise AssertionError(name)
    executor='e'*64;controller='c'*64
    receipt={'schema':1,'kind':'ROOT_owned_closed_queue_for_shared_heads_v1','status':'actual_CLOSED_reviewed',
        'allow_next_shared_head_queue':True,'next_queue_id':'synthetic_only','next_queue_script_sha256':executor,
        'next_controller_sha256':controller,'guards_retained':True,'allocator_cap_bytes':8000000000,
        'completed_update_cache_threshold_bytes':3000000000,'predecessor_state':{'path':'SYNTHETIC_NOT_READ.json','sha256':'a'*64},
        'owned_actual_exit':{'session_id':123,'exit_code':0,'tool_chunk':'abcdef','observed_at_utc':'2026-10-12T01:00:01Z'}}
    prior={'status':'trials_complete_review_required','owned_child_pid':None,'finished_at_utc':'2026-10-12T01:00:00Z'}
    validate=lambda r,p,review=None:m.validate_predecessor_metadata(r,p,queue_id='synthetic_only',executor_sha=executor,controller_sha=controller,resource_review=review)
    check(validate(receipt,prior),'ROOT CLOSED0 diagnostic successor metadata admitted')
    r1=copy.deepcopy(receipt);r1['owned_actual_exit']['exit_code']=1
    p1={**prior,'status':'repair_required','owned_controller_closure_observed':True}
    review={'kind':'ROOT_closed_queue_resource_repair_for_shared_heads_v1','passed':True,'resource_review_cleared':True,
        'unchanged_guard_thresholds':True,'predecessor_state_sha256':'a'*64,'predecessor_owned_exit':r1['owned_actual_exit'],
        'next_controller_sha256':controller,'actual_native_complete_and_sources_unchanged':True,'actual_owned_controller_child_exit_code':0}
    check(validate(r1,p1,review),'explicit hash-bound CLOSED1 reviewed resource metadata admitted')
    reject(lambda:validate(r1,p1),'CLOSED1 without ROOT resource review rejected')
    reject(lambda:validate(receipt,{**prior,'status':'running'}),'running predecessor rejected')
    reject(lambda:validate(receipt,{**prior,'owned_child_pid':42}),'known unclosed child rejected')
    reject(lambda:validate({**receipt,'next_controller_sha256':'d'*64},prior),'wrong successor source rejected')
    reject(lambda:validate(r1,{**p1,'owned_controller_closure_observed':False},review),'unobserved resource controller closure rejected')
    reject(lambda:validate(r1,p1,{**review,'predecessor_state_sha256':'b'*64}),'unbound resource review rejected')
    row={'step':7001,'adaptation_updates_completed':1,'source_phase1_step':7000,'original_source_selected_step':6500,
        'selected_acquisition_updates':500,'row_Adam_clock':501,'shared_Adam_clock':6501,'row_lr':3e-4,
        'shared_lr':1e-5,'head_lr':1e-5,'lora_lr':0.,'new_token_update_multiplier':1.,
        'old_six_character_update_multiplier_applied':False}
    m.validate_update_marker(row,1,3e-4,1e-5);checks.append('localN versus K+N and S+N actual LR marker accepted')
    for key,bad in [('row_Adam_clock',1),('shared_Adam_clock',7001),('lora_lr',3e-4),('row_lr',1e-3),('old_six_character_update_multiplier_applied',True)]:
        reject(lambda key=key,bad=bad:m.validate_update_marker({**row,key:bad},1,3e-4,1e-5),'marker mismatch rejected '+key)
    check(m.materially_improving_latest({'retention_passed':True,'TR_WER':.38},.39,.005),'qualifying material latest pauses controls')
    check(not m.materially_improving_latest({'retention_passed':False,'TR_WER':.30},.39,.005),'retention failure cannot enter fullpass')
    check(not m.materially_improving_latest({'retention_passed':True,'TR_WER':.388},.39,.005),'subthreshold change does not claim material gain')
    check(not m.materially_improving_latest({'retention_passed':True,'TR_WER':.40},.39,.005),'regression cannot enter fullpass')
    current=ast.parse(path.read_text(encoding='utf-8'));parent=ast.parse((HERE/'run_TSC_row_schedule_memory_queue_v3.py').read_text(encoding='utf-8'))
    functions=lambda t:{n.name:n for n in t.body if isinstance(n,ast.FunctionDef)}
    a,b=functions(current),functions(parent)
    for name in ('cooperative_stop_path','request_owned_stop','close_owned_after_failure','execute_owned'):
        check(ast.dump(a[name],include_attributes=False)==ast.dump(b[name],include_attributes=False),'frozen owned cleanup unchanged '+name)
    text=path.read_text(encoding='utf-8')
    check("('shared1e5',1e-5),('shared3e6',3e-6)" in text and "manifest['row_lr'] == 3e-4" in text,'only explicit ordered shared controls admitted')
    check("--adaptation-max-updates', '1000'" in text and "'--row-lr', str(row_lr), '--shared-lr', str(shared_lr)" in text,'real command uses worker phase/LR CLI')
    check('second_control_not_started=True' in text and "status='competitive_latest_full_pass_review_required'" in text,'heartbeat pause state before second control')
    check('historical_peak = 12_176_424_960' in text and 'projected / memory[\'commit_limit_bytes\'] < .95' in text,'unchanged historical-private critical projection')
    check('torch' not in sys.modules and 'trackio' not in sys.modules,'no ML or logger imported')
    report={'schema':1,'kind':'inactive_shared_head_memory_queue_tiny_metadata_verification_v1','passed':True,
        'source':{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()},'checks':checks,'assertions':len(checks),'rejections':rejections,
        'scope':'Only synthetic receipt/update/decision metadata plus source AST; no actual predecessor/source/STATE/namespace/model/checkpoint/Trackio/GPU reads or writes.',
        'actual_predecessor_admitted':False,'actual_child_or_queue_launched':False,'runtime_fit_or_tensor_proof':False}
    out=HERE/'TSC_shared_head_memory_queue_tiny_metadata_verification_v1.json'
    with out.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({'source_sha256':report['source']['sha256'],'checks':len(checks),'rejections':rejections,
        'report':str(out),'report_sha256':hashlib.sha256(out.read_bytes()).hexdigest()}))

if __name__=='__main__':main()
