"""Source/AST/hash preparation only. NO candidate import or proof operation."""
import ast,copy,hashlib,json,os
from pathlib import Path
F=Path(__file__).parent;FIX=F.parent
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish';R=P/'work/research'
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def write(p,v):
 with Path(p).open('x',encoding='utf-8',newline='\n')as out:json.dump(v,out,indent=2,ensure_ascii=False);out.write('\n')
assert os.environ['TORCH_EXTENSIONS_DIR'].lower().startswith('e:')
pending=[]
def begin(old,new):assert not new.exists();return old.read_bytes().decode(),[]
def edit(t,l,a,b,n=1):
 if '\n'in a and t.count(a.replace('\n','\r\n'))==n:a=a.replace('\n','\r\n');b=b.replace('\n','\r\n')
 assert t.count(a)==n,(a[:150],t.count(a),n);l.append({'old_literal':a,'new_literal':b,'expected_count':n});return t.replace(a,b)
def finish(old,new,t,l):
 ast.parse(t);inv=t
 for e in reversed(l):
  assert inv.count(e['new_literal'])==e['expected_count'];inv=inv.replace(e['new_literal'],e['old_literal'])
 assert inv.encode()==old.read_bytes();pending.append((old,new,t,l))
old=F/'verify_one_postDEV_refresh_finite_CPU_v1.py';new=F/'verify_one_postDEV_refresh_finite_CPU_v2.py';t,l=begin(old,new)
def f(a,b,n=1):
 global t;t=edit(t,l,a,b,n)
f('import argparse, ast, copy, gc, hashlib, json, os, queue, sys, threading, time, traceback, weakref',
  'import argparse, ast, copy, gc, hashlib, json, os, queue, random, sys, threading, time, traceback, weakref')
f('import asr18_one_postDEV_gc_refresh_v1 as h','import asr18_one_postDEV_gc_refresh_v2 as h')
anchor="        case('METHOD_STUB_independent_semantic_mismatch_is_fatal',compare_failure)\n"
extra='''        def changed_capture_value():
            s=StubHarness();m=h.RefreshSequence(0);count=0
            def capture_changed_frontier():
                nonlocal count;count+=1;value=science_METHOD_STUB()
                if count==2:value['clocks'][0]+=1
                return value
            result=expected_error(lambda:s.run(m,capture=capture_changed_frontier,compare=lambda a,b:None))
            require('Exact independent scientific fingerprint'in result['message']and not m.completed,
                'Actual helper canonical comparison rejects real changed capture value, not a throwing comparison double')
            return result
        case('METHOD_STUB_genuinely_changed_frontier_value_rejected_by_actual_helper',changed_capture_value)
        def changed_real_isolated_python_RNG():
            saved=random.getstate();count=0;s=StubHarness();m=h.RefreshSequence(0)
            try:
                def capture_changed_rng():
                    nonlocal count;count+=1
                    if count==2:random.random()
                    return dict(science_METHOD_STUB(),isolated_actual_Python_RNG=random.getstate())
                result=expected_error(lambda:s.run(m,capture=capture_changed_rng,compare=lambda a,b:None))
                require('Exact independent scientific fingerprint'in result['message'],
                    'Real fixture RNG change reaches actual equality rejection')
                return dict(result,production_RNG_read_or_changed=False,real_isolated_fixture_RNG_change=True)
            finally:random.setstate(saved)
        case('actual_isolated_Python_RNG_changed_value_negative_with_restoration',changed_real_isolated_python_RNG)
        from finite_actual_capture_bridge_METADATA_DOUBLES_v1 import (check_actual_capture_and_compare_sources,
            check_actual_periodic_eligibility_branch)
        case('actual_capture_and_compare_source_executed_under_explicit_207_threeAdam215_METADATA_DOUBLES',
            check_actual_capture_and_compare_sources)
        case('actual_source36_first_opportunity_and_original_terminal_AST_under_METADATA_DOUBLES',
            lambda:check_actual_periodic_eligibility_branch(h,P/'work/research/run_ASR18_BPE9223_resident_RAM_training_v36.py'))
'''
f(anchor,anchor+extra)
anchor="            m=h.OnePostDEVRefresh(0)\n            result=m.execute_real(barrier(),capture=science_METHOD_STUB,\n"
f(anchor,"            m=h.OnePostDEVRefresh(0)\n            actual_isolated_python_rng_before=random.getstate()\n            def capture_real_fixture_rng():\n                return dict(science_METHOD_STUB(),isolated_actual_Python_RNG=random.getstate())\n            result=m.execute_real(barrier(),capture=capture_real_fixture_rng,\n")
anchor="            require(result['passed']and observation.take_window()['calls']==0,\n"
f(anchor,"            actual_isolated_python_rng_after=random.getstate()\n            require(actual_isolated_python_rng_before==actual_isolated_python_rng_after,\n                'Real isolated Python RNG state exactly unchanged across actual CPython policy event')\n"+anchor)
anchor="                          'frozen_cycle_retention_risk_genuinely_observed':True,'maximum_simultaneous_toy_workers':ToyIdleWorker.maximum_active})"
f(anchor,"                          'frozen_cycle_retention_risk_genuinely_observed':True,'maximum_simultaneous_toy_workers':ToyIdleWorker.maximum_active,\n                          'actual_isolated_python_rng_before':actual_isolated_python_rng_before,\n                          'actual_isolated_python_rng_after':actual_isolated_python_rng_after,\n                          'real_isolated_Python_RNG_before_after_exact':True,'production_RNG_read_or_changed':False})")
finish(old,new,t,l)

old=F/'launch_one_postDEV_refresh_finite_CPU_v1.py';new=F/'launch_one_postDEV_refresh_finite_CPU_v2.py';t,l=begin(old,new)
def x(a,b,n=1):
 global t;t=edit(t,l,a,b,n)
x('verify_one_postDEV_refresh_finite_CPU_v1.py','verify_one_postDEV_refresh_finite_CPU_v2.py')
x('one_postDEV_refresh_finite_CPU_external_result_v1.json','one_postDEV_refresh_finite_CPU_external_result_v2.json')
x('one_postDEV_refresh_finite_CPU_stdout.log','one_postDEV_refresh_finite_CPU_v2_stdout.log')
x('one_postDEV_refresh_finite_CPU_stderr.log','one_postDEV_refresh_finite_CPU_v2_stderr.log')
finish(old,new,t,l)
bridge=F/'finite_actual_capture_bridge_METADATA_DOUBLES_v1.py';ast.parse(bridge.read_text())
for old,new,t,l in pending:new.write_bytes(t.encode())
records=[{'old':pin(old),'new':pin(new),'replacements':l,'full_RAW_forward_inverse_exact':True,
          'newline_policy':'Actual literal replacements preserve original RAW endings'}for old,new,t,l in pending]
ledger=F/'finite_fixture_external_v1_to_v2_exact_RAW_inverse_v1.json'
write(ledger,{'kind':'ASR18_INACTIVE_postDEV_refresh_finite_fixture_external_v2_RAW_inverse_v1',
 'sources':records,'new_capture_bridge_metadata_doubles':pin(bridge),
 'frozen_v1_unexecuted':True,'actual_candidate_import_CPU_GC_or_production_execution':False})
schema=json.loads((F/'required_completed_CPU_review_schema_v1.json').read_text())
schema['helper_source']=pin(F/'asr18_one_postDEV_gc_refresh_v2.py')
schema['actual_capture_compare_source_with_metadata_doubles_required']=True
schema['actual_isolated_Python_RNG_event_equality_required']=True
schema['actual_changed_capture_value_negative_required']=True
sp=F/'required_completed_CPU_review_schema_v2.json';write(sp,schema)
oldhandoff=json.loads((F/'inactive_one_postDEV_refresh_finite_CPU_source_handoff_v1.json').read_text())
inputs=[]
for value in oldhandoff['input_pins']:
 p=Path(value['path'])
 replace_names={'verify_one_postDEV_refresh_finite_CPU_v1.py':'verify_one_postDEV_refresh_finite_CPU_v2.py',
 'launch_one_postDEV_refresh_finite_CPU_v1.py':'launch_one_postDEV_refresh_finite_CPU_v2.py',
 'asr18_one_postDEV_gc_refresh_v1.py':'asr18_one_postDEV_gc_refresh_v2.py',
 'required_completed_CPU_review_schema_v1.json':'required_completed_CPU_review_schema_v2.json',
 'finite_CPU_external_exact_raw_inverse_v1.json':'finite_fixture_external_v1_to_v2_exact_RAW_inverse_v1.json'}
 if p.name in replace_names:p=p.with_name(replace_names[p.name])
 inputs.append(pin(p))
more=[bridge,F/'ROOT_source35_and_finite_v1_preexecution_findings_and_successor_preparation_scope_v1.json',
 F/'helper2_source35_to36_and_full33_RAW_inverse_v1.json',F/'inactive_source36_one_postDEV_refresh_source_handoff_v1.json',
 F/'proposed_source36_configuration_pending_CPU_and_actual_stop_v1.json',
 R/'run_ASR18_BPE9223_resident_RAM_training_v36.py',R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v36.py',
 S/'asr18_one_postDEV_gc_refresh_checkpoint_transition_v2.py']
for p in more:
 value=pin(p)
 if value not in inputs:inputs.append(value)
oldhandoff.update(kind='ASR18_INACTIVE_one_postDEV_refresh_finite_CPU_source_handoff_v2',
 fixture_source=records[0]['new'],launcher_source=records[1]['new'],exact_raw_inverse=pin(ledger),
 required_completed_CPU_review_schema=pin(sp),input_pins=inputs,
 actual_capture_bridge_doubles_source=pin(bridge),
 external_result_path=str(F/'one_postDEV_refresh_finite_CPU_external_result_v2.json'),
 exact_future_external_arguments=['-Xutf8','-B',str(F/'launch_one_postDEV_refresh_finite_CPU_v2.py'),
  '--ROOT-review','FUTURE_EXACT_ROOT_FINITE_CPU_V2_SCOPE_REQUIRED'],
 actual_first_observation_namespace_preserved_fresh_no_v1_job_ever_executed=True,
 new_cases=['real isolated Python RNG before/after actual CPython event',
  'actual helper rejects genuinely changed frontier/RNG capture values, no throwing compare',
  'actual capture and compare source executed with explicit207/threeAdam215 metadata/tensor/fingerprint doubles',
  'actual source36 first-opportunity/original terminal AST cleaner/budget/validation/weak missing/dead negatives'],
 preexecution_successor_scope=pin(F/'ROOT_source35_and_finite_v1_preexecution_findings_and_successor_preparation_scope_v1.json'))
hp=F/'inactive_one_postDEV_refresh_finite_CPU_source_handoff_v2.json';write(hp,oldhandoff)
print(json.dumps({'fixture':records[0]['new'],'external':records[1]['new'],'metadata_bridge':pin(bridge),
 'RAW_inverse':pin(ledger),'schema':pin(sp),'handoff':pin(hp),'input_count':len(inputs)}))
