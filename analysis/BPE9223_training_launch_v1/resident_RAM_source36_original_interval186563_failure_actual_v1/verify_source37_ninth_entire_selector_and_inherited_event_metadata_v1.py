"""INACTIVE metadata fixture. Separate exact ROOT scope required before execution.

Real new typed validators, complete driver factory/postrestore dictionaries,
wrapper entry/forwarding and actual inherited-consumed setup are exercised.
Journal/restore/GPU/scientific owners are explicit METHOD_STUBs. No live GC,
capture/sync, audio, model, tensor, original suffix or plan/payload reads.
"""
import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import types

import psutil

P = Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20')
S = P / 'outputs/parakeet_turkish'
R = P / 'work/research'
F = Path(__file__).parent
sys.path[:0] = [str(S), str(R), str(F)]
from resource_watchdog_v2 import sample_memory

BEGIN = time.monotonic()
OUTPUT = F / 'source37_ninth_typed_selector_actual_01'
SCOPE = 'ASR18_source37_ninth_original_failure_entire_selector_metadata_CPU_v1'


def require(v, m):
    if not v:
        raise ValueError(m)


def pin(p):
    p = Path(p).resolve()
    require(p.stat().st_size <= 8388608, 'Compact source/metadata only')
    b = p.read_bytes()
    return {'path': str(p), 'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()}


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def write(p, v):
    with Path(p).open('x', encoding='utf-8', newline='\n') as h:
        json.dump(v, h, indent=2)
        h.write('\n')
        h.flush()
        os.fsync(h.fileno())


def identity(p=None):
    p = psutil.Process() if p is None else p
    return {'pid': p.pid, 'create_time': p.create_time(), 'ppid': p.ppid()}


def guard():
    v = sample_memory(os.getpid(), 'source37_ninth_inherited_metadata_CPU')
    m = v['system_memory']
    require(v['paired_sampling_within_one_second'] and v['worker_private_memory']['private_bytes'] < 536870912
            and time.monotonic() - BEGIN < 100 and m['commit_available_bytes'] >= 2268435456
            and m['physical_available_bytes'] >= 1768435456 and shutil.disk_usage('E:/').free >= 100000000000,
            'Original metadata100s/512MiB/FULLprefetch/runtime floors/E100GB')
    return v


def expect(kind, call):
    try:
        call()
    except kind as error:
        return {'type': type(error).__name__, 'message': str(error)}
    raise AssertionError('Required rejection did not occur: ' + str(kind))


def execute(node, scope):
    exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(node)], type_ignores=[])),
                 '<EXACT_FROZEN_DRIVER_OR_WRAPPER_METADATA_AST>', 'exec'), scope)


def expression(node, scope):
    return eval(compile(ast.fix_missing_locations(ast.Expression(copy.deepcopy(node))),
                        '<ENTIRE_REAL_POSTRESTORE_DICTIONARY_METADATA_ONLY>', 'eval'), scope)


def assignment(nodes, name):
    return next(n for n in nodes if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))


class PrefixViewDouble:
    """No original journal read/hash, no actual checkpoint restore."""
    def __init__(self, saved, guard):
        self.saved = copy.deepcopy(saved)
    def binding(self):
        return copy.deepcopy(self.saved)


class JournalBindingDouble:
    """Only prospective new26 metadata; no file, namespace, copy or native authority."""
    def __init__(self, value):
        self.value = copy.deepcopy(value)
    def binding(self):
        return copy.deepcopy(self.value)


class ForbiddenNamespace:
    def writer(self, *a, **k):
        raise AssertionError('No original/future journal writer may be created')


def forbidden(*a, **k):
    raise AssertionError('No GC/capture/sync/restore/copy/model/state/draw/eligible or ineligible event action')


def main(a):
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '-1' and os.environ.get('PYTHONUTF8') == '1'
            and all(os.environ.get(k) == '1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'))
            and Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad')),
            'Exact process-local E CPU-only limits')
    review = read(a.ROOT_review)
    require(review['accepted'] is True and review['reviewer'] == '/root' and review['scope'] == SCOPE
            and review['fixture_source'] == pin(__file__), 'Separate exact ONE ROOT CPU scope')
    for item in review['input_pins']:
        require(pin(item['path']) == item, 'Frozen source/compact metadata before proof')
    require(not OUTPUT.exists(), 'Fresh immutable ONE actual metadata output')
    OUTPUT.mkdir()
    entry = {'producer': identity(), 'launcher': identity(psutil.Process().parent()), 'ROOT_review': pin(a.ROOT_review)}
    write(OUTPUT / 'entry.json', entry)
    results = {}
    # Persist numeric sample BEFORE this admission predicate and candidate import.
    initial = sample_memory(os.getpid(), 'source37_ninth_pre_predicate_metadata_admission')
    write(OUTPUT / 'initial_numeric_resource_sample_before_predicate.json', initial)
    require(initial['paired_sampling_within_one_second']
            and initial['system_memory']['commit_available_bytes'] >= 5055306368
            and initial['system_memory']['physical_available_bytes'] >= 2305306368
            and initial['worker_private_memory']['private_bytes'] < 536870912
            and shutil.disk_usage('E:/').free >= 100000000000, 'FULL225+prefetch+prospective512MiB unchanged')
    try:
        configuration = read(review['source_configuration']['path'])
        closure = review['ninth_original_resume']
        pointer = review['initial_checkpoint']
        branch = Path(review['branch'])
        import run_ASR18_BPE9223_resident_RAM_training_v37 as driver
        import asr18_ninth_original_interval_checkpoint_resume_v1 as lane
        import fastmss_resident_feature_stream_v14 as CPU
        require('torch' not in sys.modules and 'numpy' not in sys.modules, 'No scientific/framework imports')
        driver.checked_native_commit_pressure_policy(configuration)
        CPU.checked_resource_profile(configuration)
        CPU.checked_packet_credit_policy(configuration)
        lane.checked_ninth_original_policy(configuration)
        proper36 = lane.legacy_source36_policy_configuration(configuration)
        from asr18_one_postDEV_gc_refresh_checkpoint_transition_v2 import legacy_source33_configuration
        # Reuse the closed finite proof. Do not reopen its actual CPU report via
        # the unchanged old checked_one_postDEV_refresh_policy record() guard.
        # The real original helper constructor is operation-free at import.
        sys.path.insert(0,str(lane.EVENT_FOLDER))
        from asr18_one_postDEV_gc_refresh_v2 import OnePostDEVRefresh
        require(configuration['one_postDEV_gc_refresh_policy']==read(lane.BASE)['one_postDEV_gc_refresh_policy'],
                'Original helper/capture/proof policy byte values preserved; closed CPU proof not rerun or reread')
        legacy33 = legacy_source33_configuration(proper36)
        require(legacy33['original_clip_geometry_overlay'] == read(lane.BASE)['original_clip_geometry_overlay'],
                'Unchanged legacy policy proof uses proper source36, not future runtime7')
        actual = lane.checked_ninth_original_resume(closure, base=configuration, folder=branch, pointer_pin=pointer)
        results['actual_new_policy_and_failure_ROOT_reuse_validators'] = {
            'passed': True, 'saved': actual['saved_optimizer_updates'], 'actual': actual['actual_optimizer_updates'],
            'historical_redo_count': len(actual['preserved_tail_positive_updates']),
            'closed_tail_and_event_never_reopened': actual['closed_ROOT_receipts_reused_no_tail_or_event_read']}
        tree = ast.parse(Path(driver.__file__).read_text(encoding='utf-8-sig'))
        mainnode = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        names = ['prior_process','gc_refresh_resume','eighth_original_resume','fresh_signature_resume','canonical_archive_resume',
                 'lookahead_resume','cushion225_resume','postrestore_selector_resume','packet_credit_resume','native_headroom_resume',
                 'abnormal_resume','source_transition_resume','original_interval_resume','DEV_frontend_resume',
                 'DEV_admission_resume','periodic_quiescence_resume']
        args = types.SimpleNamespace(**{k: None for k in names}, ninth_original_resume=closure['path'], invocation=26,
                                     start_invocation=26, stream_configuration=review['source_configuration']['path'],
                                     configuration=review['source_configuration']['path'])
        scope = {'a': args, 'base': configuration, 'folder': branch, 'pointer_pin': pointer, 'record': driver.record,
                 'require': require, 'checked_ninth_original_resume': lane.checked_ninth_original_resume}
        entry_assignment = assignment(mainnode.body, 'ninth_interval_failure')
        entry_if = next(n for n in mainnode.body if isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
                        and isinstance(n.test.left, ast.Name) and n.test.left.id == 'ninth_interval_failure')
        execute(entry_assignment, scope)
        execute(entry_if, scope)
        require(scope['ninth_interval_failure'] == actual, 'Actual complete typed entry')
        scope.update(eighth_interval_failure=None, checked_original_interval_resume=forbidden)
        execute(assignment(mainnode.body, 'interval_failure'), scope)
        require(scope['interval_failure'] == actual, 'Actual shared interval selector')
        common = {k: None for k in ('gc_transition','signature_transition','canonical_transition','lookahead_transition',
                                   'abnormal','transition','frontend_failure','DEV_admission_failure','periodic_quiescence_failure')}
        scope.update(common)
        execute(assignment(mainnode.body, 'preserved_tail'), scope)
        require(scope['preserved_tail'] == actual, 'Actual saved-prefix branch binds current ninth tail')
        results['entire_actual_driver_entry_shared_selector_and_preserved_lane'] = {'passed': True,
            'entry_assignment_AST': ast.dump(entry_assignment, include_attributes=False),
            'entry_if_AST': ast.dump(entry_if, include_attributes=False)}
        bad_args = copy.deepcopy(args); bad_args.invocation = 25
        results['wrong_initial_invocation_rejected'] = expect(ValueError, lambda: execute(entry_if, {**scope,'a':bad_args}))
        bad_args = copy.deepcopy(args); bad_args.gc_refresh_resume = 'unselected_historical_stop'
        results['old_gc_lane_mix_rejected'] = expect(ValueError, lambda: execute(entry_if, {**scope,'a':bad_args}))
        bad = copy.deepcopy(pointer); bad['sha256'] = '0' * 64
        results['wrong_checkpoint_rejected'] = expect(ValueError, lambda: lane.checked_ninth_original_resume(closure,base=configuration,folder=branch,pointer_pin=bad))
        bad = copy.deepcopy(closure); bad['sha256'] = '0' * 64
        results['stale_closure_rejected'] = expect(ValueError, lambda: lane.checked_ninth_original_resume(bad,base=configuration,folder=branch,pointer_pin=pointer))
        oldbase = read(lane.BASE)
        for name, mutate in [
            ('None_actual_CPU_gate', lambda v:v['original_clip_geometry_overlay'].__setitem__('ROOT_review', None)),
            ('old_overlay9', lambda v:v['original_clip_geometry_overlay'].__setitem__('source',oldbase['original_clip_geometry_overlay']['source'])),
            ('CPU13_protocol', lambda v:v['resident_RAM_resource_scheduling_policy'].__setitem__('CPU_protocol_source',oldbase['resident_RAM_resource_scheduling_policy']['CPU_protocol_source'])),
            ('MAX1', lambda v:v['resident_RAM_packet_credit_policy'].__setitem__('maximum_logical_packets',1)),
            ('changed_225_guard', lambda v:v['native_update_commit_pressure_policy'].__setitem__('kind','wrong')),
            ('missing_inherited_policy', lambda v:v.pop(lane.INHERITED_FIELD)),
            ('new_owner_completed_true', lambda v:v[lane.INHERITED_FIELD].__setitem__('new_owner_completed',True)),
            ('second_event_allowed', lambda v:v[lane.INHERITED_FIELD].__setitem__('new_owner_GC_capture_sync_or_ineligible_event_allowed',True)),
            ('freed_bookkeeping_credit', lambda v:v['one_postDEV_gc_refresh_policy'].__setitem__('additional_bookkeeping_reserve_bytes',0))]:
            bad = copy.deepcopy(configuration); mutate(bad)
            results['policy_reject_' + name] = expect((ValueError,KeyError), lambda bad=bad:lane.checked_ninth_original_policy(bad))
        saved = actual['saved_journal_prefix']
        factory = next(n for n in ast.walk(mainnode) if isinstance(n,ast.FunctionDef)and n.name=='journal_factory')
        fs = {**scope,'source_journal':saved,'SavedJournalPrefixView':PrefixViewDouble,'guard':forbidden,
              'source_namespace':ForbiddenNamespace()}
        execute(factory, fs)
        view, before = fs['journal_factory'](saved)
        require(view.binding() == saved and before == {
            'kind':'ASR18_readonly_saved186400_prefix_after_ninth_original_interval_failure_v1',
            'failure':closure,'original_tail_modified':False,'no_tail_replay':True}, 'ENTIRE actual pre-restore branch dictionary')
        results['entire_actual_pre_restore_factory'] = {'metadata':before,'prefix_view_explicit_no_read_METHOD_STUB':True}
        results['wrong_factory_saved_prefix_rejected'] = expect(ValueError,lambda:fs['journal_factory']({**saved,'offset':saved['offset']-1}))
        post = next(n for n in ast.walk(mainnode)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)
                    and n.func.id=='write_json'and n.args and 'actual_journal_branch.json'in ast.unparse(n.args[0]))
        newbinding = {'path':str(branch/'journals/commits_000026.jsonl'),'offset':saved['offset'],'sha256':saved['sha256']}
        ps = {**fs,'journal':JournalBindingDouble(newbinding),'preserved_tail':actual}
        after = expression(post.args[1], ps)
        require(after == {'kind':'ASR18_actual_saved_prefix_recovery_journal_branch_v1','predecessor':closure,
                         'saved_prefix':saved,'preserved_original':actual['whole_journal_including_preserved_tail'],
                         'new_active_journal':newbinding,'historical_unsaved_tail_updates':list(range(186401,186564)),
                         'native_updates_or_draws_in_copy':0}and 'sha256'not in after['preserved_original'],
                'ENTIRE actual postrestore dictionary, original25 STAT only')
        results['entire_actual_post_restore_dictionary']={'metadata':after,'no_copy_binding_explicit_METHOD_STUB':True}
        postcheck=next(n for n in ast.walk(mainnode)if isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)
                      and isinstance(n.value.func,ast.Name)and n.value.func.id=='require'
                      and any(isinstance(x,ast.Constant)and x.value=='Old tail byteexact after prefix copy'for x in n.value.args))
        execute(postcheck,{**ps,'check_original_closed_journal_stat':driver.check_original_closed_journal_stat})
        results['actual_postcopy_STAT_predicate_without_copy']={'passed':True,'no_original_bytes_read':True}
        missing=copy.deepcopy(args);missing.ninth_original_resume=None
        results['None_selector_complete_postdict_rejected_METHOD_STUB_scope']=expect(TypeError,lambda:expression(post.args[1],{**ps,'a':missing}))
        wrong=copy.deepcopy(args);wrong.ninth_original_resume='E:/gpu_training_scratchpad/missing_source37_failure_selector.json'
        results['wrong_selector_complete_postdict_rejected_METHOD_STUB_scope']=expect(FileNotFoundError,lambda:expression(post.args[1],{**ps,'a':wrong}))
        # Execute the actual complete runtime c.update, with the same record() calls.
        cassign=assignment(mainnode.body,'c')
        cupdate=next(n for n in mainnode.body if isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)
                     and isinstance(n.value.func,ast.Attribute)and isinstance(n.value.func.value,ast.Name)
                     and n.value.func.value.id=='c'and n.value.func.attr=='update')
        cs={**scope,'copy':copy,'__file__':str(Path(driver.__file__).resolve())}
        execute(cassign,cs);execute(cupdate,cs)
        expected7=read(review['future_runtime7']['path'])['future_runtime_values']
        require({k:cs['c'][k]for k in lane.RUNTIME_FIELDS}==expected7,'All actual future runtime7 expressions including BOTH new config pins')
        results['complete_actual_runtime7_expression']={'values':expected7,'new_stream_equals_new_source':True}
        wt=ast.parse(Path(review['wrapper_source']['path']).read_text(encoding='utf-8-sig'))
        wm=next(n for n in wt.body if isinstance(n,ast.FunctionDef)and n.name=='main')
        route=next(n for n in wm.body if isinstance(n,ast.If)and isinstance(n.test,ast.Compare)
                   and isinstance(n.test.left,ast.Attribute)and n.test.left.attr=='prior_process')
        # Explicit in-memory structural review double, NEVER a production ROOT receipt.
        structural={'ninth_original_resume':closure,'ninth_original_transition_validator':review['ninth_original_transition_validator'],
                    'original_clip_geometry_overlay':configuration['original_clip_geometry_overlay'],
                    'packet_credit_policy':configuration['resident_RAM_packet_credit_policy'],
                    'inherited_one_postDEV_refresh_opportunity_policy':configuration[lane.INHERITED_FIELD],'initial_owner_invocation':26}
        ws={'a':args,'review':structural,'configuration':configuration,'record':driver.record,'S':S,'Path':Path,
            'NINTH_TRANSITION_FOLDER':F,'require':require}
        execute(route,ws)
        results['entire_actual_wrapper_route']={'passed':True,'structural_review_METHOD_STUB_in_memory_only':True}
        wrong=copy.deepcopy(args);wrong.start_invocation=25
        results['wrapper_wrong_invocation_rejected']=expect(ValueError,lambda:execute(route,{**ws,'a':wrong}))
        results['wrapper_missing_ninth_rejected']=expect(ValueError,lambda:execute(route,{**ws,'a':missing}))
        wrong=copy.deepcopy(args);wrong.prior_process='not_an_actual_normal0'
        results['wrapper_normal0_mix_rejected']=expect(ValueError,lambda:execute(route,{**ws,'a':wrong}))
        loop=next(n for n in wm.body if isinstance(n,ast.While))
        forward=next(n for n in loop.body if isinstance(n,ast.If)and isinstance(n.test,ast.Compare)
                     and isinstance(n.test.left,ast.Name)and n.test.left.id=='prior')
        fw={'a':args,'prior':None,'invocation':26,'command':[]};execute(forward,fw)
        require(fw['command']==['--ninth-original-resume',closure['path']],'Actual initial wrapper forwards new lane exactly once')
        results['actual_complete_wrapper_forwarding']={'command':fw['command']}
        # Actual original helper constructor and actual driver inheritance AST.
        # Construction imports/initializes no GC operation. Scientific event owners are forbidden doubles.
        owner_assign=next(n for n in ast.walk(mainnode)if isinstance(n,ast.Assign)
                          and any(isinstance(t,ast.Name)and t.id=='postDEV_refresh'for t in n.targets))
        inherit_if=next(n for n in ast.walk(mainnode)if isinstance(n,ast.If)and isinstance(n.test,ast.Name)
                        and n.test.id=='ninth_policy_active'and any(isinstance(x,ast.Call)and isinstance(x.func,ast.Name)
                            and x.func.id=='inherit_consumed_postDEV_opportunity'for x in ast.walk(n)))
        emitted=[];guards=[]
        ins={'OnePostDEVRefresh':OnePostDEVRefresh,'initial_updates':186400,'ninth_policy_active':True,'base':configuration,
             'inherit_consumed_postDEV_opportunity':lane.inherit_consumed_postDEV_opportunity,
             'guard':lambda:guards.append('original_guard'),'write_json':lambda path,value:emitted.append((str(path),copy.deepcopy(value))),
             'invocation':branch/'invocations/owner_000026'}
        execute(owner_assign,ins);owner=ins['postDEV_refresh'];execute(inherit_if,ins)
        require(owner.opportunity_consumed is True and owner.attempted is False and owner.completed is False
                and owner.operations_completed==[]and owner.failed_phase is None and owner.ineligible_reason is None
                and len(emitted)==1 and len(guards)==2 and emitted[0][1]['new_eligible_or_ineligible_event_capture_GC_sync_calls']==0,
                'Actual before-loop consumed inheritance and truthful new-owner-not-completed receipt')
        owner.execute_real=forbidden;owner.consume_ineligible=forbidden
        event_if=next(n for n in ast.walk(mainnode)if isinstance(n,ast.If)and isinstance(n.test,ast.UnaryOp)
                      and isinstance(n.test.operand,ast.Attribute)and n.test.operand.attr=='opportunity_consumed')
        execute(event_if,{'postDEV_refresh':owner})
        require(owner.completed is False and owner.operations_completed==[]and len(emitted)==1,
                'Actual entire periodic event AST skipped with all event/capture/GC/sync owners unavailable')
        results['actual_driver_inherited_setup_receipt_and_complete_event_skip']={
            'passed':True,'receipt':emitted[0][1],'new_owner_completed':False,'actual_event_body_executed':False,
            'all_scientific_event_owners_forbidden_METHOD_STUBs':True,'actual_original_guard_calls':len(guards)}
        results['duplicate_inheritance_rejected']=expect(ValueError,lambda:lane.inherit_consumed_postDEV_opportunity(owner,base=configuration,restored_updates=186400,guard=forbidden,emit=forbidden))
        for name, change in [('attempted',lambda v:setattr(v,'attempted',True)),('completed',lambda v:setattr(v,'completed',True)),
                             ('wrong_expected_frontier',lambda v:setattr(v,'expected_update',184000))]:
            wrong_owner=OnePostDEVRefresh(186400);change(wrong_owner)
            results['inherit_reject_'+name]=expect(ValueError,lambda w=wrong_owner:lane.inherit_consumed_postDEV_opportunity(w,base=configuration,restored_updates=186400,guard=forbidden,emit=forbidden))
        results['wrong_owner_type_rejected']=expect(ValueError,lambda:lane.inherit_consumed_postDEV_opportunity(types.SimpleNamespace(),base=configuration,restored_updates=186400,guard=forbidden,emit=forbidden))
        sentinel=RuntimeError('METHOD_STUB exact original guard exception')
        fresh=OnePostDEVRefresh(186400)
        def fail_guard():
            raise sentinel
        try:lane.inherit_consumed_postDEV_opportunity(fresh,base=configuration,restored_updates=186400,guard=fail_guard,emit=forbidden)
        except RuntimeError as error:require(error is sentinel and fresh.opportunity_consumed is False,'Original guard exception identity retained')
        else:raise AssertionError('Expected original guard exception')
        results['inherit_original_guard_exception_identity_and_no_consumption']={'passed':True}
        emitted_error=RuntimeError('METHOD_STUB provenance writer failure')
        fresh=OnePostDEVRefresh(186400)
        def fail_emit(value):
            raise emitted_error
        try:lane.inherit_consumed_postDEV_opportunity(fresh,base=configuration,restored_updates=186400,guard=lambda:None,emit=fail_emit)
        except RuntimeError as error:require(error is emitted_error and fresh.opportunity_consumed is True
                                           and fresh.completed is False,'No new event or retry after provenance write failure')
        else:raise AssertionError('Expected provenance writer error')
        results['inherit_writer_error_consumed_without_event_or_completed_claim']={'passed':True}
        require(not(branch/'invocations/owner_000026').exists()and not(branch/'journals/commits_000026.jsonl').exists()
                and not(F.parent/'resident_RAM_training_actual_27').exists(),'No production activation/restore/journal created')
        for item in review['input_pins']:
            require(pin(item['path'])==item,'All source/compact metadata inputs exact after proof')
        require('torch'not in sys.modules and 'numpy'not in sys.modules,'No scientific package/model activity')
        report={'kind':SCOPE+'_actual_v1','passed':True,'entry':entry,'results':results,
                'initial_guard':initial,'final_guard':guard(),'fixture_source':pin(__file__),
                'source_configuration':review['source_configuration'],'input_pins':review['input_pins'],
                'actual_driver_factory_postdict_wrapper_inheritance_AST_executed':True,
                'scientific_GPU_restore_journal_copy_state_draw_event_owners_explicit_METHOD_STUB':True,
                'original_journal_suffix_event_plan_audio_tensor_payload_read':False,
                'actual_GC_capture_sync_model_native_restore_or_sampler_operation':False,
                'production_changed':False,'all51_previous_audio_CPU_proof_not_rerun':True}
        write(OUTPUT/'report.json',report)
        print(json.dumps({'report':pin(OUTPUT/'report.json'),'result_count':len(results)}),flush=True)
    except BaseException as error:
        import traceback
        write(OUTPUT/'actual_failure.json',{'error':repr(error),'traceback':traceback.format_exc(),'entry':entry,'partial_results':results})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ROOT-review',required=True);main(p.parse_args())
