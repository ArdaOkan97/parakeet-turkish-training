"""Metadata/source text preparation only. Never import the candidate or open old journal/payload/audio."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import psutil

F = Path(__file__).parent
P = Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20')
R = P / 'work/research'
S = P / 'outputs/parakeet_turkish'
sys.path.insert(0,str(S))
from resource_watchdog_v2 import sample_memory
B = Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01')
BASE = B / 'invocations/owner_000025/configuration.json'
EVENT = F.parent / 'resident_RAM_recurrent_metadata_slowdown_fix_v1/one_postDEV_gc_refresh_v1'
TEMPLATE = F / 'inactive_configuration_source37_v1/configuration.json'


def pin(p):
    p = Path(p).resolve()
    require(p.stat().st_size <= 8388608, 'Compact source/config/ROOT receipt only')
    b = p.read_bytes()
    return {'path': str(p), 'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()}


def require(value, message):
    if not value:
        raise ValueError(message)


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def write(p, value):
    with p.open('x', encoding='utf-8', newline='\n') as h:
        json.dump(value, h, indent=2)
        h.write('\n')
        h.flush()
        os.fsync(h.fileno())


class Edit:
    def __init__(self, old, new):
        self.old, self.new = old, new
        self.before = old.read_bytes()
        self.text = self.before.decode('utf-8')
        self.records = []
        require(not new.exists(), 'Immutable new successor source only')

    def replace(self, old, new, count=1):
        actual_old = old
        actual_new = new
        if self.text.count(actual_old) != count and '\n' in old:
            actual_old = old.replace('\n', '\r\n')
            actual_new = new.replace('\n', '\r\n')
        require(self.text.count(actual_old) == count, 'Exact old literal count: ' + repr(old[:120]))
        self.records.append({'old_literal': actual_old, 'new_literal': actual_new, 'expected_count': count})
        self.text = self.text.replace(actual_old, actual_new)

    def token(self, old, new):
        count = self.text.count(old)
        require(count > 0, 'Declared source binding token exists')
        self.replace(old, new, count)

    def finish(self):
        data = self.text.encode('utf-8')
        inverse = self.text
        for r in reversed(self.records):
            require(inverse.count(r['new_literal']) == r['expected_count'], 'Whole RAW inverse unambiguous')
            inverse = inverse.replace(r['new_literal'], r['old_literal'])
        require(inverse.encode('utf-8') == self.before, 'Full old RAW bytes reconstructed exactly')
        ast.parse(self.text)
        with self.new.open('xb') as h:
            h.write(data)
            h.flush()
            os.fsync(h.fileno())
        return {'old_source': pin(self.old), 'new_source': pin(self.new), 'replacement_records': self.records,
                'newline_normalization': 'NONE. Each exact old literal retains its own CRLF/LF style.',
                'entire_RAW_forward_and_inverse_exact': True}


def ast_dump(node):
    return ast.dump(node, include_attributes=False)


def main():
    me = psutil.Process()
    me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '-1' and os.environ.get('PYTHONUTF8') == '1'
            and all(os.environ.get(k) == '1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')),
            'Process-local source preparation environment')
    sample=sample_memory(me.pid,'source37_inactive_text_preparation')
    write(F/'source37_inactive_preparation_initial_numeric_sample_v1.json',sample)
    require(sample['paired_sampling_within_one_second']
            and sample['worker_private_memory']['private_bytes']<536870912
            and sample['system_memory']['commit_available_bytes']>=5055306368
            and sample['system_memory']['physical_available_bytes']>=2305306368
            and shutil.disk_usage('E:/').free>=100000000000,'Full original metadata prospective/225/FULLprefetch/floors/E guards')
    old = read(BASE)
    require(pin(BASE)['sha256'] == 'f866afb3e9cd7e8db15c341e7df5ecbc8c6cc7d6714153783c389fafc54498a7', 'Proper actual source36 base')
    scope = F / 'ROOT_ninth_source37_current36_typed_checkpoint_transition_INACTIVE_preparation_scope_v1.json'
    require(pin(scope)['sha256'] == '1ceb6234d281ec1e4d28e285f741cc4f4487ff4c24dcecdb4e33b809bbfb2abe', 'Exact preparation-only scope')
    failure_root = F / 'ROOT_source36_failure186563_saved186400_actual_independent_verification_v1.json'
    event_root = EVENT / 'ROOT_source36_actual_first_periodic180000_idle_refresh_preservation_verification_v1.json'
    quality_root = F.parent / 'ROOT_periodic180000_full350_source36_idle_refresh_CPU13_max2_normal_saved180200_and_resumed_training_verification_v1.json'
    failure = read(failure_root)
    event = read(event_root)
    standard = read(EVENT / 'ROOT_source36_standard186400_metadata_and_PRE_SAVED_equality_verification_v1.json')
    targets = [R/'run_ASR18_BPE9223_resident_RAM_training_v37.py', R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v37.py',
               TEMPLATE, F/'source37_ninth_driver_wrapper_full_RAW_and_AST_ledger_v1.json',
               F/'source37_ninth_configuration_delta_and_future_runtime7_v1.json', F/'source37_ninth_inactive_source_handoff_v1.json']
    require(all(not p.exists() for p in targets) and not TEMPLATE.parent.exists(), 'All inactive source/config targets fresh before writes')
    d = Edit(R/'run_ASR18_BPE9223_resident_RAM_training_v36.py', targets[0])
    d.replace("sys.path[:0]=[str(S),str(P/'work/research')]", "sys.path[:0]=[str(S),str(P/'work/research')]\nNINTH_TRANSITION_FOLDER=Path('E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1/resident_RAM_source36_original_interval186563_failure_actual_v1')\nsys.path.insert(0,str(NINTH_TRANSITION_FOLDER))")
    d.token('fastmss_resident_feature_stream_v13', 'fastmss_resident_feature_stream_v14')
    d.replace('    from asr18_one_postDEV_gc_refresh_checkpoint_transition_v2 import (checked_one_postDEV_refresh_policy,',
              "    from asr18_ninth_original_interval_checkpoint_resume_v1 import (checked_ninth_original_policy,checked_ninth_original_resume,\n        legacy_source36_policy_configuration,inherit_consumed_postDEV_opportunity)\n    ninth_policy_active='prior_resident_RAM_ninth_original_interval_failure'in base\n    if ninth_policy_active:checked_ninth_original_policy(base)\n    original_gc_policy_base=legacy_source36_policy_configuration(base)if ninth_policy_active else base\n    ninth_interval_failure=None if a.ninth_original_resume is None else checked_ninth_original_resume(\n        record(a.ninth_original_resume),base=base,folder=folder,pointer_pin=pointer_pin)\n    if ninth_interval_failure is not None:\n        require(a.invocation==26 and all(x is None for x in (a.prior_process,a.gc_refresh_resume,a.eighth_original_resume,\n            a.fresh_signature_resume,a.canonical_archive_resume,a.lookahead_resume,a.cushion225_resume,\n            a.postrestore_selector_resume,a.packet_credit_resume,a.native_headroom_resume,a.abnormal_resume,\n            a.source_transition_resume,a.original_interval_resume,a.DEV_frontend_resume,a.DEV_admission_resume,\n            a.periodic_quiescence_resume)),\n            'Only exact source36 original interval186563 saved186400 ninth failure, no historical or normal0 alias')\n    from asr18_one_postDEV_gc_refresh_checkpoint_transition_v2 import (checked_one_postDEV_refresh_policy,")
    d.replace('OnePostDEVRefresh,capture_idle_state,compare_idle_state=checked_one_postDEV_refresh_policy(base)',
              'OnePostDEVRefresh,capture_idle_state,compare_idle_state=checked_one_postDEV_refresh_policy(original_gc_policy_base)')
    d.replace('legacy_base=legacy_source33_configuration(base)', 'legacy_base=legacy_source33_configuration(original_gc_policy_base)')
    d.replace('require(gc_transition is not None or a.prior_process is not None,', 'require(ninth_interval_failure is not None or gc_transition is not None or a.prior_process is not None,')
    d.replace('    interval_failure=eighth_interval_failure if eighth_interval_failure is not None else(None if a.original_interval_resume is None else checked_original_interval_resume(\n        record(a.original_interval_resume),base=base,folder=folder,pointer_pin=pointer_pin))',
              '    interval_failure=ninth_interval_failure if ninth_interval_failure is not None else(eighth_interval_failure if eighth_interval_failure is not None else(None if a.original_interval_resume is None else checked_original_interval_resume(\n        record(a.original_interval_resume),base=base,folder=folder,pointer_pin=pointer_pin)))')
    d.replace('(24 if a.eighth_original_resume is not None else 16)', '(26 if a.ninth_original_resume is not None else(24 if a.eighth_original_resume is not None else 16))')
    d.replace('        def journal_factory(saved):\n            if eighth_interval_failure is not None:',
              "        def journal_factory(saved):\n            if ninth_interval_failure is not None:\n                require(saved==source_journal,'Exact actual186400 saved prefix; historical163 COPY never replayed')\n                return SavedJournalPrefixView(saved,guard),{'kind':'ASR18_readonly_saved186400_prefix_after_ninth_original_interval_failure_v1',\n                    'failure':record(a.ninth_original_resume),'original_tail_modified':False,'no_tail_replay':True}\n            if eighth_interval_failure is not None:")
    d.replace('(a.gc_refresh_resume is not None or a.eighth_original_resume is not None or a.fresh_signature_resume is not None',
              '(a.ninth_original_resume is not None or a.gc_refresh_resume is not None or a.eighth_original_resume is not None or a.fresh_signature_resume is not None')
    d.replace("'predecessor':record(a.gc_refresh_resume if a.gc_refresh_resume is not None else a.eighth_original_resume", 
              "'predecessor':record(a.ninth_original_resume if a.ninth_original_resume is not None else a.gc_refresh_resume if a.gc_refresh_resume is not None else a.eighth_original_resume")
    d.replace('        postDEV_refresh=OnePostDEVRefresh(initial_updates)',
              "        postDEV_refresh=OnePostDEVRefresh(initial_updates)\n        if ninth_policy_active:\n            inherit_consumed_postDEV_opportunity(postDEV_refresh,base=base,restored_updates=initial_updates,guard=guard,\n                emit=lambda value:write_json(invocation/'actual_inherited_consumed180000_refresh_opportunity.json',value))")
    d.replace("p.add_argument('--gc-refresh-resume');p.add_argument('--eighth-original-resume')", "p.add_argument('--ninth-original-resume');p.add_argument('--gc-refresh-resume');p.add_argument('--eighth-original-resume')")
    driver_ledger = d.finish()
    w = Edit(R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v36.py', targets[1])
    w.replace("sys.path[:0]=[str(S),str(P/'work/research')]", "sys.path[:0]=[str(S),str(P/'work/research')]\nNINTH_TRANSITION_FOLDER=Path('E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1/resident_RAM_source36_original_interval186563_failure_actual_v1')\nsys.path.insert(0,str(NINTH_TRANSITION_FOLDER))")
    w.token('run_ASR18_BPE9223_resident_RAM_training_v36', 'run_ASR18_BPE9223_resident_RAM_training_v37')
    w.token('fastmss_resident_feature_stream_v13', 'fastmss_resident_feature_stream_v14')
    w.replace('    from asr18_one_postDEV_gc_refresh_checkpoint_transition_v2 import (checked_one_postDEV_refresh_policy,',
              "    from asr18_ninth_original_interval_checkpoint_resume_v1 import (checked_ninth_original_policy,checked_ninth_original_resume,legacy_source36_policy_configuration)\n    ninth_policy_active='prior_resident_RAM_ninth_original_interval_failure'in configuration\n    if ninth_policy_active:checked_ninth_original_policy(configuration)\n    original_gc_policy_configuration=legacy_source36_policy_configuration(configuration)if ninth_policy_active else configuration\n    from asr18_one_postDEV_gc_refresh_checkpoint_transition_v2 import (checked_one_postDEV_refresh_policy,")
    w.replace('checked_one_postDEV_refresh_policy(configuration)', 'checked_one_postDEV_refresh_policy(original_gc_policy_configuration)')
    w.replace('legacy_configuration=legacy_source33_configuration(configuration)', 'legacy_configuration=legacy_source33_configuration(original_gc_policy_configuration)')
    w.replace('require(a.gc_refresh_resume is None and a.eighth_original_resume is None', 'require(a.ninth_original_resume is None and a.gc_refresh_resume is None and a.eighth_original_resume is None')
    w.replace('    elif a.gc_refresh_resume is not None:',
              "    elif a.ninth_original_resume is not None:\n        require(all(x is None for x in (a.gc_refresh_resume,a.eighth_original_resume,a.fresh_signature_resume,\n            a.canonical_archive_resume,a.lookahead_resume,a.cushion225_resume,a.postrestore_selector_resume,\n            a.packet_credit_resume,a.native_headroom_resume,a.abnormal_resume,a.original_interval_resume,\n            a.source_transition_resume,a.DEV_frontend_resume,a.DEV_admission_resume,a.periodic_quiescence_resume))\n            and review['ninth_original_resume']==record(a.ninth_original_resume)\n            and review['ninth_original_transition_validator']==record(NINTH_TRANSITION_FOLDER/'asr18_ninth_original_interval_checkpoint_resume_v1.py')\n            and review['original_clip_geometry_overlay']==configuration['original_clip_geometry_overlay']\n            and review['packet_credit_policy']==configuration['resident_RAM_packet_credit_policy']\n            and review['inherited_one_postDEV_refresh_opportunity_policy']==configuration['inherited_one_postDEV_refresh_opportunity_policy']\n            and a.start_invocation==review['initial_owner_invocation']==26,\n            'Only exact source36 interval186563 saved186400 ninth repair and consumed180k provenance')\n    elif a.gc_refresh_resume is not None:", 1)
    w.replace('    if a.gc_refresh_resume is not None:\n        checked_one_postDEV_refresh_checkpoint_resume',
              '    if a.ninth_original_resume is not None:\n        checked_ninth_original_resume(record(a.ninth_original_resume),base=configuration,folder=branch,pointer_pin=record(a.pointer))\n    if a.gc_refresh_resume is not None:\n        checked_one_postDEV_refresh_checkpoint_resume')
    w.replace("            if a.gc_refresh_resume is not None:command+=['--gc-refresh-resume',a.gc_refresh_resume]", 
              "            if a.ninth_original_resume is not None:command+=['--ninth-original-resume',a.ninth_original_resume]\n            elif a.gc_refresh_resume is not None:command+=['--gc-refresh-resume',a.gc_refresh_resume]")
    w.replace("p.add_argument('--prior-process');p.add_argument('--gc-refresh-resume')", "p.add_argument('--prior-process');p.add_argument('--ninth-original-resume');p.add_argument('--gc-refresh-resume')")
    wrapper_ledger = w.finish()
    ot = ast.parse(d.before.decode('utf-8')); nt = ast.parse(d.text)
    old_top = {n.name:n for n in ot.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name != 'main'}
    new_top = {n.name:n for n in nt.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name != 'main'}
    require(set(old_top)==set(new_top) and all(ast_dump(n)==ast_dump(new_top[k]) for k,n in old_top.items()), 'All nonentry driver ASTs exact')
    om=next(n for n in ot.body if isinstance(n,ast.FunctionDef)and n.name=='main')
    nm=next(n for n in nt.body if isinstance(n,ast.FunctionDef)and n.name=='main')
    old_loops=[ast_dump(n)for n in ast.walk(om)if isinstance(n,ast.While)]
    new_loops=[ast_dump(n)for n in ast.walk(nm)if isinstance(n,ast.While)]
    require(old_loops==new_loops and len(old_loops)==3, 'All three original native/main/checkpoint loops exact')
    old_nested={n.name:n for n in ast.walk(om)if isinstance(n,ast.FunctionDef)and n.name not in ('main','journal_factory')}
    new_nested={n.name:n for n in ast.walk(nm)if isinstance(n,ast.FunctionDef)and n.name not in ('main','journal_factory')}
    require(set(old_nested)==set(new_nested) and all(ast_dump(n)==ast_dump(new_nested[k])for k,n in old_nested.items()), 'All non-factory nested numerical/DEV/checkpoint ASTs exact')
    candidate=copy.deepcopy(old)
    candidate['original_clip_geometry_overlay']={'prior_overlay':copy.deepcopy(old['original_clip_geometry_overlay']),
        'source':pin(S/'fastmss_original_clip_geometry_overlay_v10.py'),'certificate':pin(F/'ninth_original_full_clip_geometry_certificate_v1.json'),
        'ROOT_review':pin(F/'ROOT_ninth_CPU14_actual01_original_ingress_frontend_independent_verification_v1.json')}
    candidate['resident_RAM_resource_scheduling_policy']['CPU_protocol_source']=pin(S/'fastmss_resident_feature_stream_v14.py')
    candidate['resident_RAM_packet_credit_policy']['actual_CPU_control_review']=pin(F/'ROOT_CPU14_one_overlay_import_only_transport_compatibility_successor_v1.json')
    candidate['prior_resident_RAM_ninth_original_interval_failure']=failure['owner_closure']
    candidate['inherited_one_postDEV_refresh_opportunity_policy']={
        'kind':'ASR18_inherited_consumed180000_refresh_opportunity_after_ninth_original_interval_v1',
        'historical_event_optimizer_updates':180000,'historical_event_ROOT_review':pin(event_root),
        'historical_quality_ROOT_review':pin(quality_root),'historical_actual_event':event['event'],
        'historical_full_state_fingerprint':event['source_bound_real_original_full854_fingerprint_and_ID_storage_versions_gradNone_frontiers_fourRNG_queued_before_after_capture_exact'],
        'consumed_before_native_loop':True,'new_owner_completed':False,
        'new_owner_GC_capture_sync_or_ineligible_event_allowed':False,'additional_bookkeeping_reserve_bytes':16777216}
    restored=copy.deepcopy(candidate)
    del restored['prior_resident_RAM_ninth_original_interval_failure'];del restored['inherited_one_postDEV_refresh_opportunity_policy']
    for key in ('original_clip_geometry_overlay','resident_RAM_resource_scheduling_policy','resident_RAM_packet_credit_policy'):restored[key]=copy.deepcopy(old[key])
    require(restored==old,'Exact five operational configuration delta to proper current f866')
    TEMPLATE.parent.mkdir();write(TEMPLATE,candidate)
    future7={'worker_driver_source':pin(targets[0]),'parent_driver_source':pin(targets[0]),
        'resident_RAM_stream_configuration':pin(TEMPLATE),'resident_branch_namespace':str(B),
        'checkpoint_interval_updates':200,'source_checkpoint':standard['pointer'],'source_configuration':pin(TEMPLATE)}
    write(targets[3],{'kind':'source36_to37_full_RAW_and_AST_metadata_transition_ledger_v1','driver':driver_ledger,'wrapper':wrapper_ledger,
        'all_nonentry_AST_exact':sorted(old_top),'all_nonfactory_nested_AST_exact':sorted(old_nested),
        'all_three_main_while_AST_exact':True,'native_math_DEV_checkpoint_GC_order_body_unchanged':True,
        'new_helper_source':pin(F/'asr18_ninth_original_interval_checkpoint_resume_v1.py'),
        'no_candidate_import_or_execution':True})
    write(targets[4],{'kind':'source37_proper_f866_exact_operational_delta_and_future_runtime7_v1','proper_current_source36_configuration':pin(BASE),
        'candidate_configuration':pin(TEMPLATE),'replaced_fields':['original_clip_geometry_overlay','resident_RAM_resource_scheduling_policy','resident_RAM_packet_credit_policy'],
        'added_fields':['prior_resident_RAM_ninth_original_interval_failure','inherited_one_postDEV_refresh_opportunity_policy'],
        'all_other_actual_f866_values_exact':True,'original_actual_runtime7':{k:old[k]for k in future7},'future_runtime_values':future7,
        'both_future_stream_and_source_configuration_are_exact_same_new_serialized_pin':True,
        'future_runtime_not_materialized_into_RUN':True,'recorded_pointer_bound_from_closed_ROOT_metadata_not_read':True})
    inputs=[scope,BASE,Path(old['source_configuration']['path']),failure_root,event_root,quality_root,
        EVENT/'ROOT_source36_standard186400_metadata_and_PRE_SAVED_equality_verification_v1.json',
        F/'ROOT_ninth_CPU14_actual01_original_ingress_frontend_independent_verification_v1.json',
        F/'ROOT_ninth_overlay10_CPU14_inactive_source_constructor_review_v1.json',
        F/'ROOT_CPU14_one_overlay_import_only_transport_compatibility_successor_v1.json',
        F/'ninth_original_full_clip_geometry_certificate_v1.json',F/'asr18_ninth_original_interval_checkpoint_resume_v1.py',
        *targets[:5],d.old,w.old,S/'fastmss_resident_feature_stream_v14.py',S/'fastmss_original_clip_geometry_overlay_v10.py',
        S/'asr18_one_postDEV_gc_refresh_checkpoint_transition_v2.py',EVENT/'asr18_one_postDEV_gc_refresh_v2.py',
        S/'asr18_BPE9223_grouped_RAM_native_v1.py',P/'work/research/run_ASR18_native_throughput_component_v15.py',
        S/'asr18_BPE9223_private_scalar_prefetch_v1.py']
    write(targets[5],{'kind':'source37_ninth_inactive_source_only_handoff_v1','preparation_scope':pin(scope),
        'driver_source':pin(targets[0]),'wrapper_source':pin(targets[1]),'transition_validator':pin(F/'asr18_ninth_original_interval_checkpoint_resume_v1.py'),
        'candidate_configuration':pin(TEMPLATE),'full_RAW_AST_ledger':pin(targets[3]),'future_runtime7':pin(targets[4]),
        'actual_CPU_ROOT_gate':pin(F/'ROOT_ninth_CPU14_actual01_original_ingress_frontend_independent_verification_v1.json'),
        'ninth_original_resume':failure['owner_closure'],'initial_checkpoint':standard['pointer'],
        'intended_initial_invocation':26,'intended_active_journal':str(B/'journals/commits_000026.jsonl'),
        'intended_native_output':str(F.parent/'resident_RAM_training_actual_27'),
        'all163_historical_186401_to186563_require_genuine_redo_then_repaired186564':True,
        'natural186600_then_post186620_required_before_actual_recovery_acceptance':True,
        'inherited_consumed180k_completed_false_before_loop_zero_new_GC_capture_sync_ineligible':True,
        'all_input_pins':[pin(p)for p in dict.fromkeys(inputs)],
        'new_CPU_metadata_fixture_and_external_still_pending_frozen_source_review':True,
        'candidate_import_proof_production_or_recovery_execution_admitted':False,
        'prep_identity':{'pid':me.pid,'create_time':me.create_time(),'ppid':me.ppid()},
        'prep_launcher_identity':{'pid':me.parent().pid,'create_time':me.parent().create_time(),'ppid':me.parent().ppid()}})
    print(json.dumps({'driver':pin(targets[0]),'wrapper':pin(targets[1]),'helper':pin(F/'asr18_ninth_original_interval_checkpoint_resume_v1.py'),
        'config':pin(TEMPLATE),'RAW_AST':pin(targets[3]),'runtime7':pin(targets[4]),'handoff':pin(targets[5]),
        'prep_pid':me.pid,'prep_ctime':me.create_time(),'launcher':me.parent().pid,'launcher_ctime':me.parent().create_time()}),flush=True)


if __name__=='__main__':
    main()
