"""Independent source-bound synthetic CURRENT metadata review; no payload/ML."""
import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys

P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish'
sys.path.insert(0,str(S));sys.path.insert(0,str(P/'work/research'))
from ASR18_continuous_data_guard_v5 import DataIOGuard,ROOT
from atomic_io_retry_v1 import write_json
from cpu_preparation_guard_v2 import now
import asr18_diagnostic_serial_DEV_9223_v1 as diagnostic
import asr18_diagnostic_DEV_runtime_9223_v1 as runtime
import asr18_full_run_serial_DEV_9223_v2 as production


def main():
    g=DataIOGuard('CPU_independent_diagnostic9223_DEV_source_review_v1',heavy=False,maximum_bytes=16_000_000);g.wait()
    D=Path(__file__).resolve().parent;checks=[]
    def pin(path):
        p=Path(path);b=p.read_bytes();g.pace(len(b));return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
    prep=json.loads((D.parent/'ROOT_diagnostic9223_CURRENT_serial350DEV_source_preparation_v1.json').read_text())
    for v in prep['sources'].values():
        for k in ('source','baseline','exact_delta'):assert pin(v[k]['path'])==v[k]
    def check(name,call,reject=False):
        g.wait()
        try:call()
        except (ValueError,KeyError,TypeError,AssertionError):
            assert reject,name
        else:assert not reject,name
        checks.append(dict(name=name,passed=True,expected_rejection=reject))
    normalize={'asr18_diagnostic_DEV_runtime_v2':'asr18_diagnostic_DEV_runtime_9223_v1',
        'asr18_diagnostic_serial_DEV_v2':'asr18_diagnostic_serial_DEV_9223_v1',
        'asr18_optimizer_free_DEV_structure_v2':'asr18_optimizer_free_DEV_structure_9223_v1',
        'asr18_DEV_weight_only_mapper_v1':'asr18_DEV_weight_only_mapper_9223_v1',
        'tdt_joint_canonical_2d_fp32_v2':'tdt_joint_canonical_2d_fp32_9223_v1'}
    old=(S/'asr18_diagnostic_DEV_runtime_v2.py').read_text();new=(S/'asr18_diagnostic_DEV_runtime_9223_v1.py').read_text()
    for a,b in normalize.items():old=old.replace(a,b)
    check('runtime_entire_AST_only_exact9223_module_substitutions',lambda:assert_equal(ast.dump(ast.parse(old)),ast.dump(ast.parse(new))))
    old=(S/'asr18_diagnostic_serial_DEV_v2.py').read_text();new=(S/'asr18_diagnostic_serial_DEV_9223_v1.py').read_text()
    for a,b in normalize.items():old=old.replace(a,b)
    old=old.replace('model.config.vocab_size == 8711','model.config.vocab_size == 9223').replace("'ASR18_diagnostic_DEV_'","'ASR18_diagnostic_9223_DEV_'")
    old_funcs={n.name:ast.dump(n) for n in ast.parse(old).body if isinstance(n,ast.FunctionDef)}
    new_funcs={n.name:ast.dump(n) for n in ast.parse(new).body if isinstance(n,ast.FunctionDef)}
    for name in ('serial','same','integer','new_E','write_new','canonical_DEV_paths','_saved_dictionary',
        '_tensor_hash','_capture_RNG','_restore_RNG','check_optimizer_free_report_protocol',
        'evaluate_boundary_checkpoint','verified_completed_DEV','seal_completed_DEV','diagnostic_saved_text_rescore','checked_policy'):
        check('unchanged_full_function_'+name,lambda name=name:assert_equal(old_funcs[name],new_funcs[name]))
    # Every object below exists ONLY in this synthetic in-memory validation core.
    # No actual process/source/ROOT review or checkpoint evidence is manufactured.
    def k(name):return {'fixture_key':name}
    names=['boundary_verifier','native_worker','native_parent','recovery_worker','recovery_parent','durable_prefix_verifier']
    pins={n:k(n) for n in names};native_id={'owner_validation_only':True,'fixture_namespace':'synthetic_only'};sha=diagnostic.digest(native_id)
    pp,cp,journal=k('pointer'),k('checkpoint'),{'path':'synthetic-journal','offset':0,'sha256':'a'*64}
    pointer=dict(checkpoint=cp,sampler_state_sha256='b'*64,journal_prefix=journal)
    capture=dict(pointer=pp,checkpoint=cp,checkpoint_persistence_verified=True,exact_GPU_restore_performed=False,
        publication_role='completed_window_end',current_latest_recovery_required_before_phase_DEV_winner_advance=True,
        pre_save_fingerprint=k('PRE'),next_draw_packet=k('nextdraw'))
    request=dict(native_boundary_sources=pins,identity={'native_run_identity':native_id},optimizer_updates=20,checkpoint_pointer=pp)
    pending_head=dict(path='synthetic-same-stream',sequence=4,offset=100,framed_bytes=100)
    current_head=dict(path='synthetic-same-stream',sequence=5,offset=200,framed_bytes=100)
    pending={'publications':2,'windows':2}
    evidence=dict(kind='ASR18_full_run_native_exit_and_CURRENT_checkpoint_recovery_boundary_for_serial_DEV_v2',
        owner=diagnostic.OWNER,passed=True,diagnostic_only=True,source=pins['boundary_verifier'],
        production_recovery_scope='CURRENT_LATEST_ONLY_WITH_VERIFIED_DURABLE_PERSISTENCE_PREFIX',checkpoint_pointer=pp,
        run_identity_sha256=sha,optimizer_updates=20,sampler_state_sha256=pointer['sampler_state_sha256'],journal_prefix=journal,
        all207_threeAdam_all4RNG_exact_checkpoint_restore_verified=True,current_checkpoint_capture=capture,
        current_native_process0=k('native0'),actual_recovery_process0=k('recovery0'),recovery_report=k('recovery'),
        production_recovery_boundary=k('ACK'),durable_persistence_prefix_verification=k('prefix'),durable_head=current_head)
    table={
        'native0':dict(worker_source=pins['native_worker'],parent_source=pins['native_parent'],report=k('native')),
        'native':dict(passed=True,diagnostic_only=True,run_identity_sha256=sha,current_checkpoint=capture,
            final_optimizer_updates=20,all_exact_GPU_checkpoint_restores_deferred=True),
        'recovery0':dict(worker_source=pins['recovery_worker'],parent_source=pins['recovery_parent'],report=k('recovery')),
        'recovery':dict(kind='ASR18_full_run_actual_CURRENT_checkpoint_recovered_v2',passed=True,diagnostic_only=True,
            current_latest_checkpoint_recovered=True,historical_fresh_GPU_restores_claimed=False,current_native_process0=k('native0'),
            run_identity_sha256=sha,actual_native_updates_in_recovery_process=0,actual_draws_consumed_in_recovery=0,recovered_checkpoints=[k('recovered')]),
        'recovered':dict(passed=True,pointer=pp,checkpoint=cp,pre_save_fingerprint=capture['pre_save_fingerprint'],next_draw_packet=capture['next_draw_packet'],
            all207_threeAdam_all4RNG_match_actual_pre_save=True,both_samplers_journal_prefix_and_next_draw_exact=True,
            native_updates_performed=0,draws_consumed=0,separate_fresh_GPU_recovery=True),
        'ACK':dict(kind='ASR18_full_run_current_latest_recovery_and_verified_persistence_prefix_v1',diagnostic_only=True,
            run_identity_sha256=sha,final_checkpoint_pointer=pp,fresh_recovery_scope='current_latest_only',
            all_persistence_verified_through_pending_prefix=True,all_historical_checkpoints_freshly_restored=False,
            current_latest_all207_threeAdam_all4RNG_sampler_prefix_nextdraw_match_actual_pre_save=True,
            actual_recovery_native_updates=0,actual_recovery_draws_consumed=0,current_latest_exact_restore=k('recovered'),
            persistence_prefix_coverage_index=k('prefix'),pending_head=pending_head,pending=pending),
        'prefix':dict(kind='ASR18_verified_durable_persistence_prefix_coverage_v1',passed=True,source=pins['durable_prefix_verifier'],
            run_identity_sha256=sha,pending_head=pending_head,pending=pending,final_checkpoint_pointer=pp,
            current_checkpoint_capture_sha256=diagnostic.digest(capture),all_completed_window_PRE_saved_live_persistence_verified=True,
            historical_fresh_GPU_restores_claimed=False,native_updates_consumed_in_verification=0)}
    heads={'pending':dict(identity_sha256=sha,head=dict(owner=k('owner'),pending=pending,lifecycle={'last_pointer':pp})),
        'current':dict(identity_sha256=sha,operation='current_latest_recovery_and_persistence_prefix_acknowledged',previous_head=pending_head,
            evidence={'recovery':k('ACK')},head=dict(owner=k('owner'),identity_sha256=sha,last_recovery=k('ACK'),pending={'publications':0,'windows':0},
                lifecycle={'last_pointer':pp,'last_sampler_sha256':pointer['sampler_state_sha256']}))}
    def run(e=evidence,r=request,t=table,h=heads,func=diagnostic._checked_current_native_boundary):
        def process(value,**kw):
            if 'report' in kw:assert value['report']==kw['report']
        return func(e,request=r,pointer=pointer,read_artifact=lambda pin:t[pin['fixture_key']],verify_process=process,
            read_head=lambda pin:h['pending' if pin['sequence']==4 else 'current'])
    check('diagnostic_CURRENT_chain_positive_metadata_only',run)
    check('production_checker_keeps_strictFalse_and_rejects_same_diagnostic_chain',lambda:run(func=production._checked_current_native_boundary),True)
    for flag in (False,1,None):
        r=deepcopy(request);r['identity']['native_run_identity']['owner_validation_only']=flag
        check('reject_native_identity_owner_validation_'+str(flag),lambda r=r:run(r=r),True)
    r=deepcopy(request);r['optimizer_updates']=True
    check('reject_bool_update_frontier',lambda:run(r=r),True)
    r=deepcopy(request);r['identity']['native_run_identity']['unexpected_identity_change']=True
    check('reject_raw_identity_digest_change',lambda:run(r=r),True)
    for key in ('native','recovery','ACK'):
        t=deepcopy(table);t[key]['diagnostic_only']=False
        check('reject_production_'+key,lambda t=t:run(t=t),True)
    t=deepcopy(table);t['recovery']['recovered_checkpoints']=[]
    check('reject_missing_actual_CURRENT_recovery',lambda:run(t=t),True)
    t=deepcopy(table);t['prefix']['historical_fresh_GPU_restores_claimed']=True
    check('reject_historical_recovery_claim',lambda:run(t=t),True)
    t=deepcopy(table);t['prefix']['source']=k('unreviewed-source')
    check('reject_changed_prefix_verifier_source',lambda:run(t=t),True)
    h=deepcopy(heads);h['current']['head']['pending']['windows']=1
    check('reject_uncleared_pending_prefix',lambda:run(h=h),True)
    assert not any(n in sys.modules for n in ('numpy','torch','transformers','pyarrow','soundfile'))
    result=dict(kind='Independent_diagnostic9223_DEV_source_and_typed_CURRENT_metadata_review_v1',owner='/root/cpu_data_preparation',
        at_utc=now(),PID=os.getpid(),passed=True,findings=[],sources=prep['sources'],fixture=pin(__file__),checks=checks,
        old_production_checker_unchanged=pin(S/'asr18_full_run_serial_DEV_9223_v2.py'),
        exact9223_format_mapper_structure_canonical_route_read=True,old8711_executed_bootstrap_leak_found=False,
        fixture_scope='Synthetic in-memory I/O callbacks only; no actual process0/recovery/payload/ROOT review is manufactured',
        audio_model_GPU_NPY_or_large_payload_read=False,execution_admission_granted=False)
    write_json(D/'report.json',result);print(dict(report=pin(D/'report.json'),checks=len(checks),PID=os.getpid(),actual_tool_exit_pending=True),flush=True)


def assert_equal(left,right):assert left==right


if __name__=='__main__':main()
