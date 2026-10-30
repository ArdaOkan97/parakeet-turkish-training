"""Synthetic metadata interface and header gates only; no native idle proof."""
import argparse
import ast
from copy import deepcopy
import json
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parents[2];SOURCE=PROJECT/'outputs/parakeet_turkish'
sys.path.insert(0,str(SOURCE))
import fastmss_owned_process_closure_v2 as process
import fastmss_phase_aware_plan_bridge_v5 as bridge
import fastmss_phase_aware_feature_window_v3 as sink
import fastmss_row_serial_cpu_v5 as core


def main(output):
    core.environment();core.resource_guard();output=core.scoped(output)
    core.require(not output.exists(),'Fresh E-only synthetic metadata protocol controls');output.mkdir()
    checks=[]
    def check(name,value):core.require(value,name);checks.append(name)
    def reject(name,function):
        try:function()
        except (ValueError,KeyError,TypeError,FileNotFoundError):checks.append(name);return
        raise AssertionError('Expected rejection: '+name)
    files=[SOURCE/'fastmss_owned_process_closure_v2.py',SOURCE/'fastmss_phase_aware_plan_bridge_v5.py',
        SOURCE/'fastmss_phase_aware_feature_window_v3.py',SOURCE/'fastmss_row_serial_cpu_v5.py',
        PROJECT/'work/research/prepare_FastMSS_phase_window_CPU_v5.py',
        PROJECT/'work/research/produce_FastMSS_independent_rows_CPU_v5.py',
        PROJECT/'work/research/produce_FastMSS_serial_features_CPU_v5.py']
    for path in files:ast.parse(path.read_text(encoding='utf-8'))
    check('All seven NEW successor sources parse',True)
    check('Sink3 explicitly pins bridge5 actual bytes',sink.BRIDGE_SHA==core.record(SOURCE/'fastmss_phase_aware_plan_bridge_v5.py')['sha256'])
    def fn(path,name):
        return ast.dump(next(n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
    check('Genuine final DATA actual0 is unchanged AST-exact',fn(SOURCE/'fastmss_phase_aware_plan_bridge_v4.py','actual0')==fn(SOURCE/'fastmss_phase_aware_plan_bridge_v5.py','actual0'))
    check('Final DATA verifier is unchanged AST-exact',fn(SOURCE/'fastmss_phase_aware_plan_bridge_v4.py','validate_DATA')==fn(SOURCE/'fastmss_phase_aware_plan_bridge_v5.py','validate_DATA'))
    check('CPU4 raw plan and reservation formula remain same frozen source',
        core.raw_plan is __import__('fastmss_raw_lifetime_v1').raw_plan)
    for n in (160,319,16000,263040):
        expected=(n//160+7)//8
        value=core.clean_header_geometry(n,16000,1,expected)
        check('Clean header accepted without allocation for samples'+str(n),
              value['valid_mel_frames']==n//160 and value['physical_feature_frames']==n//160+1 and value['encoder_frames']==expected)
        reject('Wrong preallocation T rejected for samples'+str(n),lambda n=n,expected=expected:core.clean_header_geometry(n,16000,1,expected+1))
    for name,args in [('native sample rate',(16000,24000,1,13)),('stereo',(16000,16000,2,13)),
        ('boolean T',(160,16000,1,True)),('short unresolved',(159,16000,1,1)),('boolean sample',(True,16000,1,1))]:
        reject('Preallocation header rejects '+name,lambda args=args:core.clean_header_geometry(*args))
    # Keep all positive persistence-format objects in memory. They explicitly
    # model a protocol; no artifact claims an actual GPU persistence/idle exit.
    updates=2;prefix={'path':str(output/'SYNTHETIC_NEVER_WRITTEN_JOURNAL.jsonl'),'offset':0,'sha256':'0'*64}
    dummy={'path':str(output/'SYNTHETIC_NEVER_WRITTEN_checkpoint.json'),'bytes':1,'sha256':'1'*64}
    semantic={'passed':True,'fixture_only':True}
    pointer={'checkpoint':dummy,'payload':{**dummy,'path':str(output/'SYNTHETIC_NEVER_WRITTEN_payload.bin')},
        'committed_updates':updates,'sampler_state_sha256':'2'*64,'journal_prefix':prefix,'saved_semantic_state_verified':semantic}
    state={'kind':'ASR18_actual_live207_threeAdam_all4RNG_fingerprint_v1','updates':updates,
        'weights':{str(i):{'fixture_only':True} for i in range(207)},'Adam_partitions':[{'state':{}} for _ in range(3)],
        'clocks':[64252,2002,62252]}
    fingerprint={'state':state,'state_sha256':core.digest(state)}
    proof={'kind':'ASR18_actual_streamed_saved_state_matches_independent_pre_save_v1','passed':True,
        'source':core.record(SOURCE/'asr18_streamed_persistence_fingerprint_v1.py'),
        'worker_source':core.record(__file__),'checkpoint':pointer['checkpoint'],'payload':pointer['payload'],
        'optimizer_updates':updates,'weights_verified':207,'Adam_states_verified':210,'named_tensors_verified':845,
        'sampler_state_sha256':pointer['sampler_state_sha256'],'journal_prefix':prefix,
        'saved_semantic_state_verified':semantic,'maximum_payload_read_bytes':4*1024**2,
        'actual_GPU_restore_performed':False,'live_model_optimizer_or_RNG_mutated_here':False,
        'independent_post_check_live_fingerprint_comparison_required':True,
        'full_fresh_GPU_restore_required_before_overall_diagnostic_pass':True,'live_unchanged_after_persistence':True,
        'independent_pre_save_state_sha256':fingerprint['state_sha256'],'saved_state_sha256':fingerprint['state_sha256'],
        'live_after_state_sha256':fingerprint['state_sha256'],'pre_save_fingerprint':{'fixture_read':'fingerprint'},
        'next_draw_packet':{'fixture_read':'next_packet'}}
    value={'kind':'ASR18_resident_native_persistence_verified_idle_boundary_v2','checkpoint_persistence_verified':True,
        'checkpoint_exact_restore_verified':False,'deferred_restore_required_before_overall_diagnostic_pass':True,
        'checkpoint_persistence_proof':{'fixture_read':'proof'},'worker_source':proof['worker_source']}
    pointer['identity']={'caller_identity':{'pre_save_fingerprint':deepcopy(proof['pre_save_fingerprint']),
                                         'next_draw_packet':deepcopy(proof['next_draw_packet'])}}
    original_read=process.read
    def memory_read(pin,guard=lambda:None):
        guard();return deepcopy({'proof':proof,'fingerprint':fingerprint,'next_packet':{'fixture_only':True}}[pin['fixture_read']])
    process.read=memory_read
    try:
        result=process._persistence_contract(value,pointer)
        check('Synthetic pure metadata persistence branch accepts explicit restoreFalse+obligationTrue',
              result['actual_GPU_restore_performed'] is False and result['full_fresh_GPU_restore_required_before_overall_diagnostic_pass'] is True)
        for key in ('checkpoint_persistence_verified','deferred_restore_required_before_overall_diagnostic_pass'):
            bad={**value,key:False};reject('Missing '+key+' rejects',lambda bad=bad:process._persistence_contract(bad,pointer))
        bad={**value,'checkpoint_exact_restore_verified':True}
        reject('Persistence cannot be relabeled as exact restore',lambda:process._persistence_contract(bad,pointer))
        for key,wrong in [('weights_verified',206),('optimizer_updates',True),('live_after_state_sha256','f'*64),
            ('live_unchanged_after_persistence',False),('actual_GPU_restore_performed',True),
            ('full_fresh_GPU_restore_required_before_overall_diagnostic_pass',False)]:
            old=deepcopy(proof[key]);proof[key]=wrong
            try:reject('Persistence proof rejects '+key,lambda:process._persistence_contract(value,pointer))
            finally:proof[key]=old
        old=deepcopy(proof['next_draw_packet']);proof['next_draw_packet']={'fixture_read':'unbound_packet'}
        try:reject('Checkpoint identity rejects switched next-draw artifact before consume',lambda:process._persistence_contract(value,pointer))
        finally:proof['next_draw_packet']=old
        old=deepcopy(proof['worker_source']);proof['worker_source']={**old,'sha256':'f'*64}
        try:reject('Wrong live worker source rejects',lambda:process._persistence_contract(value,pointer))
        finally:proof['worker_source']=old
    finally:process.read=original_read
    reject('Persistence kind is never accepted by actual process-completion validator',lambda:
        process.verify(value,kind='ASR18_owned_GPU_process_completion_v1',owner='/root/memory_diagnosis'))
    check('No actual positive persistence/idle proof was written',not list(output.iterdir()))
    check('No numerical/model/audio imports',not any(k in sys.modules for k in ('numpy','torch','scipy','transformers','pyarrow','soundfile','av')))
    core.resource_guard()
    core.write_new(output/'report.json',{'kind':'FastMSS_synthetic_persistence_idle_and_header_metadata_controls_v1','passed':True,
        'check_count':len(checks),'checks':checks,'source':core.record(__file__),'bindings':[core.record(path) for path in files],
        'positive_contract_checks_are_in_memory_synthetic_metadata_only':True,'actual_live_idle_process_GPU_proof_forged':False,
        'actual_GPU_restore_or_persistence_payload_read':False,'full_DATA_or_audio_frontend_CACHE_job_started':False,
        'original_source_or_admitted_files_changed':False,'production_approval_claimed':False})
    print(json.dumps({'passed':True,'checks':len(checks),'synthetic_metadata_only':True}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);main(Path(p.parse_args().output))
