"""Metadata-only source/policy proposal, then exact config after ROOT reviews."""
import argparse
import ast
from collections import deque
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';R=P/'work/research'
E=Path(r'E:\gpu_training_scratchpad');A=E/'analysis/ASR18_memory_diagnosis_v1'
sys.path.insert(0,str(S))
from atomic_io_retry_v1 import read_json
from bpe_broad_production_state_v1 import canonical,digest,require
from bpe_broad_audio_v1 import record
OWNER='/root/memory_diagnosis'
FOLDER=A/'diagnostic_optimizer_free_serial350DEV_configuration_02'
ACTUAL=A/'fullpopulation_firstwindow20_conv_actual_01'


def write_new(path,value):
    path=Path(path).resolve();require(path.is_relative_to(E.resolve()),'All new config/evidence E only')
    with path.open('xb') as stream:
        stream.write(canonical(value));stream.flush();os.fsync(stream.fileno())
    return record(path)


def source_closure():
    queue=deque([S/'asr18_diagnostic_serial_DEV_v2.py',S/'asr18_diagnostic_DEV_runtime_v2.py',
                 R/'run_ASR18_diagnostic_serial_DEV_v2.py',R/'run_ASR18_frozen_CNN_lifetime_pair_v8.py',
                 R/'prepare_ASR18_diagnostic_DEV_sources_v1.py',R/'prepare_ASR18_diagnostic_DEV_successors_v2.py',Path(__file__)])
    result={}
    while queue:
        path=queue.popleft().resolve()
        if path.name in result:
            require(result[path.name]['path']==str(path),'Unique source filename required');continue
        tree=ast.parse(path.read_text(encoding='utf-8-sig'));compile(tree,str(path),'exec')
        result[path.name]=record(path)
        for node in ast.walk(tree):
            names=[]
            if isinstance(node,ast.Import):names=[item.name for item in node.names]
            elif isinstance(node,ast.ImportFrom) and node.module:names=[node.module]
            for name in names:
                if '.' in name:continue
                for directory in (S,R):
                    candidate=directory/(name+'.py')
                    if candidate.is_file():queue.append(candidate);break
    return result


def metadata_admit():
    from resource_watchdog_v2 import sample_memory
    require(E.exists() and read_json(E/'storage_config.json')['storage_root']==str(E),'Required E storage')
    require(shutil.disk_usage(E).free>100_000_000_000,'E100GB reserve')
    for key in ('TEMP','TMP','HF_HOME','TORCH_EXTENSIONS_DIR'):
        require(Path(os.environ[key]).resolve().is_relative_to(E.resolve()),'Apply E process environment first')
    require(os.environ['CUDA_VISIBLE_DEVICES']=='-1','Metadata-only CUDA mask')
    samples=[]
    for _ in range(3):
        value=sample_memory(os.getpid(),'diagnostic_DEV_metadata64MB')
        s=value['system_memory'];w=value['worker_private_memory']
        require(value['paired_sampling_within_one_second'] and s['available'] and w['available']
            and w['private_bytes']+64_000_000<2_000_000_000
            and s['commit_used_bytes']+64_000_000<.90*s['commit_limit_bytes']
            and s['commit_available_bytes']-64_000_000>2_000_000_000,'Fresh bounded metadata admission')
        samples.append(value);time.sleep(.05)
    return samples


def prepare_review():
    samples=metadata_admit();require(not FOLDER.exists(),'Fresh source proposal namespace')
    sources=source_closure()
    from asr18_diagnostic_DEV_runtime_v2 import resource_proposal
    from asr18_diagnostic_serial_DEV_v2 import REFERENCES,LIMITS,sources as adapter_sources,canonical_DEV_paths
    interface=read_json(A/'optimizer_free_serial350DEV_source_interface_v2.json')
    for pin in interface['base_assets'].values():require(record(pin['path'])==pin,'Original immutable assets')
    full=read_json(ACTUAL/'report.json')
    require(full['passed'] is True and full['actual_native_updates']==20
            and full['all_deferred_recovery_obligations_discharged'] is True,'Actual pipeline0 prerequisite')
    old_config=read_json(A/'fullpopulation_firstwindow20_conv_configuration_CPU_actual_02/finite_two_window_configuration_pending_execution_review.json')
    FOLDER.mkdir()
    policy={'kind':'ASR18_fixed_original350_diagnostic_DEV_scorer_policy_v1',
        'official_scorer':record(S/'metrics.py'),'integer_gate_contract':record(S/'tsc_bpe_full_decoder_transition_v1.py'),
        'reference_words':REFERENCES,'retention_limits':LIMITS,
        'DEV_manifests':{language:record(path) for language,path in canonical_DEV_paths().items()},
        'membership_review':record(A/'ROOT_fixed_serial350DEV_manifest_membership_review_v1.json'),
        'TEST_used':False,'winner_promotion_allowed':False}
    policy_pin=write_new(FOLDER/'fixed350_scorer_policy.json',policy)
    identity={'kind':'ASR18_actual_fullpopulation_pipeline_diagnostic_DEV_identity_v1','native_owner':OWNER,
        'source_report':record(ACTUAL/'report.json'),'native_session':old_config['session'],
        'native_engine_source':record(S/'asr18_full_DATA_native_window_v5.py'),
        'native_parent_source':record(R/'run_ASR18_full_DATA_multiwindow_v11.py'),
        'deferred_recovery_requests':record(ACTUAL/'all_actual_native_checkpoints_pending_recovery.json'),
        'source_original_clocks':[64250,2000,62250],'source_optimizer_RNG_sampler_reset':False,
        'scorer_policy':policy_pin,'diagnostic_only':True,'production_enabled':False}
    execution={'worker_source':record(S/'asr18_diagnostic_DEV_runtime_v2.py'),
        'parent_source':record(R/'run_ASR18_diagnostic_serial_DEV_v2.py'),
        'guard_source':record(R/'run_ASR18_diagnostic_serial_DEV_v2.py'),
        'bootstrap_source':record(S/'asr18_optimizer_free_DEV_structure_v2.py')}
    proposal={'kind':'ASR18_diagnostic_optimizer_free_serial350DEV_source_resource_review_handoff_v1',
        'owner':OWNER,'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'source_bindings':sources,'source_count':len(sources),'all_owned_sources_AST_compiled':True,
        'implementation':record(S/'asr18_diagnostic_serial_DEV_v2.py'),'dependencies':adapter_sources(),
        'execution_sources':execution,'diagnostic_source_identity':identity,'scorer_policy':policy_pin,
        'canonical_DEV_manifests':policy['DEV_manifests'],'membership_review':policy['membership_review'],
        'base_assets':interface['base_assets'],'resource_policy':resource_proposal(),
        'outer_parent_private_limit_bytes':128_000_000,'ALL_shims_private_limit_bytes':64_000_000,
        'whole_owned_family_private_limit_bytes':5_642_000_000,
        'actual_source_TOOL0':record(ACTUAL/'actual_owned_CLOSED0_v1.json'),
        'actual_source_summary':record(ACTUAL/'actual_firstwindow20_end_to_end_summary_v1.json'),
        'latest_checkpoint_pointer':full['windows'][-1]['latest_pointer'],
        'prior_original350_derivation_only':record(A/'feature_window_native20_DEV_01/actual_DEV_summary_v1.json'),
        'reused_source_scope':record(A/'ROOT_optimizer_free_serial350DEV_primitives_source_read_review_v1.json'),
        'new_deltas':['explicit diagnostic native TOOL0/every-capture recovered chain, no fullrun identity spoof',
            'same selective207/all4RNG/zeroAdam evaluator; private scorer-kind projection never published as fullrun',
            'early native entrypoint identity before validation; matching runtime boot before lease/admission; native5.45GB separate from ALLshims64MB',
            'exact decimal8GB allocator/peak gate; source validate_capacity(8.) uses8e9, not GiB',
            'current and OS lifetime5.45GB, unchanged phase budgets, coherent endcache3GB through model release',
            'validated conv-only2 instance hook before evaluation'],
        'maximum_GPU_child_wall_seconds':900,'metadata_fresh_samples':samples,
        'production_enabled':False,'candidate_resource_accepted':False,'ML_audio_GPU_executed':False}
    pin=write_new(FOLDER/'source_resource_review_handoff.json',proposal)
    print(json.dumps(pin),flush=True)


def make_configuration(policy_review_path,resource_review_path):
    metadata_admit();proposal=read_json(FOLDER/'source_resource_review_handoff.json')
    for pin in proposal['source_bindings'].values():require(record(pin['path'])==pin,'Frozen source changed')
    policy_review=read_json(policy_review_path);resource=read_json(resource_review_path)
    require(policy_review['accepted'] is True and policy_review['reviewer']=='/root'
            and policy_review['kind']=='ROOT_diagnostic_optimizer_free_serial350DEV_policy_and_sources_review_v1'
            and policy_review['diagnostic_source_identity']==proposal['diagnostic_source_identity']
            and policy_review['execution_sources']==proposal['execution_sources']
            and policy_review['scorer_policy']==proposal['scorer_policy'], 'Exact new ROOT policy/source review')
    require(resource['accepted'] is True and resource['reviewer']=='/root'
            and resource['kind']=='ROOT_diagnostic_optimizer_free_serial350DEV_resource_review_v1'
            and resource['resource_policy']==proposal['resource_policy']
            and resource['execution_sources']==proposal['execution_sources'],'Exact new numeric resource review')
    request={'kind':'ASR18_diagnostic_recovered_pipeline_serial350DEV_request_v1','owner':OWNER,
        'diagnostic_only':True,'qualified':False,'production_enabled':False,
        'action':{'kind':'diagnostic_checkpoint_then_serial350DEV',
            'boundary':{'kind':'diagnostic_pipeline_native_exit_and_every_capture_recovered','optimizer_updates':20}},
        'identity':proposal['diagnostic_source_identity'],'optimizer_updates':20,
        'checkpoint_pointer':proposal['latest_checkpoint_pointer'],'native_checkpoint_actual0':proposal['actual_source_TOOL0'],
        'split':'validation','clips':350,'TEST_used':False,'native_updates_in_DEV':0,
        'scorer_policy':proposal['scorer_policy'],'ROOT_policy_review':record(policy_review_path),
        'dependencies':proposal['dependencies'],'implementation':proposal['implementation'],
        'execution_sources':proposal['execution_sources']}
    request_pin=write_new(FOLDER/'diagnostic_DEV_request.json',request)
    from asr18_diagnostic_serial_DEV_v2 import checked_request
    checked_request(request)
    config={'kind':'ASR18_diagnostic_optimizer_free_serial350DEV_worker_configuration_v1',
        'worker_mode':'serial350DEV_optimizer_free','diagnostic_only':True,'production_enabled':False,
        'parent_driver_source':proposal['execution_sources']['parent_source'],'request':request_pin,
        'resource_policy':proposal['resource_policy'],'ROOT_resource_review':record(resource_review_path),
        'source_bindings':proposal['source_bindings'],'base_assets':proposal['base_assets'],
        'GPU_python':record(P/'work/nemo-native-env-v2/Scripts/python.exe'),
        'outer_parent_private_limit_bytes':128_000_000,'all_owned_shims_private_limit_bytes':64_000_000,
        'whole_owned_family_private_limit_bytes':5_642_000_000,'maximum_GPU_child_wall_seconds':900}
    config_pin=write_new(FOLDER/'configuration_pending_execution_review.json',config)
    pin=write_new(FOLDER/'configuration_handoff.json',{'configuration':config_pin,'request':request_pin,
        'scorer_policy':proposal['scorer_policy'],'ROOT_policy_review':record(policy_review_path),
        'ROOT_resource_review':record(resource_review_path),'execution_sources':proposal['execution_sources'],
        'source_bindings':proposal['source_bindings'],'base_assets':proposal['base_assets'],
        'source_resource_handoff':record(FOLDER/'source_resource_review_handoff.json'),
        'actual_new_ML_audio_GPU_executed':False,'production_enabled':False})
    print(json.dumps(pin),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-review',action='store_true')
    parser.add_argument('--ROOT-policy-review');parser.add_argument('--ROOT-resource-review');args=parser.parse_args()
    if args.prepare_review:prepare_review()
    else:
        require(args.ROOT_policy_review and args.ROOT_resource_review,'Actual policy and numeric ROOT reviews required')
        make_configuration(args.ROOT_policy_review,args.ROOT_resource_review)
