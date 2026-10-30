"""Metadata-only finite9223 DEV proposal/config; no GPU, model or payload read.

Same original fixed350 scorer/base assets and unchanged optimizer-free resource
policy. Genuine finite20 packet/owner/process joins replace durable authority.
The actual owner repeats full payload verification before its reviewed worker.
"""
import argparse
import ast
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';R=P/'work/research'
E=Path(r'E:\gpu_training_scratchpad');N=E/'analysis/BPE9223_training_launch_v1'
A=E/'analysis/ASR18_memory_diagnosis_v1';sys.path.insert(0,str(S))
from atomic_io_retry_v1 import read_json
from bpe_broad_production_state_v1 import canonical,require
from bpe_broad_audio_v1 import record
from fastmss_BPE9223_cpu_common_v4 import environment,resource_guard
import asr18_finite_BPE9223_serial_DEV_v2 as serial
import finite_BPE9223_DEV_runtime_v2 as runtime
import asr18_finite_BPE9223_checkpoint_evidence_v2 as finite
OWNER='/root/memory_diagnosis'
PARENT=R/'run_ASR18_finite_BPE9223_serial_DEV_v1.py'


def write_new(path,value):
    path=Path(path).resolve();require(path.is_relative_to(N.resolve()),'New finite metadata must stay in launch E namespace')
    with path.open('xb') as stream:stream.write(canonical(value));stream.flush();os.fsync(stream.fileno())
    return record(path)


def metadata_admit():
    environment();require(shutil.disk_usage(E).free>100_000_000_000,'E100GB reserve')
    samples=[]
    for _ in range(3):
        value=resource_guard();s=value['system_memory'];w=value['worker_private_memory']
        require(w['private_bytes']+64_000_000<2_000_000_000
            and s['commit_used_bytes']+64_000_000<.90*s['commit_limit_bytes']
            and s['commit_available_bytes']-64_000_000>2_000_000_000,'Fresh metadata64MB prospective allowance')
        samples.append(value);time.sleep(.05)
    return samples


def source_closure():
    queue=deque([S/'asr18_finite_BPE9223_serial_DEV_v2.py',S/'finite_BPE9223_DEV_runtime_v2.py',
        S/'asr18_finite_BPE9223_checkpoint_evidence_v2.py',PARENT,Path(__file__)])
    result={}
    while queue:
        resource_guard();path=queue.popleft().resolve()
        if path.name in result:
            require(result[path.name]['path']==str(path),'Unique source filename');continue
        tree=ast.parse(path.read_text(encoding='utf-8-sig'));compile(tree,str(path),'exec');result[path.name]=record(path)
        for node in ast.walk(tree):
            names=[x.name for x in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) and node.module else []
            for name in names:
                if '.' in name:continue
                for directory in (S,R):
                    candidate=directory/(name+'.py')
                    if candidate.is_file():queue.append(candidate);break
    return result


def metadata_packet(packet_pin):
    """Actual scalar/PRE/process joins; explicitly no tensor payload hashing."""
    value=serial.read_bound(packet_pin)
    from asr18_full_run_serial_DEV_9223_v2 import checkpoint_format_view
    def payload(checkpoint,pin):
        path=Path(pin['path']).resolve()
        require(path.is_relative_to(E.resolve()) and path==Path(checkpoint['path']).resolve().with_suffix('.tensors.bin')
            and type(pin['bytes']) is int and path.stat().st_size==pin['bytes'],
            'Finite declared payload signature only; complete SHA rechecked by actual owner before DEV')
    def prefix(value):
        require(type(value) is dict and set(value)=={'path','offset','sha256'}
            and type(value['offset']) is int and 0<=value['offset']<=16*1024**2,'Finite20 bounded journal prefix')
        path=Path(value['path']).resolve();require(path.is_relative_to(E.resolve()),'E finite journal only')
        hashed=hashlib.sha256();remaining=value['offset']
        with path.open('rb') as stream:
            while remaining:
                resource_guard();block=stream.read(min(remaining,1024**2));require(block,'Complete finite prefix')
                hashed.update(block);remaining-=len(block)
        require(hashed.hexdigest()==value['sha256'],'Exact finite committed prefix')
    packet,metadata,sampler=finite._checked_binding(value,read_artifact=serial.read_bound,
        check_process=finite.checked_process,check_owner=finite.checked_owner,check_format=checkpoint_format_view,
        check_payload=payload,check_prefix=prefix)
    return packet


def prepare_review(folder,packet_path):
    samples=metadata_admit();folder=Path(folder).resolve()
    require(folder.is_relative_to(N.resolve()) and not folder.exists(),'Fresh E finite configuration namespace')
    packet_pin=record(packet_path);packet=metadata_packet(packet_pin);sources=source_closure()
    interface=read_json(A/'optimizer_free_serial350DEV_source_interface_v2.json')
    for pin in interface['base_assets'].values():require(record(pin['path'])==pin,'Original immutable base tokenizer/frontend assets')
    folder.mkdir()
    policy={'kind':'ASR18_fixed_original350_diagnostic_DEV_scorer_policy_v1',
        'official_scorer':record(S/'metrics.py'),'integer_gate_contract':record(S/'tsc_bpe_full_decoder_transition_v1.py'),
        'reference_words':serial.REFERENCES,'retention_limits':serial.LIMITS,
        'DEV_manifests':{language:record(path) for language,path in serial.canonical_DEV_paths().items()},
        'membership_review':record(A/'ROOT_fixed_serial350DEV_manifest_membership_review_v1.json'),
        'TEST_used':False,'winner_promotion_allowed':False}
    policy_pin=write_new(folder/'fixed350_scorer_policy.json',policy)
    identity={'kind':'ASR18_actual_finite_BPE9223_diagnostic_DEV_identity_v1','native_owner':OWNER,
        'source_report':packet['overall_result'],'native_configuration':packet['configuration'],
        'native_parent_source':packet['native_source'],'owner_tool_actual0':packet['owner_tool_actual0'],
        'source_original_clocks':[64250,2000,62250],'source_optimizer_RNG_sampler_reset':False,'scorer_policy':policy_pin}
    execution={'worker_source':record(S/'finite_BPE9223_DEV_runtime_v2.py'),'parent_source':record(PARENT),
        'guard_source':record(PARENT),'bootstrap_source':record(S/'asr18_optimizer_free_DEV_structure_9223_v1.py')}
    boundary={'boundary_verifier':record(S/'asr18_finite_BPE9223_checkpoint_evidence_v2.py')}
    boundary.update({key:packet['native_source'] for key in ('native_worker','native_parent','recovery_worker','recovery_parent')})
    proposal={'kind':'ASR18_finite_BPE9223_serial350DEV_source_resource_review_handoff_v1','owner':OWNER,
        'metadata_preparer':'/root/cpu_data_preparation','source_bindings':sources,'source_count':len(sources),
        'all_owned_sources_AST_compiled':True,'implementation':record(S/'asr18_finite_BPE9223_serial_DEV_v2.py'),
        'dependencies':serial.sources(),'execution_sources':execution,'diagnostic_source_identity':identity,
        'native_boundary_sources':boundary,'scorer_policy':policy_pin,'canonical_DEV_manifests':policy['DEV_manifests'],
        'membership_review':policy['membership_review'],'base_assets':interface['base_assets'],
        'resource_policy':runtime.resource_proposal(),'outer_parent_private_limit_bytes':128_000_000,
        'ALL_shims_private_limit_bytes':64_000_000,'whole_owned_family_private_limit_bytes':5_642_000_000,
        'actual_source_TOOL0':packet['owner_tool_actual0'],'finite_recovered_checkpoint':packet_pin,
        'metadata_only_finite_source_PRE_854tensor_215Adam_and_process_joins_verified':True,
        'complete_tensor_payload_rehashed_in_metadata_assembler':False,
        'actual_owner_checked_request_full_payload_validation_still_required':True,
        'prior_original350_derivation_only':record(A/'feature_window_native20_DEV_01/actual_DEV_summary_v1.json'),
        'reused_source_scope':record(A/'ROOT_optimizer_free_serial350DEV_primitives_source_read_review_v1.json'),
        'maximum_GPU_child_wall_seconds':900,'metadata_fresh_samples':samples,'production_enabled':False,
        'candidate_resource_accepted':False,'ML_audio_GPU_executed':False}
    pin=write_new(folder/'source_resource_review_handoff.json',proposal);print(json.dumps(pin),flush=True)


def make_configuration(folder,policy_review_path,resource_review_path):
    metadata_admit();folder=Path(folder).resolve();proposal=read_json(folder/'source_resource_review_handoff.json')
    require(folder.is_relative_to(N.resolve()),'Existing E finite proposal only')
    for pin in proposal['source_bindings'].values():require(record(pin['path'])==pin,'Frozen source changed')
    metadata_packet(proposal['finite_recovered_checkpoint'])
    policy=read_json(policy_review_path);resource=read_json(resource_review_path)
    require(policy['accepted'] is True and policy['reviewer']=='/root'
        and policy['kind']=='ROOT_finite_BPE9223_optimizer_free_serial350DEV_policy_and_sources_review_v1'
        and policy['diagnostic_source_identity']==proposal['diagnostic_source_identity']
        and policy['execution_sources']==proposal['execution_sources'] and policy['scorer_policy']==proposal['scorer_policy']
        and policy['native_boundary_sources']==proposal['native_boundary_sources'],'Exact finite ROOT policy/source review')
    require(resource['accepted'] is True and resource['reviewer']=='/root'
        and resource['kind']=='ROOT_diagnostic_optimizer_free_serial350DEV_resource_review_v1'
        and resource['resource_policy']==proposal['resource_policy'] and resource['execution_sources']==proposal['execution_sources'],
        'Exact unchanged numeric ROOT resource review')
    request={'kind':serial.REQUEST_KIND,'owner':OWNER,'diagnostic_only':True,'qualified':False,'production_enabled':False,
        'action':{'kind':'diagnostic_checkpoint_then_serial350DEV','boundary':{'kind':'finite20_native_exit_and_fresh_checkpoint_recovered','optimizer_updates':20}},
        'identity':proposal['diagnostic_source_identity'],'optimizer_updates':20,
        'checkpoint_pointer':proposal['finite_recovered_checkpoint'],'native_checkpoint_actual0':proposal['actual_source_TOOL0'],
        'native_boundary_sources':proposal['native_boundary_sources'],'split':'validation','clips':350,'TEST_used':False,'native_updates_in_DEV':0,
        'scorer_policy':proposal['scorer_policy'],'ROOT_policy_review':record(policy_review_path),
        'dependencies':proposal['dependencies'],'implementation':proposal['implementation'],'execution_sources':proposal['execution_sources']}
    request_pin=write_new(folder/'finite_DEV_request.json',request)
    config={'kind':'ASR18_diagnostic_optimizer_free_serial350DEV_worker_configuration_v1','worker_mode':'serial350DEV_optimizer_free',
        'diagnostic_only':True,'production_enabled':False,'parent_driver_source':proposal['execution_sources']['parent_source'],
        'request':request_pin,'resource_policy':proposal['resource_policy'],'ROOT_resource_review':record(resource_review_path),
        'source_bindings':proposal['source_bindings'],'base_assets':proposal['base_assets'],
        'GPU_python':record(P/'work/nemo-native-env-v2/Scripts/python.exe'),'outer_parent_private_limit_bytes':128_000_000,
        'all_owned_shims_private_limit_bytes':64_000_000,'whole_owned_family_private_limit_bytes':5_642_000_000,
        'maximum_GPU_child_wall_seconds':900}
    cp=write_new(folder/'configuration_pending_execution_review.json',config)
    handoff={'configuration':cp,'request':request_pin,'scorer_policy':proposal['scorer_policy'],
        'ROOT_policy_review':record(policy_review_path),'ROOT_resource_review':record(resource_review_path),
        'execution_sources':proposal['execution_sources'],'source_bindings':proposal['source_bindings'],'base_assets':proposal['base_assets'],
        'source_resource_handoff':record(folder/'source_resource_review_handoff.json'),'actual_new_ML_audio_GPU_executed':False,
        'complete_payload_validation_deferred_to_actual_owner_checked_request':True,'production_enabled':False}
    print(json.dumps(write_new(folder/'configuration_handoff.json',handoff)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);p.add_argument('--prepare-review',action='store_true')
    p.add_argument('--finite-binding',default=str(N/'hard_configuration_01/finite20_recovered_checkpoint.json'))
    p.add_argument('--ROOT-policy-review');p.add_argument('--ROOT-resource-review');a=p.parse_args()
    if a.prepare_review:prepare_review(a.folder,a.finite_binding)
    else:
        require(a.ROOT_policy_review and a.ROOT_resource_review,'Actual ROOT policy and numeric reviews required')
        make_configuration(a.folder,a.ROOT_policy_review,a.ROOT_resource_review)
