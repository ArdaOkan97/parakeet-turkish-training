"""Inactive exact-owned discard for actual18107 NumPy finalization failure.

No cleanup authority without the exact independent ROOT source/inventory
review. The original TOOL1 and forty normal child0 proofs remain unchanged.
"""
import argparse,datetime,stat,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import digest
from fastmss_row_serial_cpu_v5 import environment,resource_guard
from fastmss_owned_process_closure_v2 import verify as verify_process
import fastmss_phase_aware_feature_window_v3 as sink
import psutil
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')
RUN=A/'full_DATA_two_window_native_02';STAGES=RUN/'CPU_window_0/payload'
FAILED=RUN/'actual_owned_CLOSED1_v1.json'
FAIL_SHA='7fd685f46de53a8528cb3794bac4b83e9942402456d09012e6568a85bbdd071a'
OWNER=Path(r'E:\gpu_training_scratchpad\temp\ASR18_CPU_feature_windows_v1\payload\owner.json')
OWNER_SHA='c85d8604e3a184ecab3366da65efa41055e6766f91521ad601a6379b603c4c5b'
SINK_SHA='de745c72c43bb99ae2eacc6b4e62de1c7ad53f2d35b037f5097c9d56ec438c46'


def exact(left,right,message):require(left==right and digest(left)==digest(right),message)


def dead(value):
    try:process=psutil.Process(value['pid']);return process.create_time()!=value['create_time'] or not process.is_running()
    except psutil.NoSuchProcess:return True


def verify_failure(guard):
    guard();require(record(FAILED)['sha256']==FAIL_SHA,'Exact observed failed TOOL1 required')
    failure=read_json(FAILED)
    require(failure['owned_session']==18107 and type(failure['actual_exit_code']) is int
        and failure['actual_exit_code']==1 and failure['actual_tool_chunk_id']=='69282c'
        and failure['GPU_launched'] is False and type(failure['actual_Adam_updates']) is int
        and failure['actual_Adam_updates']==0 and type(failure['actual_authoritative_draws_consumed']) is int
        and failure['actual_authoritative_draws_consumed']==0
        and failure['normal_prepared_reference_children']==20 and failure['normal_prepared_frontend_children']==20
        and failure['final_manifest_published'] is False,'Only the unconsumed failed preparation scope')
    require(record(failure['all_captured_owned_identities_observed_dead']['path'])==failure['all_captured_owned_identities_observed_dead'],
        'Exact earlier observed identity death binding')
    identities=read_json(failure['all_captured_owned_identities_observed_dead']['path'])['exact_owned_identities']
    require(identities and all(dead(item['identity']) for item in identities),'All exact captured identities must still be dead')
    lease=read_json(P/'work/gpu_training_owner.json')
    require(lease['owner']=='/root/memory_diagnosis' and lease['actual_GPU_running'] is False
        and lease.get('actual_owned_worker_pid') is None and not list(RUN.glob('GPU_process_*')),
        'Actual idle sole GPU lease and no native worker in failed run')
    require(record(S/'fastmss_phase_aware_feature_window_v3.py')['sha256']==SINK_SHA,'Frozen cleanup implementation')
    return failure


def main(review_path,folder):
    environment();guard=resource_guard;sample=guard();m=sample['system_memory'];budget=100_000_000
    require(m['commit_used_bytes']+budget<.9*m['commit_limit_bytes'] and m['commit_available_bytes']-budget>2_000_000_000,
        'Fresh100MB metadata/hash/discard forecast')
    require(folder.resolve().parent==RUN.resolve() and not folder.exists(),'Fresh E evidence outside transient cache')
    require(review_path.resolve().is_relative_to(A.resolve()),'Review artifact is E metadata')
    own=record(__file__);review=read_json(review_path)
    require(review['reviewer']=='/root' and review['accepted'] is True
        and review['scope']=='discard_failed_original_CPU_window_after_finalization_runtime_failure_v1'
        and review['production_approval_claimed'] is False,'Independent exact discard source review')
    exact(review['source'],own,'Another cleanup source reviewed');exact(review['failed_trial'],record(FAILED),'Another failed trial reviewed')
    require(record(OWNER)['sha256']==OWNER_SHA,'Frozen uncommitted owner')
    owner_pin=record(OWNER);exact(review['partial_owner'],owner_pin,'Another owner reviewed')
    failure=verify_failure(guard);inventory_pin=failure['retained_feature_inventory']
    exact(review['retained_inventory'],inventory_pin,'Another original inventory reviewed')
    require(record(inventory_pin['path'])==inventory_pin,'Original retained inventory changed')
    inventory=read_json(inventory_pin['path']);expected=inventory['actual_files']
    owner,cache,plan=sink.owner_folder(owner_pin,guard)
    require(cache.resolve()==OWNER.parent.resolve() and cache.resolve().parent==sink.CACHE.resolve()
        and owner['updates']==20 and owner['window_index']==0 and owner['prepared_is_committed'] is False
        and owner['sampler_journal_frontier_advanced'] is False and not (cache/'manifest.json').exists(),
        'Only the exact unresolved first twenty-update window')
    for path in (sink.CACHE.resolve(),cache.resolve(),*cache.iterdir()):
        require(not (getattr(path.lstat(),'st_file_attributes',0)&stat.FILE_ATTRIBUTE_REPARSE_POINT),
            'No reparse point may redirect an owned cleanup target')
    paths=sorted(cache.iterdir());actual=[]
    require(len(paths)==681 and all(path.is_file() and path.resolve().parent==cache.resolve() for path in paths),
        'Exact681 regular files only')
    for path in paths:
        guard();pin=record(path);pin['mtime_ns']=path.stat().st_mtime_ns;actual.append(pin)
    exact(actual,expected,'Any retained source/feature bytes/signature changed; preserve all')
    require(sum(pin['bytes'] for pin in actual)==123547924,'Exact reviewed payload size only')
    closures=[]
    for i in range(20):
        value=read_json(STAGES/('completed_update_%02d.json'%i))
        require(value['passed'] is True and value['update_index']==i and value['actual_optimizer_updates']==0
            and value['sampler_journal_frontier_advanced'] is False,'Every original prepared-only child stage')
        for role in ('reference','frontend'):
            pin=value[role]['actual0'];require(record(pin['path'])==pin,'Original child proof changed')
            verify_process(read_json(pin['path']),kind='ASR18_owned_CPU_process_completion_v1',owner='/root/memory_diagnosis',guard=guard)
        pin=value['frontend']['feature_closure'];require(record(pin['path'])==pin,'Original closure changed')
        proof=sink.verify_feature_closure(pin,owner_pin,success=True,guard=guard)
        require(proof['update_index']==i,'Exact original frontend closure order');closures.append(pin)
    folder.mkdir(exist_ok=False)
    write_json(folder/'pre_discard_verification.json',{'passed':True,'review':record(review_path),
        'failure':record(FAILED),'owner':owner_pin,'inventory':inventory_pin,'files':681,'bytes':123547924,
        'fresh100MB_admission':sample,'actual_normal_CPU_children':40,'actual_GPU_or_Adam_or_draws':0})
    boundary={'kind':'discard_uncommitted_feature_window','owner_sha256':owner_pin['sha256'],
        'session':owner['session'],'phase':owner['phase'],'native_work_started':False,'committed_window_updates':0,
        'reason':'Observed controller1 at NumPy-free finalization; all40 child0 preserved, no native work',
        'actual_failed_trial':record(FAILED),'actual_ROOT_discard_review':record(review_path)}
    def verify_boundary(value):
        exact(value,boundary,'Exact specific supported discard only');verify_failure(guard)
        return True
    result=sink.cleanup_window(owner_pin,closures=closures,manifest_binding=None,boundary=boundary,
        verify_boundary=verify_boundary,evidence_path=folder/'durable_owned_metadata_archive.json',guard=guard)
    require(len(result['removed_exact_owned_files'])==681 and not cache.exists()
        and not list(sink.CACHE.iterdir()),'All exact owned files removed; shared slot empty')
    require(record(FAILED)['sha256']==FAIL_SHA,'Original TOOL1 proof unchanged after cleanup')
    result.update(source=own,actual_failed_trial=record(FAILED),review=record(review_path),
        observed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),original_TOOL1_unchanged=True,
        actual_GPU_or_Adam_updates=0,shared_feature_slot_empty=True)
    write_json(folder/'report.json',result);print({'passed':True,'files_removed':681,'bytes_removed':123547924,
        'original_tool1_preserved':True,'report':record(folder/'report.json')},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--review',required=True,type=Path);parser.add_argument('--folder',required=True,type=Path)
    args=parser.parse_args();main(args.review,args.folder)
