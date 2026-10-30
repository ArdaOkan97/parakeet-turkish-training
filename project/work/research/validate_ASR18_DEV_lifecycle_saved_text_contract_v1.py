"""READONLY historical350 text -> unchanged scorer and callback-consumer contract.

All new report/request/process/storage proof projections stay explicitly
synthetic IN MEMORY. This does not rerun DEV, read audio/tensors, validate actual
new processes, or promote the old diagnostic checkpoint to a fullrun source.
"""
import argparse
from copy import deepcopy
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import shutil
import sys
import time

PROJECT=Path(__file__).resolve().parents[2]
SOURCE=PROJECT/'outputs/parakeet_turkish'
ROOT=Path(r'E:\gpu_training_scratchpad')
AREA=ROOT/'analysis/ASR18_memory_diagnosis_v1'
sys.path.insert(0,str(SOURCE))
CONTRACT={}


class NoNumerical(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'torch','numpy','scipy','transformers','tokenizers','soundfile','av','pyarrow'}:
            raise ImportError('Saved-text controls forbid numerical/audio import: '+fullname)


def record(path):
    path=Path(path).resolve()
    assert path.suffix in ('.py','.json','.jsonl') and (path.is_relative_to(PROJECT) or path.is_relative_to(ROOT))
    raw=path.read_bytes()
    return {'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


class SyntheticSchedule:
    def __init__(self,state):self.value=deepcopy(state)
    def state_dict(self):return deepcopy(self.value)


def owner_admit_operation(request):
    assert CONTRACT['synthetic_only'] and request['action']['kind']=='checkpoint_then_serial350DEV'
    return True


def owner_storage_boundary(request):
    assert CONTRACT['synthetic_only'] and request['preserve_all_existing_and_original_checkpoints'] is True
    return CONTRACT['storage_token']


def owner_serial_DEV(request):
    # No evaluator/worker executes; historical text is projected only in memory.
    assert request['action']['kind']=='checkpoint_then_serial350DEV' and request['split']=='validation' and request['clips']==350
    assert request['checkpoint_pointer']==CONTRACT['pointer']
    CONTRACT['serial_calls']+=1
    return CONTRACT['report_token']


def owner_verify_DEV(request):
    CONTRACT['verify_calls']+=1
    # Execute the NEW adapter method; only its future actual-proof IO/verifier
    # is replaced. The unchanged real scorer is exercised by the lifecycle below.
    return CONTRACT['adapter'].verify_DEV(request)


def owner_preserve_winner(request):
    assert request['replace'] is False and request['score']['strict_all_six_passed'] is False
    assert request['old_winner'] is None
    CONTRACT['winner_calls']+=1
    return None


def owner_unused_callback(request):
    raise AssertionError('Training/resume/winner callback is forbidden in saved-text fixture')


def main(folder):
    folder=Path(folder).resolve();assert folder.is_relative_to(ROOT.resolve()) and not folder.exists()
    from resource_watchdog_v2 import sample_memory
    admission=[]
    for index in range(3):
        value=sample_memory(os.getpid(),'DEV_saved_text_contract_admission')
        s=value['system_memory'];w=value['worker_private_memory']
        assert s['available'] and w['available'] and value['paired_sampling_within_one_second']
        assert w['private_bytes']+128_000_000<2_000_000_000
        assert (s['commit_used_bytes']+128_000_000)/s['commit_limit_bytes']<.90
        assert s['commit_limit_bytes']-s['commit_used_bytes']-128_000_000>2_000_000_000
        assert shutil.disk_usage(ROOT).free>100_000_000_000
        admission.append(value);time.sleep(.05)
    sys.meta_path.insert(0,NoNumerical())
    import asr18_full_run_serial_DEV_v6 as dev
    import bpe_full_run_lifecycle_v2 as lifecycle
    from bpe_broad_production_state_v1 import digest
    from bpe_training_schedule_v1 import KIND as SCHEDULE_KIND,WEAK,CLEANER
    checks=[]
    def check(value,name):
        assert value,name
        checks.append(name)
    def reject(operation,name):
        try:operation()
        except (ValueError,KeyError,TypeError,AssertionError):checks.append(name);return
        raise AssertionError('Unexpected acceptance: '+name)
    sources={name:record(SOURCE/name) for name in ('asr18_full_run_serial_DEV_v6.py','asr18_full_run_DEV_runtime_v3.py',
        'asr18_optimizer_free_DEV_structure_v2.py','bpe_full_run_lifecycle_v2.py','metrics.py',
        'tsc_bpe_full_decoder_transition_v1.py')}
    result=ROOT/'results/ASR18_feature_window_native20_DEV_01'
    summary_pin=record(result/'summary.json');summary=json.loads((result/'summary.json').read_text(encoding='utf-8'))
    historical=record(AREA/'feature_window_native20_DEV_01/actual_DEV_summary_v1.json')
    check(historical['sha256']=='17b2244761f30d281f933e08f09a7af1af84fbcf0ad1af6df423795ef41421ca',
          'Historical diagnostic350 closure summary independently pinned; no new DEV execution')
    membership=record(AREA/'ROOT_fixed_serial350DEV_manifest_membership_review_v1.json')
    manifests={language:record(path) for language,path in dev.canonical_DEV_paths().items()}
    predictions={language:record(result/('validation_'+language+'.jsonl')) for language in lifecycle.REFERENCES}
    policy={'kind':'ASR18_saved_historical_text_contract_fixture_policy_v1',
        'official_scorer':record(SOURCE/'metrics.py'),'integer_gate_contract':record(SOURCE/'tsc_bpe_full_decoder_transition_v1.py'),
        'reference_words':deepcopy(lifecycle.REFERENCES),'retention_limits':deepcopy(lifecycle.LIMITS),'DEV_manifests':manifests,
        'historical_text_fixture_only':True,'ROOT_execution_approval_claimed':False}
    pointer={'synthetic_in_memory_fixture_pointer':True,'historical_diagnostic_not_promoted':historical}
    report={'kind':dev.REPORT_KIND,'optimizer_free_protocol':dev.OPTIMIZER_FREE_PROTOCOL,'passed':True,
        'serial_optimizer_free':True,'full350_DEV_complete':True,'split':'validation','TEST_used':False,
        'checkpoint_pointer':pointer,'scorer_policy':policy,'predictions':predictions,'scores':summary,
        'all207_weight_only_restore_byte_exact':True,'all207_after_DEV_hashes_unchanged':True,
        'all4_child_RNG_restored_exactly':True,'optimizer_objects_constructed_or_loaded_in_DEV_worker':False,
        'sampler_replay_and_augmentation_objects_created_or_advanced':False,'actual_native_updates_in_DEV':0,
        'Adam_tensor_views_created':0,'selected_saved_CUDA_RNG_states':1,
        'selective_weight_RNG_mapping':{'weights':207,'Adam_tensor_views':0,'optimizer_objects_constructed':0,'tensor_views':209},
        'synthetic_contract_projection_only':True,'actual_new_optimizer_free_restore_executed':False,
        'actual_worker_report':{'synthetic_worker_report_token':True},'actual_owned_DEV_process0':{'synthetic_process0_token':True}}
    check(dev.check_optimizer_free_report_protocol(report) is True,'NEW protocol marker/all207/RNG/zeroAdam metadata contract')
    score=lifecycle.official_DEV(report,pointer,policy)
    expected=dict(tr=367,en=69,es=52,it=24,fr=65,de=73,pt=69)
    check(score['word_errors']==expected,'Unchanged actual official scorer recomputes all7*50 historical saved-text counts')
    check(score['strict_all_six_passed'] is False and score['word_errors']['de']>70 and score['word_errors']['pt']>66,
          'Historical DE/PT failures remain failures; no diagnostic promotion')
    report['official_saved_text_rescore']=deepcopy(score)
    old=deepcopy(report);old['kind']='ASR18_selective_optimizer_free_serial_complete350_DEV_v2'
    reject(lambda:lifecycle.official_DEV(old,pointer,policy),'Preserved DEV5 reportkind failure reproduced against unchanged scorer')
    reject(lambda:dev.check_optimizer_free_report_protocol(old),'New protocol rejects obsolete noncanonical reportkind')
    for key,value,label in (
        ('optimizer_free_protocol','wrong','Missing/wrong optimizer-free marker rejected'),
        ('Adam_tensor_views_created',False,'Boolean Adam viewcount rejected'),
        ('actual_native_updates_in_DEV',True,'Boolean native updatecount rejected'),
        ('selected_saved_CUDA_RNG_states',0,'Missing saved CUDA RNG rejected'),
        ('all207_after_DEV_hashes_unchanged',False,'Changed post-DEV weights rejected')):
        changed=deepcopy(report);changed[key]=value
        reject(lambda changed=changed:dev.check_optimizer_free_report_protocol(changed),label)
    corrupted=deepcopy(report);corrupted['scores']['validation_tr']['asr']['word_errors']+=1
    reject(lambda:lifecycle.official_DEV(corrupted,pointer,policy),'False saved score summary rejected by actual official scorer')
    # Whole consumer path: real FullRunLifecycle.advance -> source-bound serial
    # callback -> NEW verify_DEV method -> unchanged official_DEV -> accept_DEV.
    # Pending process/storage evidence is synthetic and never written as real.
    state={'kind':SCHEDULE_KIND,'broad_cursor':{'population':14,'draws':14,'passes':2},
        'cleaner_cursor':{'population':352768,'draws':0,'passes':1},'phase':WEAK,
        'carried_replay':{'updates':1},'population_mean_U':{WEAK:38.56568679058118,CLEANER:11972886/352768}}
    schedule=SyntheticSchedule(state);source_pin=record(__file__)
    identity={'kind':'ASR18_reviewed_full_two_broad_one_cleaner_run_identity_v1','native_owner':'/root/memory_diagnosis',
        'required_broad_passes':2,'required_cleaner_passes':1,'source_original_clocks':[64250,2000,62250],
        'source_optimizer_RNG_sampler_reset':False,'full_run_source_contract':source_pin,'scorer_policy':source_pin,
        'checkpoint_storage_policy':source_pin,'synthetic_in_memory_fixture_only':True}
    action=lifecycle.next_boundary_request(state,0)
    report_token={'synthetic_report_token':True};storage_token={'synthetic_storage_token':True};request_token={'synthetic_request_token':True}
    request={'checkpoint_pointer':pointer,'identity':identity,'action':action}
    report['source_request']=request_token
    storage_proof={'kind':'ASR18_reviewed_owned_checkpoint_storage_boundary_v1','passed':True,
        'policy':source_pin,'owner':'/root/memory_diagnosis','bounded_publication_and_retirement_implementation_reviewed':True,
        'preserve_qualified_winner_coherent_latest_source_history_DEV_and_journal':True,
        'existing_or_original_artifact_deletion_authorized':False,'unbounded_per_window_checkpoint_retention':False,
        'maximum_additional_owned_bytes':1,'synthetic_in_memory_fixture_only':True}
    original_lifecycle_read=lifecycle.read_bound;original_dev_read=dev.read_bound;original_verified=dev.verified_completed_DEV
    def lifecycle_read(pin,*args,**kwargs):
        if pin==report_token:return deepcopy(report)
        if pin==storage_token:return deepcopy(storage_proof)
        if pin==source_pin:return deepcopy(policy)
        return original_lifecycle_read(pin,*args,**kwargs)
    def dev_read(pin,*args,**kwargs):
        if pin==report_token:return deepcopy(report)
        if pin==request_token:return deepcopy(request)
        return original_dev_read(pin,*args,**kwargs)
    def synthetic_completed(req,worker,process):
        assert req==request_token and worker==report['actual_worker_report'] and process==report['actual_owned_DEV_process0']
        CONTRACT['completion_adapter_calls']+=1
        return deepcopy(report)
    CONTRACT.update(synthetic_only=True,pointer=pointer,report_token=report_token,storage_token=storage_token,
        serial_calls=0,verify_calls=0,winner_calls=0,completion_adapter_calls=0,
        adapter=dev.SerialDEVCallbacks.__new__(dev.SerialDEVCallbacks))
    callbacks={name:owner_unused_callback for name in lifecycle.CALLBACKS}
    callbacks.update(admit_operation=owner_admit_operation,checkpoint_storage_boundary=owner_storage_boundary,
        serial_DEV=owner_serial_DEV,verify_DEV=owner_verify_DEV,preserve_winner=owner_preserve_winner)
    bound_callbacks=lifecycle.SourceBoundOwnerCallbacks({name:source_pin for name in callbacks},callbacks)
    controller=lifecycle.FullRunLifecycle(schedule,identity=identity,callbacks=bound_callbacks)
    controller.protocol.state['last_pointer']=deepcopy(pointer) # explicitly synthetic assumed tail input, not a native proof
    try:
        lifecycle.read_bound=lifecycle_read;dev.read_bound=dev_read;dev.verified_completed_DEV=synthetic_completed
        outcome=controller.advance()
    finally:
        lifecycle.read_bound=original_lifecycle_read;dev.read_bound=original_dev_read;dev.verified_completed_DEV=original_verified
    check(CONTRACT['serial_calls']==CONTRACT['verify_calls']==CONTRACT['winner_calls']==CONTRACT['completion_adapter_calls']==1,
          'Real lifecycle consumes callback report and NEW verify_DEV output exactly once')
    check(outcome['winner_changed'] is False and outcome['retention_early_stop'] is False
          and outcome['declared_training_shortened'] is False,'Failed retention preserves winner/remaining passes')
    check(outcome['next']['kind']=='bounded_training_window' and outcome['next']['boundary']['name']=='broad_pass_2',
          'Actual lifecycle callback output advances to pass2 preparation, never full completion')
    check(digest(schedule.state_dict())==digest(state) and controller.state_dict()['winner'] is None
          and len(controller.state_dict()['completed_boundaries'])==1,'Callback consumption does not change sampler or fabricate a qualified winner')
    check(all(record(pin['path'])==pin for pin in [*predictions.values(),*manifests.values(),summary_pin,historical,membership,*sources.values()]),
          'All350 saved prediction files/manifests/scorer/source/checkpoint summary unchanged')
    check(not any(name.split('.')[0] in {'torch','numpy','scipy','transformers','tokenizers','soundfile','av','pyarrow'}
                  for name in sys.modules),'No numerical/model/audio runtime imported')
    folder.mkdir()
    result={'kind':'ASR18_historical_saved350_text_and_NEW_callback_contract_controls_v1','passed':True,
        'check_count':len(checks),'checks':checks,'producer':{'pid':os.getpid(),'ppid':os.getppid()},
        'sources':sources,'official_historical_counts':score,'historical_summary':historical,
        'historical_predictions':predictions,'canonical_manifests':manifests,'original_summary':summary_pin,'membership_review':membership,
        'callback_consumer_outcome':outcome,'fresh_three_samples':admission,'prospective_increment_bytes':128_000_000,
        'synthetic_report_and_future_process_IO_stayed_in_memory':True,
        'actual_future_process_or_native_boundary_verified':False,'actual_new_optimizer_free_bootstrap_or_DEV_executed':False,
        'historical_checkpoint_promoted':False,'canonical_lifecycle_scorer_modified':False,
        'model_audio_tensor_payload_or_GPU_used':False,'production_enabled':False}
    (folder/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'passed':True,'checks':len(checks),'report':record(folder/'report.json')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--folder',required=True);main(parser.parse_args().folder)
