"""Inactive first-window deterministic replay gate; no numerical imports.

The failed controller stays TOOL1. Only a separately completed new CPU window
may pass, after exact owned retirement and equality to the durable old hashes.
"""
from pathlib import Path
import re
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import digest
from fastmss_phase_aware_plan_bridge_v5 import read_bound
from fastmss_owned_process_closure_v2 import verify as verify_process

FAIL_SHA='7fd685f46de53a8528cb3794bac4b83e9942402456d09012e6568a85bbdd071a'
ARRAY=re.compile(r'u\d{2}_r\d{2}_(features|mask)\.npy')
ROW_KEYS=('context','source_id','language','role','full_target_ids','target_ids_u32_sha256',
    'encoder_frames','clean_PCM_sha256','prepared_PCM_sha256','samples','sample_rate',
    'original_row_sha256','raw_text_sha256','target_tokens','cleaner_original_ordinal')


def exact(a,b,message):require(a==b and digest(a)==digest(b),message)


def validate_retirement(binding,*,prior_failure,guard):
    failure=read_bound(prior_failure,guard)
    require(prior_failure['sha256']==FAIL_SHA and failure['actual_exit_code']==1
        and type(failure['actual_exit_code']) is int and failure['actual_tool_chunk_id']=='69282c'
        and failure['owned_session']==18107 and failure['GPU_launched'] is False
        and failure['actual_Adam_updates']==failure['actual_authoritative_draws_consumed']==0,
        'Preserve exact original TOOL1/no-native failure')
    actual=read_bound(binding,guard)
    require(actual['kind']=='ASR18_actual_exact_owned_failed_window_discard_TOOL0_v2'
        and actual['owner']=='/root/memory_diagnosis' and type(actual['actual_exit_code']) is int
        and actual['actual_exit_code']==0 and type(actual['actual_tool_chunk_id']) is str
        and bool(actual['actual_tool_chunk_id']) and type(actual['native_PID']) is int,
        'Genuine operator-observed new exact-owned cleanup TOOL0 required')
    source=Path(__file__).resolve().parents[2]/'work/research/discard_ASR18_failed_original_CPU_window_v2.py'
    exact(actual['source'],record(source),'Exact reviewed cleanup implementation')
    report=read_bound(actual['report'],guard);closed=read_bound(actual['guard'],guard)
    require(closed['pid']==actual['native_PID'] and type(closed['actual_exit_code']) is int
        and closed['actual_exit_code']==0 and closed['monitor_clean'] is True and closed['monitor_joined'] is True
        and closed['observed']['monitor_error'] is None and type(closed['observed']['samples']) is int
        and closed['observed']['samples']>=3,'Actual same-process clean cleanup guard0')
    require(report['passed'] is True and report['actual_failed_trial']==prior_failure
        and report['source']==actual['source'] and report['original_TOOL1_unchanged'] is True
        and report['shared_feature_slot_empty'] is True and len(report['removed_exact_owned_files'])==681,
        'Real exact-owned retirement, never controller1 relabeled')
    review=read_bound(report['review'],guard)
    require(review['reviewer']=='/root' and review['accepted'] is True
        and review['scope']=='discard_failed_original_CPU_window_after_finalization_runtime_failure_v1'
        and review['production_approval_claimed'] is False and review['source']==actual['source']
        and review['failed_trial']==prior_failure,'Independent exact reviewed cleanup authority')
    archive=read_bound(report['preserved_metadata'],guard)
    require(archive['manifest_binding'] is None and archive['owner_binding']['sha256']==
        'c85d8604e3a184ecab3366da65efa41055e6766f91521ad601a6379b603c4c5b'
        and len(archive['closures'])==20,'Complete original unresolved-window metadata archive')
    return failure,report,archive


def verify_repreparation(*,prior_failure,prior_retirement,new_CPU_actual,new_owner,new_manifest,
                         coordinator_plan,output,guard):
    failure,retirement,archive=validate_retirement(prior_retirement,prior_failure=prior_failure,guard=guard)
    process=read_bound(new_CPU_actual,guard)
    verify_process(process,kind='ASR18_owned_CPU_process_completion_v1',owner='/root/memory_diagnosis',guard=guard)
    prepared=read_bound(process['report'],guard)
    require(prepared['passed'] is True and prepared['owner']==new_owner and prepared['manifest']==new_manifest
        and type(prepared['actual_optimizer_updates']) is int and prepared['actual_optimizer_updates']==0
        and prepared['all_children_actually_closed0'] is True
        and prepared['sampler_journal_frontier_advanced'] is False,'Actual successful NEW prepared-only CPU process0')
    manifest=read_bound(new_manifest,guard);owner=read_bound(new_owner,guard);plan=read_bound(coordinator_plan,guard)
    require(owner['source']==record(Path(__file__).with_name('fastmss_phase_aware_feature_window_v4.py'))
        and owner['window_index']==0 and owner['updates']==20 and owner['prepared_is_committed'] is False
        and manifest['owner']==new_owner and len(manifest['updates'])==20
        and coordinator_plan['sha256']=='6bec4cb6781805def767d0cd4b1e98363845941d6ee96ed4aa18623014e2b36b',
        'NEW sink4 actual owner and identical source0 plan, never owner3 alias')
    original=read_bound(failure['retained_feature_inventory'],guard)
    old={Path(pin['path']).name:pin for pin in original['actual_files'] if ARRAY.fullmatch(Path(pin['path']).name)}
    require(len(old)==640,'All640 original full NPY file hashes remain durable')
    names=set();rows=0
    for i,part in enumerate(manifest['updates']):
        guard();new=read_bound(part['update'],guard);previous=archive['metadata']['update_%02d.json'%i]
        for key in ('batch_id','native_transaction_sha256','before_sampler_sha256','after_sampler_sha256','prepared_payload'):
            exact(new[key],previous[key],'Deterministic prepared update '+key+' changed')
        require(len(new['rows'])==len(previous['rows'])==16,'Full14TR+2original replay rows retained')
        exact(new['native_transaction_sha256'],plan['updates'][i]['native_transaction_sha256'],'Original actual draw transaction')
        for current,prior in zip(new['rows'],previous['rows']):
            exact({k:current[k] for k in ROW_KEYS},{k:prior[k] for k in ROW_KEYS},
                'Context/fulltargets/clean+prepared PCM/raw source replay changed')
            for key in ('input_features','attention_mask'):
                c,p=current[key],prior[key];path=Path(c['file']['path']);name=path.name
                require(name in old and name not in names and path.resolve().parent==Path(new_owner['path']).resolve().parent,
                    'One exact new-owned array for every original name')
                exact({k:c[k] for k in ('shape','dtype','array_sha256')},
                    {k:p[k] for k in ('shape','dtype','array_sha256')},'Feature/mask numerical identity changed')
                actual=record(path);exact(actual,c['file'],'New actual NPY bytes differ from closed producer')
                require(actual['bytes']==old[name]['bytes'] and actual['sha256']==old[name]['sha256'],
                    'All640 full original NPY files must be BYTE EXACT before any native work')
                names.add(name)
            rows+=1
    require(names==set(old) and rows==320,'No original array/row omitted or duplicated')
    lease=read_json(Path(__file__).resolve().parents[2]/'work/gpu_training_owner.json')
    require(lease['owner']=='/root/memory_diagnosis' and lease['actual_GPU_running'] is False
        and lease['actual_owned_worker_pid'] is None,'No GPU work before exact re-preparation proof')
    output=Path(output).resolve();require(output.is_relative_to(Path(r'E:\gpu_training_scratchpad').resolve())
        and not output.exists(),'Fresh E-only actual comparison artifact')
    result={'kind':'ASR18_actual_first_window_repreparation_byte_equivalence_v1','passed':True,
        'source':record(__file__),'prior_failed_trial':prior_failure,'prior_actual_owned_retirement0':prior_retirement,
        'original_array_inventory':failure['retained_feature_inventory'],'new_CPU_actual0':new_CPU_actual,
        'new_owner':new_owner,'new_manifest':new_manifest,'original_source0_plan':coordinator_plan,
        'byte_exact_full_NPY_files':640,'exact_full_rows':320,'exact_update_payloads':20,
        'actual_GPU_or_Adam_or_authoritative_draws':0,'old_controller_TOOL1_preserved':True,'production_enabled':False}
    write_json(output,result);return record(output)
