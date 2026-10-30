"""ONE inactive CURRENT overlay9 ordered14 original-ingress observation.

No tracing hooks, local/frame introspection, baseline, frontend, transform, model,
GPU, new draw or production action. Stop at FIRST propagated runtime exception.
All14 success is explicitly INCONCLUSIVE, never an invitation to retry.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import time
import traceback
import types
import weakref

import psutil

P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20')
S=P/'outputs/parakeet_turkish';sys.path[:0]=[str(S),str(P/'work/research')]
F=Path(__file__).parent
OUTPUT=F/'recorded186564_ordered14_current_decoder_actual_01'
INPUTS=F/'recorded186564_ordered14_current_decoder_frozen_inputs_v1.json'
SCOPE='ASR18_source36_saved186400_recorded186564_ordered14_current_decoder_CPU_v1'
PRIVATE=8000000000;PREFETCH=268435456;PROSPECTIVE=2500000000


def require(value,message):
    if not value:raise ValueError(message)


def pin(path):
    path=Path(path).resolve();before=path.stat()
    require(before.st_size<=8*1024**2,'Compact/source input only, never journal/payload/source media pin')
    data=path.read_bytes();after=path.stat()
    require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Pinned input changed')
    return {'path':str(path),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def canonical(value):return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf-8')
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def identity():
    process=psutil.Process();return {'pid':process.pid,'create_time':process.create_time(),'ppid':process.ppid()}


def write_new(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n')as file:
        json.dump(value,file,ensure_ascii=False,indent=2,allow_nan=False);file.write('\n');file.flush();os.fsync(file.fileno())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--ROOT-review',required=True);args=parser.parse_args()
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    review=read(args.ROOT_review);fixed=read(INPUTS)
    require(review['accepted']is True and review['reviewer']=='/root'and review['scope']==SCOPE
        and review['fixture_source']==pin(__file__)and review['frozen_inputs']==pin(INPUTS)
        and Path(review['output_path']).resolve()==OUTPUT.resolve(),'Exact ROOT current fixture/input/output scope')
    require(review['input_pins']==fixed['input_pins'],'Exact frozen compact/source input set')
    for value in fixed['input_pins']:require(pin(value['path'])==value,'Every frozen input before import/decode')
    require(not OUTPUT.exists(),'ONE fresh bounded observation, preserve prior outcomes')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'and sys.flags.utf8_mode==1
        and all(os.environ.get(k)=='1'for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'))
        and all(Path(os.environ[k]).resolve().is_relative_to(Path('E:/gpu_training_scratchpad').resolve())
            for k in ('TEMP','TMP','HF_HOME','TORCH_EXTENSIONS_DIR')),'Original E CPU-only environment/caps')
    candidate=read(fixed['candidate']['path']);config=read(fixed['configuration']['path'])
    plan=read(candidate['plan']['path']);update=plan['updates'][3];rows=candidate['ordered14_TR_candidates']
    require(pin(candidate['plan']['path'])==candidate['plan']and digest(plan)==candidate['canonical_plan_sha256']
        and update==candidate['complete_original_update']and update['local_update_index']==3
        and update['global_update']==186564 and candidate['original_array_position_index']==3
        and candidate['next_CPU_preparation_global_update']==186564
        and update['phase']=='weak_label_continued_training'and rows==update['rows'][:14]and len(rows)==14,
        'Exact recorded plan389/index3/global186564, no successor guessed from published frontier')
    policies=[];policy=config['original_clip_geometry_overlay']
    while policy is not None:
        certificate=read(policy['certificate']['path']);raw=certificate['original_row']
        require(all(v['ordinal']!=raw['ordinal']and v['source_id']!=raw['unique_id']for v in rows),
            'All14 nonmatch BOTH dispatch keys of each existing certificate')
        policies.append({'source':policy['source'],'certificate':policy['certificate'],'ROOT_review':policy['ROOT_review'],
            'ordinal':raw['ordinal'],'unique_id':raw['unique_id']});policy=policy.get('prior_overlay')
    require(len(policies)==8 and policies==fixed['eight_overlay_nonmatches'],'Current overlay9/EIGHT exact chain')
    from fastmss_resident_feature_stream_v13 import AvailableMemoryPolicy,_raw_finish_without_full_heap_collection,checked_resource_profile
    import fastmss_BPE9223_cpu_common_v12 as core
    from fastmss_original_clip_geometry_overlay_v9 import OriginalRows
    require(checked_resource_profile(config)==fixed['original_resource_policy'],'Exact CURRENT CPU13 resource policy')
    core.environment();raw_contract=core.verify_raw_review(config['raw_lifetime_review'])
    memory=AvailableMemoryPolicy(private_limit_bytes=PRIVATE)
    require(memory.prospective_reserve_bytes==PROSPECTIVE and memory.allocation_wait_seconds==300,
        'Original CPU2.5GB prospective/private8GB/allocation300s retained')
    began=time.monotonic();guard_error=None
    def guard():
        nonlocal guard_error
        try:
            require(time.monotonic()-began<180,'Finite ordered14 current decoder wall180s')
            value=memory();system=value['system_memory']
            require(system['commit_available_bytes']>=2000000000+PREFETCH
                and system['physical_available_bytes']>=1500000000+PREFETCH
                and shutil.disk_usage('E:/').free>=100000000000,'Original floors + FULL prefetch/E100GB')
            return value
        except BaseException as error:guard_error=repr(error);raise
    initial_sample=guard();core.numpy_runtime(frontend=False);core.admit=memory.admit
    validated=deepcopy(fixed['source_derived_validated_metadata_context'])
    require(validated['population_evidence']==config['planning_arguments']['DATA_evidence']
        and validated['retained_rows']==2631115 and validated['vocabulary']==9223,
        'Closed immutable original DATA +9223 context, no new DATA audit')
    normalized=core.bridge.normalize_updates(plan,population=validated['retained_rows'],guard=guard)
    require(normalized[3]['coordinator_update']==update and normalized[3]['rows']==fixed['expected_normalized_rows'],
        'Actual original pure normalizer agrees with complete frozen targets/contexts, no authority draw')
    OUTPUT.mkdir();producer=identity();results=[];current=None;reader=None;failure=None;cleanup_error=None
    report={'kind':SCOPE,'passed':False,'producer':producer,'ROOT_review':pin(args.ROOT_review),
        'input_pins_before':fixed['input_pins'],'eight_overlay_dispatch_nonmatches':policies,
        'execution_path':'CURRENT CPU13 OriginalRows overlay9->7->6->5->4->3->2->1->core; real resolve/locks/full labels/decode/geometry/hash/heldout; no direct decoder substitution',
        'validated_context_basis':fixed['validated_context_basis'],'initial_resource_sample':initial_sample,
        'results':results,'first_actual_exception':None,'observation_outcome':'INCOMPLETE',
        'all14_decodes_passed':False,'no_trace_hooks_frame_or_local_introspection':True,
        'no_sampler_authority_RNG_journal_frontend_transform_model_GPU':True,'production_admitted':False,
        'no_fullmember_baseline_tolerance_or_source_change':True,'reader_metadata_close_returned':False}
    write_new(OUTPUT/'producer_identity.json',producer)
    try:
        reader=OriginalRows(validated,geometry_overlay=config['original_clip_geometry_overlay'],raw_contract=raw_contract,
            private_limit_bytes=PRIVATE,frontend_enabled=True,guard=guard)
        reader.ledger.finish=types.MethodType(_raw_finish_without_full_heap_collection,reader.ledger)
        for j,(original,expected,context)in enumerate(zip(rows,normalized[3]['rows'][:14],update['contexts'][:14])):
            guard();row=deepcopy(original);row['audio_draw_context']=deepcopy(context);row['augmentation_role']=expected['role']
            key=str(j+1).zfill(2);declaration=original['declared_geometry'];budget=fixed['original_per_row_prospective'][j]
            current={'global_update':186564,'plan':candidate['plan'],'canonical_plan_sha256':candidate['canonical_plan_sha256'],
                'local_update_index':3,'original_update_row_1based':j+1,'ordinal':original['ordinal'],
                'source_id':original['source_id'],'dataset':original['dataset'],'original_plan_row':deepcopy(original),
                'runtime_call_row':deepcopy(row),'raw_original_annotation':deepcopy(original['broad_original']),
                'phase':update['phase'],'expected_argument':deepcopy(expected),'full_target_ids':deepcopy(expected['full_target_ids']),
                'declared_geometry':deepcopy(declaration),'original_raw_and_full_numerical_prospective':budget,
                'call':'CURRENT overlay9 OriginalRows(row, expected, exact recorded phase)',
                'call_preceded_by_fsynced_exact_identity_and_full_targets':True}
            write_new(OUTPUT/('before_row_'+key+'.json'),current)
            try:
                wave=reader(row,expected,update['phase'])
            except BaseException as error:
                # The propagated exception is authoritative. Copy only text and
                # explicitly owned metadata; never traverse or retain frames,
                # traceback objects, caller locals, errors, or waveform arrays.
                text=traceback.format_exc()
                failure={'identity':deepcopy(current),'exception_type':type(error).__name__,
                    'exception_module':type(error).__module__,'exception_message':str(error),'exception_repr':repr(error),
                    'traceback_text':text,'guard_error':guard_error,
                    'propagated_from_current_runtime_ingress_call':True,
                    'original_decoder_source_in_propagated_traceback_text':'fastmss_bounded_original_audio_v2.py'in text,
                    'no_trace_event_or_handled_internal_exception_used':True}
                write_new(OUTPUT/'first_actual_exception.json',failure)
                report['first_actual_exception']=failure
                report['observation_outcome']='LOCALIZED_FIRST_PROPAGATED_ORIGINAL_DECODER_EXCEPTION'if guard_error is None and failure['original_decoder_source_in_propagated_traceback_text']else'FAILED_RESOURCE_OR_NONDECODER_INGRESS_EXCEPTION'
                if guard_error is not None or not isinstance(error,Exception):raise
                break
            else:
                wave_ref=weakref.ref(wave)
                shape=[int(v)for v in wave.shape]
                scalar={'dtype':str(wave.dtype),'shape':shape,'sample_rate':16000,'samples':int(len(wave)),
                    'bytes':int(wave.nbytes),'C_contiguous':bool(wave.flags.c_contiguous)}
                reference=core.reference_entry(row,expected,reader)
                require(reference['full_target_ids']==expected['full_target_ids']and reference['context']==context
                    and reference['expected_audio']['samples']==len(wave),'Actual original clean-reader reference full targets/context')
                p=reader.ledger.jobs[reader.active_job]['plan']
                work=p['workspace_incremental_envelopes']
                require(p==budget['raw_plan']and p['raw_peak_bytes']+p['compressed_media_cap_bytes']+max(work.values())+raw_contract['numerical_runtime_reserve_bytes']==budget['prospective_bytes'],
                    'Actual raw plan equals frozen historical full transform/frontend/numerical envelope')
                del reference,wave
                reader.finish_raw_row();require(wave_ref()is None,'Actual raw return buffer dead before retirement')
                receipt=reader.raw_receipt()
                record={'original_update_row_1based':j+1,'ordinal':original['ordinal'],'source_id':original['source_id'],
                    'scalar_waveform_metadata':scalar,'full_targets_and_context_matched':True,
                    'raw_return_weakref_dead':True,'original_raw_plan':p,'raw_ledger_success_receipt':receipt}
                write_new(OUTPUT/('success_row_'+key+'.json'),record);results.append(record);current=None
        else:
            report['all14_decodes_passed']=True;report['observation_outcome']='INCONCLUSIVE_ALL14_SUCCEEDED_NO_AUTOMATIC_RETRY'
    except BaseException as error:
        report['protocol_exception']={'exception_type':type(error).__name__,'exception_message':str(error),
            'exception_repr':repr(error),'traceback_text':traceback.format_exc(),'identity':deepcopy(current)}
        if failure is None:report['observation_outcome']='FAILED_PROTOCOL_RESOURCE_OR_SOURCE_BINDING'
        raise
    finally:
        if reader is not None:
            report['raw_ledger_final_scalar_state']={'failed':bool(reader.ledger.failed),'active_job':reader.active_job,
                'unretired_jobs':len(reader.ledger.jobs),'tracked_return_refs':sum(len(v['arrays'])for v in reader.ledger.jobs.values()),
                'live_return_refs':sum(ref()is not None for v in reader.ledger.jobs.values()for _,ref,_ in v['arrays']),
                'events':deepcopy(reader.ledger.events),'failed_ledger_NOT_fabricated_as_success_or_retired':bool(reader.ledger.failed)}
            locks=reader.locks
            try:
                reader.close();report['reader_metadata_close_returned']=True
                report['original_metadata_connections_empty']=reader.connections==[]
                report['original_metadata_lock_handles_empty']=locks is not None and locks.handles==[] and reader.locks is None
            except BaseException as error:
                cleanup_error={'repr':repr(error),'traceback_text':traceback.format_exc()}
            reader=None;locks=None
        current=None
        report['cleanup_error']=cleanup_error;report['CPU_elapsed_seconds']=time.monotonic()-began
        report['CPU_peak_private_bytes']=memory.peak_private;report['allocation_wait_count']=memory.allocation_wait_count
        report['allocation_wait_seconds_total']=memory.allocation_wait_seconds_total
        report['last_resource_sample']=memory.last_observed;report['guard_error']=guard_error
        report['input_pins_after']=[pin(value['path'])for value in fixed['input_pins']]
        report['input_pins_before_after_exact']=report['input_pins_after']==fixed['input_pins']
        report['torch_imported']='torch'in sys.modules
        report['passed']=bool(guard_error is None and cleanup_error is None and not report.get('protocol_exception')
            and report['input_pins_before_after_exact']and report['reader_metadata_close_returned']
            and report.get('original_metadata_connections_empty')and report.get('original_metadata_lock_handles_empty')
            and not report['torch_imported']and report['observation_outcome']in(
                'LOCALIZED_FIRST_PROPAGATED_ORIGINAL_DECODER_EXCEPTION','INCONCLUSIVE_ALL14_SUCCEEDED_NO_AUTOMATIC_RETRY'))
        report['passed_means_bounded_observation_completed_NOT_audio_success_cause_or_remedy_proof']=True
        write_new(OUTPUT/'report.json',report)
        write_new(OUTPUT/'CPU_completion.json',{'producer':producer,'report':pin(OUTPUT/'report.json'),
            'actual_process_return_still_unobserved_inside_child':True,'elapsed_seconds':time.monotonic()-began})
    print(json.dumps({'report':pin(OUTPUT/'report.json'),'diagnostic_completed':report['passed'],
        'outcome':report['observation_outcome'],'preceding_successes':len(results),
        'first_failed_row_1based':failure['identity']['original_update_row_1based']if failure else None}),flush=True)
    require(report['passed'],'Preserve failed/inconclusive observation honestly; no automatic retry or promotion')


if __name__=='__main__':main()
