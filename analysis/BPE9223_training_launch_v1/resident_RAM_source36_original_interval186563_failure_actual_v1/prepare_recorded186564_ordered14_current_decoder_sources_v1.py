"""Inactive metadata/source preparation only: never import or run candidate ingress."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct
import sys
import os
import shutil
import psutil

P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish'
N=Path('E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1')
F=N/'resident_RAM_source36_original_interval186563_failure_actual_v1'
OLD=N/'resident_RAM_original_interval170469_failure_actual_v1'
B=Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01')
I=B/'invocations/owner_000025'
FIXTURE=F/'observe_recorded186564_ordered14_current_decoder_v1.py'
EXTERNAL=F/'launch_recorded186564_ordered14_current_decoder_external_v1.py'
INPUTS=F/'recorded186564_ordered14_current_decoder_frozen_inputs_v1.json'
SCOPE='ASR18_source36_saved186400_recorded186564_ordered14_current_decoder_CPU_v1'


def require(value,message):
    if not value:raise ValueError(message)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def canonical(value):return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf-8')
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def pin(path):
    p=Path(path).resolve();before=p.stat();require(before.st_size<=8*1024**2,'Only compact metadata/source, not media/journal/payload')
    data=p.read_bytes();after=p.stat();require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Preparation input stable')
    return {'path':str(p),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
def sig(path):
    p=Path(path).resolve();st=p.stat();return {'path':str(p),'bytes':st.st_size,'mtime_ns':st.st_mtime_ns}
def encoded(value):return (json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8')

psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
sys.path.insert(0,str(S))
from resource_watchdog_v2 import sample_memory
def preparation_guard():
    value=sample_memory(os.getpid(),'inactive_current186564_decoder_source_metadata_preparation')
    require(value['paired_sampling_within_one_second']and value['worker_private_memory']['private_bytes']<=536870912
        and value['system_memory']['commit_available_bytes']>=5055306368
        and value['system_memory']['physical_available_bytes']>=2305306368
        and shutil.disk_usage('E:/').free>=100000000000,'Metadata-only preparation full prospective/floors/E guard')
    return value
preparation_initial=preparation_guard()


# The sole admitted preparation process only reads compact/source inputs and
# serializes inactive files. No candidate module is imported or executed here.
scope_pin=pin(F/'ROOT_source36_current186564_ordered_decoder_inactive_preparation_scope_v1.json')
require(scope_pin['bytes']==4121 and scope_pin['sha256']=='854ec6f256290806eed3c87d4a8e53ff994ef8d2b959d526420de12229841bd5','Exact preparation-only scope')
candidate_pin=pin(F/'current_recorded_CPU186564_plan389_index3_ordered14_candidates_v1.json')
candidate=read(candidate_pin['path']);config_pin=pin(I/'configuration.json');config=read(config_pin['path'])
plan_pin=candidate['plan'];require(pin(plan_pin['path'])==plan_pin,'Accepted current plan pin')
plan=read(plan_pin['path']);update=plan['updates'][3];rows=candidate['ordered14_TR_candidates']
require(digest(plan)==candidate['canonical_plan_sha256'] and update==candidate['complete_original_update']
    and candidate['original_array_position_index']==3 and candidate['next_CPU_preparation_global_update']==186564
    and update['local_update_index']==3 and update['global_update']==186564
    and update['phase']=='weak_label_continued_training' and rows==update['rows'][:14]
    and len(rows)==14 and len(update['rows'])==16,'Current exact recorded 389/index3/186564 ordered14')

# Reconstruct ONLY the immutable metadata context returned by the original
# validate_CPU_population source. This reuses closed source/label verification;
# it is NOT another DATA audit or a claim that the old validation ran again.
evidence=config['planning_arguments']['DATA_evidence']
old_pin=evidence['original_DATA_evidence']['validated'];old=read(old_pin['path'])
runtime_pin=evidence['BPE9223_population'];runtime=read(runtime_pin['path'])
geometry_pin=runtime['declared_geometry'];geometry=read(geometry_pin['path'])
bindings=deepcopy(old['bindings'])
bindings.update(final_labels={k:runtime['databases']['broad_labels'][k]for k in('path','bytes','sha256')},
    declared_geometry_index=runtime['databases']['declared_geometry'],cleaner_labels=runtime['databases']['cleaner_labels'])
signatures=[sig(bindings[k]['path'])for k in('final_catalog','final_labels','declared_geometry_index')]
for value,signature in zip([runtime['source_catalog_unchanged'],runtime['databases']['broad_labels'],runtime['databases']['declared_geometry']],signatures):
    require(signature=={k:value[k]for k in('path','bytes','mtime_ns')},'Original current immutable DB signatures; no full DB hash/read')
validated={'kind':'ASR18_validated_9223_training_population_v1','passed':True,'complete':True,'vocabulary':9223,'blank_id':8192,
    'retained_rows':2631115,'label_rows':2631115,'frontier':2631115,'total_target_tokens':90746310,
    'population_mean_U':90746310/2631115,'cleaner_rows':352768,'cleaner_target_tokens':10621773,
    'cleaner_mean_U':10621773/352768,'bindings':bindings,
    'declared_geometry':{'passed':True,'TRAIN_admitted':False,'native_allocation_or_encoder_fit_approved':False,
        'scalar_audit':{'passed':True,'population':geometry['population'],'source_signatures':signatures},
        'new_geometry_binding':geometry_pin},
    'runtime_bindings':deepcopy(runtime),'new_population':runtime_pin,'population_evidence':deepcopy(evidence),
    'heldout_reference_PCM_binding_complete':True,'production_enabled':False,'TRAIN_admitted':False,
    'native_envelope_or_forecast_approved':False,'actual_optimizer_updates':0}
require(geometry['geometry_index']==runtime['databases']['declared_geometry']
    and geometry['broad_labels']==runtime['databases']['broad_labels']
    and geometry['population']['retained_rows']==2631115
    and geometry['population']['total_target_tokens']==90746310,'Closed current 9223 metadata linkage')

policies=[];policy_pins=[];policy=config['original_clip_geometry_overlay']
while policy is not None:
    cert=read(policy['certificate']['path']);raw=cert['original_row']
    require(all(v['ordinal']!=raw['ordinal']and v['source_id']!=raw['unique_id']for v in rows),'Every current candidate nonmatches both keys of each existing certificate')
    policies.append({'source':policy['source'],'certificate':policy['certificate'],'ROOT_review':policy['ROOT_review'],
        'ordinal':raw['ordinal'],'unique_id':raw['unique_id']})
    policy_pins.extend((policy['source'],policy['certificate'],policy['ROOT_review']));policy=policy.get('prior_overlay')
require(len(policies)==8 and Path(policies[0]['source']['path']).name=='fastmss_original_clip_geometry_overlay_v9.py','Exact current overlay9/EIGHT chain')

# Original bridge normalize_updates entry formula, frozen as a comparison
# oracle only. The future fixture invokes the actual original normalizer and
# requires byte-identical expected entries before its first runtime reader call.
expected=[]
for j,(row,context)in enumerate(zip(update['rows'],update['contexts'])):
    ids=row['tr_metadata_labels']if row['language']=='tr'else row['clean_target_ids']
    expected.append({'row_index':j,'original_row_sha256':digest(row),'source_id':row['source_id'],'language':row['language'],
        'context':deepcopy(context),'full_target_ids':deepcopy(ids),
        'target_ids_u32_sha256':hashlib.sha256(b''.join(struct.pack('<I',v)for v in ids)).hexdigest(),
        'raw_text_sha256':hashlib.sha256(row['text'].encode('utf-8')).hexdigest(),
        'role':'student_tr'if j<14 else'replay','route':'frozen_weak_TR_augmentation_CPU_only'if j<14 else'direct_original_clean',
        'independent_clean_PCM_still_required':True})

raw_review=config['raw_lifetime_review'];budgets=[]
for j,row in enumerate(rows):
    d=row['declared_geometry'];c=d['clean16k_samples'];n=d['native_samples'];sr=d['native_sr']
    ch=1 if row['dataset']=='kiraat'else raw_review['maximum_runtime_channels'];frame=raw_review['maximum_decoder_frame_samples']
    require(c==(n*16000+sr-1)//sr and sr in(16000,24000,48000),'Original declared integer geometry')
    hold=4*c;frame_raw=ch*frame*8*6+frame*4*2;decode=4*n+frame_raw+4*(c+128)+256*1024
    weak=c<=120*16000;retained=hold+(hold if weak else 0);transform=32*c+1024**2 if weak else 0
    peaks={'decode_resample':hold+decode,'canonical_hash':hold+65536*8,
        'weak_transform':retained+transform,'original_frontend_waveform':retained+24*c+64*1024}
    fft=1<<(max(1,c+24000-1)-1).bit_length()
    work={'decoder':8*1024**2,'weak_transform_FFT':6*16*fft if weak else 0,
        'original_frontend_spectral':(c//160+1)*257*8*10+8*1024**2}
    original_plan={'kind':'FastMSS_preallocation_raw_lifetime_plan_v1','native_samples':n,'native_rate':sr,
        'clean16k_samples':c,'channels_bound':ch,'frame_samples_bound':frame,'stage_raw_peaks':peaks,
        'raw_peak_bytes':max(peaks.values()),'clean_hold_bytes':hold,'prepared_hold_bytes':hold if weak else 0,
        'workspace_incremental_envelopes':work,'compressed_media_cap_bytes':33554432,'maximum_live_raw_jobs':2,
        'weak_transform_possible':weak,'source_excluded_or_cropped':False,'draw_consumed':False}
    require(max(peaks.values())<=raw_review['raw_capacity_bytes']and work['decoder']<=raw_review['decoder_workspace_allowance_bytes']
        and work['weak_transform_FFT']<=raw_review['transform_workspace_allowance_bytes']
        and work['original_frontend_spectral']<=raw_review['frontend_workspace_allowance_bytes'],'Original full numerical/raw support retained')
    budgets.append({'original_update_row_1based':j+1,'source_id':row['source_id'],'raw_plan':original_plan,
        'prospective_bytes':max(peaks.values())+33554432+max(work.values())+raw_review['numerical_runtime_reserve_bytes']})
maximum=max(v['prospective_bytes']for v in budgets)

old_fixture=OLD/'observe_recorded170470_ordered14_decoder_v1.py';old_external=OLD/'launch_recorded170470_ordered14_decoder_external_v1.py'
old_text=old_fixture.read_bytes().decode('utf-8');fixture_text=old_text;fixture_changes=[]
def change(which,before,after,count=1):
    global fixture_text,external_text
    text=fixture_text if which=='fixture'else external_text
    require(text.count(before)==count,'Exact declared replacement count: '+before[:90])
    text=text.replace(before,after)
    if which=='fixture':fixture_text=text;fixture_changes.append({'old_literal':before,'new_literal':after,'expected_count':count})
    else:external_text=text;external_changes.append({'old_literal':before,'new_literal':after,'expected_count':count})

change('fixture',old_text[:old_text.index('import argparse')],'''"""ONE inactive CURRENT overlay9 ordered14 original-ingress observation.

No tracing hooks, local/frame introspection, baseline, frontend, transform, model,
GPU, new draw or production action. Stop at FIRST propagated runtime exception.
All14 success is explicitly INCONCLUSIVE, never an invitation to retry.
"""
''')
change('fixture','from collections import deque\nimport gc\n','from copy import deepcopy\n')
change('fixture','import hashlib\nimport inspect\nimport json\n','import hashlib\nimport json\n')
change('fixture','import shutil\n','import shutil\nimport struct\n')
change('fixture','import types\n','import types\nimport weakref\n')
change('fixture',"OUTPUT=F/'ordered14_decoder_actual_01'","OUTPUT=F/'recorded186564_ordered14_current_decoder_actual_01'")
change('fixture',"INPUTS=F/'ordered14_decoder_frozen_inputs_v1.json'","INPUTS=F/'recorded186564_ordered14_current_decoder_frozen_inputs_v1.json'")
change('fixture',"SCOPE='ASR18_source31_saved170400_ordered14_original_decoder_reproduction_CPU_v1'",'SCOPE='+repr(SCOPE))
new_body='''def main():
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
'''
change('fixture',old_text[old_text.index('def observe_original_call'):],new_body)

external_text=old_external.read_bytes().decode('utf-8');external_changes=[]
change('external','ONE ordered14 original decoder reproduction, no frontend/model/GPU/production; external original full guards.',
    'ONE CURRENT overlay9 ordered14 decoder observation; external original full guards, no production action.')
change('external',"FIXTURE=F/'observe_recorded170470_ordered14_decoder_v1.py'","FIXTURE=F/'observe_recorded186564_ordered14_current_decoder_v1.py'")
change('external',"RESULT=F/'ordered14_decoder_external_result_v1.json'","RESULT=F/'recorded186564_ordered14_current_decoder_external_result_v1.json'")
change('external',"OUTPUT=F/'ordered14_decoder_actual_01'","OUTPUT=F/'recorded186564_ordered14_current_decoder_actual_01'")
change('external','PROSPECTIVE=903023104','PROSPECTIVE='+str(maximum))
change('external',"'ASR18_source31_saved170400_ordered14_original_decoder_reproduction_CPU_v1'",repr(SCOPE))
change('external',"ordered14_decoder_frozen_inputs_v1.json","recorded186564_ordered14_current_decoder_frozen_inputs_v1.json",2)
change('external',"sample_memory(proc.pid,'ordered14_original_decoder_external_CPU')","sample_memory(proc.pid,'recorded186564_current_overlay9_decoder_external_CPU')")
change('external',"stdout=F/'ordered14_decoder_stdout.log';stderr=F/'ordered14_decoder_stderr.log'",
    "stdout=F/'recorded186564_ordered14_current_decoder_stdout.log';stderr=F/'recorded186564_ordered14_current_decoder_stderr.log'")
change('external',"'ASR18_ordered14_original_decoder_external_result_v1'","'ASR18_recorded186564_current_overlay9_ordered14_external_result_v1'")
change('external','(2456,38964,26892,41672,34044,40012,31044)','(26892,30128,21912,37108,29600,39856,35440)',2)

fixture_bytes=fixture_text.encode('utf-8');external_bytes=external_text.encode('utf-8')
ast.parse(fixture_bytes);ast.parse(external_bytes)
old_defs={v.name:ast.dump(v,include_attributes=False)for v in ast.parse(old_fixture.read_bytes()).body if isinstance(v,ast.FunctionDef)}
new_defs={v.name:ast.dump(v,include_attributes=False)for v in ast.parse(fixture_bytes).body if isinstance(v,ast.FunctionDef)}
require(all(new_defs[name]==old_defs[name]for name in('require','pin','read','canonical','digest','identity','write_new')),'All seven retained metadata primitives exact AST')
for text,old_bytes,changes in((fixture_text,old_fixture.read_bytes(),fixture_changes),(external_text,old_external.read_bytes(),external_changes)):
    inverse=text
    for item in reversed(changes):
        require(inverse.count(item['new_literal'])==item['expected_count'],'Exact inverse literal count')
        inverse=inverse.replace(item['new_literal'],item['old_literal'])
    require(inverse.encode('utf-8')==old_bytes,'ENTIRE original RAW bytes reconstructed')

source_names=('fastmss_resident_feature_stream_v13.py','fastmss_original_clip_geometry_overlay_v9.py',
    'fastmss_bounded_original_audio_v2.py','fastmss_raw_lifetime_v1.py','fastmss_BPE9223_cpu_common_v12.py',
    'fastmss_BPE9223_plan_bridge_v12.py','fastmss_BPE9223_feature_engine_v12.py','fastmss_singlepass_CPU_v1.py',
    'asr18_BPE9223_source_context_v9.py','bpe_runtime_declared_geometry_v1.py','bpe_declared_geometry_index_v1.py',
    'bpe_windows_final_metadata_lock_v1.py','bpe_broad_audio_v1.py','bpe_broad_audio_FLAC_records_v1.py',
    'bpe_broad_source_identity_v1.py','bpe_broad_production_state_v1.py','resource_watchdog_v2.py')
values=[scope_pin,candidate_pin,config_pin,plan_pin,
    pin(F/'ROOT_source36_failure186563_saved186400_actual_independent_verification_v1.json'),
    pin(F/'ROOT_source36_current_plan389_index3_186564_candidate_localization_review_v1.json'),
    pin(N/'resident_RAM_recurrent_metadata_slowdown_fix_v1/one_postDEV_gc_refresh_v1/ROOT_source36_standard186400_metadata_and_PRE_SAVED_equality_verification_v1.json'),
    old_pin,runtime_pin,geometry_pin,bindings['source_identity_inventory'],bindings['heldout_PCM'],bindings['preparation_plan'],
    evidence['BPE9223_report'],evidence['BPE9223_actual0'],
    pin('E:/gpu_training_scratchpad/storage_config.json'),pin('E:/gpu_training_scratchpad/use_storage.ps1'),
    pin('E:/gpu_training_scratchpad/scratch/BPE_broad_CPU_preflight_env_v2/Scripts/python.exe'),
    pin(old_fixture),pin(old_external),pin(P/'work/research/run_ASR18_BPE9223_resident_RAM_training_v36.py')]
values+=policy_pins+[pin(S/name)for name in source_names]
unique={}
for value in values:
    exact={k:value[k]for k in('path','bytes','sha256')};actual=pin(value['path'])
    require(actual==exact,'Frozen compact/source input exact: '+value['path']);unique[actual['path']]=actual
inputs=list(unique.values())
frozen={'kind':'ASR18_INACTIVE_recorded186564_current_overlay9_ordered14_inputs_v1','candidate':candidate_pin,
    'configuration':config_pin,'plan':plan_pin,'input_pins':inputs,'eight_overlay_nonmatches':policies,
    'original_resource_policy':config['resident_RAM_resource_scheduling_policy'],'original_raw_contract':raw_review,
    'source_derived_validated_metadata_context':validated,'expected_normalized_rows':expected,
    'original_per_row_prospective':budgets,'maximum_original_prospective_bytes':maximum,
    'current_DB_STAT_signatures_only':signatures,
    'validated_context_basis':{'source':pin(S/'asr18_BPE9223_source_context_v9.py'),'function':'validate_CPU_population',
        'original_closed_DATA_validated':old_pin,'closed_9223_runtime':runtime_pin,'closed_geometry_metadata':geometry_pin,
        'same_original_return_schema_values_from_closed_metadata':True,'fresh_original_validation_execution':False,
        'no_full_DB_hash_or_corpus_reaudit':True,'actual_OriginalRows_locks_STAT_SQL_full_targets_still_required':True,
        'actual_original_normalizer_before_decode_required':True}}
def future_pin(path,data):return {'path':str(path.resolve()),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
fixture_pin=future_pin(FIXTURE,fixture_bytes);external_pin=future_pin(EXTERNAL,external_bytes);input_bytes=encoded(frozen)
ledger_path=F/'recorded186564_ordered14_current_decoder_full_raw_inverses_v1.json'
ledger={'kind':'ASR18_INACTIVE_full_RAW_source_inverse_ledger_v1','newline_normalization':'NONE; original exact LF bytes retained',
    'sources':[{'old':pin(old_fixture),'new':fixture_pin,'replacements':fixture_changes,'entire_old_RAW_reconstructed':True},
        {'old':pin(old_external),'new':external_pin,'replacements':external_changes,'entire_old_RAW_reconstructed':True}],
    'all_candidate_sources_AST_parsed_without_import_execution':True,
    'fixture_shared_pin_read_canonical_digest_identity_write_functions_exact_AST':True,
    'external_other_functions_and_resource_ownership_failure_behavior_unchanged_after_declared_literal_replacements':True}
ledger_bytes=encoded(ledger)
handoff_path=F/'recorded186564_ordered14_current_decoder_inactive_handoff_v1.json'
handoff={'kind':'ASR18_INACTIVE_current186564_overlay9_ordered14_decoder_observation_handoff_v1','execution_admitted':False,
    'preparation_ROOT_scope':scope_pin,'required_execution_ROOT_scope':SCOPE,'fixture_source':fixture_pin,
    'launcher_source':external_pin,'frozen_inputs':future_pin(INPUTS,input_bytes),'full_RAW_inverse_ledger':future_pin(ledger_path,ledger_bytes),
    'input_pins':inputs,'recorded_plan':plan_pin,'recorded_plan_canonical_sha256':candidate['canonical_plan_sha256'],
    'global_update':186564,'local_update_index':3,'original_phase':update['phase'],
    'ordered14_rows':[{'row_1based':j+1,'ordinal':v['ordinal'],'source_id':v['source_id'],'dataset':v['dataset'],
        'full_U':len(v['tr_metadata_labels']),'T':v['encoder_frames']}for j,v in enumerate(rows)],
    'current_runtime_path':'actual CPU13 resource validator + original NumPy-only import + actual overlay9 OriginalRows constructor + original real resolve/3lockedDB/fulltargets/currentrow call + original reference_entry + CPU13 raw retirement',
    'no_CPU_startup_frontend_or_transport_constructor':True,'all8_overlay_policies_constructor_checked_and_all14_bothkey_nonmatches':True,
    'original_metadata_context_reconstruction':frozen['validated_context_basis'],
    'exception_observation':'FIRST propagated runtime-call exception copied immediately as type/message/repr/traceback TEXT + pre-fsynced fullrow/annotation/targets/context; no tracing/local/frame/traceback-object retention; no later row',
    'no_direct_unwrapped_baseline_or_second_decoder':True,'all14_success':'Explicit INCONCLUSIVE; no physical cause/remedy success, no automatic new observation',
    'resource_or_source_error':'Failed protocol/guard evidence, not successful row localization or audio correctness',
    'limits':{'fixture_wall_seconds':180,'external_wall_seconds':240,'sampled_whole_family_private_bytes':8000000000,
        'CPU_original_private_bytes':8000000000,'CPU_prospective_commit_reserve_bytes':2500000000,
        'CPU_actual_commit_headroom_bytes':1500000000,'original_allocation_wait_seconds':300,
        'FULL_retained_prefetch_bytes':268435456,'raw_capacity_bytes':134217728,'raw_maximum_jobs':2,
        'max_original_raw_media_fullworkspace_numerical_prospective_bytes':maximum,
        'external_admission_commit_bytes':2500000000+268435456+maximum,
        'external_admission_physical_bytes':1500000000+268435456+maximum,
        'runtime_commit_floor_plus_FULLprefetch_bytes':2268435456,'runtime_physical_floor_plus_FULLprefetch_bytes':1768435456,
        'E_free_bytes':100000000000,'CUDA_VISIBLE_DEVICES':'-1','all4threadcaps':1,'priority':'BelowNormal',
        'family_limit_is_sampled_not_hard_allocation':True,'no_freed_reservation_credit':True,
        'maximum_rows':14,'compact_single_file_pin_bytes':8388608,
        'planned_output_inventory_max_files':32,'metadata_report_bound_8MiB_at_external_pin_not_hard_allocation':True},
    'output_inventory':'producer_identity + at most14 before_row + at most14 success_row + at most1 first_actual_exception + report + CPU_completion; external result/stdout/stderr outside owned output',
    'original_frontend_transform_numerical_reservations_retained_without_executing_those_stages':True,
    'output_path':str((F/'recorded186564_ordered14_current_decoder_actual_01').resolve()),
    'future_exact_external_arguments':['E:/gpu_training_scratchpad/scratch/BPE_broad_CPU_preflight_env_v2/Scripts/python.exe','-Xutf8','-B',str(EXTERNAL),'--ROOT-review','<FUTURE_EXACT_ROOT_EXECUTION_SCOPE_PATH>'],
    'no_new_audio_execution_source_import_GC_state_draw_frontend_model_GPU_or_production_action_by_preparation':True,
    'all_closed_failure_checkpoint_tail_and_prior_audio_proofs_reused_without_raw_reread_or_rerun':True}
prepared={FIXTURE:fixture_bytes,EXTERNAL:external_bytes,INPUTS:input_bytes,ledger_path:ledger_bytes,handoff_path:encoded(handoff)}
require(all(not p.exists()for p in prepared),'All inactive immutable destinations fresh BEFORE writes')
preparation_final=preparation_guard()
for path,data in prepared.items():path.write_bytes(data)
result_path=F/'recorded186564_ordered14_current_decoder_source_preparation_result_v1.json'
require(not result_path.exists(),'Fresh metadata preparation result')
result={'kind':'ASR18_INACTIVE_current186564_ordered14_preparation_result_v1','prepared':[pin(path)for path in prepared],
    'input_count':len(inputs),'max_original_prospective_bytes':maximum,'no_candidate_import_or_execution':True,
    'preparation_identity':{'pid':os.getpid(),'create_time':psutil.Process().create_time(),'ppid':psutil.Process().ppid()},
    'preparation_launcher_identity':{'pid':psutil.Process().parent().pid,'create_time':psutil.Process().parent().create_time(),'ppid':psutil.Process().parent().ppid()},
    'preparation_initial_resource_sample':preparation_initial,'preparation_final_resource_sample':preparation_final}
result_path.write_bytes(encoded(result));print(json.dumps({'result':pin(result_path),**result},ensure_ascii=False))
