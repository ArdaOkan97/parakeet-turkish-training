"""ONE inactive ordered14 decoder-only reproduction; no production admission.

The already-certified overlay7 chain delegates these exact nonmatching rows to
decoder2. Each row is decoded ONCE in the original order, stopping at the first
actual exception. No baseline/full-member/frontend/transform operation is used.
"""
import argparse
from collections import deque
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import types

import psutil

P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20')
S=P/'outputs/parakeet_turkish';sys.path[:0]=[str(S),str(P/'work/research')]
F=Path(__file__).parent
OUTPUT=F/'ordered14_decoder_actual_01'
INPUTS=F/'ordered14_decoder_frozen_inputs_v1.json'
SCOPE='ASR18_source31_saved170400_ordered14_original_decoder_reproduction_CPU_v1'
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


def observe_original_call(reader,raw,ledger,key,declaration,current):
    """Trace ONLY scalar decoder locals; never keep frames or ndarray objects."""
    method=reader.kiraat if raw['dataset']=='kiraat'else reader.yodas
    code=method.__func__.__code__;frames=deque(maxlen=8);first=[];state={}
    lines,start=inspect.getsourcelines(method.__func__)
    marker=next((start+i for i,line in enumerate(lines)if 'if frame_end<=begin:continue'in line),None)
    def snapshot(frame):
        local=frame.f_locals
        for name in ('written','count','decoded','begin','finish','sr','channels','sample_cursor','name','offset','size'):
            value=local.get(name)
            if type(value)in(int,str,bool)or value is None:state[name]=value
        for name in ('origin','start','end','first_selected_pts'):
            if name in local:state[name]=str(local[name])
    def trace(frame,event,arg):
        if frame.f_code is not code:return None
        if event=='line'and frame.f_lineno==marker:
            local=frame.f_locals;frame_audio=local['frame']
            item={'decoded':local['decoded'],'PTS_ticks':frame_audio.pts,
                'time_base':str(frame_audio.time_base),'samples':frame_audio.samples,
                'frame_begin':local['frame_begin'],'frame_end':local['frame_end'],
                'written_before':local['written']}
            if len(first)<5:first.append(item)
            frames.append(item)
        if event in('exception','return'):snapshot(frame)
        if event=='exception':
            first_error=OUTPUT/'first_decoder_exception_immediate.json'
            if not first_error.exists():
                write_new(first_error,{'identity':current,'exception_type':arg[0].__name__,
                    'exception_module':arg[0].__module__,'exception_message':str(arg[1]),
                    'exception_repr':repr(arg[1]),'traceback':''.join(traceback.format_exception(*arg)),
                    'scalar_coverage':dict(state),'first_frames':list(first),'last_frames':list(frames),
                    'only_explicit_scalar_locals_copied_no_frame_or_traceback_retained':True})
        return trace
    require(sys.gettrace()is None,'No preexisting trace observer to replace')
    sys.settrace(trace)
    try:
        wave=reader.read_original(raw,job_key=key,declaration=declaration)
        actual=dict(reader.last)
        del wave
        return {'passed':True,'actual_decoder_metadata':actual,'scalar_coverage':state,
            'first_frames':first,'last_frames':list(frames),'decoder_method_source':pin(inspect.getfile(method.__func__))}
    finally:sys.settrace(None)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--ROOT-review',required=True);args=parser.parse_args()
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    review=read(args.ROOT_review);fixed=read(INPUTS)
    require(review['accepted']is True and review['reviewer']=='/root'and review['scope']==SCOPE
        and review['fixture_source']==pin(__file__)and review['frozen_inputs']==pin(INPUTS)
        and Path(review['output_path']).resolve()==OUTPUT.resolve(),'Exact ROOT fixture/input/output scope')
    require(review['input_pins']==fixed['input_pins'],'Exact canonical frozen input set')
    for value in fixed['input_pins']:require(pin(value['path'])==value,'All fixed compact/source inputs before decode')
    require(not OUTPUT.exists(),'ONE fresh decoder observation output')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'and sys.flags.utf8_mode==1
        and all(os.environ.get(k)=='1'for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'))
        and all(Path(os.environ[k]).resolve().is_relative_to(Path('E:/gpu_training_scratchpad').resolve())
            for k in ('TEMP','TMP','HF_HOME','TORCH_EXTENSIONS_DIR')),'Original E CPU-only environment and four thread caps')
    candidate=read(fixed['candidate']['path']);config=read(fixed['configuration']['path'])
    plan=read(candidate['recorded_target_plan']['path']);update=plan['updates'][9]
    require(pin(candidate['recorded_target_plan']['path'])==candidate['recorded_target_plan']
        and digest(plan)==candidate['recorded_target_plan_canonical_sha256']
        and update['local_update_index']==9 and update['global_update']==170470
        and digest(update)==candidate['exact_target_entry_canonical_sha256']
        and candidate['completed_sent_packets']==2469 and candidate['origin_initial_updates']==168000
        and candidate['next_unsent_CPU_update_index']==9,'Exact recorded CPU2469 plan154/index9, not inferred from native frontier')
    rows=candidate['ordered14_original_TR_candidates']
    require(len(rows)==14 and [v['original_update_row_1based']for v in rows]==list(range(1,15))
        and [v['full_original_plan_row']for v in rows]==update['rows'][:14]
        and all(v['raw_original_annotation']==v['full_original_plan_row']['broad_original']
            and digest(v['full_original_plan_row'])==v['original_row_canonical_sha256']
            and digest(v['raw_original_annotation'])==v['raw_annotation_canonical_sha256']for v in rows),
        'Exact all14 original TR rows, annotations, labels and original row order')
    policies=[];policy=config['original_clip_geometry_overlay']
    while policy is not None:
        certificate=read(policy['certificate']['path']);raw=certificate['original_row']
        require(all(v['ordinal']!=raw['ordinal']and v['source_id']!=raw['unique_id']for v in rows),
            'Every candidate is an ordinary nonmatch of each certified overlay')
        policies.append({'source':policy['source'],'certificate':policy['certificate'],
            'ordinal':raw['ordinal'],'unique_id':raw['unique_id']});policy=policy.get('prior_overlay')
    require(len(policies)==7 and policies==fixed['seven_overlay_nonmatches'],
        'Overlay7 to core ordinary path delegates once to unchanged decoder2')
    from fastmss_resident_feature_stream_v11 import AvailableMemoryPolicy,_raw_finish_without_full_heap_collection,checked_resource_profile
    import fastmss_BPE9223_cpu_common_v12 as core
    from fastmss_bounded_original_audio_v2 import BoundedOriginalReader
    from fastmss_raw_lifetime_v1 import RawLifetimeLedger,raw_plan
    require(checked_resource_profile(config)==fixed['original_resource_policy'],'Exact original CPU11 resource policy')
    raw_contract=core.verify_raw_review(config['raw_lifetime_review'])
    heldout=read(fixed['heldout_PCM']['path'])
    require(heldout['complete']is True and heldout['reference_clips']==12273 and len(heldout['PCM_i16_sha256'])==12273,
        'Original full heldout PCM metadata set, no new heldout decode')
    heldout_hashes=set(heldout['PCM_i16_sha256'])
    memory=AvailableMemoryPolicy(private_limit_bytes=PRIVATE)
    require(memory.prospective_reserve_bytes==PROSPECTIVE and memory.allocation_wait_seconds==300,
        'Unchanged original 2.5GB prospective reserve/private8GB/allocation300s')
    began=time.monotonic();guard_error=None
    def guard():
        nonlocal guard_error
        try:
            require(time.monotonic()-began<180,'Finite decoder observation wall180s')
            value=memory();system=value['system_memory']
            require(system['commit_available_bytes']>=2000000000+PREFETCH
                and system['physical_available_bytes']>=1500000000+PREFETCH
                and shutil.disk_usage('E:/').free>=100000000000,'Original floors PLUS FULL retained256MiB prefetch/E100GB')
            return value
        except BaseException as error:guard_error=repr(error);raise
    def admit(name,budget):return memory.admit(budget,guard)
    guard();core.numpy_runtime(frontend=False)
    OUTPUT.mkdir()
    producer=identity();results=[];current=None;reader=None;ledger=None;failure=None
    report={'kind':SCOPE,'passed':False,'producer':producer,'ROOT_review':pin(args.ROOT_review),
        'input_pins_before':fixed['input_pins'],'seven_overlay_dispatch_nonmatches':policies,
        'execution_path':'Single unchanged decoder2 read_original per ordinary overlay7-delegated row; no overlay duplicate/baseline rerun',
        'results':results,'first_actual_exception':None,'no_sampler_RNG_journal_frontend_transform_model_GPU':True,
        'no_fullmember_scan_or_source_geometry_tolerance_change':True,'production_admitted':False}
    write_new(OUTPUT/'producer_identity.json',producer)
    try:
        for item in rows:
            guard();row=item['full_original_plan_row'];raw=item['raw_original_annotation'];d=row['declared_geometry']
            key=str(item['original_update_row_1based'])
            current={'global_update':170470,'plan':candidate['recorded_target_plan'],'plan_canonical_sha256':digest(plan),
                'local_update_index':9,'original_update_row_1based':item['original_update_row_1based'],
                'candidate':item,'full_target_ids':row['tr_metadata_labels'],'declared_geometry':d,
                'row_text_unchanged':True,'decoder_call_started':False}
            write_new(OUTPUT/('before_row_'+key.zfill(2)+'.json'),current)
            p=raw_plan(native_samples=d['native_samples'],native_rate=d['native_sr'],clean_samples=d['clean16k_samples'],
                phase=update['phase'],role=row['augmentation_role'],
                channels_bound=1 if raw['dataset']=='kiraat'else raw_contract['maximum_runtime_channels'],
                frame_samples_bound=raw_contract['maximum_decoder_frame_samples'])
            require(p['workspace_incremental_envelopes']['decoder']<=raw_contract['decoder_workspace_allowance_bytes'],
                'Original decoder workspace envelope')
            work=p['workspace_incremental_envelopes']
            require(work['weak_transform_FFT']<=raw_contract['transform_workspace_allowance_bytes']
                and work['original_frontend_spectral']<=raw_contract['frontend_workspace_allowance_bytes'],
                'Retain full original transform/frontend prospective envelopes even though not executed')
            budget=p['raw_peak_bytes']+p['compressed_media_cap_bytes']+max(work.values())+raw_contract['numerical_runtime_reserve_bytes']
            admission=memory.admit(budget,guard)
            ledger=RawLifetimeLedger(guard=guard)
            ledger.finish=types.MethodType(_raw_finish_without_full_heap_collection,ledger)
            ledger.begin(key,p)
            reader=BoundedOriginalReader(ledger=ledger,source_inventory_binding=fixed['source_identity_inventory'],
                guard=guard,admit=admit,heldout_PCM_hashes=heldout_hashes)
            current['decoder_call_started']=True
            observation=observe_original_call(reader,raw,ledger,key,d,current)
            ledger.finish(key);receipt=ledger.receipt()
            record={'identity':current,'original_raw_plan':p,'original_prospective_admission':admission,
                'observation':observation,'raw_ledger':receipt}
            write_new(OUTPUT/('success_row_'+key.zfill(2)+'.json'),record)
            results.append(record);reader=None;ledger=None;current=None
        report['all14_decodes_passed']=True
    except BaseException as error:
        # Persist before finally/drop/cleanup can replace any original failure.
        failure={'identity':current,'exception_type':type(error).__name__,'exception_module':type(error).__module__,
            'exception_message':str(error),'exception_repr':repr(error),'traceback':traceback.format_exc(),
            'guard_error':guard_error,'raw_ledger_failed':ledger.failed if ledger is not None else None,
            'raw_ledger_events':list(ledger.events)if ledger is not None else None}
        # Extract only scalar locals from the existing failing decoder traceback.
        scalar={};tb=error.__traceback__
        while tb is not None:
            if tb.tb_frame.f_code.co_filename==str(S/'fastmss_bounded_original_audio_v2.py'):
                for name in ('written','count','decoded','begin','finish','sr','channels','sample_cursor','name','offset','size'):
                    value=tb.tb_frame.f_locals.get(name)
                    if type(value)in(int,str,bool)or value is None:scalar[name]=value
                for name in ('origin','start','end','first_selected_pts'):
                    if name in tb.tb_frame.f_locals:scalar[name]=str(tb.tb_frame.f_locals[name])
            tb=tb.tb_next
        failure['decoder_scalar_locals']=scalar
        write_new(OUTPUT/'first_actual_exception.json',failure)
        report['first_actual_exception']=failure;report['all14_decodes_passed']=False
        if guard_error is not None:raise
    finally:
        sys.settrace(None);reader=None;ledger=None;current=None
        report['CPU_elapsed_seconds']=time.monotonic()-began
        report['CPU_peak_private_bytes']=memory.peak_private
        report['allocation_wait_count']=memory.allocation_wait_count
        report['allocation_wait_seconds_total']=memory.allocation_wait_seconds_total
        report['last_resource_sample']=memory.last_observed
        report['guard_error']=guard_error
        report['input_pins_after']=[pin(value['path'])for value in fixed['input_pins']]
        report['input_pins_before_after_exact']=report['input_pins_after']==fixed['input_pins']
        report['passed']=guard_error is None and report['input_pins_before_after_exact']
        report['passed_means_observation_completed_not_original_audio_success']=True
        report['failure_localization_not_physical_cause_or_remedy_proof']=True
        require('torch'not in sys.modules,'No Torch/model/frontend/GPU import')
        write_new(OUTPUT/'report.json',report)
        write_new(OUTPUT/'CPU_completion.json',{'producer':producer,'report':pin(OUTPUT/'report.json'),
            'actual_process_return_still_unobserved_inside_child':True,'elapsed_seconds':time.monotonic()-began})
    print(json.dumps({'report':pin(OUTPUT/'report.json'),'diagnostic_completed':report['passed'],
        'preceding_successes':len(results),'first_failed_row_1based':failure['identity']['original_update_row_1based']if failure and failure['identity']else None}),flush=True)


if __name__=='__main__':main()
