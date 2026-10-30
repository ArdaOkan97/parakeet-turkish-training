"""ONE inactive genuine row6 original ingress/frontend proof plus labelled controls.

No resident packet transport constructor, model, sampler draw, GPU or production
operation. Prior seven/ordinary delegation and one adversarial metadata-loader
geometry rejection are explicitly METHOD_STUB controls; no old audio is replayed.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import psutil

P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20')
S=P/'outputs/parakeet_turkish';sys.path[:0]=[str(S),str(P/'work/research')]
F=Path(__file__).parent;OUTPUT=F/'eighth_ingress_frontend_actual_02'
INPUTS=F/'eighth_ingress_frontend_frozen_inputs_v2.json'
SCOPE='ASR18_eighth_exact_original_head_member_CPU_ingress_frontend_v1'
def require(value,message):
    if not value:raise ValueError(message)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def pin(path):
    p=Path(path).resolve();before=p.stat();require(before.st_size<=8*1024**2,'Compact/source pin only')
    data=p.read_bytes();after=p.stat()
    require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Pinned input changed')
    return {'path':str(p),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
def canonical(value):return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf-8')
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def write_new(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n')as file:
        json.dump(value,file,indent=2,ensure_ascii=False,allow_nan=False);file.write('\n');file.flush();os.fsync(file.fileno())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--ROOT-review',required=True);args=parser.parse_args()
    process=psutil.Process();process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    who={'pid':process.pid,'create_time':process.create_time(),'ppid':process.ppid()}
    review=read(args.ROOT_review);fixed=read(INPUTS)
    require(review['accepted']is True and review['reviewer']=='/root'and review['scope']==SCOPE
        and review['fixture_source']==pin(__file__)and review['frozen_inputs']==pin(INPUTS)
        and review['input_pins']==fixed['input_pins']and review['overlay_source_review']==fixed['overlay_source_review']
        and Path(review['output_path']).resolve()==OUTPUT.resolve(),'Exact ROOT CPU scope/source/input/output')
    for value in fixed['input_pins']:require(pin(value['path'])==value,'All frozen inputs exact before one ingress')
    source_review=read(fixed['overlay_source_review']['path'])
    require(source_review['accepted']is True and source_review['reviewer']=='/root'
        and source_review['scope']=='ASR18_source_bound_eighth_original_full_clip_geometry_overlay_v1'
        and source_review['overlay_source']==fixed['overlay_source']and source_review['certificate']==fixed['certificate'],
        'Genuine source-only ROOT gate, not a fixture-generated fake completed review')
    require(not OUTPUT.exists(),'ONE fresh real ingress/frontend proof output')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'and sys.flags.utf8_mode==1
        and all(os.environ.get(k)=='1'for k in('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'))
        and all(Path(os.environ[k]).resolve().is_relative_to(Path('E:/gpu_training_scratchpad').resolve())
            for k in('TEMP','TMP','HF_HOME','TORCH_EXTENSIONS_DIR')),'E CPU-only masks and all four environment thread caps')
    from fastmss_resident_feature_stream_v13 import _startup,AvailableMemoryPolicy,checked_resource_profile,checked_packet_credit_policy
    import fastmss_original_clip_geometry_overlay_v9 as eighth
    from fastmss_original_clip_geometry_overlay_v9 import match_original,decode_full_member
    began=time.monotonic();guard_error=None
    class BoundedPolicy(AvailableMemoryPolicy):
        def __call__(self):
            nonlocal guard_error
            try:
                require(time.monotonic()-began<180,'Finite180s ingress/frontend proof')
                value=super().__call__();system=value['system_memory']
                require(system['commit_available_bytes']>=2000000000+268435456
                    and system['physical_available_bytes']>=1500000000+268435456
                    and shutil.disk_usage('E:/').free>=100000000000,'Original floors PLUS FULL256MiB retained prefetch/E100GB')
                return value
            except BaseException as error:guard_error=repr(error);raise
    policy=BoundedPolicy(8000000000);guard=policy
    require(policy.prospective_reserve_bytes==2500000000 and policy.allocation_wait_seconds==300,
        'Original CPU8GB/2.5GB prospective/300s allocation wait unchanged')
    initial=policy.admit(fixed['maximum_original_prospective_bytes'],guard)
    base=read(fixed['proper_source_base']['path']);candidate=read(fixed['candidate_configuration']['path'])
    require(candidate['original_clip_geometry_overlay']['ROOT_review']==fixed['overlay_source_review'],
        'Genuine completed source-only overlay gate, not completed real-ingress admission')
    expected=copy.deepcopy(base)
    expected['original_clip_geometry_overlay']={'prior_overlay':copy.deepcopy(base['original_clip_geometry_overlay']),
        'source':fixed['overlay_source'],'certificate':fixed['certificate'],'ROOT_review':fixed['overlay_source_review']}
    expected['prior_resident_RAM_original_interval_failure']=fixed['actual_failure_closure']
    expected['resident_RAM_resource_scheduling_policy']['CPU_protocol_source']=fixed['CPU13_source']
    expected['resident_RAM_packet_credit_policy']['actual_CPU_control_review']=fixed['CPU13_transport_compatibility_review']
    require(candidate==expected,'Exactly4 source operational fields including genuine derived transport provenance; all other values exact')
    require(checked_resource_profile(candidate)==candidate['resident_RAM_resource_scheduling_policy'],
        'Actual CPU13 resource policy binding, all original values')
    require(checked_packet_credit_policy(candidate)==candidate['resident_RAM_packet_credit_policy'],
        'Actual unchanged metadata-only credit-policy validator accepts genuine derived CPU13 compatibility; NO transport rerun')
    runtime=copy.deepcopy(candidate)
    OUTPUT.mkdir();configuration=OUTPUT/'configuration.json';write_new(configuration,runtime)
    reader=None;negative=[];report={'kind':SCOPE,'passed':False,'producer':who,'ROOT_review':pin(args.ROOT_review),
        'ROOT_source_review':fixed['overlay_source_review'],'configuration':pin(configuration),
        'overlay_source':fixed['overlay_source'],'CPU_stream_source':fixed['CPU13_source'],'certificate':fixed['certificate'],
        'input_pins_before':fixed['input_pins'],'initial_full_original_prospective_admission':initial,
        'negative_controls':negative,'actual_native_updates':0,'draws_consumed':0,'no_GPU_or_model':True,
        'no_source_index_plan_journal_or_authoritative_state_mutation':True,
        'no_old_audio_or_transport_rerun':True,
        'CPU13_transport_compatibility_is_genuine_derived_metadata_gate_no_new_transport_test':fixed['CPU13_transport_compatibility_review']}
    write_new(OUTPUT/'producer_identity.json',who)
    try:
        core,single,source,validated,reader,augmenter,clean_reader,frontend=_startup(pin(configuration),policy)
        plan=read(fixed['recorded_plan']['path'])
        require(digest(plan)==fixed['recorded_plan_canonical_sha256'],'Actual fixed plan154 canonical identity')
        normalized=core.bridge.normalize_updates(plan,population=validated['retained_rows'],guard=guard)
        update=normalized[9];expected=update['rows'][5]
        row=copy.deepcopy(update['coordinator_update']['rows'][5]);context=update['coordinator_update']['contexts'][5]
        row['audio_draw_context']=copy.deepcopy(context);row['augmentation_role']=expected['role']
        require(row['broad_original']['ordinal']==332560
            and row['broad_original']['unique_id']=='alania:aedf14e56cf50416170171503e0dee9a_00000'
            and row['tr_metadata_labels']==reader.eighth_certificate['full_target_ids']
            and len(expected['full_target_ids'])==99,'Exact real normalized ingress of recorded plan154/index9/row6/full99')
        write_new(OUTPUT/'before_real_ingress.json',{'recorded_plan':fixed['recorded_plan'],
            'plan_canonical_sha256':digest(plan),'local_update_index':9,'original_row_1based':6,'global_update':170470,
            'row':row,'expected':expected,'original_phase':update['phase']})
        wave,reference=clean_reader(row,expected,update['phase']);arrays=augmenter.frontend(frontend,wave)
        import numpy as np
        geometry=core.check_frontend(arrays,len(wave),np)
        require(geometry==dict(clean16k_samples=273624,valid_mel_frames=1710,physical_feature_frames=1711,encoder_frames=214)
            and reference['full_target_ids']==reader.eighth_certificate['full_target_ids']
            and len(reference['full_target_ids'])==99
            and reference['expected_audio']['waveform_sha256']==reader.eighth_certificate['clean_f32_sha256'],
            'Genuine full273624 clean PCM/all99/original1711features1710validmask/T214')
        require(arrays['input_features'].dtype==np.float32 and arrays['input_features'].shape==(1,1711,128)
            and np.isfinite(arrays['input_features']).all()and arrays['attention_mask'].dtype==np.bool_
            and int(arrays['attention_mask'].sum())==1710,'Original actual frontend finite normalization and complete mask')
        array_reports={key:{'shape':list(value.shape),'dtype':str(value.dtype),
            'array_sha256':hashlib.sha256(memoryview(value).cast('B')).hexdigest()}for key,value in arrays.items()}
        geometry_proof=reference['before_draw_geometry_proof']
        require(geometry_proof['historical_declaration']['native_samples']==820896
            and geometry_proof['historical_declaration']['clean16k_samples']==273632
            and geometry_proof['actual_geometry']['native_samples']==820872
            and geometry_proof['actual_geometry']['clean16k_samples']==273624
            and geometry_proof['historical_mel_frames']==geometry_proof['actual_physical_mel_frames']==1710,
            'Historical larger capacities retained, full actual physical samples once; no pad/crop/freed credit')
        actual_reference=copy.deepcopy(reference);del arrays,wave
        clean_reader.finish_row();require(not reader.ledger.jobs and reader.active_job is None,'Actual raw buffers released')
        successful_raw=reader.ledger.receipt()
        certificate=reader.eighth_certificate;raw=copy.deepcopy(certificate['original_row']);decl=copy.deepcopy(certificate['original_declaration'])
        def rejects(name,call,*,message=None,method_stub=False):
            try:call()
            except ValueError as error:
                require(message is None or str(error)==message,'Strict negative failed at unexpected validation path: '+name)
                negative.append({'name':name,'rejected':True,'error':repr(error),'METHOD_STUB_metadata_loader':method_stub})
            else:raise AssertionError('Required genuine strict rejection: '+name)
        changed=copy.deepcopy(raw);changed['end']=17.101
        rejects('changed_original_annotation',lambda:match_original(certificate,changed,decl,expected))
        changed_decl=copy.deepcopy(decl);changed_decl['native_samples']-=1
        rejects('changed_historical_declaration',lambda:match_original(certificate,raw,changed_decl,expected))
        changed_expected=copy.deepcopy(expected);changed_expected['full_target_ids'][0]=(changed_expected['full_target_ids'][0]+1)%8192
        rejects('changed_full_target',lambda:match_original(certificate,raw,decl,changed_expected))
        changed_row=copy.deepcopy(raw);changed_row['ordinal']+=1
        rejects('changed_original_row',lambda:match_original(certificate,changed_row,decl,expected))
        # Both following negatives fail BEFORE any PCM allocation/decode. The
        # second intentionally hashes only this same compressed member extent.
        wrong_source=copy.deepcopy(certificate);wrong_source['original_TAR']['recorded_sha256']='0'*64
        policy.admit(fixed['maximum_original_prospective_bytes'],guard)
        rejects('changed_original_source_SHA_before_decode',lambda:decode_full_member(reader.broad,raw,wrong_source,
            guard=guard,channels_bound=8,frame_samples_bound=16384),message='Original source identity/signature')
        wrong_member=copy.deepcopy(certificate);wrong_member['member_sha256']='0'*64
        policy.admit(fixed['maximum_original_prospective_bytes'],guard)
        rejects('changed_original_member_SHA_before_decode',lambda:decode_full_member(reader.broad,raw,wrong_member,
            guard=guard,channels_bound=8,frame_samples_bound=16384),message='Exact original compressed bytes')
        # METHOD_STUB adversarial immutable-metadata loader: original source and
        # genuine ROOT pin remain exact; no production-usable fake review/file.
        original_read_bound=core.read_bound;bad_geometry=copy.deepcopy(certificate)
        bad_geometry['actual_geometry']['native_samples']+=1
        def adversarial_read(bound,check):
            if bound==fixed['certificate']:return copy.deepcopy(bad_geometry)
            return original_read_bound(bound,check)
        try:
            core.read_bound=adversarial_read
            rejects('changed_certified_physical_geometry_METHOD_STUB_loader',
                lambda:eighth.checked_certificate(runtime['original_clip_geometry_overlay'],guard),
                message='Exact measured EOF, unchanged T/U',method_stub=True)
        finally:core.read_bound=original_read_bound
        # Only eighth-layer delegation is invoked; prior actual row audio is NOT
        # regenerated. The parent method is restored before closure/reporting.
        original_call=eighth.prior.OriginalRows.__call__;delegated=[];sentinel=object();prior_rows=[]
        previous=runtime['original_clip_geometry_overlay']['prior_overlay']
        while previous is not None:
            prior_certificate=core.read_bound(previous['certificate'],guard)
            prior_rows.append({'ordinal':prior_certificate['original_row']['ordinal'],
                'source_id':prior_certificate['original_row']['unique_id']});previous=previous.get('prior_overlay')
        require(len(prior_rows)==7,'All seven original policy dispatch bindings')
        def delegation_stub(self,r,e,p):delegated.append((r,e,p));return sentinel
        try:
            eighth.prior.OriginalRows.__call__=delegation_stub
            for prior_row in prior_rows:
                prior_expected={}
                require(reader(prior_row,prior_expected,update['phase'])is sentinel
                    and delegated[-1][0]is prior_row and delegated[-1][1]is prior_expected,
                    'Exact unchanged prior7 arguments delegated; METHOD_STUB route only')
            ordinary={'ordinal':-1,'source_id':'metadata_only_unmatched_route'};ordinary_expected={}
            require(reader(ordinary,ordinary_expected,update['phase'])is sentinel
                and delegated[-1][0]is ordinary and delegated[-1][1]is ordinary_expected,'Ordinary exact delegation METHOD_STUB')
        finally:eighth.prior.OriginalRows.__call__=original_call
        require(runtime['original_clip_geometry_overlay']['prior_overlay']==base['original_clip_geometry_overlay'],
            'Entire unchanged original seven-certificate policy tree')
        bad_prior=copy.deepcopy(base['original_clip_geometry_overlay']['certificate']);bad_prior['sha256']='0'*64
        rejects('changed_prior7_certificate_pin',lambda:core.read_bound(bad_prior,guard))
        require(len(negative)==8 and core.read_bound is original_read_bound
            and eighth.prior.OriginalRows.__call__ is original_call,'All eight strict negatives and every stub restored')
        previous_policy={'passed':True,'METHOD_STUB_call_route_control':True,'no_old_audio_replayed':True,
            'all_seven_prior_row_routes':prior_rows,'ordinary_unmatched_route':True,'parent_method_restored':True,
            'prior7_policy_exact':True}
        reader.close();require(reader.connections==[]and reader.locks is None,'Original scalar DB connections/locks actually closed')
        reader=None
        import torch
        require(torch.__version__=='2.7.1+cpu'and torch.version.cuda is None and not torch.cuda.is_initialized(),
            'Original CPU-only frontend Torch, no model or CUDA initialized')
        report.update(passed=True,actual_geometry=geometry,arrays=array_reports,reference=actual_reference,
            previous_policy_delegation=previous_policy,raw_buffers_released_and_metadata_handles_closed=True,
            successful_raw_ledger=successful_raw,all_stubs_restored=True,
            actual_one_full_source_ingress_and_original_frontend_only=True,
            original_historical_reservations_retained=True,actual_native_updates=0,draws_consumed=0)
    except BaseException as error:
        report['actual_exception']={'type':type(error).__name__,'repr':repr(error),'traceback':traceback.format_exc()}
        write_new(OUTPUT/'actual_exception_immediate.json',report['actual_exception']);raise
    finally:
        if reader is not None:reader.close()
        report.update(guard_error=guard_error,elapsed_seconds=time.monotonic()-began,peak_private_bytes=policy.peak_private,
            last_resource_sample=policy.last_observed,allocation_wait_count=policy.allocation_wait_count,
            input_pins_after=[pin(v['path'])for v in fixed['input_pins']])
        report['input_pins_before_after_exact']=report['input_pins_after']==fixed['input_pins']
        write_new(OUTPUT/'report.json',report)
        write_new(OUTPUT/'CPU_completion.json',{'producer':who,'report':pin(OUTPUT/'report.json'),
            'elapsed_seconds':time.monotonic()-began,'actual_process_return_still_unobserved_inside_child':True})
    print(json.dumps({'report':pin(OUTPUT/'report.json'),'geometry':report['actual_geometry'],
        'negatives':len(negative),'CPU_peak_private_bytes':policy.peak_private}),flush=True)


if __name__=='__main__':main()
