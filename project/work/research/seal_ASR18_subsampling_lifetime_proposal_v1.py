"""Seal inactive source/locality/allocation proposal; never instantiate Torch/CNN."""
import ast
from datetime import datetime,timezone
import hashlib
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


def record(path):
    path=Path(path).resolve();assert path.is_relative_to(PROJECT) or path.is_relative_to(ROOT)
    assert path.stat().st_size<16*1024**2,'Metadata/source only'
    raw=path.read_bytes()
    return {'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,ensure_ascii=False)
    return record(path)


def main():
    from resource_watchdog_v2 import sample_memory
    samples=[]
    for index in range(3):
        value=sample_memory(os.getpid(),'source_only_subsampling_lifetime_proposal')
        s=value['system_memory'];w=value['worker_private_memory']
        assert s['available'] and w['available'] and value['paired_sampling_within_one_second']
        assert w['private_bytes']+64_000_000<2_000_000_000
        assert (s['commit_used_bytes']+64_000_000)/s['commit_limit_bytes']<.90
        assert s['commit_limit_bytes']-s['commit_used_bytes']-64_000_000>2_000_000_000
        assert shutil.disk_usage(ROOT).free>100_000_000_000
        samples.append(value);time.sleep(.05)
    import asr18_temporal_tiled_subsampling_v1 as proposal
    module=record(SOURCE/'asr18_temporal_tiled_subsampling_v1.py')
    original=record(proposal.ORIGINAL_SOURCE);assert original['sha256']==proposal.ORIGINAL_SHA256
    compile(ast.parse(Path(module['path']).read_text()),module['path'],'exec')
    base=Path((PROJECT/'work/base_model_path.txt').read_text().strip()).resolve()
    original_config=record(base/'config.json')
    handoff=read(AREA/'full_DATA_two_window_execution_review_handoff_v5.json')
    finite_config=read(handoff['configuration']['path']);calibration=finite_config['forecast_calibration']
    archive=read(calibration['archive_binding']['path']);assert record(calibration['archive_binding']['path'])==calibration['archive_binding']
    rows=[row for name,value in archive['metadata'].items() if name.startswith('update_') and name.endswith('.json')
          for row in value['rows']]
    assert len(rows)==320
    envelope={}
    for row in rows:
        role='student_tr' if row['role']=='student_tr' else 'replay';e=envelope.setdefault(role,{'physical_frames':0,'encoder_frames':0,'samples':0})
        e['physical_frames']=max(e['physical_frames'],row['input_features']['shape'][1])
        e['encoder_frames']=max(e['encoder_frames'],row['encoder_frames']);e['samples']=max(e['samples'],row['samples'])
    cases=[('one_second_physical101',101,100),('mask_multiple8_physical_extra',105,104),
           ('measured20_TR_max',envelope['student_tr']['physical_frames'],envelope['student_tr']['samples']//160),
           ('finite_T1115_upper',8921,8920),('full_declared_hard159s',15925,15924)]
    logical={name:{'plan':proposal.tile_plan(f,v),'BF16':proposal.logical_allocation(f,v,activation_bytes=2),
                   'FP32_conservative':proposal.logical_allocation(f,v,activation_bytes=4)} for name,f,v in cases}
    pins={
        'proposed_module':module,'original_Parakeet_implementation':original,'original_config':original_config,
        'current_precomputed_row_serial_native':record(SOURCE/'asr18_precomputed_native_v2.py'),
        'current_objective':record(SOURCE/'asr18_narrow_objective_v1.py'),
        'current_streaming_weight_hooks':record(SOURCE/'stream_frozen_encoder_matrices_v1.py'),
        'current_per_block_checkpoint_bootstrap':record(SOURCE/'bpe_broad_streamed_bootstrap_v2.py'),
        'current_measured20_summary':calibration['summary_binding'],'archived_actual320_feature_rows':calibration['archive_binding'],
        'current_native_forecast6':record(SOURCE/'asr18_calibrated_geometry_forecast_v6.py'),
        'current_separated_whole_forecast':record(SOURCE/'asr18_separated_worker_forecast_v1.py'),
        'latest_failed_admission04':record(AREA/'full_DATA_two_window_prelaunch_full_forecast_04.json'),
        'sealer':record(__file__)}
    operations=[
        {'index':0,'kind':'Conv2d','channels':[1,256],'kernel':[3,3],'stride':[2,2],'padding':[1,1],'dilation':[1,1],'groups':1,'zero_padding':True,'mask_after':True},
        {'index':1,'kind':'ReLU','inplace':False,'local':True},
        {'index':2,'kind':'Conv2d','channels':[256,256],'kernel':[3,3],'stride':[2,2],'padding':[1,1],'dilation':[1,1],'groups':256,'zero_padding':True,'mask_after':True},
        {'index':3,'kind':'Conv2d','channels':[256,256],'kernel':[1,1],'stride':[1,1],'padding':[0,0],'dilation':[1,1],'groups':1,'mask_after':True},
        {'index':4,'kind':'ReLU','inplace':False,'local':True},
        {'index':5,'kind':'Conv2d','channels':[256,256],'kernel':[3,3],'stride':[2,2],'padding':[1,1],'dilation':[1,1],'groups':256,'zero_padding':True,'mask_after':True},
        {'index':6,'kind':'Conv2d','channels':[256,256],'kernel':[1,1],'stride':[1,1],'padding':[0,0],'dilation':[1,1],'groups':1,'mask_after':True},
        {'index':7,'kind':'ReLU','inplace':False,'local':True},
        {'index':8,'kind':'transpose_then_reshape','from':'[1,256,T,16]','to':'[1,T,4096]','time_local':True},
        {'index':9,'kind':'Linear','in_features':4096,'out_features':1024,'time_local':True,'original_bias':True}]
    result={'kind':'ASR18_inactive_frozen_CNN_temporal_lifetime_source_proposal_v1','owner':'/root/memory_diagnosis',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'production_enabled':False,'execution_admitted':False,
        'source':module,'bindings':pins,'actual_calibration_scalar_geometry':envelope,
        'reason':'Native2 already runs physical1 row and immediate backward/GC. Further microbatch serialization alone changes no lifetime. Target the whole256-channel frozen subsampling intermediates and out-of-place ReLU overlap.',
        'original_operation_sequence':operations,'normalization_in_target_subgraph':None,'stochastic_operations_in_target_subgraph':[],
        'mask_rule':'Exact right-prefix input mask; after every Conv2d original current_lengths=(length+padH+padW-kernel)//stride+1 when stride!=1; same channel mask and inplace multiplication, then same final bias/physical padding',
        'output_rule':'Physical output ceil(physicalF/8); valid encoder length ceil(validmaskF/8). Extra physical right-padding row is retained even when valid count divides8.',
        'locality_derivation':{'temporal_strided_kernels':[3,3,3],'strides':[2,2,2],'effective_strides':[2,4,8],
            'receptive_widths':[3,7,15],'radius':7,'output_index_j_input_interval_inclusive':'[8*j-7,8*j+7]',
            'tile_core_output_frames':32,'crop_start':'max(0,8*core_start-8), a multiple of8',
            'crop_stop_exclusive':'min(physicalF,8*core_stop+8)',
            'stitch':'output[core_start:core_stop] <- original_crop_forward[core_start-crop_start/8:core_stop-crop_start/8]',
            'left_edge':'Clipped to0, original k3/p1 zero padding supplies only the true utterance edge; no invented audio',
            'right_edge':'Clipped to physicalF, original mask and zero padding preserved; no truncation or crop of physical output',
            'internal_edges':'One aligned8-input halo covers radius7; crop-boundary-dependent output cells are discarded, never stitched'},
        'logical_tensor_allocations_only':logical,
        'memory_domains':'Logical CUDA tensor storage union is separate from unbounded backend workspace, CUDA allocator reserve, process private and system commit. No CUDA savings are subtracted from historical4.683GB native peak or5.624813440GB whole forecast.',
        'full_native_resource_forecast_unchanged':{'native_bytes':5_560_813_440,'write_persistence_bytes':64_000_000,
            'whole_resident_bytes':5_624_813_440,'variation_bytes':750_000_000,'replay_extension_bytes':128_000_000,
            'running_high_water_margin_bytes':250_000_000,'CPU_raw_capacity_bytes':134_217_728},
        'proposed_one_bounded_comparison':{
            'status':'PROPOSAL_NOT_LAUNCHABLE; concrete comparison harness/config/source/resource review and fresh admission still required',
            'workload':'One paired original-vs-tiled frozen-CNN forward comparison, each in a separate fresh sole-owned process, no24-block encoder/LoRA/Adam/teacher/training/source-state restore',
            'weights':'Exact existing six subsampling matrix shards and original bias values only, explicit ACCESS_READ and temporary GPU matrix hooks; no complete model or COW checkpoint load',
            'inputs':'Same deterministic full-shape finite synthetic FP32 feature/mask packets in both workers, pinned all-file/full-array SHA. Edge/multiple8 shapes plus measured finiteTR max and full159s declared shape. No new audio decode or sample exclusion.',
            'numeric_gate':'Require whole physical output dtype/shape/mask and raw bytes EXACT initially. Record maxdiff too. A mismatch stops, no tolerance widening or claimed old-trajectory equality.',
            'RNG_gate':'Actual before/after Python/NumPy/Torch/CUDA state unchanged; all CNN parameters/frozen backing unchanged; no source draw/optimizer advance',
            'measurement':'Separate fresh paired system/private and CUDA allocated/reserved highwaters for each process; source-bound original input/output and phase-tagged managed lifetimes; actualwait0/clean guard before accepting either result',
            'conservative_entire_GPU_worker_candidate_bytes':5_624_813_440,
            'candidate_borrows_existing_larger_full_training_ceiling_without_subtracting_tensor_bytes':True,
            'actual_pre_operation_private_upper_bound_bytes':3_000_000_000,
            'existing750MB_variation128MB_replay64MB_write_allowances_retained':True,
            'whole_parent_and_launcher_additional_fresh_accounting_required':True,
            'system_commit_projected_limit_percent':90,'fresh_paired_samples':3,'critical_percent':95,
            'minimum_headroom_bytes':2_000_000_000,'minimum_E_free_bytes':100_000_000_000,'CUDA_allocator_cap_GiB':8,
            'if_cannot_fit':'No launch/unchanged retry; no app/pagefile change or reduced margin'},
        'before_any_native40':[
            'Actual paired numeric/resource comparison0 before any reduction to host forecast is proposed',
            'Source-bound evidence that the removed tensor lifetime affects actual owned private/commit, not only logicalCUDA',
            'Separately reviewed integrated mature207/threeAdam/all4RNG/sampler/fulltarget/audio/CPU6 comparison and checkpoint every-save recovery with conservative new whole forecast',
            'ROOT exact source/config/resource review plus fresh sole lease/global admission; no diagnostic output becomes a production seed'],
        'old_trajectory_byte_equivalence_claimed':False,'new_numeric_or_private_fit_proved':False,
        'additional_production_scaffolding_created':False,'numerical_control_or_model_payload_audio_GPU_job_executed':False,
        'source_only_seal_fresh_samples':samples}
    for pin in pins.values():assert record(pin['path'])==pin
    pin=write_new(AREA/'subsampling_temporal_lifetime_source_proposal_v1.json',result)
    pointer_path=AREA/'current_launch_readiness.json';pointer=read(pointer_path);previous=pointer['current_readiness'];status=read(previous['path'])
    status.update(kind='ASR18_compact_memory_and_production_preparation_readiness_v59',previous_history=previous,
        updated_at_utc=datetime.now(timezone.utc).isoformat(),
        current_stage='GPU_idle_admission04_denied_DEV6_source_accepted_NEW_subsampling_lifetime_proposal_unadmitted',
        inactive_subsampling_lifetime_proposal=pin,actual_GPU_running=False,actual_owned_GPU_PID=None,current_CPU_metadata_child=None,
        production_enabled=False,production_ready=False,
        remaining='No unchanged config5 retry. Source-only frozen-CNN lifetime proposal is ready for ROOT review; no tensor/private savings or new whole native forecast is proved. Exact paired comparison harness/config/resource review and fresh admission required. DEV6 remains inactive source accepted only.')
    readiness=write_new(AREA/'launch_readiness_v59.json',status)
    pointer.update(current_stage=status['current_stage'],updated_at_utc=status['updated_at_utc'],current_readiness=readiness,
        inactive_subsampling_lifetime_proposal=pin,current_CPU_metadata_child=None)
    temporary=pointer_path.with_suffix('.updating.json');temporary.write_text(json.dumps(pointer,indent=2),encoding='utf-8');temporary.replace(pointer_path)
    for name in ('memory_diagnosis_status.json','production_preparation_status_v1.json'):
        (AREA/name).write_text(json.dumps(pointer,indent=2),encoding='utf-8')
    print(json.dumps({'proposal':pin,'source':module,'readiness':readiness,'numerical_job':False,'GPU_used':False}))


if __name__=='__main__':main()
