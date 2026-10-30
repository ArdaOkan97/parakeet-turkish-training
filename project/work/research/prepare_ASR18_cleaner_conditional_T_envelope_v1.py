"""Conditional full cleaner resource arithmetic. No media/header/frontend execution."""
import heapq
import os
from pathlib import Path
import sqlite3
import sys

from ASR18_continuous_data_guard_v5 import AREA, DataIOGuard
from cpu_preparation_guard_v2 import now
from finalize_ASR18_declared_training_geometry_v1 import MetadataIO

P = Path(__file__).resolve().parents[2]
S = P / 'outputs/parakeet_turkish'
sys.path.insert(0, str(S))
from atomic_io_retry_v1 import write_json
from fastmss_raw_lifetime_v1 import raw_plan, CAPACITY, MAX_JOBS

FOLDER = AREA / 'cleaner_resource_accounting_v1'


def main():
    guard = DataIOGuard('cleaner_conditional_T_resource_envelope', heavy=False, maximum_bytes=1_000_000_000)
    guard.wait()
    io = MetadataIO(guard)
    output = FOLDER / 'conditional_T_resource_envelope_v1.json'
    assert not output.exists()
    report_path = FOLDER / 'report.json'
    actual_path = FOLDER / 'metadata_owned_tool_actual_CLOSED0_v1.json'
    actual = io.json(actual_path)
    assert actual['kind'] == 'ASR18_source_bound_actual_tool_exit_v1' and actual['owner'] == '/root/cpu_data_preparation'
    assert type(actual['actual_exit_code']) is int and actual['actual_exit_code'] == 0 and actual['actual_tool_chunk_id']
    assert actual['published_marker'] == io.binding(report_path)
    report = io.json(report_path)
    assert report['complete_metadata_scan'] is True and report['rows'] == 352768 and report['full_target_tokens'] == 11972886
    index_pin = report['bindings']['metadata_index']
    assert io.binding(index_pin['path']) == index_pin
    db = sqlite3.connect(Path(index_pin['path']).as_uri()+'?mode=ro&immutable=1',uri=True)
    db.execute('PRAGMA cache_size=-2048')
    metrics = {name: [] for name in ('stored_T','full_BPE_U','conditional_N_upper','raw_peak_bytes','decoder_workspace_bytes','original_frontend_spectral_workspace_bytes','physical_frontend_frames_upper','physical_feature_and_mask_bytes_upper')}
    counts = dict(rows=0,tokens=0,nonpositive_T=0,one_job_over_raw128MiB=0,two_simultaneous_same_worst_over_raw128MiB=0,weak_transform_possible=0)
    for ordinal,group,sid,path,t,u in db.execute('SELECT ordinal,source_group,source_id,audio_filepath,stored_T,new_U FROM rows ORDER BY ordinal'):
        counts['rows']+=1;counts['tokens']+=u
        assert ordinal==counts['rows'] and type(t) is int and t>0 and type(u) is int and u>0
        upper=8*t*160+159
        plan=raw_plan(native_samples=upper,native_rate=16000,clean_samples=upper,
            phase='cleaner_data_fine_tuning',role='student_tr',channels_bound=1,frame_samples_bound=16384)
        assert plan['weak_transform_possible'] is False and plan['prepared_hold_bytes']==0
        counts['one_job_over_raw128MiB']+=int(plan['raw_peak_bytes']>CAPACITY)
        counts['two_simultaneous_same_worst_over_raw128MiB']+=int(2*plan['raw_peak_bytes']>CAPACITY)
        values=dict(stored_T=t,full_BPE_U=u,conditional_N_upper=upper,raw_peak_bytes=plan['raw_peak_bytes'],
            decoder_workspace_bytes=plan['workspace_incremental_envelopes']['decoder'],
            original_frontend_spectral_workspace_bytes=plan['workspace_incremental_envelopes']['original_frontend_spectral'],
            physical_frontend_frames_upper=8*t+1,physical_feature_and_mask_bytes_upper=(8*t+1)*513)
        row=dict(ordinal=ordinal,source_group=group,source_id=sid,audio_filepath=path,declared_encoder_T=t,
            full_BPE_U=u,conditional_N_upper=upper,exact_N=None,header_checked=False,
            raw_plan_at_conditional_upper=plan)
        for key,value in values.items():
            item=(value,-ordinal,row);heap=metrics[key]
            if len(heap)<8:heapq.heappush(heap,item)
            elif item[:2]>heap[0][:2]:heapq.heapreplace(heap,item)
        if ordinal%2048==0:guard.pace(2048*160)
    db.close()
    assert counts['rows']==352768 and counts['tokens']==11972886
    maxima={key:[dict(value=value,**row) for value,_,row in sorted(heap,reverse=True)] for key,heap in metrics.items()}
    cpu4=io.binding(S/'fastmss_row_serial_cpu_v4.py')
    value=dict(kind='ASR18_full_cleaner_conditional_declared_T_resource_envelope_v1',schema_version=1,
        owner='/root/cpu_data_preparation',at_utc=now(),PID=os.getpid(),complete_scalar_population=True,
        rows=352768,full_target_tokens=11972886,population_mean_U=11972886/352768,
        counts=counts,maxima=maxima,raw_capacity_bytes=CAPACITY,raw_maximum_jobs=MAX_JOBS,
        conditional_basis=dict(sample_rate=16000,channels=1,
            N_upper_expression='N <= 8*T*160 + 159',physical_frontend_frames_upper_expression='N//160+1 <= 8*T+1',
            target_frames_basis='Stored declared encoder T; runtime mono16k header N must reproduce T=ceil((N//160)/8) before allocation. This upper bound is not an exact sample count.',
            augmentation_bypassed=True,weak_transform_possible=False,augmentation_transport_or_noise_FFT_buffers=0,
            original_reader_no_trim_resample_or_crop=True),
        exact_N_unknown_rows=report['exact_native_sample_unknown_rows'],exact_canonical_channels_unknown_rows=report['exact_canonical_channel_unknown_rows'],
        actual_header_or_decode_coverage_added=0,
        frozen_CPU4_source_review=dict(source=cpu4,
            actual_header_rate_channels_N_checked_before_allocation=True,
            actual_header_derived_raw_plan_and_private_admission_before_allocation=True,
            header_rechecked_before_full_read=True,
            declared_T_checked_before_full_read=False,declared_T_checked_after_full_read=True,
            reviewed_lines=dict(header=222,actual_N_raw_plan=227,preallocation_admission=235,header_recheck=258,postread_T=263),
            new_successor_required='Before allocation enforce geometry(header.samples).encoder_frames == exact row.encoder_frames; retain actual-header raw/private/workspace admission and post-read checks. Frozen Parent7/CPU4 remain unchanged.',
            new_source_owner='/root/fastmss_integration'),
        resource_domains=dict(raw_float32_lifetime='Frozen raw_plan at conditional N upper; not a whole-process private bound',
            compressed_media='Separate unchanged32MiB/job domain in frozen raw_plan even though cleaner reads original canonical WAV directly',
            spectral_workspace='Frozen frontend spectral expression plus8MiB; no augmentation FFT in cleaner',
            numerical_import_frontend_runtime_and_whole_process_reserve='Separate measured/reviewed admission required; not proved by scalar arithmetic'),
        bindings=dict(full_cleaner_metadata=io.binding(report_path),full_cleaner_metadata_actual_closed=io.binding(actual_path),
            metadata_index=index_pin,frozen_raw_lifetime_source=io.binding(S/'fastmss_raw_lifetime_v1.py'),
            frozen_CPU4=cpu4,producer_source=io.binding(__file__)),
        upfront_header_plan_executed=False,duration_converted_to_exact_samples=False,
        no_row_selection_or_target_truncation=True,guard_limits_changed=False,
        whole_frontend_private_fit_or_production_admission_claimed=False,TRAIN_admitted=False,production_enabled=False)
    io.unchanged();write_json(output,value)
    assert not any(name in sys.modules for name in ('torch','av','soundfile','numpy','scipy','pyarrow','transformers'))
    print(dict(conditional_cleaner_resource_envelope=io.binding(output),rows=352768,
        MAX_T=maxima['stored_T'][0]['value'],MAX_conditional_N=maxima['conditional_N_upper'][0]['value'],
        MAX_raw_peak_bytes=maxima['raw_peak_bytes'][0]['value'],MAX_frontend_workspace_bytes=maxima['original_frontend_spectral_workspace_bytes'][0]['value'],
        actual_owned_exit_still_required=True))


if __name__=='__main__':main()
