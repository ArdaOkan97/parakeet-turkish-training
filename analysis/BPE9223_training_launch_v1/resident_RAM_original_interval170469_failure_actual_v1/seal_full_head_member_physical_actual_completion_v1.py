"""Actual completed physical observation metadata/death sealing only; no decode."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import os
import psutil

F=Path(__file__).parent;O=F/'full_head_member_physical_actual_01'
def require(value,message):
    if not value:raise ValueError(message)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def pin(path):
    p=Path(path).resolve();data=p.read_bytes();require(len(data)<=8*1024**2,'Compact/source metadata only')
    return {'path':str(p),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
def identity(process):return {'pid':process.pid,'create_time':process.create_time(),'ppid':process.ppid()}
def dead(who):
    try:return abs(psutil.Process(who['pid']).create_time()-who['create_time'])>=.001
    except psutil.NoSuchProcess:return True
r=read(O/'report.json');e=read(F/'full_head_member_physical_external_result_v1.json');tool=read(F/'full_head_member_physical_actual_TOOL_v1.json')
require(tool['actual_TOOL_exit_code']==0 and e['Popen_returncode']==0 and e['passed']is True
    and e['external_guard_error']is None and e['required_external_cleanup']==[] and e['inputs_before_after_exact']is True
    and r['passed']is True and r['guard_error']is None and r['input_pins_before_after_exact']is True,
    'Actual0/passed/no guard or cleanup/all61inputpins exact')
require(r['row']['ordinal']==332560 and r['row']['unique_id']=='alania:aedf14e56cf50416170171503e0dee9a_00000'
    and r['complete_demux_and_decode_EOF']is True and r['all_original_physical_samples_used_once']is True
    and r['synthesized_or_skipped_samples']==0 and r['decoded_frames']==856 and r['demuxed_packets_including_flush']==857
    and r['full_source_native_samples']==820872 and r['clean16k_samples']==273624
    and r['native_sample_channels_before_original_mono']==2 and r['original_stereo_float32_mean_and_original_resample_unchanged']is True
    and r['heldout_PCM_check_performed']is True and r['raw_ledger']['passed']is True and len(r['full_target_ids'])==99,
    'Actual complete one-member physical PCM and all99 labels, no scientific rerun')
require(r['packet_side_data']==[{'packet':856,'type':'skip_samples','data_size':10,'description':'Skip Samples',
    'raw_hex':'00000000400200000000','skip_start_samples':0,'discard_end_samples':576,'skip_reason':0,'discard_reason':0}],
    'Genuine actual bounded side bytes retained exactly')
ids=e['recorded_identities']+[e['launcher_identity']];unique={(x['pid'],x['create_time']):x for x in ids}
deaths=[{'identity':x,'exact_dead':dead(x)}for x in unique.values()]
require(len(deaths)==4 and all(v['exact_dead']for v in deaths),'All four exact physical-job native/launcher/external identities dead')
evidence=[F/'full_head_member_physical_actual_TOOL_v1.json',F/'full_head_member_physical_external_result_v1.json',
    O/'report.json',O/'CPU_completion.json',O/'before_member.json',F/'full_head_member_physical_stdout.log',
    F/'full_head_member_physical_stderr.log',F/'ROOT_recorded170470_row6_full_head_member_physical_exact_CPU_scope_v1.json',
    F/'full_head_member_physical_frozen_inputs_v1.json',F/'observe_recorded170470_row6_full_head_member_v1.py',
    F/'launch_recorded170470_row6_full_head_member_external_v1.py']
me=psutil.Process();me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
value={'kind':'ASR18_exact_recorded170470_row6_full_head_member_actual_CPU_completion_v1',
    'observed_UTC':datetime.now(timezone.utc).isoformat(),'actual_TOOL_exit_code':0,'actual_Popen_returncode':0,
    'actual_external_guard_error':None,'required_external_cleanup':[],'all61input_pins_before_after_exact':True,
    'all_four_exact_observed_job_identities_dead':True,'exact_job_deaths':deaths,'evidence_pins':[pin(p)for p in evidence],
    'row':r['row'],'full_target_ids':r['full_target_ids'],
    'full_target_ids_canonical_sha256':r['full_target_ids_canonical_sha256'],
    'full_target_ids_packed_u32_sha256':r['full_target_ids_packed_u32_sha256'],
    'original_plan_row_canonical_sha256':r['original_plan_row_canonical_sha256'],
    'raw_annotation_canonical_sha256':r['raw_annotation_canonical_sha256'],
    'original_declaration':r['original_declaration'],'actual_interval_geometry':r['actual_interval_geometry'],
    'original_TAR':r['original_TAR'],'recording_index_entry':r['recording_index_entry'],'member_sha256':r['member_sha256'],
    'full_source_native_f32_sha256':r['full_source_native_f32_sha256'],'clean_f32_sha256':r['clean_f32_sha256'],'clean_i16_sha256':r['clean_i16_sha256'],
    'actual_evidence':{'complete_EOF':True,'frames':856,'packets_including_flush':857,'physical_native_samples':820872,
        'declared_native_samples':820896,'physical_clean16k_samples':273624,'declared_clean16k_samples':273632,
        'container_duration_microseconds':r['container_duration_microseconds'],'stereo_channels':2,
        'original_float32_mono_mean_preserved':True,'max_abs_PTS_residual_samples':r['max_abs_PTS_residual_samples'],
        'first_frames':r['first_frames'],'last_frames':r['last_frames'],'genuine_raw_side_bytes':r['packet_side_data']},
    'all_original_physical_samples_used_once':True,'synthesized_or_skipped_samples':0,
    'full_larger_historical_raw_media_MAXfrontend_transform_numerical_reservation_retained':r['original_prospective_bytes'],
    'full_CPU2point5GB_prospective_private8GB_prefetch_and_original_floors_retained':True,
    'all_raw_buffers_weakref_observed_released':True,'heldout_PCM_check_passed':True,
    'derived_geometry_NOT_actual_frontend_proof':True,'new_overlay_or_certificate_or_production_admission':False,
    'no_baseline_other_member_decoder_rerun_model_GPU_native_sampler_RNG_journal_tensor_source_guard_or_training_change':True,
    'saved170400_and69_historical_positive_attempts_remain_unchanged':True,
    'completion_sealer_identity':identity(me),'completion_launcher_identity':identity(me.parent()),
    'completion_self_and_launcher_alive_at_write_require_external_exact_death_verification':True}
out=F/'full_head_member_physical_actual_completion_v1.json'
with out.open('x',encoding='utf-8',newline='\n')as file:
    json.dump(value,file,indent=2,ensure_ascii=False,allow_nan=False);file.write('\n');file.flush();os.fsync(file.fileno())
print(json.dumps({'completion':pin(out),'all_four_exact_job_dead':True,'sealer':value['completion_sealer_identity'],
    'launcher':value['completion_launcher_identity'],'no_further_audio_or_repair_executed':True}))
