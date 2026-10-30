"""Current physical-result metadata closure only; no decoder/other target work."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import psutil

S=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20/outputs/parakeet_turkish');sys.path.insert(0,str(S))
from resource_watchdog_v2 import sample_memory
F=Path('E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1/resident_RAM_source36_original_interval186563_failure_actual_v1')
O=F/'recorded186564_row5_full_original_member_physical_actual_01'
T=Path('E:/gpu_training_scratchpad/analysis/BPE9223_Trackio_live_v1')
def require(value,message):
    if not value:raise ValueError(message)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def pin(path):
    p=Path(path).resolve();before=p.stat();require(before.st_size<=8388608,'NEW compact metadata/source only')
    data=p.read_bytes();after=p.stat();require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Stable compact result')
    return {'path':str(p),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
def identity(p):return {'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()}
def same_alive(who):
    try:return abs(psutil.Process(who['pid']).create_time()-who['create_time'])<.001
    except psutil.NoSuchProcess:return False
def write_new(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n')as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())

me=psutil.Process();me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);began=time.monotonic()
require(os.environ['CUDA_VISIBLE_DEVICES']=='-1'and all(os.environ[k]=='1'for k in('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'))
    and all(Path(os.environ[k]).resolve().is_relative_to(Path('E:/gpu_training_scratchpad').resolve())for k in('TEMP','TMP','HF_HOME','TORCH_EXTENSIONS_DIR')),'Original process-local E CPU-only metadata env')
sample=sample_memory(me.pid,'recorded186564_row5_physical_actual_completion_metadata_only')
require(sample['paired_sampling_within_one_second']and sample['worker_private_memory']['private_bytes']<=536870912
    and sample['system_memory']['commit_available_bytes']>=5055306368
    and sample['system_memory']['physical_available_bytes']>=2305306368
    and shutil.disk_usage('E:/').free>=100000000000,'Full metadata prospective512 + original floors/prefetch/E100GB')
scope_path=F/'ROOT_recorded186564_row5_full_original_member_physical_exact_ONCE_execution_scope_v1.json'
scope=read(scope_path);ext_path=F/'recorded186564_row5_full_original_member_physical_external_result_v1.json'
external=read(ext_path);report_path=O/'report.json';report=read(report_path)
gate_path=F/'recorded186564_row5_physical_manual_prelaunch_gate_v1.json';gate=read(gate_path)
initial_path=F/'recorded186564_row5_physical_actual_initial_resource_sample_BEFORE_predicate_v1.json'
require(scope['scope']==report['kind']and scope['maximum_actual_attempts']==1 and scope['sole_owner']=='/root/memory_diagnosis','Same ONE admitted physical observation')
require(external['passed']is True and external['Popen_returncode']==0 and external['external_guard_error']is None
    and external['required_external_cleanup']==[]and not external.get('cleanup_error')and not external.get('not_closed_exact_identities'),'Real external0, no guard/cleanup')
require(report['passed']is True and report['guard_error']is None and report['input_pins_before_after_exact']is True
    and report['complete_demux_and_decode_EOF']is True and report['all_original_physical_samples_used_once']is True
    and report['synthesized_or_skipped_samples']==0 and report['raw_ledger']['passed']is True,'Actual complete EOF/raw release/no padding or skipping')
require(report['input_pins_before']==report['input_pins_after']==external['input_pins_before']==external['input_pins_after']==scope['input_pins']
    and len(scope['input_pins'])==26 and external['report']==pin(report_path),'All26 actual before/after pins and exact report; no original closed-input reread')
require(report['row']['ordinal']==156134 and report['row']['source_id']=='a280321c852958fbb53e9d7a2044d1a6'
    and len(report['full_target_ids'])==66 and report['original_declaration']['encoder_frames']==165
    and report['original_prospective_bytes']==901306112 and report['original_actual_row_arithmetic_bytes']==885176752,'Exact original row/full66/larger preserved envelope')
require(gate['gate_passed']is True and gate['actual_initial_numeric_sample_persisted_BEFORE_predicate']==pin(initial_path),'Actual initial full numeric sample before predicate retained')
known={}
for who in [*external['recorded_identities'],external['launcher_identity'],gate['producer'],gate['launcher']]:known[(who['pid'],who['create_time'])]=who
deaths=[{'identity':who,'exact_dead':not same_alive(who),'separate_numeric_OS_exit':'UNKNOWN unless represented by actual Popen/TOOL result'}for who in known.values()]
require(all(value['exact_dead']for value in deaths),'All recorded preflight/fixture/external metadata identities exact-dead')
old=[{'pid':21912,'create_time':1792780981.1092248},{'pid':37108,'create_time':1792780981.1024935},{'pid':29600,'create_time':1792780979.698954},{'pid':39856,'create_time':1792790240.3032615},{'pid':35440,'create_time':1792790240.2869232}]
reader=[{'pid':26892,'create_time':1792652904.3359847},{'pid':30128,'create_time':1792652904.283631}]
require(all(not same_alive(w)for w in old)and all(same_alive(w)for w in reader),'Training stays stopped and exact reader2 healthy unchanged')
status_path=T/'current_status.json';before=status_path.stat();require(before.st_size<=262144,'Only one compact CURRENT reader status, no timing/log/journal')
status_bytes=status_path.read_bytes();after=status_path.stat();json.loads(status_bytes.decode('utf-8-sig'))
status_snapshot=F/'recorded186564_row5_physical_CURRENT_reader_status_snapshot_v1.json'
tool_path=F/'recorded186564_row5_physical_actual_TOOL_result_v1.json'
completion_path=F/'recorded186564_row5_physical_actual_completion_v1.json'
require(all(not path.exists()for path in(status_snapshot,tool_path,completion_path)),'Fresh immutable current completion outputs')
with status_snapshot.open('xb')as stream:stream.write(status_bytes);stream.flush();os.fsync(stream.fileno())
tool={'kind':'ASR18_actual_owned_exec_TOOL_result_v1','owner':'/root/memory_diagnosis',
    'actual_owned_exec_session':None,'actual_execution_completed_in_initial_TOOL_call':True,
    'actual_TOOL_exit_code':0,'actual_TOOL_chunk_id':'42955f','actual_external_reported_Popen_returncode':0,
    'external_result':pin(ext_path),'exact_external_command':scope['exact_external_command'],'ROOT_execution_scope':pin(scope_path),
    'actual_manual_preflight_TOOL0_chunk_id':'497e56','manual_preflight':pin(gate_path),
    'separate_individual_native_fixture_CPU_wrapper_OS_exits_NOT_inferred':True,'one_actual_attempt_no_retry_rebind_or_production_action':True}
write_new(tool_path,tool)
inventory=[pin(path)for path in sorted(O.iterdir())]
require(len(inventory)<=4,'Bounded physical metadata outputs, no payload export')
completion={'kind':'ASR18_recorded186564_row5_full_original_member_physical_actual_completion_v1',
    'owner':'/root/memory_diagnosis','producer':identity(me),'launcher':identity(me.parent()),'completion_source':pin(__file__),
    'completion_pair_alive_at_write_requires_exact_death_check_after_TOOL_returns':True,
    'initial_full_metadata_guard_sample':sample,'actual_TOOL':pin(tool_path),'actual_external':pin(ext_path),
    'actual_report':pin(report_path),'actual_manual_gate':pin(gate_path),'actual_numeric_initial_before_predicate':pin(initial_path),
    'actual_metadata_outputs':inventory,'all_recorded_preflight_and_observation_identity_deaths':deaths,
    'actual_full_EOF_physical_native_samples':report['full_source_native_samples'],
    'actual_full_EOF_clean16k_samples':report['clean16k_samples'],
    'actual_native_f32_sha256':report['full_source_native_f32_sha256'],'actual_clean_f32_sha256':report['clean_f32_sha256'],
    'actual_clean_i16_sha256':report['clean_i16_sha256'],'actual_member_sha256':report['member_sha256'],
    'actual_channels_before_original_mono':report['native_sample_channels_before_original_mono'],
    'actual_decoded_frames':report['decoded_frames'],'actual_demuxed_packets_including_flush':report['demuxed_packets_including_flush'],
    'actual_derived_geometry_NOT_frontend':report['actual_interval_geometry'],
    'actual_packet_side_data':report['packet_side_data'],'actual_max_abs_PTS_residual_samples':report['max_abs_PTS_residual_samples'],
    'complete_original_physical_samples_once_zero_synthesized_skipped':True,'full66targets_original_annotation_T165_preserved':True,
    'larger901306112_original_full_numerical_media_raw_budget_and_all_original_guards_preserved':True,
    'actual_raw_ledger_successful_release_source_receipt':report['raw_ledger'],
    'index_DB_closed_in_original_fixture_finally_SOURCE_no_independent_method_handle_counter_receipt':True,
    'actual_guard_error':None,'actual_external_cleanup':[],'actual26before_after_exact':True,
    'actual_external_wall_seconds':external['external_wall_seconds'],'actual_sampled_owned_family_peak_bytes':external['sampled_family_private_peak_bytes'],
    'current_reader_status_snapshot':{'original_path':str(status_path),'snapshot':pin(status_snapshot),
        'source_STAT_before':{'bytes':before.st_size,'mtime_ns':before.st_mtime_ns},'source_STAT_after':{'bytes':after.st_size,'mtime_ns':after.st_mtime_ns},
        'same_STAT_before_after':(before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),
        'raw_compact_snapshot_no_guessed_nested_schema_or_reader_change':True},
    'reader2_same_exact_alive':[{'identity':w,'same_identity_alive':same_alive(w)}for w in reader],
    'training5_remain_exact_dead':[{'identity':w,'exact_dead':not same_alive(w)}for w in old],
    'physical_measurement_requires_ROOT_review_NO_certificate_overlay_CPU_source_or_recovery_preparation':True,
    'no_new_decode_baseline_frontend_model_GPU_GC_state_journal_LR_reader_or_production_action':True,
    'completion_elapsed_seconds':time.monotonic()-began}
write_new(completion_path,completion)
print(json.dumps({'completion':pin(completion_path),'actual_TOOL':pin(tool_path),'report':pin(report_path),
    'producer':completion['producer'],'launcher':completion['launcher'],'recorded_preflight_observation_exact_dead_count':len(deaths),
    'actual_native_samples':report['full_source_native_samples'],'actual_clean16k_samples':report['clean16k_samples'],
    'actual_frames':report['decoded_frames'],'actual_packets':report['demuxed_packets_including_flush'],
    'actual_derived_geometry':report['actual_interval_geometry'],'member_sha256':report['member_sha256'],
    'native_f32_sha256':report['full_source_native_f32_sha256'],'clean_f32_sha256':report['clean_f32_sha256'],
    'clean_i16_sha256':report['clean_i16_sha256'],'actual_side_data':report['packet_side_data'],
    'current_reader_status_snapshot':pin(status_snapshot)}),flush=True)
