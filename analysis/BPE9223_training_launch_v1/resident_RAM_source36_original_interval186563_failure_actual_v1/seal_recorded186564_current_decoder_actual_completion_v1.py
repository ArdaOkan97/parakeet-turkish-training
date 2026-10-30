"""Metadata-only completion for ONE actual observation; no decoder/target work."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import psutil

S=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20/outputs/parakeet_turkish')
sys.path.insert(0,str(S))
from resource_watchdog_v2 import sample_memory
F=Path('E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1/resident_RAM_source36_original_interval186563_failure_actual_v1')
O=F/'recorded186564_ordered14_current_decoder_actual_01'

def require(value,message):
    if not value:raise ValueError(message)
def pin(path):
    p=Path(path).resolve();before=p.stat();require(before.st_size<=8388608,'NEW compact metadata/source only')
    data=p.read_bytes();after=p.stat();require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Stable compact metadata')
    return {'path':str(p),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def identity(p):return {'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()}
def same_alive(who):
    try:return abs(psutil.Process(who['pid']).create_time()-who['create_time'])<.001
    except psutil.NoSuchProcess:return False
def write_new(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n')as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())

me=psutil.Process();me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);began=time.monotonic()
require(os.environ['CUDA_VISIBLE_DEVICES']=='-1'and all(os.environ[k]=='1'for k in('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'))
    and all(Path(os.environ[k]).resolve().is_relative_to(Path('E:/gpu_training_scratchpad').resolve())for k in('TEMP','TMP','HF_HOME','TORCH_EXTENSIONS_DIR')),'Process-local E CPU metadata environment')
sample=sample_memory(me.pid,'recorded186564_current_decoder_actual_completion_metadata_only')
require(sample['paired_sampling_within_one_second']and sample['worker_private_memory']['private_bytes']<=536870912
    and sample['system_memory']['commit_available_bytes']>=5055306368
    and sample['system_memory']['physical_available_bytes']>=2305306368
    and shutil.disk_usage('E:/').free>=100000000000,'Full metadata prospective512/floors/FULLprefetch/E100GB')
scope_path=F/'ROOT_recorded186564_current_overlay9_ordered14_exact_ONCE_execution_scope_v1.json'
scope=read(scope_path);external_path=F/'recorded186564_ordered14_current_decoder_external_result_v1.json'
external=read(external_path);report=read(O/'report.json')
manual_path=F/'recorded186564_ordered14_current_decoder_manual_prelaunch_gate_v1.json';manual=read(manual_path)
require(scope['scope']==report['kind']and scope['allowed_executions']==1 and scope['only_execution_owner']=='/root/memory_diagnosis','Same ONE actual execution')
require(external['passed']is True and external['Popen_returncode']==0 and external['external_guard_error']is None
    and external['required_external_cleanup']==[]and not external.get('cleanup_error')and not external.get('not_closed_exact_identities'),'Actual external Popen0/no guard/no cleanup')
require(report['passed']is True and report['guard_error']is None and report['cleanup_error']is None
    and report['reader_metadata_close_returned']is True and report['original_metadata_connections_empty']is True
    and report['original_metadata_lock_handles_empty']is True and report['torch_imported']is False,'Actual clean original metadata handle closure, no Torch')
require(report['input_pins_before']==report['input_pins_after']==external['input_pins_before']==external['input_pins_after']==scope['input_pins']
    and len(scope['input_pins'])==61,'Actual before/after source61 exact; no re-read of closed inputs')
require(external['report']==pin(O/'report.json')and manual['gate_passed']is True,'Exact actual report and persisted manual gate')
failure=report['first_actual_exception'];where=failure['identity']
require(where['original_update_row_1based']==5 and where['ordinal']==156134
    and where['source_id']=='alania:a280321c852958fbb53e9d7a2044d1a6_00000'
    and where['local_update_index']==3 and where['global_update']==186564
    and failure['exception_type']=='ValueError'and failure['exception_message']=='Original interval beyond source unresolved'
    and len(report['results'])==4,'Actual first propagated original decoder row5; no physical cause inference')
require(read(O/'first_actual_exception.json')==failure,'Immediate propagated exception equals final report')
expected={'producer_identity.json','first_actual_exception.json','report.json','CPU_completion.json'}
expected.update('before_row_%02d.json'%j for j in range(1,6));expected.update('success_row_%02d.json'%j for j in range(1,5))
require({p.name for p in O.iterdir()}==expected,'Only four successes/five attempts, no rows after first failure or payload artifacts')
known={}
for who in [*external['recorded_identities'],external['launcher_identity'],manual['producer'],manual['launcher']]:
    known[(who['pid'],who['create_time'])]=who
deaths=[{'identity':who,'exact_dead':not same_alive(who),'separate_numeric_OS_exit':'UNKNOWN unless represented by actual Popen/TOOL result'}for who in known.values()]
require(len(deaths)==6 and all(value['exact_dead']for value in deaths),'All four actual observation plus two manual-gate identities exact-dead')
old=[{'pid':21912,'create_time':1792780981.1092248},{'pid':37108,'create_time':1792780981.1024935},{'pid':29600,'create_time':1792780979.698954},{'pid':39856,'create_time':1792790240.3032615},{'pid':35440,'create_time':1792790240.2869232}]
reader=[{'pid':26892,'create_time':1792652904.3359847},{'pid':30128,'create_time':1792652904.283631}]
require(all(not same_alive(w)for w in old)and all(same_alive(w)for w in reader),'Production remains stopped, healthy unchanged reader2')
tool_path=F/'recorded186564_current_decoder_actual_owned1953_TOOL_result_v1.json'
completion_path=F/'recorded186564_current_decoder_actual_completion_v1.json'
require(not tool_path.exists()and not completion_path.exists(),'Fresh actual completion namespace')
tool={'kind':'ASR18_actual_owned_exec_TOOL_result_v1','owner':'/root/memory_diagnosis','actual_owned_exec_session':1953,
    'initial_tool_chunk_id':'418b04','actual_terminal_TOOL_exit_code':0,'actual_terminal_TOOL_chunk_id':'679c56',
    'actual_external_reported_Popen_returncode':0,'actual_terminal_stdout_external_result':pin(external_path),
    'exact_prefix':scope['exact_prefix'],'exact_arguments':scope['exact_arguments'],
    'separate_native_fixture_and_individual_wrapper_OS_exits_NOT_inferred_from_TOOL_or_Popen':True,
    'actual_manual_preflight_TOOL0_chunk':'a5e0f3','manual_prelaunch':pin(manual_path),'execution_scope':pin(scope_path),
    'no_second_execution_retry_or_production_action':True}
write_new(tool_path,tool)
completion={'kind':'ASR18_recorded186564_CURRENT_overlay9_ordered14_actual_observation_completion_v1',
    'owner':'/root/memory_diagnosis','producer':identity(me),'launcher':identity(me.parent()),
    'completion_source':pin(__file__),'initial_full_metadata_resource_sample':sample,
    'ROOT_execution_scope':pin(scope_path),'actual_TOOL':pin(tool_path),'actual_external_result':pin(external_path),
    'actual_report':pin(O/'report.json'),'actual_immediate_first_exception':pin(O/'first_actual_exception.json'),
    'manual_prelaunch':pin(manual_path),'all_new_metadata_output_pins':[pin(O/name)for name in sorted(expected)],
    'all_recorded_six_preflight_and_observation_job_identities_exact_dead':deaths,
    'completion_pair_alive_at_write_and_exact_death_must_be_checked_after_actual_TOOL_returns':True,
    'actual_four_preceding_successes_released':True,'first_failed_original_row_1based':5,
    'failed_row_identity':{k:where[k]for k in('ordinal','source_id','dataset','global_update','local_update_index')},
    'full_target_count':len(where['full_target_ids']),'encoder_frames_T':where['original_plan_row']['encoder_frames'],
    'original_annotation':{k:where['raw_original_annotation'][k]for k in('shard','row_index','start','end','duration')},
    'actual_propagated_exception_type':failure['exception_type'],'actual_propagated_exception_message':failure['exception_message'],
    'actual_original_reader_DB_connections_and_lock_handles_closed':True,
    'failed_raw_job_NOT_fabricated_as_released_success':report['raw_ledger_final_scalar_state'],
    'actual_before_after61_pins_exact':True,'actual_inner_guard_error':None,'actual_external_guard_error':None,'actual_external_cleanup':[],
    'actual_external_wall_seconds':external['external_wall_seconds'],'actual_sampled_family_peak_private_bytes':external['sampled_family_private_peak_bytes'],
    'physical_samples_EOF_decoder_scalar_internals_cause_remedy_NOT_proved':True,
    'production5_remain_exact_dead':[{'identity':w,'exact_dead':not same_alive(w)}for w in old],
    'reader2_healthy_unchanged':[{'identity':w,'same_identity_alive':same_alive(w)}for w in reader],
    'summary_only_metadata_parser_TOOL0_4c45fb_historical_PID_ctime_UNRECORDED':True,
    'no_rerun_new_decode_fullmember_baseline_GC_frontend_model_GPU_state_source_recovery_action':True,
    'completion_elapsed_seconds':time.monotonic()-began}
write_new(completion_path,completion)
print(json.dumps({'completion':pin(completion_path),'actual_TOOL':pin(tool_path),'producer':completion['producer'],
    'launcher':completion['launcher'],'all_six_observed_job_identities_exact_dead':True,'first_failed_row_1based':5}),flush=True)
