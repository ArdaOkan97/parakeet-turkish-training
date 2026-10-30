import sys,json,time,os
from pathlib import Path
P=Path(r'C:/Users/ardao/Documents/Codex/2026-10-10/i-x20')
sys.path[:0]=[str(P/'outputs/parakeet_turkish'),str(P/'work/research')]
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
from fastmss_owned_process_closure_v2 import dead
from run_ASR18_BPE9223_full_run_owner_v20 import storage,sample_memory
F=Path(__file__).parent;N=F.parent
storage();began=time.monotonic()
def guard():
    v=sample_memory(os.getpid(),'seal_actual_preCPU_failure')
    require(time.monotonic()-began<60 and v['worker_private_memory']['private_bytes']<128000000
        and v['system_memory']['commit_fraction']<.95 and v['system_memory']['commit_available_bytes']>2000000000,'Original metadata guards')
launch=read_json(F/'launch.json');c=read_json(launch['configuration']['path']);q=read_json(launch['resume_request']['path'])
prior_pin=c['prior_ACKed260_metadata_path_failure'];prior=read_json(prior_pin['path']);run=Path(c['run_namespace'])
guard();require(dead(launch['producer']),'Actual parent PID/create-time dead')
joined=read_json(F/'owner_guard_completion.json');require(joined['producer']==launch['producer'] and joined['monitor_joined'] and joined['monitor_clean'],'Actual clean joined guard')
pins={k:record(v['path'])for k,v in prior['frozen_current_state_pins'].items()}
changed=[k for k,v in pins.items()if v!=prior['frozen_current_state_pins'][k]]
require(changed==['control/worker_control_pool/registry.json'],'Only real protection-registry revision changed')
registry=read_json(run/'control/worker_control_pool/registry.json')
require(registry['revision']==5 and registry['reservation_count']==2 and registry['archive_count']==0,'Actual unchanged live reservations and protection revision5')
request_dir=F/'singlepass_request_000000013'
require(request_dir.is_dir() and not list(request_dir.iterdir()) and not (F/'singlepass_CPU_000000013').exists(),'Failure before CPU request/report/Popen')
guard()
result={'kind':'ASR18_actual_ACKed260_singlepass_control_namespace_failure_TOOL1_v1','actual_tool_exit_code':1,
 'actual_exec_tool_exit_code':1,'actual_tool_chunk_id':'da511c','actual_owned_session':36626,
 'producer':launch['producer'],'actual_parent_dead':True,'owner_guard_completion':record(F/'owner_guard_completion.json'),
 'configuration':launch['configuration'],'execution_review':launch['review'],'resume_request':launch['resume_request'],
 'source':c['full_owner_controller_source'],'parent_source':c['parent_driver_source'],
 'prior_ACKed260_metadata_path_failure':prior_pin,'preserved_production_updates':260,'new_optimizer_updates':0,
 'new_CPU_children':0,'new_training_GPU_children':0,'new_recovery_GPU_children':0,
 'actual_native_normal_exit0_claimed':False,'durable_ACK_committed':True,'pending_prefix_publications':0,
 'failure_message':'Only registered live or exactly archived worker directories',
 'failure_call_chain':'FullOwner._execute_prepared_CPU -> _singlepass_request -> parent.check -> original2.control_guard',
 'failed_empty_request_directory':str(request_dir),'CPU_control_directory_created':False,
 'frozen_current_state_pins':pins,'changed_original_pins':changed,
 'current_checkpoint_capture':prior['current_checkpoint_capture'],'resource_prefix_acknowledgement':prior['resource_prefix_acknowledgement'],
 'coverage':prior['coverage'],'actual_normal_fresh_recovery0':prior['actual_normal_fresh_recovery0'],
 'actual_current_recovery_report':prior['actual_current_recovery_report'],'checkpoint_pointer':prior['checkpoint_pointer'],
 'durable_head':prior['durable_head'],'durable_head_state':prior['durable_head_state'],
 'worker_control_pool_state':registry,'counters':{'actual_owner_invocation_count':3,'native_sequence':1,
 'recovery_sequence':1,'recovery_control_sequence':1,'proof_sequence':50,'last_window_index':12},
 'prior_request_owner_state':q['owner_state'],'prior_original_captured_identities_dead':prior['all_captured_identities_dead'],
 'outer_launcher_identity_not_captured':{'pid':launch['producer']['ppid'],'create_time':None},
 'old_failure_and_partial_p14_preserved':True,'metadata_seal_elapsed_seconds':time.monotonic()-began}
out=F/'actual_control_namespace_failure_TOOL1_v1.json';require(not out.exists(),'New failure receipt');write_json(out,result)
print(json.dumps(record(out)),flush=True)
