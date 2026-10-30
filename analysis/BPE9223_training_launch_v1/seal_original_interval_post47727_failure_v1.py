"""Read-only closed-owner observation and preserved saved-prefix tail, no relaunch."""
import hashlib,json,sys
from pathlib import Path
import psutil
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
N=Path(__file__).parent;F=N/'resident_RAM_original_interval_post47727_failure_actual_v1';require(not F.exists(),'Fresh E failure observation');F.mkdir()
B=Path(r'E:\gpu_training_scratchpad\runs\BPE9223_resident_RAM_20261018_01');I=B/'invocations/owner_000008';O=N/'resident_RAM_training_actual_09'
failure=read_json(I/'actual_failure.json');latest=read_json(B/'latest_checkpoint.json');ptr=read_json(latest['pointer']['path'])
require(failure['actual_optimizer_updates']==47727 and failure['last_checkpoint']==latest['committed_updates']==ptr['committed_updates']==47600,'Observed closed47727/saved47600')
require(latest['full_state_verified'] is True and record(latest['pointer']['path'])==latest['pointer'],'Actual published pointer pin')
proof=read_json(latest['persistence']['path']);require(proof['passed'] and proof['weights_verified']==207 and proof['named_tensors_verified']==854 and proof['Adam_states_verified']==215,'Existing actual persistence, no new GPU restore claim')
child=read_json(O/'current_child.json');ready=read_json(I/'CPU/actual_CPU_ready.json')
pairs=[failure['producer'],child['launcher'],ready['source']['producer']]
deaths=[]
for who in pairs:
    try:current=psutil.Process(who['pid']).create_time();same=abs(current-who['create_time'])<.001
    except psutil.NoSuchProcess:current=None;same=False
    require(not same,'Exact old identity is dead')
    deaths.append({'identity':who,'same_identity_alive':False,'current_create_time_if_PID_reused':current})
unknown=[]
for pid,role in ((child['launcher']['ppid'],'old_wrapper'),(ready['source']['producer']['ppid'],'CPU_interpreter_launcher')):
    require(not psutil.pid_exists(pid),'Old PID without historical ctime currently absent')
    unknown.append({'pid':pid,'role':role,'historical_create_time_available':False,'PID_currently_absent':True})
monitor=read_json(I/'monitor_completion.json');require(monitor['joined'] is True and monitor['clean'] is True and monitor['error'] is None,'Real monitor completion remains clean')
journal=Path(ptr['journal_prefix']['path']);remaining=ptr['journal_prefix']['offset'];h=hashlib.sha256()
with journal.open('rb') as stream:
    while remaining:
        block=stream.read(min(4*1024**2,remaining));require(bool(block),'Whole saved journal prefix present');h.update(block);remaining-=len(block)
    tail=stream.read()
require(h.hexdigest()==ptr['journal_prefix']['sha256'],'Exact saved whole journal prefix')
events=[json.loads(line) for line in tail.splitlines() if line];ids=[e['optimizer_updates'] for e in events]
require(ids==list(range(47601,47728)),'All105 actual positive unsaved events preserved in order')
tail_path=F/'preserved_actual_positive_tail_47601_47727.jsonl';tail_path.write_bytes(tail)
plan_paths=[I/'plans/plan_000127.json',I/'plans/plan_000128.json'];plan=read_json(plan_paths[0])
require(plan['initial_committed_updates']==47720 and len(plan['updates'])==20,'Only current47721..47740 submitted; next128 failed at submit before enqueue')
failed=F/'actual_current_plan47721_47740_remaining47728_47740.json';write_json(failed,{'initial_committed_updates':47720,'first_unconsumed_index':7,'updates':plan['updates'][7:]})
write_json(F/'observed_latest_checkpoint_at_failure.json',latest);write_json(F/'observed_progress_at_failure.json',read_json(B/'progress/latest.json'))
lease=P/'work/gpu_training_owner.json';write_json(F/'observed_closed_lease.json',read_json(lease))
write_json(F/'actual_owned_tool_exit.json',{'session_id':39651,'actual_tool_exit_code':1,'actual_tool_chunk':'33ed22','actual_wrapper_command_exit_code':1,'native_numeric_OS_exit_code_not_persisted':True})
config_path=child['command'][child['command'].index('--configuration')+1]
result={'kind':'ASR18_actual_resident_RAM_original_interval_failure_saved47600_tail47727_v1','actual_owned_session':39651,
 'actual_exec_tool_exit_code':1,'actual_tool_chunk':'33ed22','actual_wrapper_command_exit_code':1,'actual_native_OS_exit_code':None,
 'actual_native_nonzero_wait_observed_by_wrapper':True,'actual_CPU_OS_exit_code':None,'numeric_OS_exit_codes_not_persisted':True,
 'normal_native0_claimed':False,'intentional_stop_claimed':False,'branch':str(B),'producer':failure['producer'],
 'captured_identity_pairs':pairs,'observed_identity_deaths':deaths,'all_captured_identity_pairs_dead':True,'known_old_PIDs_without_recorded_create_time':unknown,
 'actual_optimizer_updates_before_failure':47727,'saved_optimizer_updates':47600,
 'original_failure':record(I/'actual_failure.json'),'original_monitor_completion':record(I/'monitor_completion.json'),
 'original_CPU_stderr':record(I/'CPU/CPU_stderr.log'),'original_CPU_ready':record(I/'CPU/actual_CPU_ready.json'),
 'original_native_log':record(O/'native_000008.log'),'original_current_child':record(O/'current_child.json'),
 'source_configuration':record(config_path),'effective_configuration':record(I/'configuration.json'),
 'worker_source':record(P/'work/research/run_ASR18_BPE9223_resident_RAM_training_v9.py'),
 'ROOT_execution_review':record(N/'ROOT_second_original_full_clip_saved45200_execution_review_v1.json'),
 'checkpoint_pointer':latest['pointer'],'checkpoint_persistence':latest['persistence'],
 'latest_checkpoint_snapshot':record(F/'observed_latest_checkpoint_at_failure.json'),'saved_journal_prefix':ptr['journal_prefix'],
 'whole_journal_including_preserved_tail':record(journal),'preserved_unsaved_tail':record(tail_path),'preserved_tail_positive_updates':ids,
 'original_journal_or_saved_state_mutated':False,'actual_current_preparation_plan':record(plan_paths[0]),'unsubmitted_next_plan':record(plan_paths[1]),'first_unconsumed_update_index':7,'candidate_failed_global_updates':list(range(47728,47741)),
 'failed_update_metadata':record(failed),'failed_global_update_and_row_not_yet_identified':True,
 'cause_scope':'Top submit guard masked already failed CPU. CPU stderr ordinary yodas written!=count. Queue was not drained, so failed update is only bounded47728..47740, not assumed47728. No reset, retry or media repair.'}
path=F/'actual_failure_closure.json';write_json(path,result)
print(json.dumps({'receipt':record(path),'saved':47600,'actual_native':47727,'preserved_positive_tail':len(ids),'same_identity_deaths':len(deaths),'unknown_ctime_PIDs_absent':len(unknown),'failed_global_update_unknown':True}),flush=True)
