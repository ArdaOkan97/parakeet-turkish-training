"""Seal actual closed24600, preserving the failed observer and absent teardown report."""
import ast, difflib, hashlib, json, sys
from pathlib import Path
import psutil
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish';R=P/'work/research'
N=Path(__file__).parent;B=Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01');I=B/'invocations/owner_000005'
sys.path[:0]=[str(S),str(R)];psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p):
 p=Path(p);b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def write(p,v):
 assert not Path(p).exists(),str(p)
 Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def replace(s,old,new):assert s.count(old)==1,old[:100];return s.replace(old,new)
old=S/'asr18_resident_source_transition_v1.py';new=S/'asr18_resident_source_transition_v2.py';assert not new.exists()
s=old.read_text(encoding='utf-8').split('\ndef watch(')[0]
s=replace(s,"and value['actual_native_exit_code'] == value['actual_wrapper_exit_code'] == 1,", """and value['actual_native_exit_code'] is None and value['native_numeric_OS_exit_available'] is False
            and value['actual_native_nonzero_wait_observed_by_old_wrapper'] is True
            and value['actual_exec_tool_exit_code'] == 1 and value['actual_exec_tool_chunk'] == 'cfcee9'
            and value['observer_actual_exec_tool_exit_code'] == 1 and value['observer_actual_tool_chunk'] == '3f3f34',""")
s=replace(s,"    require([x['identity'] for x in value['actual_process_exit_codes']] == intent['captured_identities']\n            and all(x['actual_exit_code'] != 259 for x in value['actual_process_exit_codes']), 'All captured process handles actually exited')", """    require(len(intent['captured_identities']) == 7 and value['captured_identities'] == intent['captured_identities']
            and value['all_old_processes_observed_dead'] is True and value['saved_optimizer_updates'] == 24600,
            'Only the exact seven closed old identities and actual24600 transition')""")
s=replace(s,"'stop_request', 'original_failure', 'original_monitor', 'original_native_log',", "'stop_request', 'original_failure', 'original_native_log', 'observer_failure', 'old_session_actual_TOOL1',")
s=replace(s,"    monitor = read_json(value['original_monitor']['path'])\n    require(monitor['joined'] is True and monitor['clean'] is True and monitor['error'] is None,\n            'No concurrent memory-monitor failure hidden as intentional transition')", """    require(value['original_monitor'] is None and value['original_monitor_absent'] is True
            and value['clean_monitor_report_claimed'] is False
            and not (Path(intent['old_invocation'])/'monitor_completion.json').exists(),
            'Guarded teardown did not finish; never fabricate joined/clean monitor proof')
    require('queued=submit_plan(private)' in failure['traceback'] and 'Actual owner cooperative stop' in failure['traceback'],
            'Actual stop was in private future planning, before new positive native work')
    with Path(value['original_native_log']['path']).open('rb') as stream:
        stream.seek(max(0,stream.seek(0,2)-16384)); teardown=stream.read().decode('utf-8')
    require('scopes.close()' in teardown and 'resident_gc_scope' in teardown and 'live_state_fingerprint' in teardown
            and 'ValueError: Actual owner cooperative stop' in teardown,
            'Exact preserved stop guard re-entry during source4 GC-scope teardown')""")
s=replace(s,"    require(tail == value['unsaved_tail_bytes'], 'Truthful actual tail length')", "    require(tail == value['unsaved_tail_bytes'] == 0 and value['actual_optimizer_updates_before_exit'] == 24600,\n            'Exact actual zero-tail saved24600; no broadened failure adoption')")
ast.parse(s);new.write_text(s,encoding='utf-8',newline='\n')
N.joinpath('validator1_to2.diff.txt').write_text(''.join(difflib.unified_diff(old.read_text(encoding='utf-8').splitlines(True),s.splitlines(True),fromfile=old.name,tofile=new.name)),encoding='utf-8')
for role in('training','autocontinue'):
 old=R/f'run_ASR18_BPE9223_resident_RAM_{role}_v6.py';new=R/f'run_ASR18_BPE9223_resident_RAM_{role}_v7.py';assert not new.exists()
 text=old.read_text(encoding='utf-8').replace('asr18_resident_source_transition_v1','asr18_resident_source_transition_v2')
 if role=='autocontinue':text=text.replace('run_ASR18_BPE9223_resident_RAM_training_v6.py','run_ASR18_BPE9223_resident_RAM_training_v7.py')
 ast.parse(text);new.write_text(text,encoding='utf-8',newline='\n')
 N.joinpath(role+'6_to7.diff.txt').write_text(''.join(difflib.unified_diff(old.read_text(encoding='utf-8').splitlines(True),text.splitlines(True),fromfile=old.name,tofile=new.name)),encoding='utf-8')
from fastmss_owned_process_closure_v2 import dead
intent_pin=pin(N/'transition_intent.json');intent=read(intent_pin['path']);assert all(dead(w)for w in intent['captured_identities'])
latest=read(B/'latest_checkpoint.json');failure=read(I/'actual_failure.json');pointer=read(latest['pointer']['path']);saved=pointer['journal_prefix'];whole=pin(saved['path'])
assert latest['committed_updates']==failure['actual_optimizer_updates']==failure['last_checkpoint']==24600
assert whole=={'path':saved['path'],'bytes':saved['offset'],'sha256':saved['sha256']}
assert not (I/'monitor_completion.json').exists()
out=N/'actual_transition_02'
write(out/'observer_actual_TOOL1.json',{'kind':'ASR18_actual_transition_observer_seal_failure_TOOL1_v1','actual_owned_session':95468,
 'actual_exec_tool_exit_code':1,'actual_tool_chunk':'3f3f34','error':'FileNotFoundError monitor_completion.json after all actual process handle waits returned',
 'source':pin(S/'asr18_resident_source_transition_v1.py'),'old_guard_stop_already_performed':True,
 'native_numeric_exit_codes_not_persisted_or_claimed':True})
write(out/'old_session_actual_TOOL1.json',{'kind':'ASR18_actual_owned_source4_wrapper_TOOL1_v1','actual_owned_session':45407,
 'actual_exec_tool_exit_code':1,'actual_tool_chunk':'cfcee9','normal0_claimed':False,
 'observed_wrapper_error':'ValueError: Actual native process failed; preserve all state/logs, no unchanged retry'})
closure={'kind':'ASR18_actual_cooperative_prefetch_source_transition_TOOL1_v1','intent':intent_pin,'intentional_source_transition':True,
 'normal_native0_claimed':False,'branch':str(B),'actual_native_exit_code':None,'native_numeric_OS_exit_available':False,
 'actual_native_nonzero_wait_observed_by_old_wrapper':True,'actual_exec_tool_exit_code':1,'actual_exec_tool_chunk':'cfcee9',
 'observer_actual_exec_tool_exit_code':1,'observer_actual_tool_chunk':'3f3f34','captured_identities':intent['captured_identities'],
 'all_old_processes_observed_dead':True,'checkpoint_pointer':latest['pointer'],'checkpoint_persistence':latest['persistence'],
 'saved_optimizer_updates':24600,'actual_optimizer_updates_before_exit':24600,'saved_journal_prefix':saved,
 'whole_journal_including_preserved_tail':whole,'unsaved_tail_bytes':0,'preserved_unsaved_tail':None,'preserved_tail_positive_updates':[],
 'stop_request':pin(I/'stop.json'),'original_failure':pin(I/'actual_failure.json'),'original_monitor':None,
 'original_monitor_absent':True,'clean_monitor_report_claimed':False,'original_native_log':pin(intent['old_native_log']),
 'observer_failure':pin(out/'observer_actual_TOOL1.json'),'old_session_actual_TOOL1':pin(out/'old_session_actual_TOOL1.json'),
 'original_CPU_stderr':pin(I/'CPU/CPU_stderr.log'),'new_stop_or_recovery_performed_while_sealing':False}
closure_path=out/'actual_transition_closure_v2.json';write(closure_path,closure)
from asr18_resident_source_transition_v2 import checked_source_transition_resume
configuration=read(N/'configuration.json');checked_source_transition_resume(pin(closure_path),base=configuration,folder=B,pointer_pin=latest['pointer'])
lease_path=P/'work/gpu_training_owner.json';raw=lease_path.read_bytes();lease=json.loads(raw)
assert lease['owner']=='/root/memory_diagnosis' and lease['actual_owned_worker_pid']==2956 and dead(intent['old_native'])
saved_lease=out/'previous_closed_source4_lease.json';saved_lease.write_bytes(raw)
lease.update(running=False,actual_GPU_running=False,actual_training_optimizer_running=False,actual_owned_worker_pid=None,
 status='production_intentional_prefetch_source_transition_closed',committed_updates=24600,current_owned_session=None,
 actual_source_transition=pin(closure_path),previous_closed_owned_session=45407)
from atomic_io_retry_v1 import write_json
write_json(lease_path,lease)
handoff={'kind':'ASR18_actual_saved24600_prefetch_transition_execution_handoff_v1','driver':pin(R/'run_ASR18_BPE9223_resident_RAM_training_v7.py'),
 'wrapper':pin(R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v7.py'),'source_transition_validator':pin(S/'asr18_resident_source_transition_v2.py'),
 'source_transition_resume':pin(closure_path),'source_configuration':pin(N/'configuration.json'),'initial_checkpoint':latest['pointer'],
 'checkpoint_persistence':latest['persistence'],'latest_failure':pin('E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1/pipeline_overlap_RUN07_actual_01/actual_control_namespace_failure_TOOL1_v1.json'),
 'unchanged_RAM_native':pin(S/'asr18_BPE9223_grouped_RAM_native_v1.py'),'unchanged_CPU':pin(S/'fastmss_resident_feature_stream_v1.py'),
 'unchanged_DEV_helper':pin(S/'asr18_resident_serial350_DEV_v2.py'),'transition_intent':intent_pin,
 'diffs':{x:pin(N/x)for x in('validator1_to2.diff.txt','training6_to7.diff.txt','autocontinue6_to7.diff.txt')},
 'source6_reviewed_preparation':pin(N/'source_handoff.json'),'old_family_all_dead':True,'actual_whole_journal_equals_saved_prefix':True,
 'native_numeric_exit_unknown':True,'absent_monitor_not_promoted_to_clean':True,'actual_serialized_validator_passed':True,
 'old_lease_preserved':pin(saved_lease),'idle_lease':pin(lease_path),'new_branch_or_journal_reset':False,
 'output':'E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1/resident_RAM_training_actual_07',
 'start_invocation':6,'queue_bytes':268435456,'full_target':401072,'administrative_update_stop':None}
write(N/'actual_execution_handoff.json',handoff);print(json.dumps({'handoff':pin(N/'actual_execution_handoff.json'),'closure':pin(closure_path),'validator':handoff['source_transition_validator'],'driver':handoff['driver'],'wrapper':handoff['wrapper']}))
