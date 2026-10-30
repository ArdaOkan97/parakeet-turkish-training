"""Exact current combined-milestone process closure; no training evidence reaudit."""
import hashlib,json,os,shutil,sys,time
from pathlib import Path
import psutil
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20')
sys.path.insert(0,str(P/'outputs/parakeet_turkish'))
from resource_watchdog_v2 import sample_memory
A=Path(__file__).resolve().parent.parent.parent/'resident_RAM_training_actual_27'
began=time.monotonic();psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
def pin(p):
 raw=p.read_bytes();return {'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def write(p,v):
 with p.open('x',encoding='utf-8',newline='\n')as f:
  json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def who(pid):
 process=psutil.Process(pid);return {'pid':pid,'create_time':process.create_time(),'ppid':process.ppid()}
def alive(identity):
 try:return abs(psutil.Process(identity['pid']).create_time()-identity['create_time'])<.001
 except psutil.NoSuchProcess:return False
sample=sample_memory(os.getpid(),'firstpass188000_Backlog_completion_initial')
ours=[who(os.getpid()),who(os.getppid()),who(psutil.Process(os.getppid()).ppid())]
initial=A/'actual_firstpass188000_backlog_completion_initial_numeric_before_predicate_v1.json'
write(initial,{'sample':sample,'required_commit':5055306368,'required_physical':2305306368,
 'metadata_identities':ours,'persisted_before_predicate':True})
assert sample['paired_sampling_within_one_second']and sample['worker_private_memory']['private_bytes']<536870912
assert sample['system_memory']['commit_available_bytes']>=5055306368 and sample['system_memory']['physical_available_bytes']>=2305306368
assert shutil.disk_usage('E:/').free>=100000000000
result_path=A/'actual_firstpass187937_periodic188000_backlog_result_v1.json'
assert pin(result_path)['sha256']=='e4a5924836590c74b629039373d21d8f9b9d58552cb2853ae88cbbe8dbb3324a'
result=json.loads(result_path.read_text(encoding='utf-8'))
assert result['passed']and result['actual_CLI_edit_exit']==result['actual_refresh_exit']==0
assert result['marker_count_before']==0 and result['marker_count_after']==1 and result['CLI_edit_invocations']==1
identities=result['metadata_identities_alive_at_write']
unknown=[]
for process in result['subprocesses']:
 identity=process['Popen_identity']
 if isinstance(identity['create_time'],(int,float)):identities.append(identity)
 else:unknown.append(identity)
closed=[dict(identity,observed_exact_dead=not alive(identity))for identity in identities]
assert all(v['observed_exact_dead']for v in closed)
assert time.monotonic()-began<180
path=A/'actual_firstpass187937_periodic188000_backlog_completion_v1.json'
write(path,{'kind':'ASR18_source37_firstpass187937_periodic188000_Backlog_completion_v1','passed':True,
 'actual_TOOL_exit_code':0,'actual_TOOL_chunk':'c1a9ce','actual_result':pin(result_path),
 'original_metadata_and_recorded_Popen_identities':closed,'unrecorded_subprocess_identity_details':unknown,
 'completion_identities_alive_at_write':ours,'initial_numeric_sample':pin(initial),
 'no_repeat_edit_refresh_or_quality_source_journal_plan_archive_tensor_audio_reads':True,
 'production26493_unchanged':True,'source':pin(Path(__file__))})
print(json.dumps({'completion':pin(path),'original_exact_death_count':len(closed),
 'unknown_subprocess_identities':unknown,'completion_identities':ours}),flush=True)
