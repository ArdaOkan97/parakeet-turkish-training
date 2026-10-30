"""ONE accepted first broad-pass plus scheduled188k milestone and normal refresh."""
import datetime,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
import psutil
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish'
sys.path.insert(0,str(S))
from resource_watchdog_v2 import sample_memory
F=Path(__file__).resolve().parent;A=F.parent.parent/'resident_RAM_training_actual_27'
OUT=A/'backlog_firstpass187937_periodic188000_01'
RESULT=A/'actual_firstpass187937_periodic188000_backlog_result_v1.json'
TASK=S/'backlog/tasks/asr-18 - Train-BPE-on-broad-weak-labels-then-fine-tune-with-multilingual-retention.md'
MARKER='BPE9223_SOURCE37_FULLPASS187937_DEV188000_TR205_SAVED188200_RESUMED_20261024'
began=time.monotonic();psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
def pin(p):
 p=Path(p);raw=p.read_bytes();return {'path':str(p.resolve()),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def write(p,v):
 with Path(p).open('x',encoding='utf-8',newline='\n')as f:
  json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def identity(pid):
 process=psutil.Process(pid);return {'pid':pid,'create_time':process.create_time(),'ppid':process.ppid()}
def observed_alive(who):
 try:return abs(psutil.Process(who['pid']).create_time()-who['create_time'])<.001
 except psutil.NoSuchProcess:return False
assert Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad'))
assert not OUT.exists()and not RESULT.exists(),'One fresh operation; no ambiguous retry'
before=TASK.read_text(encoding='utf-8').count(MARKER)
assert before==0,('Canonical marker already present; no duplicate edit or refresh',before)
initial=sample_memory(os.getpid(),'firstpass187937_periodic188000_Backlog_initial')
initial_path=A/'actual_firstpass187937_periodic188000_backlog_initial_numeric_before_predicate_v1.json'
metadata_identities=[identity(os.getpid()),identity(os.getppid()),identity(psutil.Process(os.getppid()).ppid())]
write(initial_path,{'sample':initial,'required_commit':5055306368,'required_physical':2305306368,
 'captured_before_predicate_and_mutation':True,'metadata_identities':metadata_identities})
assert initial['paired_sampling_within_one_second']and initial['worker_private_memory']['private_bytes']<536870912
assert initial['system_memory']['commit_available_bytes']>=5055306368 and initial['system_memory']['physical_available_bytes']>=2305306368
assert shutil.disk_usage('E:/').free>=100000000000
inputs={
 'owner_periodic188000_status':(A/'actual_periodic188000_and_resumed_training_status.json','f8adfa4b1a04d0f6505e871b34d12b6dd997c7cce31ee5b3cf1d90f8b3ea5194'),
 'ROOT_actual_first_fullpass187937_review':(F/'ROOT_source37_first_fullpass187937_full350_normal_CPU14_and_successor_interposition_verification_v1.json','893c06226eea08e33858658224013e7ed30fe0cf667b7d8b4f168952288c65d8'),
 'ROOT_actual_periodic188000_review':(F/'ROOT_periodic188000_full350_source37_CPU14_max2_normal_saved188200_and_resumed_training_verification_v1.json','056d3d839f6c3d7f63edd5ba989716c395b5a2e86b466be2b2f4ca3520078e75')}
pins={}
for key,(path,digest)in inputs.items():
 pins[key]=pin(path);assert pins[key]['sha256']==digest,(key,pins[key])
note=f'''{MARKER}
The first genuine broad-data pass completed naturally at optimizer187937 and advanced to broad pass2 through the original lifecycle. Its original ordered seven-language 350-example DEV was independently verified by ROOT: Turkish207/825=25.090909%, with only Spanish missing its retention cap by9 and strict retention false. The old CPU14 cohort20436/42644 closed normally with1537 acquires and matching posts, genuine wait0, joined transport, closed pipes and both credit-handle sets; no guessed4020 count or padding/reset/drop was used at the original pass boundary.
The separate scheduled188000 DEV completed the original seven ordered50-example sets. ROOT independently recomputed Turkish205/825=24.848485%, versus the accepted184k212/825=25.696970% (7 fewer errors, -0.848485 percentage points). Other word errors EN62, ES61, IT23, FR72, DE69, PT64 compare with caps69/52/30/74/70/66; only Spanish misses, by9. Strict six-language retention remains false. The periodic monitor does not advance a pass or winner; no qualified winner, early stop or TEST evaluation is claimed.
The actual post-fullpass CPU32124/39960 then closed at the188k periodic barrier with63 genuine native preparations plus20 original same-plan drain packets=83 total acquires and posts. ROOT verified actual wait0, exact old-cohort deaths, transport joins, closed pipes and both credit-handle sets, MAX2/LIVE1/final return1-to2, and preservation of207 weights/three Adam groups/four RNG/both cursors/journal/nextdraw and the SAME unconsumed plan. New CPU36416/4092 resumed that immutable plan, with the first20 positive updates independently verified under the original broad-pass2 cursor semantics: broad_draws=2631115+14*(u-187937)=14u-3, replay_updates=u, cleaner_draws=0. The original partial-tail semantics are preserved without padding, dropping or resetting samples.
Natural188000 and188200 checkpoints verified all854 named tensors/207 weights/215 Adam states, three Adam partitions, four RNG states, original semantic state and independent PRE-SAVED equality. Positive training continued after188200; ROOT's accepted current snapshot was188440/saved188400, with all five production identities and both unchanged reader identities healthy. Full401072/adminNone continues under sole session26493. The original consumed180k event remains inherited; no second GC event, capture, synchronization or worker refresh is performed.
All original science, LR, save200 cadence, native cleanup, 2250000000B cushion, full packet/queue/prefetch reservations, CPU2.5GB reserve/private8GB, MAX2/LIVE1, 16MiB bookkeeping reserve and resource floors remain unchanged. The throughput-fix request remains OPEN. This scientific-quality/normal-phase/resumed-training milestone makes no permanent throughput, fit, memory-saving or cause claim and repeats no diagnostic note.
Evidence: owner188k status {pins['owner_periodic188000_status']['sha256']}; ROOT firstpass {pins['ROOT_actual_first_fullpass187937_review']['sha256']}; ROOT188k {pins['ROOT_actual_periodic188000_review']['sha256']}.
'''
OUT.mkdir()
with(OUT/'milestone_note.txt').open('x',encoding='utf-8',newline='\n')as f:f.write(note);f.flush();os.fsync(f.fileno())
write(OUT/'entry.json',{'kind':'ASR18_source37_firstpass187937_periodic188000_Backlog_entry_v1',
 'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'marker':MARKER,'marker_count_before':before,
 'inputs':pins,'task_before':pin(TASK),'one_edit_only':True,'metadata_identities_at_entry':metadata_identities})
result={'kind':'ASR18_source37_firstpass187937_periodic188000_Backlog_result_v1','marker':MARKER,
 'marker_count_before':before,'inputs':pins,'CLI_edit_invocations':0,'actual_CLI_edit_exit':None,
 'actual_refresh_exit':None,'passed':False,'error':None,'subprocesses':[],
 'refresh_interpreter':str(P/'work/training-env/Scripts/python.exe')}
def run(label,command):
 with(OUT/(label+'.stdout.log')).open('xb')as out,(OUT/(label+'.stderr.log')).open('xb')as err:
  process=subprocess.Popen(command,cwd=S,stdout=out,stderr=err)
  try:who=identity(process.pid)
  except psutil.NoSuchProcess:who={'pid':process.pid,'create_time':'UNRECORDED_UNKNOWN'}
  row={'label':label,'Popen_identity':who,'actual_Popen_wait_code':None};result['subprocesses'].append(row)
  code=process.wait(timeout=60);row['actual_Popen_wait_code']=code
  row['recorded_Popen_identity_exact_dead']=None if who['create_time']=='UNRECORDED_UNKNOWN'else not observed_alive(who)
  return code
try:
 result['CLI_edit_invocations']=1
 result['actual_CLI_edit_exit']=run('edit',[str(S/'tools/backlog.exe'),'task','edit','ASR-18','--append-notes',note])
 assert result['actual_CLI_edit_exit']==0
 assert TASK.read_text(encoding='utf-8').count(MARKER)==1
 result['actual_refresh_exit']=run('refresh',[result['refresh_interpreter'],'-Xutf8','-B',str(S/'refresh_backlog.py')])
 assert result['actual_refresh_exit']==0
 result['marker_count_after']=TASK.read_text(encoding='utf-8').count(MARKER);assert result['marker_count_after']==1
 result['passed']=True
except BaseException as error:
 result['error']={'type':type(error).__name__,'message':str(error),'ambiguous_retry_forbidden':True}
finally:
 result['marker_count_after']=TASK.read_text(encoding='utf-8').count(MARKER);result['task_after']=pin(TASK)
 result['entry']=pin(OUT/'entry.json');result['note']=pin(OUT/'milestone_note.txt');result['source']=pin(Path(__file__))
 result['logs']={p.name:pin(p)for p in sorted(OUT.glob('*.log'))}
 result['input_pins_unchanged']=all(pin(path)==pins[key]for key,(path,_)in inputs.items())
 result['metadata_identities_alive_at_write']=metadata_identities
 result['initial_numeric_sample']=pin(initial_path)
 result['no_training_reader_source_environment_LR_GC_guard_or_policy_change']=True
 result['wall_seconds']=time.monotonic()-began
 write(RESULT,result)
print(json.dumps({'result':pin(RESULT),'passed':result['passed'],'edit':result['actual_CLI_edit_exit'],
 'refresh':result['actual_refresh_exit'],'marker_before':before,'marker_after':result['marker_count_after'],
 'metadata_identities_at_write':metadata_identities,'error':result['error']}),flush=True)
sys.exit(0 if result['passed']else 1)
