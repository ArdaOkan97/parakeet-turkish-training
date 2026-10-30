"""ONE accepted196k quality milestone and established training-env normal refresh."""
import datetime,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
import psutil
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish'
sys.path.insert(0,str(S))
from resource_watchdog_v2 import sample_memory
F=Path(__file__).resolve().parent;A=F.parent.parent/'resident_RAM_training_actual_27'
OUT=A/'backlog_periodic196000_source37_01';RESULT=A/'actual_periodic196000_source37_backlog_result_v1.json'
TASK=S/'backlog/tasks/asr-18 - Train-BPE-on-broad-weak-labels-then-fine-tune-with-multilingual-retention.md'
MARKER='BPE9223_DEV196000_SOURCE37_CPU14_MAX2_SAVED196200_RESUMED_20261024'
TRAINING_PY=P/'work/training-env/Scripts/python.exe'
began=time.monotonic();psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
def pin(p):
 p=Path(p);raw=p.read_bytes();return {'path':str(p.resolve()),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def write(p,v):
 with Path(p).open('x',encoding='utf-8',newline='\n')as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def ident(pid):
 p=psutil.Process(pid);return {'pid':pid,'create_time':p.create_time(),'ppid':p.ppid()}
def alive(who):
 try:return abs(psutil.Process(who['pid']).create_time()-who['create_time'])<.001
 except psutil.NoSuchProcess:return False
assert Path(sys.executable).resolve()==TRAINING_PY.resolve(),'Use only established training-env interpreter'
assert Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad'))
assert not OUT.exists()and not RESULT.exists(),'One fresh operation; ambiguous retry forbidden'
before=TASK.read_text(encoding='utf-8').count(MARKER);assert before==0,('Canonical marker already exists',before)
ids=[ident(os.getpid()),ident(os.getppid()),ident(psutil.Process(os.getppid()).ppid())]
sample=sample_memory(os.getpid(),'periodic196000_Backlog_initial')
initial=A/'actual_periodic196000_backlog_initial_numeric_before_predicate_v1.json'
write(initial,{'sample':sample,'required_commit':5055306368,'required_physical':2305306368,'persisted_before_predicate_and_mutation':True,'metadata_identities':ids})
assert sample['paired_sampling_within_one_second']and sample['worker_private_memory']['private_bytes']<536870912
assert sample['system_memory']['commit_available_bytes']>=5055306368 and sample['system_memory']['physical_available_bytes']>=2305306368
assert shutil.disk_usage('E:/').free>=100000000000
review=F/'ROOT_periodic196000_full350_source37_CPU14_max2_normal_saved196200_and_resumed_training_verification_v1.json'
review_pin=pin(review);assert review_pin['bytes']==12123 and review_pin['sha256']=='eb761afb9d34e2b7df8b29f39215dfb730bfdea65dbdc8d8b1e0dbc6dd555f38'
inputs={'ROOT_periodic196000_accepted_quality':review_pin}
note=f'''{MARKER}
The original scheduled196000 DEV completed all350 clean examples in seven ordered50-example language sets. ROOT independently recomputed Turkish196/825=23.7575758%, eight fewer word errors and -0.96969697 percentage points versus192k204/825. Other word errors EN62, ES62, IT24, FR73, DE71, PT62 compare with caps69/52/30/74/70/66; Spanish misses by10 and German by1. Strict six-language retention remains false. No qualified winner, periodic full-pass advancement, early stop or TEST evaluation is claimed.
Actual CPU14/helper3/MAX2 normal closure covered4000 native preparations plus the actual20 same-plan drain packets=4020 acquires and matching posts, genuine wait0, transport joins, closed pipes and both owned credit-handle sets with zero remaining handles, final previous1-to-full2 return/no cancellation. The SAME immutable queued plan and all207 weights/three Adam groups/four RNG states/both cursors/journal/nextdraw were preserved. Current CPU13472/launcher39752 resumed positive training. ROOT's separate ONLY-new20 proof accepted the exact same-plan tokens, positive updates and original broad2 counters14u-3/replayu/cleaner0.
Both natural196000 and196200 checkpoints passed full854 named tensors/207 weights/215 Adam states, three Adam partitions, four RNG states, original semantic state and independent canonical PRE-SAVED equality. ROOT accepted actual196400/saved196200 with the same seven production/reader identities healthy. Full401072/adminNone continues under sole session26493. The original180k once-only event remains inherited CONSUMED with no second GC/capture/sync/worker refresh. Source37 science, LR1e-5/1e-4/3e-5, save200, native cleanup,2250000000B cushion, full packet/queue/prefetch reservations, CPU2.5GB reserve/private8GB, MAX2/LIVE1,16MiB event reserve and all resource guards remain unchanged.
The throughput-fix request remains OPEN. This scientific quality/resumed-training milestone makes no speed, durable-fix, fit, memory-saving or cause claim and repeats no diagnostic note.
Evidence: ROOT accepted196k {review_pin['sha256']}; closed directROOT new20 receipt SHA44f3e53374231283a79c84c7bd50f2805351e5b99402300430687c557685dec2. No closed actual evidence is re-audited here.
'''
OUT.mkdir()
with(OUT/'milestone_note.txt').open('x',encoding='utf-8',newline='\n')as f:f.write(note);f.flush();os.fsync(f.fileno())
write(OUT/'entry.json',{'kind':'ASR18_periodic196000_source37_Backlog_entry_v1','UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'marker':MARKER,'marker_count_before':before,'inputs':inputs,'task_before':pin(TASK),'one_edit_only':True,'metadata_identities':ids})
result={'kind':'ASR18_periodic196000_source37_Backlog_result_v1','marker':MARKER,'marker_count_before':before,'inputs':inputs,
 'CLI_edit_invocations':0,'actual_CLI_edit_exit':None,'actual_refresh_exit':None,'passed':False,'error':None,'subprocesses':[],
 'job_interpreter':str(TRAINING_PY),'refresh_interpreter':str(TRAINING_PY)}
def run(label,command):
 with(OUT/(label+'.stdout.log')).open('xb')as out,(OUT/(label+'.stderr.log')).open('xb')as err:
  process=subprocess.Popen(command,cwd=S,stdout=out,stderr=err)
  try:who=ident(process.pid)
  except psutil.NoSuchProcess:who={'pid':process.pid,'create_time':'UNRECORDED_UNKNOWN'}
  row={'label':label,'Popen_identity':who,'actual_Popen_wait_code':None};result['subprocesses'].append(row)
  code=process.wait(timeout=60);row['actual_Popen_wait_code']=code
  row['recorded_Popen_identity_exact_dead']=None if who['create_time']=='UNRECORDED_UNKNOWN'else not alive(who)
  return code
try:
 result['CLI_edit_invocations']=1
 result['actual_CLI_edit_exit']=run('edit',[str(S/'tools/backlog.exe'),'task','edit','ASR-18','--append-notes',note]);assert result['actual_CLI_edit_exit']==0
 assert TASK.read_text(encoding='utf-8').count(MARKER)==1
 result['actual_refresh_exit']=run('refresh',[str(TRAINING_PY),'-Xutf8','-B',str(S/'refresh_backlog.py')]);assert result['actual_refresh_exit']==0
 result['marker_count_after']=TASK.read_text(encoding='utf-8').count(MARKER);assert result['marker_count_after']==1
 result['passed']=True
except BaseException as error:result['error']={'type':type(error).__name__,'message':str(error),'ambiguous_retry_forbidden':True}
finally:
 result['marker_count_after']=TASK.read_text(encoding='utf-8').count(MARKER);result['task_after']=pin(TASK)
 result['entry']=pin(OUT/'entry.json');result['note']=pin(OUT/'milestone_note.txt');result['source']=pin(Path(__file__))
 result['logs']={path.name:pin(path)for path in sorted(OUT.glob('*.log'))};result['ROOT_input_pin_unchanged']=pin(review)==review_pin
 result['metadata_identities_alive_at_write']=ids;result['initial_numeric_sample']=pin(initial)
 result['no_training_reader_source_environment_LR_GC_guard_or_policy_change']=True;result['wall_seconds']=time.monotonic()-began
 write(RESULT,result)
print(json.dumps({'result':pin(RESULT),'passed':result['passed'],'edit':result['actual_CLI_edit_exit'],'refresh':result['actual_refresh_exit'],
 'marker_before':before,'marker_after':result['marker_count_after'],'metadata_identities_at_write':ids,'error':result['error']}),flush=True)
sys.exit(0 if result['passed']else 1)
