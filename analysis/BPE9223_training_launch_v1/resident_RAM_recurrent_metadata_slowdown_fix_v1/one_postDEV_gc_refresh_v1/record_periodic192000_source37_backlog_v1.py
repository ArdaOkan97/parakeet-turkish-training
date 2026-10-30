"""ONE accepted scheduled192k scientific milestone and normal Backlog refresh."""
import datetime,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
import psutil
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish'
sys.path.insert(0,str(S))
from resource_watchdog_v2 import sample_memory
F=Path(__file__).resolve().parent;A=F.parent.parent/'resident_RAM_training_actual_27'
OUT=A/'backlog_periodic192000_source37_01'
RESULT=A/'actual_periodic192000_source37_backlog_result_v1.json'
TASK=S/'backlog/tasks/asr-18 - Train-BPE-on-broad-weak-labels-then-fine-tune-with-multilingual-retention.md'
MARKER='BPE9223_DEV192000_SOURCE37_CPU14_MAX2_SAVED192200_RESUMED_20261024'
began=time.monotonic();psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
def pin(p):
 p=Path(p);raw=p.read_bytes();return {'path':str(p.resolve()),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def write(p,v):
 with Path(p).open('x',encoding='utf-8',newline='\n')as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def ident(pid):
 process=psutil.Process(pid);return {'pid':pid,'create_time':process.create_time(),'ppid':process.ppid()}
def alive(who):
 try:return abs(psutil.Process(who['pid']).create_time()-who['create_time'])<.001
 except psutil.NoSuchProcess:return False
assert Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad'))
assert not OUT.exists()and not RESULT.exists(),'Fresh exactly-once operation; ambiguous retry forbidden'
before=TASK.read_text(encoding='utf-8').count(MARKER);assert before==0,('Canonical marker already exists',before)
metadata_ids=[ident(os.getpid()),ident(os.getppid()),ident(psutil.Process(os.getppid()).ppid())]
sample=sample_memory(os.getpid(),'periodic192000_Backlog_initial')
initial=A/'actual_periodic192000_backlog_initial_numeric_before_predicate_v1.json'
write(initial,{'sample':sample,'required_commit':5055306368,'required_physical':2305306368,
 'persisted_before_predicate_and_mutation':True,'metadata_identities':metadata_ids})
assert sample['paired_sampling_within_one_second']and sample['worker_private_memory']['private_bytes']<536870912
assert sample['system_memory']['commit_available_bytes']>=5055306368 and sample['system_memory']['physical_available_bytes']>=2305306368
assert shutil.disk_usage('E:/').free>=100000000000
inputs={
 'owner_periodic192000_status':(A/'actual_periodic192000_and_resumed_training_status.json','50c35d775ee1b22bca0538230ce94d4675c444c6df61d8cf7092520913c4c99f'),
 'ROOT_periodic192000_review':(F/'ROOT_periodic192000_full350_source37_CPU14_max2_normal_saved192200_and_resumed_training_verification_v1.json','84de69e1940f77a24fae9c3ce4dd24b601ec2844f3a14b8406bb5661c44bb864'),
 'ROOT_periodic192000_direct20_review':(F/'ROOT_periodic192000_direct20_sameplan_broad2_resumed_positive_verification_v1.json','a524dc835b058150d989169750659c53ab85780aead045eed5bfba481f610d96')}
pins={}
for key,(path,digest)in inputs.items():pins[key]=pin(path);assert pins[key]['sha256']==digest,(key,pins[key])
note=f'''{MARKER}
The original scheduled192000 DEV completed all350 clean examples in seven ordered50-example language sets. ROOT independently recomputed Turkish204/825=24.7272727%, versus188k205/825=24.8484848%: one fewer word error and -0.121212 percentage points. Other word errors EN62, ES62, IT23, FR75, DE71, PT64 compare with caps69/52/30/74/70/66; Spanish misses by10, French by1 and German by1. Strict six-language retention remains false. No qualified winner, periodic full-pass advancement, early stop or TEST evaluation is claimed.
The actual normal CPU14/helper3/MAX2 periodic phase covered4000 native preparations from the accepted188k cohort plus the ACTUAL20 original unconsumed same-plan drain packets=4020 acquires and matching release posts. Old CPU36416/launcher4092 completed genuine wait0, transport joins, closed pipes and both child/parent credit-handle sets with zero remaining handles, final previous-count1 return to full2 and no cancellation. All207 weights/three Adam groups/four RNG states/both cursors/journal/nextdraw and the SAME immutable queued plan were preserved. New CPU38116/launcher20996 resumed that plan; ROOT independently verified the first20 genuine positive updates. Original broad-pass2 counters remain broad_draws=2631115+14*(u-187937)=14u-3, replay_updates=u and cleaner_draws=0, without padding, dropping or resetting samples.
Natural192000 and192200 checkpoints verified all854 named tensors/207 weights/215 Adam states, three Adam partitions, four RNG states, original semantic state and independent canonical PRE-SAVED equality. Actual positive training continued beyond192200; ROOT's accepted current snapshot was192680/saved192600, with all five production identities and both unchanged reader identities healthy. Full401072/adminNone continues under sole session26493. The original180k once-only event remains inherited as CONSUMED, with no second GC, capture, synchronization or worker refresh.
Original science, LR1e-5/1e-4/3e-5, save200 cadence, native cleanup,2250000000B cushion, FULL packet/queue/prefetch reservations, CPU2.5GB reserve/private8GB, MAX2/LIVE1,16MiB event bookkeeping reserve and original resource floors remain unchanged. The throughput-fix request remains OPEN. This scientific-quality/normal-phase/resumed-training milestone makes no performance, fit, memory-saving, durable-fix or cause claim and repeats no diagnostic note.
Evidence: owner status {pins['owner_periodic192000_status']['sha256']}; ROOT quality {pins['ROOT_periodic192000_review']['sha256']}; ROOT direct20 {pins['ROOT_periodic192000_direct20_review']['sha256']}.
'''
OUT.mkdir()
with(OUT/'milestone_note.txt').open('x',encoding='utf-8',newline='\n')as f:f.write(note);f.flush();os.fsync(f.fileno())
write(OUT/'entry.json',{'kind':'ASR18_periodic192000_source37_Backlog_entry_v1','UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'marker':MARKER,'marker_count_before':before,'inputs':pins,'task_before':pin(TASK),'one_edit_only':True,'metadata_identities':metadata_ids})
result={'kind':'ASR18_periodic192000_source37_Backlog_result_v1','marker':MARKER,'marker_count_before':before,'inputs':pins,
 'CLI_edit_invocations':0,'actual_CLI_edit_exit':None,'actual_refresh_exit':None,'passed':False,'error':None,'subprocesses':[],
 'refresh_interpreter':str(P/'work/training-env/Scripts/python.exe')}
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
 result['actual_refresh_exit']=run('refresh',[result['refresh_interpreter'],'-Xutf8','-B',str(S/'refresh_backlog.py')]);assert result['actual_refresh_exit']==0
 result['marker_count_after']=TASK.read_text(encoding='utf-8').count(MARKER);assert result['marker_count_after']==1
 result['passed']=True
except BaseException as error:result['error']={'type':type(error).__name__,'message':str(error),'ambiguous_retry_forbidden':True}
finally:
 result['marker_count_after']=TASK.read_text(encoding='utf-8').count(MARKER);result['task_after']=pin(TASK)
 result['entry']=pin(OUT/'entry.json');result['note']=pin(OUT/'milestone_note.txt');result['source']=pin(Path(__file__))
 result['logs']={path.name:pin(path)for path in sorted(OUT.glob('*.log'))}
 result['input_pins_unchanged']=all(pin(path)==pins[key]for key,(path,_)in inputs.items())
 result['metadata_identities_alive_at_write']=metadata_ids;result['initial_numeric_sample']=pin(initial)
 result['no_training_reader_source_environment_LR_GC_guard_or_policy_change']=True;result['wall_seconds']=time.monotonic()-began
 write(RESULT,result)
print(json.dumps({'result':pin(RESULT),'passed':result['passed'],'edit':result['actual_CLI_edit_exit'],'refresh':result['actual_refresh_exit'],
 'marker_before':before,'marker_after':result['marker_count_after'],'metadata_identities_at_write':metadata_ids,'error':result['error']}),flush=True)
sys.exit(0 if result['passed']else 1)
