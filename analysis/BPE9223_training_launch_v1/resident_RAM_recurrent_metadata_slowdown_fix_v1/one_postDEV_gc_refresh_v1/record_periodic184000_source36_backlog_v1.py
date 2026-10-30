"""ONE ROOT-accepted184k scientific quality/normal-phase/resumed milestone and normal refresh."""
import datetime,hashlib,json,os,subprocess,sys
from pathlib import Path
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish'
F=Path(__file__).resolve().parent;N=F.parent.parent;A=N/'resident_RAM_training_actual_26'
OUT=A/'backlog_periodic184000_source36_01';RESULT=A/'actual_periodic184000_source36_backlog_result_v1.json'
TASK=S/'backlog/tasks/asr-18 - Train-BPE-on-broad-weak-labels-then-fine-tune-with-multilingual-retention.md'
MARKER='BPE9223_DEV184000_SOURCE36_CPU13_MAX2_SAVED184200_RESUMED_20261023'

def pin(p):
 b=p.read_bytes();return {'path':str(p.resolve()),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}

def write(p,v):
 with p.open('x',encoding='utf-8',newline='\n')as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())

assert Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad'))
before=TASK.read_text(encoding='utf-8').count(MARKER)
assert before==0,('Canonical marker already present; no duplicate or ambiguous retry',before)
assert not OUT.exists()and not RESULT.exists(),'One fresh exactly-once operation'
inputs={
 'owner_periodic184000_status':(A/'actual_periodic184000_and_resumed_training_status.json','5acb0039e9c91440872c344c85c881268314a72af4002dd59d14835a43150a4b'),
 'ROOT_actual_periodic184000_review':(N/'ROOT_periodic184000_full350_source36_CPU13_max2_normal_saved184200_and_resumed_training_verification_v1.json','b12238cc7101d82fa2b70d252761d3bbd4e966671514bfc42a9d103546428843')}
pins={}
for key,(path,digest)in inputs.items():
 pins[key]=pin(path);assert pins[key]['sha256']==digest
note=f'''{MARKER}
The scheduled 184000 DEV completed all 350 original examples in seven ordered 50-example language sets. ROOT independently recomputed Turkish 212/825 = 25.696970%, unchanged from 180k 212/825. Other word errors: EN61, ES61, IT24, FR73, DE70, PT64. Spanish misses its cap by 9; German is exactly at its passing cap of 70. Strict six-language retention remains false. No qualified winner, full-pass advance, early stop or TEST evaluation is claimed.
The normal CPU13/helper3/MAX2 phase preserved all 207 weights, three Adam groups, four RNG states, both cursors, journal/next draw and the exact unconsumed plan. All 4020 preparation acquires and release posts (4000 native packets plus 20 original same-plan drain packets) completed with actual Popen wait 0, joined transport threads, closed pipes and both credit-handle sets, final previous-count 1 return to 2 and LIVE receiver limit 1. Old CPU40004/launcher21652 exited; new CPU39856/launcher35440 resumed the unchanged queued plan261 with a fresh nonce. ROOT independently verified the first 20 genuine resumed positive updates.
Natural 184000 and 184200 checkpoints verified all 854 named tensors/207 weights/215 Adam states, three Adam groups, four RNG states, semantic state and independent PRE-SAVED equality; positive training resumed beyond 184200. ROOT's accepted snapshot was 184380/saved184200, with native21912, the new CPU cohort and unchanged healthy reader verified184200. Exclusive session42997 continues full401072/adminNone. Original science, save200, 2 GB commit/1.5 GB physical floors, the 2250000000 B cushion, full packet/queue/prefetch reservations and MAX2/LIVE1 remain exact. The actual metadata observer's original numeric prelaunch full-resource sample was persisted before its unchanged full-prospective threshold check and DEV metadata reads.
The recurring throughput-fix request remains OPEN. This milestone records scientific quality, normal phase closure, durability and resumed training; it makes no new GC-event, diagnostic attribution or permanent-speed claim.
Evidence: owner status {pins['owner_periodic184000_status']['sha256']}; ROOT accepted 184k {pins['ROOT_actual_periodic184000_review']['sha256']}.
'''
OUT.mkdir();(OUT/'milestone_note.txt').write_text(note,encoding='utf-8',newline='\n')
write(OUT/'entry.json',{'kind':'ASR18_periodic184000_source36_Backlog_entry_v1','UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'marker':MARKER,'marker_count_before':before,'inputs':pins,'task_before':pin(TASK),'one_edit_only':True})
result={'kind':'ASR18_periodic184000_source36_Backlog_result_v1','marker':MARKER,'marker_count_before':before,'inputs':pins,'CLI_edit_invocations':0,'actual_CLI_edit_exit':None,'actual_refresh_exit':None,'passed':False,'error':None,'refresh_interpreter':str(P/'work/training-env/Scripts/python.exe')}
try:
 result['CLI_edit_invocations']=1
 with(OUT/'edit.stdout.log').open('xb')as out,(OUT/'edit.stderr.log').open('xb')as err:
  done=subprocess.run([str(S/'tools/backlog.exe'),'task','edit','ASR-18','--append-notes',note],cwd=S,stdout=out,stderr=err,timeout=60,check=False)
 result['actual_CLI_edit_exit']=done.returncode;assert done.returncode==0
 assert TASK.read_text(encoding='utf-8').count(MARKER)==1
 with(OUT/'refresh.stdout.log').open('xb')as out,(OUT/'refresh.stderr.log').open('xb')as err:
  done=subprocess.run([result['refresh_interpreter'],'-Xutf8','-B',str(S/'refresh_backlog.py')],cwd=S,stdout=out,stderr=err,timeout=60,check=False)
 result['actual_refresh_exit']=done.returncode;assert done.returncode==0
 result['marker_count_after']=TASK.read_text(encoding='utf-8').count(MARKER);assert result['marker_count_after']==1
 result['passed']=True
except BaseException as error:
 result['error']={'type':type(error).__name__,'message':str(error),'ambiguous_retry_forbidden':True}
finally:
 result['marker_count_after']=TASK.read_text(encoding='utf-8').count(MARKER);result['task_after']=pin(TASK)
 result['entry']=pin(OUT/'entry.json');result['note']=pin(OUT/'milestone_note.txt');result['source']=pin(Path(__file__))
 result['logs']={p.name:pin(p)for p in sorted(OUT.glob('*.log'))};result['input_pins_unchanged']=all(pin(path)==pins[key]for key,(path,digest)in inputs.items())
 result['no_training_reader_source_environment_or_policy_change']=True;write(RESULT,result)
print(json.dumps({'result':pin(RESULT),'passed':result['passed'],'edit':result['actual_CLI_edit_exit'],'refresh':result['actual_refresh_exit'],'marker_before':before,'marker_after':result['marker_count_after'],'error':result['error']}),flush=True)
sys.exit(0 if result['passed']else 1)
