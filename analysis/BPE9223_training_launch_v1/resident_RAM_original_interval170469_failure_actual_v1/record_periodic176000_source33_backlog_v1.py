"""ONE ROOT-accepted 176k quality/normal-phase milestone and normal refresh."""
import datetime, hashlib, json, os, subprocess, sys
from pathlib import Path
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20'); S=P/'outputs/parakeet_turkish'
F=Path(__file__).resolve().parent; N=F.parent; A=N/'resident_RAM_training_actual_25'
OUT=A/'backlog_periodic176000_CPU13_normal_01'; RESULT=A/'actual_periodic176000_CPU13_normal_backlog_result.json'
TASK=S/'backlog/tasks/asr-18 - Train-BPE-on-broad-weak-labels-then-fine-tune-with-multilingual-retention.md'
MARKER='BPE9223_DEV176000_CPU13_MAX2_NORMAL_SAVED176200_RESUMED_20261023'
def pin(p):
 b=p.read_bytes(); return {'path':str(p.resolve()),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def write(p,v):
 with p.open('x',encoding='utf-8',newline='\n') as f: json.dump(v,f,ensure_ascii=False,indent=2); f.write('\n')
assert Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad'))
before=TASK.read_text(encoding='utf-8').count(MARKER)
assert before==0,('Exact canonical marker already present; no duplicate or ambiguous retry',before)
assert not OUT.exists() and not RESULT.exists(),'One fresh exactly-once operation'
inputs={
 'owner_periodic176000_status':(A/'actual_periodic176000_and_resumed_training_status.json','1c45ed7199cb59bca80388115139923b9a283e0b3781318a4f597008f9cb4d87'),
 'ROOT_actual_periodic176000_review':(N/'ROOT_periodic176000_full350_CPU13_max2_normal_quiescence_saved176200_and_resumed_training_verification_v1.json','457d6e51b52c2b611cf910ec012f3ac24ccf4718668285bb9058cc92df8c7fdd')}
pins={}
for key,(path,digest) in inputs.items():
 pins[key]=pin(path); assert pins[key]['sha256']==digest
note=f'''{MARKER}
The real scheduled 176000 DEV completed all 350 original examples (seven ordered 50-example language sets). Independently ROOT-recomputed Turkish word errors are 207/825 = 25.090909%, versus 172k 204/825: +3 errors/+0.363636 percentage points. Other word errors: EN61, ES61, IT24, FR71, DE70, PT63. Only Spanish misses its retention cap, by 9 errors; German is exactly at its cap and passes. Strict retention remains false. No qualified winner, full-pass advance, early stop or TEST evaluation is claimed.
The genuine normal CPU13/helper3/MAX2 phase preserved all 207 weights, three Adam groups, four RNG states, both cursors, journal/next draw and the exact unconsumed plan. All 4020 preparation acquires and release posts (4000 native packets plus 20 original same-plan drain packets) completed with actual CPU Popen wait 0, joined transport threads, closed pipes and both kernel-handle sets, final previous-count 1 return to 2, and LIVE receiver limit 1. Old CPU31568/launcher8608 exited; new CPU31032/launcher40160 resumed retained same plan 281. ROOT independently checked the first 20 genuine resumed positive updates without changing training.
Natural 176000 and 176200 standard checkpoints both verified all 854 named tensors/207 weights/215 Adam states, semantic state and independent PRE-SAVED equality. Positive training continued past 176200; ROOT's later accepted snapshot was 176760 with saved 176600. The same native3652, healthy unchanged reader and sole-owned 65067 continue toward full 401072 with no administrative stop. Original science, 200-update saving, 2250000000B extra cushion, original 2GB commit/1.5GB physical floors, full packet/queue/prefetch reservations and MAX2/LIVE1 remain unchanged.
The recurring slowdown request remains OPEN; this quality milestone makes no speed-fix, future-fit or prevention claim. No extra DEV/inference/export or old repair/quality note is included.
Evidence: owner status {pins['owner_periodic176000_status']['sha256']}; ROOT accepted 176k {pins['ROOT_actual_periodic176000_review']['sha256']}.
'''
OUT.mkdir(); (OUT/'milestone_note.txt').write_text(note,encoding='utf-8',newline='\n')
write(OUT/'entry.json',{'kind':'ASR18_periodic176000_CPU13_normal_Backlog_entry_v1',
 'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'marker':MARKER,'marker_count_before':before,
 'inputs':pins,'task_before':pin(TASK),'one_edit_only':True})
result={'kind':'ASR18_periodic176000_CPU13_normal_Backlog_result_v1','marker':MARKER,'marker_count_before':before,
 'inputs':pins,'CLI_edit_invocations':0,'actual_CLI_edit_exit':None,'actual_refresh_exit':None,'passed':False,
 'error':None,'refresh_interpreter':str(P/'work/training-env/Scripts/python.exe')}
try:
 result['CLI_edit_invocations']=1
 with (OUT/'edit.stdout.log').open('xb') as out,(OUT/'edit.stderr.log').open('xb') as err:
  done=subprocess.run([str(S/'tools/backlog.exe'),'task','edit','ASR-18','--append-notes',note],cwd=S,stdout=out,stderr=err,timeout=60,check=False)
 result['actual_CLI_edit_exit']=done.returncode; assert done.returncode==0
 assert TASK.read_text(encoding='utf-8').count(MARKER)==1
 with (OUT/'refresh.stdout.log').open('xb') as out,(OUT/'refresh.stderr.log').open('xb') as err:
  done=subprocess.run([result['refresh_interpreter'],'-Xutf8','-B',str(S/'refresh_backlog.py')],cwd=S,stdout=out,stderr=err,timeout=60,check=False)
 result['actual_refresh_exit']=done.returncode; assert done.returncode==0
 result['marker_count_after']=TASK.read_text(encoding='utf-8').count(MARKER); assert result['marker_count_after']==1
 result['passed']=True
except BaseException as error:
 result['error']={'type':type(error).__name__,'message':str(error),'ambiguous_retry_forbidden':True}
finally:
 result['marker_count_after']=TASK.read_text(encoding='utf-8').count(MARKER); result['task_after']=pin(TASK)
 result['entry']=pin(OUT/'entry.json'); result['note']=pin(OUT/'milestone_note.txt')
 result['logs']={p.name:pin(p) for p in sorted(OUT.glob('*.log'))}
 result['input_pins_unchanged']=all(pin(path)==pins[key] for key,(path,digest) in inputs.items())
 result['no_training_reader_source_environment_or_policy_change']=True; write(RESULT,result)
print(json.dumps({'result':pin(RESULT),'passed':result['passed'],'edit':result['actual_CLI_edit_exit'],
 'refresh':result['actual_refresh_exit'],'marker_before':before,'marker_after':result['marker_count_after'],'error':result['error']}),flush=True)
sys.exit(0 if result['passed'] else 1)
