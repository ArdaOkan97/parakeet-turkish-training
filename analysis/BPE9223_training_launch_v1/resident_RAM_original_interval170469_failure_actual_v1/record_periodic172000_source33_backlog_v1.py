"""ONE ROOT-accepted172k quality/normalphase milestone and normal refresh."""
import datetime,hashlib,json,os,subprocess,sys
from pathlib import Path
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish'
F=Path(__file__).resolve().parent;N=F.parent;A=N/'resident_RAM_training_actual_25'
OUT=A/'backlog_periodic172000_CPU13_normal_01';RESULT=A/'actual_periodic172000_CPU13_normal_backlog_result.json'
TASK=S/'backlog/tasks/asr-18 - Train-BPE-on-broad-weak-labels-then-fine-tune-with-multilingual-retention.md'
MARKER='BPE9223_DEV172000_CPU13_MAX2_NORMAL_SAVED172200_RESUMED_20261023'
def pin(p):
 b=p.read_bytes();return {'path':str(p.resolve()),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def write(p,v):
 with p.open('x',encoding='utf-8',newline='\n')as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
assert Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad'))
before=TASK.read_text(encoding='utf-8').count(MARKER)
assert before==0,('Exact canonical marker already present; no duplicate or ambiguous retry',before)
assert not OUT.exists()and not RESULT.exists(),'One fresh exactly-once operation'
inputs={
 'owner_periodic172000_status':(A/'actual_periodic172000_and_resumed_training_status.json','d60fcab2ee20008fb3be0476ed5c74c364d83b83556991fda570fb0c05827654'),
 'ROOT_actual_periodic172000_review':(N/'ROOT_periodic172000_full350_CPU13_max2_normal_quiescence_saved172200_and_resumed_training_verification_v1.json','ab201acbf657f4bf0c29b646cf883c7ac722019d0bc76e1597b3587983bec54b')}
pins={}
for key,(path,digest)in inputs.items():
 pins[key]=pin(path);assert pins[key]['sha256']==digest
note=f'''{MARKER}
The real scheduled172000 DEV completed all350 original examples (seven ordered50-example language sets). Independently ROOT-recomputed Turkish word errors are204/825 =24.727273%, versus168k202/825: +2 errors/+0.242424 percentage points. Other word errors: EN60, ES62, IT24, FR75, DE71, PT62. Spanish misses its retention cap by10, French by1 and German by1; strict retention remains false. No qualified winner, full-pass advance, early stop or TEST evaluation is claimed.
The genuine normal CPU13/helper3/MAX2 phase preserved all207 weights, three Adam groups, four RNG states, both cursors, journal/nextdraw and the exact unconsumed plan. All1620 preparation acquires and release posts (1600 native packets plus20 original same-plan drain packets) completed with actual CPU Popen wait0, joined transport threads, closed pipes and both kernel-handle sets, final previous-count1 return to2, and LIVE receiver limit1. The old CPU cohort exited; new CPU31568/launcher8608 resumed the retained same plan. ROOT independently checked the first20 genuine resumed positive updates without changing training.
Natural172000 and172200 standard checkpoints both verified all854 named tensors/207 weights/215 Adam states, semantic state and independent PRE-SAVED equality. Positive training continued past172200; the same native3652, healthy unchanged reader and sole-owned65067 continue toward full401072 with no administrative stop. Original science,200-update saving,2250000000B extra cushion, original2GB commit/1.5GB physical floors, full packet/queue/prefetch reservations and MAX2/LIVE1 remain unchanged.
The recurring slowdown request remains OPEN; this quality milestone makes no speed-fix, future-fit or prevention claim. No extra DEV/inference/export or old repair/quality note is included.
Evidence: owner status {pins['owner_periodic172000_status']['sha256']}; ROOT accepted172k {pins['ROOT_actual_periodic172000_review']['sha256']}.
'''
OUT.mkdir();(OUT/'milestone_note.txt').write_text(note,encoding='utf-8',newline='\n')
write(OUT/'entry.json',{'kind':'ASR18_periodic172000_CPU13_normal_Backlog_entry_v1',
 'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'marker':MARKER,'marker_count_before':before,
 'inputs':pins,'task_before':pin(TASK),'one_edit_only':True})
result={'kind':'ASR18_periodic172000_CPU13_normal_Backlog_result_v1','marker':MARKER,'marker_count_before':before,
 'inputs':pins,'CLI_edit_invocations':0,'actual_CLI_edit_exit':None,'actual_refresh_exit':None,'passed':False,
 'error':None,'refresh_interpreter':str(P/'work/training-env/Scripts/python.exe')}
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
 result['entry']=pin(OUT/'entry.json');result['note']=pin(OUT/'milestone_note.txt')
 result['logs']={p.name:pin(p)for p in sorted(OUT.glob('*.log'))}
 result['input_pins_unchanged']=all(pin(path)==pins[key]for key,(path,digest)in inputs.items())
 result['no_training_reader_source_environment_or_policy_change']=True;write(RESULT,result)
print(json.dumps({'result':pin(RESULT),'passed':result['passed'],'edit':result['actual_CLI_edit_exit'],
 'refresh':result['actual_refresh_exit'],'marker_before':before,'marker_after':result['marker_count_after'],'error':result['error']}),flush=True)
sys.exit(0 if result['passed']else 1)
