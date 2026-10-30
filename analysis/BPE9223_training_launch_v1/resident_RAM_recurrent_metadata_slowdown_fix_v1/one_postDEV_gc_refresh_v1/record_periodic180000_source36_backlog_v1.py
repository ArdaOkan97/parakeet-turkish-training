"""ONE ROOT-accepted180k quality/normalphase/idle-event milestone and refresh."""
import datetime,hashlib,json,os,subprocess,sys
from pathlib import Path
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish'
F=Path(__file__).resolve().parent;N=F.parent.parent;A=N/'resident_RAM_training_actual_26'
OUT=A/'backlog_periodic180000_source36_once_idle_refresh_01';RESULT=A/'actual_periodic180000_source36_backlog_result_v1.json'
TASK=S/'backlog/tasks/asr-18 - Train-BPE-on-broad-weak-labels-then-fine-tune-with-multilingual-retention.md'
MARKER='BPE9223_DEV180000_SOURCE36_ONCE_IDLE_REFRESH_CPU13_MAX2_SAVED180200_RESUMED_20261023'
def pin(p):
 b=p.read_bytes();return {'path':str(p.resolve()),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def write(p,v):
 with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
assert Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad'))
before=TASK.read_text(encoding='utf-8').count(MARKER)
assert before==0,('Exact canonical marker already present; no duplicate or ambiguous retry',before)
assert not OUT.exists() and not RESULT.exists(),'One fresh exactly-once operation'
inputs={
 'owner_periodic180000_status':(A/'actual_periodic180000_and_resumed_training_status.json','eccd285b258d294a0d53d2031310e622b3ffd660a73d98a00964431d6386af7f'),
 'ROOT_actual_periodic180000_review':(N/'ROOT_periodic180000_full350_source36_idle_refresh_CPU13_max2_normal_saved180200_and_resumed_training_verification_v1.json','3fb24d985e3ddbd29477b712e8ffdaa535a997ce363baa48b24fa349133097e5')}
pins={}
for key,(path,digest) in inputs.items():
 pins[key]=pin(path);assert pins[key]['sha256']==digest
note=f'''{MARKER}
The scheduled 180000 DEV completed all 350 original examples in seven ordered 50-example language sets. ROOT independently recomputed Turkish 212/825 = 25.696970%, versus 176k 207/825: +5 errors/+0.606061 percentage points. Other word errors: EN61, ES62, IT24, FR72, DE71, PT64. Spanish misses its cap by 10 and German by 1; strict retention remains false. No qualified winner, full-pass advance, early stop or TEST evaluation is claimed.
The normal CPU13/helper3/MAX2 phase preserved 207 weights, three Adam groups, four RNG states, both cursors, journal/next draw and the exact unconsumed plan. All 1220 preparation acquires and release posts (1200 native packets plus 20 original same-plan drain packets) completed with actual Popen wait 0, joined transport threads, closed pipes and both credit-handle sets, final previous-count 1 return to 2 and LIVE receiver limit 1. Old CPU33268/launcher18884 exited; new CPU40004/launcher21652 resumed the same retained plan. ROOT independently verified the first 20 genuine resumed positive updates.
At the first periodic DEV opportunity only, source36 normally joined and dropped the empty scalar worker, preserved the exact queued plan, and completed the admitted idle unfreeze -> captured original builtin full collect -> freeze event. Actual before/after full-state semantic capture matched exactly; the fresh scalar worker was admitted afterward. The event charged a separate 16777216 B bookkeeping reserve with no freed CPU/DEV credit. Original native numerical/per-update GC/synchronize/empty-cache calls and startup/final GC scope remain unchanged. Opaque-cycle retention exclusion, future fit, memory savings, prevention and a sole GC-cause claim are not proved.
Natural 180000 and 180200 checkpoints verified all 854 named tensors/207 weights/215 Adam states, semantic state and independent PRE-SAVED equality; positive training resumed beyond 180200. ROOT's accepted snapshot was 180460/saved180400, with native21912, new CPU40004, unchanged healthy reader and exclusive session42997 continuing full401072/adminNone. Original science, save200, 2 GB commit/1.5 GB physical floors, the 2250000000 B cushion, full packet/queue/prefetch reservations and MAX2/LIVE1 remain exact. The original metadata observer's numeric prelaunch sample is UNRECORDED; its successful source-bound full cushion plus prospective 512 MiB threshold checks are preserved, without a reconstructed sample or rerun.
The recurring speed-fix request remains OPEN pending the frozen 178820->181020 interval: all 2200 updates, 11 natural saves and real 180k DEV+1020. This milestone makes no warm-window or durable speed-success claim. No extra DEV, study or older repair/quality note is included.
Evidence: owner status {pins['owner_periodic180000_status']['sha256']}; ROOT accepted 180k {pins['ROOT_actual_periodic180000_review']['sha256']}.
'''
OUT.mkdir();(OUT/'milestone_note.txt').write_text(note,encoding='utf-8',newline='\n')
write(OUT/'entry.json',{'kind':'ASR18_periodic180000_source36_once_idle_refresh_Backlog_entry_v1','UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'marker':MARKER,'marker_count_before':before,'inputs':pins,'task_before':pin(TASK),'one_edit_only':True})
result={'kind':'ASR18_periodic180000_source36_once_idle_refresh_Backlog_result_v1','marker':MARKER,'marker_count_before':before,'inputs':pins,'CLI_edit_invocations':0,'actual_CLI_edit_exit':None,'actual_refresh_exit':None,'passed':False,'error':None,'refresh_interpreter':str(P/'work/training-env/Scripts/python.exe')}
try:
 result['CLI_edit_invocations']=1
 with (OUT/'edit.stdout.log').open('xb') as out,(OUT/'edit.stderr.log').open('xb') as err:
  done=subprocess.run([str(S/'tools/backlog.exe'),'task','edit','ASR-18','--append-notes',note],cwd=S,stdout=out,stderr=err,timeout=60,check=False)
 result['actual_CLI_edit_exit']=done.returncode;assert done.returncode==0
 assert TASK.read_text(encoding='utf-8').count(MARKER)==1
 with (OUT/'refresh.stdout.log').open('xb') as out,(OUT/'refresh.stderr.log').open('xb') as err:
  done=subprocess.run([result['refresh_interpreter'],'-Xutf8','-B',str(S/'refresh_backlog.py')],cwd=S,stdout=out,stderr=err,timeout=60,check=False)
 result['actual_refresh_exit']=done.returncode;assert done.returncode==0
 result['marker_count_after']=TASK.read_text(encoding='utf-8').count(MARKER);assert result['marker_count_after']==1
 result['passed']=True
except BaseException as error:
 result['error']={'type':type(error).__name__,'message':str(error),'ambiguous_retry_forbidden':True}
finally:
 result['marker_count_after']=TASK.read_text(encoding='utf-8').count(MARKER);result['task_after']=pin(TASK)
 result['entry']=pin(OUT/'entry.json');result['note']=pin(OUT/'milestone_note.txt');result['source']=pin(Path(__file__))
 result['logs']={p.name:pin(p) for p in sorted(OUT.glob('*.log'))};result['input_pins_unchanged']=all(pin(path)==pins[key] for key,(path,digest) in inputs.items())
 result['no_training_reader_source_environment_or_policy_change']=True;write(RESULT,result)
print(json.dumps({'result':pin(RESULT),'passed':result['passed'],'edit':result['actual_CLI_edit_exit'],'refresh':result['actual_refresh_exit'],'marker_before':before,'marker_after':result['marker_count_after'],'error':result['error']}),flush=True)
sys.exit(0 if result['passed'] else 1)
