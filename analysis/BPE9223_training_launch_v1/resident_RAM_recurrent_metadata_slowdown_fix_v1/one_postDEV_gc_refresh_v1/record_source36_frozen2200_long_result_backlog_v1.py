"""Exactly one accepted long-result milestone; no training action or new study."""
import datetime,hashlib,json,os,subprocess,sys
from pathlib import Path
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish'
F=Path(__file__).resolve().parent;N=F.parent.parent;A=N/'resident_RAM_training_actual_26'
OUT=A/'backlog_source36_frozen2200_181020_01';RESULT=A/'actual_source36_frozen2200_181020_backlog_result_v1.json'
TASK=S/'backlog/tasks/asr-18 - Train-BPE-on-broad-weak-labels-then-fine-tune-with-multilingual-retention.md'
MARKER='BPE9223_SOURCE36_FROZEN2200_181020_RATE76_DRIFT8P12_FIX_OPEN_20261023'
def pin(p):
 b=p.read_bytes();return {'path':str(p.resolve()),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def write(p,v):
 with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
assert Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad'))
before=TASK.read_text(encoding='utf-8').count(MARKER);assert before==0,('Canonical marker already exists; no duplicate or ambiguous retry',before)
assert not OUT.exists() and not RESULT.exists(),'One fresh milestone operation'
inputs={
 'owner_frozen2200_actual_report':(F/'actual_source36_dynamic_long_validation_01/report.json','ce347891d2eb8b21fb387c5a9e846291c985341ed1a05e38814a8482b4bbb867'),
 'ROOT_actual_long_result_review':(F/'ROOT_source36_frozen2200_178820_181020_actual_long_result_verification_v1.json','eaca93edb0392d0877b15717d2f44b06e493f38326a23cc8afa8b884908ab0a7')}
pins={}
for key,(path,digest) in inputs.items():
 pins[key]=pin(path);assert pins[key]['sha256']==digest
note=f'''{MARKER}
The predeclared source36 interval 178820->181020 completed all 2200 updates, all 110 original contiguous timing rows, all 12 emitted 200-update endpoints, all 11 natural saves and the scheduled 180000 DEV plus 1020 resumed updates. ROOT independently recomputed the same complete interval. Whole throughput was 76.172885 audio minutes per wall minute, including evaluation and saves.
First 400 throughput was 81.905326 including saves (85.349788 with recorded save time excluded); final 400 was 76.196912 including saves (78.421702 derived without save time). The 8.117286% save-excluded decline exceeds the predeclared 5% limit, so durable speed-fix acceptance is FALSE and the human speed request remains OPEN. All comparable non-DEV 400-update rates exceeded the 75 engineering floor. The two fixed post-DEV 400-update blocks improved from 75.295300 to 76.196912 including saves (77.532836 to 78.421702 excluding recorded save time); this is not a renewed post-DEV collapse or a proven pure-GC cause.
Original collect call-wall sums fell from 85.136753 to 11.433421 seconds per first/final 400, while native+commit fell from 548.075 to 503.194 seconds. Endpoint wall outside native+commit grew from 70.628 to 160.510 seconds. Collect and scalar-worker wall can overlap other work; these are timing scopes, not an exclusive CPU or causal decomposition. No extra profiler, GC action, measurement span, LR change, restart or tuning was selected.
Natural 181000 remains fully verified: 854 named tensors, 207 weights, 215 Adam states, semantic state and independent PRE-SAVED equality. ROOT's accepted current snapshot was 181200/saved181000, with the same five production identities and healthy unchanged reader. Exclusive session42997/source36 continues toward full401072/adminNone with original science, save200, 2250000000 B cushion, original floors, full reservations and MAX2/LIVE1 intact. This milestone records a completed measurement and partial gain, not completion of the speed fix, future-fit proof or guaranteed future throughput.
Evidence: owner actual report {pins['owner_frozen2200_actual_report']['sha256']}; ROOT accepted long result {pins['ROOT_actual_long_result_review']['sha256']}.
'''
OUT.mkdir();(OUT/'milestone_note.txt').write_text(note,encoding='utf-8',newline='\n')
write(OUT/'entry.json',{'kind':'ASR18_source36_frozen2200_Backlog_entry_v1','UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'marker':MARKER,'marker_count_before':before,'inputs':pins,'task_before':pin(TASK),'one_edit_only':True})
result={'kind':'ASR18_source36_frozen2200_Backlog_result_v1','marker':MARKER,'marker_count_before':before,'inputs':pins,'CLI_edit_invocations':0,'actual_CLI_edit_exit':None,'actual_refresh_exit':None,'passed':False,'error':None,'refresh_interpreter':str(P/'work/training-env/Scripts/python.exe')}
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
