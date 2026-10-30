"""One exactly-once accepted audio failure/repair/recovery milestone and refresh."""
import datetime,hashlib,json,os,subprocess,sys
from pathlib import Path
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish'
F=Path(__file__).resolve().parent;A=F.parent/'resident_RAM_training_actual_25'
OUT=A/'backlog_eighth_saved170400_recovered170600_01';RESULT=A/'actual_eighth_saved170400_recovered170600_backlog_result.json'
TASK=S/'backlog/tasks/asr-18 - Train-BPE-on-broad-weak-labels-then-fine-tune-with-multilingual-retention.md'
MARKER='BPE9223_EIGHTH_FULLCLIP_SAVED170400_RECOVERED170600_20261023'
def pin(p):
 b=p.read_bytes();return {'path':str(p.resolve()),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def write(p,v):
 with p.open('x',encoding='utf-8',newline='\n')as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
inputs={
 'owner_recovery_status':(A/'actual_saved170600_recovery_01/actual_recovery_status.json','48584c607f137f1d8be25c64c67d0920bb181796ec574fd912c74aba95c4f63d'),
 'ROOT_actual_recovery_review':(F/'ROOT_source33_eighth_original_saved170400_recovered170600_actual_verification_v1.json','cf4e7d83d691a4b698f1739a033869a1896439e28f35a5a9a404612c97484f62')}
pins={}
for key,(path,digest)in inputs.items():
 pins[key]=pin(path);assert pins[key]['sha256']==digest,(key,pins[key])
assert Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad'))
assert not OUT.exists()and not RESULT.exists(),'One fresh exactly-once operation; no ambiguous rerun'
before=TASK.read_text(encoding='utf-8').count(MARKER);assert before==0,('Exact canonical marker already present',before)
note=f'''{MARKER}
Actual source31 training stopped at170469 after the original CPU interval decoder failed; the last fully saved checkpoint was170400. The69 later positive journal events remain historical and were preserved. Separate native/CPU/wrapper numeric OS exits remain UNKNOWN; they are not inferred from outer TOOL1. The original Monitor was joined and clean.
The eighth correction is bound to the one exact original row/member/annotation: all820872 original physical native samples are used once, yielding273624 clean samples, with all99 target IDs and encoder T214 unchanged. The larger historical820896 native/273632 clean capacities and every original raw/media/frontend reservation remain charged. The failed alias8 proof remains failed; the separately reviewed corrected alias9/CPU13 passed genuine original ingress/frontend, finite features/mask and strict binding negatives.
Source33 genuinely restored full170400 model/three Adam groups/four RNG/both samplers/nextdraw BEFORE native work. Only the saved prefix was copied AFTER restore with zero native updates/draws. All69 historical positives were genuinely redone, and the new170470 repaired-row update committed. Natural170600 full854 named tensors/207 weights/215 Adam states passed semantic and independent PRE-SAVED equality checks, followed by positive training and a healthy caught-up unchanged reader. ROOT independently accepted this audio recovery.
Same sole-owned65067/source33 training continues toward full401072 with no administrative update stop. Original2GB commit/1.5GB physical floors,2250000000B extra cushion, FULL packet/queue/prefetch reservations, CPU2.5GB reserve/private8GB, MAX2/LIVE1, science and200-update save cadence remain unchanged. This file-specific repair makes no future-fit, memory-saving, prevention or speed guarantee. The recurring slowdown fix remains OPEN and unresolved; no old quality or speed success is repeated here.
Evidence: owner status {pins['owner_recovery_status']['sha256']}; ROOT accepted recovery {pins['ROOT_actual_recovery_review']['sha256']}.
'''
OUT.mkdir();(OUT/'milestone_note.txt').write_text(note,encoding='utf-8',newline='\n')
write(OUT/'entry.json',{'kind':'ASR18_eighth_saved170400_recovered170600_Backlog_entry_v1',
 'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'marker':MARKER,'marker_count_before':before,
 'inputs':pins,'task_before':pin(TASK),'one_edit_only':True})
result={'kind':'ASR18_eighth_saved170400_recovered170600_Backlog_result_v1','marker':MARKER,'marker_count_before':before,
 'inputs':pins,'CLI_edit_invocations':0,'actual_CLI_edit_exit':None,'actual_refresh_exit':None,'passed':False,
 'error':None,'refresh_interpreter':str(P/'work/training-env/Scripts/python.exe')}
try:
 result['CLI_edit_invocations']=1
 with(OUT/'edit.stdout.log').open('xb')as out,(OUT/'edit.stderr.log').open('xb')as err:
  completed=subprocess.run([str(S/'tools/backlog.exe'),'task','edit','ASR-18','--append-notes',note],cwd=S,stdout=out,stderr=err,timeout=60,check=False)
 result['actual_CLI_edit_exit']=completed.returncode;assert completed.returncode==0
 assert TASK.read_text(encoding='utf-8').count(MARKER)==1
 with(OUT/'refresh.stdout.log').open('xb')as out,(OUT/'refresh.stderr.log').open('xb')as err:
  completed=subprocess.run([result['refresh_interpreter'],'-Xutf8','-B',str(S/'refresh_backlog.py')],cwd=S,stdout=out,stderr=err,timeout=60,check=False)
 result['actual_refresh_exit']=completed.returncode;assert completed.returncode==0
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
