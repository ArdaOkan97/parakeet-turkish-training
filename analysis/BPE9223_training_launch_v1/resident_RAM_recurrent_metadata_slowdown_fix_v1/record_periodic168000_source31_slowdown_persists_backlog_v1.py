import datetime,hashlib,json,os,subprocess,sys
from pathlib import Path
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish'
N=Path('E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1');F=N/'resident_RAM_recurrent_metadata_slowdown_fix_v1';A=N/'resident_RAM_training_actual_24'
OUT=A/'backlog_periodic168000_source31_slowdown_persists_01';RESULT=A/'actual_periodic168000_source31_slowdown_persists_backlog_result.json'
TASK=S/'backlog/tasks/asr-18 - Train-BPE-on-broad-weak-labels-then-fine-tune-with-multilingual-retention.md'
MARKER='BPE9223_DEV168000_FRESH_SIGNATURE_RECOVERED167600_LONG1600_SLOWDOWN_PERSISTS_20261023'
def pin(p):
 b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def write(p,v):
 with p.open('x',encoding='utf-8',newline='\n')as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
inputs={
'owner_quality_status':(A/'actual_periodic168000_and_resumed_training_status.json','12c592737f35d8e3d71555edf7339587ba390af2d576d367ccbf776500f5b621'),
'ROOT_quality_review':(N/'ROOT_periodic168000_full350_CPU11_max2_normal_quiescence_saved168200_and_resumed_training_verification_v1.json','48156f342df29683738998f393d8800112eefd67dcd75e512264bd7f0f5f8b12'),
'ROOT_recovery_review':(F/'ROOT_source31_fresh_signature_saved167400_recovered167600_actual_verification_v1.json','42a0e76f69b7133cb1876dcef16357fef4cbabdf04a592926f4c14b085baf989'),
'owner_long_measurement':(F/'actual_source31_dynamic_long_validation_01/report.json','860232d91dad741ab7415f2e74efb6aefad91fe108b28f12bda9e7a3b2ca54d0'),
'ROOT_long_measurement_review':(F/'ROOT_source31_fresh_signature_collect_actual_long1600_result_verification_v1.json','a2204eeef81d19b293ae1743173eec122d373f7b58268a4c49ccd66b90732ea4'),
'readonly_next_measurement':(F/'source31_remaining_slowdown_readonly_targeted_CPU_attribution_proposal_v1.json','f20eef359e3da0595ee82ad0631002efba24d468fa631c7c401cc3f5dc2b488a')}
pins={}
for k,(p,h)in inputs.items():
 pins[k]=pin(p);assert pins[k]['sha256']==h,(k,pins[k])
assert os.environ['TORCH_EXTENSIONS_DIR'].lower().startswith('e:')
assert not OUT.exists()and not RESULT.exists(),'Fresh exactly-once operation'
before=TASK.read_text(encoding='utf-8').count(MARKER);assert before==0,('Canonical marker already present',before)
note=f'''{MARKER}
168,000 scheduled fixed DEV completed all350 original ordered rows. Turkish202/825=24.484848%, three more errors (+0.363636 percentage points) than164k. Retention EN61/ES61/IT23/FR72/DE71/PT64; Spanish misses by9 and German by1. No qualified winner, pass advance, early stop or TEST evaluation.
Source31 genuinely restored167400, redid all36 historical positives and reached natural full167600 durability; original separate numeric OS exits remained UNKNOWN. Its real168k CPU11/MAX2 normal phase completed620 matched acquires/posts (600native+20drain), wait0, closed transport/pipes/both kernel handle sets, preserved all model/Adam/RNG/cursors/journal/nextdraw state and resumed the same plan. Full standard168200 durability was independently accepted.
The fresh-signature candidate produced an early partial gain but did NOT hold durable speed. Frozen complete167420..169020 (1600updates, all8 natural saves and real168kDEV) measured75.814181 audio min/wall min overall. First400 vs final400, each including two saves:82.30042 ->71.62423; save-time-excluded85.58212 ->73.52272, a14.091019% decline. Two predeclared postDEV400 blocks:80.06562 then71.62423 inclusive. Both durability criteria failed.
Original collect-call wall83.19036 ->114.88921seconds/400 grew, with no raised calls or clock errors; these inclusive durations can overlap and do not establish pure GC CPU or an exclusive cost decomposition. Gate2.206seconds/zero waited updates; metadata end-wait/fallback0. No durable-speed success or GC-alone cause claim.
Human-requested fix remains OPEN. Same healthy source31 training/full401072, original safeguards/science,2250000000B extra cushion, FULL packet/queue/prefetch reservations, MAX2/LIVE1 and reader/environment remain unchanged. Next targeted CPU/thread attribution is an inactive proposal pending exact ROOT scope; no process/GC/cache tuning or restart selected.
Evidence: ROOTquality {pins['ROOT_quality_review']['sha256']}; ROOTrecovery {pins['ROOT_recovery_review']['sha256']}; ROOTlongfailed-durability {pins['ROOT_long_measurement_review']['sha256']}; ownerquality {pins['owner_quality_status']['sha256']}; ownerlong {pins['owner_long_measurement']['sha256']}; readonly {pins['readonly_next_measurement']['sha256']}.
'''
OUT.mkdir();(OUT/'milestone_note.txt').write_text(note,encoding='utf-8',newline='\n')
write(OUT/'entry.json',{'kind':'ASR18_periodic168000_source31_long_slowdown_persists_Backlog_entry_v1','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'marker':MARKER,'marker_count_before':before,'inputs':pins,'task_before':pin(TASK),'one_edit_only':True})
result={'kind':'ASR18_periodic168000_source31_long_slowdown_persists_Backlog_result_v1','marker':MARKER,'marker_count_before':before,'inputs':pins,'CLI_edit_invocations':0,'actual_CLI_edit_exit':None,'actual_refresh_exit':None,'passed':False,'error':None,'refresh_interpreter':str(P/'work/training-env/Scripts/python.exe')}
try:
 result['CLI_edit_invocations']=1
 with(OUT/'edit.stdout.log').open('xb')as o,(OUT/'edit.stderr.log').open('xb')as e:r=subprocess.run([str(S/'tools/backlog.exe'),'task','edit','ASR-18','--append-notes',note],cwd=S,stdout=o,stderr=e,timeout=60,check=False)
 result['actual_CLI_edit_exit']=r.returncode;assert r.returncode==0
 assert TASK.read_text(encoding='utf-8').count(MARKER)==1
 with(OUT/'refresh.stdout.log').open('xb')as o,(OUT/'refresh.stderr.log').open('xb')as e:r=subprocess.run([result['refresh_interpreter'],'-Xutf8','-B',str(S/'refresh_backlog.py')],cwd=S,stdout=o,stderr=e,timeout=60,check=False)
 result['actual_refresh_exit']=r.returncode;assert r.returncode==0
 result['marker_count_after']=TASK.read_text(encoding='utf-8').count(MARKER);assert result['marker_count_after']==1
 result['passed']=True
except BaseException as e:result['error']={'type':type(e).__name__,'message':str(e),'ambiguous_retry_forbidden':True}
finally:
 result['marker_count_after']=TASK.read_text(encoding='utf-8').count(MARKER);result['task_after']=pin(TASK)
 result['entry']=pin(OUT/'entry.json');result['note']=pin(OUT/'milestone_note.txt');result['logs']={p.name:pin(p)for p in sorted(OUT.glob('*.log'))}
 result['input_pins_unchanged']=all(pin(p)==pins[k]for k,(p,h)in inputs.items())
 result['no_training_source_environment_reader_or_child_change']=True;write(RESULT,result)
print(json.dumps({'result':pin(RESULT),'passed':result['passed'],'edit':result['actual_CLI_edit_exit'],'refresh':result['actual_refresh_exit'],'marker_count':result['marker_count_after'],'error':result['error']}))
sys.exit(0 if result['passed']else 1)
