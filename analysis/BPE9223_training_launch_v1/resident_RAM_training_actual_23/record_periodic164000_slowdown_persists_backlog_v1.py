import datetime, hashlib, json, os, subprocess, sys
from pathlib import Path

P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20')
S=P/'outputs/parakeet_turkish'
N=Path(r'E:\gpu_training_scratchpad\analysis\BPE9223_training_launch_v1')
A=N/'resident_RAM_training_actual_23'
F=N/'resident_RAM_recurrent_metadata_slowdown_fix_v1'
OUT=A/'backlog_periodic164000_slowdown_persists_01'
RESULT=A/'actual_periodic164000_slowdown_persists_backlog_result.json'
TASK=S/'backlog/tasks/asr-18 - Train-BPE-on-broad-weak-labels-then-fine-tune-with-multilingual-retention.md'
MARKER='BPE9223_DEV164000_CANONICAL_ARCHIVE_LONG1600_SLOWDOWN_PERSISTS_20261023'

def pin(p):
    b=p.read_bytes()
    return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}

def write(p,v):
    with p.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')

inputs={
 'owner_quality_status':(A/'actual_periodic164000_and_resumed_training_status.json','8461803c0085b7ce8fc6b06a455a208fe6f96969773c84f3f6b367cff8f50e95'),
 'ROOT_quality_review':(N/'ROOT_periodic164000_full350_CPU11_max2_normal_quiescence_saved164200_and_resumed_training_verification_v1.json','0cc118c0391c84c9a932227d4654d6c6fe48cc823092a8131a16a2a42feab1b1'),
 'owner_long_measurement':(F/'actual_source30_full_validation163020_164620_01/report.json','10d0e9c1aa2324d86de0d9697253f5d3e5a943cd0f3deca54568de8016b35424'),
 'ROOT_long_measurement_review':(F/'ROOT_source30_long1600_slowdown_persists_actual_measurement_review_v1.json','cf574b9311063275362f35dd53881a354bdf1d898b04496e3a12ff7396377f4d')
}
pins={}
for k,(p,h) in inputs.items():
    pins[k]=pin(p);assert pins[k]['sha256']==h,(k,pins[k])
assert os.environ['TORCH_EXTENSIONS_DIR'].lower().startswith('e:'), 'E storage environment missing'
assert not OUT.exists() and not RESULT.exists(), 'Fresh exactly-once operation required'
before=TASK.read_text(encoding='utf-8').count(MARKER)
assert before==0,('Canonical marker already exists',before)
note=f'''{MARKER}
164,000 scheduled fixed DEV completed all 350 rows (7 x 50), with the original manifest order and independently recomputed word errors. Turkish: 199/825 = 24.121212%, 7 fewer errors (-0.848485 percentage points) than 160k. Retention EN59/ES62/IT23/FR72/DE70/PT63; only Spanish misses its immutable cap (62 vs 52, +10). No qualified winner, pass advance, early stop or TEST evaluation.
The real CPU11/max2 normal phase closed with wait0, 1,020 matched acquires/posts including the original 20-packet drain, closed transport/kernel handles, and unchanged model/Adam/RNG/cursors/journal/next draw. Resumed same-plan positive training and full standard 164,200 durability were independently accepted.
The canonical-byte archive candidate did NOT hold the speed improvement. Frozen complete 1,600-update interval 163020..164620 included all 8 natural saves and scheduled 164k DEV: 71.427147 audio min/wall min overall. First 400 vs final 400, each including two saves: 77.8975 -> 68.5596 (-11.9874%); corresponding save-time-excluded rates 80.8576 -> 70.2338. Composite cleanup boundary mean grew 0.176085 -> 0.255675 s/update; this does not isolate garbage collection from synchronization/state checks. Gate waits were 2.063 s with zero waited updates; no metadata end-wait or fallback. No speed-success or GC-alone cause claim.
Human-requested fix remains active. Healthy source30 training, all original guards/science, approved 2.25 GB extra cushion, MAX2/LIVE1, reader and environment remain unchanged while the next targeted diagnosis is reviewed.
Evidence: ROOT quality {pins['ROOT_quality_review']['sha256']}; owner quality {pins['owner_quality_status']['sha256']}; ROOT failed-fix measurement {pins['ROOT_long_measurement_review']['sha256']}; owner full-span report {pins['owner_long_measurement']['sha256']}.
'''
OUT.mkdir()
(OUT/'milestone_note.txt').write_text(note,encoding='utf-8',newline='\n')
entry={'kind':'ASR18_periodic164000_and_long1600_slowdown_persists_Backlog_entry_v1','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'marker':MARKER,'marker_count_before':before,'inputs':pins,'task_before':pin(TASK),'one_edit_only':True}
write(OUT/'entry.json',entry)
result={'kind':'ASR18_periodic164000_and_long1600_slowdown_persists_Backlog_result_v1','marker':MARKER,'marker_count_before':before,'inputs':pins,'CLI_edit_invocations':0,'actual_CLI_edit_exit':None,'actual_refresh_exit':None,'passed':False,'error':None,'refresh_interpreter':str(P/'work/training-env/Scripts/python.exe')}
try:
    result['CLI_edit_invocations']=1
    with (OUT/'edit.stdout.log').open('xb') as o,(OUT/'edit.stderr.log').open('xb') as e:
        r=subprocess.run([str(S/'tools/backlog.exe'),'task','edit','ASR-18','--append-notes',note],cwd=S,stdout=o,stderr=e,timeout=60,check=False)
    result['actual_CLI_edit_exit']=r.returncode
    assert r.returncode==0,('Backlog edit failed',r.returncode)
    assert TASK.read_text(encoding='utf-8').count(MARKER)==1,'Marker count after edit differs'
    with (OUT/'refresh.stdout.log').open('xb') as o,(OUT/'refresh.stderr.log').open('xb') as e:
        r=subprocess.run([result['refresh_interpreter'],'-Xutf8','-B',str(S/'refresh_backlog.py')],cwd=S,stdout=o,stderr=e,timeout=60,check=False)
    result['actual_refresh_exit']=r.returncode
    assert r.returncode==0,('Backlog refresh failed',r.returncode)
    result['marker_count_after']=TASK.read_text(encoding='utf-8').count(MARKER)
    assert result['marker_count_after']==1
    result['passed']=True
except BaseException as e:
    result['error']={'type':type(e).__name__,'message':str(e),'ambiguous_retry_forbidden':True}
finally:
    result['marker_count_after']=TASK.read_text(encoding='utf-8').count(MARKER)
    result['task_after']=pin(TASK)
    result['entry']=pin(OUT/'entry.json');result['note']=pin(OUT/'milestone_note.txt')
    result['logs']={p.name:pin(p) for p in sorted(OUT.glob('*.log'))}
    result['input_pins_unchanged']=all(pin(p)==pins[k] for k,(p,h) in inputs.items())
    result['no_training_source_environment_reader_or_child_change']=True
    write(RESULT,result)
print(json.dumps({'result':pin(RESULT),'passed':result['passed'],'edit':result['actual_CLI_edit_exit'],'refresh':result['actual_refresh_exit'],'marker_count':result['marker_count_after'],'error':result['error']}))
sys.exit(0 if result['passed'] else 1)
