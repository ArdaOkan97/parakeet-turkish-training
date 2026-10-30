"""One ROOT-accepted saved156200/recovered156400 resource milestone and refresh."""
import datetime,hashlib,json,subprocess
from pathlib import Path
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish';A=Path(__file__).parent;N=A.parent
def pin(p):
 p=Path(p);raw=p.read_bytes();return {'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def require(v,m):
 if not v:raise ValueError(m)
status=A/'actual_native_pressure_saved156400_recovery_status.json'
review=N/'ROOT_native_commit_pressure_saved156200_recovered156400_actual_verification_v1.json'
require(pin(status)['sha256']=='fe241202c140de4c32937a66e6a3e59ca6c24422003c643698c056c2d3e88152'
 and pin(review)['sha256']=='3fc4e73fe1135e84bef34a75c473019dbba11043364bb1a0b0ad26d462f78f6e','Exact accepted new recovery status and ROOT verification')
v=json.loads(status.read_text(encoding='utf-8'));require(v['passed']and v['actual_optimizer_updates']>156400 and v['training_continues'],'Actual durable recovery and positive ongoing training')
marker='BPE9223_NATIVE_COMMIT_GUARD_SAVED156200_RECOVERED156400_20261022'
task=S/'backlog/tasks/asr-18 - Train-BPE-on-broad-weak-labels-then-fine-tune-with-multilingual-retention.md'
receipt=A/'actual_native_pressure_saved156400_backlog_result.json'
require(not receipt.exists(),'One immutable result');require(task.read_text(encoding='utf-8').count(marker)==0,'No duplicate canonical recovery milestone')
note=(datetime.datetime.now(datetime.timezone.utc).isoformat()+': '+marker+' — New system-commit guard failure and exact saved156200 recovery independently accepted. '
 'The old source20/session47103 stopped with actual TOOL1 after156300; its persisted sample recorded1,943,629,824B system commit headroom below the unchanged2,000,000,000B floor, while physical available memory remained10,373,808,128B. Separate historical native/CPU/wrapper numeric OS exits remain unknown. Saved156200 and all100 later positive attempts156201..156300 were preserved; those attempts remain historical, never replayed as durable state. '
 'Reviewed source22 adds an explicit conservative2.5GB pre-update scheduling buffer plus the existing full256MiB prefetch reservation, requiring4,768,435,456B available commit before packet/propose/native. The original actual2GB commit/1.5GB physical, CUDA, storage, CPU9 single-live-packet/full256MiB, numerical, sampler, checkpoint200 and periodic4000 policies remain unchanged. Bounded300s timeout preserves its exact last denied sample and thresholds. CPU proof comprised nine honest method-stub controls plus genuine unchanged original Monitor sampling/thread/close/cooperative-stop composition, not native allocation or failure-prevention proof. '
 'Actual source22 restored all207 weights/three Adam groups/four RNGs/both samplers/nextdraw before any native work, then copied only the saved316408290B prefix into journal17 with0native/draw. ROOT independently verified101 ordered NEW positive events156201..156301, genuinely redoing every historical100 and passing the old failure frontier. Standard156400 persistence verified854 tensors/207 weights/215 Adam states with PRE/SAVED and semantic publication equality. Owner status156440/new240 and independent ROOT156480/new280 showed all3 clocks exact/both poison flags false and all5 owned family identities alive. '
 'First200 production windows recorded38.204s of admitted scheduling waits across14updates. These are waiting observations, not measured freeing, required-allocation, workload-forecast, speed or future-protection claims; unrelated mid-update pressure can still trip the unchanged monitor. Trackio reader26892 caught up to journal17/new native12328/workertrue/errorNone/durable156400 without a restart. '
 'Same sole owner35083/source22/CPU9/native12328/currentCPU28556 and full401072/adminNone continue unchanged; next scheduled quality160000. This is one new recovery milestone, not a repeated156k quality note. '
 'Accepted owner status '+str(status)+' SHAfe241202; independent ROOT '+str(review)+' SHA3fc4e73f.')
note_path=A/'native_pressure_saved156400_milestone_note.txt';require(not note_path.exists(),'One note');note_path.write_text(note+'\n',encoding='utf-8')
r=subprocess.run([str(S/'tools/backlog.exe'),'task','edit','ASR-18','--append-notes',note],cwd=S,capture_output=True,text=True,encoding='utf-8');edit=r.returncode
(A/'native_pressure_saved156400_CLI_output.txt').write_text(r.stdout+'\n'+r.stderr,encoding='utf-8')
refresh=None
if edit==0:
 r=subprocess.run([str(P/'work/training-env/Scripts/python.exe'),str(S/'refresh_backlog.py')],cwd=P,capture_output=True,text=True,encoding='utf-8');refresh=r.returncode
 (A/'native_pressure_saved156400_refresh_output.txt').write_text(r.stdout+'\n'+r.stderr,encoding='utf-8')
count=task.read_text(encoding='utf-8').count(marker)
receipt.write_text(json.dumps({'kind':'ASR18_actual_native_pressure_saved156400_Backlog_milestone_v1','marker':marker,'actual_task_edit_exit_code':edit,'actual_refresh_exit_code':refresh,'canonical_marker_count_after_edit':count,'status':pin(status),'ROOT_actual_verification':pin(review),'note':pin(note_path),'training_changes':False,'duplicate_prior_note':False},indent=2)+'\n',encoding='utf-8')
print(json.dumps(pin(receipt)),flush=True);require(edit==refresh==0 and count==1,'ActualCLI0/refresh0/markerexactly1')
