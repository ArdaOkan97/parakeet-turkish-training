"""One accepted148k quality/normal quiescence milestone and refresh."""
import datetime,hashlib,json,subprocess
from pathlib import Path
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish';A=Path(__file__).parent;N=A.parent
def pin(p):
 p=Path(p);b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def require(v,m):
 if not v:raise ValueError(m)
status=A/'actual_periodic148000_and_resumed_training_status.json';review=N/'ROOT_periodic148000_full350_normal_quiescence_saved148200_and_resumed_training_verification_v1.json'
require(pin(status)['sha256']=='8c68e1a3e24e133dd0a2116489aad3059c34bfdc90e36a6682e67c17afb1c9dd'and pin(review)['sha256']=='e409de30bfc1310ae2f39d59e2966d9baf20f471a9369ebf3c903bb3a6d49397','Exact accepted148k status and ROOT verification')
v=json.loads(status.read_text(encoding='utf-8'));require(v['passed']and v['actual_optimizer_updates']>148200 and v['training_continues'],'Actual full350/resumedpositive/nextdurability')
marker='BPE9223_DEV148000_NORMAL_QUIESCENCE_SAVED148200_RESUMED_20261022'
task=S/'backlog/tasks/asr-18 - Train-BPE-on-broad-weak-labels-then-fine-tune-with-multilingual-retention.md'
receipt=A/'actual_periodic148000_backlog_result.json';require(not receipt.exists(),'One result only');require(task.read_text(encoding='utf-8').count(marker)==0,'No duplicate canonical milestone')
note=(datetime.datetime.now(datetime.timezone.utc).isoformat()+': '+marker+' — Genuine scheduled148000 full350 DEV, normal helper2 CPU quiescence, resumed positive training and standard148200 persistence independently accepted. '
 'The unchanged fixed7x50 manifests/scorer produced Turkish208/825=25.212121%, two fewer errors/−0.242424 percentage points versus144000. Retention errors EN61/ES61/IT23/FR74/DE71/PT65: Spanish exceeds52 by9 and German exceeds70 by1; French equals cap74 and all other caps pass. No qualified winner, full-pass advance, early stop or TEST. '
 'Actual CPU9 drained/released20 unconsumed private packets for exact pendingplan261, completed4020 prepared updates and returned genuine Popen wait0 with transport threads joined and pipes closed. No authoritative sampler/replay/journal advance or private-plan regeneration/reordering occurred. All207 weights/three Adam groups/four RNGs/both cursors/journal/nextdraw and pendingplan261 remained exact across DEV. '
 'After DEV the retained plan was resubmitted to actual new resident CPU3252; ROOT directly verified20 ordered positive updates148001..148020 from the same plan. Standard148000 and148200 checkpoint persistence verified854 tensors/207 weights/215 Adam states with PRE/SAVED equality. Positive status148220/new5420 and independent ROOT148940/new6140/latestdurable148800 had both poison flags false and native24824/CPU3252 exact identities alive; reader26892 remained live. '
 'Same sole owner47103/source20/CPU9/environment and full401072/adminNone continue unchanged. CPU normal0 is not a GPU-process exit. Original numerical, resource, reservation, checkpoint and DEV policies remain unchanged; no extra evaluator/control/study/export/inference, saving, speed or future-protection claim. Next scheduled quality152000. '
 'Accepted status '+str(status)+' SHA8c68e1a3; independent ROOT '+str(review)+' SHAe409de30.')
note_path=A/'periodic148000_milestone_note.txt';require(not note_path.exists(),'One preserved note');note_path.write_text(note+'\n',encoding='utf-8')
r=subprocess.run([str(S/'tools/backlog.exe'),'task','edit','ASR-18','--append-notes',note],cwd=S,capture_output=True,text=True,encoding='utf-8');edit=r.returncode
(A/'periodic148000_CLI_output.txt').write_text(r.stdout+'\n'+r.stderr,encoding='utf-8')
refresh=None
if edit==0:
 r=subprocess.run([str(P/'work/training-env/Scripts/python.exe'),str(S/'refresh_backlog.py')],cwd=P,capture_output=True,text=True,encoding='utf-8');refresh=r.returncode
 (A/'periodic148000_refresh_output.txt').write_text(r.stdout+'\n'+r.stderr,encoding='utf-8')
count=task.read_text(encoding='utf-8').count(marker)
receipt.write_text(json.dumps({'kind':'ASR18_actual_periodic148000_Backlog_milestone_v1','marker':marker,'actual_task_edit_exit_code':edit,'actual_refresh_exit_code':refresh,'canonical_marker_count_after_edit':count,'status':pin(status),'ROOT_actual_verification':pin(review),'note':pin(note_path),'training_changes':False,'duplicate_prior_note':False},indent=2)+'\n',encoding='utf-8')
print(json.dumps(pin(receipt)),flush=True);require(edit==refresh==0 and count==1,'ActualCLI0/refresh0/markerexactly1')