"""One accepted152k quality/normal quiescence milestone and refresh."""
import datetime,hashlib,json,subprocess
from pathlib import Path
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish';A=Path(__file__).parent;N=A.parent
def pin(p):
 p=Path(p);b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def require(v,m):
 if not v:raise ValueError(m)
status=A/'actual_periodic152000_and_resumed_training_status.json';review=N/'ROOT_periodic152000_full350_normal_quiescence_saved152200_and_resumed_training_verification_v1.json'
require(pin(status)['sha256']=='284e233723282b9dc9764d61110574ebd47a4329f3c0bfd2cb0099c16beb2586'and pin(review)['sha256']=='7191ffd096ff81aa07d47758fdb0625807950e27357329fa17c7c4ec82aa6d45','Exact accepted152k status and ROOT verification')
v=json.loads(status.read_text(encoding='utf-8'));require(v['passed']and v['actual_optimizer_updates']>152200 and v['training_continues'],'Actual full350/resumedpositive/nextdurability')
marker='BPE9223_DEV152000_NORMAL_QUIESCENCE_SAVED152200_RESUMED_20261022'
task=S/'backlog/tasks/asr-18 - Train-BPE-on-broad-weak-labels-then-fine-tune-with-multilingual-retention.md'
receipt=A/'actual_periodic152000_backlog_result.json';require(not receipt.exists(),'One result only');require(task.read_text(encoding='utf-8').count(marker)==0,'No duplicate canonical milestone')
note=(datetime.datetime.now(datetime.timezone.utc).isoformat()+': '+marker+' — Genuine scheduled152000 full350 DEV, normal CPU quiescence, resumed positive training and standard152200 persistence independently accepted. '
 'The unchanged fixed7x50 manifests/scorer produced Turkish204/825=24.727273%, four fewer errors/−0.484848 percentage points versus148000. Retention errors EN62/ES60/IT23/FR72/DE68/PT63. German now passes cap70 again; only Spanish exceeds its cap52 by8. No qualified winner, full-pass advance, early stop or TEST. '
 'Actual CPU9 drained/released20 unconsumed private packets for exact pendingplan461, completed4020 prepared updates and returned genuine Popen wait0 with transport threads joined and pipes closed. No authoritative sampler/replay/journal advance or private-plan regeneration/reordering occurred. All207 weights/three Adam groups/four RNGs/both cursors/journal/nextdraw and pendingplan461 remained exact across DEV. '
 'After DEV the retained plan was resubmitted to actual new resident CPU37808; ROOT directly verified20 ordered positive updates152001..152020 from that same plan. Standard152000 and152200 checkpoint persistence verified854 tensors/207 weights/215 Adam states with PRE/SAVED equality. Positive status152240/new9440 and independent ROOT152520/new9720/latestdurable152400 had both poison flags false; native24824/currentCPU37808/reader26892 exact identities remained live. '
 'Same sole owner47103/source20/CPU9/environment and full401072/adminNone continue unchanged. CPU normal0 is not a GPU-process exit. Original numerical, resource, reservation, checkpoint and DEV policies remain unchanged; no extra evaluator/control/study/export/inference, saving, speed or future-protection claim. Next scheduled quality156000. '
 'Accepted status '+str(status)+' SHA284e2337; independent ROOT '+str(review)+' SHA7191ffd0.')
note_path=A/'periodic152000_milestone_note.txt';require(not note_path.exists(),'One preserved note');note_path.write_text(note+'\n',encoding='utf-8')
r=subprocess.run([str(S/'tools/backlog.exe'),'task','edit','ASR-18','--append-notes',note],cwd=S,capture_output=True,text=True,encoding='utf-8');edit=r.returncode
(A/'periodic152000_CLI_output.txt').write_text(r.stdout+'\n'+r.stderr,encoding='utf-8')
refresh=None
if edit==0:
 r=subprocess.run([str(P/'work/training-env/Scripts/python.exe'),str(S/'refresh_backlog.py')],cwd=P,capture_output=True,text=True,encoding='utf-8');refresh=r.returncode
 (A/'periodic152000_refresh_output.txt').write_text(r.stdout+'\n'+r.stderr,encoding='utf-8')
count=task.read_text(encoding='utf-8').count(marker)
receipt.write_text(json.dumps({'kind':'ASR18_actual_periodic152000_Backlog_milestone_v1','marker':marker,'actual_task_edit_exit_code':edit,'actual_refresh_exit_code':refresh,'canonical_marker_count_after_edit':count,'status':pin(status),'ROOT_actual_verification':pin(review),'note':pin(note_path),'training_changes':False,'duplicate_prior_note':False},indent=2)+'\n',encoding='utf-8')
print(json.dumps(pin(receipt)),flush=True);require(edit==refresh==0 and count==1,'ActualCLI0/refresh0/markerexactly1')