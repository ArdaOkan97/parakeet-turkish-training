"""One accepted144k quality/real periodic CPU quiescence milestone."""
import datetime,hashlib,json,subprocess
from pathlib import Path
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish';A=Path(__file__).parent;N=A.parent
def pin(p):
 p=Path(p);b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def require(v,m):
 if not v:raise ValueError(m)
status=A/'actual_periodic144000_and_resumed_training_status.json';review=N/'ROOT_periodic144000_full350_real_quiescence_saved144200_and_resumed_training_verification_v1.json'
require(pin(status)['sha256']=='8f2b831399a4b84826ff9ce80a0819b1184296294ae38b05a0f95d4af29dfcc3'and pin(review)['sha256']=='fbd4f6254340594dcc2cdd1ea90d2a57358e8fb4a4c4e3ae01fe714ed14e59be','Exact accepted144k status and ROOT verification')
v=json.loads(status.read_text(encoding='utf-8'));require(v['passed']and v['actual_optimizer_updates']>144000 and v['training_continues'],'Actual full350/resumedpositive')
marker='BPE9223_DEV144000_REAL_HELPER2_QUIESCENCE_AND_RESUMED_20261022'
task=S/'backlog/tasks/asr-18 - Train-BPE-on-broad-weak-labels-then-fine-tune-with-multilingual-retention.md'
receipt=A/'actual_periodic144000_backlog_result.json';require(not receipt.exists(),'One result only');require(task.read_text(encoding='utf-8').count(marker)==0,'No duplicate canonical milestone')
note=(datetime.datetime.now(datetime.timezone.utc).isoformat()+': '+marker+' — Genuine scheduled144000 full350 DEV and first actual normal helper2 production CPU-quiescence phase independently accepted with resumed positive training. '
 'The unchanged fixed7x50 manifests/scorer produced Turkish210/825=25.454545%, two more errors/+0.242424 percentage points versus140000. Retention errors EN60/ES62/IT23/FR71/DE76/PT63: Spanish exceeds52 by10 and German exceeds70 by6; all other caps pass. No qualified winner, full-pass advance, early stop or TEST. '
 'Before DEV, real CPU9 drained/released exactly20 unconsumed private packets for pendingplan61, completed1220 prepared updates, actually returned Popen wait0, joined transport threads and closed pipe handles. All authoritative sampler/replay/journal/model/optimizer frontiers were preserved, with no private-plan regeneration or reordering. This is CPU normal0, not a GPU-process exit. '
 'Actual resident optimizer-idle DEV took81.406s with unchanged2.1GB incremental allowance, full256MiB extra prefetch reservation and original memory/CUDA/storage guards. All207 weights/three Adam groups/four RNGs/both cursors/journal/nextdraw and pendingplan61 were unchanged. After DEV the exact retained plan was resubmitted to genuine new resident CPU3680, and ROOT directly verified20 ordered positive updates144001..144020 from that same plan. '
 'Checkpoint144000 verified854 tensors/207 weights/215 Adam states and PRE/SAVED equality. Positive status144080/new1280 and independent ROOT144440/latestdurable144400 had both poison flags false; native24824 and newCPU3680 exact identities alive. Same sole owner session47103/source20/CPU9 continues full401072/adminNone unchanged; reader26892 remains live. This proves this actual phase only, with no saving, speed or future-protection claim. Next scheduled quality148000. '
 'Accepted status '+str(status)+' SHA8f2b8313; independent ROOT '+str(review)+' SHAfbd4f625.')
note_path=A/'periodic144000_milestone_note.txt';require(not note_path.exists(),'One preserved note');note_path.write_text(note+'\n',encoding='utf-8')
r=subprocess.run([str(S/'tools/backlog.exe'),'task','edit','ASR-18','--append-notes',note],cwd=S,capture_output=True,text=True,encoding='utf-8');edit=r.returncode
(A/'periodic144000_CLI_output.txt').write_text(r.stdout+'\n'+r.stderr,encoding='utf-8')
refresh=None
if edit==0:
 r=subprocess.run([str(P/'work/training-env/Scripts/python.exe'),str(S/'refresh_backlog.py')],cwd=P,capture_output=True,text=True,encoding='utf-8');refresh=r.returncode
 (A/'periodic144000_refresh_output.txt').write_text(r.stdout+'\n'+r.stderr,encoding='utf-8')
count=task.read_text(encoding='utf-8').count(marker)
receipt.write_text(json.dumps({'kind':'ASR18_actual_periodic144000_Backlog_milestone_v1','marker':marker,'actual_task_edit_exit_code':edit,'actual_refresh_exit_code':refresh,'canonical_marker_count_after_edit':count,'status':pin(status),'ROOT_actual_verification':pin(review),'note':pin(note_path),'training_changes':False,'duplicate_prior_note':False},indent=2)+'\n',encoding='utf-8')
print(json.dumps(pin(receipt)),flush=True);require(edit==refresh==0 and count==1,'ActualCLI0/refresh0/markerexactly1')