"""One authorized combined136k recovery/quality Backlog milestone."""
import datetime,hashlib,json,subprocess
from pathlib import Path
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish';A=Path(__file__).parent;N=A.parent

def pin(p):
 p=Path(p);raw=p.read_bytes();return {'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def require(ok,message):
 if not ok:raise ValueError(message)
status=A/'actual_periodic136000_recovery_and_saved136200_resumed_training_status.json';review=N/'ROOT_periodic136000_recovery_saved136200_and_resumed_training_verification_v1.json'
require(pin(status)['sha256']=='d40d2d5fd5d46949e558cbc8ca65c0b5f75d23440047fe20caef828088d3cd7e' and pin(review)['sha256']=='28b610d407e689771833dc9bdede6f3ef3d30825c02719d3a625fe42a2331cd4','Exact independently accepted combined recovery/quality evidence')
v=json.loads(status.read_text());require(v['passed'] and v['actual_optimizer_updates_at_observation']>136200 and v['periodic_result']['score']['word_errors']['tr']==203,'Actual full350 and persisted resumed training')
task=S/'backlog/tasks/asr-18 - Train-BPE-on-broad-weak-labels-then-fine-tune-with-multilingual-retention.md';marker='BPE9223_DEV136000_ADMISSION_RECOVERED136200_AND_FULL350_20261022';receipt=A/'actual_periodic136000_recovery_backlog_result.json'
require(not receipt.exists(),'One receipt');before=task.read_text(encoding='utf-8').count(marker);require(before in (0,1),'No duplicate marker')
note=(datetime.datetime.now(datetime.timezone.utc).isoformat()+': '+marker+' — The separate prospective DEV memory-admission stop preserved all 136000 actual updates in a verified checkpoint; the whole journal exactly equals its saved prefix, with zero unsaved tail. Actual old TOOL1/deaths and unknown native numeric OS code remain recorded. '
 'Reviewed source17 keeps CPU7, native1, DEV2, all five exact original-clip certificates, numerical/transaction/science bodies and every existing resource guard/reservation unchanged. It schedules periodic CPU preparation closure/restart around DEV, retaining the exact unconsumed plan for re-preparation; denial samples are now persisted before the unchanged admission guard. No historical failing bytes or measured memory saving is inferred. '
 'Fresh136000 restore matched all207 weights/threeAdam/fourRNG/both samplers/journal/nextdraw before any new update. The pending original full350 DEV ran BEFORE creating a CPU preparation producer or committing136001; source17 reused the same complete commits12 without truncation, branch copy or tail replay. '
 'Actual DEV covered all seven fixed50-row sets with zero native updates and exact state/nextdraw/fullpass/winner preservation. Turkish203/825=24.606061%,13 fewer errors and1.575758 percentage points better than132000. Retention word errors EN61/ES61/IT23/FR70/DE71/PT64; ES exceeds immutable52 by9 and DE exceeds70 by1, so no qualified winner. TEST, early stop and full-pass advancement remain false. '
 'Twenty new ordered positive commits136001..136020 were independently verified; checkpoint136200 verified854 named tensors/207 weights/215 Adam states with PRE/SAVED and semantic equality, followed by actual136220 and later ROOT137100. Same exclusive owner session86052/native41200/CPU26908 continues full401072/adminNone; no optimizer/RNG/sampler/LR reset. '
 'This recovery proves startup DEV without a resident producer. The first real future periodic private-plan drain/CPU wait0 is140000 and is not claimed yet. Next scheduled quality140000; no extra inference, samples, benchmark or diagnostic activation. '
 'Status '+str(status)+' SHAd40d2d5f; independent ROOT acceptance '+str(review)+' SHA28b610d4.')
note_path=A/'periodic136000_recovery_milestone_note.txt';require(not note_path.exists(),'One preserved note');note_path.write_text(note+'\n',encoding='utf-8')
edit_code=0
if before==0:
 r=subprocess.run([str(S/'tools/backlog.exe'),'task','edit','ASR-18','--append-notes',note],cwd=S,capture_output=True,text=True,encoding='utf-8');edit_code=r.returncode;(A/'periodic136000_recovery_task_edit.log').write_text(r.stdout+'\n'+r.stderr,encoding='utf-8')
else:(A/'periodic136000_recovery_task_edit.log').write_text('Existing unique marker; no duplicate edit.\n',encoding='utf-8')
refresh_code=None
if edit_code==0:
 r=subprocess.run([str(P/'work/training-env/Scripts/python.exe'),str(S/'refresh_backlog.py')],cwd=P,capture_output=True,text=True,encoding='utf-8');refresh_code=r.returncode;(A/'periodic136000_recovery_refresh.log').write_text(r.stdout+'\n'+r.stderr,encoding='utf-8')
count=task.read_text(encoding='utf-8').count(marker)
result={'kind':'ASR18_actual_combined136000_recovery_quality_Backlog_milestone_refresh_v1','marker':marker,'deduplicated_existing_marker':before==1,'actual_task_edit_exit_code':edit_code,'actual_refresh_exit_code':refresh_code,'canonical_marker_count_after_edit':count,'status':pin(status),'ROOT_actual_verification':pin(review),'note':pin(note_path),'training_changes':False,'duplicate_prior_note':False}
receipt.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(pin(receipt)),flush=True);require(edit_code==refresh_code==0 and count==1,'One actual edit0/refresh0/marker1')
