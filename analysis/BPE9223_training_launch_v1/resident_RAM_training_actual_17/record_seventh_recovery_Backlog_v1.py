"""One accepted seventh-source recovery Backlog milestone and refresh."""
import datetime,hashlib,json,subprocess
from pathlib import Path
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish';A=Path(__file__).parent;N=A.parent
def pin(p):
 p=Path(p);b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def require(v,m):
 if not v:raise ValueError(m)
status=A/'actual_corrected_seventh_full_clip_saved143000_status.json';review=N/'ROOT_seventh_original_full_clip_saved143000_actual_verification_v1.json'
require(pin(status)['sha256']=='e32ea7a0812e81ae793238e2bf7b62b88858e8e9745c10c99abbf61e78a6bbb9'and pin(review)['sha256']=='386ffdb06ed0a20023a7bb05fde8399aefb07b1b02d6fa262c8d0fd895e1f346','Exact accepted actualstatus and independent ROOT verification')
v=json.loads(status.read_text(encoding='utf-8'));require(v['passed']and v['actual_optimizer_updates']>143000 and v['training_continues'],'Actual corrected consumption/fullsave/resumedpositive')
marker='BPE9223_SEVENTH_FULL_CLIP_SAVED142800_RECOVERED143000_20261022'
task=S/'backlog/tasks/asr-18 - Train-BPE-on-broad-weak-labels-then-fine-tune-with-multilingual-retention.md'
receipt=A/'actual_seventh_recovery_backlog_result.json';require(not receipt.exists(),'One result only');require(task.read_text(encoding='utf-8').count(marker)==0,'No duplicate canonical milestone')
note=(datetime.datetime.now(datetime.timezone.utc).isoformat()+': '+marker+' — Seventh exact original-source coverage recovery independently accepted. '
 'The source19 stop at142894 left coherent saved142800 and94 actual positive unsaved attempts142801..142894; all oldcommits15 bytes and the separately pinned tail remain historical and unchanged. Native and CPU numeric OS exits remain unknown; actual outer TOOL1 and all seven recorded process deaths were sealed. '
 'Exact alania:0ce0d55b73c946f80abea8bbe41f3b27_00000 physically contains693255 native48k samples versus historical693264, yielding231085 clean16k versus231088 reserved. Source/member/annotation/full61-label-bound certificate030535 and overlay7 preserve every physical sample once, validmel1444/features1445/encoderT181, original plans/indexes/labels/order/fullpass denominator and larger reservations. All six prior policies remain exact; no filler/crop/skip/filter or generic EOF tolerance. '
 'Actual original frontend/finite features/mask/full61 targets and eight strict rejection controls completed CPU0 with exact deaths; the final serialized typed gate also completed0. Source20/CPU9 restored saved142800 all207 weights/three Adam groups/four RNGs/both samplers/journal/nextdraw before native; only saved290088487B/e45e1adb prefix was copied AFTER restore with0native/draw into commits16. '
 '95 new ordered positives142801..142895 genuinely redid all94 historical attempts and consumed row6 at142895. Exact consumed metadata and fullnative commit digests match; archived JSON and integer-route-key reconstructed digest remain separately recorded. Checkpoint143000 verified854 tensors/207 weights/215 Adam states, PRE/SAVED and semantic equality; actual143040/new240 then ROOT143140/new340 had both poison flags false. '
 'Same healthy source20/CPU9 native24824 and residentCPU34876, sole owner session47103 continue full401072/adminNone unchanged. No optimizer/RNG/sampler/LR/science/resource/cadence reset, extra evaluation/export/inference or speed claim. Next scheduled144000 full350 and first genuine normal helper2 CPU-quiescence proof remain pending. '
 'Status '+str(status)+' SHAe32ea7a0; independent ROOT '+str(review)+' SHA386ffdb0.')
note_path=A/'seventh_recovery_milestone_note.txt';require(not note_path.exists(),'One preserved note');note_path.write_text(note+'\n',encoding='utf-8')
r=subprocess.run([str(S/'tools/backlog.exe'),'task','edit','ASR-18','--append-notes',note],cwd=S,capture_output=True,text=True,encoding='utf-8');edit=r.returncode
(A/'seventh_recovery_CLI_output.txt').write_text(r.stdout+'\n'+r.stderr,encoding='utf-8')
refresh=None
if edit==0:
 r=subprocess.run([str(P/'work/training-env/Scripts/python.exe'),str(S/'refresh_backlog.py')],cwd=P,capture_output=True,text=True,encoding='utf-8');refresh=r.returncode
 (A/'seventh_recovery_refresh_output.txt').write_text(r.stdout+'\n'+r.stderr,encoding='utf-8')
count=task.read_text(encoding='utf-8').count(marker)
receipt.write_text(json.dumps({'kind':'ASR18_actual_seventh_recovery_Backlog_milestone_v1','marker':marker,'actual_task_edit_exit_code':edit,'actual_refresh_exit_code':refresh,'canonical_marker_count_after_edit':count,'status':pin(status),'ROOT_actual_verification':pin(review),'note':pin(note_path),'training_changes':False,'duplicate_prior_note':False},indent=2)+'\n',encoding='utf-8')
print(json.dumps(pin(receipt)),flush=True);require(edit==refresh==0 and count==1,'ActualCLI0/refresh0/markerexactly1')