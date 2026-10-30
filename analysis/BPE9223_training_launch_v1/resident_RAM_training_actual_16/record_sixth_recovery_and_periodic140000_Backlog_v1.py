"""Exactly one authorized combined recovery/140k milestone and refresh."""
import datetime,hashlib,json,subprocess
from pathlib import Path
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish';A=Path(__file__).parent;N=A.parent
def pin(p):
 p=Path(p);b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def require(ok,message):
 if not ok:raise ValueError(message)
status=A/'actual_corrected_sixth_terminal_saved140400_status.json'
review=N/'ROOT_sixth_original_terminal_interval_saved140400_actual_verification_v1.json'
quality=N/'ROOT_periodic140000_quality_and_new140236_failure_compact_review_v1.json'
require(pin(status)['sha256']=='1cdae4f42327e306a1f7b0fb727c149067dd1fb86b0126663195bbf3f57a172c'
 and pin(review)['sha256']=='416c21a7b1b932863e9146dd0c42ae8ff9d99c1824fe99c64d2d35ed67a52043'
 and pin(quality)['sha256']=='d689f9f07ca0e8906cdaf8a24240d5e669f4bc7de9cfd5d12c6d9fec65c7cd99','Exact accepted recovery status, ROOT verification and historical140k quality')
value=json.loads(status.read_text());require(value['passed']and value['actual_optimizer_updates']>140400 and value['training_continues'],'Genuine corrected consumption/fullsave/resumed positive training')
task=S/'backlog/tasks/asr-18 - Train-BPE-on-broad-weak-labels-then-fine-tune-with-multilingual-retention.md'
marker='BPE9223_DEV140000_AND_SIXTH_TERMINAL_SAVED140200_RECOVERED140400_20261022'
receipt=A/'actual_sixth_recovery_and_periodic140000_backlog_result.json';require(not receipt.exists(),'One combined result receipt only')
before=task.read_text(encoding='utf-8').count(marker);require(before in (0,1),'No duplicate canonical marker')
note=(datetime.datetime.now(datetime.timezone.utc).isoformat()+': '+marker+' — Genuine full350 DEV at140000 and subsequent sixth source-bound terminal-interval recovery, independently accepted together. '
 'The140000 evaluation used the original fixed7x50 manifests/scorer with0 native updates: Turkish208/825=25.212121%,5 more errors/+0.606061 percentage points versus136000. '
 'Retention errors EN61/ES61/IT23/FR72/DE72/PT64; Spanish exceeds52 by9 and German exceeds70 by2. No qualified winner, full-pass advance, early stop or TEST. All207 weights/three Adam groups/four RNGs/both samplers/journal/nextdraw and pending-plan state were preserved across that actual evaluation. '
 'The later140236 stop was a separate original-source coverage failure; its native numeric OS exit remains unknown. Saved140200, the entire oldcommits12 lineage and all36 positive unsaved attempts140201..140236 remain preserved. '
 'The exact original terminal interval alania:38198deb36bb10f9e58129b2c9d100fe_00003 physically contains925487 selected48k samples versus the925488 derived declaration. A narrowly source/member/annotation/full-label-bound certificate and delegating overlay use every selected physical sample once, without synthesis/crop/skip/filter or generic EOF tolerance: clean308496, mel1928, features1929, encoderT241 and all121 targets. Original plans/indexes/annotations/labels/sampler order/larger historical reservations remain unchanged; all five prior policies remain exact. '
 'Actual CPU ingress/original frontend, eight strict rejection controls and the real serialized typed gate completed0 with exact process deaths. Fresh140200 restore matched all207 weights/three Adam groups/four RNGs/both samplers/journal/nextdraw before native; only the saved284977815-byte prefix was copied AFTER restore to newcommits15 with0 native/draws. '
 '37 new ordered positives140201..140237 genuinely redid all36 historical attempts and consumed the corrected row4 at140237. Consumed metadata3e09e4f1 and fullnativef08aa6a4 exactly bind the new journal commit; only source-declared integer joint-route keys were reconstructed for the commit digest, retaining the raw archived JSON digest32f9cd28 separately. '
 'Checkpoint140400 independently verified854 tensors/207 weights/215 Adam states with PRE/SAVED and semantic equality; positive140440/new240 followed, and ROOT independently observed140540 with both poison flags false. The same live native37316/CPU38236, source19/CPU8 and sole owner session1903 continue full401072/adminNone. '
 'No optimizer/RNG/sampler/LR/science/resource/cadence reset, extra inference/DEV/export or speed claim. Next scheduled periodic144000 and real CPU quiescence proof remain pending. '
 'Recovery status '+str(status)+' SHA1cdae4f4; ROOT recovery '+str(review)+' SHA416c21a7; ROOT actual140k quality '+str(quality)+' SHAd689f9f0.')
note_path=A/'sixth_recovery_and_periodic140000_milestone_note.txt';require(not note_path.exists(),'One preserved combined note');note_path.write_text(note+'\n',encoding='utf-8')
edit=0
if before==0:
 r=subprocess.run([str(S/'tools/backlog.exe'),'task','edit','ASR-18','--append-notes',note],cwd=S,capture_output=True,text=True,encoding='utf-8');edit=r.returncode
 (A/'sixth_recovery_and_periodic140000_CLI_output.txt').write_text(r.stdout+'\n'+r.stderr,encoding='utf-8')
else:(A/'sixth_recovery_and_periodic140000_CLI_output.txt').write_text('Existing unique marker; no duplicate edit.\n',encoding='utf-8')
refresh=None
if edit==0:
 r=subprocess.run([str(P/'work/training-env/Scripts/python.exe'),str(S/'refresh_backlog.py')],cwd=P,capture_output=True,text=True,encoding='utf-8');refresh=r.returncode
 (A/'sixth_recovery_and_periodic140000_refresh_output.txt').write_text(r.stdout+'\n'+r.stderr,encoding='utf-8')
count=task.read_text(encoding='utf-8').count(marker)
receipt.write_text(json.dumps({'kind':'ASR18_actual_sixth_recovery_and_140k_quality_Backlog_milestone_v1','marker':marker,
 'deduplicated_existing_marker':before==1,'actual_task_edit_exit_code':edit,'actual_refresh_exit_code':refresh,'canonical_marker_count_after_edit':count,
 'status':pin(status),'ROOT_actual_verification':pin(review),'ROOT_actual140k_quality':pin(quality),'note':pin(note_path),
 'training_changes':False,'duplicate_prior_note':False},indent=2)+'\n',encoding='utf-8')
print(json.dumps(pin(receipt)),flush=True);require(edit==refresh==0 and count==1,'Exactly one milestone CLI0/refresh0/marker1')
