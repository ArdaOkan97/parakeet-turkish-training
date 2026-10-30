"""One authorized, deduplicated actual recovery milestone through Backlog CLI."""
import datetime, hashlib, json, subprocess
from pathlib import Path
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20'); OUT=P/'outputs/parakeet_turkish'; A=Path(__file__).parent; N=A.parent

def pin(path):
    path=Path(path); raw=path.read_bytes()
    return {'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}

def require(ok, message):
    if not ok: raise ValueError(message)

status=A/'actual_corrected_fifth_full_clip_saved134600_status.json'
review=N/'ROOT_fifth_original_full_clip_saved134600_actual_verification_v1.json'
require(pin(status)['sha256']=='73512902faa0becfd72140cb7e566cb6e1d70e84bf93012339ce40b42c6d5f0a' and pin(review)['sha256']=='8731a299c6f6203a633ef5451e1daac0a0ece3c4e4121d0c73de8a7e8e09132d','Exact accepted status and ROOT review')
v=json.loads(status.read_text()); require(v['passed'] and v['actual_optimizer_updates']>134600 and v['training_continues'],'Actual recovery and resumed positive training')
task=OUT/'backlog/tasks/asr-18 - Train-BPE-on-broad-weak-labels-then-fine-tune-with-multilingual-retention.md'
marker='BPE9223_FIFTH_FULL_CLIP_SAVED134400_RECOVERED134600_20261022'
receipt=A/'actual_fifth_full_clip_recovery_backlog_result.json'; require(not receipt.exists(),'One receipt only')
before=task.read_text(encoding='utf-8').count(marker); require(before in (0,1),'No duplicate milestone marker')
note=(datetime.datetime.now(datetime.timezone.utc).isoformat()+': '+marker+' — Genuine saved-state recovery after the source-coverage failure at update 134529. '
      'Saved checkpoint 134400, the entire previous journal and all 129 positive unsaved historical attempts remain preserved. The prior native numeric OS exit remains unknown. '
      'The exact original clip alania:9105d11d7d12c9c166028f98935dc107_00000 contains 600190 native samples, two fewer than its derived declaration. A narrowly source/member/annotation/label-bound certificate and delegating overlay use every physical sample exactly once: clean PCM 200064, physical mel 1250, features 1251, encoder T157 and all 82 target IDs. Original plans, annotations, indexes, labels, sampler order and larger conservative reservations remain unchanged; no padding, cropping, filtering or generic EOF tolerance. '
      'Independent actual CPU ingress/frontend and seven rejection controls passed; all prior four overlay policies and the source15 resource scheduling remain exact. '
      'Fresh saved134400 restore matched all 207 weights, three Adam states, four RNGs, both samplers, journal and next draw before native. Only the saved 273582680-byte prefix was copied after restore to active commits12. '
      '130 new ordered positive updates 134401..134530 genuinely redid the historical attempts and consumed the corrected clip in native update134530/row3. The consumed metadata and full native receipt exactly bind the commit; archived JSON route keys were restored only to their source-declared integer type for digest comparison, with raw digest retained separately. '
      'Checkpoint134600 verified 854 named tensors, 207 weights and 215 Adam states with independent PRE/SAVED and semantic checks; actual134780/new380 followed with both poison flags false and the same live native27232/CPU548, exclusive session64681. '
      'Full401072/adminNone continues. No model, optimizer, RNG, sampler, LR, DEV, winner or full-pass reset/advance; no speed claim or extra evaluation. '
      'Actual status '+str(status)+' SHA73512902; independent ROOT acceptance '+str(review)+' SHA8731a299.')
note_path=A/'fifth_full_clip_recovery_milestone_note.txt'; require(not note_path.exists(),'One preserved note'); note_path.write_text(note+'\n',encoding='utf-8')
edit_code=0
if before==0:
    r=subprocess.run([str(OUT/'tools/backlog.exe'),'task','edit','ASR-18','--append-notes',note],cwd=OUT,capture_output=True,text=True,encoding='utf-8')
    edit_code=r.returncode; (A/'fifth_full_clip_recovery_CLI_output.txt').write_text(r.stdout+'\n'+r.stderr,encoding='utf-8')
else:
    (A/'fifth_full_clip_recovery_CLI_output.txt').write_text('Existing unique marker; no duplicate CLI edit.\n',encoding='utf-8')
refresh_code=None
if edit_code==0:
    r=subprocess.run([str(P/'work/training-env/Scripts/python.exe'),str(OUT/'refresh_backlog.py')],cwd=P,capture_output=True,text=True,encoding='utf-8')
    refresh_code=r.returncode; (A/'fifth_full_clip_recovery_refresh_output.txt').write_text(r.stdout+'\n'+r.stderr,encoding='utf-8')
count=task.read_text(encoding='utf-8').count(marker)
result={'kind':'ASR18_actual_fifth_full_clip_saved134400_recovery_CLI_milestone_and_refresh_v1','marker':marker,'deduplicated_existing_marker':before==1,'actual_task_edit_exit_code':edit_code,'actual_refresh_exit_code':refresh_code,'canonical_marker_count_after_edit':count,'status':pin(status),'ROOT_actual_verification':pin(review),'note':pin(note_path),'training_changes':False,'duplicate_prior_note':False}
receipt.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8'); print(json.dumps(pin(receipt)),flush=True)
require(edit_code==refresh_code==0 and count==1,'One actual milestone edit and refresh0')
