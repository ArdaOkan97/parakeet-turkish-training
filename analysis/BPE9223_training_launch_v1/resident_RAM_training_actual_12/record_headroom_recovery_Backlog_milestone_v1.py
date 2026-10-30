"""One authorized recovery milestone through the established Backlog CLI."""
import datetime,hashlib,json,subprocess
from pathlib import Path
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');OUT=P/'outputs/parakeet_turkish';A=Path(__file__).parent;N=A.parent
def pin(p):
    p=Path(p);raw=p.read_bytes();return {'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def require(ok,message):
    if not ok:raise ValueError(message)
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
status=A/'actual_saved133200_recovery_status.json';review=N/'ROOT_single_live_packet_saved133200_actual_recovery_verification_v1.json'
require(pin(status)['sha256']=='e63e9beb0a5570e0d31acca6aec72a26edc6a8dff8a099ea7393cee360c2d999'
    and pin(review)['sha256']=='4eb688d3eab0b8e4cd3440a0368624efc1fa78031ca95caee1676a857efc760b','Exact accepted actual recovery/status evidence')
v=json.loads(status.read_text());require(v['passed']and v['actual_optimizer_updates']>=133220 and v['same_native_and_CPU_actually_alive'],'Genuine new recovery plus continued training')
task=OUT/'backlog/tasks/asr-18 - Train-BPE-on-broad-weak-labels-then-fine-tune-with-multilingual-retention.md'
marker='BPE9223_HEADROOM133137_SAVED133000_RECOVERED133200_SINGLE_PACKET_20261022'
receipt=A/'actual_headroom133137_recovery_backlog_result.json';require(not receipt.exists(),'One new receipt only')
before=task.read_text(encoding='utf-8').count(marker);require(before in (0,1),'Never duplicate a milestone marker')
note=(datetime.datetime.now(datetime.timezone.utc).isoformat()+': '+marker+' — Exact saved-state recovery after the actual system-commit headroom failure at133137. '
    'Checkpoint133000 and all137 positive unsaved historical attempts were preserved, with unknown native numeric OS exit kept unknown. '
    'New reviewed source15/CPUstream6 keeps the full256MiB packet/queue and additional256MiB private-metadata reservation, every numerical/native/commit/DEV body and existing actual memory/CUDA/E-reserve guards. '
    'Receiver credit now permits one live packet including delivered-unreleased; stronger prospective CPU allocation reserve2.5GB with bounded300s pre-allocation wait. This is changed resource scheduling, not a proved historical queue cause or measured memory-saving claim. '
    'Real CPU protocol controls accepted the full256MiB packet and verified ordered release/backpressure/error/abort joins and closed handles (TOOL0,7.703s,602730496B fixture-family peak; no model/audio/GPU). '
    'Genuine fresh133000 restore matched all207 weights/threeAdam/allfourRNG/both cursors/journal/nextdraw before native; only the saved270832675-byte prefix was copied to commits11. '
    '138 new ordered positive updates133001..133138 prove the old137 attempts were actually redone and the failed frontier passed. '
    'Checkpoint133200 verified854 named tensors/207 weights/215 Adam states and independent PRE/SAVED/live semantic equality, followed by real positive training; same sole owner session60255/native39588/CPU27116. '
    'Full401072/adminNone continues unchanged; no optimizer/RNG/sampler/LR reset, new inference/evaluation, winner or full-pass advance. '
    'Actual status '+str(status)+' SHAe63e9beb; independent ROOT acceptance '+str(review)+' SHA4eb688d3.')
note_path=A/'headroom133137_recovery_milestone_note.txt';require(not note_path.exists(),'Preserve one exact note');note_path.write_text(note+'\n',encoding='utf-8')
edit_code=0
if before==0:
    edited=subprocess.run([str(OUT/'tools/backlog.exe'),'task','edit','ASR-18','--append-notes',note],cwd=OUT,capture_output=True,text=True,encoding='utf-8')
    edit_code=edited.returncode;(A/'headroom133137_recovery_CLI_output.txt').write_text(edited.stdout+'\n'+edited.stderr,encoding='utf-8')
else:(A/'headroom133137_recovery_CLI_output.txt').write_text('Existing unique marker; no duplicate CLI edit.\n',encoding='utf-8')
refresh_code=None
if edit_code==0:
    refreshed=subprocess.run([str(P/'work/training-env/Scripts/python.exe'),str(OUT/'refresh_backlog.py')],cwd=P,capture_output=True,text=True,encoding='utf-8')
    refresh_code=refreshed.returncode;(A/'headroom133137_recovery_refresh_output.txt').write_text(refreshed.stdout+'\n'+refreshed.stderr,encoding='utf-8')
count=task.read_text(encoding='utf-8').count(marker)
result={'kind':'ASR18_actual_headroom133137_saved133000_recovery_CLI_milestone_and_refresh_v1','marker':marker,
    'deduplicated_existing_marker':before==1,'actual_task_edit_exit_code':edit_code,'actual_refresh_exit_code':refresh_code,
    'canonical_marker_count_after_edit':count,'status':pin(status),'ROOT_actual_verification':pin(review),'note':pin(note_path),
    'training_changes':False,'duplicate_prior_note':False}
write(receipt,result);print(json.dumps(pin(receipt)),flush=True)
require(edit_code==refresh_code==0 and count==1,'Exact one actual milestone edit and refresh0')
