"""Exactly-once authorized failure/required-action Backlog bookkeeping; no training operations."""
import datetime,hashlib,json,os,subprocess
from pathlib import Path
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish'
F=Path('E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1/resident_RAM_native_pressure_timeout156769_failure_actual_v1');N=F.parent
TASK=S/'backlog/tasks/asr-18 - Train-BPE-on-broad-weak-labels-then-fine-tune-with-multilingual-retention.md'
MARKER='BPE9223_ADDED_BUFFER_TIMEOUT156769_SAVED156600_AWAITING_POLICY_20261023'
ROOT=N/'ROOT_source24_native_pressure_timeout156769_saved156600_compact_review_v1.json'
FINDINGS=F/'readonly_source24_pressure_and_packet_lifetime_findings_v1.json'
NOTE=F/'added_buffer_timeout156769_required_action_note.txt'
ATTEMPT=F/'added_buffer_timeout156769_backlog_attempt.json'
RESULT=F/'actual_added_buffer_timeout156769_backlog_result.json'
def pin(p):
 raw=p.read_bytes();return {'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def write(p,v):
 with p.open('x',encoding='utf-8',newline='\n')as f:json.dump(v,f,indent=2);f.write('\n')
assert Path(os.environ['GPU_TRAINING_STORAGE_ROOT'])==Path('E:/gpu_training_scratchpad')
assert Path(os.environ['TORCH_EXTENSIONS_DIR'])==Path('E:/gpu_training_scratchpad/cache/torch_extensions')
assert pin(ROOT)['sha256']=='02de44ce16fecb8cf776b6ca1a3119484475d1eb2e9a0b5c8f85972540f60180'
assert pin(FINDINGS)['sha256']=='1f6e3230612d708cf09df34e7eeef27894b02e28e40c655e952bd06645940f33'
assert not RESULT.exists()and not ATTEMPT.exists()and not NOTE.exists()
assert TASK.read_text(encoding='utf-8-sig').count(MARKER)==0
note=(datetime.datetime.now(datetime.timezone.utc).isoformat()+': '+MARKER+' — '
 'New source24 stop and required human policy decision; training is STOPPED. Actual native156769, last fully saved156600. '
 'All169 later ordered positive attempts156601..156769 remain preserved as historical evidence and require genuine redo after an admitted saved-state continuation. '
 'The FIRST pre-take gate timed out after300s/600observations before packet readiness/take, schedule proposal or native update156770. '
 'Last available system commitment4,669,493,248B was98,942,208B below the unchanged4,768,435,456B requirement comprising the original2GB floor, full256MiB reservation and conservative additional2.5GB buffer. '
 'The original memory Monitor joined cleanly with no error; this is an added-buffer admission timeout, not a measured original-floor crossing or OOM. '
 'Owned session96057 returned actual outer TOOL1/102679; separate native/CPU/wrapper numeric OS exits remain UNKNOWN. Allfive production identities and the two metadata-sealer identities were verified dead. '
 'Checkpoint156600 retains full854/207/215 PRE-SAVED and semantic persistence; no checkpoint tensor, original prefix or whole-journal read was needed for the new compact failure review. '
 'The max1 logical packet-credit change did not prevent this timeout. Existing source already drops sender/receiver payload references, and no proven missing cleanup or small preserving remedy was found. '
 'The exact preparing/transferring/queued state and CPU private memory at denial were not recorded; no memory-saving, speed, required-allocation or prevention claim is made. '
 'Human direction is PENDING with no answer received: review a lower additional buffer, retain2.5GB and design CPU/native separation, or keep stopped. '
 'No buffer reduction, unchanged retry, control, source/config/guard change or training launch is authorized now. Original guards/reservations/science remain unchanged and the full401072 objective remains unfinished. '
 'Accepted failure review '+str(ROOT)+' SHA02de44ce; readonly findings '+str(FINDINGS)+' SHA1f6e3230. '
 'This is one new failure/required-action note; no old quality or recovery note is repeated.')
with NOTE.open('x',encoding='utf-8',newline='\n')as f:f.write(note+'\n')
write(ATTEMPT,{'kind':'ASR18_added_buffer_timeout156769_single_Backlog_attempt_v1','marker':MARKER,'canonical_marker_count_before':0,'note':pin(NOTE),'state':'before_task_edit'})
edit=subprocess.run([str(S/'tools/backlog.exe'),'task','edit','ASR-18','--append-notes',note],cwd=str(S),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
(F/'added_buffer_timeout156769_edit.stdout.txt').write_bytes(edit.stdout);(F/'added_buffer_timeout156769_edit.stderr.txt').write_bytes(edit.stderr)
count=TASK.read_text(encoding='utf-8-sig').count(MARKER)
write(F/'added_buffer_timeout156769_edit_result.json',{'actual_exit_code':edit.returncode,'canonical_marker_count_after_edit':count})
assert edit.returncode==0 and count==1,'Preserve actual edit outcome; do not repeat ambiguously'
refresh=subprocess.run([str(P/'work/training-env/Scripts/python.exe'),str(S/'refresh_backlog.py')],cwd=str(P),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
(F/'added_buffer_timeout156769_refresh.stdout.txt').write_bytes(refresh.stdout);(F/'added_buffer_timeout156769_refresh.stderr.txt').write_bytes(refresh.stderr)
write(F/'added_buffer_timeout156769_refresh_result.json',{'actual_exit_code':refresh.returncode})
count=TASK.read_text(encoding='utf-8-sig').count(MARKER)
assert refresh.returncode==0 and count==1,'Preserve actual refresh outcome; edit already completed'
write(RESULT,{'kind':'ASR18_actual_added_buffer_timeout156769_saved156600_required_action_Backlog_milestone_v1','marker':MARKER,'actual_task_edit_exit_code':edit.returncode,'actual_refresh_exit_code':refresh.returncode,'canonical_marker_count_before':0,'canonical_marker_count_after_edit_and_refresh':count,'ROOT_failure_review':pin(ROOT),'readonly_findings':pin(FINDINGS),'failure_closure':pin(F/'actual_failure_closure.json'),'note':pin(NOTE),'training_stopped':True,'human_policy_direction_pending':True,'training_source_config_guards_or_environment_changed':False,'duplicate_old_quality_or_recovery_note':False})
print(json.dumps({'passed':True,'receipt':pin(RESULT),'actual_task_edit_exit_code':edit.returncode,'actual_refresh_exit_code':refresh.returncode,'canonical_marker_count':count}),flush=True)

