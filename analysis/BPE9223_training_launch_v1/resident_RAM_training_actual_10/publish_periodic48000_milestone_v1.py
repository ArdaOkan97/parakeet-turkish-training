"""One independently accepted periodic quality CLI note; no training action."""
import json,subprocess,sys
from pathlib import Path
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
N=Path(__file__).parent;marker='BPE9223_PERIODIC48000_FULL350_30P30303_STATE_PRESERVED_RESUMED_20261020'
status=N/'actual_periodic48000_and_resumed_training_status.json';r=read_json(status)
root_review=N.parent/'ROOT_periodic48000_and_resumed_training_verification_v1.json'
require(record(root_review)['sha256']=='3f34a93c7f94e266649fb2f55bba6dfccc78fe0ab8a3f9279602a527cf9d63a8','Exact ROOT quality acceptance')
review=read_json(root_review);require(review['accepted'] is True,'ROOT accepted actual full350 and resumed work')
cli=S/'tools/backlog.exe';view=subprocess.run([str(cli),'task','view','ASR-18','--json'],cwd=S,capture_output=True,text=True,encoding='utf-8');require(view.returncode==0 and marker not in view.stdout,'Canonical task view and no duplicate marker')
note=(f'[{marker}]\nActual periodic DEV at optimizer update 48,000 completed all fixed seven-language 50-clip sets (350 examples), using the original manifests, scorer and decoding. Turkish WER is 250/825 = 30.30303%, compared with 248/825 = 30.06061% at 44,000: two more word errors and 0.24242 percentage points increase. Retention counts are EN 60/1045, ES 56/1289, IT 21/1158, FR 70/1072, DE 72/1004 and PT 62/1095. Spanish exceeds cap 52 by 4 and German exceeds cap 70 by 2; French is back within cap 74. All-six retention remains false. No qualified winner, full-pass advancement, retention early stop or TEST use is claimed.\n'
f'DEV elapsed {r["DEV_wall_seconds"]:.3f} s with sampled native-owner private peak {r["DEV_private_peak_bytes_native_owner"]:,} bytes. All 207 trainable tensors, all three Adam optimizers, all four RNG states, both cursors/replay, journal/next draw and exact pending plan 21 were preserved. DEV consumed zero native updates or authoritative draws. Checkpoint 48,000 passed 854-tensor/215-Adam-state persistence. Same admitted source10/native8252/session62574 genuinely resumed (immutable observation {r["actual_optimizer_updates_at_observation"]:,}), both poison flags false. Full 401072/adminNone automatic continuation remains active; next periodic 52,000. Training source, environment, child and resource settings were unchanged during DEV.\n'
f'Evidence: {status}\nROOT verification: {root_review}\nDEV: {r["periodic_result"]["DEV"]["path"]}\n')
note_path=N/'periodic48000_milestone_note.txt';require(not note_path.exists(),'One note');note_path.write_text(note,encoding='utf-8')
edit=subprocess.run([str(cli),'task','edit','ASR-18','--status','In Progress','--append-notes',note],cwd=S,capture_output=True,text=True,encoding='utf-8');(N/'periodic48000_task_edit.log').write_text(edit.stdout+edit.stderr,encoding='utf-8');require(edit.returncode==0,'Actual task edit0')
refresh=subprocess.run([str(P/'work/training-env/Scripts/python.exe'),str(S/'refresh_backlog.py')],cwd=P,capture_output=True,text=True,encoding='utf-8');(N/'periodic48000_refresh.log').write_text(refresh.stdout+refresh.stderr,encoding='utf-8');require(refresh.returncode==0,'Actual board refresh0')
verify=subprocess.run([str(cli),'task','view','ASR-18','--json'],cwd=S,capture_output=True,text=True,encoding='utf-8');require(verify.returncode==0 and verify.stdout.count(marker)==1,'Exactly one canonical marker after edit')
receipt={'kind':'ASR18_actual_periodic48000_CLI_milestone_and_refresh_v1','marker':marker,'deduplicated_existing_marker':False,'actual_task_edit_exit_code':edit.returncode,'actual_refresh_exit_code':refresh.returncode,'status':record(status),'ROOT_actual_verification':record(root_review),'note':record(note_path),'canonical_marker_count_after_edit':1,'training_changes':False,'duplicate_prior_note':False}
path=N/'actual_periodic48000_backlog_result.json';write_json(path,receipt);print(json.dumps({'receipt':record(path),'task_edit_exit':edit.returncode,'refresh_exit':refresh.returncode,'marker_unique':True}),flush=True)
