"""One deduplicated official CLI milestone; no training action."""
import json,subprocess,sys
from pathlib import Path
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
N=Path(__file__).parent;marker='BPE9223_PERIODIC36000_FULL350_32P72727_STATE_PRESERVED_RESUMED_20261019'
status=N/'actual_periodic36000_and_resumed_training_status.json';r=read_json(status)
root_review=N.parent/'ROOT_periodic36000_and_resumed_training_verification_v1.json';require(root_review.exists(),'ROOT actual quality acceptance exists')
cli=S/'tools/backlog.exe';view=subprocess.run([str(cli),'task','view','ASR-18','--json'],cwd=S,capture_output=True,text=True,encoding='utf-8');require(view.returncode==0,'Actual official task view')
require(marker not in view.stdout,'New distinct marker; do not duplicate an existing milestone')
note=(f'[{marker}]\nActual periodic DEV at optimizer update 36,000 completed all fixed seven-language 50-clip sets (350 examples) with unchanged manifests, scorer and decoding. Turkish WER is 270/825 = 32.72727%, compared with 274/825 = 33.21212% at 32,000: four fewer word errors and 0.48485 percentage points improvement. Retention counts are EN 62/1045, ES 56/1289, IT 21/1158, FR 73/1072, DE 73/1004 and PT 63/1095. Spanish exceeds cap52 by4 and German exceeds cap70 by3; all-six retention is false. No qualified winner or full-pass advancement, retention early stop or TEST use is claimed.\n'
'DEV elapsed82.484s with sampled native-owner private peak5,908,594,688bytes. All207 trainable tensors, all three Adam optimizers, all four RNG states, both cursors/replay, journal/nextdraw and exact pending plan181 were preserved; DEV consumed zero native updates or authoritative draws. Checkpoint36,000 passed854-tensor/215-Adam-state persistence. The same admitted source8/native35964/session63811 genuinely resumed (immutable observation36,080), both poison flags false. Full401072/adminNone automatic continuation remains active, next periodic40,000. No source, environment, child, resource or numerical change occurred during DEV.\n'
f'Evidence: {status}\nROOT verification: {root_review}\nDEV: {r["periodic_result"]["DEV"]["path"]}\n')
note_path=N/'periodic36000_milestone_note.txt';require(not note_path.exists(),'One note');note_path.write_text(note,encoding='utf-8')
edit=subprocess.run([str(cli),'task','edit','ASR-18','--status','In Progress','--append-notes',note],cwd=S,capture_output=True,text=True,encoding='utf-8');(N/'periodic36000_task_edit.log').write_text(edit.stdout+edit.stderr,encoding='utf-8');require(edit.returncode==0,'Actual task edit0')
refresh=subprocess.run([str(P/'work/training-env/Scripts/python.exe'),str(S/'refresh_backlog.py')],cwd=P,capture_output=True,text=True,encoding='utf-8');(N/'periodic36000_refresh.log').write_text(refresh.stdout+refresh.stderr,encoding='utf-8');require(refresh.returncode==0,'Actual board refresh0')
verify=subprocess.run([str(cli),'task','view','ASR-18','--json'],cwd=S,capture_output=True,text=True,encoding='utf-8');require(verify.returncode==0 and verify.stdout.count(marker)==1,'Exactly one canonical marker after edit')
receipt={'kind':'ASR18_actual_periodic36000_CLI_milestone_and_refresh_v1','marker':marker,'deduplicated_existing_marker':False,'actual_task_edit_exit_code':edit.returncode,'actual_refresh_exit_code':refresh.returncode,'status':record(status),'ROOT_actual_verification':record(root_review),'note':record(note_path),'canonical_marker_count_after_edit':1,'training_changes':False,'duplicate_prior_note':False}
path=N/'actual_periodic36000_backlog_result.json';write_json(path,receipt);print(json.dumps({'receipt':record(path),'task_edit_exit':edit.returncode,'refresh_exit':refresh.returncode,'marker_unique':True}),flush=True)
