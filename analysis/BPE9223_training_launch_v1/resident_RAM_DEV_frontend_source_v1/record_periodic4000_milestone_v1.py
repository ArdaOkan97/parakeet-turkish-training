"""One actual fixed350 milestone via canonical CLI; does not touch training."""
import json,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
P=Path(r'C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
N=Path(r'E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1');A=N/'resident_RAM_DEV_frontend_source_v1'
status_pin=record(N/'resident_RAM_training_actual_04/actual_periodic4000_and_resumed_training_status.json')
status=read_json(status_pin['path']);entry=status['periodic_result'];score=entry['score']
require(entry['optimizer_updates']==4000 and score['word_errors']['tr']==372 and score['word_errors']['pt']==67
    and score['strict_all_six_passed'] is False and status['actual_positive_updates_at_observation']>=4020,
    'Only genuine full350 result with real continuation')
marker='BPE9223_PERIODIC4000_FULL350_STATE_PRESERVED_RESUMED_20261019'
note=(marker+' '+datetime.now(timezone.utc).isoformat()+': First complete periodic fixed350 DEV at actual4000, '
    'using the original seven50-clip validation sets, saved official scorer and integer retention gates; TEST untouched. '
    'Turkish372/825=45.090909% WER. Retention errors/references: EN69/1045, ES51/1289, IT22/1158, FR68/1072, DE67/1004, PT67/1095. '
    'All-six gate fails only Portuguese67>66; this periodic monitor does not qualify a winner, end a pass, alter replay/teacher/augmentation, or stop training. '
    'Actual DEV81.140s, resident private peak4,824,973,312B within measured before3,300,306,944+2.1GB allowance; '
    'CUDA lifetime peak after DEV1,610,139,648B. All207 trainable weights, three Adam states, all4 RNG families, both cursors, journal, next draw and full-pass/winner state unchanged; zero native updates/draws during DEV. '
    'The preceding source3 first-row TokenOnlyCarrier frontend failure, actual TOOL1, saved4000/all854 and original failed result directories are preserved. '
    'Source4/helper2 now composes the real original local AutoProcessor with the identical live9223 tokenizer/blank8192; a real CPU fixed-row inputs_for test verified exact frontend features/masks, with only final CUDA transfer intercepted. '
    'Actual fresh4000 restore passed before this evaluation; same whole commits_000004 lineage reused without copy/truncation/tail adoption. '
    'Training genuinely resumed to at least4020, both poison flags false, session45407/native2956/ctime1792395250.2468245, source4 SHA807269145a34dbef4ea299c2c499bb76be8591bd87cd95050c5f62ab91a5bce2. '
    'Full401072 automatic continuation remains active, checkpoint every200, next periodic8000, no finite administrative stop. '
    'Exact evidence: '+status_pin['path']+' SHA'+status_pin['sha256']+'; DEV '+entry['DEV']['sha256']+'; frontier preservation '+entry['frontier_preservation']['sha256']+'.')
binary=S/'tools/backlog.exe'
def cli(*args):return subprocess.run([str(binary),*args],cwd=S,capture_output=True,text=True,encoding='utf-8',check=True)
current=json.loads(cli('task','view','ASR-18','--json').stdout)['task']
require(marker not in current.get('implementationNotes',''),'Exactly one milestone, no duplicate append')
added=cli('task','edit','ASR-18','--append-notes',note,'--add-ref',status_pin['path'],'--add-ref',entry['DEV']['path'])
refresh=subprocess.run([str(P/'work/training-env/Scripts/python.exe'),str(S/'refresh_backlog.py')],cwd=P,capture_output=True,text=True,encoding='utf-8',check=True)
result={'kind':'ASR18_actual_periodic4000_CLI_milestone_and_refresh_v1','marker':marker,'task':'ASR-18',
    'actual_task_edit_exit_code':added.returncode,'actual_refresh_exit_code':refresh.returncode,
    'evidence':status_pin,'training_source_or_environment_changed':False,'duplicate_note_appended':False}
write_json(A/'actual_backlog_milestone_result.json',result)
print(json.dumps({'passed':True,'result':record(A/'actual_backlog_milestone_result.json')}))
