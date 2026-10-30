"""One truthful final tokenizer milestone; existing canonical CLI then refresh."""
import importlib,json,subprocess,sys
from pathlib import Path
D=Path(r'E:\gpu_training_scratchpad\analysis\Turkish_labels_BPE_circumflex_v1')
sys.path.insert(0,str(D));m=importlib.import_module('audit_and_fit_append_only_v5')
OUT=m.PROJECT/'outputs/parakeet_turkish';m.AREA=D/'final_backlog_milestone_v2';m.AREA.mkdir(exist_ok=True)

def main():
 g=m.Guard();g.phase='single_meaningful_completed_BPE_tokenizer_Backlog_milestone'
 marker='BPE9223_FINAL_ALL_TRAIN_20261018'
 task=OUT/'backlog/tasks/asr-18 - Train-BPE-on-broad-weak-labels-then-fine-tune-with-multilingual-retention.md'
 assert marker not in task.read_text(encoding='utf8')
 assert json.loads((D/'ROOT_final_BPE9223_delivery_review_v1.json').read_text())['accepted_for_requested_tokenizer_delivery']
 note=(m.now()+': '+marker+' — CPU-only corpus-driven tokenizer candidate finalized and independently ROOT reviewed. '
  'All 2,983,883 available Turkish TRAIN clip occurrences (2,631,115 raw broad weak-label +352,768 cleaner) '
  'were audited against actual existing source text/label IDs and used at their observed frequencies; no DEV/TEST, '
  'quality filters, transcript corrections or synthetic capitalization. Candidate9223 retains every existing8711ID, '
  '13988 old merge ranks, normalizer/case/special settings and adds512 positive-frequency Latin merges; '
  'all512 emit in actual TRAIN text. No forced uppercaseÛ: raw/normalizedTRAIN count0. '
  'Full actual Rust validation CLOSED0/tool74dc28: 113,443,643→101,368,083 tokens (10.644545% fewer), '
  'same decoded text and unknown counts on all rows,33064 existingUNK rows remain. '
  'Actual1914 six-language TRAIN replay comparison:76 encoding changes,0 decoded-text changes; '
  'existing protected replay labels were NOT rewritten and retention is not inferred from stable IDs. '
  'Fit CLOSED0/tool982907, exact256-prefix replay,28.64s/374.29MBpeak; full validation416.375s/65.27MBpeak; '
  'ZIP/CRC/byte/ID/merge independent seal CLOSED0/tooldbc6c4. '
  'Final tokenizer SHA f2d80ffee7ed1b006a83b52cc77bfb28bed4ba850352b5222bb3a49d6cfbac4a; '
  'package E:/gpu_training_scratchpad/analysis/Turkish_labels_BPE_circumflex_v1/Turkish_all_TRAIN_BPE9223_candidate_v2.zip '
  'SHA9b513b590ad30a0e9c7be336876e114029a1ef66320875f9195ce61ef6faeb79. '
  'Finalized handoff finalized_delivery_handoff_v3.json; ROOT review ROOT_final_BPE9223_delivery_review_v1.json. '
  'Tokenizer candidate only: no GPU/model/active tokenizer/config/label mutations or ASR quality gain. '
  'Current8711 pipeline remains unchanged; explicit head/embedding/optimizer migration, new TRAIN catalogs/means/geometry '
  'and six-language retention validation are required before adoption. Optional8712 manualÛ coverage patch remains separate, untrained and unadopted.')
 cli=subprocess.run([str(OUT/'tools/backlog.exe'),'task','edit','ASR-18','--append-notes',note],cwd=OUT,capture_output=True,text=True,encoding='utf8')
 (m.AREA/'actual_cli_output.txt').write_text(cli.stdout+'\n'+cli.stderr,encoding='utf8')
 assert cli.returncode==0,cli.stderr
 g.check()
 refresh=subprocess.run([str(m.PROJECT/'work/training-env/Scripts/python.exe'),str(OUT/'refresh_backlog.py')],cwd=m.PROJECT,capture_output=True,text=True,encoding='utf8')
 (m.AREA/'actual_refresh_output.txt').write_text(refresh.stdout+'\n'+refresh.stderr,encoding='utf8')
 assert refresh.returncode==0,refresh.stderr
 assert task.read_text(encoding='utf8').count(marker)==1
 result={'kind':'Actual_final_BPE_completion_Backlog_milestone_v1','at_utc':m.now(),'marker':marker,
  'task':'ASR-18','CLI_actual_exit0':True,'refresh_actual_exit0':True,'marker_exactly_once':True,
  'CLI_output':m.record(m.AREA/'actual_cli_output.txt'),'refresh_output':m.record(m.AREA/'actual_refresh_output.txt'),
  'final_delivery_handoff':m.record(D/'finalized_delivery_handoff_v3.json'),'no_model_or_data_jobs':True}
 m.write('actual_Backlog_CLI0_refresh0.json',result);g.phase='completion_Backlog_recorded_once';g.publish(passed=True)
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
