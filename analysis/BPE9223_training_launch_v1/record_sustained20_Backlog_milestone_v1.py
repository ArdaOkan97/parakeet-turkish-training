"""Exactly one truthful completed sustained20 milestone; no model/data job."""
import importlib,json,subprocess,sys
from pathlib import Path
L=Path(r'E:\gpu_training_scratchpad\analysis\Turkish_labels_BPE_circumflex_v1');sys.path.insert(0,str(L));m=importlib.import_module('audit_and_fit_append_only_v5')
N=Path(r'E:\gpu_training_scratchpad\analysis\BPE9223_training_launch_v1');O=N/'sustained20_Backlog_milestone_v1';O.mkdir(exist_ok=False);m.AREA=O
def main():
 g=m.Guard();g.phase='single_completed_sustained20_Backlog_milestone'
 path=N/'sustained20_CPU_native_actual_01/actual_sustained20_summary_v1.json';assert m.sha(path)=='b6e9f566bb0c1c1a591cf37719652c6cae8ad191e5d1f2a7859248a61c7ee541'
 summary=json.loads(path.read_text());actualpin=summary['actual_tool0'];assert m.sha(actualpin['path'])=='3168a2045e980dcbe1db059ea5c2f4950c508dfdaacb2e283c20bba3a5b140c3'
 actual=json.loads(Path(actualpin['path']).read_text());assert actual['actual_exec_tool_exit_code']==actual['actual_native_exit_code']==0 and actual['all_owned_CPU_native_fresh_processes_closed0'] is True
 assert summary['passed'] is True and summary['actual_native_updates']==20 and summary['TR_rows']==280 and summary['replay_rows']==40 and summary['named_tensors_verified']==854 and summary['Adam_states_verified']==215
 assert summary['fresh_all207_threeAdam_all4RNG_sampler_journal_nextdraw_exact'] is True and summary['production_enabled'] is False
 out=m.PROJECT/'outputs/parakeet_turkish';task=out/'backlog/tasks/asr-18 - Train-BPE-on-broad-weak-labels-then-fine-tune-with-multilingual-retention.md'
 marker='BPE9223-sustained20-20261018-v1';assert marker not in task.read_text(encoding='utf8')
 note=(m.now()+': '+marker+' — Original-source BPE9223 sustained diagnostic actually completed: one new CPU window,280 real full Turkish rows plus40 protected carried replay rows,20 positive native all207 updates, native exit0, separate fresh recovery0 and outer TOOL0. '
  'All854 named checkpoint tensors/215 mature+appended Adam states, all4 RNG families, both scientific samplers, journal prefix and provisional next draw exactly recovered. '
  'Appended512 Adam clock20; inherited source clocks64270/2020/62270. Original BPE64250 plus qualified parkedLoRA62250 sources, not diagnostic seeds. '
  'Native cold update6.359s; updates2–20 mean1.788684s/median1.765s. Full CPU preparation129.906s; native startup/training/checkpoint/exit83.188s; fresh recovery28.094s; whole owned parent241.797s. '
  'Recorded noise/16-room augmentation and full inputs/targets included in CPU preparation, protected replay/teacher unchanged. Timings are this finite window, not corpus throughput or launch ETA. '
  'Truthful diagnostic-only result: no measured ASR quality gain, qualified winner, production ancestry or full-pass stability claim. '
  'Remaining actual launch checks include full retained hard9223 row/current lifecycle, finite serial350DEV, complete production rollover/controller/config review. '
  'Evidence E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1/sustained20_CPU_native_actual_01/actual_sustained20_summary_v1.json; actual0 tool443d94.')
 cli=subprocess.run([str(out/'tools/backlog.exe'),'task','edit','ASR-18','--append-notes',note],cwd=out,capture_output=True,text=True,encoding='utf8')
 (O/'actual_CLI_output.txt').write_text(cli.stdout+'\n'+cli.stderr,encoding='utf8')
 refresh=None
 if cli.returncode==0:
  refresh=subprocess.run([str(m.PROJECT/'work/training-env/Scripts/python.exe'),str(out/'refresh_backlog.py')],cwd=m.PROJECT,capture_output=True,text=True,encoding='utf8')
  (O/'actual_refresh_output.txt').write_text(refresh.stdout+'\n'+refresh.stderr,encoding='utf8')
 count=task.read_text(encoding='utf8').count(marker);assert cli.returncode!=0 or count==1
 report={'kind':'ASR18_one_completed_sustained20_Backlog_milestone_actual_outcome_v1','at_utc':m.now(),'task':'ASR-18','marker':marker,'note_occurrences':count,
  'CLI_actual_exit_code':cli.returncode,'refresh_actual_exit_code':refresh.returncode if refresh else None,
  'CLI_output':m.record(O/'actual_CLI_output.txt'),'refresh_output':m.record(O/'actual_refresh_output.txt') if refresh else None,
  'actual_sustained_summary':m.record(path),'actual_native_outer0':actualpin,'no_new_model_data_audio_GPU_job':True}
 (O/'actual_Backlog_CLI_refresh_outcome_v1.json').write_text(json.dumps(report,indent=2)+'\n');g.publish(completed=True,receipt=m.record(O/'actual_Backlog_CLI_refresh_outcome_v1.json'))
 print(json.dumps(report));assert cli.returncode==0
if __name__=='__main__':main()
