"""INACTIVE one read-only typed source22 metadata check; no model/audio/GPU."""
import argparse,json,os,shutil,sys,time
from pathlib import Path
import psutil
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish';sys.path[:0]=[str(S),str(P/'work/research')]
from resource_watchdog_v2 import sample_memory
from bpe_broad_audio_v1 import record,require
from atomic_io_retry_v1 import read_json,write_json
ROOT=Path(r'E:\gpu_training_scratchpad')
def main(a):
 from asr18_memory_diagnosis_v2 import storage
 storage();require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'and all(os.environ.get(k)=='1'for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')),'Exact CPU-only metadata environment')
 began=time.monotonic();psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
 def guard():
  s=sample_memory(os.getpid(),'source22_saved156200_typed_metadata_control');m=s['system_memory']
  require(s['paired_sampling_within_one_second']and s['worker_private_memory']['private_bytes']<512*1024**2 and m['commit_available_bytes']>=2000000000+268435456 and m['physical_available_bytes']>=1500000000+268435456 and shutil.disk_usage(ROOT).free>=100000000000 and time.monotonic()-began<180,'Original metadata/headroom/full256MiB/E guards')
  return s
 initial=guard();require(initial['system_memory']['commit_available_bytes']>=2000000000+268435456+512*1024**2 and initial['system_memory']['physical_available_bytes']>=1500000000+268435456+512*1024**2,'Full512MiB metadata prospective admission')
 handoff_pin=record(a.handoff);handoff=read_json(a.handoff);scope_pin=record(a.ROOT_review);scope=read_json(a.ROOT_review)
 require(scope['accepted']is True and scope['reviewer']=='/root'and scope['scope']=='ASR18_native_pressure_saved156200_typed_metadata_control_v1'
  and scope['source_handoff']==handoff_pin and scope['metadata_control_source']==record(__file__),'Separate exact ROOT typed metadata control scope')
 out=Path(a.output).resolve();require(out.is_relative_to(ROOT)and not out.exists(),'One fresh E metadata output');out.mkdir()
 me=psutil.Process();parent=me.parent()
 identities={'producer':{'pid':me.pid,'create_time':me.create_time(),'ppid':me.ppid()},'launcher':None if parent is None else {'pid':parent.pid,'create_time':parent.create_time(),'ppid':parent.ppid()}}
 write_json(out/'actual_process_entry.json',identities)
 before={}
 for key,pin in handoff['typed_metadata_inputs'].items():
  require(record(pin['path'])==pin,'Exact frozen metadata input '+key);before[key]=pin;guard()
 config=read_json(before['source_configuration']['path']);folder=Path(handoff['branch'])
 pointer=before['pointer'];failure=before['actual_failure_closure']
 journal=Path(handoff['saved_prefix']['path']);journal_stat_before=journal.stat().st_size
 absence={'invocation':folder/'invocations/owner_000017','journal':folder/'journals/commits_000017.jsonl','GPU_output':Path(handoff['future_GPU_output'])}
 require(all(not path.exists()for path in absence.values()),'No future GPU owner/journal/output before metadata check')
 from run_ASR18_BPE9223_resident_RAM_training_v22 import checked_native_commit_pressure_policy,checked_current_native_headroom_resume
 policy,gate=checked_native_commit_pressure_policy(config)
 validated=checked_current_native_headroom_resume(failure,base=config,folder=folder,pointer_pin=pointer)
 require(validated['saved_optimizer_updates']==156200 and validated['saved_journal_prefix']==handoff['saved_prefix']
  and validated['preserved_tail_positive_updates']==list(range(156201,156301))
  and policy==config['native_update_commit_pressure_policy'] and callable(gate),'Exact typed saved156200/100-tail/policy bindings only')
 require('torch'not in sys.modules and 'numpy'not in sys.modules,'No model/NumPy/Torch/GPU imports')
 require(journal.stat().st_size==journal_stat_before==handoff['whole_old_journal']['bytes']
  and all(not path.exists()for path in absence.values()),'Original journal stat and absent future names unchanged')
 after={key:record(pin['path'])for key,pin in before.items()};require(after==before,'All frozen inputs unchanged after actual metadata check')
 result={'kind':'ASR18_actual_source22_saved156200_readonly_typed_metadata_control_v1','passed':True,
  'ROOT_scope':scope_pin,'source_handoff':handoff_pin,'metadata_control_source':record(__file__),
  'actual_process_identities':identities,'inputs_before':before,'inputs_after':after,
  'saved_optimizer_updates':156200,'saved_journal_prefix':validated['saved_journal_prefix'],
  'whole_old_journal_STAT_only':{'path':str(journal),'bytes':journal_stat_before},
  'preserved_unsaved_positive_tail':validated['preserved_unsaved_tail'],'preserved_tail_updates':list(range(156201,156301)),
  'historical_outer_TOOL1_preserved':True,'unknown_native_CPU_wrapper_numeric_OS_exits_not_invented':True,
  'historical_failed_monitor_clean_false_preserved':True,'native_pressure_policy':policy,
  'new_invocation_journal_GPU_output_absent_before_after':{key:str(path)for key,path in absence.items()},
  'read_scope':'Only exact compact source/config/review/failure metadata and196884B preserved100 tail; old journal STAT only, no whole journal/prefix/tensor/audio read.',
  'actual_native_optimizer_updates':0,'actual_sampler_draws':0,'actual_journal_writes':0,'actual_model_audio_or_GPU':False,
  'no_production_state_or_environment_mutation':True,'guard_initial':initial,'guard_final':guard()}
 write_json(out/'report.json',result);print(json.dumps({'report':record(out/'report.json'),'identities':identities}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--handoff',required=True);p.add_argument('--ROOT-review',required=True);p.add_argument('--output',required=True);main(p.parse_args())
