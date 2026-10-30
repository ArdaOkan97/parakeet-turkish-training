"""INACTIVE typed saved156600 entry proof; no audio/model/NumPy/Torch/GPU.
Reads only the preserved169-tail COPY, compact/source records and original journal STAT.
Requires final completed-CPU-bound configuration and separate exact ROOT scope.
"""
import argparse,copy,json,os,shutil,sys,time
from pathlib import Path
import psutil
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish';sys.path[:0]=[str(S),str(P/'work/research')]
from bpe_broad_audio_v1 import record,require
from atomic_io_retry_v1 import read_json,write_json
from resource_watchdog_v2 import sample_memory
F=Path(__file__).parent;N=F.parents[1];B=Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01')
began=time.monotonic();peak=0
def guard():
 global peak
 s=sample_memory(os.getpid(),'readonly_saved156600_cushion225_typed_metadata');m=s['system_memory'];peak=max(peak,s['worker_private_memory']['private_bytes'])
 require(s['paired_sampling_within_one_second']and peak<512*1024**2
  and m['commit_available_bytes']>=2000000000+268435456 and m['physical_available_bytes']>=1500000000+268435456
  and shutil.disk_usage('E:/').free>=100000000000 and time.monotonic()-began<180,'Original512MiB/180s/full256MiB/headroom/E100GB')
 return s
def main(a):
 psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
 require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'and all(os.environ.get(k)=='1'for k in('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')),'CPU-only fixed thread masks')
 require(Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad')),'E storage applied')
 initial=guard();m=initial['system_memory']
 require(m['commit_available_bytes']>=2000000000+268435456+512*1024**2 and m['physical_available_bytes']>=1500000000+268435456+512*1024**2,'Full prospective512MiB metadata allowance')
 scope=read_json(a.ROOT_review);configuration=record(a.configuration)
 require(scope['accepted']is True and scope['reviewer']=='/root'
  and scope['scope']=='ASR18_readonly_saved156600_cushion225_typed_metadata_v1'
  and scope['typed_fixture_source']==record(__file__)and scope['source_configuration']==configuration,'Exact separate ROOT typed scope')
 for item in scope['input_pins']:require(record(item['path'])==item,'Exact scoped typed input before execution')
 output=Path(a.output).resolve();require(output.is_relative_to(F)and not output.exists(),'ONE fresh typed metadata E output');output.mkdir()
 me=psutil.Process();parent=me.parent();identities={'producer':{'pid':me.pid,'create_time':me.create_time(),'ppid':me.ppid()},'launcher':{'pid':parent.pid,'create_time':parent.create_time(),'ppid':parent.ppid()}}
 write_json(output/'actual_process_entry.json',identities)
 from run_ASR18_BPE9223_resident_RAM_training_v26 import checked_cushion225_resume,checked_native_commit_pressure_policy
 from fastmss_resident_feature_stream_v10 import checked_resource_profile,checked_packet_credit_policy
 base=read_json(a.configuration);old_path=N/'resident_RAM_native_pressure_timeout156537_failure_actual_v1/inactive_packet_credit_v1/production_configuration_v1/configuration.json'
 old=read_json(old_path);restored=copy.deepcopy(base)
 del restored['prior_resident_RAM_native_pressure156769_failure'];restored['native_update_commit_pressure_policy']=old['native_update_commit_pressure_policy']
 require(restored==old,'Proper source24 base: only native-pressure policy and new typed failure fields')
 checked_resource_profile(base);credit=checked_packet_credit_policy(base);policy,_=checked_native_commit_pressure_policy(base)
 require(policy['parameters']['additional_native_commit_pressure_buffer_bytes']==2250000000
  and base['resident_RAM_resource_scheduling_policy']['CPU_prospective_commit_reserve_bytes']==2500000000,'Approved native cushion only; CPU reserve unchanged')
 pointer=record(B/'checkpoints/update_000156600/pointer.json')
 validated=checked_cushion225_resume(base['prior_resident_RAM_native_pressure156769_failure'],base=base,folder=B,pointer_pin=pointer)
 require(validated['saved_journal_prefix']['offset']==317194933 and len(validated['preserved_tail_positive_updates'])==169,'Exact saved156600 plus169 historical tail')
 require(not(B/'invocations/owner_000019').exists()and not(B/'journals/commits_000019.jsonl').exists()and not(N/'resident_RAM_training_actual_20').exists(),'No future owner/output/journal before final execution')
 runtime={'worker_driver_source':record(P/'work/research/run_ASR18_BPE9223_resident_RAM_training_v26.py'),'parent_driver_source':record(P/'work/research/run_ASR18_BPE9223_resident_RAM_training_v26.py'),
  'resident_RAM_stream_configuration':configuration,'resident_branch_namespace':str(B),'checkpoint_interval_updates':200,'source_checkpoint':pointer,'source_configuration':configuration}
 require('numpy'not in sys.modules and 'torch'not in sys.modules,'No NumPy/Torch/model/GPU imports')
 for item in scope['input_pins']:require(record(item['path'])==item,'Exact scoped typed input after execution')
 write_json(output/'report.json',{'kind':'ASR18_actual_readonly_saved156600_cushion225_typed_metadata_v1','passed':True,'actual_training_native_audio_sampler_journal_operations':0,
  'actual_kernel_credit_objects_created':0,'only_two_operational_config_fields':True,'all_old_source24_config_values_exact':True,'source_configuration':configuration,
  'saved_pointer':pointer,'pressure_policy':policy,'completed_packet_credit_policy':credit,'saved_prefix':validated['saved_journal_prefix'],
  'original18_closed_journal_STAT':validated['whole_journal_including_preserved_tail'],'preserved169tail':validated['preserved_unsaved_tail'],
  'original_whole_or_prefix_journal_read_in_this_control':False,'only_preserved_tail_copy_bytes_read':330892,'expected_future_7_runtime_bindings_NOT_actual_boot':runtime,
  'future_output_absent':True,'numpy_or_torch_imported':False,'identities':identities,'guard_initial':initial,'guard_final':guard(),'peak_private_bytes':peak,
  'input_pins_before_after_exact':scope['input_pins'],'no_restore_prefixcopy_launch_or_production_admission':True})
 print(json.dumps({'report':record(output/'report.json'),'identities':identities}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--ROOT-review',required=True);p.add_argument('--configuration',required=True);p.add_argument('--output',required=True);main(p.parse_args())

