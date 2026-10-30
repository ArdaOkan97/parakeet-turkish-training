"""One readonly serialized typed gate; no model/audio/GPU/sampler/journal writes."""
import argparse,os,shutil,sys,time
from pathlib import Path
import psutil
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish';R=P/'work/research';sys.path[:0]=[str(S),str(R)]
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
from resource_watchdog_v2 import sample_memory
from run_ASR18_BPE9223_resident_RAM_training_v20 import checked_original_interval_resume
from fastmss_resident_feature_stream_v9 import checked_resource_profile
a=argparse.ArgumentParser();a.add_argument('--configuration',required=True);a.add_argument('--output',required=True);args=a.parse_args()
began=time.monotonic();out=Path(args.output);require(not out.exists()and out.is_relative_to(Path(r'E:\gpu_training_scratchpad')),'One fresh E metadata output');out.mkdir()
process=psutil.Process();process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);who={'pid':process.pid,'create_time':process.create_time(),'ppid':process.ppid()}
write_json(out/'entry_identity.json',who)
require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'and sys.flags.utf8_mode==1,'CPU-only metadata with upfront UTF8')
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):require(os.environ.get(key)=='1','Four thread caps1')
def guard():
 v=sample_memory(os.getpid(),'exact_saved142800_readonly_metadata_gate');s=v['system_memory']
 require(v['paired_sampling_within_one_second']and v['worker_private_memory']['private_bytes']<512*1024**2
  and s['commit_available_bytes']>=2000000000+268435456 and s['physical_available_bytes']>=1500000000+268435456
  and shutil.disk_usage('E:/').free>=100000000000 and time.monotonic()-began<180,'Unchanged original headroom plus full extra256MiB/private512MiB/wall180/E100GB')
 return v
initial=guard();require(initial['system_memory']['commit_available_bytes']>=2000000000+268435456+512*1024**2
 and initial['system_memory']['physical_available_bytes']>=1500000000+268435456+512*1024**2,'Full metadata prospective512MiB plus unchanged headroom/full256MiB')
config_pin=record(args.configuration);config=read_json(args.configuration);review=read_json(config['original_clip_geometry_overlay']['ROOT_review']['path'])
require(record(config['original_clip_geometry_overlay']['ROOT_review']['path'])==config['original_clip_geometry_overlay']['ROOT_review']
 and review['accepted']is True and review['reviewer']=='/root'
 and review['scope']=='ASR18_source_bound_seventh_original_full_clip_geometry_overlay_v1'
 and review['overlay_source']==config['original_clip_geometry_overlay']['source']and review['certificate']==config['original_clip_geometry_overlay']['certificate'],
 'Exact completed CPU scoped review/config/overlay/certificate binding')
checked_resource_profile(config);guard()
failure_pin=config['prior_resident_RAM_original_interval_failure'];failure=read_json(failure_pin['path']);pointer=failure['inputs']['pointer']
result=checked_original_interval_resume(failure_pin,base=config,folder=Path(r'E:\gpu_training_scratchpad\runs\BPE9223_resident_RAM_20261018_01'),pointer_pin=pointer)
require(result['saved_optimizer_updates']==142800 and result['preserved_tail_positive_updates']==list(range(142801,142895)),
 'Actual serialized exact typed saved142800 and all94 retained positives')
require('torch'not in sys.modules and 'numpy'not in sys.modules,'Metadata only: no model/frontend/NumPy runtime')
write_json(out/'report.json',{'kind':'ASR18_actual_saved142800_typed_metadata_gate_v1','passed':True,'producer':who,'source':record(__file__),
 'configuration':config_pin,'failure':failure_pin,'pointer':pointer,'saved_updates':142800,'preserved_historical_tail_updates':94,
 'remaining_schema_exact':True,'saved_prefix_only_copy_after_genuine_GPU_restore_required':True,'actual_GPU_restore_claimed':False,
 'no_model_audio_GPU_sampler_or_journal_operation':True,'last_resource_sample':guard(),'elapsed_seconds':time.monotonic()-began,
 'actual_child_exit_unobserved_inside_child':True})
print(record(out/'report.json'),flush=True)
