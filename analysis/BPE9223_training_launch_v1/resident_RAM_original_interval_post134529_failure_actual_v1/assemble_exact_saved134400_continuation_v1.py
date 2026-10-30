"""Metadata only: bind actual ROOT CPU acceptance, real typed gate0/death, handoff."""
import argparse,copy,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
import psutil
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish';R=P/'work/research';N=Path(__file__).parent
sys.path[:0]=[str(S),str(R)]
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
from resource_watchdog_v2 import sample_memory
parser=argparse.ArgumentParser();parser.add_argument('--CPU-review',required=True);parser.add_argument('--gate-child',action='store_true');parser.add_argument('--configuration');a=parser.parse_args()
began=time.monotonic()
def guard():
    v=sample_memory(os.getpid(),'saved134400_metadata_only_gate');s=v['system_memory']
    require(v['paired_sampling_within_one_second']and v['worker_private_memory']['private_bytes']<512*1024**2
        and s['commit_available_bytes']>=2000000000+268435456 and s['physical_available_bytes']>=1500000000+268435456
        and shutil.disk_usage('E:/').free>=100000000000 and time.monotonic()-began<180,'Finite actual metadata/private/headroom/E guards')
    return v
guard();require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1','MetadataCPU only');psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
source_handoff=read_json(N/'owner16_stream7_inactive_source_handoff_v1.json');failure_pin=record(N/'actual_failure_closure.json');failure=read_json(failure_pin['path'])
branch=Path(r'E:\gpu_training_scratchpad\runs\BPE9223_resident_RAM_20261018_01');pointer=failure['source_evidence']['saved_pointer']
D=N/'production_configuration_v1'
if a.gate_child:
    config=read_json(a.configuration)
    from run_ASR18_BPE9223_resident_RAM_training_v16 import checked_original_interval_resume
    from fastmss_resident_feature_stream_v7 import checked_resource_profile
    checked_resource_profile(config)
    result=checked_original_interval_resume(failure_pin,base=config,folder=branch,pointer_pin=pointer)
    require(result['saved_optimizer_updates']==134400 and len(result['preserved_tail_positive_updates'])==129,'Actual serialized final typed entry')
    require('torch'not in sys.modules,'No model/GPU imported')
    who={'pid':os.getpid(),'create_time':psutil.Process().create_time(),'ppid':os.getppid()}
    write_json(D/'actual_final_typed_metadata_gate.json',{'passed':True,'producer':who,'source':record(__file__),
        'configuration':record(a.configuration),'checkpoint':pointer,'actual_failure':failure_pin,'saved_updates':134400,
        'historical_positive_tail_count':129,'profile':config['resident_RAM_resource_scheduling_policy'],'no_model_GPU_audio_draws_or_native':True,
        'actual_child_exit_still_unobserved_inside_child':True,'last_resource_sample':guard()})
    raise SystemExit(0)
review=read_json(a.CPU_review);CPU=read_json(N/'ingress_frontend_actual_01/report.json');CPUclosure=read_json(N/'actual_fifth_ingress_CPU_closure.json')
config=read_json(N/'ingress_frontend_actual_01/configuration.json');policy=config['original_clip_geometry_overlay']
require(review['accepted']is True and review['reviewer']=='/root'
    and review['scope']=='ASR18_source_bound_fifth_original_full_clip_geometry_overlay_v1'
    and review['overlay_source']==policy['source']and review['certificate']==policy['certificate'],'Genuine exact scoped completed-CPU ROOT acceptance')
require(CPU['passed']and CPUclosure['actual_TOOL_exit_code']==0 and CPUclosure['exact_original_CPU_identity_dead'],'Actual accepted finite CPU proof')
policy['ROOT_review']=record(a.CPU_review)
prior=read_json(source_handoff['prior_configuration']['path']);undo=copy.deepcopy(config)
for key in ('original_clip_geometry_overlay','prior_resident_RAM_original_interval_failure','resident_RAM_resource_scheduling_policy'):undo[key]=prior[key]
require(undo==prior and policy['prior_overlay']==prior['original_clip_geometry_overlay'],'Only three operational fields and exact prior4 policy')
for key,value in prior['resident_RAM_resource_scheduling_policy'].items():
    require(key=='CPU_protocol_source'or config['resident_RAM_resource_scheduling_policy'][key]==value,'Source15 resource numeric values unchanged')
require(not D.exists()and not(branch/'invocations/owner_000012').exists()and not(branch/'journals/commits_000012.jsonl').exists(),'No new training invocation or journal')
D.mkdir();config_path=D/'configuration.json';write_json(config_path,config)
CPU_python=Path(r'E:\gpu_training_scratchpad\scratch\BPE_broad_CPU_preflight_env_v2\Scripts\python.exe')
command=[str(CPU_python),'-Xutf8','-B',str(Path(__file__).resolve()),'--CPU-review',a.CPU_review,'--gate-child','--configuration',str(config_path)]
child=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8',creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
launcher={'pid':child.pid,'create_time':psutil.Process(child.pid).create_time()};out,err=child.communicate(timeout=180)
(D/'actual_typed_gate_stdout_stderr.txt').write_text(out+'\n'+err,encoding='utf-8')
require(child.returncode==0,'Preserve any actual final typed metadata failure')
gate=read_json(D/'actual_final_typed_metadata_gate.json');deaths=[]
for who in (launcher,gate['producer']):
    try:now=psutil.Process(who['pid']).create_time()
    except psutil.NoSuchProcess:now=None
    require(now is None or abs(now-who['create_time'])>=.001,'Actual final metadata worker and interpreter launcher dead')
    deaths.append({'original_identity':who,'observed_current_create_time':now,'original_identity_dead':True})
write_json(D/'actual_final_typed_metadata_gate0_closure.json',{'actual_Popen_wait_code':child.returncode,'actual_dead_identities':deaths,
    'worker_report':record(D/'actual_final_typed_metadata_gate.json'),'source':record(__file__),'no_GPU_or_audio_job':True})
latest_failure=read_json(branch/'invocations/owner_000011/boot.json')['latest_failure']
output=N.parent/'resident_RAM_training_actual_13';require(not output.exists(),'Fresh future production output')
sources=source_handoff['sources'];review_fields={'accepted':True,'reviewer':'/root','driver_source':sources['driver'],'wrapper_source':sources['wrapper'],
    'RAM_native_source':record(S/'asr18_BPE9223_grouped_RAM_native_v1.py'),'CPU_source':sources['CPU'],
    'DEV_helper_source':record(S/'asr18_resident_serial350_DEV_v2.py'),'source_configuration':record(config_path),
    'initial_checkpoint':pointer,'latest_failure':latest_failure,'bounded_RAM_queue_bytes':268435456,'periodic_DEV_interval_updates':4000,
    'private_scalar_prefetch_source':config['private_scalar_prefetch_policy']['source'],
    'private_scalar_prefetch_actual_CPU_review':config['private_scalar_prefetch_policy']['actual_CPU_control_review'],
    'additional_prefetch_host_reservation_bytes':268435456,'original_interval_resume':failure_pin,
    'original_clip_geometry_overlay':policy,'resource_scheduling_policy':config['resident_RAM_resource_scheduling_policy'],'initial_owner_invocation':12}
arguments=['--review','<FINAL_ROOT_EXECUTION_REVIEW_REQUIRED>','--configuration',str(config_path),'--pointer',pointer['path'],
    '--latest-failure',latest_failure['path'],'--branch',str(branch),'--output',str(output),'--queue-bytes','268435456',
    '--start-invocation','12','--original-interval-resume',failure_pin['path']]
handoff={'kind':'ASR18_INACTIVE_exact_saved134400_final_execution_handoff_v1','production_admitted':False,'sources':sources,
    'configuration':record(config_path),'ROOT_actual_CPU_review':record(a.CPU_review),'CPU_report':record(N/'ingress_frontend_actual_01/report.json'),
    'actual_CPU0_closure':record(N/'actual_fifth_ingress_CPU_closure.json'),'final_typed_metadata0_closure':record(D/'actual_final_typed_metadata_gate0_closure.json'),
    'actual_failure':failure_pin,'checkpoint':pointer,'historical129_positive_tail':{k:failure['preserved_unsaved_positive_tail'][k]for k in ('path','bytes','sha256')},
    'ROOT_execution_review_required_fields':review_fields,'exact_future_wrapper_arguments':arguments,'future_output':str(output),
    'saved_prefix_only_copy_AFTER_fresh_exact_restore':True,'no_original_whole_journal_payload_reread_by_this_assembler':True,
    'full401072_adminNone':True,'prior4_overlay_and_resource_values_and_all_numerical_bodies_unchanged':True}
write_json(D/'final_execution_handoff.json',handoff);print(json.dumps({'configuration':record(config_path),'handoff':record(D/'final_execution_handoff.json'),
    'metadata_gate_actual_wait_code':child.returncode,'actual_recorded_gate_identities_dead':True}),flush=True)
