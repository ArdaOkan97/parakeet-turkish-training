"""Metadata completion only; no journal/tail reread, candidate import or GC operation."""
import datetime,hashlib,json,os,shutil,sys
from pathlib import Path
import psutil
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');sys.path.insert(0,str(P/'outputs/parakeet_turkish'))
from resource_watchdog_v2 import sample_memory
F=Path(__file__).resolve().parent
C=F/'actual_source33_one_postDEV_refresh_checkpoint_transition_01/actual_transition_closure.json'
O=F/'one_postDEV_refresh_checkpoint_stop_observer_actual_01'
OUT=F/'actual_source33_checkpoint_stop_metadata_completion_v1.json'
def pin(p):
 p=Path(p).resolve();assert p.stat().st_size<=8*1024**2;b=p.read_bytes()
 return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def who(p):return {'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()}
def alive(w):
 try:return abs(psutil.Process(w['pid']).create_time()-w['create_time'])<.001
 except psutil.NoSuchProcess:return False
assert not OUT.exists() and Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad'))
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1' and all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'))
me=psutil.Process();me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);observer=[who(me),who(me.parent())]
samples=[sample_memory(w['pid'],'source33_actual_stop_metadata_completion') for w in observer]
assert all(s['paired_sampling_within_one_second'] for s in samples)
assert sum(s['worker_private_memory']['private_bytes'] for s in samples)<=536870912
m=samples[-1]['system_memory'];assert m['commit_available_bytes']>=5055306368 and m['physical_available_bytes']>=2305306368
assert shutil.disk_usage('E:/').free>=100000000000
assert pin(C)['sha256']=='824cb18cc5ad72582a026ba7273c2472061a87593b2d045df2bec75d1037301f'
c=read(C);entry=read(O/'entry.json');report=read(O/'report.json')
assert report['all_five_exact_runtime_dead'] is True and report['stop_requested'] is True
runtime=[{'identity':w,'exact_dead':not alive(w)} for w in c['all_five_runtime_identities']]
assert len(runtime)==5 and all(x['exact_dead'] for x in runtime)
metadata=[{'identity':w,'exact_dead':not alive(w)} for w in (entry['observer'],entry['launcher'])]
assert all(x['exact_dead'] for x in metadata)
assert c['actual_optimizer_updates']==c['published_journal_optimizer_updates']==178820 and c['last_fully_saved_optimizer_updates']==178800
assert c['optimizer_vs_journal_frontier']['unpublished_optimizer_counter_advance']==0
assert c['monitor_completion']['joined'] is True and c['monitor_completion']['clean'] is True and c['monitor_completion']['error'] is None
lease=read(P/'work/gpu_training_owner.json')
assert lease['owner']=='/root/memory_diagnosis' and lease['actual_GPU_running'] is False and lease['actual_owned_worker_pid'] is None
reader={'pid':26892,'create_time':1792652904.3359847,'ppid':30128};assert alive(reader)
sealer_pid=c['final_guard']['worker_private_memory']['pid']
unknown={'recorded_sampled_pid':sealer_pid,'historical_create_time':'UNRECORDED','historical_launcher_identity':'UNRECORDED',
 'current_pid_exists':psutil.pid_exists(sealer_pid),'exact_historical_identity_death_claimed':False,
 'actual_sealer_TOOL_exit_code':0,'actual_sealer_TOOL_chunk_id':'22d740',
 'reason':'Reviewed sealer records sample PID but no ctime/launcher; direct exact invocation added no external identity recorder. No fabricated identity and no rerun.'}
value={'kind':'ASR18_actual_source33_checkpoint_stop_metadata_completion_v1','UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'actual_transition_closure':pin(C),'owned_training_session':65067,'actual_owned_training_TOOL':pin(F/'actual_source33_checkpoint_stop_TOOL_result_v1.json'),
 'actual_owned_training_TOOL_exit_code':1,'actual_owned_training_TOOL_chunk_id':'f00322','normal0_claimed':False,
 'observer_owned_session':94233,'observer_actual_TOOL_exit_code':0,'observer_actual_TOOL_chunk_id':'9d54e0',
 'observer_entry':pin(O/'entry.json'),'observer_report':pin(O/'report.json'),'observer_stop':report['stop'],
 'actual_sealer_TOOL_exit_code':0,'actual_sealer_TOOL_chunk_id':'22d740','actual_sealer_source':pin(F/'seal_actual_source33_one_postDEV_refresh_checkpoint_transition_v1.py'),
 'actual_saved_updates':178800,'actual_optimizer_and_published_updates':178820,'optimizer_journal_gap':0,
 'all5_runtime_exact_dead':runtime,'known_observer_metadata_exact_deaths':metadata,'historical_sealer_identity_limitation':unknown,
 'original_Monitor_joined_clean_errorNone':True,'lease_released':True,'lease_pin':pin(P/'work/gpu_training_owner.json'),
 'reader_exact_alive':reader,'completion_identity':observer[0],'completion_launcher_identity':observer[1],'completion_pair_live_at_write':True,
 'completion_source':pin(__file__),'sampled_metadata_completion_resources':samples,
 'no_tail_journal_prefix_whole_tensor_audio_read_or_hash':True,'no_repeated_control_GC_stop_launch_or_reader_mutation':True,
 'separate_native_CPU_wrapper_numeric_OS_exits':'UNKNOWN_NOT_INFERRED','human_fix_open':True}
with OUT.open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps({'completion':pin(OUT),'sealer_identity_limitation':unknown,'completion_pair_live_at_write':observer}),flush=True)
