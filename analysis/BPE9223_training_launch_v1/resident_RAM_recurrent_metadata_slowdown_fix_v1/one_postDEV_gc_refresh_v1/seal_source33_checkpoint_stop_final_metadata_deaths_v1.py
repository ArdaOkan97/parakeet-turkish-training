"""Final known metadata death observation. Historical sealer ctime stays UNKNOWN."""
import hashlib,json,os,shutil,sys
from pathlib import Path
import psutil
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');sys.path.insert(0,str(P/'outputs/parakeet_turkish'))
from resource_watchdog_v2 import sample_memory
F=Path(__file__).resolve().parent;IN=F/'actual_source33_checkpoint_stop_metadata_completion_v1.json'
OUT=F/'actual_source33_checkpoint_stop_final_known_metadata_deaths_v1.json'
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();assert len(b)<=8*1024**2
 return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def who(p):return {'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()}
def alive(w):
 try:return abs(psutil.Process(w['pid']).create_time()-w['create_time'])<.001
 except psutil.NoSuchProcess:return False
assert not OUT.exists() and Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad'))
me=psutil.Process();me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);current=[who(me),who(me.parent())]
samples=[sample_memory(x['pid'],'source33_stop_final_metadata_death_observer') for x in current]
assert all(s['paired_sampling_within_one_second'] for s in samples) and sum(s['worker_private_memory']['private_bytes'] for s in samples)<=536870912
m=samples[-1]['system_memory'];assert m['commit_available_bytes']>=5055306368 and m['physical_available_bytes']>=2305306368
assert shutil.disk_usage('E:/').free>=100000000000
assert pin(IN)['sha256']=='e9cc31c301bd57466fde3981613af69fcc13afa9092ee3834070049b8d05af96'
c=json.loads(IN.read_text(encoding='utf-8'))
known=[x['identity'] for x in c['known_observer_metadata_exact_deaths']]+[c['completion_identity'],c['completion_launcher_identity']]
deaths=[{'identity':x,'exact_dead':not alive(x)} for x in known];assert len(deaths)==4 and all(x['exact_dead'] for x in deaths)
assert all(not alive(x['identity']) for x in c['all5_runtime_exact_dead'])
unknown=c['historical_sealer_identity_limitation'];unknown['current_pid_exists']=psutil.pid_exists(unknown['recorded_sampled_pid'])
value={'kind':'ASR18_actual_source33_checkpoint_stop_final_known_metadata_deaths_v1','completion':pin(IN),
 'actual_completion_TOOL_exit_code':0,'actual_completion_TOOL_chunk_id':'204ee9',
 'all_four_RECORDED_observer_and_completion_identities_exact_dead':True,'exact_recorded_metadata_identity_deaths':deaths,
 'historical_sealer_identity_limitation':unknown,'all_sealer_identities_exact_dead_claimed':False,
 'final_metadata_observer_source':pin(__file__),'final_observer_pair_live_at_write':current,
 'all5_old_runtime_exact_dead':True,'reader26892_exact_alive':alive(c['reader_exact_alive']),
 'no_journal_tail_prefix_whole_tensor_audio_GC_or_production_operation':True}
with OUT.open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps({'final_closure':pin(OUT),'all_recorded_metadata_exact_dead':True,'unknown_sealer_ctime_preserved':True,
 'final_observer_pair_live_at_write':current}),flush=True)
