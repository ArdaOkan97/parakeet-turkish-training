"""Read-only process identities and lightweight final closure; no candidate import or GC call."""
import datetime, hashlib, json, os, shutil, sys
from pathlib import Path
import psutil
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20'); sys.path.insert(0,str(P/'outputs/parakeet_turkish'))
from resource_watchdog_v2 import sample_memory
F=Path(__file__).resolve().parent
IN=F/'source36_pre_stop_typed_metadata_actual_completion_v1.json'
OUT=F/'source36_pre_stop_typed_final_exact_metadata_deaths_v1.json'
def pin(p):
 p=Path(p).resolve(); b=p.read_bytes(); assert len(b)<=8*1024**2
 return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def identity(p): return {'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()}
def alive(w):
 try: return abs(psutil.Process(w['pid']).create_time()-w['create_time'])<.001
 except psutil.NoSuchProcess: return False
assert not OUT.exists()
assert Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad'))
me=psutil.Process(); me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
observer=[identity(me),identity(me.parent())]
samples=[sample_memory(w['pid'],'one_postDEV_finite_final_death_metadata') for w in observer]
assert all(s['paired_sampling_within_one_second'] for s in samples)
assert sum(s['worker_private_memory']['private_bytes'] for s in samples)<=536870912
system=samples[-1]['system_memory']
assert system['commit_available_bytes']>=5055306368 and system['physical_available_bytes']>=2305306368
assert shutil.disk_usage('E:/').free>=100000000000
assert pin(IN)['sha256']=='6c6078daac5083516be5df7e46819e374fed414d2caf9a3872574977133152b9'
c=json.loads(IN.read_text(encoding='utf-8'))
known=[x['identity'] for x in c['exact_job_identity_closure_records']]+[c['metadata_sealer_identity'],c['metadata_sealer_launcher_identity']]
deaths=[{'identity':w,'exact_dead':not alive(w)} for w in known]
assert len({(w['pid'],w['create_time']) for w in known})==6 and all(x['exact_dead'] for x in deaths)
protected=[{'identity':x['identity'],'exact_alive':alive(x['identity'])} for x in c['protected_runtime_exact_alive']]
assert all(x['exact_alive'] for x in protected)
value={'kind':'ASR18_source36_pre_stop_typed_final_exact_metadata_deaths_v1',
 'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completion':pin(IN),
 'actual_completion_TOOL_exit_code':0,'actual_completion_TOOL_chunk_id':'da0483',
 'all_six_job_and_completion_exact_metadata_identities_dead':True,'exact_identity_closure_records':deaths,
 'final_metadata_observer_source':pin(__file__),'final_metadata_observer_pair_live_at_write':observer,
 'protected_runtime_exact_alive':protected,'no_training_reader_source_environment_or_policy_change':True}
with OUT.open('x',encoding='utf-8',newline='\n') as f:
 json.dump(value,f,ensure_ascii=False,indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
print(json.dumps({'final_closure':pin(OUT),'all_six_exact_dead':True,'observer_pair_live_at_write':observer}),flush=True)
