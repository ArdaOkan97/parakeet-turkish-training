"""Owned launch metadata only; does not consume its exec or operate on GPU/state."""
import hashlib,json,os,shutil,sys
from pathlib import Path
import psutil
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');sys.path.insert(0,str(P/'outputs/parakeet_turkish'))
from resource_watchdog_v2 import sample_memory
F=Path(__file__).resolve().parent;N=F.parent.parent;A=N/'resident_RAM_training_actual_26'
B=Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01');I=B/'invocations/owner_000025'
OUT=A/'actual_owned_source36_boot_session.json'
def pin(p):
 p=Path(p).resolve();assert p.stat().st_size<=8*1024**2;b=p.read_bytes()
 return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def who(p):return {'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()}
def alive(w):
 try:return abs(psutil.Process(w['pid']).create_time()-w['create_time'])<.001
 except psutil.NoSuchProcess:return False
assert not OUT.exists() and Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad'))
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
me=psutil.Process();me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
samples=[sample_memory(p.pid,'source36_actual_boot_compact_capture') for p in (me,me.parent())]
assert all(s['paired_sampling_within_one_second'] for s in samples) and sum(s['worker_private_memory']['private_bytes'] for s in samples)<=536870912
m=samples[-1]['system_memory'];assert m['commit_available_bytes']>=5055306368 and m['physical_available_bytes']>=2305306368
assert shutil.disk_usage('E:/').free>=100000000000
review_path=F/'ROOT_source36_one_postDEV_gc_refresh_saved178800_final_execution_review_v1.json'
assert pin(review_path)['sha256']=='2a23e3143e8e91586f46214e5af1c650f6057887a43e169352fd2be6c99b04d2'
r=read(review_path);child=read(A/'current_child.json');boot=read(I/'boot.json');configuration=read(I/'configuration.json')
expected=read(r['source_configuration']['path']);expected.update(r['future_seven_runtime_bindings'])
assert configuration==expected and boot['source']==r['driver_source'] and boot['source_checkpoint']==r['initial_checkpoint']
assert child['invocation']==25 and child['source_checkpoint']==r['initial_checkpoint']
native=boot['producer'];launcher=child['launcher'];wrapper=who(psutil.Process(launcher['ppid']))
family=[{'role':role,**w,'exact_alive':alive(w)} for role,w in (('native',native),('native_interpreter_launcher',launcher),('wrapper',wrapper))]
assert all(x['exact_alive'] for x in family) and native['ppid']==launcher['pid'] and launcher['ppid']==wrapper['pid']
pins={'boot':pin(I/'boot.json'),'configuration':pin(I/'configuration.json'),'current_child':pin(A/'current_child.json'),
 'ROOT_execution_review':pin(review_path),'prelaunch_metadata':pin(F/'actual_source36_single_launch_prelaunch_metadata_v1.json')}
stage={}
for name in ('actual_parent_restore.json','actual_journal_branch.json','actual_failure.json'):
 p=I/name;stage[name]={'present':p.exists(),'pin':pin(p) if p.exists() else None}
ready=I/'CPU/actual_CPU_ready.json';stage['CPU/actual_CPU_ready.json']={'present':ready.exists(),'pin':pin(ready) if ready.exists() else None}
value={'kind':'ASR18_actual_source36_owned_initial_boot_session_v1','owner':'/root/memory_diagnosis','owned_session':42997,
 'actual_initial_exec_chunk_id':'a4bdd3','actual_initial_exec_running':True,'sole_launch_invocation':25,
 'exact_wrapper_command':r['exact_initial_launch_command_prefix']+r['exact_initial_wrapper_arguments'],
 'actual_three_boot_runtime':family,'all7_runtime_bindings_and_every_remaining_configuration_value_exact':True,
 'future_seven_runtime_bindings':r['future_seven_runtime_bindings'],'new_native_configuration':pins['configuration'],
 'boot_pins':pins,'stage_presence_at_snapshot':stage,'restore_or_positive_or_recovery_not_inferred_from_boot':True,
 'production_environment_record':'OriginalCUDA0/all4caps1/E storage applied before launch; CPUcapture localmask does not change production',
 'reader26892_exact_alive':alive({'pid':26892,'create_time':1792652904.3359847}),
 'capture_identity':who(me),'capture_launcher_identity':who(me.parent()),'metadata_resource_samples':samples,
 'no_GPU_probe_state_draw_control_or_owned_exec_consumption_by_capture':True}
with OUT.open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps({'owned_boot_receipt':pin(OUT),'actual_three':family,'stage_presence':stage}),flush=True)
