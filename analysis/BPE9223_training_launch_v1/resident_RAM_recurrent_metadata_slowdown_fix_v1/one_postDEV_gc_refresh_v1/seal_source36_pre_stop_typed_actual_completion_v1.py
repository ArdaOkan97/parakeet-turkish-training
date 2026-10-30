"""One actual typed-job metadata closure only; no candidate import or GC call."""
import datetime,hashlib,json,os,shutil,sys
from pathlib import Path
import psutil
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');sys.path.insert(0,str(P/'outputs/parakeet_turkish'))
from resource_watchdog_v2 import sample_memory
F=Path(__file__).resolve().parent
EXT=F/'source36_pre_stop_typed_metadata_external_result_v1.json'
REPORT=F/'source36_pre_stop_typed_selector_actual_01/report.json'
SCOPE=F/'ROOT_source36_pre_stop_entire_selector_metadata_exact_CPU_scope_v1.json'
TOOL=F/'source36_pre_stop_typed_metadata_actual_TOOL_v1.json';OUT=F/'source36_pre_stop_typed_metadata_actual_completion_v1.json'
def pin(p):
 p=Path(p).resolve();assert p.stat().st_size<=8*1024**2;b=p.read_bytes()
 return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def who(p):return {'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()}
def alive(w):
 try:return abs(psutil.Process(w['pid']).create_time()-w['create_time'])<.001
 except psutil.NoSuchProcess:return False
def write(p,v):
 with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
assert not OUT.exists() and not TOOL.exists()
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1' and os.environ.get('PYTHONUTF8')=='1'
assert all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'))
assert Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad'))
me=psutil.Process();me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);observer=[who(me),who(me.parent())]
samples=[sample_memory(w['pid'],'source36_typed_metadata_completion_only') for w in observer]
assert all(s['paired_sampling_within_one_second'] for s in samples)
assert sum(s['worker_private_memory']['private_bytes'] for s in samples)<=536870912
m=samples[-1]['system_memory'];assert m['commit_available_bytes']>=5055306368 and m['physical_available_bytes']>=2305306368
assert shutil.disk_usage('E:/').free>=100000000000
assert pin(EXT)['sha256']=='267f2745f2d063ca7f77c54100b47cb1b7ea03bc74d25f3ad8ec8c60bb794d57'
assert pin(SCOPE)['sha256']=='8f535f3e68b04ca34623936aee39a04d94cbac5e256076cca5d2b2969fd72f4d'
ext=json.loads(EXT.read_text(encoding='utf-8'));report=json.loads(REPORT.read_text(encoding='utf-8'));scope=json.loads(SCOPE.read_text(encoding='utf-8'))
assert ext['passed'] is True and ext['Popen_returncode']==0 and ext['external_guard_error'] is None and ext['required_external_cleanup']==[]
assert ext['inputs_before_after_exact'] is True and ext['report']==pin(REPORT) and report['passed'] is True
assert not (REPORT.parent/'actual_failure.json').exists()
assert report['results']['entire_actual_eligible_driver_event_and_per_instance_final_error_close_METHOD_STUB']['passed'] is True
assert report['production_stop_or_final_config_or_authority_write'] is False
assert report['actual_candidate_gc_event_or_capture_or_restore_copy_sampler_native_audio_GPU_operation'] is False
assert all(pin(x['path'])==x for x in scope['input_pins']) and len(scope['input_pins'])==38
known={}
for w in [ext['external_identity'],ext['launcher_identity'],ext['original_Popen_identity'],*ext['recorded_identities'],report['entry']['observer'],report['entry']['launcher']]:
 known[(w['pid'],w['create_time'])]=w
deaths=[{'identity':w,'exact_dead':not alive(w)} for w in known.values()]
assert len(deaths)==4 and all(x['exact_dead'] for x in deaths)
protected=[{'identity':w,'exact_alive':alive(w)} for w in scope['protected_runtime_identities']]
assert all(x['exact_alive'] for x in protected)
write(TOOL,{'kind':'ASR18_source36_pre_stop_typed_actual_TOOL_v1','actual_TOOL_exit_code':0,'actual_TOOL_chunk_id':'db722f',
 'actual_Popen_returncode':0,'exact_external_arguments':scope['exact_external_arguments'],'external_result':pin(EXT),
 'actual_report':pin(REPORT),'guard_error':None,'required_cleanup':[],'one_actual_invocation_only':True,'no_retry':True})
value={'kind':'ASR18_source36_pre_stop_typed_metadata_actual_completion_v1','passed':True,
 'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'ROOT_scope':pin(SCOPE),'actual_TOOL':pin(TOOL),
 'external_result':pin(EXT),'report':pin(REPORT),'stdout':pin(ext['stdout']['path']),'stderr':pin(ext['stderr']['path']),
 'actual_results_count':len(report['results']),'all38_input_pins_before_after_and_current_exact':True,
 'all_four_exact_job_identities_dead':True,'exact_job_identity_closure_records':deaths,
 'metadata_sealer_identity':observer[0],'metadata_sealer_launcher_identity':observer[1],'metadata_sealer_pair_live_at_write':True,
 'metadata_sealer_source':pin(__file__),'protected_runtime_exact_alive':protected,
 'scientific_and_future_stop_owners_remain_in_memory_METHOD_STUBS':True,'no_actual_production_GC_stop_restore_or_draw':True,
 'sampled_metadata_resource_samples':samples,'production_and_reader_unchanged':True}
write(OUT,value)
print(json.dumps({'completion':pin(OUT),'TOOL':pin(TOOL),'report':pin(REPORT),'results':len(report['results']),
 'job_deaths':deaths,'sealer_pair_live_at_write':observer}),flush=True)
