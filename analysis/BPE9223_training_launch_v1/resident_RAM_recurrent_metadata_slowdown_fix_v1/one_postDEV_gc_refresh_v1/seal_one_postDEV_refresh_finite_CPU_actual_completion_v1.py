"""Metadata-only closure of the one actual isolated finite CPU fixture; no candidate import."""
import datetime, hashlib, json, os, shutil, sys
from pathlib import Path
import psutil
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20')
S=P/'outputs/parakeet_turkish'; sys.path.insert(0,str(S))
from resource_watchdog_v2 import sample_memory
F=Path(__file__).resolve().parent
EXT=F/'one_postDEV_refresh_finite_CPU_external_result_v2.json'
REPORT=F/'finite_one_postDEV_refresh_actual_CPU_01/report.json'
SCOPE=F/'ROOT_one_postDEV_refresh_finite_CPU_v2_exact_execution_scope_v1.json'
TOOL=F/'one_postDEV_refresh_finite_CPU_actual_TOOL_v1.json'
OUT=F/'one_postDEV_refresh_finite_CPU_actual_completion_v1.json'
def pin(p):
 p=Path(p).resolve(); b=p.read_bytes(); assert len(b)<=8*1024**2
 return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def identity(p): return {'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()}
def same_alive(w):
 try: return abs(psutil.Process(w['pid']).create_time()-w['create_time'])<.001
 except psutil.NoSuchProcess: return False
def write(p,v):
 with p.open('x',encoding='utf-8',newline='\n') as f:
  json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
assert not TOOL.exists() and not OUT.exists(),'Fresh actual closure only; no overwrite or rerun'
me=psutil.Process(); me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
mine=identity(me); launcher=identity(me.parent())
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1' and os.environ.get('PYTHONUTF8')=='1'
assert all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'))
assert Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad'))
samples=[sample_memory(w['pid'],'one_postDEV_finite_metadata_completion') for w in (mine,launcher)]
assert all(x['paired_sampling_within_one_second'] for x in samples)
assert sum(x['worker_private_memory']['private_bytes'] for x in samples)<=536870912
system=samples[-1]['system_memory']
assert system['commit_available_bytes']>=5055306368 and system['physical_available_bytes']>=2305306368
assert shutil.disk_usage('E:/').free>=100000000000
assert pin(EXT)['sha256']=='b228555ba1de5f0ec2073f60c6e362bebc2d7c014dbeed94a3453dfbf99aea54'
assert pin(SCOPE)['sha256']=='0623526a4b37510f910fd676ed7d2d1b07319096bede5854c9d2c39a3c30fb82'
ext=json.loads(EXT.read_text(encoding='utf-8')); report=json.loads(REPORT.read_text(encoding='utf-8'))
scope=json.loads(SCOPE.read_text(encoding='utf-8'))
assert ext['passed'] is True and ext['Popen_returncode']==0 and ext['external_guard_error'] is None
assert ext['required_external_cleanup']==[] and ext['inputs_before_after_exact'] is True
assert report['passed'] is True and report['actual_exception'] is None and report['required_fixture_cleanup_errors']==[]
assert report['input_pins_before_after_exact'] is True and report['final_frozen_objects']==0
assert report['final_original_collect_alias_restored'] is True
assert ext['report']==pin(REPORT) and all(x['passed'] is True for x in report['cases'])
assert all(pin(w['path'])==w for w in scope['input_pins'])
known={}
for w in [ext['external_identity'],ext['launcher_identity'],ext['original_Popen_identity'],*ext['recorded_identities'],report['fixture_identity']]:
 known[(w['pid'],w['create_time'])]=w
deaths=[{'identity':w,'exact_dead':not same_alive(w)} for w in known.values()]
assert len(deaths)==4 and all(x['exact_dead'] for x in deaths)
protected=[{'identity':w,'exact_alive':same_alive(w)} for w in scope['protected_runtime_identities']]
assert all(w['exact_alive'] for w in protected)
write(TOOL,{'kind':'ASR18_one_postDEV_refresh_finite_CPU_actual_TOOL_v1','actual_TOOL_exit_code':0,
 'actual_TOOL_chunk_id':'5b0014','actual_external_Popen_returncode':0,
 'exact_external_arguments':scope['exact_external_arguments'],'external_result':pin(EXT),
 'actual_output_passed':True,'external_guard_error':None,'required_external_cleanup':[],
 'sampled_family_private_peak_bytes':ext['sampled_family_private_peak_bytes'],
 'one_invocation_only':True,'no_retry':True})
value={'kind':'ASR18_one_postDEV_refresh_finite_CPU_actual_completion_v1',
 'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':True,
 'ROOT_execution_scope':pin(SCOPE),'actual_TOOL':pin(TOOL),'external_result':pin(EXT),'report':pin(REPORT),
 'stdout':pin(ext['stdout']['path']),'stderr':pin(ext['stderr']['path']),
 'all_four_exact_job_identities_dead':True,'exact_job_identity_closure_records':deaths,
 'metadata_sealer_identity':mine,'metadata_sealer_launcher_identity':launcher,
 'metadata_sealer_pair_live_at_write':True,'metadata_sealer_source':pin(__file__),
 'protected_runtime_exact_alive':protected,'case_count':len(report['cases']),
 'all_case_results_passed':True,'input_pins_before_after_and_current_exact':True,
 'sampled_metadata_completion_resource_samples':samples,
 'scientific_GPU_full854_equality':'METHOD_STUB and source proof only; not established by this CPU fixture',
 'genuine_production_DB_lock_lifecycle_tested':False,'opaque_framework_cycle_absence_proved':False,
 'production_fit_saving_speed_or_prevention_proved':False,'production_and_reader_unchanged':True,
 'no_candidate_import_or_GC_call_in_this_metadata_sealer':True}
write(OUT,value)
print(json.dumps({'completion':pin(OUT),'TOOL':pin(TOOL),'report':pin(REPORT),
 'cases':len(report['cases']),'job_deaths':deaths,'sealer_live_at_write':[mine,launcher]}),flush=True)
