"""Lightweight actual TOOL/result/job-death seal; no source/control re-execution."""
import hashlib,json,os,shutil,sys,time
from pathlib import Path
import psutil
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from resource_watchdog_v2 import sample_memory
F=Path(__file__).resolve().parent
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 with Path(p).open('x',encoding='utf-8',newline='\n')as f:json.dump(v,f,indent=2);f.write('\n')
def who(p):return {'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()}
def dead(w):
 try:return abs(psutil.Process(w['pid']).create_time()-w['create_time'])>=.001
 except psutil.NoSuchProcess:return True
psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
require=lambda v,m:None if v else(_ for _ in ()).throw(ValueError(m))
require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'and Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad')),'Applied E CPU metadata runtime')
sample=sample_memory(os.getpid(),'source33_typed_actual01_compact_completion');m=sample['system_memory']
require(sample['paired_sampling_within_one_second']and sample['worker_private_memory']['private_bytes']<536870912
 and m['commit_available_bytes']>=2805306368 and m['physical_available_bytes']>=2305306368
 and shutil.disk_usage('E:/').free>=100000000000,'Full512MiB prospective/FULLprefetch/originalfloors/E100GB')
scope=F/'ROOT_source33_eighth_actual_failure_entire_selector_exact_one_CPU_scope_v1.json';review=read(scope)
external=F/'source33_typed_eighth_selector_actual_01_external_result.json';result=read(external)
reportpath=F/'source33_typed_eighth_selector_actual_01/report.json';report=read(reportpath)
require(result['actual_Popen_wait_code']==0 and result['external_guard_error']is None and result['targeted_cleanup']==[]
 and report['passed']is True and result['report']==pin(reportpath),'Actual clean finite metadata pass, no cleanup or failure promotion')
for item in review['input_pins']:require(pin(item['path'])==item,'All39 frozen inputs exact')
known={}
for w in [result['external_identity'],result['external_launcher_identity'],report['entry']['parent'],report['entry']['launcher'],
 *[v['identity']for v in result['actual_owned_identities']]]:known[(w['pid'],w['create_time'])]=w
deaths=[{'identity':w,'exact_dead':dead(w)}for w in known.values()]
require(all(v['exact_dead']for v in deaths),'All actual fixture/external exact identities dead')
toolpath=F/'source33_typed_metadata_actual01_TOOL_v1.json'
write(toolpath,{'kind':'ASR18_actual_owned_source33_typed_metadata_TOOL_result_v1','actual_TOOL_exit_code':0,
 'actual_TOOL_chunk':'0e6796','actual_TOOL_wall_seconds':1.0724335,'actual_Popen_wait_code':0,
 'external_result':pin(external),'report':pin(reportpath),'external_guard_error':None,'no_automatic_retry':True})
completion={'kind':'ASR18_source33_typed_metadata_actual01_completion_v1','actual_TOOL':pin(toolpath),'ROOT_scope':pin(scope),
 'fixture_report':pin(reportpath),'external_result':pin(external),'actual_Popen_wait_code':0,
 'external_guard_error':None,'targeted_cleanup':[],'all39_input_pins_exact':review['input_pins'],
 'actual_case_count':len(report['results']),'job_identity_deaths':deaths,
 'metadata_sealer_identity':who(psutil.Process()),'metadata_sealer_interpreter_launcher_identity':who(psutil.Process().parent()),
 'sealer_and_its_launcher_live_at_write_exact_death_checked_after_TOOL_returns':True,
 'sealer_source':pin(__file__),'sample':sample,'source33_executed_only_as_metadata_import_and_AST':True,
 'no_original_journal_payload_audio_model_GPU_restore_copy_sampler_native_or_kernel_operation':True,
 'source_configuration_unchanged':review['source_configuration'],'actual_failure_closure_unchanged':review['eighth_original_resume'],
 'source_handoff':pin(F/'source33_eighth_actual_failure_inactive_execution_handoff_v1.json'),
 'corrected_runtime7':pin(F/'source33_configuration_and_future7_runtime_bindings_v2.json'),
 'production_launch_admitted':False}
path=F/'source33_typed_metadata_actual01_completion_v1.json';write(path,completion)
print(json.dumps({'completion':pin(path),'actual_TOOL_receipt':pin(toolpath),'job_deaths':deaths,
 'sealer':completion['metadata_sealer_identity'],'sealer_launcher':completion['metadata_sealer_interpreter_launcher_identity']}),flush=True)
