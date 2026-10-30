"""Actual METHOD_STUB controller proof completion; no target/helper/profiler."""
import hashlib,json,os,shutil,sys
from pathlib import Path
import psutil
F=Path(__file__).resolve().parent
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');sys.path.insert(0,str(P/'outputs/parakeet_turkish'))
from resource_watchdog_v2 import sample_memory
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def write(p,v):
 with Path(p).open('x',encoding='utf-8',newline='\n')as f:json.dump(v,f,indent=2);f.write('\n')
def who(p):return {'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()}
def dead(w):
 try:return abs(psutil.Process(w['pid']).create_time()-w['create_time'])>=.001
 except psutil.NoSuchProcess:return True
psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
assert os.environ['CUDA_VISIBLE_DEVICES']=='-1'and Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad'))
sample=sample_memory(os.getpid(),'controller_METHOD_STUB_completion');parent_sample=sample_memory(os.getppid(),'controller_METHOD_STUB_completion_launcher')
family=sample['worker_private_memory']['private_bytes']+parent_sample['worker_private_memory']['private_bytes'];m=sample['system_memory']
assert sample['paired_sampling_within_one_second']and parent_sample['paired_sampling_within_one_second']and family<536870912
assert m['commit_available_bytes']>=5055306368 and m['physical_available_bytes']>=2305306368 and shutil.disk_usage('E:/').free>=100000000000
external=F/'fixed_counter_controller_external_result_v1.json';value=read(external);reportpath=Path(value['report']['path']);report=read(reportpath)
assert value['passed']is True and value['Popen_returncode']==0 and value['external_guard_error']is None
assert value['required_external_cleanup']==[]and value['inputs_before_after_exact']is True
assert report['passed']is True and report['error']is None and report['all_input_pins_unchanged']is True
assert report['clock_profiler_process_counter_inputs_are_METHOD_STUB']is True
assert report['actual_win32_thread_profile_or_training_operation']is False and report['all_owned_scalar_files_closed']is True
assert report['owned_scalar_files_opened']==report['owned_scalar_files_closed']
for item in value['input_pins_before']:assert pin(item['path'])==item
known={}
for w in [*value['recorded_identities'],value['external_identity'],value['launcher_identity'],
 report['fixture_identity'],report['fixture_launcher_identity']]:known[(w['pid'],w['create_time'])]=w
deaths=[{'identity':w,'exact_dead':dead(w)}for w in known.values()];assert all(x['exact_dead']for x in deaths)
tool=F/'fixed_counter_controller_actual_TOOL_v1.json'
write(tool,{'kind':'ASR18_fixed_counter_controller_actual_METHOD_STUB_TOOL_result_v1','actual_TOOL_exit':0,
 'actual_TOOL_chunk':'95f6de','actual_TOOL_wall_seconds':0.957935,'actual_Popen_returncode':0,
 'external_guard_error':None,'required_external_cleanup':[],'external_result':pin(external),'report':pin(reportpath)})
completion={'kind':'ASR18_fixed_counter_controller_actual_METHOD_STUB_completion_v1','actual_TOOL':pin(tool),
 'scope':pin(F/'ROOT_fixed60_counter_controller_METHOD_STUB_CPU_execution_scope_v1.json'),
 'actual_report':pin(reportpath),'actual_external_result':pin(external),'actual_Popen_returncode':0,
 'external_guard_error':None,'required_external_cleanup':[],'all_input_pins_unchanged':value['input_pins_before'],
 'actual_cases_count':len(report['cases']),'actual_owned_scalar_files_opened_closed':[report['owned_scalar_files_opened'],report['owned_scalar_files_closed']],
 'actual_job_identity_deaths':deaths,'controller_source':pin(F/'fixed_counter_window_and_scalar_files_v1.py'),
 'metadata_sealer':who(psutil.Process()),'metadata_sealer_launcher':who(psutil.Process().parent()),
 'completion_sampled_family_private_bytes':family,'completion_guard':sample,
 'sealer_pair_live_at_write_exact_death_checked_after_TOOL_return':True,'sealer_source':pin(__file__),
 'all_clocks_process_counter_values_METHOD_STUB_actual_E_scalar_files_only':True,
 'no_real_helper_profiler_target_training_model_audio_GPU_or_oldtoy_operation':True,
 'no_live_or_production_execution_admitted_by_this_owner_completion':True,'ROOT_actual_acceptance_pending':True}
p=F/'fixed_counter_controller_actual_completion_v1.json';write(p,completion)
print(json.dumps({'completion':pin(p),'actual_TOOL':pin(tool),'report':pin(reportpath),
 'cases':completion['actual_cases_count'],'job_deaths':deaths,'metadata_sealer':completion['metadata_sealer'],
 'metadata_sealer_launcher':completion['metadata_sealer_launcher']}),flush=True)
