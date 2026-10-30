import hashlib, importlib.util, json, os, pathlib, sys, time
import psutil

W=pathlib.Path(__file__).resolve().parent
OUT=W/'source39_entire_transition_integration_CPU_failed_actual01_completion_v1.json'
INITIAL=W/'source39_entire_transition_integration_CPU_failed_actual01_completion_initial_v1.json'
TOOL=W/'source39_entire_transition_integration_CPU_actual01_owned_TOOL1_v1.json'
EXT=W/'source39_entire_transition_integration_CPU_external_actual_01'/'external_result.json'
READER=pathlib.Path('E:/gpu_training_scratchpad/analysis/BPE9223_Trackio_live_v1/current_status.json')

def require(value,message):
    if not value:raise ValueError(message)
def identity(p):return dict(pid=p.pid,create_time=p.create_time(),ppid=p.ppid())
def alive(who):
    try:
        p=psutil.Process(who['pid'])
        return p.create_time()==who['create_time']
    except psutil.NoSuchProcess:return False
def pin(path):
    raw=path.read_bytes()
    return dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def write(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,sort_keys=True,ensure_ascii=False,allow_nan=False)
        f.write('\n');f.flush();os.fsync(f.fileno())

me=psutil.Process();me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
own=identity(me);parent=identity(me.parent())
entry=dict(identity=own,launcher=parent,recorded_before_predicate=True)
require(not OUT.exists() and not INITIAL.exists() and not TOOL.exists(),'Fresh scalar metadata only')
spec=importlib.util.spec_from_file_location('source39_failure_completion_watchdog',pathlib.Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20/outputs/parakeet_turkish/resource_watchdog_v2.py'))
watch=importlib.util.module_from_spec(spec);spec.loader.exec_module(watch)
sample=watch.sample_memory(me.pid,'source39_failed_CPU_metadata_completion')
initial=dict(entry=entry,sample=sample,E_free=__import__('shutil').disk_usage('E:/gpu_training_scratchpad').free,
             private_bytes=me.memory_info().private,launcher_private_bytes=me.parent().memory_info().private,
             TORCH_EXTENSIONS_DIR=os.environ.get('TORCH_EXTENSIONS_DIR'))
write(INITIAL,initial)
sysmem=sample['system_memory']
require(sysmem['commit_available_bytes']>=5130803840 and sysmem['physical_available_bytes']>=2380803840,'Unchanged full prospective initial guard')
require(initial['private_bytes']+initial['launcher_private_bytes']<=536870912 and initial['E_free']>=100000000000,'Original private and E guards')
write(TOOL,dict(kind='ASR18_owned_actual_TOOL_result_v1',tool_chunk_id='fcec2d',exit_code=1,wall_time_seconds=1.2767185,
               external_Popen_returncode=1,guard_error=None,automatic_retry=False,
               output_summary='Actual metadata CPU external failed; original failure preserved in report/stderr',
               separate_native_OS_exit_codes='NOT_APPLICABLE_CPU_metadata_only_no_production_started'))
ext=json.loads(EXT.read_text(encoding='utf-8'))
report_path=pathlib.Path(ext['report']['path']);report=json.loads(report_path.read_text(encoding='utf-8'))
job=[ext['external'],ext['external_launcher']]+ext['owned_identities']
deaths=[dict(identity=who,exact_dead=not alive(who)) for who in job]
scope=json.loads((W/'ROOT_source39_actual204800_entire_transition_main_path_CPU_exact_ONCE_scope_v1.json').read_text(encoding='utf-8'))
protected=[dict(identity=who,exact_alive=alive(who),expected_alive=who['expected_alive']) for who in scope['protected_runtime_identities']]
reader_raw=READER.read_bytes();require(len(reader_raw)<=1048576,'Bounded compact reader')
reader=json.loads(reader_raw)
require(all(d['exact_dead'] for d in deaths),'All four exact job identities dead')
require(all(d['exact_alive']==d['expected_alive'] for d in protected),'Protected training dead and reader alive unchanged')
require(ext['input_pins_before']==ext['input_pins_after'],'All36 original inputs before/after equality')
require(ext['guard_error'] is None and ext['cleanup']==[],'Preserve actual guard/cleanup')
paths=[EXT,report_path,INITIAL,TOOL,pathlib.Path(ext['stdout']['path']),pathlib.Path(ext['stderr']['path']),
       W/'source39_entire_transition_integration_CPU_actual_01'/'entry.json',
       W/'source39_entire_transition_integration_CPU_actual_01'/'initial_numeric_before_predicate.json',
       W/'source39_entire_transition_integration_CPU_external_actual_01'/'initial_numeric_before_predicate.json',pathlib.Path(__file__)]
result=dict(kind='ASR18_source39_actual204800_CPU_actual01_failed_metadata_completion_v1',entry=entry,
    actual_TOOL=dict(exit_code=1,chunk_id='fcec2d'),Popen_returncode=ext['Popen_returncode'],
    guard_error=ext['guard_error'],cleanup=ext['cleanup'],passed=False,
    actual_error=report['error'],completed_results_count=len(report['results']),completed_result_names=list(report['results']),
    all36_input_pins_equal=ext['input_pins_before']==ext['input_pins_after'],
    external_wall_seconds=ext['wall_seconds'],family_private_peak=ext['family_private_peak'],
    all_four_exact_job_identities_dead=True,exact_job_deaths=deaths,protected=protected,
    compact_reader=dict(path=str(READER),bytes=len(reader_raw),sha256=hashlib.sha256(reader_raw).hexdigest(),value=reader),
    input_pins=[pin(p) for p in paths],
    sealer_and_launcher_alive_at_write_not_yet_exact_dead=True,
    no_candidate_or_fixture_retry=True,no_source_or_production_change=True,
    limitations=['Metadata harness failed before entire wrapper requirement test; later integrated main-path/phase/resource cases not reached.',
      'No production/GPU/model/audio/restore/copy/GC executed. Closed historical tail bytes not read.',
      'Completion own interpreter and launcher require later exact-death check by independent ROOT after return.'])
write(OUT,result)
print(json.dumps(dict(completion=pin(OUT),initial=pin(INITIAL),TOOL=pin(TOOL),entry=entry,
    exact_job_deaths=deaths,completed_results_count=len(report['results'])),sort_keys=True))
