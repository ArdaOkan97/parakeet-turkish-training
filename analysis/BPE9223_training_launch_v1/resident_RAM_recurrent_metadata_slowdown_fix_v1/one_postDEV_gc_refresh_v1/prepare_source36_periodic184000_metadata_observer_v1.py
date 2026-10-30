"""Inactive routine184k metadata source preparation only; never execute it."""
import ast,hashlib,json,os,shutil,sys,time
from pathlib import Path
import psutil
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from resource_watchdog_v2 import sample_memory
F=Path(__file__).resolve().parent;N=F.parent.parent;A=N/'resident_RAM_training_actual_26'
OLD=F/'seal_periodic180000_source36_CPU13_helper3_max2_saved180200_v1.py'
NEW=F/'seal_periodic184000_source36_CPU13_helper3_max2_saved184200_v1.py'
LEDGER=F/'source36_periodic184000_metadata_observer_entire_RAW_inverse_v1.json'
HANDOFF=F/'inactive_source36_periodic184000_metadata_observer_handoff_v1.json'
RESULT=F/'source36_periodic184000_inactive_preparation_result_v1.json'
SCOPE=F/'ROOT_periodic184000_source36_CPU13_metadata_observer_inactive_preparation_scope_v1.json'
ME=psutil.Process();ME.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);BEGIN=time.monotonic()
def require(v,m):
 if not v:raise ValueError(m)
def data_pin(p,b):return {'path':str(p.resolve()),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def pin(p):
 require(p.stat().st_size<=8*1024**2,'Bounded source/metadata pin');return data_pin(p,p.read_bytes())
def write(p,v):
 with p.open('xb')as f:f.write((json.dumps(v,indent=2)+'\n').encode());f.flush();os.fsync(f.fileno())
def guard():
 s=sample_memory(ME.pid,'source36_inactive_periodic184000_metadata_preparation');m=s['system_memory']
 require(s['paired_sampling_within_one_second'] and s['worker_private_memory']['private_bytes']<536870912 and time.monotonic()-BEGIN<180 and m['commit_available_bytes']>=5055306368 and m['physical_available_bytes']>=2305306368 and shutil.disk_usage('E:/').free>=100000000000,'Original fullprospective512/full225/FULLprefetch/floors/E100GB metadata limits')
 return s
require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1' and os.environ.get('PYTHONUTF8')=='1' and all(os.environ.get(k)=='1'for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')) and Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad')),'Process-local E CPU-only metadata preparation')
initial=guard();require(pin(SCOPE)['sha256']=='48a5bbfe6644516eeef10a45b59b2aa5d2172324362e41865995e0f8098d3e47' and pin(OLD)['sha256']=='e37bc503fd4e4e8e173ddde7e264c217335f1157ed4b0d4a9af328b780e1690f','Exact scope and frozen180k source')
future=[A/'actual_periodic184000_and_resumed_training_status.json',A/'actual_periodic184000_postpositive_progress.json',A/'actual_periodic184000_initial_full_resource_sample_v1.json']
require(all(not p.exists()for p in [NEW,LEDGER,HANDOFF,RESULT,*future]),'Fresh immutable sources and future output paths')
raw=OLD.read_bytes();out=raw;edits=[]
def replace(a,b,expected=None):
 global out
 aa,bb=a.encode(),b.encode();count=out.count(aa);require(count>0 and(expected is None or count==expected),'Exact declared literal '+a)
 out=out.replace(aa,bb);edits.append({'old_literal':a,'new_literal':b,'expected_count':count})
for a,b in [('180000','184000'),('180200','184200'),('180k','184k'),('176000','180000'),('176k','180k'),('1220','4020')]:replace(a,b)
newline='\r\n'if b'\r\n'in raw else'\n'
capture="""initial=guard()
initial_path=A/'actual_periodic184000_initial_full_resource_sample_v1.json'
with initial_path.open('x',encoding='utf-8',newline='\\n')as initial_file:
 json.dump({'kind':'ASR18_periodic184000_original_initial_full_metadata_resource_sample_v1',
  'sample':initial,'original_runtime_guard_returned':True,
  'required_commit_bytes':4518435456+512*1024**2,
  'required_physical_bytes':1500000000+268435456+512*1024**2,
  'full_prospective_predicate_passed':initial['system_memory']['commit_available_bytes']>=4518435456+512*1024**2 and initial['system_memory']['physical_available_bytes']>=1500000000+268435456+512*1024**2,
  'captured_before_original_full_prospective_require_and_DEV_metadata_reads':True,
  'same_original_sample_no_extra_resource_query':True,
  'sealer_identity':{'pid':os.getpid(),'create_time':psutil.Process().create_time(),'ppid':os.getppid()},
  'sealer_launcher_identity':{'pid':os.getppid(),'create_time':psutil.Process(os.getppid()).create_time(),'ppid':psutil.Process(os.getppid()).ppid()},
  'metadata_source':pin(Path(__file__))},initial_file,indent=2)
 initial_file.write('\\n');initial_file.flush();os.fsync(initial_file.fileno())
require(""".replace('\n',newline)
replace('initial=guard();require(',capture,1)
replace("'last_metadata_resource_sample':guard()","'initial_metadata_full_resource_sample':pin(initial_path),'last_metadata_resource_sample':guard()",1)
inverse=out
for e in reversed(edits):
 a,b=e['old_literal'].encode(),e['new_literal'].encode();require(inverse.count(b)==e['expected_count'],'Unambiguous entire RAW inverse');inverse=inverse.replace(b,a)
require(inverse==raw,'FULL original180k raw bytes reconstructed')
tree=ast.parse(out.decode('utf-8-sig'))
initial_writes=[n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and isinstance(n.func.value,ast.Name)and n.func.value.id=='initial_file'and n.func.attr=='write']
initial_opens=[n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and isinstance(n.func.value,ast.Name)and n.func.value.id=='initial_path'and n.func.attr=='open']
require(len(initial_writes)==len(initial_opens)==1 and initial_writes[0].args[0].value=='\n'and next(k.value.value for k in initial_opens[0].keywords if k.arg=='newline')=='\n','New receipt uses real LF newline constants, never literal backslash-n data')
ledger={'kind':'ASR18_periodic184000_source36_metadata_observer_entire_RAW_inverse_v1','preparation_scope':pin(SCOPE),'old_source':data_pin(OLD,raw),'new_source':data_pin(NEW,out),'edits_in_forward_order':edits,'newline_normalization':'NONE; unchanged bytes exact; new explicit receipt block uses detected base line endings','full_RAW_forward_inverse_exact':True,'explicit_semantic_additions':['Persist exact original first returned guard sample before unchanged fullprospective predicate, including its boolean outcome and actual metadata identities','Include actual initial receipt pin in successful status'],'original_runtime_and_full_prospective_predicates_thresholds_and_resource_query_count_unchanged':True,'no_GC_event_or_scientific_operation_added':True}
ledger_bytes=(json.dumps(ledger,indent=2)+'\n').encode();ledger_pin=data_pin(LEDGER,ledger_bytes)
handoff={'kind':'ASR18_inactive_periodic184000_source36_metadata_observer_handoff_v1','execution_admitted':False,'preparation_scope':pin(SCOPE),'source':data_pin(NEW,out),'full_RAW_inverse':ledger_pin,'immutable_base':data_pin(OLD,raw),'exact_future_command':['E:/gpu_training_scratchpad/scratch/BPE_broad_CPU_preflight_env_v2/Scripts/python.exe','-Xutf8','-B',str(NEW.resolve())],'requires_separate_ROOT_source_and_execution_scope':True,'current_native':{'pid':21912,'create_time':1792780981.1092248,'ppid':37108},'expected_old_CPU':{'pid':40004,'create_time':1792783315.0228927,'ppid':21652},'expected_old_CPU_launcher':{'pid':21652,'create_time':1792783315.0045743,'ppid':21912},'owned_production_session':42997,'invocation':25,'future_actual_prerequisites':['Original184000 ordered7x50cleanDEV350/normalCPU13helper3MAX2/LIVE1 phase','Actual4020acquires=posts/4000native+20wholeunconsumeddrain/wait0/joins/pipes/bothcreditsetsclosed/oldcohortexactdead','Same original queued plan/no regeneration or reorder/newCPU13MAX2freshnonce/resumedpositive','184000+184200 full854/207/215semantic independentPRE-SAVED and actual>=184220/no failure/same native/source/config'],'future_output_paths':[str(p.resolve())for p in future],'initial_numeric_sample_capture':'One actual original guard() result is exclusively written before the unchanged fullprospective require and any DEV metadata read; receipt therefore survives fullprospective rejection when writing succeeds. A rejection inside original first runtime guard before return remains UNRECORDED; no reconstructed sample or false all-failure capture claim.','initial_sample_adds_no_memory_query_and_changes_no_original_predicate_or_budget':True,'limits':{'private_bytes':536870912,'wall_seconds':180,'prelaunch_commit':5055306368,'prelaunch_physical':2305306368,'runtime_commit':2268435456,'runtime_physical':1768435456,'E_free':100000000000,'not_a_new_fullfamily_CPUproof':True},'historical_once180k_refresh_ROOT_receipt':{'path':str((F/'ROOT_source36_actual_first_periodic180000_idle_refresh_preservation_verification_v1.json').resolve()),'bytes':2877,'sha256':'7caeec0f471f63027b0566e8869a131920a79a732421994b6bd6a6a6ca708e9a'},'historical_once_refresh_reference_only_not_reread_or_reaudited':True,'new184k_GC_event_capture_sync_or_worker_refresh':False,'environment':'Fresh E storage_config plus process-local use_storage includingTORCH_EXTENSIONS_DIR; CPU-1/fourcaps1/UTF8/-B/BelowNormal; current process environments untouched','actual_TOOL_failure_and_completion_exact_metadata_identities_or_explicit_UNKNOWN_required':True,'no_owner_journal_tensor_audio_archive_read_or_extraDEV_or_science_LR_source_guard_or_reader_change':True,'no_automatic_rerun':True}
handoff_bytes=(json.dumps(handoff,indent=2)+'\n').encode();final=guard()
for p,b in [(NEW,out),(LEDGER,ledger_bytes),(HANDOFF,handoff_bytes)]:
 with p.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
result={'kind':'ASR18_periodic184000_source36_inactive_preparation_result_v1','passed':True,'source':pin(NEW),'ledger':pin(LEDGER),'handoff':pin(HANDOFF),'writer':pin(Path(__file__)),'initial_guard':initial,'final_guard':final,'metadata_identity':{'pid':ME.pid,'create_time':ME.create_time(),'ppid':ME.ppid()},'metadata_launcher_identity':{'pid':ME.ppid(),'create_time':ME.parent().create_time(),'ppid':ME.parent().ppid()},'candidate_import_or_execution':False,'future_outputs_absent':all(not p.exists()for p in future)}
write(RESULT,result);print(json.dumps({'result':pin(RESULT),'source':pin(NEW),'ledger':pin(LEDGER),'handoff':pin(HANDOFF),'execution_admitted':False}),flush=True)
