"""Preparation only: exact source replacement ledger, pins and inactive proof scope."""
import ast,hashlib,json,os
from pathlib import Path
F=Path(__file__).parent;BASE=F.parent
OLD=BASE/'launch_fresh_signature_collect_observation_CPU_v2.py';NEW=F/'launch_finite_thread_counter_mapping_CPU_v1.py'
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def write(p,v):
 with p.open('x',encoding='utf-8',newline='\n')as f:json.dump(v,f,indent=2);f.write('\n')
assert os.environ['TORCH_EXTENSIONS_DIR'].lower().startswith('e:')
raw=OLD.read_bytes();text=raw.decode('utf-8');edits=[]
def edit(old,new,count=1):
 global text
 assert text.count(old)==count,(old,text.count(old),count)
 text=text.replace(old,new);edits.append({'old_literal':old,'new_literal':new,'expected_count':count})
edit('One reviewed CPU metadata observation with external original resource guards.','One NEW finite thread-counter/nativeTID proof; no production attach; external full guards.')
edit("FIXTURE=F/'verify_fresh_signature_and_collect_observation_CPU_v1.py'","FIXTURE=F/'finite_thread_counter_mapping_fixture_v1.py'")
edit("RESULT=F/'fresh_signature_collect_observation_external_result_v1.json'","RESULT=F/'finite_thread_counter_mapping_external_result_v1.json'")
edit("OUTPUT=F/'fresh_signature_collect_observation_actual_CPU_01'","OUTPUT=F/'finite_thread_counter_mapping_actual_01'")
edit('ASR18_fresh_single_open_signature_and_collect_observation_CPU_v1','ASR18_GetThreadTimes_new_toy_nativeTID_mapping_finite_CPU_v1')
edit("Path(review['output']).resolve()","Path(review['output_path']).resolve()")
edit("if popen is not None:\n            try:\n                root=psutil.Process(popen.pid);family+=[root,*root.children(recursive=True)]","if popen is not None and popen_identity is not None and same_alive(popen_identity):\n            try:\n                root=psutil.Process(popen_identity['pid']);family+=[root,*root.children(recursive=True)]")
edit('fresh_signature_collect_external_CPU','finite_thread_counter_mapping_external_CPU')
edit("time.monotonic()-began<240,'Sampled family512MiB/external240s'","time.monotonic()-began<120,'Sampled family512MiB/external120s'")
edit("system['commit_available_bytes']>=2000000000+PREFETCH+prospective","system['commit_available_bytes']>=(4518435456+prospective if admission else 2000000000+PREFETCH)")
edit("Original floors plus FULLprefetch/prospective512MiB/E100GB","Admission FULL225+prefetch+prospective512MiB; runtime originalfloors+FULLprefetch; E100GB")
edit("stdout=F/'fresh_signature_collect_stdout.log';stderr=F/'fresh_signature_collect_stderr.log'","stdout=F/'finite_thread_counter_mapping_stdout.log';stderr=F/'finite_thread_counter_mapping_stderr.log'")
edit('ASR18_fresh_signature_collect_observation_external_result_v1','ASR18_GetThreadTimes_new_toy_finite_CPU_external_result_v1')
edit('(40012,31044,26892)','(2456,38964,26892,41672,34044,40012,31044)',2)
ast.parse(text)
rebuilt=text
for e in reversed(edits):
 assert rebuilt.count(e['new_literal'])==e['expected_count'],('Inverse count',e)
 rebuilt=rebuilt.replace(e['new_literal'],e['old_literal'])
assert rebuilt.encode('utf-8')==raw,'Entire original external launcher byte inverse'
with NEW.open('xb')as f:f.write(text.encode('utf-8'))
ledger={'kind':'ASR18_new_thread_CPU_finite_external_raw_inverse_v1','old_source':pin(OLD),'new_source':pin(NEW),'replacements':edits,'newline_normalization':False,'full_old_raw_bytes_reconstructed':True,'new_source_imported_or_executed':False}
write(F/'finite_external_exact_raw_inverse_v1.json',ledger)
sources=[F/'owned_thread_times_v1.py',F/'toy_native_threads_v1.py',F/'finite_thread_counter_mapping_fixture_v1.py',NEW]
for p in sources:ast.parse(p.read_text(encoding='utf-8'))
inputs=sources+[F/'finite_external_exact_raw_inverse_v1.json',BASE/'source31_remaining_slowdown_readonly_targeted_CPU_attribution_proposal_v1.json',BASE/'ROOT_source31_fresh_signature_collect_actual_long1600_result_verification_v1.json',BASE/'ROOT_pinned_py_spy_0_4_2_isolated_E_preparation_review_v1.json',BASE/'py_spy_0_4_2_win_amd64_v1/py-spy.exe',Path('E:/gpu_training_scratchpad/storage_config.json'),Path('E:/gpu_training_scratchpad/use_storage.ps1')]
h={'kind':'ASR18_new_GetThreadTimes_nativeTID_finite_CPU_source_handoff_v1','production_or_control_execution_admitted':False,'requested_exact_scope':'ASR18_GetThreadTimes_new_toy_nativeTID_mapping_finite_CPU_v1','fixture_source':pin(sources[2]),'launcher_source':pin(NEW),'input_pins':[pin(p)for p in inputs],'output_path':str((F/'finite_thread_counter_mapping_actual_01').resolve()),'external_result_path':str((F/'finite_thread_counter_mapping_external_result_v1.json').resolve()),'exact_future_external_arguments':[str(NEW),'--ROOT-review','ACTUAL_ROOT_SCOPE_REQUIRED'],'CPU_environment':{'interpreter':'E:/gpu_training_scratchpad/scratch/BPE_broad_CPU_preflight_env_v2/Scripts/python.exe','CUDA_VISIBLE_DEVICES':'-1','all4_math_thread_caps':'1','PYTHONUTF8':'1','python_flags':['-Xutf8','-B'],'priority':'BelowNormal','storage_script_applied_before_new_process':True},'resources':{'external_wall_seconds':120,'fixture_own_bound_seconds':60,'owned_family_private_bytes':536870912,'full_prospective_reserve_bytes':536870912,'full_prefetch_reserve_bytes':268435456,'admission_commit_available_minimum_bytes':5055306368,'admission_physical_available_minimum_bytes':2305306368,'runtime_commit_available_minimum_bytes':2268435456,'runtime_physical_available_minimum_bytes':1768435456,'E_free_floor_bytes':100000000000,'sampler_max_current_threads':128,'sampler_max_total_thread_identities':256,'sampler_max_polls':200,'sampler_max_scalar_records':20000,'profile_byte_cap':67108864,'new_toy_wall_seconds':10,'new_toy_profile_seconds':8,'new_toy_profile_rate_Hz':100},'proof_scope':['NEW real nativeTID/creation GetThreadTimes monotonic counters','retained same owned query handle across actual thread exit/join/process exit','wrongctime/wrongPID/invalidhandle/closedhandle/doubleclose negatives','NEW synthetic two-thread nemo-native CPU target and one8s pinnednonblocking profile solely for actual nativeTID+creation role mapping','actual MainThread and ToyScalarWorker CPU deltas, no PID=TID assumption','bounded thread inventory/polls/records; unknown/terminated/open-race explicit','METHOD_STUB reused-TID ambiguity and duplicate role rejection, not actual forced OSreuse','all owned query handles closed; actual Popen outcomes/owned process exactdeath; no production target attach'], 'limitations':['synthetic scalar math target; not training/native/model/GC speed or fit proof','py-spy nonblocking nonatomic errors retained','thread CPU is user+kernel actual CPU, not exclusive phase cost','sampled wall callsite weights and CPU interval cannot be added or called exact CPU by stack','no extra gc.collect/freezing/threshold/import policy','does not rerun old closed py-spy toy proof; this is a NEW counter/nativeTID pairing proof'], 'live_execution_pending_separate_scope':{'target_pid':2456,'target_create_time':1792752131.0863705,'source':'source31','owned_training_session':2901,'live_profile_seconds':60,'counter_poll_seconds':.1,'never_signal_suspend_inject_or_restart_target':True}}
write(F/'finite_thread_counter_mapping_source_handoff_v1.json',h)
print(json.dumps({'handoff':pin(F/'finite_thread_counter_mapping_source_handoff_v1.json'),'ledger':pin(F/'finite_external_exact_raw_inverse_v1.json'),'sources':[pin(p)for p in sources],'no_fixture_or_helper_imported':True}))
