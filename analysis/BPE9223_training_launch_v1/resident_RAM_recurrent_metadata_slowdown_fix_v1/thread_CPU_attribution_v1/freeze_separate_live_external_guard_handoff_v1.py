"""Inactive preparation only after genuine NEWcounter proof; never attaches to target."""
import ast,hashlib,json,os
from pathlib import Path
F=Path(__file__).parent;BASE=F.parent
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');B=Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01');I=B/'invocations/owner_000023'
OLD=F/'launch_finite_thread_counter_mapping_CPU_v3.py';NEW=F/'launch_existing_source31_thread_CPU_external_v1.py'
LIVE=F/'observe_existing_source31_thread_CPU_and_stacks_v2.py'
PROOF=F/'ROOT_new_thread_counter_nativeTID_mapping_actual_CPU_review_v1.json'
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 with p.open('x',encoding='utf-8',newline='\n')as f:json.dump(v,f,indent=2);f.write('\n')
assert os.environ['TORCH_EXTENSIONS_DIR'].lower().startswith('e:')
assert pin(PROOF)['sha256']=='8f2ee5f17afed28f420bdfa131e3e178965e0921bb92d6356d4b6a0caacffd52'
proof=read(PROOF)
assert proof['accepted']is True and proof['reviewer']=='/root'and proof['scope']=='ASR18_GetThreadTimes_new_toy_nativeTID_mapping_actual_CPU_v1'and proof['thread_counter_helper_source']==pin(F/'owned_thread_times_v2.py')and proof['all_exact_metadata_identities_dead']is True
assert pin(LIVE)['sha256']=='f36955cf7fb284b55c6ac8ac2fbbf7ef8714e763ab2f5c7b243e15cecf83f8cc'
raw=OLD.read_bytes();text=raw.decode('utf-8');edits=[]
def edit(old,new,n=1):
 global text
 assert text.count(old)==n,(old,text.count(old),n)
 text=text.replace(old,new);edits.append({'old_literal':old,'new_literal':new,'expected_count':n})
edit('One NEW finite thread-counter/nativeTID proof; no production attach; external full guards.','One separately reviewed existing-source31 nonblocking observation; external guard remains active through all parsing/writing/cleanup.')
edit("FIXTURE=F/'finite_thread_counter_mapping_fixture_v3.py'","FIXTURE=F/'observe_existing_source31_thread_CPU_and_stacks_v2.py'")
edit("RESULT=F/'finite_thread_counter_mapping_external_result_v2.json'","RESULT=F/'existing_source31_thread_CPU_external_result_v1.json'")
edit("OUTPUT=F/'finite_thread_counter_mapping_actual_02'","OUTPUT=F/'existing_source31_thread_CPU_actual_01'")
edit('ASR18_GetThreadTimes_new_toy_nativeTID_mapping_finite_CPU_v1','ASR18_existing_source31_GetThreadTimes_nonblocking60s_attribution_v1')
edit("review['fixture_source']==pin(FIXTURE)and review['launcher_source']==pin(__file__)","review['live_observer_source']==pin(FIXTURE)and review['external_live_launcher_source']==pin(__file__)")
edit('finite_thread_counter_mapping_external_CPU','existing_source31_thread_CPU_external_guard')
edit("stdout=F/'finite_thread_counter_mapping_v2_stdout.log';stderr=F/'finite_thread_counter_mapping_v2_stderr.log'","stdout=F/'existing_source31_thread_CPU_stdout.log';stderr=F/'existing_source31_thread_CPU_stderr.log'")
edit('ASR18_GetThreadTimes_new_toy_finite_CPU_external_result_v1','ASR18_existing_source31_thread_CPU_nonblocking_external_result_v1')
ast.parse(text);inverse=text
for e in reversed(edits):
 assert inverse.count(e['new_literal'])==e['expected_count'];inverse=inverse.replace(e['new_literal'],e['old_literal'])
assert inverse.encode('utf-8')==raw and not NEW.exists()
f2=ast.parse((F/'finite_thread_counter_mapping_fixture_v2.py').read_text(encoding='utf-8'));f3=ast.parse((F/'finite_thread_counter_mapping_fixture_v3.py').read_text(encoding='utf-8'))
def function(tree,name):return ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name),include_attributes=False)
assert function(f2,'profile_profiles')==function(f3,'profile_profiles'),'Live pure parser imported from frozenv2 is exact to real passedv3 proof'
with NEW.open('xb')as f:f.write(text.encode('utf-8'))
ledger={'kind':'ASR18_separate_existing_source31_LIVE_external_guard_exact_raw_inverse_v1','old_source':pin(OLD),'new_source':pin(NEW),'replacements':edits,'newline_normalization':False,'whole_old_raw_bytes_reconstructed':True,'actual_passed_v3_profile_parser_AST_exact_to_imported_v2':True,'no_helper_fixture_live_source_imported_or_executed':True}
write(F/'existing_source31_live_external_guard_exact_raw_inverse_v1.json',ledger)
sources=[LIVE,NEW,F/'owned_thread_times_v2.py',F/'finite_thread_counter_mapping_fixture_v2.py',F/'finite_thread_counter_mapping_fixture_v3.py',PROOF,F/'finite_thread_counter_mapping_actual02_completion_v1.json',F/'existing_source31_live_external_guard_exact_raw_inverse_v1.json',BASE/'py_spy_0_4_2_win_amd64_v1/py-spy.exe',BASE/'ROOT_pinned_py_spy_0_4_2_isolated_E_preparation_review_v1.json',P/'work/research/run_ASR18_BPE9223_resident_RAM_training_v31.py',P/'outputs/parakeet_turkish/asr18_BPE9223_private_scalar_prefetch_v1.py',I/'configuration.json',I/'boot.json',Path('E:/gpu_training_scratchpad/storage_config.json'),Path('E:/gpu_training_scratchpad/use_storage.ps1')]
for p in sources[:5]:ast.parse(p.read_text(encoding='utf-8'))
h={'kind':'ASR18_existing_source31_thread_CPU_and_stack_attribution_inactive_execution_handoff_v1','production_or_observation_execution_admitted':False,'requested_scope':'ASR18_existing_source31_GetThreadTimes_nonblocking60s_attribution_v1','live_observer_source':pin(LIVE),'external_live_launcher_source':pin(NEW),'completed_new_counter_CPU_review':pin(PROOF),'input_pins':[pin(p)for p in sources],'target_identity':{'pid':2456,'create_time':1792752131.0863705,'ppid':41672},'source31_owned_session':2901,'production_session_never_consumed':True,'output_path':str((F/'existing_source31_thread_CPU_actual_01').resolve()),'external_result_path':str((F/'existing_source31_thread_CPU_external_result_v1.json').resolve()),'exact_future_external_arguments':[str(NEW.resolve()),'--ROOT-review','EXACT_FINAL_ROOT_EXECUTION_SCOPE_REQUIRED'],'guard_scope':'External sampler watches exact own native+exact interpreter-launcher parent+observer/profiler descendants through startup, targetcounter/profile work, CJSON parse/serialization and finallycleanup. Own live parent is sampled but excluded from descendantwait/kill. Production target/family/reader never signaled or counted as owned cleanup.','resource_policy':{'duration_seconds':60,'profile_rate_Hz':100,'counter_poll_seconds':.1,'external_wall_seconds':120,'live_internal_wall_seconds':120,'sampled_total_owned_family_private_cap_bytes':536870912,'full_prospective_extra_bytes':536870912,'full_prefetch_reservation_bytes':268435456,'admission_commit_available_minimum_bytes':5055306368,'admission_physical_available_minimum_bytes':2305306368,'runtime_commit_available_minimum_bytes':2268435456,'runtime_physical_available_minimum_bytes':1768435456,'E_free_minimum_bytes':100000000000,'maximum_current_threads':128,'maximum_total_thread_identities':256,'maximum_counter_polls':700,'maximum_scalar_counter_records':80000,'counter_snapshot_cap_bytes':33554432,'profile_cap_bytes':67108864,'profile_logs_cap_each_bytes':4194304,'all4_math_thread_caps':1,'priority':'BelowNormal','observer_CUDA_VISIBLE_DEVICES':'-1','parent_production_environment_unchanged':True,'E_storage_config_and_script_applied_before_launch':True},'expected_actual_outputs':['Actual root/tool/Popen results and exact owned-process closure','Raw all-thread nonblocking profile and complete scalar counter JSONL','All bounded poll start/end brackets and launch/end alignment timestamps','Actual unique nativeTID+FILETIMEcreation mapping by MainThread exactdriver31/main and private-broad-row-prefetch exactprefetch1/_run name AND source frame','Unknown unsupported/hex/reused/terminated/open-race identities explicitly retained','Target process IO before/after and logical delta, not physicaldisk/perthreadphase evidence','Original target health/source/runtime before/after; no state change'],'forbidden':['SuspendThread/ResumeThread/debugger/injection/targetsignal/restart','--native/--gil/--locals/--subprocesses/script_launch','extraGC/GCpolicy/cushion/source/science/state changes','GPU/model/audio/journal/archive/tensor operation','second interval/automatic retry or closedproof rerun'],'interpretation_limits':['GetThreadTimes supplies actual per-thread user+kernel CPU across its counter brackets, inclusive of any C/finalizer work','All-thread profile supplies sampled wall occupancy; nonatomic failures/stderr preserved','Sequential thread query brackets and profiler launch/end uncertainty forbid exact exclusive CPU-by-stack/phase claims','No target MainThread=PID assumption, rank inference or scalar role by CPU usage','Guard samples can miss between-poll instantaneous peaks; limits are sampled guard enforcement, not a hard allocation bound','A completed observation is not a speed fix or a pureGC cause demonstration'],'finite_proof_audits_closed_no_rerun':True}
write(F/'existing_source31_thread_CPU_live_source_handoff_v1.json',h)
print(json.dumps({'handoff':pin(F/'existing_source31_thread_CPU_live_source_handoff_v1.json'),'external_source':pin(NEW),'live_observer_source':pin(LIVE),'inverse':pin(F/'existing_source31_live_external_guard_exact_raw_inverse_v1.json'),'no_attach_or_job_executed':True}))
