"""Source/hash/AST metadata only; no candidate imports or GC operations."""
import ast,difflib,hashlib,json,os
from pathlib import Path
F=Path(__file__).parent;FIX=F.parent;D=FIX/'thread_CPU_attribution_v1'
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish'
def pin(path):
 p=Path(path).resolve();raw=p.read_bytes();return {'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def write(path,value):
 with Path(path).open('x',encoding='utf-8',newline='\n')as out:json.dump(value,out,indent=2,ensure_ascii=False);out.write('\n')
assert os.environ['TORCH_EXTENSIONS_DIR'].lower().startswith('e:')
old=D/'launch_fixed_counter_controller_METHOD_STUB_v1.py';new=F/'launch_one_postDEV_refresh_finite_CPU_v1.py'
raw=old.read_bytes();text=raw.decode()
edits=[]
def change(a,b,n=1):
 global text
 if '\n'in a and text.count(a.replace('\n','\r\n'))==n:a=a.replace('\n','\r\n');b=b.replace('\n','\r\n')
 assert text.count(a)==n,(a,text.count(a),n);edits.append({'old_literal':a,'new_literal':b,'expected_count':n});text=text.replace(a,b)
change('One NEW finite thread-counter/nativeTID proof; no production attach; external full guards.',
       'One focused CPython postDEV-refresh finite proof; no production operation; full external guards.')
change('verify_fixed_counter_controller_METHOD_STUB_v1.py','verify_one_postDEV_refresh_finite_CPU_v1.py')
change('fixed_counter_controller_external_result_v1.json','one_postDEV_refresh_finite_CPU_external_result_v1.json')
change('fixed_counter_controller_actual_01','finite_one_postDEV_refresh_actual_CPU_01')
change("    review=json.loads(Path(args.ROOT_review).read_text(encoding='utf-8'))\n",
       "    review=json.loads(Path(args.ROOT_review).read_text(encoding='utf-8'))\n    protected_runtime_pids={3652,31268,30732,26892,30128}\n    protected_runtime_pids.update(who['pid']for who in review['protected_runtime_identities'])\n")
change('ASR18_fixed60_counter_window_persistence_METHOD_STUB_CPU_v1','ASR18_one_postDEV_refresh_finite_CPU_v1')
change('finite_thread_counter_mapping_external_CPU','one_postDEV_refresh_finite_external_CPU')
change('fixed_counter_controller_stdout.log','one_postDEV_refresh_finite_CPU_stdout.log')
change('fixed_counter_controller_stderr.log','one_postDEV_refresh_finite_CPU_stderr.log')
change('ASR18_fixed_counter_window_METHOD_STUB_external_result_v1','ASR18_one_postDEV_refresh_finite_CPU_external_result_v1')
change('not in (2456,38964,26892,41672,34044,40012,31044)','not in protected_runtime_pids',2)
change("and not result.get('not_closed_exact_identities')and result.get('actual_report_passed')is True",
       "and not result.get('not_closed_exact_identities')and not result.get('cleanup_error')and result.get('actual_report_passed')is True")
ast.parse(text);assert not new.exists();new.write_bytes(text.encode())
inverse=text
for e in reversed(edits):
 assert inverse.count(e['new_literal'])==e['expected_count'];inverse=inverse.replace(e['new_literal'],e['old_literal'])
assert inverse.encode()==raw
fixture=F/'verify_one_postDEV_refresh_finite_CPU_v1.py';ast.parse(fixture.read_text(encoding='utf-8'))
ledger=F/'finite_CPU_external_exact_raw_inverse_v1.json'
write(ledger,{'kind':'ASR18_one_postDEV_finite_external_full_RAW_inverse_v1','old':pin(old),'new':pin(new),
 'newline_policy':'Actual RAW literals preserve original mixed line endings exactly; no normalization',
 'replacements':edits,'full_old_RAW_inverse_exact':True,'new_scope_is_not_actual_execution_authorization':True})
schema=F/'required_completed_CPU_review_schema_v1.json'
write(schema,{'kind':'ASR18_one_postDEV_refresh_required_completed_CPU_schema_v1',
 'accepted':{'type':'boolean','required_value':True},'reviewer':{'required_value':'/root'},
 'scope':{'required_value':'ASR18_one_first_periodic_DEV_quiescent_gc_refresh_policy_v1'},
 'helper_source':pin(F/'asr18_one_postDEV_gc_refresh_v1.py'),
 'capture_source':pin(F/'asr18_postDEV_refresh_state_capture_v1.py'),
 'actual_CPU_control_report':{'type':'canonical_exact_file_pin','must_be_actual':True},
 'all_exact_metadata_identities_dead':{'type':'boolean','required_value':True},
 'scientific_GPU_full854_equality_in_CPU_proof':False,'opaque_framework_cycle_absence_proved':False,
 'production_admitted_by_CPU_gate':False})
inputs=[fixture,new,F/'asr18_one_postDEV_gc_refresh_v1.py',F/'asr18_postDEV_refresh_state_capture_v1.py',
 F/'inactive_one_postDEV_refresh_helpers_handoff_v1.json',F/'state_capture_options_and_explicit_amendment_v1.json',
 schema,ledger,FIX/'ROOT_one_postDEV_gc_refresh_inactive_preparation_scope_v1.json',
 FIX/'ROOT_one_postDEV_gc_refresh_inactive_preparation_semantic_verification_amendment_v1.json',
 S/'asr18_live_state_fingerprint_v1.py',S/'asr18_BPE9223_grouped_RAM_native_v1.py',
 S/'asr18_BPE9223_private_scalar_prefetch_v1.py',P/'work/research/run_ASR18_native_throughput_component_v15.py',
 FIX/'asr18_observe_original_gc_collect_v2.py',S/'resource_watchdog_v2.py',
 Path('E:/gpu_training_scratchpad/storage_config.json'),Path('E:/gpu_training_scratchpad/use_storage.ps1')]
handoff=F/'inactive_one_postDEV_refresh_finite_CPU_source_handoff_v1.json'
write(handoff,{'kind':'ASR18_INACTIVE_one_postDEV_refresh_finite_CPU_source_handoff_v1',
 'fixture_source':pin(fixture),'launcher_source':pin(new),'exact_raw_inverse':pin(ledger),
 'required_completed_CPU_review_schema':pin(schema),'input_pins':[pin(p)for p in inputs],
 'exact_future_external_arguments':['-Xutf8','-B',str(new),'--ROOT-review','FUTURE_EXACT_ROOT_FINITE_CPU_SCOPE_REQUIRED'],
 'output_path':str(F/'finite_one_postDEV_refresh_actual_CPU_01'),
 'external_result_path':str(F/'one_postDEV_refresh_finite_CPU_external_result_v1.json'),
 'limits':{'fixture_internal_seconds':100,'external_seconds':120,'sampled_owned_family_private_bytes':536870912,
  'exact_external_interpreter_launcher_included_in_sampling_not_owned_cleanup':True,
  'additional_prospective_bytes':536870912,'full_prefetch_bytes':268435456,
  'full_admission_commit':5055306368,'full_admission_physical':2305306368,
  'runtime_commit':2268435456,'runtime_physical':1768435456,'E_free_bytes':100000000000,
  'all4_math_threadcaps':1,'CUDA_VISIBLE_DEVICES':'-1','priority':'BelowNormal'},
 'honest_scope':{'actual_CPythongc_cycles_weakrefs_and_builtin_alias':True,
  'actual_original_close_AST_on_toy_constructor_and_reader':True,'genuine_production_DB_locks_tested':False,
  'scientific_torch_fingerprint':'METHOD_STUB plus original-source binding only',
  'no_authoritative_sampler_draw_or_production_state_write':True,
  'freeze_retention_negative_is_expected_hazard_not_a_memory_saving_claim':True},
 'production_trial_or_actual_CPU_execution_admitted':False,'completed_CPU_review':None})
print(json.dumps({'fixture':pin(fixture),'external':pin(new),'ledger':pin(ledger),'schema':pin(schema),'handoff':pin(handoff)}))
