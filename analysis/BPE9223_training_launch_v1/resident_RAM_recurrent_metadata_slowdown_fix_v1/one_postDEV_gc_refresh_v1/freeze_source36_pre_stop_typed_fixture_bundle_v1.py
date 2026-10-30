"""Freeze source/metadata only. AST parsing is not candidate import or proof execution."""
import ast,difflib,hashlib,json,os
from pathlib import Path
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish';R=P/'work/research'
F=Path(__file__).resolve().parent;FIX=F.parent;FAIL=FIX.parent/'resident_RAM_original_interval170469_failure_actual_v1'
def pin(p):
 p=Path(p).resolve();raw=p.read_bytes();return {'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(v,f,indent=2,ensure_ascii=False);f.write('\n')
def successor(old,new,changes):
 raw=old.read_bytes();text=raw.decode();ledger=[]
 for a,b,n in changes:
  assert text.count(a)==n,(a,text.count(a),n);text=text.replace(a,b);ledger.append({'old_literal':a,'new_literal':b,'expected_count':n})
 ast.parse(text);inverse=text
 for x in reversed(ledger):
  assert inverse.count(x['new_literal'])==x['expected_count'],x['new_literal']
  inverse=inverse.replace(x['new_literal'],x['old_literal'])
 assert inverse.encode()==raw and not new.exists();new.write_bytes(text.encode())
 diff=new.with_suffix('.diff')
 with diff.open('x',encoding='utf-8',newline='\n') as f:f.write(''.join(difflib.unified_diff(old.read_text().splitlines(True),text.replace('\r\n','\n').splitlines(True),fromfile=str(old),tofile=str(new))))
 return {'old':pin(old),'new':pin(new),'replacements':ledger,'newline_normalization':'NONE: raw literal replacements',
  'entire_RAW_forward_and_inverse_exact':True,'diff':pin(diff)}
assert Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad'))
fixture=F/'verify_source36_pre_stop_entire_selector_metadata_v1.py'
bridge=F/'source36_eligible_periodic_branch_METADATA_DOUBLES_v1.py'
for p in (fixture,bridge):ast.parse(p.read_text(encoding='utf-8-sig'))
external=F/'launch_source36_pre_stop_typed_metadata_external_v1.py'
ext=successor(F/'launch_one_postDEV_refresh_finite_CPU_v2.py',external,[
 ('One focused CPython postDEV-refresh finite proof; no production operation; full external guards.','One source36 entire-selector metadata proof; no production operation; full external guards.',1),
 ('verify_one_postDEV_refresh_finite_CPU_v2.py','verify_source36_pre_stop_entire_selector_metadata_v1.py',1),
 ('one_postDEV_refresh_finite_CPU_external_result_v2.json','source36_pre_stop_typed_metadata_external_result_v1.json',1),
 ('finite_one_postDEV_refresh_actual_CPU_01','source36_pre_stop_typed_selector_actual_01',1),
 ('ASR18_one_postDEV_refresh_finite_CPU_v1','ASR18_source36_pre_stop_entire_selector_metadata_CPU_v1',1),
 ('one_postDEV_refresh_finite_external_CPU','source36_pre_stop_typed_metadata_external_CPU',1),
 ('one_postDEV_refresh_finite_CPU_v2_stdout.log','source36_pre_stop_typed_metadata_stdout_v1.log',1),
 ('one_postDEV_refresh_finite_CPU_v2_stderr.log','source36_pre_stop_typed_metadata_stderr_v1.log',1),
 ('ASR18_one_postDEV_refresh_finite_CPU_external_result_v1','ASR18_source36_pre_stop_typed_metadata_external_result_v1',1)])
old_fixture=FAIL/'verify_source33_eighth_actual_failure_entire_selector_metadata_v1.py'
newraw=fixture.read_bytes();oldraw=old_fixture.read_bytes()
fixture_record={'old':pin(old_fixture),'new':pin(fixture),'replacements':[{
 'old_literal':oldraw.decode(),'new_literal':newraw.decode(),'expected_count':1}],
 'purpose':'Entire new explicit in-memory future-stop framework; this is a broad fixture rewrite, NOT an unchanged-algorithm claim.',
 'newline_normalization':'NONE: whole RAW literal replacement','entire_RAW_forward_and_inverse_exact':True}
assert oldraw.replace(oldraw,newraw,1)==newraw and newraw.replace(newraw,oldraw,1)==oldraw
ledger=F/'source36_pre_stop_typed_fixture_external_exact_RAW_ledger_v1.json'
write(ledger,{'fixture_entire_literal_rewrite':fixture_record,'external_narrow_successor':ext,
 'eligible_bridge_is_new_source_not_an_equivalence_claim':pin(bridge),
 'production_source36_unchanged_and_reuses_closed_direct33_inverse':pin(F/'helper2_source35_to36_and_full33_RAW_inverse_v1.json'),
 'fixture_and_bridge_AST_syntax_only_checked_no_execution':True})
scope=F/'ROOT_source36_typed_transition_checkpoint_stop_inactive_preparation_scope_v1.json'
proof=F/'ROOT_one_postDEV_refresh_actual_finite_CPU_review_v1.json'
candidate=F/'candidate_source36_configuration_completed_CPU_pending_actual_stop_v1.json'
c=read(candidate);assert c['prior_resident_RAM_one_postDEV_gc_refresh_checkpoint_transition'] is None
assert c['one_postDEV_gc_refresh_policy']['completed_CPU_review']==pin(proof)
paths=[fixture,external,bridge,candidate,proof,Path(read(proof)['actual_CPU_control_report']['path']),scope,
 F/'inactive_source36_one_postDEV_refresh_source_handoff_v1.json',F/'helper2_source35_to36_and_full33_RAW_inverse_v1.json',
 F/'inactive_source36_checkpoint_stop_source_handoff_v1.json',F/'inactive_source33_one_postDEV_refresh_checkpoint_transition_intent_v1.json',
 F/'observe_one_postDEV_refresh_natural_checkpoint_stop_v1.py',F/'seal_actual_source33_one_postDEV_refresh_checkpoint_transition_v1.py',
 F/'source33_to36_checkpoint_observer_sealer_exact_RAW_inverse_v1.json',ledger,
 R/'run_ASR18_BPE9223_resident_RAM_training_v36.py',R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v36.py',
 S/'asr18_one_postDEV_gc_refresh_checkpoint_transition_v2.py',F/'asr18_one_postDEV_gc_refresh_v2.py',F/'asr18_postDEV_refresh_state_capture_v1.py',
 S/'asr18_live_state_fingerprint_v1.py',R/'run_ASR18_native_throughput_component_v15.py',S/'asr18_BPE9223_grouped_RAM_native_v1.py',
 S/'asr18_BPE9223_private_scalar_prefetch_v1.py',FIX/'asr18_observe_original_gc_collect_v2.py',
 FIX/'ROOT_one_postDEV_gc_refresh_inactive_preparation_scope_v1.json',FIX/'ROOT_one_postDEV_gc_refresh_inactive_preparation_semantic_verification_amendment_v1.json',
 F/'ROOT_source35_and_finite_v1_preexecution_findings_and_successor_preparation_scope_v1.json',
 FAIL/'production_configuration_source32_v1/configuration.json',FIX/'ROOT_direct_human_then_fix_authorization_v1.json',
 R/'run_ASR18_BPE9223_resident_RAM_training_v33.py',R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v33.py',
 S/'fastmss_resident_feature_stream_v13.py',S/'resource_watchdog_v2.py',S/'bpe_broad_audio_v1.py',S/'atomic_io_retry_v1.py',
 Path('E:/gpu_training_scratchpad/storage_config.json'),Path('E:/gpu_training_scratchpad/use_storage.ps1')]
inputs=[pin(p) for p in paths];assert len({x['path'] for x in inputs})==len(inputs)
manifest=F/'source36_pre_stop_typed_metadata_input_manifest_v1.json';write(manifest,{'kind':'ASR18_source36_pre_stop_typed_metadata_input_manifest_v1','input_pins':inputs})
schema=F/'required_source36_pre_stop_typed_metadata_actual_review_schema_v1.json'
write(schema,{'scope':'ASR18_source36_pre_stop_entire_selector_metadata_CPU_v1','accepted':True,'reviewer':'/root',
 'fixture_source':pin(fixture),'driver_source':pin(R/'run_ASR18_BPE9223_resident_RAM_training_v36.py'),
 'actual_CPU_control_report':'EXACT actual scoped metadata report pin, not placeholder promoted to real acceptance',
 'all_exact_metadata_identities_dead':True,'all_state_GPU_or_actual_intentional_stop_tested':False,
 'production_stop_or_launch_admitted_by_metadata_gate':False})
handoff=F/'inactive_source36_pre_stop_typed_metadata_fixture_handoff_v1.json'
write(handoff,{'kind':'ASR18_INACTIVE_source36_pre_stop_entire_selector_metadata_fixture_handoff_v1',
 'preparation_scope':pin(scope),'fixture_source':pin(fixture),'external_launcher_source':pin(external),'eligible_branch_bridge_source':pin(bridge),
 'source_configuration':pin(candidate),'driver_source':pin(R/'run_ASR18_BPE9223_resident_RAM_training_v36.py'),
 'wrapper_source':pin(R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v36.py'),
 'transition_validator':pin(S/'asr18_one_postDEV_gc_refresh_checkpoint_transition_v2.py'),'actual_CPU_review':pin(proof),
 'transition_intent':pin(F/'inactive_source33_one_postDEV_refresh_checkpoint_transition_intent_v1.json'),
 'manifest':pin(manifest),'input_pins':inputs,'exact_RAW_ledger':pin(ledger),'required_actual_review_schema':pin(schema),
 'future_exact_external_arguments':[str(Path('E:/gpu_training_scratchpad/scratch/BPE_broad_CPU_preflight_env_v2/Scripts/python.exe')),
  '-Xutf8','-B',str(external),'--ROOT-review','SEPARATE_FUTURE_ROOT_EXACT_CPU_SCOPE_REQUIRED'],
 'output_path':str(F/'source36_pre_stop_typed_selector_actual_01'),'external_result_path':str(F/'source36_pre_stop_typed_metadata_external_result_v1.json'),
 'limits':{'internal_seconds':100,'external_seconds':120,'sampled_full_owned_family_private_bytes':536870912,
  'exact_external_interpreter_launcher_in_sampled_family_not_owned_cleanup':True,'additional_prospective_bytes':536870912,
  'full_admission_commit_bytes':5055306368,'full_admission_physical_bytes':2305306368,
  'runtime_commit_floor_with_fullprefetch_bytes':2268435456,'runtime_physical_floor_with_fullprefetch_bytes':1768435456,
  'E_free_bytes':100000000000,'CUDA_VISIBLE_DEVICES':'-1','math_thread_caps':1,'priority':'BelowNormal','automatic_rerun':False},
 'honest_doubles':['future closure/intended stop/pre-stop review/checkpoint/persistence/Monitor/deaths are STRICT IN-MEMORY structural doubles only',
  'old-journal STAT and preserved tail bytes are virtual, never original24 or other journal access',
  'prefix view/journal binding/validated-result forwarding and wrapper execution review are explicit in-memory doubles',
  'eligible driver worker/DB/GC/capture owners and scalar final/error closure use METHOD_STUBS; actual source AST only, no real event',
  'Scientific full854 GPU equality and real DB/opaque-framework graph behavior remain unproved'],
 'coverage':['actual full typed validator and real completedCPU/source policy; proper372 inverse; stale/None/source/checkpoint/cohort/prefix/Monitor negatives',
  'complete actual driver entry/shared pre/post selectors, whole factory and postrestore dictionaries; record(None) reproducer; strict STAT-only condition',
  'whole wrapper lane requirements/initial25 forwarding; actual both config arguments and original7 runtime binding KINDS',
  'entire eligible driver AST: exact drops, joined old weakref death, barrier/admission/capture owner args, SAME queued tuple/pin, recreate order and new-generation final/error close'],
 'proof_execution_admitted':False,'production_stop_admitted':False,'production_launch_admitted':False,
 'candidate_driver_helper_validator_imported_during_preparation':False,'any_GC_call_during_preparation':False})
print(json.dumps({'fixture':pin(fixture),'external':pin(external),'eligible_bridge':pin(bridge),'handoff':pin(handoff),
 'manifest':pin(manifest),'RAW_ledger':pin(ledger),'input_count':len(inputs)}),flush=True)
