"""Preparation only: AST syntax, immutable metadata/runtime correction and handoff.
Never imports or executes fixture, successor, validator or policy modules.
"""
import ast,copy,hashlib,json
from pathlib import Path
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish';R=P/'work/research'
F=Path(__file__).resolve().parent;FIX=F.parent/'resident_RAM_recurrent_metadata_slowdown_fix_v1'
B=Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01')
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 p=Path(p)
 if p.name in ('source33_configuration_and_future7_runtime_bindings_v2.json','source33_runtime7_v1_to_v2_report_only_correction_v1.json')and p.exists():
  assert read(p)==v;return
 assert not p.exists(),p;p.write_bytes((json.dumps(v,indent=2)+'\n').encode())
driver=R/'run_ASR18_BPE9223_resident_RAM_training_v33.py';wrapper=R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v33.py'
validator=F/'asr18_eighth_original_interval_checkpoint_resume_v1.py'
fixture=F/'verify_source33_eighth_actual_failure_entire_selector_metadata_v1.py';external=F/'launch_source33_eighth_selector_metadata_external_v1.py'
for path in (driver,wrapper,validator,fixture,external):ast.parse(path.read_bytes())
configpin=pin(F/'production_configuration_source32_v1/configuration.json');config=read(configpin['path'])
v1=read(F/'source33_configuration_and_future7_runtime_bindings_v1.json');v2=copy.deepcopy(v1)
v2['future_runtime_values']['resident_RAM_stream_configuration']=configpin
assert v2['future_runtime_values']['source_configuration']==configpin
write(F/'source33_configuration_and_future7_runtime_bindings_v2.json',v2)
write(F/'source33_runtime7_v1_to_v2_report_only_correction_v1.json',{'old':pin(F/'source33_configuration_and_future7_runtime_bindings_v1.json'),
 'new':pin(F/'source33_configuration_and_future7_runtime_bindings_v2.json'),'only_changed_field':'future_runtime_values.resident_RAM_stream_configuration',
 'old_value':v1['future_runtime_values']['resident_RAM_stream_configuration'],'new_value':configpin,
 'source_basis':'actual unchanged wrapper33 --configuration and --stream-configuration both receive a.configuration',
 'source_configuration_control_or_production_changed':False})
closure=pin(F/'actual_failure_closure.json');value=read(closure['path']);pointer=value['saved_full_checkpoint']['pointer']
oldhandoff=read(FIX/'final_source31_signature_collect_transition_execution_handoff_v1.json')
args=['--review','__PENDING_FINAL_ROOT_EXECUTION_REVIEW__','--configuration',configpin['path'],'--pointer',pointer['path'],
 '--latest-failure',oldhandoff['latest_failure']['path'],'--branch',str(B),'--output',str(F.parent/'resident_RAM_training_actual_25'),
 '--queue-bytes','268435456','--start-invocation','24','--eighth-original-resume',closure['path']]
assert len(args)==18
inputs=[];seen=set()
def add(p):
 item=pin(p)
 if item['path']not in seen:seen.add(item['path']);inputs.append(item)
for path in (fixture,external,driver,wrapper,validator,configpin['path'],FIX/'production_configuration_source31_v1/configuration.json',
 F/'source31_to33_entire_RAW_inverse_v1.json',F/'source33_original_numerical_and_nonentry_AST_equivalence_v1.json',
 F/'source33_configuration_and_future7_runtime_bindings_v2.json',F/'source33_runtime7_v1_to_v2_report_only_correction_v1.json',
 F/'ROOT_source33_eighth_original_interval_inactive_preCPU_source_review_v1.json',
 F/'ROOT_source31_original_interval170469_saved170400_compact_failure_and_candidate_review_v1.json',
 F/'ROOT_eighth_alias9_CPU13_actual02_original_ingress_frontend_CPU_review_v1.json',
 F/'ROOT_CPU13_overlay_import_only_transport_compatibility_successor_v1.json',
 F/'eighth_original_full_clip_geometry_certificate_v1.json',F/'actual_failure_closure.json',
 F/'preserved69_unsaved_positive_journal_suffix.jsonl',S/'fastmss_resident_feature_stream_v13.py',S/'fastmss_original_clip_geometry_overlay_v9.py',
 S/'resource_watchdog_v2.py',S/'atomic_io_retry_v1.py',S/'bpe_broad_audio_v1.py',
 'E:/gpu_training_scratchpad/storage_config.json','E:/gpu_training_scratchpad/use_storage.ps1'):
 add(path)
for key in ('actual_failure','actual_CPU_failure','originalMonitor_completion','private_metadata_worker_completion','CPU_ready',
 'boot','runtime','current_child','saved_pointer','saved_persistence','owned2901_actual_TOOL'):add(value['evidence'][key]['path'])
for item in config['native_update_commit_pressure_policy'].values():
 if isinstance(item,dict)and set(('path','bytes','sha256')).issubset(item):add(item['path'])
add(read(F/'ROOT_eighth_alias9_CPU13_actual02_original_ingress_frontend_CPU_review_v1.json')['actual_CPU_control_report']['path'])
assert all('commits_000023.jsonl'not in x['path']and not x['path'].endswith('checkpoint.tensors.bin')for x in inputs)
scope={'kind':'ASR18_source33_eighth_actual_failure_typed_metadata_exact_scope_proposal_v1',
 'scope':'ASR18_source33_eighth_original_failure_entire_selector_metadata_CPU_v1','execution_admitted':False,
 'fixture_source':pin(fixture),'external_launcher_source':pin(external),'source_configuration':configpin,
 'driver_source':pin(driver),'wrapper_source':pin(wrapper),'eighth_original_transition_validator':pin(validator),
 'eighth_original_resume':closure,'initial_checkpoint':pointer,'branch':str(B),'input_pins':inputs,
 'runtime':'E:/gpu_training_scratchpad/scratch/BPE_broad_CPU_preflight_env_v2/Scripts/python.exe',
 'exact_external_arguments':['-Xutf8','-B',str(external),'--ROOT-review','__PENDING_EXACT_ROOT_CPU_SCOPE__'],
 'output':str(F/'source33_typed_eighth_selector_actual_01'),
 'limits':{'internal_seconds':180,'external_seconds':240,'sampled_full_family_private_bytes':536870912,
  'prospective_family_bytes':536870912,'FULL_prefetch_bytes':268435456,
  'admission_commit_available_bytes':2805306368,'admission_physical_available_bytes':2305306368,
  'runtime_commit_available_bytes':2268435456,'runtime_physical_available_bytes':1768435456,'E_free_bytes':100000000000},
 'external_exact_interpreter_launcher_parent_in_sampled_family_not_cleanup_targets':True,
 'tests':['real current policy and actual failure validators','actual complete new driver entry and shared interval assignment',
  'entire original journal_factory metadata dictionary under new typed branch','entire postrestore branch dictionary',
  'actual postcopy STAT-only require expression without copy','entire wrapper entry require and initial forwarding',
  'wrong initial/mixedhistorical/staleclosure/wrongcheckpoint/missingCPU/failedalias8/wrongCPU/MAX2 strict negatives',
  'missing/wrong selector full postdict negatives explicitly METHOD_STUB'],
 'doubles':'ONLY no-read prefix view, prospective journal.binding and in-memory structural review; never production-valid fake ROOT review',
 'reads':'compact/source metadata plus preserved135633B69tailCOPY; original23 STAT only',
 'no_operations':['audio','frontend','NumPy','Torch','model','GPU','restore','prefixcopy','originaljournalread/hash','sampler/RNGdraw','kernelcredit','productionwrite'],
 'no_automatic_retry':True,'all_original_science_guards_reservations_225_MAX2_and_source31_performance_semantics_exact':True}
write(F/'source33_eighth_actual_failure_typed_metadata_scope_proposal_v1.json',scope)
stale={'fresh_signature_resume','exact_future_wrapper_arguments','actual_pre_stop_review','actual_stop_ROOT_review',
 'metadata_materialization_scope','actual_stop_closure','required_next_steps','actual_CPU_review','configuration_inverse_and_runtime7'}
handoff={k:copy.deepcopy(v)for k,v in oldhandoff.items()if k not in stale}
handoff.update(kind='ASR18_source33_eighth_actual170469_saved170400_inactive_execution_handoff_v1',production_admitted=False,
 driver_source=pin(driver),wrapper_source=pin(wrapper),CPU_source=pin(S/'fastmss_resident_feature_stream_v13.py'),
 source_configuration=configpin,initial_checkpoint=pointer,eighth_original_resume=closure,eighth_original_transition_validator=pin(validator),
 initial_owner_invocation=24,original_clip_geometry_overlay=config['original_clip_geometry_overlay'],
 resource_scheduling_policy=config['resident_RAM_resource_scheduling_policy'],packet_credit_policy=config['resident_RAM_packet_credit_policy'],
 native_update_commit_pressure_policy=config['native_update_commit_pressure_policy'],runtime7=pin(F/'source33_configuration_and_future7_runtime_bindings_v2.json'),
 exact_future_wrapper_arguments=args,exact_arguments_are_DRAFT_pending_final_ROOT_binding=True,
 source_preCPU_review=pin(F/'ROOT_source33_eighth_original_interval_inactive_preCPU_source_review_v1.json'),
 typed_metadata_scope_proposal=pin(F/'source33_eighth_actual_failure_typed_metadata_scope_proposal_v1.json'),
 direct_RAW_inverse=pin(F/'source31_to33_entire_RAW_inverse_v1.json'),
 original_failure_ROOT_review=pin(F/'ROOT_source31_original_interval170469_saved170400_compact_failure_and_candidate_review_v1.json'),
 completed_ingress_CPU_review=pin(F/'ROOT_eighth_alias9_CPU13_actual02_original_ingress_frontend_CPU_review_v1.json'),
 actual_CPU_review=pin(F/'ROOT_eighth_alias9_CPU13_actual02_original_ingress_frontend_CPU_review_v1.json'),
 configuration_inverse_and_runtime7=pin(F/'source33_configuration_and_future7_runtime_bindings_v2.json'),
 derived_transport_review=pin(F/'ROOT_CPU13_overlay_import_only_transport_compatibility_successor_v1.json'),
 saved170400_fullstate_restore_before_native=True,savedprefix_only_copy_AFTER_restore_no_native_or_draws=True,
 preserved_historical_updates=list(range(170401,170470)),new_required_positives=[170401,170470],
 next_natural_fullcheckpoint=170600,postpositive_required_at_least=170620,full_training_target=401072,admin_stop=None,
 actual_reader_catchup_expected_no_restart=True,human_recurrent_speed_fix_still_OPEN_no_new_speed_claim=True)
handoff['required_next_steps']=['ROOT exact typed CPU scope','ONE actual typed metadata proof and exact death seal',
 'ROOT actual typed acceptance and final exact ONE wrapper execution binding','Only then full saved170400 allstate restore before native',
 'After restore savedprefix-only branch,69 genuine redo and actual170470 positive','Natural170600 full persistence/post170620/current family and healthy reader before recovery acceptance']
# Remove old long-test/frontier details; this is an audio repair, no speed tuning.
for key in list(handoff):
 if ('validation'in key or 'long_observer'in key or 'recovery_sealer'in key)and key not in ('validation_required_after_actual_activation',):del handoff[key]
write(F/'source33_eighth_actual_failure_inactive_execution_handoff_v1.json',handoff)
print(json.dumps({'runtime7_v2':pin(F/'source33_configuration_and_future7_runtime_bindings_v2.json'),
 'fixture':pin(fixture),'external':pin(external),'scope_proposal':pin(F/'source33_eighth_actual_failure_typed_metadata_scope_proposal_v1.json'),
 'handoff':pin(F/'source33_eighth_actual_failure_inactive_execution_handoff_v1.json'),'inputs':len(inputs)}),flush=True)
