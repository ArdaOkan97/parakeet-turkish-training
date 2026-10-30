"""Metadata/AST preparation only; never import the inactive candidate."""
import ast, hashlib, json, os
from pathlib import Path
F=Path(__file__).parent
FIX=F.parent
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20')
S=P/'outputs/parakeet_turkish'
def pin(path):
 p=Path(path).resolve();raw=p.read_bytes()
 return {'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def write(path,value):
 with Path(path).open('x',encoding='utf-8',newline='\n')as out:json.dump(value,out,indent=2,ensure_ascii=False);out.write('\n')
assert os.environ['TORCH_EXTENSIONS_DIR'].lower().startswith('e:')
scope=FIX/'ROOT_one_postDEV_gc_refresh_inactive_preparation_scope_v1.json'
amendment=FIX/'ROOT_one_postDEV_gc_refresh_inactive_preparation_semantic_verification_amendment_v1.json'
assert pin(scope)['sha256']=='0b98ef3185077a2bdfcbec5e07686874f556b6c40d0697838ceba7d50d557ea9'
assert pin(amendment)['sha256']=='24c976c05da96240f87588ea765345d4071db3c550b2be29c88955b3caba848e'
helper=F/'asr18_one_postDEV_gc_refresh_v1.py'
capture=F/'asr18_postDEV_refresh_state_capture_v1.py'
for path in (helper,capture):ast.parse(path.read_text(encoding='utf-8'))
unchanged=[S/'asr18_BPE9223_grouped_RAM_native_v1.py',S/'fastmss_resident_feature_stream_v13.py',
 P/'work/research/run_ASR18_native_throughput_component_v15.py',S/'asr18_live_state_fingerprint_v1.py',
 S/'asr18_BPE9223_private_scalar_prefetch_v1.py',S/'asr18_resident_periodic_CPU_quiescence_v3.py',
 FIX/'asr18_observe_original_gc_collect_v2.py']
comparison=F/'state_capture_options_and_explicit_amendment_v1.json'
write(comparison,{'kind':'ASR18_INACTIVE_postDEV_state_capture_comparison_v1',
 'option_A':{'selected':False,'identity_version_storage_manifest_only':True,
  'limitation':'No independent tensor-byte semantic equality. Cannot satisfy original full854 state event requirement; must never be called full byte equality.'},
 'option_B':{'selected_for_inactive_preparation_only':True,'ROOT_amendment':pin(amendment),
  'original_fingerprint_source':pin(S/'asr18_live_state_fingerprint_v1.py'),
  'before_after_original_live_state_fingerprint_and_require_exact_live_restore':True,
  'two_additional_explicit_normal_CUDA_synchronizations_at_idle_event':True,
  'maximum_contiguous_CPU_transfer_slice_bytes':4194304,'no_whole_CPU_model_copy_or_payload_artifact':True,
  'original_per_update_native_collect_sync_emptycache_bytes_unchanged':True,
  'CPU_fixture_scientific_torch_part':'METHOD_STUB/source AST only, not real854 GPU equality'},
 'bookkeeping':{'additional_bytes':16777216,'full_commit_threshold':4535212672,'full_physical_threshold':1785212672,
  'capture_serialized_bytes_max_each':4194304,'known_before_after_scalar_trees_intentionally_live_during_event':True,
  'four_MiB_transfer_slice_transient_within_explicit_proposal':True,
  'Python_tree_allocator_total_hard_bound_proved':False,'no_CPU_or_DEV_freed_credit':True,
  'if_unfit_or_bound_denied':'fatal/preserve actual; no automatic reserve raise'},
 'retention_risk':'Known scalar trees are acyclic metadata. A reachable opaque framework cycle frozen now can later become unreachable yet survive until original teardown unfreeze. Synthetic tests cannot exclude it.',
 'runtime_GC_or_production_admitted':False})
handoff=F/'inactive_one_postDEV_refresh_helpers_handoff_v1.json'
write(handoff,{'kind':'ASR18_INACTIVE_one_postDEV_refresh_initial_helpers_handoff_v1',
 'preparation_scope':pin(scope),'semantic_verification_amendment':pin(amendment),
 'helper_source':pin(helper),'capture_source':pin(capture),'state_capture_comparison':pin(comparison),
 'unchanged_source_pins':[pin(p)for p in unchanged],
 'new_source_has_no_prior_byte_stream':'Fresh helper modules; original perupdate native and original GC scope remain raw-byte unchanged.',
 'method_stub_only_entry':'RefreshSequence injected callbacks/operations for strict finite CPU negatives; production uses OnePostDEVRefresh.execute_real actual builtins and original collect observer class identity.',
 'all_actual_import_control_GC_stop_launch_admitted':False})
print(json.dumps({'handoff':pin(handoff),'helper':pin(helper),'capture':pin(capture),'comparison':pin(comparison)}))
