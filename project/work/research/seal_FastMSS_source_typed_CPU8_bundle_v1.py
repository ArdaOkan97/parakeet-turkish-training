"""Source-only AST/pin seal. No producer imports, arrays, model or fixture jobs."""
import ast
import hashlib
import json
from pathlib import Path

P = Path(__file__).resolve().parents[2]
S, R = P/'outputs/parakeet_turkish', P/'work/research'
F = Path(r'E:\gpu_training_scratchpad\analysis\FastMSS_integration_v1')

def record(path):
    path = Path(path).resolve(); raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}

def tree(path):
    source = Path(path).read_text(encoding='utf-8-sig')
    value = ast.parse(source, filename=str(path)); compile(value, str(path), 'exec')
    return value

def functions(path):
    return {v.name: ast.dump(v, include_attributes=False) for v in tree(path).body
            if isinstance(v, (ast.FunctionDef, ast.ClassDef))}

def write_new(path, value):
    raw = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf8')
    with Path(path).open('xb') as stream:
        stream.write(raw)
    return record(path)

paths = {'bridge': S/'fastmss_phase_aware_plan_bridge_v7.py',
         'sink': S/'fastmss_phase_aware_feature_window_v6.py',
         'common': S/'fastmss_row_serial_cpu_v8.py',
         'controller': R/'prepare_FastMSS_phase_window_CPU_v8.py',
         'reference_worker': R/'produce_FastMSS_independent_rows_CPU_v8.py',
         'feature_worker': R/'produce_FastMSS_serial_features_CPU_v8.py'}
for path in paths.values(): tree(path)

checks = []
for key, old, changed in (
    ('bridge', S/'fastmss_phase_aware_plan_bridge_v5.py',
     {'normalize_updates', 'bridge_plan', 'bind_verified_update_inputs', 'validate_resume'}),
    ('sink', S/'fastmss_phase_aware_feature_window_v5.py', {'_load_bridge'}),
    ('common', S/'fastmss_row_serial_cpu_v7.py', {'verify_contract', 'load_request'}),
    ('controller', R/'prepare_FastMSS_phase_window_CPU_v7.py', {'main'}),
    ('reference_worker', R/'produce_FastMSS_independent_rows_CPU_v7.py', set()),
    ('feature_worker', R/'produce_FastMSS_serial_features_CPU_v7.py', set())):
    before, after = functions(old), functions(paths[key])
    preserved = sorted(set(before)-changed)
    assert all(after[name] == before[name] for name in preserved), (key, 'unexpected operation delta')
    checks.append({'source': key, 'baseline': record(old), 'unchanged_definitions': preserved,
                   'explicit_changed_definitions': sorted(changed)})

bridge_hash = record(paths['bridge'])['sha256']
assert 'BRIDGE_SHA = "'+bridge_hash+'"' in paths['sink'].read_text()
assert 'bridge.PURPOSE' not in paths['sink'].read_text()
assert 'bridge.PURPOSE' not in paths['common'].read_text()
template = json.loads((F/'FastMSS_row_serial_CPU_contract_TEMPLATE_NOT_LAUNCHABLE_v7.json').read_bytes())
template['kind'] = 'FastMSS_source_typed_row_serial_CPU_preparation_contract_v1'
template['mode'] = 'full_two_broad_one_cleaner'
template['purpose'] = 'reviewed_full_two_broad_one_cleaner_production_CPU_preparation'
template['planned_native_updates'] = 401072
del template['maximum_session_windows'], template['maximum_session_native_updates']
for key, path in paths.items(): template['sources'][key] = record(path)
template_pin = write_new(F/'FastMSS_row_serial_CPU_contract_TEMPLATE_NOT_LAUNCHABLE_v8.json', template)

interface = {
 'kind': 'FastMSS_source_typed_CPU8_source_only_interface_v1',
 'source_only': True, 'production_enabled': False, 'execution_or_resource_approved': False,
 'template': template_pin,
 'bridge_API': 'bridge_plan(coordinator_plan_binding, *, DATA_evidence, session_binding, window_index, prior_window_boundary=None, verify_prior_boundary=None, guard)',
 'reader_API_unchanged': ['take_row(update_index,row_index,context,transactionSHA,beforeSHA)', 'update_record(index)', 'prepared_payload(index)'],
 'request_kind': 'FastMSS_source_typed_row_serial_CPU_update_request_v1',
 'full_mode': {'mode': 'full_two_broad_one_cleaner', 'session_kind': 'FastMSS_full_run_CPU_preparation_session_v1',
   'purpose': template['purpose'], 'diagnostic_only': False, 'planned_native_updates': 401072,
   'forbidden_session_fields': ['maximum_windows', 'maximum_native_updates', 'any diagnostic_context field', 'hard_context_verifier_source'],
   'required_full_session_fields': ['full_run_identity_sha256', 'durable_owner'],
   'window_index': 'scalar >=0; no history arrays, no two-window/40-update ceiling',
   'pass_tail': 'all updates same phase/epoch; exact original permutation; exact min(14,remaining-in-pass); pass tail is last update',
   'later_boundary_kind': 'FastMSS_actual_coherent_full_run_CPU_window_boundary_v2',
   'later_boundary': 'same old full pointer/process2+actual owned cleanup metadata fields; completed_window_index=current_index-1; durable_head logical descriptor mandatory',
   'durable_head': 'one <=64KiB immutable adapter2 frame; identity/owner/current pointer/sampler exact; native-owner callback independently proves CURRENT head',
   'pass_barrier': 'prior tail requires completed_boundaries acknowledgement at expected count, current recovery receipt and empty pending index before next preparation',
   'recovery_scope': 'CURRENT_LATEST_ONLY_WITH_VERIFIED_DURABLE_PERSISTENCE_PREFIX_BEFORE_PASS_DEV_WINNER_ADVANCE'},
 'hard_mode': {'mode': 'one_hard_envelope_diagnostic',
   'session_kind': 'FastMSS_one_full_hard_envelope_diagnostic_CPU_session_v1',
   'purpose': 'bounded_one_full_hard_envelope_diagnostic', 'diagnostic_only': True, 'production_source_eligible': False,
   'maximum_windows': 1, 'maximum_native_updates': 1, 'initial_optimizer_updates': 0, 'window_index': 0,
   'session_plan_equal_pin': 'hard_envelope_diagnostic_context',
   'session_helper_pin': 'hard_context_verifier_source',
   'verifier': record(S/'asr18_hard_envelope_schedule_context_v1.py'),
   'helper_API': 'verify_bridge_state(plan, context_pin, *, population, guard)',
   'contract_delta_from_full_template': 'change mode/purpose; remove planned_native_updates; set maximum_session_windows=1 and maximum_session_native_updates=1',
   'no_full_pass_or_DEV_claim': True, 'recovery_scope': 'EVERY_DIAGNOSTIC_CAPTURE_BEFORE_OVERALL_PASS'},
 'controller': {'CLI': '--bridge <recordJSON> --output <fresh E folder> [--contract <same pin>] [--prior-boundary-verification <recordJSON>]',
   'prior_verification_required': 'exactly when window_index>0',
   'attestation_kind_unchanged': 'FastMSS_native_owner_prior_window_CPU_handoff_v1',
   'full_attestation_additions': ['durable_head equals prior boundary descriptor', 'current_durable_owner_head_and_persistence_prefix_verified=true'],
   'source_bound_outer_parent_unchanged': True},
 'unchanged_limits': {'per_window_updates': 20, 'feature_bytes': 536870912, 'raw_bytes': 134217728,
                      'raw_jobs': 2, 'CPU_child_private_hard_bytes': 2000000000, 'controller_reserve_bytes': 64000000},
 'actual_metadata_bridge_call_required_before_audio': True,
 'not_proven': ['actual CPU8/hard real-plan bridge call', 'new CPU8 array preparation', 'hard effective16 native resource fit',
   'actual long-run durable/DEV/storage integration', 'production activation'],
 'unchanged_numerics': 'all original decoder/PTS/resampling/chunked hashes/raw ledger/frontend/NumPy and stdlib NPY validation/transform policy methods are AST-identical; import pins and protocol routing only'
}
interface_pin = write_new(F/'FastMSS_source_typed_CPU8_interface_v1.json', interface)
manifest = {'kind': 'FastMSS_source_typed_CPU8_source_review_manifest_v1', 'owner': '/root/fastmss_integration',
 'source_only': True, 'production_enabled': False, 'new_sources': {k:record(v) for k,v in paths.items()},
 'hard_context_verifier': record(S/'asr18_hard_envelope_schedule_context_v1.py'),
 'original_schedule_state': record(S/'bpe_broad_production_state_v1.py'),
 'template': template_pin, 'interface': interface_pin, 'AST_syntax_compile_all_six': True,
 'AST_preserved_operation_checks': checks, 'seal_source': record(__file__),
 'actual_fixture_or_native_execution_claimed': False}
print(json.dumps(write_new(F/'FastMSS_source_typed_CPU8_source_review_manifest_v1.json', manifest)))
