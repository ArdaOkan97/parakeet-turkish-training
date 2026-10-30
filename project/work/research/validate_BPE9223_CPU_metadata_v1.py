"""Focused synthetic-metadata controls; NO audio, arrays, ROOT authority or model."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

P = Path(__file__).resolve().parents[2]
S = P / 'outputs/parakeet_turkish'
R = P / 'work/research'
sys.path.insert(0, str(S))
from ASR18_continuous_data_guard_v5 import DataIOGuard, ROOT
from cpu_preparation_guard_v2 import now
import fastmss_BPE9223_feature_protocol_v1 as public
import fastmss_BPE9223_cpu_common_v1 as core
import fastmss_BPE9223_recorded_room_train_v1 as augmentation


def main():
    guard = DataIOGuard('BPE9223_focused_CPU_metadata_controls', heavy=False, maximum_bytes=32_000_000)
    guard.wait()
    area = ROOT / 'analysis/BPE9223_training_launch_v1/CPU_preparation_v1/metadata_fixture_v1'
    assert not area.exists()
    area.mkdir()
    tests = []

    def check(name, operation, *, reject=False):
        guard.wait()
        try:
            value = operation()
        except (ValueError, KeyError, TypeError, AssertionError):
            assert reject, name
            tests.append(dict(name=name, expected_rejection=True, passed=True))
        else:
            assert not reject, name
            tests.append(dict(name=name, expected_rejection=False, passed=True))

    def save(path, value):
        core.write_new(path, value)
        return core.record(path, guard.wait)

    bridge = public.bridge
    base = dict(kind=bridge.FULL_SESSION_KIND, mode=bridge.FULL_MODE, purpose=bridge.FULL_PURPOSE,
        production_enabled=False, owner_validation_only=True, checkpoint_diagnostic_only=True,
        diagnostic_only=True, production_source_eligible=False, planned_native_updates=401072,
        initial_optimizer_updates=0, initial_sampler_sha256='a'*64, full_run_identity_sha256='b'*64)
    check('full_planner_body_can_remain_typed_qualification', lambda: bridge.validate_session_scope(base))
    production = dict(base, owner_validation_only=False, checkpoint_diagnostic_only=False,
        diagnostic_only=False, production_source_eligible=True)
    check('separate_production_scope_is_explicit', lambda: bridge.validate_session_scope(production))
    for field, value in (('production_source_eligible', True), ('diagnostic_only', False),
                         ('checkpoint_diagnostic_only', False), ('owner_validation_only', 1),
                         ('planned_native_updates', 40)):
        bad = dict(base, **{field: value})
        check('reject_scope_' + field, lambda bad=bad: bridge.validate_session_scope(bad), reject=True)
    for language, ids, allowed in (('tr', [0, 8198, 9222], True), ('tr', list(range(517)), True),
        ('en', [0, 8191], True), ('en', [9222], False), ('en', [8193], False),
        ('tr', [9223], False), ('tr', [8192], False), ('en', [8192], False), ('tr', [True], False)):
        row = dict(language=language, target_tokens=len(ids), tr_metadata_labels=ids, clean_target_ids=ids)
        check('target_domain_' + language + '_' + str(len(tests)), lambda row=row: bridge.full_targets(row), reject=not allowed)

    paths = {'protocol': Path(public.__file__), 'engine': Path(public.engine.__file__),
        'reference': R/'produce_FastMSS_BPE9223_reference_CPU_v1.py',
        'frontend': R/'produce_FastMSS_BPE9223_frontend_CPU_v1.py',
        'controller': R/'run_ASR18_BPE9223_CPU_window_v1.py', 'core': Path(core.__file__),
        'guard': R/'ASR18_bounded_CPU_phase_v3.py', 'process': Path(public.process.__file__),
        'augmentation_adapter': Path(augmentation.__file__),
        'real_noise_mixer': Path(augmentation.mixer.__file__),
        'real_noise_decoder': Path(augmentation.decoder.__file__), 'room_provenance': Path(augmentation.rooms.__file__)}
    sources = {key: core.record(path, guard.wait) for key, path in paths.items()}
    policy = core.record(area.parent/'recorded_room_TRAIN_policy_v1.json', guard.wait)
    _, identity = augmentation.policy_identity(policy, guard.wait)
    # This real source pin is deliberately only a fixture placeholder for a
    # context source. A live core/context/ROOT review rejects this fixture; no
    # ROOT review, actual process0, feature owner or authority is ever written.
    placeholder = sources['protocol']
    label_population = core.record(ROOT/'analysis/BPE9223_training_launch_v1/verified_population_v1/final_runtime_bindings_9223_v1.json', guard.wait)
    python = core.record(core.CPU_ENV/'Scripts/python.exe', guard.wait)
    resources = dict(leaf_maximum_private_bytes=1_800_000_000, controller_private_reserve_bytes=64*1024**2,
        inner_shim_private_reserve_bytes=32*1024**2, outer_shim_private_reserve_bytes=32*1024**2,
        whole_inner_family_bytes=1_900_663_296, whole_family_forecast_bytes=1_934_217_728)

    def scenario(count, phase, row_count=16):
        folder = area / ('case_' + str(len(tests))); folder.mkdir()
        numerical = save(folder/'numerical.json', dict(vocabulary=9223, source_context_source=placeholder,
            label_population_contract=label_population, outer_parent_source=sources['controller']))
        session = save(folder/'session.json', dict(base, CPU_preparation_contract=numerical,
            vocabulary=9223, source_context_source=placeholder, label_population_contract=label_population))
        plan = dict(kind=bridge.KIND, mode=bridge.FULL_MODE, purpose=bridge.FULL_PURPOSE,
            production_enabled=False, diagnostic_only=True, owner_validation_only=True, checkpoint_diagnostic_only=True,
            bridge_source_sha256=core.record(bridge.__file__)['sha256'], maximum_feature_window_bytes=public.CAPACITY,
            raw_PCM_capacity_bytes=128*1024**2, raw_PCM_maximum_jobs=2, session=session, window_index=0,
            phase=phase, augmentation_identity=identity, updates=[dict(rows=[{} for _ in range(row_count)]) for _ in range(count)])
        bridge_pin = save(folder/'bridge.json', plan)
        contract = dict(kind=public.CONTRACT, mode=public.MODE, production_enabled=False, production_source_eligible=False,
            validation_only=False, native_benchmark_only=False, owner_validation_only=True, checkpoint_diagnostic_only=True,
            vocabulary=9223, sources=sources, maximum_window_updates=20, feature_capacity_bytes=public.CAPACITY,
            raw_PCM_capacity_bytes=128*1024**2, raw_PCM_maximum_jobs=2, one_actual_reference_window0_before_frontend=True,
            per_update_child0_claimed=False, freeze_after_setup=True, process_owner='/root/cpu_data_preparation',
            resource_review=resources, bridge=bridge_pin, numerical_CPU_contract=numerical,
            augmentation_policy=policy, augmentation_identity=identity, original_augmentation_identity=identity)
        contract_pin = save(folder/'source_contract.json', contract)
        static = dict(production_enabled=False, native_benchmark_only=False, vocabulary=9223,
            sources=dict(parent=sources['controller'], reference=sources['reference'], frontend=sources['frontend'],
                         core=sources['core'], guard=sources['guard']),
            python=python, outer_parent_source=sources['controller'], freeze_after_setup=True,
            namespace='synthetic_metadata_controls_only', owner_validation_only=True, checkpoint_diagnostic_only=True)
        template = dict(static, kind=public.TEMPLATE,
            source_contract_template={key: value for key, value in contract.items() if key != 'bridge'},
            session=session, full_run_identity_sha256='b'*64, label_population_contract=label_population,
            source_context_source=placeholder, output_namespace=str(folder/'outputs'), maximum_window_updates=20)
        template_pin = save(folder/'template.json', template)
        proposal = dict(static, kind=public.CONFIGURATION, source_contract=contract_pin,
            output_root=str(folder/'outputs/window0'), window_id='window0', window_index=0, template_configuration=template_pin)
        return template, proposal, contract_pin

    successful = []
    for count, phase, row_count in ((1, bridge.WEAK, 3), (10, bridge.WEAK, 16),
                                    (20, bridge.WEAK, 16), (20, bridge.CLEANER, 16)):
        template, proposal, pin = scenario(count, phase, row_count)
        check('generic_' + phase + '_' + str(count) + '_updates_' + str(row_count) + '_rows',
            lambda t=template, p=proposal: public.verify_window_derivative(t, p, guard.wait))
        successful.append((template, proposal))
    for count, phase, row_count in ((0, bridge.WEAK, 16), (21, bridge.WEAK, 16), (1, 'test', 16), (1, bridge.WEAK, 17)):
        template, proposal, pin = scenario(count, phase, row_count)
        check('reject_window_bounds_' + str(count) + '_' + phase + '_' + str(row_count),
            lambda pin=pin: public.verify_source_contract(pin, guard=guard.wait), reject=True)
    template, proposal = successful[-1]
    for field, value in (('namespace', 'production_promoted'), ('freeze_after_setup', False),
                         ('checkpoint_diagnostic_only', False), ('window_index', 1),
                         ('output_root', str(ROOT/'temp/forbidden_adoption')), ('extra_override', True)):
        bad = dict(proposal, **{field: value})
        check('reject_derivative_' + field,
            lambda bad=bad: public.verify_window_derivative(template, bad, guard.wait), reject=True)
    assert len(tests) >= 30
    assert not any(name in sys.modules for name in ('numpy', 'torch', 'transformers', 'pyarrow', 'soundfile'))
    for pin in sources.values():
        if pin['path'].endswith('.py'):
            ast.parse(Path(pin['path']).read_text(encoding='utf-8'))
    report = dict(kind='BPE9223_generic_CPU_synthetic_metadata_controls_v1', at_utc=now(), passed=True,
        source=core.record(__file__), sources=sources, tests=tests, checks=len(tests),
        metadata_only=True, synthetic_case_artifacts_not_authority=True, ROOT_review_written=False,
        actual_CPU_audio_or_frontend_job=False, PCM_or_NPY_files_written=0, models_or_GPU_used=False,
        production_enabled=False, phase_CLEANER_or_replay_retokenized=False)
    save(area/'report.json', report)
    print(dict(report=core.record(area/'report.json'), checks=len(tests), actual_tool_exit_still_required=True), flush=True)


if __name__ == '__main__':
    main()
