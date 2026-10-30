"""Create NEW unadmitted CPU sources; never mutate the reviewed CPU8 benchmark."""
import ast
import hashlib
from pathlib import Path

P = Path(__file__).resolve().parents[2]
S = P / 'outputs/parakeet_turkish'
R = P / 'work/research'
MAPPING = {
    'fastmss_phase_aware_plan_bridge_v7': 'fastmss_BPE9223_plan_bridge_v1',
    'fastmss_phase_aware_feature_window_v7': 'fastmss_BPE9223_feature_engine_v1',
    'fastmss_row_serial_cpu_v8': 'fastmss_BPE9223_cpu_common_v1',
    'fastmss_recorded_room_feature_protocol_v2': 'fastmss_BPE9223_feature_protocol_v1',
    'fastmss_recorded_room_train_v2': 'fastmss_BPE9223_recorded_room_train_v1',
    'run_ASR18_recorded_room_CPU_window_v2': 'run_ASR18_BPE9223_CPU_window_v1',
    'produce_FastMSS_recorded_room_reference_CPU_v2': 'produce_FastMSS_BPE9223_reference_CPU_v1',
    'produce_FastMSS_recorded_room_frontend_CPU_v2': 'produce_FastMSS_BPE9223_frontend_CPU_v1',
}


def replace(text, before, after, count=1):
    assert text.count(before) == count, (before, text.count(before), count)
    return text.replace(before, after)


def function(text, name, body):
    start = text.index('def ' + name + '(')
    end = text.find('\ndef ', start + 1)
    if end < 0:
        end = text.find('\nclass ', start + 1)
    assert end > start
    return text[:start] + body.rstrip() + '\n\n' + text[end+1:]


def base(name):
    folder = R if name.startswith(('run_', 'produce_')) else S
    text = (folder / (name + '.py')).read_text(encoding='utf-8')
    for old, new in MAPPING.items():
        text = text.replace(old, new)
    return text


def main():
    from ASR18_continuous_data_guard_v5 import DataIOGuard
    guard = DataIOGuard('BPE9223_CPU_source_preparation', heavy=False, maximum_bytes=5_000_000)
    guard.wait()
    result = {}
    bridge = base('fastmss_phase_aware_plan_bridge_v7')
    bridge = replace(bridge, 'FastMSS_source_typed_phase_aware_feature_window_bridge_v2', 'FastMSS_BPE9223_phase_aware_feature_window_bridge_v1')
    bridge = replace(bridge, '(8711 if language == "tr" else 8193)', '(9223 if language == "tr" else 8193)')
    bridge = replace(bridge, '11972886 / 352768', '10621773 / 352768')
    bridge = function(bridge, 'frozen_augmentation', '''def frozen_augmentation(identity, guard):
    """Metadata-only exact NEW recorded-noise/16RIR policy, no old active pilot."""
    import fastmss_BPE9223_recorded_room_train_v1 as augmentation
    _, actual = augmentation.policy_identity(identity['policy_file'], guard)
    require(same(actual, identity), 'Exact NEW9223 recorded-room policy/runtime/source identity')
    return deepcopy(identity)
''')
    bridge = function(bridge, 'validate_DATA', '''def validate_DATA(evidence, guard):
    """NEW9223 population audit; original source/heldout facts remain bound."""
    from asr18_BPE9223_source_context_v1 import validate_CPU_population
    data = validate_CPU_population(evidence, guard=guard)
    require(type(data) is dict and data['retained_rows'] == 2631115
        and data['total_target_tokens'] == 90746310 and data['population_mean_U'] == 90746310 / 2631115,
        'Complete9223 broad denominator, no partial/full8711 substitution')
    require(data['vocabulary'] == 9223 and data['blank_id'] == 8192,
        'Explicit9223 TR labels; replay retains original8193 cached IDs')
    return data
''')
    # The new native context is lazy and metadata-only; its exact pin is supplied
    # by the immutable session/ROOT-reviewed configuration before any audio work.
    bridge = replace(bridge, '    augmentation_identity = frozen_augmentation(session["augmentation_identity"], guard)', '''    context_source = session['source_context_source']
    require(Path(context_source['path']).resolve() == Path(__file__).with_name('asr18_BPE9223_source_context_v1.py').resolve(), 'Exact NEW9223 context source path')
    verify_file(context_source, guard, maximum=256*1024)
    require(session['vocabulary'] == 9223 and same(session['label_population_contract'], DATA_evidence['BPE9223_population']), 'Session and9223 label population binding differ')
    augmentation_identity = frozen_augmentation(session["augmentation_identity"], guard)''')
    result[MAPPING['fastmss_phase_aware_plan_bridge_v7']] = bridge

    engine = base('fastmss_phase_aware_feature_window_v7')
    engine = replace(engine, 'e1a484e851cde2f9309a3b4905a1b892ca874f46918fdd0fc5729cdbfaa8f413', hashlib.sha256(bridge.encode()).hexdigest())
    engine = engine.replace('temp/ASR18_CPU_feature_windows_v1', 'temp/BPE9223_original_CPU_feature_engine_v1')
    result[MAPPING['fastmss_phase_aware_feature_window_v7']] = engine

    core = base('fastmss_row_serial_cpu_v8')
    core = replace(core, 'import fastmss_phase_aware_feature_window_v6 as sink', 'import fastmss_BPE9223_feature_engine_v1 as sink')
    core = core.replace('produce_FastMSS_independent_rows_CPU_v8.py', 'produce_FastMSS_BPE9223_reference_CPU_v1.py')
    core = core.replace('produce_FastMSS_serial_features_CPU_v8.py', 'produce_FastMSS_BPE9223_frontend_CPU_v1.py')
    core = core.replace('prepare_FastMSS_phase_window_CPU_v8.py', 'run_ASR18_BPE9223_CPU_window_v1.py')
    core = replace(core, "'cleaner':Path(__file__).with_name('bpe_cleaner_data_v1.py')", "'cleaner':Path(__file__).with_name('asr18_BPE9223_source_context_v1.py')")
    core = replace(core, '    mode=bridge.validate_session_scope(session)', '''    mode=bridge.validate_session_scope(session)
    require(contract['vocabulary'] == session['vocabulary'] == 9223
        and same(contract['source_context_source'], session['source_context_source'])
        and same(contract['label_population_contract'], session['label_population_contract']), 'New9223 session/context/labels are inseparable')''')
    core = replace(core, '            from bpe_cleaner_data_v1 import CleanerRows\n            if self.cleaner is None:self.cleaner=CleanerRows()', '''            from asr18_BPE9223_source_context_v1 import CleanerRows9223
            if self.cleaner is None:self.cleaner=CleanerRows9223(self.validated, guard=self.guard)''')
    result[MAPPING['fastmss_row_serial_cpu_v8']] = core

    adapter = base('fastmss_recorded_room_train_v2')
    adapter = adapter.replace("policy['benchmark_enabled'] is True", "policy['bounded_preparation_enabled'] is True")
    adapter = adapter.replace('recorded_noise_16room_aggregate_CPU_first4_benchmark_v1', 'BPE9223_recorded_noise_16room_bounded_CPU_window_v1')
    adapter = adapter.replace("KIND='FastMSS_recorded_noise_16room_TRAIN_policy_v1'", "KIND='FastMSS_BPE9223_recorded_noise_16room_TRAIN_policy_v1'")
    adapter = adapter.replace("IDENTITY='FastMSS_recorded_noise_16room_ordered_draw_identity_v1'", "IDENTITY='FastMSS_BPE9223_recorded_noise_16room_ordered_draw_identity_v1'")
    adapter = adapter.replace('a separately admitted benchmark namespace', 'a separately reviewed9223 preparation namespace')
    result[MAPPING['fastmss_recorded_room_train_v2']] = adapter

    protocol = base('fastmss_recorded_room_feature_protocol_v2')
    protocol = protocol.replace('temp/ASR18_recorded_room_CPU_benchmark_v1', 'temp/BPE9223_recorded_room_CPU_windows_v1')
    protocol = protocol.replace('recorded_room_first4_native_benchmark_only', 'BPE9223_recorded_room_bounded_CPU_window_v1')
    protocol = protocol.replace('FastMSS_recorded_room_CPU_source_and_numerical_contract_v1', 'FastMSS_BPE9223_CPU_source_and_numerical_contract_v1')
    protocol = protocol.replace('FastMSS_recorded_room_CPU_preparation_request_v1', 'FastMSS_BPE9223_CPU_preparation_request_v1')
    protocol = protocol.replace('FastMSS_owned_recorded_room_CPU_feature_window_v1', 'FastMSS_owned_BPE9223_recorded_room_CPU_feature_window_v1')
    protocol = protocol.replace("['production_source_eligible'] is False", "['production_source_eligible'] is True")
    protocol = protocol.replace("['native_benchmark_only'] is True", "['native_benchmark_only'] is False")
    protocol = protocol.replace("'native_benchmark_only': True", "'native_benchmark_only': False")
    protocol = protocol.replace("'production_source_eligible': False", "'production_source_eligible': True")
    protocol = protocol.replace("'/root/fastmss_integration'", "'/root/cpu_data_preparation'")
    protocol = protocol.replace('numerical_CPU8_contract', 'numerical_CPU_contract')
    protocol = replace(protocol, "        and plan['phase'] == bridge.WEAK and len(plan['updates']) == 4\n        and sum(len(v['rows']) for v in plan['updates']) == 64,", "        and plan['phase'] in (bridge.WEAK, bridge.CLEANER) and 1 <= len(plan['updates']) <= 20\n        and all(3 <= len(v['rows']) <= 16 for v in plan['updates']),")
    protocol = protocol.replace('native-benchmark', 'bounded9223')
    result[MAPPING['fastmss_recorded_room_feature_protocol_v2']] = protocol

    reference = base('produce_FastMSS_recorded_room_reference_CPU_v2')
    reference = reference.replace('numerical_CPU8_contract', 'numerical_CPU_contract')
    reference = reference.replace('production-disabled recorded-room benchmark slot', 'review-required generic9223 recorded-room slot')
    result[MAPPING['produce_FastMSS_recorded_room_reference_CPU_v2']] = reference
    frontend = base('produce_FastMSS_recorded_room_frontend_CPU_v2')
    frontend = frontend.replace("'native_benchmark_only': True", "'native_benchmark_only': False")
    result[MAPPING['produce_FastMSS_recorded_room_frontend_CPU_v2']] = frontend

    original_parent = base('run_ASR18_recorded_room_CPU_window_v2')
    parent = original_parent[:original_parent.index('def main(')]
    parent = parent.replace('ASR18_actual_recorded_room_CPU_window_benchmark_v1', 'ASR18_actual_BPE9223_recorded_room_CPU_window_v1')
    parent = parent.replace('ASR18_recorded_room_aggregate_CPU_window_configuration_v1', 'ASR18_BPE9223_aggregate_CPU_window_configuration_v1')
    parent = parent.replace("OWNER = '/root/fastmss_integration'", "OWNER = '/root/cpu_data_preparation'")
    parent += GENERIC_MAIN
    result[MAPPING['run_ASR18_recorded_room_CPU_window_v2']] = parent

    for name, text in result.items():
        ast.parse(text)
        destination = (R if name.startswith(('run_', 'produce_')) else S) / (name + '.py')
        assert not destination.exists(), 'Never overwrite an existing/admitted successor'
    for name, text in result.items():
        destination = (R if name.startswith(('run_', 'produce_')) else S) / (name + '.py')
        with destination.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(text)
        guard.pace(len(text.encode()))
        print(name, len(text.encode()), hashlib.sha256(text.encode()).hexdigest(), flush=True)


GENERIC_MAIN = r'''def main(proposal_path, ROOT_review, output):
    began = time.monotonic(); core.environment()
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    proposal_pin = core.record(proposal_path); proposal = core.read_bound(proposal_pin)
    review_pin = core.record(ROOT_review); review = core.read_bound(review_pin, core.resource_guard)
    core.require(proposal['kind'] == PROPOSAL and proposal['production_enabled'] is False
        and proposal['native_benchmark_only'] is False and proposal['vocabulary'] == 9223
        and review['accepted'] is True and review['reviewer'] == '/root'
        and core.same(review['configuration'], proposal_pin)
        and review['scope'] == 'BPE9223_recorded_noise_16room_bounded_CPU_window_v1',
        'Exact ROOT-reviewed generic9223 bounded CPU configuration')
    root = private_output(output)
    core.require(str(root) == proposal['output_root'] and not root.exists(), 'Exact fresh private output reservation')
    sources = proposal['sources']; parent_pin = core.record(__file__)
    core.require(core.same(sources['parent'], parent_pin)
        and core.same(sources['reference'], core.record(reference.__file__))
        and core.same(sources['core'], core.record(core.__file__))
        and type(proposal['freeze_after_setup']) is bool, 'Exact new finite sources and explicit GC mode')
    for pin in sources.values(): core.require(core.same(core.record(pin['path']), pin), 'Bound source differs')
    python_pin = proposal['python']; guard_pin = sources['guard']
    core.require(Path(python_pin['path']).resolve() == core.CPU_ENV / 'Scripts/python.exe'
        and core.same(core.record(python_pin['path']), python_pin)
        and core.same(guard_pin, core.record(PROJECT / 'work/research/ASR18_bounded_CPU_phase_v3.py')),
        'Exact unchanged E CPU interpreter and guard3')
    root.mkdir(parents=True, exist_ok=False)
    def guard():
        core.require(time.monotonic() - began <= 480, 'Bounded whole CPU window480s')
        value = core.resource_guard()
        core.require(value['worker_private_memory']['private_bytes'] <= PARENT_BYTES, 'Unchanged64MiB parent reserve')
        return value
    contract_pin = proposal['source_contract']
    source, plan = public.verify_source_contract(contract_pin, guard=guard)
    numerical = core.read_bound(source['numerical_CPU_contract'], guard)
    core.require(core.same(source['sources']['controller'], parent_pin)
        and core.same(source['sources']['reference'], sources['reference'])
        and core.same(source['sources']['frontend'], sources['frontend'])
        and core.same(source['sources']['core'], sources['core'])
        and source['freeze_after_setup'] == proposal['freeze_after_setup'], 'Exact reviewed source contract')
    outer_parent = proposal['outer_parent_source']
    core.require(core.same(outer_parent, numerical['outer_parent_source'])
        and core.same(core.record(outer_parent['path'], guard), outer_parent), 'Actual external native-owner source binding')
    numerical_request = root / 'numerical_request.json'
    core.write_new(numerical_request, {'kind': core.REQUEST_KIND, 'production_enabled': False,
        'bridge': source['bridge'], 'contract': source['numerical_CPU_contract'], 'update_index': 0})
    common_request = {'kind': public.REQUEST, 'mode': public.MODE, 'production_enabled': False,
        'freeze_after_setup': proposal['freeze_after_setup'], 'source_contract': contract_pin,
        'parent_source': parent_pin, 'bridge': source['bridge'], 'numerical_request': core.record(numerical_request, guard),
        'output_root': str(root), 'ROOT_review': review_pin, 'configuration': proposal_pin}
    target = root / 'reference_request.json'; core.write_new(target, common_request)
    _, reference_proof = run_leaf('reference', core.record(target, guard), sources['reference'], root,
                                  python_pin, guard_pin, parent_pin, guard)
    inventory(root)
    public.verify_reference_window(reference_proof['report'], reference_proof['actual0'], contract_pin=contract_pin, guard=guard)
    from asr18_BPE9223_source_context_v1 import verify_CPU_prior_boundary
    owner = public.create_window(contract_pin, window_id=proposal['window_id'], frontend=numerical['frontend'],
        guard=guard, verify_prior_boundary=lambda boundary: verify_CPU_prior_boundary(boundary, guard=guard))
    frontend_request = {**common_request, 'owner': owner, 'reference_window_report': reference_proof['report'],
                        'reference_window_actual0': reference_proof['actual0']}
    target = root / 'frontend_request.json'; core.write_new(target, frontend_request)
    frontend_child, frontend_proof = run_leaf('frontend', core.record(target, guard), sources['frontend'], root,
                                             python_pin, guard_pin, parent_pin, guard)
    inventory(root)
    closure = public.observe_feature_window(frontend_child, owner_pin=owner, actual_process0=frontend_proof['actual0'],
        report=frontend_proof['report'], output=root / 'actual_frontend_window_closure.json', guard=guard)
    manifest = public.finalize_window(owner, closure, guard=guard)
    reader = public.FeatureWindowReader(owner, manifest, guard=guard)
    frontend = core.read_bound(frontend_proof['report'], guard)
    total_rows = sum(len(update['rows']) for update in plan['updates'])
    core.require(frontend['passed'] is True and frontend['all_updates_reported'] is True
        and frontend['full_rows'] == total_rows and frontend['individual_array_raw_hashes'] == 2 * total_rows
        and frontend['new_NPY_files_written'] == 2 * total_rows and frontend['native_features_ready'] is False,
        'Every full row and both original numeric arrays from the actual aggregate frontend0')
    rows, payloads, operations = [], [], []
    for index, update in enumerate(plan['updates']):
        item = core.read_bound(reader.update_record(index), guard)
        core.require(item['prepared_payload']['state'] == 'prepared_not_committed'
            and item['per_update_child0_claimed'] is False and len(item['rows']) == len(update['rows']),
            'All full copied provisional rows; no fake per-update child0')
        rows.extend({'update_index': index, 'row_index': j, **row} for j, row in enumerate(item['rows']))
        payloads.append(item['prepared_payload'])
        operations.extend({'draw': event['draw'], **operation}
            for event in item['prepared_payload']['rows'] for operation in event['augmentation'].get('operations', []))
    new_owner, new_folder, _ = public.owner_folder(owner, guard)
    store = public.engine._Store(new_folder, new_owner, guard, public.capacity_lock)
    logical = store.inventory(); actual_files = list(new_folder.iterdir())
    physical = ALLOCATION + sum(((p.stat().st_size + ALLOCATION - 1) // ALLOCATION) * ALLOCATION for p in actual_files)
    core.require(logical <= public.CAPACITY and len(actual_files) == len(new_owner['owned_names'])
        and physical <= 512 * 1024**2, 'Exact complete owned logical and cluster-rounded physical512MiB reservation')
    result = {'kind': KIND, 'passed': True, 'producer': core.identity(), 'worker_source': parent_pin,
        'parent_source': outer_parent, 'proposal': proposal_pin, 'source_contract': contract_pin,
        'owner': owner, 'manifest': manifest, 'actual_frontend_window_closure': closure,
        'bridge': source['bridge'], 'phase': plan['phase'], 'window_index': plan['window_index'],
        'reference_actual_aggregate0': reference_proof, 'frontend_actual_aggregate0': frontend_proof,
        'real_actual_child0_count': 2, 'per_update_child0_count_claimed': 0,
        'complete_raw_array_hash_rows': rows, 'complete_ordered_prepared_payloads': payloads,
        'updates': len(plan['updates']), 'full_rows': total_rows, 'individual_array_raw_hashes': 2 * total_rows,
        'new_reference_worker_timings': core.read_bound(reference_proof['report'], guard)['timings'],
        'new_frontend_worker_timings': frontend['timings'], 'freeze_after_setup': proposal['freeze_after_setup'],
        'recorded_room_operations': operations, 'recorded_room_operation_counts': {kind: sum(v['kind'] == kind for v in operations) for kind in ('noise', 'rir')},
        'actual_NEW_feature_logical_bytes': logical, 'actual_NEW_feature_cluster_rounded_bytes': physical,
        'private_output_inventory': inventory(root), 'native_features_ready': False,
        'actual_optimizer_updates': 0, 'sampler_journal_frontier_advanced': False,
        'production_enabled': False, 'native_benchmark_only': False,
        'actual_full_CPU_preparation_speedup_claimed': False, 'wall_seconds': time.monotonic() - began}
    core.write_new(root / 'report.json', result)
    inventory(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True); parser.add_argument('--ROOT-review', required=True)
    parser.add_argument('--folder', required=True)
    args = parser.parse_args(); main(args.config, args.ROOT_review, args.folder)
'''


if __name__ == '__main__':
    main()
