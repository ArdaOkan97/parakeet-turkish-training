"""Pure metadata derivation for a reviewed9223 CPU window; never writes files.

The native owner writes each returned dictionary in its registered fresh AUX,
then binds the actual immutable file. A derived dictionary is not ROOT review,
process completion, feature readiness or sampler/optimizer authority.
"""
from copy import deepcopy
from pathlib import Path
import re

import fastmss_BPE9223_feature_protocol_v10 as protocol

PROTOCOL_SHA256 = 'bd21ce30b41cf85ec5398ff4f340786c413477197bd235702fade8992bdaaab1'
SHARED = frozenset({'production_enabled', 'native_benchmark_only', 'vocabulary', 'sources',
    'owner_validation_only', 'checkpoint_diagnostic_only', 'python', 'outer_parent_source',
    'freeze_after_setup', 'namespace'})
TEMPLATE_KEYS = SHARED | {'kind', 'source_contract_template', 'session', 'full_run_identity_sha256',
    'label_population_contract', 'source_context_source', 'output_namespace', 'maximum_window_updates'}


def _template(template_pin, guard):
    protocol.require(protocol.record(protocol.__file__, guard)['sha256'] == PROTOCOL_SHA256,
        'Exact frozen generic9223 protocol2 required')
    value = protocol.read_bound(template_pin, guard)
    protocol.require(type(value) is dict and set(value) in (TEMPLATE_KEYS, TEMPLATE_KEYS | {'checkpoint_storage_namespace'}) and value['kind'] == protocol.TEMPLATE
        and value['production_enabled'] is False and value['native_benchmark_only'] is False
        and type(value['vocabulary']) is int and value['vocabulary'] == 9223
        and type(value['maximum_window_updates']) is int and value['maximum_window_updates'] == 20
        and type(value['owner_validation_only']) is bool and type(value['checkpoint_diagnostic_only']) is bool
        and value['owner_validation_only'] == value['checkpoint_diagnostic_only']
        and type(value['freeze_after_setup']) is bool,
        'Exact reviewed-template schema, limits and qualification flags')
    protocol.require(type(value['source_contract_template']) is dict
        and 'bridge' not in value['source_contract_template'], 'Static contract excludes only the variable bridge binding')
    return value


def derive_contract(template_pin, bridge_pin, guard=lambda: None):
    """Return an uncommitted exact static contract plus one immutable bridge."""
    value = _template(template_pin, guard)
    planned = protocol.engine._load_bridge(bridge_pin, guard)
    session = protocol.read_bound(planned['session'], guard)
    protocol.require(protocol.same(planned['session'], value['session'])
        and protocol.same(session['full_run_identity_sha256'], value['full_run_identity_sha256'])
        and protocol.same(session['label_population_contract'], value['label_population_contract'])
        and protocol.same(session['source_context_source'], value['source_context_source'])
        and protocol.same(session['owner_validation_only'], value['owner_validation_only'])
        and protocol.same(session['checkpoint_diagnostic_only'], value['checkpoint_diagnostic_only'])
        and planned['phase'] in (protocol.bridge.WEAK, protocol.bridge.CLEANER)
        and 1 <= len(planned['updates']) <= value['maximum_window_updates'],
        'Only this complete same-session/identity/qualification bounded bridge may vary')
    result = deepcopy(value['source_contract_template'])
    result['bridge'] = deepcopy(bridge_pin)
    return result


def derive_configuration(template_pin, source_contract_pin, window_index, window_id, output_root,
                         guard=lambda: None):
    """Return an exact proposal after the caller has written/bound its contract."""
    value = _template(template_pin, guard)
    protocol.integer(window_index, 0, protocol.bridge.PLANNED_UPDATES - 1)
    protocol.require(type(window_id) is str and re.fullmatch(r'[A-Za-z0-9_-]{1,64}', window_id),
        'Exact finite fresh window identifier')
    output = protocol.scoped(output_root)
    namespace = protocol.scoped(value['output_namespace'])
    expected = namespace / window_id / 'CPU' / 'payload' if 'checkpoint_storage_namespace' in value else namespace / window_id
    protocol.require(output == expected, 'Exact template-derived private E output')
    result = {key: deepcopy(value[key]) for key in SHARED}
    result.update(kind=protocol.CONFIGURATION, source_contract=deepcopy(source_contract_pin),
        output_root=str(output), window_id=window_id, window_index=window_index,
        template_configuration=deepcopy(template_pin))
    protocol.require(protocol.verify_window_derivative(value, result, guard) is True,
        'All exact source/population/resource/identity/owned-root derivation checks must pass')
    return result
