"""Complete DATA aggregate using a separately published geometry-bound two-pass successor."""
import argparse
import os
from pathlib import Path
import sys

from ASR18_continuous_data_guard_v5 import DataIOGuard, AREA, ROOT
from cpu_preparation_guard_v2 import now
from finalize_ASR18_declared_training_geometry_v1 import MetadataIO, base, integer

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import write_json

OWNER = '/root/cpu_data_preparation'
EXPECTED_CONTRACT = '2e46ddc915b166e1195f14bb1d5391b37b51a99b25f35bce1d003ed4b9c222c0'
EXPECTED_VALIDATOR = 'f8536cb791e1f2e66a5a4f7565c3b569bda067e01f77f296f908ed7d12e050ee'
ORDERING = dict(owner='/root/memory_diagnosis', representation='virtual CompletePassCursor',
                explicit_per_pass_ordinal_files=False, caller_preserves_original_source_RNG=True)


def actual(io, wrapper, marker, *, owner=OWNER, four=False):
    proof = io.json(AREA / wrapper)
    assert proof['kind'] == 'ASR18_source_bound_actual_tool_exit_v1' and proof['owner'] == owner
    assert integer(proof['actual_exit_code']) == 0 and proof['actual_tool_chunk_id']
    integer(proof['native_PID'], 1)
    pin = io.binding(marker)
    assert proof['published_marker'] == (pin if four else base(pin))
    assert io.binding(proof['source_path'])['sha256'] == proof['source_SHA256']
    return proof, io.json(marker)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    assert output == (AREA / 'ASR18_final_data_population_for_bounded_preparation_v1.json').resolve()
    assert not output.exists()
    assert all(os.environ.get(key) == '1' for key in (
        'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'))
    guard = DataIOGuard('final_DATA_population_metadata', heavy=False, maximum_bytes=32_000_000_000)
    guard.wait()
    io = MetadataIO(guard)
    paths = dict(
        preparation_plan=AREA / 'preparation_plan_v1.json',
        finalization_contract=AREA / 'production_finalization_contract_v5.json',
        completed_source_inventory=AREA / 'completed_source_inventory_v1.json',
        source_identity_inventory=AREA / 'final_source_identity_inventory_v1.json',
        acquisition_actual_closed=AREA / 'acquisition_owned_tool_actual_CLOSED0_v6.json',
        header_actual_closed=AREA / 'YODAS_header_owned_tool_actual_CLOSED0_v3.json',
        catalog_actual_closed=AREA / 'broad_metadata_owned_tool_actual_CLOSED0_v3.json',
        labels_actual_closed=AREA / 'BPE_labels_owned_tool_actual_CLOSED0_v3.json',
        final_catalog=ROOT / 'indexes/BPE_broad_weak_labels_v1.sqlite',
        final_labels=ROOT / 'indexes/BPE_broad_labels_8711_v1.sqlite',
        heldout_metadata=AREA / 'heldout_metadata_bindings_v1.json',
        heldout_PCM=ROOT / 'analysis/ASR18_memory_diagnosis_v1/heldout_PCM_binding_v1/report.json',
        carrier_coverage=AREA / 'KIRAAT_original_FLAC_carrier_coverage_v1.json',
        carrier_coverage_contract=AREA / 'KIRAAT_FLAC_carrier_coverage_contract_v1.json',
        carrier_stage_contract=AREA / 'KIRAAT_FLAC_normalized_stage_contract_v1.json',
        carrier_coverage_actual_closed=AREA / 'KIRAAT_FLAC_carrier_coverage_owned_tool_actual_CLOSED0_v1.json',
        declared_geometry_envelope=AREA / 'ASR18_final_declared_training_geometry_v1.json',
        declared_geometry_contract=AREA / 'declared_training_geometry_envelope_receipt_contract_v2.json',
        declared_geometry_companion=AREA / 'declared_geometry_envelope_concrete_interface_companion_v1.json',
        declared_geometry_actual_closed=AREA / 'ASR18_final_declared_training_geometry_owned_tool_actual_CLOSED0_v1.json',
        declared_geometry_index=ROOT / 'indexes/ASR18_declared_training_geometry_v1.sqlite')
    # Inspect the reviewed interface without importing its audio-reader dependency.
    import ast
    validator = PROJECT / 'outputs/parakeet_turkish/bpe_final_DATA_population_v1.py'
    parsed = ast.parse(validator.read_text(encoding='utf-8-sig'))
    required = next(node.value.args[0] for node in parsed.body
                    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'REQUIRED' for t in node.targets))
    assert set(paths) == set(ast.literal_eval(required))
    assert io.binding(validator)['sha256'] == EXPECTED_VALIDATOR
    # Missing canonical artifacts fail before any large final DB hash begins.
    assert all(path.is_file() for path in paths.values())
    catalog_marker = AREA / 'broad_metadata_actual_CLOSED0_v3.json'
    labels_marker = AREA / 'BPE_labels_actual_CLOSED0_v3.json'
    contract = io.json(paths['finalization_contract'])
    assert io.binding(paths['finalization_contract'])['sha256'] == EXPECTED_CONTRACT
    _, acquisition = actual(io, paths['acquisition_actual_closed'].name, Path(contract['acquisition_marker_path']))
    _, headers = actual(io, paths['header_actual_closed'].name, Path(contract['header_marker_path']))
    _, catalog = actual(io, paths['catalog_actual_closed'].name, catalog_marker)
    _, labels = actual(io, paths['labels_actual_closed'].name, labels_marker)
    actual(io, 'completed_source_inventory_owned_tool_actual_CLOSED0_v1.json', paths['completed_source_inventory'])
    identity_exit, identity = actual(io, 'final_source_identity_inventory_owned_tool_actual_CLOSED0_v1.json',
                                    paths['source_identity_inventory'], owner='/root/memory_diagnosis')
    identity_guard = io.checked(identity_exit['actual_guarded_completion'])
    assert identity_guard['kind'] == 'ASR18_guarded_CPU_tool_completion_v1'
    assert integer(identity_guard['actual_native_exit_code']) == integer(identity_guard['actual_exec_tool_exit_code']) == 0
    assert identity_guard['native_PID'] == identity_exit['native_PID']
    identity_footer = io.checked(identity_guard['guard_closure'])
    assert integer(identity_footer['actual_exit_code']) == 0 and identity_footer['pid'] == identity_exit['native_PID']
    assert identity_footer['monitor_joined'] is True and identity_footer['monitor_clean'] is True
    assert identity_footer['observed']['monitor_error'] is None
    assert integer(identity_footer['observed']['samples'], 3) >= 3
    assert integer(identity_footer['observed']['peak_private_bytes'], 1) < 2_000_000_000
    assert identity_footer['CPU_private_cap_bytes'] == 2_000_000_000 and identity_footer['CUDA_masked'] is True
    assert identity_footer['thread_environment'] == {key: '1' for key in (
        'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}
    _, carrier = actual(io, paths['carrier_coverage_actual_closed'].name, paths['carrier_coverage'])
    _, geometry = actual(io, paths['declared_geometry_actual_closed'].name, paths['declared_geometry_envelope'], four=True)
    two_pass_path = AREA / 'ASR18_final_two_pass_population_metadata_with_declared_geometry_v1.json'
    _, two_pass = actual(io, 'ASR18_final_two_pass_population_metadata_with_declared_geometry_owned_tool_actual_CLOSED0_v1.json',
                         two_pass_path, four=True)
    assert two_pass['complete'] is True and type(two_pass['pass_count']) is int and two_pass['pass_count'] == 2
    assert two_pass['declared_geometry_bound'] is True
    assert two_pass['ordering'] == ORDERING and two_pass['independent_final_denominator_audit']['passed'] is True
    rows = integer(labels['label_rows'], 1)
    tokens = integer(labels['total_target_tokens'], rows)
    assert acquisition['status'] == 'completed' and acquisition['completed_files'] == 773
    assert headers['status'] == 'completed' and headers['completed_archives'] == 51
    assert catalog['metadata_complete'] is True and catalog['committed_metadata_files'] == 730
    assert catalog['metadata_retained_rows'] == labels['frontier'] == two_pass['retained_rows'] == rows
    assert labels['complete'] is True and labels['vocabulary'] == 8711
    assert two_pass['total_target_tokens'] == tokens and labels['population_mean_U'] == two_pass['population_mean_U'] == tokens / rows
    assert two_pass['ordinal_domain'] == dict(min=1, max=rows, contiguous_unique=True)
    assert carrier['complete'] is True and carrier['source_files'] == len(carrier['sources']) == 729
    assert carrier['all_physical_rows_and_original_compressed_bytes_independently_verified'] is True
    assert geometry['complete'] is True and geometry['KIRAAT_source_sidecars']['sources_count'] == 729
    assert geometry['population']['retained_rows'] == rows and geometry['population']['total_target_tokens'] == tokens
    assert integer(geometry['unknown_invalid_counts']['retained_unresolved_rows']) == 0
    assert identity['complete'] is True and len(identity['sources']) == 784
    bindings = {key: (io.binding(path) if key == 'declared_geometry_index' else base(io.binding(path)))
                for key, path in paths.items()}
    assert bindings['final_catalog']['sha256'] == catalog['catalog_sha256']
    assert bindings['final_labels']['sha256'] == labels['catalog_sha256']
    assert bindings['final_catalog'] == base(two_pass['bindings']['final_catalog']) == base(geometry['bindings']['final_catalog'])
    assert bindings['final_labels'] == base(two_pass['bindings']['final_labels']) == base(geometry['bindings']['final_labels'])
    assert bindings['declared_geometry_index'] == geometry['geometry_index']
    assert two_pass['bindings']['declared_geometry_envelope'] == io.binding(paths['declared_geometry_envelope'])
    assert two_pass['bindings']['declared_geometry_index'] == bindings['declared_geometry_index']
    assert two_pass['bindings']['declared_geometry_actual_closed'] == io.binding(paths['declared_geometry_actual_closed'])
    references = io.json(paths['heldout_PCM'])
    assert references['complete'] is True and references['passed'] is True and references['reference_clips'] == 12273
    assert bindings['heldout_PCM']['sha256'] == contract['heldout_reference']['report_SHA256']
    io.unchanged()
    assert not any(name in sys.modules for name in ('torch', 'av', 'soundfile', 'scipy', 'pyarrow', 'transformers'))
    value = dict(kind='ASR18_final_data_population_for_bounded_preparation_v1', schema_version=1,
        owner=OWNER, created_at_utc=now(), PID=os.getpid(), complete=True, pass_count=2,
        retained_rows=rows, label_rows=rows, frontier=rows, total_target_tokens=tokens,
        population_mean_U=tokens / rows, ordinal_domain=dict(min=1, max=rows, contiguous_unique=True),
        quality_filters=[], language_filters=[], unresolved_geometry_rows=0,
        heldout_reference_PCM_binding_complete=True, bindings=bindings, ordering=two_pass['ordering'],
        two_pass_population_metadata=io.binding(two_pass_path),
        source_identity_inventory_actual_closed=io.binding(AREA / 'final_source_identity_inventory_owned_tool_actual_CLOSED0_v1.json'),
        independent_final_validator_source=io.binding(validator), producer_source=io.binding(__file__),
        runtime_heldout_PCM_and_actual_geometry_checks_before_consumed_draw_still_required=True,
        full_corpus_PCM_decode_validation_claimed=False, native_envelope_or_forecast_approved=False,
        sampler_seed_or_frontier_chosen_by_DATA=False, production_enabled=False, TRAIN_admitted=False)
    write_json(output, value)
    print(dict(final_DATA_population=io.binding(output), retained_rows=rows, total_target_tokens=tokens,
        actual_owned_tool_exit_still_required=True, independent_validator_and_ROOT_arm_still_required=True), flush=True)


if __name__ == '__main__':
    main()
