"""Full finalized DATA population handoff; never choose a sampler or arm GPU."""
import argparse
from pathlib import Path
import os
import sys

from ASR18_continuous_data_guard_v5 import DataIOGuard, AREA, ROOT
from cpu_preparation_guard_v2 import now
from finalize_ASR18_declared_training_geometry_v1 import MetadataIO, observed_wrapper
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import write_json
from bpe_final_denominator_v1 import audit

ORDERING = {'owner': '/root/memory_diagnosis', 'representation': 'virtual CompletePassCursor',
            'explicit_per_pass_ordinal_files': False, 'caller_preserves_original_source_RNG': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--with-declared-geometry', action='store_true')
    args = parser.parse_args()
    target = Path(args.output).resolve()
    assert target.is_relative_to(AREA.resolve()) and not target.exists()
    assert all(os.environ.get(key) == '1' for key in (
        'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'))
    guard = DataIOGuard('final_two_pass_population_metadata', heavy=False, maximum_bytes=32_000_000_000)
    guard.wait()
    io = MetadataIO(guard)
    catalog_marker = AREA / 'broad_metadata_actual_CLOSED0_v3.json'
    labels_marker = AREA / 'BPE_labels_actual_CLOSED0_v3.json'
    inventory_path = AREA / 'completed_source_inventory_v1.json'
    for wrapper, marker in (
        ('broad_metadata_owned_tool_actual_CLOSED0_v3.json', catalog_marker),
        ('BPE_labels_owned_tool_actual_CLOSED0_v3.json', labels_marker),
        ('completed_source_inventory_owned_tool_actual_CLOSED0_v1.json', inventory_path)):
        observed_wrapper(io, AREA / wrapper, marker)
    catalog = io.json(catalog_marker)
    labels = io.json(labels_marker)
    inventory = io.json(inventory_path)
    assert catalog['metadata_complete'] is True and catalog['committed_metadata_files'] == 730
    assert labels['complete'] is True and labels['vocabulary'] == 8711
    rows, tokens = labels['label_rows'], labels['total_target_tokens']
    assert type(rows) is int and rows > 0 and type(tokens) is int and tokens >= rows
    assert catalog['metadata_retained_rows'] == labels['frontier'] == rows
    assert labels['population_mean_U'] == tokens / rows
    assert inventory['sources_count'] == 784 and inventory['total_source_bytes'] == 718_862_913_959
    paths = dict(preparation_plan=AREA / 'preparation_plan_v1.json',
        finalization_contract=AREA / 'production_finalization_contract_v5.json',
        completed_source_inventory=inventory_path,
        completed_source_inventory_actual_closed=AREA / 'completed_source_inventory_owned_tool_actual_CLOSED0_v1.json',
        final_catalog=ROOT / 'indexes/BPE_broad_weak_labels_v1.sqlite',
        catalog_actual_closed=AREA / 'broad_metadata_owned_tool_actual_CLOSED0_v3.json',
        final_labels=ROOT / 'indexes/BPE_broad_labels_8711_v1.sqlite',
        labels_actual_closed=AREA / 'BPE_labels_owned_tool_actual_CLOSED0_v3.json',
        heldout_metadata=AREA / 'heldout_metadata_bindings_v1.json')
    bindings = {key: io.binding(path) for key, path in paths.items()}
    assert bindings['final_catalog']['sha256'] == catalog['catalog_sha256']
    assert bindings['final_labels']['sha256'] == labels['catalog_sha256']
    previous_io = guard.process.io_counters().read_bytes

    def bounded_sql():
        nonlocal previous_io
        current = guard.process.io_counters().read_bytes
        if current > previous_io:
            guard.pace(current - previous_io)
        previous_io = guard.process.io_counters().read_bytes
        guard.wait()

    denominator = audit(paths['final_catalog'], paths['final_labels'], rows=rows, tokens=tokens, guard=bounded_sql)
    assert denominator['passed'] is True
    if args.with_declared_geometry:
        geometry_path = AREA / 'ASR18_final_declared_training_geometry_v1.json'
        exit_path = AREA / 'ASR18_final_declared_training_geometry_owned_tool_actual_CLOSED0_v1.json'
        actual = io.json(exit_path)
        assert actual['kind'] == 'ASR18_source_bound_actual_tool_exit_v1' and actual['owner'] == '/root/cpu_data_preparation'
        assert type(actual['actual_exit_code']) is int and actual['actual_exit_code'] == 0 and actual['actual_tool_chunk_id']
        assert actual['published_marker'] == io.binding(geometry_path)
        geometry = io.json(geometry_path)
        assert geometry['complete'] is True and geometry['population']['retained_rows'] == rows
        assert geometry['population']['total_target_tokens'] == tokens
        assert geometry['unknown_invalid_counts']['retained_unresolved_rows'] == 0
        bindings.update(declared_geometry_envelope=io.binding(geometry_path),
            declared_geometry_actual_closed=io.binding(exit_path),
            declared_geometry_index=io.binding(ROOT / 'indexes/ASR18_declared_training_geometry_v1.sqlite'))
    io.unchanged()
    assert not any(name in sys.modules for name in ('torch', 'av', 'soundfile', 'scipy', 'pyarrow', 'transformers'))
    value = dict(kind='ASR18_final_two_pass_population_metadata_v1', schema_version=1,
        owner='/root/cpu_data_preparation', created_at_utc=now(), PID=os.getpid(), complete=True,
        pass_count=2, retained_rows=rows, label_rows=rows, frontier=rows,
        total_target_tokens=tokens, population_mean_U=tokens / rows,
        ordinal_domain=dict(min=1, max=rows, contiguous_unique=True),
        quality_filters=[], language_filters=[], bindings=bindings, ordering=ORDERING,
        declared_geometry_bound=args.with_declared_geometry,
        sampler_seed_or_frontier_chosen_by_DATA=False, retention_replay_order_chosen_by_DATA=False,
        retention_replay_clock_separate=True,
        independent_final_denominator_audit=denominator,
        producer_source=io.binding(__file__),
        scope='Final current weak-label catalog and full8711 targets only; source cache, geometry/native admission, retention and cleaner phase remain separate.',
        TRAIN_admitted=False, production_training_started=False)
    write_json(target, value)
    print(dict(final_two_pass_metadata=io.binding(target), retained_rows=rows, total_target_tokens=tokens,
        population_mean_U=tokens / rows, actual_owned_tool_exit_still_required=True,
        production_arm_receipt=False), flush=True)


if __name__ == '__main__':
    main()
