"""Produce a full source-bound declared envelope; no waveform/model imports.

Canonical output is impossible without all final actual closures and all729
source sidecars. Header declarations remain forecasts. Runtime must compare the
entire decoded/resampled waveform and encoder length before a consumed draw.
"""
import argparse
import collections
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys

from ASR18_continuous_data_guard_v5 import DataIOGuard, ROOT, AREA
from cpu_preparation_guard_v2 import now
from finalize_ASR18_FLAC_carrier_coverage_v1 import collect

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import read_json, write_json

OWNER = '/root/cpu_data_preparation'
CONTRACT = AREA / 'declared_training_geometry_envelope_receipt_contract_v2.json'
CONTRACT_SHA = '75c28129882d076ad6abc7789b3fed61d3a7f26ccefc1254d7fcf451e6823a7c'
PARSER_SHA = 'f358b187e83d3df1cfb06ebd6d3cdf733937f87964f98341f95b101442c4ceca'
WRITER_SHA = 'e4668593803e831d88c7e89c82841af5b42e845ff366bcd64cc6fb78890f458a'
SUCCESS = 'parsed_declared_geometry_runtime_match_required'
UNRESOLVED = {
    'unresolved_unknown_total_samples': 'unknown_samples',
    'unresolved_invalid_STREAMINFO': 'invalid_header',
    'unresolved_noncard_properties_require_canonical_reader_admission': 'noncard_property',
}


def integer(value, minimum=0):
    assert type(value) is int and value >= minimum, 'Exact integer required; bool is not an integer'
    return value


def base(pin):
    return {key: pin[key] for key in ('path', 'bytes', 'sha256')}


def signature(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


class MetadataIO:
    def __init__(self, guard):
        self.guard = guard
        self.pins = {}

    def binding(self, path, maximum=16_000_000_000):
        path = Path(path).resolve()
        assert path.is_relative_to(ROOT.resolve()) or path.is_relative_to(PROJECT)
        stat = signature(path)
        assert stat['bytes'] <= maximum
        old = self.pins.get(str(path))
        if old:
            assert all(old[key] == value for key, value in stat.items()), 'Frozen binding changed'
            return dict(old)
        digest = hashlib.sha256()
        with path.open('rb', buffering=0) as stream:
            while True:
                self.guard.wait()
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                self.guard.pace(len(chunk))
                digest.update(chunk)
        assert signature(path) == stat
        pin = {**stat, 'sha256': digest.hexdigest()}
        self.pins[str(path)] = pin
        return dict(pin)

    def checked(self, pin):
        actual = self.binding(pin['path'])
        assert set(pin).issubset(actual) and all(actual[key] == value for key, value in pin.items())
        assert actual['bytes'] <= 32_000_000, 'Bound JSON metadata before parsing'
        return read_json(pin['path'])

    def json(self, path):
        return self.checked(self.binding(path, maximum=32_000_000))

    def unchanged(self):
        for pin in self.pins.values():
            assert signature(pin['path']) == {key: pin[key] for key in ('path', 'bytes', 'mtime_ns')}


def observed_wrapper(io, path, marker):
    value = io.json(path)
    assert value['kind'] == 'ASR18_source_bound_actual_tool_exit_v1' and value['owner'] == OWNER
    assert integer(value['actual_exit_code']) == 0 and value['actual_tool_chunk_id']
    integer(value['native_PID'], 1)
    assert value['published_marker'] == base(io.binding(marker))
    return value


def actual_parent(io, path, *, legacy=False):
    value = io.json(path)
    assert value['owner'] == OWNER and integer(value['actual_exit_code']) == 0
    assert value.get('actual_tool_chunk_id') or value.get('actual_tool_chunk')
    assert value['no_owned_carrier_child_running'] is True
    if legacy:
        assert value['kind'] == 'ASR18_bounded_legacy_geometry_completion_actual_tool_CLOSED0_v1'
    return value


def actual_child(io, pin, source_sha, mode):
    proof = io.checked(pin)
    assert proof['kind'] == 'ASR18_actual_owned_FLAC_child_exit_v1'
    assert integer(proof['actual_exit_code']) == 0 and proof['parent_observed_wait'] is True
    assert proof['mode'] == mode and proof['source_sha256'] == source_sha
    pid = integer(proof['native_PID'], 1)
    footer = io.checked(proof['native_closure'])
    assert integer(footer['PID'], 1) == pid and integer(footer['native_return_code']) == 0
    assert footer['watchdog_clean'] is True and integer(footer['samples'], 3) >= 3
    assert integer(footer['peak_private_bytes'], 1) < 2_000_000_000
    assert footer['CPU_private_cap_bytes'] == 2_000_000_000 and footer['CUDA_masked'] is True
    assert footer['thread_caps'] == {key: '1' for key in (
        'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}
    worker = io.binding(proof['worker_source']['path'], maximum=8_000_000)
    assert all(worker[key] == value for key, value in proof['worker_source'].items())
    result_path = Path(pin['path']).parent / 'result.json'
    result = io.json(result_path)
    assert result['mode'] == mode and integer(result['PID'], 1) == pid
    return result, io.binding(proof['native_closure']['path'])


def geometry_definitions(io, last_phase, legacy_paths):
    definitions = {}
    for sha, (item, parent_path) in collect(last_phase).items():
        result_path = Path(item['audited_actual_exit']['path']).parent / 'result.json'
        result = io.json(result_path)
        if 'declared_STREAMINFO_geometry' not in result:
            continue
        actual_parent(io, parent_path)
        checked_result, footer = actual_child(io, item['audited_actual_exit'], sha, 'audit')
        assert checked_result == result
        validation = io.checked(result['validation'])
        assert validation['every_compressed_byte_and_physical_row_identical'] is True
        assert validation['declared_STREAMINFO_geometry'] == result['declared_STREAMINFO_geometry']
        definitions[sha] = (result['declared_STREAMINFO_geometry'], item['audited_actual_exit'],
            footer, io.binding(parent_path), 'during_original_independent_allbyte_audit')
    for parent_path in legacy_paths:
        parent = actual_parent(io, parent_path, legacy=True)
        for item in io.checked(parent['completed_sources']):
            sha = item['job']['sha256']
            assert sha not in definitions, 'Duplicate completed sidecars need explicit versioned reconciliation'
            result, footer = actual_child(io, item['geometry_completion_actual_exit'], sha,
                                          'declared_geometry_completion')
            assert result == item['geometry_result']
            assert result['all_record_SHA_and_payload_SHA_verified'] is True
            definitions[sha] = (result['declared_STREAMINFO_geometry'],
                item['geometry_completion_actual_exit'], footer, io.binding(parent_path),
                'readonly_existing_carrier_completion')
    return definitions


def declared_scalars(row):
    status = row['status']
    value = row['geometry']
    if status in UNRESOLVED:
        return None, UNRESOLVED[status]
    assert status == SUCCESS
    rate = integer(value['sample_rate'], 1)
    native = integer(value['total_native_samples'], 1)
    assert rate == 24000 and integer(value['channels'], 1) == 1
    assert value['known_total_samples'] is True and value['native24k_mono_as_dataset_card'] is True
    clean = (native * 16000 + rate - 1) // rate
    mel, encoder = clean // 160, (clean // 160 + 7) // 8
    assert integer(value['clean16k_samples_from_header'], 1) == clean
    assert integer(value['mel_frames_from_header']) == mel
    assert integer(value['encoder_frames_from_header']) == encoder
    assert isinstance(value['STREAMINFO_prefix_SHA256'], str) and len(value['STREAMINFO_prefix_SHA256']) == 64
    return (rate, native, clean, mel, encoder), None


def sidecar_rows(io, entry, *, return_rows=True):
    """Bounded per-source map; compare all physical rows to original carrier index."""
    pin = entry['geometry_sidecar']
    assert io.binding(pin['path'], maximum=8_000_000) == pin
    index = entry['row_index']
    assert io.binding(index['path'], maximum=32_000_000) == index
    source_rows = sqlite3.connect(Path(index['path']).as_uri() + '?mode=ro&immutable=1', uri=True)
    source_rows.execute('PRAGMA cache_size=-1024')
    expected = source_rows.execute('SELECT row_index,clip_id,source_id,sha256 FROM records ORDER BY row_index')
    output = {}
    counts = collections.Counter()
    total = 0
    try:
        io.guard.pace(pin['bytes'])
        with Path(pin['path']).open('r', encoding='utf-8') as stream:
            first = stream.readline(64 * 1024)
            assert first.endswith('\n')
            header = json.loads(first)
            assert header['kind'] == 'ASR18_FLAC_STREAMINFO_declared_geometry_rows_v1'
            assert header['source_SHA256'] == entry['source_SHA256']
            assert all(header[key] == base(entry[target]) for key, target in (
                ('carrier_receipt', 'carrier_receipt'),))
            assert header['payload'] == entry['payload'] and header['row_index'] == entry['row_index']
            assert header['parser'] == entry['geometry_parser'] and header['writer'] == entry['geometry_writer']
            while True:
                line = stream.readline(64 * 1024)
                if not line:
                    break
                assert line.endswith('\n') and len(line) < 64 * 1024
                row = json.loads(line)
                io.guard.wait()
                assert expected.fetchone() == (row['row_index'], row['clip_id'], row['source_id'],
                                                row['original_compressed_record_SHA256'])
                assert integer(row['row_index']) == total
                scalars, unresolved = declared_scalars(row)
                if unresolved:
                    counts[unresolved] += 1
                if return_rows:
                    output[total] = (row['clip_id'], row['source_id'], scalars, unresolved)
                total += 1
            assert expected.fetchone() is None
    finally:
        source_rows.close()
    assert total == entry['physical_rows']
    assert sum(counts.values()) == entry['unresolved_physical_rows']
    return output, counts


def alania_scalars(start, end):
    assert type(start) in (int, float) and type(end) in (int, float)
    first, last = Fraction(str(start)), Fraction(str(end))
    assert 0 <= first < last
    native = round(last * 48000) - round(first * 48000)
    assert native > 0
    clean = (native * 16000 + 47999) // 48000
    mel = clean // 160
    return 48000, native, clean, mel, (mel + 7) // 8


def group():
    return dict(retained_rows=0, total_target_tokens=0, MAX_T=0, MAX_U=0,
                MAX_T_times_U_plus_1=0, MAX_native_samples=0, MAX_clean16k_samples=0,
                unresolved_retained_rows=0)


def add_group(value, scalars, tokens):
    integer(tokens, 1)
    value['retained_rows'] += 1
    value['total_target_tokens'] += tokens
    value['MAX_U'] = max(value['MAX_U'], tokens)
    if scalars is None:
        value['unresolved_retained_rows'] += 1
        return
    _, native, clean, _, encoder = scalars
    value['MAX_T'] = max(value['MAX_T'], encoder)
    value['MAX_T_times_U_plus_1'] = max(value['MAX_T_times_U_plus_1'], encoder * (tokens + 1))
    value['MAX_native_samples'] = max(value['MAX_native_samples'], native)
    value['MAX_clean16k_samples'] = max(value['MAX_clean16k_samples'], clean)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--last-actual-phase', required=True)
    parser.add_argument('--legacy-actual-closure', action='append', default=[])
    args = parser.parse_args()
    assert all(os.environ.get(key) == '1' for key in (
        'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'))
    guard = DataIOGuard('final_declared_geometry_metadata_producer', heavy=False,
                        maximum_bytes=32_000_000_000)
    guard.wait()
    io = MetadataIO(guard)
    assert io.binding(CONTRACT)['sha256'] == CONTRACT_SHA
    contract = io.json(CONTRACT)
    paths = {key: Path(value) for key, value in contract['required_binding_paths'].items()}
    target = Path(contract['canonical_receipt_path'])
    index_target = paths['declared_geometry_index']
    inventory_target = paths['KIRAAT_declared_sidecar_inventory']
    assert not any(path.exists() for path in (target, index_target, inventory_target))
    assert paths['producer_source'].resolve() == Path(__file__).resolve()
    assert io.binding(paths['canonical_clean_decoder_length_recipe']) == contract['canonical_clean_decoder_length_recipe']
    assert io.binding(contract['ROOT_declared_geometry_acknowledgment']['path']) == contract['ROOT_declared_geometry_acknowledgment']
    for wrapper, marker in (
        ('completed_source_inventory_actual_closed', paths['completed_source_inventory']),
        ('original_FLAC_carrier_coverage_actual_closed', paths['original_FLAC_carrier_coverage']),
        ('catalog_actual_closed', AREA / 'broad_metadata_actual_CLOSED0_v3.json'),
        ('labels_actual_closed', AREA / 'BPE_labels_actual_CLOSED0_v3.json')):
        observed_wrapper(io, paths[wrapper], marker)
    source_inventory = io.json(paths['completed_source_inventory'])
    coverage = io.json(paths['original_FLAC_carrier_coverage'])
    assert source_inventory['sources_count'] == 784
    assert coverage['complete'] is True and coverage['source_files'] == 729
    assert coverage['all_physical_rows_and_original_compressed_bytes_independently_verified'] is True
    plan = io.json(paths['preparation_plan'])
    jobs = {item['sha256']: item for item in plan['download_jobs']
            if item['repository'] == 'serdarcaglar/kiraat'
            and Path(item['path']).name.startswith(('train-', 'rest-'))}
    assert len(jobs) == 729
    definitions = geometry_definitions(io, args.last_actual_phase, args.legacy_actual_closure)
    assert set(definitions) == set(jobs), 'Exactly all729 declared sidecars and actual parent/child closures required'
    originals = {item['source_sha256']: item for item in coverage['sources']}
    assert len(originals) == 729 and set(originals) == set(jobs)
    sources, physical_counts = [], collections.Counter()
    for sha, job in sorted(jobs.items(), key=lambda pair: pair[1]['path']):
        guard.wait()
        original = originals[sha]
        declaration, child, footer, parent, mode = definitions[sha]
        assert declaration['source_SHA256'] == sha
        assert declaration['physical_rows'] == original['physical_rows']
        assert signature(job['destination']) == original['source_signature']
        assert declaration['parser']['sha256'] == PARSER_SHA and declaration['writer']['sha256'] == WRITER_SHA
        receipt = io.checked(original['carrier_receipt'])
        assert receipt['complete'] is True and receipt['source']['source_sha256'] == sha
        assert declaration['physical_rows'] == receipt['physical_rows']
        assert declaration['unresolved_declared_geometry_rows'] >= 0
        assert io.binding(declaration['parser']['path']) == declaration['parser']
        assert io.binding(declaration['writer']['path']) == declaration['writer']
        for key in ('records', 'index'):
            assert signature(receipt[key]['path']) == {field: receipt[key][field] for field in ('path', 'bytes', 'mtime_ns')}
        normalized = original['independent_audit_actual_closed']
        assert normalized['mode'] == 'audit' and integer(normalized['actual_exit_code']) == 0
        normalized_proof = io.checked(normalized['actual_exit_proof'])
        assert normalized_proof['source_sha256'] == sha and normalized_proof['mode'] == 'audit'
        assert integer(normalized_proof['actual_exit_code']) == 0
        entry = dict(source_SHA256=sha, source_signature=signature(job['destination']),
            source_acquisition_receipt=io.binding(original['source_acquisition_receipt']['path']),
            carrier_receipt=io.binding(original['carrier_receipt']['path']), payload=dict(receipt['records']),
            row_index=io.binding(receipt['index']['path']), geometry_sidecar=dict(declaration['sidecar']),
            geometry_parser=dict(declaration['parser']), geometry_writer=dict(declaration['writer']),
            physical_rows=receipt['physical_rows'], original_FLAC_bytes=receipt['records']['bytes'],
            unresolved_physical_rows=declaration['unresolved_declared_geometry_rows'], collection_mode=mode,
            declared_geometry_actual_child_exit=io.binding(child['path']),
            declared_geometry_native_guard_footer=footer, declared_geometry_actual_parent_tool_closure=parent,
            original_normalized_independent_allbyte_actual_exit=io.binding(normalized['actual_exit_proof']['path']))
        assert set(entry) == set(contract['KIRAAT_source_entry_exact_keys'])
        _, counts = sidecar_rows(io, entry, return_rows=False)
        physical_counts.update(counts)
        sources.append(entry)
        write_json(AREA / 'final_declared_geometry_producer_status_v1.json',
            dict(at_utc=now(), PID=os.getpid(), stage='validating_all729_source_sidecars',
                 verified_source_sidecars=len(sources), required_source_sidecars=729, TRAIN_admitted=False))
    inventory = dict(kind='ASR18_complete_KIRAAT_declared_geometry_sidecar_inventory_v1', schema_version=1,
        complete=True, owner=OWNER, created_at_utc=now(), preparation_plan=io.binding(paths['preparation_plan']),
        completed_source_inventory=io.binding(paths['completed_source_inventory']),
        original_FLAC_carrier_coverage=io.binding(paths['original_FLAC_carrier_coverage']), sources_count=729,
        all_physical_rows=sum(item['physical_rows'] for item in sources),
        all_original_FLAC_bytes=sum(item['original_FLAC_bytes'] for item in sources),
        all_sidecar_bytes=sum(item['geometry_sidecar']['bytes'] for item in sources),
        sources=sources, TRAIN_admitted=False)
    assert set(inventory) == set(contract['KIRAAT_sidecar_inventory_exact_keys'])
    by_source = {entry['source_signature']['path']: entry for entry in sources}
    meta_marker = io.json(AREA / 'broad_metadata_actual_CLOSED0_v3.json')
    labels_marker = io.json(AREA / 'BPE_labels_actual_CLOSED0_v3.json')
    assert io.binding(paths['final_catalog'])['sha256'] == meta_marker['catalog_sha256']
    assert io.binding(paths['final_labels'])['sha256'] == labels_marker['catalog_sha256']
    catalog = sqlite3.connect(paths['final_catalog'].as_uri() + '?mode=ro&immutable=1', uri=True)
    labels = sqlite3.connect(paths['final_labels'].as_uri() + '?mode=ro&immutable=1', uri=True)
    for db in (catalog, labels):
        db.execute('PRAGMA cache_size=-2048')
        db.execute('PRAGMA temp_store=MEMORY')
        db.set_progress_handler(lambda: (guard.wait() or 0), 10000)
    rows, low, high = catalog.execute('SELECT COUNT(*),MIN(ordinal),MAX(ordinal) FROM clips').fetchone()
    assert integer(rows, 1) == meta_marker['metadata_retained_rows'] == labels_marker['label_rows']
    assert low == 1 and high == rows
    lrows, llow, lhigh, tokens = labels.execute(
        'SELECT COUNT(*),MIN(ordinal),MAX(ordinal),SUM(target_tokens) FROM labels').fetchone()
    assert (lrows, llow, lhigh) == (rows, 1, rows) and tokens == labels_marker['total_target_tokens']
    assert labels.execute('SELECT frontier,rows,tokens FROM progress WHERE id=1').fetchone() == (rows, rows, tokens)
    assert catalog.execute("SELECT rows_seen FROM files WHERE file='alania_all_metadata'").fetchone() == (835267,)
    work = index_target.with_name(index_target.stem + '.build.sqlite')
    assert not work.exists(), 'Preserve any unresolved earlier index; explicit reconciliation required'
    geometry = sqlite3.connect(work)
    geometry.execute('PRAGMA cache_size=-2048')
    geometry.execute('PRAGMA temp_store=MEMORY')
    geometry.execute('CREATE TABLE geometry (' + contract['geometry_index_schema']['geometry'] + ')')
    groups = {key: group() for key in contract['group_keys_exact']}
    counters = collections.Counter()
    active_source, source_map, ordinal, completed = None, None, 0, 0
    cr = catalog.execute('SELECT ordinal,dataset,unique_id,source_id,shard,row_index,start,end FROM clips ORDER BY ordinal')
    lr = labels.execute('SELECT ordinal,target_tokens,length(target_ids_u32) FROM labels ORDER BY ordinal')
    try:
        while True:
            batch = cr.fetchmany(512)
            if not batch:
                break
            guard.pace(len(batch) * 256)
            for number, component, unique, source_id, shard, physical, start, end in batch:
                ordinal += 1
                assert integer(number, 1) == ordinal
                label = lr.fetchone()
                assert label is not None and label[0] == ordinal
                u = integer(label[1], 1)
                assert integer(label[2], 4) == 4 * u
                counters['U_gt_458'] += int(u > 458)
                counters['U_plus_1_gt_1024'] += int(u + 1 > 1024)
                reason = None
                if component == 'alania_yodas3':
                    category, source_component = component, 0
                    try:
                        scalars = alania_scalars(start, end)
                    except (AssertionError, ValueError, OverflowError):
                        scalars, reason = None, 'invalid_interval'
                else:
                    assert component == 'kiraat'
                    category, source_component = component, 1
                    if shard not in by_source:
                        scalars, reason = None, 'missing_sidecar'
                    else:
                        if shard != active_source:
                            source_map, _ = sidecar_rows(io, by_source[shard])
                            active_source = shard
                        assert integer(physical) in source_map
                        clip, original_source_id, scalars, reason = source_map[physical]
                        assert unique == 'kiraat:' + clip and source_id == original_source_id
                if scalars is not None and scalars[4] == 0:
                    scalars, reason = None, 'zero_encoder_frame'
                status = 'declared'
                if reason:
                    counters[reason] += 1
                    status = 'unresolved_' + {
                        'unknown_samples': 'unknown_samples', 'invalid_header': 'invalid_header',
                        'noncard_property': 'noncard_properties', 'invalid_interval': 'invalid_interval',
                        'zero_encoder_frame': 'zero_encoder_frames', 'missing_sidecar': 'missing_sidecar'}[reason]
                for key in ('all', category):
                    add_group(groups[key], scalars, u)
                values = scalars if scalars is not None else (None,) * 5
                geometry.execute('INSERT INTO geometry VALUES (?,?,?,?,?,?,?,?,?)',
                    (ordinal, source_component, *values, u, status))
            geometry.commit()
            completed += len(batch)
            assert work.stat().st_size < 2_000_000_000
            write_json(AREA / 'final_declared_geometry_producer_status_v1.json',
                dict(at_utc=now(), PID=os.getpid(), stage='joining_final_retained_catalog_targets_geometry',
                     final_rows=rows, geometry_rows=completed, retained_unresolved_rows=groups['all']['unresolved_retained_rows'],
                     TRAIN_admitted=False))
        assert lr.fetchone() is None and completed == ordinal == rows
    finally:
        geometry.commit()
        geometry.close()
        catalog.close()
        labels.close()
    assert groups['all']['total_target_tokens'] == tokens
    unknown_counts = dict(
        all_physical_KIRAAT_unknown_samples_rows=physical_counts['unknown_samples'],
        all_physical_KIRAAT_invalid_header_rows=physical_counts['invalid_header'],
        all_physical_KIRAAT_noncard_property_rows=physical_counts['noncard_property'],
        retained_unknown_samples_rows=counters['unknown_samples'],
        retained_invalid_header_rows=counters['invalid_header'],
        retained_noncard_property_rows=counters['noncard_property'],
        retained_invalid_interval_rows=counters['invalid_interval'],
        retained_zero_encoder_frame_rows=counters['zero_encoder_frame'],
        retained_missing_sidecar_rows=counters['missing_sidecar'],
        retained_unresolved_rows=groups['all']['unresolved_retained_rows'])
    assert set(unknown_counts) == set(contract['unknown_invalid_count_keys_exact'])
    if unknown_counts['retained_unresolved_rows']:
        write_json(AREA / 'final_declared_geometry_BLOCKED_preserved_v1.json',
            dict(kind='ASR18_declared_geometry_finalization_blocked_v1', at_utc=now(), owner=OWNER,
                 preserved_index_work=str(work), retained_rows=rows, unknown_invalid_counts=unknown_counts,
                 labels_unchanged=True, no_rows_dropped=True, TRAIN_admitted=False))
        raise RuntimeError('Unresolved retained declarations block canonical publication; preserve work and reconcile')
    io.unchanged()
    guard.wait()
    assert not inventory_target.exists() and not index_target.exists()
    write_json(inventory_target, inventory)
    work.rename(index_target)
    index_pin = io.binding(index_target)
    bindings = {key: io.binding(path) for key, path in paths.items()}
    assert bindings['declared_geometry_index'] == index_pin
    all_group = groups['all']
    population = dict(retained_rows=rows, catalog_min_ordinal=1, catalog_max_ordinal=rows,
        labels_min_ordinal=1, labels_max_ordinal=rows, geometry_min_ordinal=1, geometry_max_ordinal=rows,
        total_target_tokens=tokens, **{key: all_group[key] for key in (
            'MAX_T', 'MAX_U', 'MAX_T_times_U_plus_1', 'MAX_native_samples', 'MAX_clean16k_samples')},
        count_U_gt_458=counters['U_gt_458'], count_U_plus_1_gt_1024=counters['U_plus_1_gt_1024'])
    assert set(population) == set(contract['population_exact_keys'])
    receipt = dict(kind=contract['final_receipt_kind'], schema_version=1, complete=True, owner=OWNER,
        created_at_utc=now(), PID=os.getpid(), bindings=bindings, scope=contract['scope_constants'],
        population=population, groups=groups, unknown_invalid_counts=unknown_counts,
        KIRAAT_source_sidecars=dict(inventory=bindings['KIRAAT_declared_sidecar_inventory'], sources_count=729,
            all_physical_rows=inventory['all_physical_rows'], all_original_FLAC_bytes=inventory['all_original_FLAC_bytes'],
            all_sidecar_bytes=inventory['all_sidecar_bytes']),
        Alania_interval_declaration=contract['Alania_interval_declaration'], geometry_index=index_pin,
        runtime_contract=contract['runtime_contract'],
        geometry_basis='Source-bound declared STREAMINFO and retained rational48k intervals; full waveform/runtime encoder match still mandatory',
        TRAIN_admitted=False)
    assert set(receipt) == set(contract['final_receipt_exact_keys'])
    io.unchanged()
    assert not any(name in sys.modules for name in ('torch', 'av', 'soundfile', 'scipy', 'pyarrow', 'transformers'))
    write_json(target, receipt)
    print(dict(final_declared_envelope=io.binding(target), retained_rows=rows, MAX_T=all_group['MAX_T'],
        MAX_U=all_group['MAX_U'], actual_owned_tool_exit_still_required=True, TRAIN_admitted=False), flush=True)


if __name__ == '__main__':
    main()
