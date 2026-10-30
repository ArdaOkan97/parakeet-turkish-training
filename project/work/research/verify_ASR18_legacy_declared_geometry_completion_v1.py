"""Independent metadata-only review of all32 completed legacy sidecars."""
import argparse
from pathlib import Path
import sqlite3
import sys

from ASR18_continuous_data_guard_v5 import DataIOGuard, AREA, ROOT
from cpu_preparation_guard_v2 import now
from finalize_ASR18_declared_training_geometry_v1 import (
    MetadataIO, actual_parent, actual_child, sidecar_rows, alania_scalars,
    declared_scalars, integer)

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import write_json


def rejected(call):
    try:
        call()
    except (AssertionError, ValueError, TypeError):
        return True
    raise AssertionError('Expected exact metadata rejection')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--actual-phase', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    target = Path(args.report).resolve()
    assert target.is_relative_to(ROOT.resolve()) and not target.exists()
    guard = DataIOGuard('legacy_declared_geometry_independent_binding_review', heavy=False,
                        maximum_bytes=400_000_000)
    io = MetadataIO(guard)
    closed = actual_parent(io, args.actual_phase, legacy=True)
    finished = io.checked(closed['completed_sources'])
    rows = bytes_verified = unresolved = 0
    hard = {}
    for item in finished:
        result, footer = actual_child(io, item['geometry_completion_actual_exit'],
                                      item['job']['sha256'], 'declared_geometry_completion')
        assert result == item['geometry_result']
        assert result['all_record_SHA_and_payload_SHA_verified'] is True
        geometry = result['declared_STREAMINFO_geometry']
        sidecar = geometry['sidecar']
        guard.pace(64 * 1024)
        with Path(sidecar['path']).open('r', encoding='utf-8') as stream:
            import json
            header = json.loads(stream.readline(64 * 1024))
        receipt = io.checked(header['carrier_receipt'])
        entry = dict(source_SHA256=item['job']['sha256'],
            geometry_sidecar=geometry['sidecar'], row_index=receipt['index'],
            carrier_receipt=io.binding(header['carrier_receipt']['path']), payload=receipt['records'],
            geometry_parser=geometry['parser'], geometry_writer=geometry['writer'],
            physical_rows=geometry['physical_rows'], unresolved_physical_rows=geometry['unresolved_declared_geometry_rows'])
        mapping, counts = sidecar_rows(io, entry)
        rows += len(mapping)
        bytes_verified += result['existing_FLAC_payload_bytes_read_and_SHA_verified']
        unresolved += sum(counts.values())
        if item['job']['path'] == 'data/rest-00104-of-00137.parquet':
            assert mapping[2239][2] == (24000, 3821760, 2547840, 15924, 1991)
            hard['U574_T1991'] = dict(source_SHA256=item['job']['sha256'], physical_row=2239,
                sidecar=sidecar, scalars=mapping[2239][2])
        if item['job']['path'] == 'data/rest-00105-of-00137.parquet':
            assert mapping[110][2] == (24000, 2888400, 1925600, 12035, 1505)
            hard['U484_T1505'] = dict(source_SHA256=item['job']['sha256'], physical_row=110,
                sidecar=sidecar, scalars=mapping[110][2])
        assert not counts
    assert len(finished) == 32 and set(hard) == {'U574_T1991', 'U484_T1505'}
    assert rows == closed['physical_rows'] and unresolved == 0
    runtime = ROOT / 'analysis/ASR18_memory_diagnosis_v1/runtime_geometry_CPU_actual_02/report.json'
    reference = io.json(runtime)
    db = sqlite3.connect((ROOT / 'indexes/BPE_broad_weak_labels_v1.sqlite').as_uri() + '?mode=ro', uri=True, timeout=1)
    db.execute('PRAGMA cache_size=-1024')
    actual_alania = []
    for case in reference['cases']:
        declaration = case['declaration']
        if declaration['source_component'] != 0:
            continue
        number = declaration['ordinal']
        row = db.execute('SELECT dataset,unique_id,start,end FROM clips WHERE ordinal=?', (number,)).fetchone()
        assert row[0] == 'alania_yodas3' and row[1] == case['unique_id']
        scalars = alania_scalars(row[2], row[3])
        assert scalars == tuple(declaration[key] for key in (
            'native_sr', 'native_samples', 'clean16k_samples', 'mel_frames', 'encoder_frames'))
        actual_alania.append(number)
    db.close()
    assert len(actual_alania) == 6
    controls = []
    for name, fn in (
        ('bool scalar rejected', lambda: integer(True)),
        ('reversed interval rejected', lambda: alania_scalars(2., 1.)),
        ('negative interval rejected', lambda: alania_scalars(-1., 1.)),
        ('bool interval rejected', lambda: alania_scalars(False, 1.)),
        ('invalid scalar row status rejected', lambda: declared_scalars({'status':'invented','geometry':{}})),
    ):
        assert rejected(fn)
        controls.append(name)
    io.unchanged()
    assert not any(name in sys.modules for name in ('torch', 'av', 'soundfile', 'scipy', 'pyarrow', 'transformers'))
    value = dict(kind='ASR18_all32_legacy_declared_geometry_independent_mapping_review_v1',
        owner='/root/cpu_data_preparation', at_utc=now(), PID=guard.process.pid, passed=True,
        actual_phase=io.binding(args.actual_phase), source_count=32, physical_rows=rows,
        previously_worker_verified_original_FLAC_bytes=bytes_verified,
        metadata_verifier_audio_payload_bytes_read=0, unresolved_declared_geometry_rows=0,
        every_sidecar_SHA_row_index_clipID_sourceID_recordSHA_formula_checked=True,
        saved_actual_hard_row_geometry_matches=hard, saved_actual_six_Alania_interval_matches=actual_alania,
        controls=controls, producer_source=io.binding(Path(__file__).with_name('finalize_ASR18_declared_training_geometry_v1.py')),
        verifier_source=io.binding(__file__), own_private_bytes=guard.process.memory_info().private,
        full729_geometry_complete=False, TRAIN_admitted=False)
    write_json(target, value)
    print(dict(report=io.binding(target), source_count=32, physical_rows=rows,
        hard_row_matches=2, Alania_actual_matches=6, TRAIN_admitted=False), flush=True)


if __name__ == '__main__':
    main()
