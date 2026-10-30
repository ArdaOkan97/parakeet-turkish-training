"""Metadata-only CPU payload forecast; no decoder, source-media scan or admission."""
import argparse
import heapq
import os
from pathlib import Path
import sqlite3
import sys
from ASR18_continuous_data_guard_v5 import DataIOGuard, AREA, ROOT
from cpu_preparation_guard_v2 import now
from finalize_ASR18_declared_training_geometry_v1 import MetadataIO, integer, signature
P = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(P / 'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import write_json


def buffers(row, channel_cap):
    """Explicit float32 array model, conditional on runtime channels <= cap."""
    n, sr, out = row['native_samples'], row['native_sr'], row['clean16k_samples']
    for value in (n, sr, out, channel_cap):
        integer(value, 1)
    channels = 1 if row['source_component'] == 1 else channel_cap
    # KIRAAT has a full mono STREAMINFO declaration. YODAS has only an interval
    # declaration; allow 2s seek preroll plus1s frame/rounding allowance here.
    native_with_allowance = n + (3 * sr if row['source_component'] == 0 else 0)
    return 4 * (2 * channels * native_with_allowance + 2 * native_with_allowance + 4 * out) + 1024**2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    assert output == (AREA / 'ASR18_final_declared_CPU_resource_envelope_v1.json').resolve() and not output.exists()
    guard = DataIOGuard('final_declared_CPU_resource_envelope', heavy=False, maximum_bytes=16_000_000_000)
    guard.wait()
    io = MetadataIO(guard)
    envelope_path = AREA / 'ASR18_final_declared_training_geometry_v1.json'
    proof_path = AREA / 'ASR18_final_declared_training_geometry_owned_tool_actual_CLOSED0_v1.json'
    proof = io.json(proof_path)
    assert proof['kind'] == 'ASR18_source_bound_actual_tool_exit_v1' and proof['owner'] == '/root/cpu_data_preparation'
    assert integer(proof['actual_exit_code']) == 0 and proof['actual_tool_chunk_id']
    assert proof['published_marker'] == io.binding(envelope_path)
    envelope = io.json(envelope_path)
    assert envelope['complete'] is True and envelope['unknown_invalid_counts']['retained_unresolved_rows'] == 0
    assert envelope['KIRAAT_source_sidecars']['sources_count'] == 729
    index_pin = envelope['bindings']['declared_geometry_index']
    assert io.binding(index_pin['path']) == index_pin == envelope['geometry_index']
    catalog_pin = envelope['bindings']['final_catalog']
    assert signature(catalog_pin['path']) == {key: catalog_pin[key] for key in ('path', 'bytes', 'mtime_ns')}
    db = sqlite3.connect(Path(index_pin['path']).as_uri() + '?mode=ro&immutable=1', uri=True)
    db.execute('PRAGMA cache_size=-4096')
    db.set_progress_handler(lambda: (guard.wait() or 0), 10000)
    tops = {key: [] for key in ('native_samples', 'native_sr', 'clean16k_samples', 'encoder_frames',
                              'per_channel_native_float32_bytes', 'buffer_model_channel_cap1',
                              'buffer_model_channel_cap2', 'buffer_model_channel_cap8')}
    counts = {0: 0, 1: 0}
    total = 0
    cursor = db.execute('SELECT ordinal,source_component,native_sr,native_samples,clean16k_samples,mel_frames,encoder_frames,target_tokens,status FROM geometry ORDER BY ordinal')
    while batch := cursor.fetchmany(2048):
        guard.pace(len(batch) * 128)
        for ordinal, component, sr, native, clean, mel, frames, u, status in batch:
            total += 1
            assert integer(ordinal, 1) == total and component in (0, 1) and status == 'declared'
            for value in (sr, native, clean, frames, u):
                integer(value, 1)
            counts[component] += 1
            row = dict(ordinal=ordinal, source_component=component, native_sr=sr, native_samples=native,
                       clean16k_samples=clean, mel_frames=mel, encoder_frames=frames, target_tokens=u,
                       original_channels_declared=(1 if component == 1 else None),
                       channel_basis=('complete mono STREAMINFO declarations' if component == 1 else 'original WebM channels not established by rational interval declaration'))
            values = {key: row[key] for key in ('native_samples', 'native_sr', 'clean16k_samples', 'encoder_frames')}
            values['per_channel_native_float32_bytes'] = native * 4
            values.update({f'buffer_model_channel_cap{cap}': buffers(row, cap) for cap in (1, 2, 8)})
            for metric, value in values.items():
                candidate = (value, -ordinal, row)
                if len(tops[metric]) < 8:
                    heapq.heappush(tops[metric], candidate)
                elif candidate[:2] > tops[metric][0][:2]:
                    heapq.heapreplace(tops[metric], candidate)
    db.close()
    assert total == envelope['population']['retained_rows']
    assert max(item[0] for item in tops['native_samples']) == envelope['population']['MAX_native_samples']
    assert max(item[0] for item in tops['clean16k_samples']) == envelope['population']['MAX_clean16k_samples']
    catalog = sqlite3.connect(Path(catalog_pin['path']).as_uri() + '?mode=ro&immutable=1', uri=True)
    catalog.execute('PRAGMA cache_size=-2048')
    selected = {}
    for values in tops.values():
        for _, _, row in values:
            if row['ordinal'] in selected:
                continue
            found = catalog.execute('SELECT dataset,unique_id,source_id,shard,row_index,start,end,duration FROM clips WHERE ordinal=?', (row['ordinal'],)).fetchone()
            assert found is not None
            selected[row['ordinal']] = dict(zip(('dataset', 'unique_id', 'source_id', 'shard', 'physical_row_index', 'start', 'end', 'duration'), found))
    catalog.close()
    maxima = {metric: [dict(value=value, **row, source_identity=selected[row['ordinal']])
                       for value, _, row in sorted(values, reverse=True)] for metric, values in tops.items()}
    assert signature(catalog_pin['path']) == {key: catalog_pin[key] for key in ('path', 'bytes', 'mtime_ns')}
    io.unchanged()
    assert not any(name in sys.modules for name in ('torch', 'av', 'soundfile', 'scipy', 'pyarrow', 'transformers'))
    report = dict(kind='ASR18_declared_CPU_array_payload_resource_envelope_v1', schema_version=1,
        owner='/root/cpu_data_preparation', created_at_utc=now(), PID=os.getpid(), complete_declared_scalar_population=True,
        retained_rows=total, bindings=dict(declared_envelope=io.binding(envelope_path),
            declared_envelope_actual_closed=io.binding(proof_path), declared_geometry_index=index_pin,
            final_catalog=catalog_pin, producer_source=io.binding(__file__)),
        exact_original_channel_coverage=dict(KIRAAT_mono_rows=counts[1],
            Alania_original_WebM_channels_unknown_rows=counts[0], exact_global_original_MAX_channels=None),
        canonical_output_channels=1, maxima_and_source_correspondence=maxima,
        conservative_float32_array_model=dict(
            expression='4*(2*channels*(native_samples+native_allowance)+2*(native_samples+native_allowance)+4*clean16k_samples)+1048576',
            native_allowance='Alania3*native_sr (2s seek preroll+1s frame allowance); KIRAAT0/full clip',
            live_buffers='two native interleaved, two mono-native, four canonical16k, plus1MiB array padding',
            source_channel_caps_are_conditional_runtime_checks=[1, 2, 8],
            KIRAAT_uses_exact_declared_mono_in_every_conditional_model=True,
            two_simultaneous_jobs_payload_bound_expression='2*per_job_buffer_model',
            total_private_RSS_or_library_workspace_bound_claimed=False),
        current_original_channels_not_inferred_for_Alania=True,
        catalog_hash_reused_from_genuine_final_geometry_producer_with_current_signature_check=True,
        media_bytes_read=0, audio_decoded=False, rows_filtered=0,
        runtime_requirement='Read actual stream channels/rate before allocation; enforce native/clean16k/encoder-length match and actual byte/lifetime/private caps before any consumed draw. Stop and reconcile on mismatch; never crop/drop.',
        actual_full_corpus_decode_or_CPU_allocation_proof_claimed=False,
        production_enabled=False, TRAIN_admitted=False)
    write_json(output, report)
    print(dict(resource_envelope=io.binding(output), retained_rows=total,
               declared_original_channels_unknown_rows=counts[0], actual_owned_exit_still_required=True))


if __name__ == '__main__':
    main()
