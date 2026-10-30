"""Full retained declared JOINT/DP audit; physical frontend frames are explicit.

No waveform/model/target selection lives here. Unsupported rows remain in the
denominator and block the result. The caller must bind genuine final DATA and
its actual producer closures separately; this pure audit cannot admit TRAIN.
"""
from pathlib import Path
from bpe_broad_production_state_v1 import require
from bpe_broad_audio_v1 import record
from bpe_declared_geometry_index_v1 import open_final, signature
from bpe_windows_final_metadata_lock_v1 import FinalMetadataReadLocks
from asr18_physical_feature_geometry_v2 import from_samples
from tdt_full_native_allocation_budget_v1 import allocation_budget


def audit_index_joint_budget(path, *, expected_rows, expected_tokens, guard=lambda: None):
    require(type(expected_rows) is int and expected_rows > 0 and type(expected_tokens) is int
            and expected_tokens >= expected_rows, 'Exact full retained denominator required')
    path = Path(path).resolve(); before = signature(path)
    require(before['bytes'] <= 2_000_000_000, 'Bounded final scalar index only')
    count = tokens = unsupported = 0; examples = []; largest = None
    maximum = {key: 0 for key in ('encoder_frames', 'target_tokens', 'full16k_samples', 'valid_mask_frames',
        'physical_feature_frames', 'clean_PCM_float32_bytes_per_row', 'original_feature_and_mask_bytes_per_row',
        'original_feature_and_mask_NPY_upper_bytes_per_row')}
    extrema = {}
    guard()
    with FinalMetadataReadLocks([path]):
        connection = open_final(path)
        try:
            connection.set_progress_handler(lambda: (guard() or 0), 100_000)
            rows = connection.execute('SELECT ordinal,encoder_frames,target_tokens,clean16k_samples,mel_frames,status FROM geometry ORDER BY ordinal')
            for ordinal, t, u, n, mel, status in rows:
                if count % 512 == 0: guard()
                count += 1
                require(type(ordinal) is int and ordinal == count <= expected_rows and type(u) is int and u > 0,
                        'Every contiguous retained ordinal and all targets required')
                tokens += u
                try:
                    require(status == 'declared' and all(type(value) is int and value > 0 for value in (t, u, n, mel)),
                            'Unresolved/missing/invalid full declared geometry blocks, never excludes')
                    shape = from_samples(n)
                    require(shape['valid_mask_frames'] == mel and shape['encoder_frames'] == t,
                            'Declared canonical full sample/valid-mask/encoder rounding differs')
                    physical = shape['physical_feature_frames']; feature_bytes = physical * (128*4+1)
                    values = {'encoder_frames': t, 'target_tokens': u, 'full16k_samples': n,
                        'valid_mask_frames': mel, 'physical_feature_frames': physical,
                        'clean_PCM_float32_bytes_per_row': 4*n,
                        'original_feature_and_mask_bytes_per_row': feature_bytes,
                        'original_feature_and_mask_NPY_upper_bytes_per_row': feature_bytes+2*4096}
                    for key, value in values.items():
                        if value > maximum[key]: maximum[key] = value; extrema[key] = {'ordinal': ordinal, 'value': value}
                    budget = allocation_budget(t, u, projected_elements=(t+u+1)*640,
                        head_elements=8716*641, hidden_width=640)
                    if largest is None or budget['total_budget_bytes'] > largest['budget']['total_budget_bytes']:
                        largest = {'ordinal': ordinal, 'budget': budget}
                except ValueError as error:
                    unsupported += 1
                    if len(examples) < 16:
                        examples.append({'ordinal': ordinal, 'declared_T': t, 'full_U': u,
                            'full16k_samples': n, 'status': status, 'reason': str(error)})
            require(count == expected_rows and tokens == expected_tokens, 'Full retained row/token denominator differs')
        finally: connection.close()
        guard(); require(before == signature(path), 'Final scalar index changed during logical audit')
    return {'kind': 'ASR18_all_retained_declared_joint_DP_logical_budget_audit_v2', 'complete': True,
        'passed': unsupported == 0, 'retained_rows': count, 'total_target_tokens': tokens,
        'unsupported_rows': unsupported, 'first_unsupported_rows': examples,
        'maximum_declared_geometry': maximum, 'maximum_declared_geometry_ordinals': extrema,
        'largest_total_joint_budget': largest, 'final_index_signature': before,
        'frontend_geometry': {'physical_frames': 'N//160+1', 'valid_mask_frames': 'N//160',
            'encoder_frames': 'ceil((N//160)/8)', 'feature_dtype': 'float32', 'mask_dtype': 'bool',
            'NPY_header_allowance_bytes_each': 4096},
        'all_original_targets_counted_no_quality_or_language_exclusions': True,
        'actual_audio_decoded': False, 'actual_training_draws_consumed': 0,
        'encoder_decoder_or_worker_peak_fit_approved': False, 'raw_PCM_lifetime_fit_approved': False,
        'production_native_forecast_approved': False, 'TRAIN_admitted': False,
        'allocation_policy_source': record(Path(__file__).with_name('tdt_full_native_allocation_budget_v1.py')),
        'physical_geometry_source': record(Path(__file__).with_name('asr18_physical_feature_geometry_v2.py')),
        'source': record(__file__)}
