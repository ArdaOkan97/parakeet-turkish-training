"""One source-bound physical EOF reconciliation, never a general decoder tolerance.

Original plans, annotations and scalar indexes remain historical evidence.
Every ordinary row uses the frozen reader. This sole certified full recording
uses every physical decoded sample, the original mono/resample, and the same T/U.
"""
from copy import deepcopy
from fractions import Fraction
import hashlib
from pathlib import Path
import sqlite3
import sys

import fastmss_BPE9223_cpu_common_v12 as core
from bpe_broad_audio_v1 import AV_TARGET, Extent, resample
from fastmss_bounded_original_audio_v2 import BoundedOriginalReader
from fastmss_raw_lifetime_v1 import raw_plan, chunked_hashes, canonical_source_path, source_identity_binding

ROW_ID = 'alania:6619bf48d0238fefe91cea29f78c5287_00000'
ORDINAL = 400485
require, digest, same, record = core.require, core.digest, core.same, core.record


def checked_certificate(policy, guard):
    require(policy['source'] == record(__file__, guard), 'Exact physical-geometry overlay source')
    certificate = core.read_bound(policy['certificate'], guard)
    review = core.read_bound(policy['ROOT_review'], guard)
    require(review['accepted'] is True and review['reviewer'] == '/root'
            and review['scope'] == 'ASR18_source_bound_original_full_clip_geometry_overlay_v1'
            and review['overlay_source'] == policy['source']
            and review['certificate'] == policy['certificate'], 'Exact reviewed one-row correction')
    require(certificate['kind'] == 'ASR18_source_bound_one_original_full_clip_geometry_certificate_v1'
            and certificate['original_row']['ordinal'] == ORDINAL
            and certificate['original_row']['unique_id'] == ROW_ID
            and certificate['original_row']['start'] == 0
            and certificate['original_row']['end'] == 15.488
            and digest(certificate['original_row']) == certificate['original_row_sha256']
            and digest(certificate['full_target_ids']) == certificate['full_target_ids_sha256']
            and len(certificate['full_target_ids']) == 82, 'Exact original row/annotation/full labels')
    old, actual = certificate['original_declaration'], certificate['actual_geometry']
    require(old == dict(ordinal=ORDINAL, source_component=0, native_sr=48000,
            native_samples=743424, clean16k_samples=247808, mel_frames=1548,
            encoder_frames=194, target_tokens=82, status='declared')
            and actual == dict(native_sr=48000, native_samples=743411,
            clean16k_samples=247804, mel_frames=1548, encoder_frames=194, target_tokens=82)
            and certificate['decoded_frames'] == 775
            and certificate['max_abs_PTS_residual_samples'] == '24', 'Exact measured EOF, unchanged T/U')
    observed = core.read_bound(certificate['independent_full_member_observation'], guard)
    require(observed['full_original_frames_used_once'] is True
            and observed['synthesized_or_skipped_samples'] == 0
            and observed['actual_geometry'] == actual
            and all(observed[k] == certificate[k] for k in ('member_sha256', 'native_f32_sha256',
                'clean_f32_sha256', 'clean_i16_sha256', 'recording_index_entry', 'original_TAR')),
            'Exact independent full-member physical PCM observation')
    return certificate


def match_original(certificate, row, declaration, expected):
    require(row == certificate['original_row']
            and declaration == certificate['original_declaration']
            and expected['full_target_ids'] == certificate['full_target_ids']
            and expected['raw_text_sha256'] == certificate['raw_label_sha256'],
            'No source/annotation/label/declaration substitution for certified row')


def decode_full_member(reader, row, certificate, *, guard, channels_bound, frame_samples_bound):
    """Decode to genuine EOF, with no seek, trimming, padding or sample repair."""
    import numpy as np
    if str(AV_TARGET) not in sys.path: sys.path.insert(0, str(AV_TARGET))
    import av
    require(av.__version__ == '16.0.1', 'Pinned original PyAV')
    path, job = reader.verify_source(canonical_source_path(row, core.ROOT))
    original = certificate['original_TAR']; stat = path.stat()
    require(str(path) == original['path'] and stat.st_size == original['bytes']
            and stat.st_mtime_ns == original['mtime_ns']
            and job['sha256'] == original['recorded_sha256'], 'Original source identity/signature')
    index = core.ROOT / ('indexes/yodas3_tar_' + row['shard'] + '.sqlite')
    db = sqlite3.connect(index.as_uri() + '?mode=ro&immutable=1', uri=True)
    try:
        entry = db.execute('SELECT member_name,offset,bytes FROM recordings WHERE recording_id=?',
                           (row['source_id'],)).fetchone()
    finally: db.close()
    require(entry is not None and dict(zip(('member_name', 'offset', 'bytes'), entry))
            == certificate['recording_index_entry'], 'Exact original TAR member, no replacement')
    name, offset, size = entry
    sha = hashlib.sha256()
    with path.open('rb') as source:
        source.seek(offset); left = size
        while left:
            guard(); block = source.read(min(1024**2, left))
            require(block, 'Entire original member'); sha.update(block); left -= len(block)
    require(sha.hexdigest() == certificate['member_sha256'], 'Exact original compressed bytes')
    n = certificate['actual_geometry']['native_samples']
    guard(); pcm = np.empty(n, dtype=np.float32); cursor = 0; decoded = 0
    with Extent(path, offset, size, guard) as extent, av.open(extent, mode='r', format='webm') as container:
        require(len(container.streams.audio) == 1, 'One original audio stream')
        stream = container.streams.audio[0]; stream.codec_context.thread_count = 1
        require(stream.codec_context.name == 'opus' and stream.time_base == Fraction(1, 1000),
                'Certified Opus millisecond clock')
        origin = None; channels = None
        for frame in container.decode(stream):
            guard(); decoded += 1
            require(frame.pts is not None and frame.sample_rate == 48000
                    and frame.time_base == stream.time_base and frame.format.is_planar
                    and frame.format.bytes <= 8 and 0 < frame.samples <= frame_samples_bound,
                    'Original per-frame raw envelope')
            if origin is None:
                origin = frame.pts * frame.time_base; channels = len(frame.layout.channels)
                require(origin == 0 and 1 <= channels <= channels_bound, 'Certified recording origin/channels')
            require(len(frame.layout.channels) == channels
                    and abs((frame.pts * frame.time_base - origin) * 48000 - cursor) <= 24,
                    'Exact half-tick clock; true gaps remain unresolved')
            require(cursor + frame.samples <= n, 'No extra samples beyond certified physical EOF')
            raw = frame.to_ndarray()
            require(raw.shape == (channels, frame.samples), 'Original planar shape')
            if np.issubdtype(raw.dtype, np.integer):
                raw = raw.astype(np.float32) / float(2**(np.iinfo(raw.dtype).bits - 1))
            else: raw = raw.astype(np.float32, copy=False)
            mono = raw.mean(axis=0, dtype=np.float32) if channels > 1 else raw[0]
            pcm[cursor:cursor + frame.samples] = mono; cursor += frame.samples
            del raw, mono
        require(cursor == n and decoded == certificate['decoded_frames'], 'Exact genuine EOF count, no clamping')
        logical = extent.logical_bytes_read
    require(hashlib.sha256(memoryview(pcm).cast('B')).hexdigest() == certificate['native_f32_sha256'],
            'Every physical original sample exactly once')
    output = resample(pcm, 48000)
    f32, i16 = chunked_hashes(output, guard=guard)
    require(len(output) == certificate['actual_geometry']['clean16k_samples']
            and f32 == certificate['clean_f32_sha256'] and i16 == certificate['clean_i16_sha256'],
            'Original full resample and canonical PCM hashes')
    return output, dict(format='YODAS_Opus_WebM_TAR_extent', source_file_sha256=job['sha256'],
        member_name=name, member_offset=offset, member_bytes=size, member_logical_read_bytes=logical,
        native_sample_rate=48000, native_samples=n, actual_native_channels=channels,
        decoded_frames=decoded, PTS_origin=str(origin), source_interval_seconds=['0', str(row['end'])],
        full_physical_recording_used=True, original_annotation_end_preserved=True,
        original_derived_native_samples=certificate['original_declaration']['native_samples'],
        linear_reference=True, codec_preroll_seconds=0, initial_decoded_frame_retained=True,
        PTS_quantization_half_tick_samples='24', temporary_member_or_PCM_written=False, PyAV_version=av.__version__)


class OriginalRows(core.OriginalRows):
    def __init__(self, validated, *, geometry_overlay, **arguments):
        self.overlay_policy = deepcopy(geometry_overlay)
        self.certificate = checked_certificate(geometry_overlay, arguments['guard'])
        super().__init__(validated, **arguments)

    def __call__(self, row, expected, phase):
        if row.get('ordinal') != ORDINAL and row.get('source_id') != ROW_ID:
            return super().__call__(row, expected, phase)
        import numpy as np
        raw, declaration = self.resolve(row, expected, phase)
        match_original(self.certificate, raw, declaration, expected)
        self.original_source_files = []; self.geometry_proof = None; self.replay_proof = None
        require(self.active_job is None, 'Prior raw buffers actually released')
        # Charge the unchanged larger historical envelope, never claim freed credit.
        p = raw_plan(native_samples=declaration['native_samples'], native_rate=declaration['native_sr'],
            clean_samples=declaration['clean16k_samples'], phase=phase, role=expected['role'],
            channels_bound=self.raw_contract['maximum_runtime_channels'],
            frame_samples_bound=self.raw_contract['maximum_decoder_frame_samples'])
        work = p['workspace_incremental_envelopes']; review = self.raw_contract
        require(work['decoder'] <= review['decoder_workspace_allowance_bytes'], 'Same decoder workspace')
        if self.frontend_enabled:
            require(work['weak_transform_FFT'] <= review['transform_workspace_allowance_bytes']
                    and work['original_frontend_spectral'] <= review['frontend_workspace_allowance_bytes'],
                    'Same original transform/frontend workspace')
        workspace = max(work.values()) if self.frontend_enabled else work['decoder']
        prospective = p['raw_peak_bytes'] + p['compressed_media_cap_bytes'] + workspace + review['numerical_runtime_reserve_bytes']
        observation = self.guard()
        require(observation['worker_private_memory']['private_bytes'] + prospective <= self.private_limit,
                'Unchanged current-private plus complete original raw/media/workspace before allocation')
        core.admit(prospective, self.guard)
        key = digest(expected['context']); self.ledger.begin(key, p); self.active_job = key
        if self.broad is None:
            self.broad = BoundedOriginalReader(ledger=self.ledger,
                source_inventory_binding=self.validated['bindings']['source_identity_inventory'],
                guard=self.guard, heldout_PCM_hashes=self.heldout)
        with self.ledger.stage(key, 'decode_resample'):
            wave, info = decode_full_member(self.broad, raw, self.certificate, guard=self.guard,
                channels_bound=p['channels_bound'], frame_samples_bound=p['frame_samples_bound'])
            self.ledger.observe(key, wave, name='whole_clean_float32', maximum_bytes=4*declaration['clean16k_samples'])
        with self.ledger.stage(key, 'canonical_hash'): f32, h = chunked_hashes(wave, guard=self.guard)
        require(h not in self.heldout, 'Exact heldout PCM overlap remains unresolved')
        c = self.certificate['actual_geometry']['clean16k_samples']
        require(wave.dtype == np.float32 and wave.ndim == 1 and wave.flags.c_contiguous
                and len(wave) == c and core.geometry(c)['encoder_frames'] == row['encoder_frames'],
                'Entire actual waveform and unchanged encoder T')
        actual = dict(info, unique_id=raw['unique_id'], source_id=raw['source_id'],
            actual_original_source_path=str(canonical_source_path(raw, core.ROOT)),
            decoded_sample_rate=16000, decoded_samples=c, encoder_frames=row['encoder_frames'],
            raw_label_sha256=self.certificate['raw_label_sha256'], waveform_sha256=f32, PCM_i16_sha256=h)
        item = self.broad.final_source_identity.entries[actual['actual_original_source_path']]
        self.original_source_files = [source_identity_binding(raw, item, actual, core.ROOT),
            item['index'], item['receipt'], self.overlay_policy['certificate']]
        self.geometry_proof = dict(kind='ASR18_exact_original_full_clip_geometry_reconciliation_v1',
            certificate=self.overlay_policy['certificate'], source=self.overlay_policy['source'],
            historical_declaration=deepcopy(declaration), actual_geometry=deepcopy(self.certificate['actual_geometry']),
            original_annotations_plan_labels_and_frame_geometry_unchanged=True,
            all_original_physical_samples_used_once=True, sampler_or_Adam_frontier_advanced=False)
        self.last = dict(actual, **core.geometry(c)); self.stable(); return wave
