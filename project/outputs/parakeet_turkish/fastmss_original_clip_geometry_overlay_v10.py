"""Ninth exact source-bound EOF certificate, with unchanged prior/ordinary delegation.

Historical declarations reserve the larger original envelope. This row uses all
physical samples and explicitly reports mel1314 versus declared1314, with T165/U66.
"""
from copy import deepcopy
import fastmss_original_clip_geometry_overlay_v9 as prior
from fastmss_original_clip_geometry_overlay_v9 import core, BoundedOriginalReader, raw_plan, chunked_hashes, canonical_source_path, source_identity_binding
ROW_ID='alania:a280321c852958fbb53e9d7a2044d1a6_00000'
ORDINAL=156134
require,digest,same,record=core.require,core.digest,core.same,core.record
match_original=prior.match_original
decode_full_member=prior.decode_full_member

def checked_certificate(policy, guard):
    require(policy['source'] == record(__file__, guard), 'Exact physical-geometry overlay source')
    certificate = core.read_bound(policy['certificate'], guard)
    review = core.read_bound(policy['ROOT_review'], guard)
    require(review['accepted'] is True and review['reviewer'] == '/root'
            and review['scope'] == 'ASR18_source_bound_ninth_original_full_clip_geometry_overlay_v1'
            and review['overlay_source'] == policy['source']
            and review['certificate'] == policy['certificate'], 'Exact reviewed one-row correction')
    require(certificate['kind'] == 'ASR18_source_bound_one_original_full_clip_geometry_certificate_v1'
            and certificate['original_row']['ordinal'] == ORDINAL
            and certificate['original_row']['unique_id'] == ROW_ID
            and certificate['original_row']['start'] == 0
            and certificate['original_row']['end'] == 13.143
            and digest(certificate['original_row']) == certificate['original_row_sha256']
            and digest(certificate['full_target_ids']) == certificate['full_target_ids_sha256']
            and len(certificate['full_target_ids']) == 66, 'Exact original row/annotation/full labels')
    old, actual = certificate['original_declaration'], certificate['actual_geometry']
    require(old == dict(ordinal=ORDINAL, source_component=0, native_sr=48000,
            native_samples=630864, clean16k_samples=210288, mel_frames=1314,
            encoder_frames=165, target_tokens=66, status='declared')
            and actual == dict(native_sr=48000, native_samples=630840,
            clean16k_samples=210280, mel_frames=1314, encoder_frames=165, target_tokens=66)
            and certificate['decoded_frames'] == 658
            and certificate['max_abs_PTS_residual_samples'] == '24', 'Exact measured EOF, unchanged T/U')
    observed = core.read_bound(certificate['independent_physical_ROOT_review'], guard)
    require(observed['accepted'] is True and observed['reviewer'] == '/root'
            and observed['kind'] == 'ASR18_recorded186564_row5_full_original_member_physical_actual01_independent_ROOT_verification_v1',
            'Exact independently accepted ROOT physical receipt, no closed raw observation re-audit')
    physical = observed['physical']; identity = observed['exact_original_row_identity']
    source = observed['source_artifact']; entry = observed['recording_index_entry']
    require(physical['complete_demux_decode_EOF'] is True
            and physical['all_original_physical_samples_used_once'] is True
            and physical['synthesized_or_skipped_samples'] == 0
            and identity['raw_original_annotation'] == certificate['original_row']
            and identity['full_target_ids'] == certificate['full_target_ids']
            and {k:observed['independent_arithmetic']['derived_geometry'][k] for k in actual} == actual
            and physical['native_f32_sha256'] == certificate['native_f32_sha256']
            and physical['clean_f32_sha256'] == certificate['clean_f32_sha256']
            and physical['clean_i16_sha256'] == certificate['clean_i16_sha256']
            and physical['compressed_member_sha256'] == certificate['member_sha256']
            and dict(zip(('member_name', 'offset', 'bytes'), entry)) == certificate['recording_index_entry']
            and dict(path=source['path'], bytes=source['bytes'], mtime_ns=source['mtime_ns'],
                     recorded_sha256=source['sha256']) == certificate['original_TAR']
            and observed['resource']['original_prospective'] == 901306112,
            'Exact ROOT physical EOF/full labels/source/member/hashes and larger original reservations')
    return certificate


class OriginalRows(prior.OriginalRows):
    def __init__(self,validated,*,geometry_overlay,**arguments):
        self.ninth_overlay_policy=deepcopy(geometry_overlay)
        self.ninth_certificate=checked_certificate(geometry_overlay,arguments['guard'])
        require(geometry_overlay['prior_overlay']['source']==record(prior.__file__,arguments['guard']),
                'Exact original overlay remains the authority for its eight prior certified rows')
        super().__init__(validated,geometry_overlay=geometry_overlay['prior_overlay'],**arguments)

    def __call__(self, row, expected, phase):
        if row.get('ordinal') != ORDINAL and row.get('source_id') != ROW_ID:
            return super().__call__(row, expected, phase)
        import numpy as np
        raw, declaration = self.resolve(row, expected, phase)
        match_original(self.ninth_certificate, raw, declaration, expected)
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
            wave, info = decode_full_member(self.broad, raw, self.ninth_certificate, guard=self.guard,
                channels_bound=p['channels_bound'], frame_samples_bound=p['frame_samples_bound'])
            self.ledger.observe(key, wave, name='whole_clean_float32', maximum_bytes=4*declaration['clean16k_samples'])
        with self.ledger.stage(key, 'canonical_hash'): f32, h = chunked_hashes(wave, guard=self.guard)
        require(h not in self.heldout, 'Exact heldout PCM overlap remains unresolved')
        c = self.ninth_certificate['actual_geometry']['clean16k_samples']
        require(wave.dtype == np.float32 and wave.ndim == 1 and wave.flags.c_contiguous
                and len(wave) == c and core.geometry(c)['encoder_frames'] == row['encoder_frames'],
                'Entire actual waveform and unchanged encoder T')
        actual = dict(info, unique_id=raw['unique_id'], source_id=raw['source_id'],
            actual_original_source_path=str(canonical_source_path(raw, core.ROOT)),
            decoded_sample_rate=16000, decoded_samples=c, encoder_frames=row['encoder_frames'],
            raw_label_sha256=self.ninth_certificate['raw_label_sha256'], waveform_sha256=f32, PCM_i16_sha256=h)
        item = self.broad.final_source_identity.entries[actual['actual_original_source_path']]
        self.original_source_files = [source_identity_binding(raw, item, actual, core.ROOT),
            item['index'], item['receipt'], self.ninth_overlay_policy['certificate']]
        self.geometry_proof = dict(kind='ASR18_exact_original_full_clip_geometry_reconciliation_v1',
            certificate=self.ninth_overlay_policy['certificate'], source=self.ninth_overlay_policy['source'],
            historical_declaration=deepcopy(declaration), actual_geometry=deepcopy(self.ninth_certificate['actual_geometry']),
            original_annotations_plan_labels_and_encoder_geometry_unchanged=True,
            historical_mel_frames=declaration['mel_frames'], actual_physical_mel_frames=self.ninth_certificate['actual_geometry']['mel_frames'],
            all_original_physical_samples_used_once=True, sampler_or_Adam_frontier_advanced=False)
        self.last = dict(actual, **core.geometry(c)); self.stable(); return wave

