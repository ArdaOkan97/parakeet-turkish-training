"""Append-only declared-geometry metadata during an existing all-byte audit."""
import hashlib
import json
import os
from pathlib import Path

from ASR18_FLAC_streaminfo_geometry_v1 import streaminfo_geometry
from ASR18_FLAC_carrier_guard_v3 import ROOT


GEOMETRY_CACHE = ROOT / 'cache/KIRAAT_original_FLAC_declared_geometry_v1'


class GeometrySidecar:
    MAXIMUM_BYTES = 8_000_000

    def __init__(self, config, guard, receipt_path, receipt):
        self.guard = guard
        self.count = self.bytes = self.unresolved = 0
        self.maximum_clean16k_samples = self.maximum_encoder_frames = 0
        self.closed = False
        self.source_sha = config['job']['sha256']
        folder = GEOMETRY_CACHE / self.source_sha
        folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / (guard.run.parent.name+'__'+guard.run.name+'.jsonl')
        self.stream = self.path.open('xb', buffering=0)
        parser_path = Path(__file__).with_name('ASR18_FLAC_streaminfo_geometry_v1.py')
        self.parser = {**guard.binding(parser_path), 'mtime_ns': parser_path.stat().st_mtime_ns}
        self.writer = {**guard.binding(__file__), 'mtime_ns': Path(__file__).stat().st_mtime_ns}
        self.emit({'kind': 'ASR18_FLAC_STREAMINFO_declared_geometry_rows_v1',
            'source_SHA256': self.source_sha, 'carrier_receipt': guard.binding(receipt_path),
            'payload': receipt['records'], 'row_index': receipt['index'],
            'parser': self.parser, 'writer': self.writer,
            'geometry_basis': 'STREAMINFO_declaration_not_full_decoded_validation',
            'canonical_runtime_length_match_required_before_consumed_draw': True})

    def emit(self, value):
        data = (json.dumps(value, ensure_ascii=False, separators=(',', ':'))+'\n').encode('utf-8')
        assert self.bytes+len(data) <= self.MAXIMUM_BYTES, 'Preserve unresolved sidecar if its finite bound is exceeded'
        self.guard.check_storage(self.bytes+len(data))
        self.guard.pace(len(data))
        assert self.stream.write(data) == len(data)
        self.bytes += len(data)

    def add(self, index, row, actual):
        assert index == self.count and len(actual) == row[3]
        try:
            geometry = streaminfo_geometry(memoryview(actual)[:42])
            if not geometry['known_total_samples']:
                status = 'unresolved_unknown_total_samples'
            elif not geometry['native24k_mono_as_dataset_card']:
                status = 'unresolved_noncard_properties_require_canonical_reader_admission'
            else:
                status = 'parsed_declared_geometry_runtime_match_required'
        except ValueError as error:
            status = 'unresolved_invalid_STREAMINFO'
            geometry = {'reason': str(error),
                'STREAMINFO_prefix_SHA256': hashlib.sha256(memoryview(actual)[:42]).hexdigest()}
        if status.startswith('unresolved_'):
            self.unresolved += 1
        self.maximum_clean16k_samples = max(self.maximum_clean16k_samples,
            geometry.get('clean16k_samples_from_header') or 0)
        self.maximum_encoder_frames = max(self.maximum_encoder_frames,
            geometry.get('encoder_frames_from_header') or 0)
        self.emit({'row_index': index, 'clip_id': row[0], 'source_id': row[1],
            'original_compressed_record_SHA256': row[4], 'status': status, 'geometry': geometry})
        self.count += 1
        if self.count % 128 == 0:
            self.flush()

    def flush(self):
        if not self.closed:
            self.stream.flush()
            os.fsync(self.stream.fileno())

    def close_partial(self):
        if not self.closed:
            self.flush()
            self.stream.close()
            self.closed = True

    def finish(self, expected_rows):
        assert self.count == expected_rows
        self.close_partial()
        pin = {**self.guard.binding(self.path), 'mtime_ns': self.path.stat().st_mtime_ns}
        return {'kind': 'ASR18_all_physical_rows_declared_FLAC_geometry_v1',
            'source_SHA256': self.source_sha, 'sidecar': pin, 'physical_rows': self.count,
            'unresolved_declared_geometry_rows': self.unresolved,
            'maximum_clean16k_samples_from_header': self.maximum_clean16k_samples,
            'maximum_encoder_frames_from_header': self.maximum_encoder_frames,
            'parser': self.parser, 'writer': self.writer,
            'every_header_taken_during_original_row_byte_equality_comparison': True,
            'new_audio_read_or_decode_for_header_collection': False,
            'canonical_runtime_length_match_required_before_consumed_draw': True,
            'labels_or_source_files_modified': False, 'TRAIN_admitted': False}
