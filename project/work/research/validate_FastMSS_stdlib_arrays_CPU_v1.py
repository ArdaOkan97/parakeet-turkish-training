"""Explicitly synthetic struct/NPY controls only; no real arrays or cache use."""
import argparse
import ast
from copy import deepcopy
import hashlib
import importlib.abc
import json
import math
import os
from pathlib import Path
import struct
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'outputs/parakeet_turkish'))


class NoNumericalRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy', 'scipy', 'torch', 'transformers', 'pyarrow', 'librosa', 'soundfile', 'av'}:
            raise ImportError('Synthetic stdlib fixture forbids numerical/audio import: ' + fullname)


def pin(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def raw_npy(data, shape, dtype, *, version=(1, 0), header_text=None, fortran=False):
    """Independent tiny format writer: exact literal header, no NumPy arrays."""
    width = 2 if version[0] == 1 else 4
    body = repr({'descr': dtype, 'fortran_order': fortran, 'shape': shape}) if header_text is None else header_text
    length = len(body.encode('latin1'))
    padding = (- (8 + width + length + 1)) % 64
    header = body.encode('latin1') + b' ' * padding + b'\n'
    return b'\x93NUMPY' + bytes(version) + len(header).to_bytes(width, 'little') + header + data


def main(output):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    assert all(os.environ.get(key) == '1' for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'))
    output = Path(output).resolve()
    assert output.is_relative_to(Path(r'E:\gpu_training_scratchpad').resolve()) and not output.exists()
    assert not output.is_relative_to(Path(r'E:\gpu_training_scratchpad\temp\ASR18_CPU_feature_windows_v1').resolve())
    output.mkdir()
    sys.meta_path.insert(0, NoNumericalRuntime())
    import fastmss_stdlib_feature_validation_v1 as verifier
    import fastmss_phase_aware_feature_window_v4 as sink
    import fastmss_row_serial_cpu_v6 as core
    checks = []
    sequence = 0

    def check(condition, name):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    def case(*, frames=17, valid=16, words=None, mask_data=None, feature_options=None, mask_options=None):
        nonlocal sequence
        sequence += 1
        folder = output / ('synthetic_case_' + str(sequence))
        folder.mkdir()
        if words is None:
            words = [0x3f800000] * (valid * 128) + [0x80000000] * ((frames - valid) * 128)
        feature_data = b''.join(struct.pack('<I', word) for word in words)
        mask_data = bytes([1] * valid + [0] * (frames - valid)) if mask_data is None else mask_data
        feature_path, mask_path = folder / 'u00_r00_features.npy', folder / 'u00_r00_mask.npy'
        feature_path.write_bytes(raw_npy(feature_data, (1, frames, 128), '<f4', **(feature_options or {})))
        mask_path.write_bytes(raw_npy(mask_data, (1, frames), '|b1', **(mask_options or {})))
        entry = {'input_features': {'file': pin(feature_path), 'shape': [1, frames, 128], 'dtype': 'float32', 'array_sha256': hashlib.sha256(feature_data).hexdigest()},
                 'attention_mask': {'file': pin(mask_path), 'shape': [1, frames], 'dtype': 'bool', 'array_sha256': hashlib.sha256(mask_data).hexdigest()},
                 'encoder_frames': (valid + 7) // 8}
        return folder, entry

    def validate(folder, entry):
        # The actual outer v3 watchdog retains all real resource guards; this
        # callback is explicitly a synthetic protocol checkpoint, not telemetry.
        return verifier.validate_feature_arrays(entry, folder=folder,
            owned_names=['u00_r00_features.npy', 'u00_r00_mask.npy'], guard=lambda: None)

    def rejected(folder, entry, name):
        try:
            validate(folder, entry)
        except (ValueError, KeyError, TypeError, FileNotFoundError):
            checks.append(name)
            return
        raise AssertionError('Unexpectedly accepted: ' + name)

    folder, entry = case()
    snapshot = [pin(folder / name) for name in ('u00_r00_features.npy', 'u00_r00_mask.npy')]
    result = validate(folder, entry)
    check(result['passed'] and result['valid_mask_frames'] == 16 and result['physical_feature_frames'] == 17 and result['encoder_frames'] == 2,
          'Physical STFT17 vs valid16 preserved with original T2')
    check(snapshot == [pin(folder / name) for name in ('u00_r00_features.npy', 'u00_r00_mask.npy')], 'Validation leaves every synthetic byte unchanged')
    check(result['sampler_or_optimizer_committed'] is False and result['process_exit_or_window_completion_attested'] is False,
          'Array validation cannot attest native child0/window0 or commit a draw')
    check(result['NumPy_imported_here'] is False and 'numpy' not in sys.modules, 'Controller finalization path does not import NumPy')
    check(all(math.isfinite(value) for (value,) in struct.iter_unpack('<f', b''.join(struct.pack('<I', word) for word in [0, 0x80000000, 1, 0x7f7fffff]))),
          'Independent Python float reference includes both signed zeros and finite subnormal/extreme values')
    store = sink._Store(folder, {'owned_names': ['u00_r00_features.npy', 'u00_r00_mask.npy']}, lambda: None, None)
    check(store.validate_arrays(entry)['passed'], 'New validation-only _Store method reaches stdlib checker without NumPy')
    folder2, entry2 = case(feature_options={'version': (2, 0)}, mask_options={'version': (2, 0)})
    check(validate(folder2, entry2)['passed'], 'Both NPY2 headers supported')
    folder2, entry2 = case(frames=1, valid=1)
    check(validate(folder2, entry2)['passed'], 'All-valid mask with no padded feature frame is supported as before')
    folder2, entry2 = case(frames=257, valid=256)
    check(validate(folder2, entry2)['maximum_payload_read_bytes'] == 65536, 'Larger synthetic row streams across64KiB chunks without array allocation')
    folder2, entry2 = case(frames=101, valid=100)
    check(validate(folder2, entry2)['encoder_frames'] == 13, 'Actual original physical101/valid100/T13 shape contract represented synthetically')
    for word, name in ((0x7f800000, 'Positive infinity rejected'), (0xff800000, 'Negative infinity rejected'),
                       (0x7fc00001, 'NaN rejected'), (0x7f800001, 'Signalling NaN bit pattern rejected')):
        words = [0x3f800000] * (16 * 128) + [0] * 128
        words[0] = word
        f, e = case(words=words)
        rejected(f, e, name)
    for word, name in ((1, 'Nonzero subnormal padding rejected'), (0x3f800000, 'Nonzero ordinary padding rejected')):
        words = [0x3f800000] * (16 * 128) + [0] * 128
        words[-1] = word
        f, e = case(words=words)
        rejected(f, e, name)
    f, e = case(words=[0x80000000] * (17 * 128))
    check(validate(f, e)['passed'], 'Signed negative zero padded values match NumPy.any zero semantics')
    for mask, name in ((bytes([1, 0, 1] + [0] * 14), 'Mask true after padding rejected'),
                       (bytes([0] * 17), 'Empty valid mask rejected'),
                       (bytes([2] * 16 + [0]), 'Noncanonical bool byte2 rejected'),
                       (bytes([255] * 16 + [0]), 'Noncanonical bool byte255 rejected')):
        f, e = case(mask_data=mask)
        rejected(f, e, name)
    for options, name in (({'fortran': True}, 'Fortran-order feature header rejected'),
                          ({'header_text': "{'descr': '>f4', 'fortran_order': False, 'shape': (1,17,128)}"}, 'Wrong endian rejected'),
                          ({'header_text': "{'descr': '|O', 'fortran_order': False, 'shape': (1,17,128)}"}, 'Object dtype rejected'),
                          ({'header_text': "{'descr': '<f8', 'fortran_order': False, 'shape': (1,17,128)}"}, 'Float64 dtype rejected'),
                          ({'header_text': "{'descr': '<f4', 'fortran_order': False, 'shape': (True,17,128)}"}, 'Boolean header shape rejected'),
                          ({'header_text': "{'descr': '<f4', 'fortran_order': 0, 'shape': (1,17,128)}"}, 'Untyped fortran False rejected'),
                          ({'header_text': "{'descr': '<f4', 'descr': '<f4', 'fortran_order': False, 'shape': (1,17,128)}"}, 'Duplicate header key rejected'),
                          ({'header_text': "{'descr': '<f4', 'fortran_order': False, 'shape': (1,17,128), 'extra': 1}"}, 'Unknown header key rejected'),
                          ({'header_text': "{'descr': '<f4', 'fortran_order': False, 'shape': list((1,17,128))}"}, 'Executable header expression rejected'),
                          ({'header_text': "{'descr': '<f4', 'fortran_order': False, 'shape': [1,17,128]}"}, 'Non-tuple header shape rejected'),
                          ({'header_text': ' ' * 1025}, 'Oversized header rejected'),
                          ({'version': (3, 0)}, 'Unpinned NPY3 version rejected')):
        f, e = case(feature_options=options)
        rejected(f, e, name)
    for field, value, name in (('encoder_frames', 3, 'Incorrect encoder T rejected'), ('encoder_frames', True, 'Boolean encoder T rejected')):
        f, e = case(); e[field] = value
        rejected(f, e, name)
    f, e = case(); e['input_features']['shape'][1] = True
    rejected(f, e, 'Boolean declared shape rejected')
    f, e = case(); e['input_features']['dtype'] = 'float64'
    rejected(f, e, 'Wrong declared dtype rejected')
    f, e = case(); e['attention_mask']['array_sha256'] = '0' * 64
    rejected(f, e, 'Raw-array hash drift rejected')
    f, e = case(); e['input_features']['file']['sha256'] = '0' * 64
    rejected(f, e, 'Whole-file hash drift rejected')
    f, e = case(); e['input_features']['file']['bytes'] = True
    rejected(f, e, 'Boolean file length rejected')
    for change, name in ((lambda raw: raw + b'x', 'Trailing payload byte rejected'),
                         (lambda raw: raw[:-1], 'Truncated payload rejected'),
                         (lambda raw: b'x' + raw[1:], 'Bad magic rejected'),
                         (lambda raw: raw[:9], 'Truncated header length rejected')):
        f, e = case(); path = f / 'u00_r00_features.npy'; path.write_bytes(change(path.read_bytes())); e['input_features']['file'] = pin(path)
        rejected(f, e, name)
    f, e = case(); e['attention_mask']['file'] = deepcopy(entry['attention_mask']['file'])
    rejected(f, e, 'Different owned folder cannot substitute an array')
    def definitions(path):
        root = ast.parse(path.read_text(encoding='utf-8-sig'))
        return {node.name: node for node in root.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
    old = definitions(PROJECT / 'outputs/parakeet_turkish/fastmss_phase_aware_feature_window_v3.py')
    new = definitions(Path(sink.__file__))
    def method(defs, cls, name):
        return ast.dump(next(node for node in defs[cls].body if isinstance(node, ast.FunctionDef) and node.name == name), include_attributes=False)
    check(method(old, '_Store', 'arrays') == method(new, '_Store', 'arrays'), 'Native NumPy _Store.arrays AST is unchanged')
    check(method(old, '_Store', 'save') == method(new, '_Store', 'save'), 'Actual producer np.save bytes/format path is unchanged')
    check(method(old, '_ReadCursor', 'take_row') == method(new, '_ReadCursor', 'take_row'), 'Native consume-once take_row NumPy behavior is unchanged')
    check(ast.dump(old['_validate_update'], include_attributes=False).replace("attr='arrays'", "attr='validate_arrays'") == ast.dump(new['_validate_update'], include_attributes=False),
          'Closed update source/context/payload checks change only validation method')
    check(core.CONTRACT_KIND.endswith('_v6') and core.REQUEST_KIND.endswith('_v6'), 'New caller contract/request identities cannot alias frozen CPU5')
    check('numpy' not in sys.modules and not any(name.split('.')[0] in {'torch', 'scipy', 'transformers', 'pyarrow', 'soundfile', 'av'} for name in sys.modules),
          'Complete synthetic fixture/import/finalizer validation remains numerical/audio-runtime-free')
    report = {'kind': 'FastMSS_explicitly_synthetic_stdlib_NPY_controls_v1', 'passed': True,
              'check_count': len(checks), 'checks': checks, 'source': pin(__file__),
              'validator': pin(verifier.__file__), 'sink': pin(sink.__file__), 'common': pin(core.__file__),
              'maximum_read_chunk_bytes': verifier.CHUNK_BYTES, 'all_fixture_bytes_generated_by_struct': True,
              'actual_original_audio_or_features_read': False, 'shared_CACHE_created_or_changed': False,
              'actual_frontend_or_native_execution': False, 'actual_window_or_process_success_attested': False,
              'production_enabled': False, 'synthetic_cases': sequence}
    (output / 'report.json').write_text(json.dumps(report, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'passed': True, 'checks': len(checks), 'explicitly_synthetic': True}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True)
    main(parser.parse_args().output)
