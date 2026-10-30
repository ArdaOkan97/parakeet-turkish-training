"""Independent native Windows atomic-I/O tests; CPU only, new scratch files."""
import ctypes
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / 'outputs/parakeet_turkish'
sys.path.insert(0, str(OUTPUT))
import atomic_io_retry_v1 as atomic


def main(output_path=None):
    assert os.name == 'nt', 'Native Windows tests are required.'
    directory = Path(tempfile.mkdtemp(prefix='recovery_io_independent_', dir=ROOT / 'work'))
    checks = []
    def verify(condition, label):
        assert condition, label
        checks.append(label)

    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateFileW.argtypes = (ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32,
                                  ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p)
    kernel.CreateFileW.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
    kernel.CloseHandle.restype = ctypes.c_int
    def handle(path, sharing):
        result = kernel.CreateFileW(str(path), 0x80000000, sharing, None, 3, 0x80, None)
        if result == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        return result

    target = directory / 'state.json'
    atomic.write_json(target, {'step': 7, 'text': '\u0131\u015f\u0131k'})
    verify(atomic.read_json(target) == {'step': 7, 'text': '\u0131\u015f\u0131k'}, 'native delete-sharing read and Unicode JSON roundtrip')
    opened = handle(target, 7)
    shared_releaser = threading.Thread(target=lambda: (time.sleep(.08), kernel.CloseHandle(opened)))
    shared_releaser.start()
    try:
        atomic.write_json(target, {'step': 8}, retry_seconds=1)
        verify(atomic.read_json(target) == {'step': 8}, 'replacement completes after an actual native delete-sharing read closes; no live-handle replacement guarantee')
    finally:
        shared_releaser.join()

    events = []
    opened = handle(target, 3)
    releaser = threading.Thread(target=lambda: (time.sleep(.08), kernel.CloseHandle(opened)))
    releaser.start()
    atomic.write_json(target, {'step': 9}, retry_seconds=1, event=events.append)
    releaser.join()
    verify(atomic.read_json(target) == {'step': 9}, 'replacement eventually commits after an actual Windows non-delete-sharing reader closes')
    verify(len(events) == 1 and events[0]['retries'] > 0, 'actual sharing violation retries are recorded')
    verify(events[0]['elapsed_seconds'] < 1.1, 'actual retry is bounded')

    opened = handle(target, 3)
    old_bytes = target.read_bytes()
    before = set(directory.glob('state.json.atomic-v1-*.tmp'))
    start = time.monotonic()
    try:
        try:
            atomic.write_json(target, {'step': 10}, retry_seconds=.03)
            raise AssertionError('Expected native sharing timeout')
        except OSError as exception:
            verify(exception.winerror in (5, 32, 33), 'bounded native timeout preserves actual Windows sharing/access error')
    finally:
        kernel.CloseHandle(opened)
    verify(time.monotonic() - start < .3, 'failed native retry respects its short bounded budget')
    verify(target.read_bytes() == old_bytes, 'failed commit leaves original destination byte-identical')
    failed = set(directory.glob('state.json.atomic-v1-*.tmp')) - before
    verify(len(failed) == 1, 'failed commit retains exactly its private diagnostic temporary')
    verify(json.loads(next(iter(failed)).read_text(encoding='utf-8')) == {'step': 10}, 'failed temporary retains complete intended JSON')

    checkpoint = directory / 'checkpoint.pt'
    payload = b'original serializer exact state\x00\xff'
    def original_serializer(private):
        inner = Path(str(private) + '.tmp')
        inner.write_bytes(payload)
        inner.replace(private)
    atomic.commit_writer(checkpoint, original_serializer)
    verify(checkpoint.read_bytes() == payload, 'native post-serializer fsync and atomic checkpoint commit preserve exact bytes')
    writer_calls = []
    nested_releasers = []
    def transient_nested_serializer(private):
        writer_calls.append(private)
        nested = Path(str(private) + '.tmp')
        nested.write_bytes(payload + b' nested commit')
        opened = handle(private, 3)
        releaser = threading.Thread(target=lambda: (time.sleep(.08), kernel.CloseHandle(opened)))
        nested_releasers.append(releaser)
        releaser.start()
        nested.replace(private)
    atomic.commit_writer(checkpoint, transient_nested_serializer, retry_seconds=1)
    for releaser in nested_releasers:
        releaser.join()
    verify(checkpoint.read_bytes() == payload + b' nested commit', 'native private nested serializer commit is recovered without discarding serialized bytes')
    verify(len(writer_calls) == 1, 'private nested commit recovery never calls the serializer twice')
    names = [atomic.temporary_path(checkpoint) for _ in range(128)]
    verify(len(set(names)) == 128, '128 temporary names are unique')
    verify(all(path.parent == checkpoint.parent for path in names), 'all temporaries stay on the destination directory/filesystem')

    original_replace = atomic.os.replace
    calls = []
    def fatal_error(source, destination):
        calls.append((source, destination))
        raise FileNotFoundError(2, 'missing source')
    atomic.os.replace = fatal_error
    try:
        try:
            atomic.retry_replace(directory / 'absent', checkpoint, retry_seconds=.1)
            raise AssertionError('Expected missing-source failure')
        except FileNotFoundError:
            verify(len(calls) == 1, 'nonsharing errors are never retried')
    finally:
        atomic.os.replace = original_replace
    for bad in [0, -1, 61]:
        try:
            atomic.retry_replace(directory / 'absent', checkpoint, retry_seconds=bad)
            raise AssertionError('Expected invalid retry budget')
        except ValueError:
            verify(True, 'invalid unbounded/nonpositive retry budget rejected: ' + str(bad))
    verify(atomic.read_json(directory / 'missing', {'fallback': True}) == {'fallback': True}, 'missing JSON preserves caller default semantics')
    result = {'schema': 1, 'passed': True, 'cpu_only': True, 'native_windows': True,
              'checks': checks, 'passed_check_count': len(checks), 'scratch_directory': str(directory),
              'helper_sha256': hashlib.sha256((OUTPUT / 'atomic_io_retry_v1.py').read_bytes()).hexdigest(),
              'test_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'Actual Windows file handles, bounded retries, fsync, atomic commit, private temps; no GPU or training calls.'}
    path = output_path or OUTPUT / 'recovery_io_independent_verification.json'
    assert path.resolve().is_relative_to(OUTPUT.resolve()), 'Report must stay inside experiment outputs.'
    assert not path.exists(), 'Preserve a previous independent report.'
    path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'passed': True, 'checks': len(checks), 'helper_sha256': result['helper_sha256'], 'report': str(path)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    main(parser.parse_args().output)
