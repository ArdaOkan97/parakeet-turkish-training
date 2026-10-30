"""Versioned Windows-safe atomic commits; no training or scoring changes.

Readers permit delete sharing and close before parsing. Writers use an
unpredictable same-directory temporary, flush it, and retry only Windows
sharing/access violations. Failed commits retain their temporary for diagnosis.
"""
from __future__ import annotations
import ctypes
import json
import os
import tempfile
import time
from pathlib import Path

RETRY_WINERRORS = frozenset((5, 32, 33))
DEFAULT_RETRY_SECONDS = 30.0


def read_bytes_shared(path):
    path = Path(path)
    if os.name != 'nt':
        return path.read_bytes()
    import msvcrt
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = (ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32,
                       ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p)
    create.restype = ctypes.c_void_p
    close = kernel.CloseHandle
    close.argtypes = (ctypes.c_void_p,)
    close.restype = ctypes.c_int
    # FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE.
    handle = create(str(path.resolve()), 0x80000000, 7, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except BaseException:
        close(handle)
        raise
    # fdopen owns the descriptor and therefore the underlying Win32 handle.
    with os.fdopen(descriptor, 'rb') as stream:
        return stream.read()


def read_json(path, default=None):
    try:
        payload = read_bytes_shared(path)
        return json.loads(payload.decode('utf-8-sig'))
    except (OSError, ValueError):
        return default


def retry_replace(source, destination, *, retry_seconds=DEFAULT_RETRY_SECONDS,
                  event=None):
    if not 0 < retry_seconds <= 60:
        raise ValueError('Atomic replace retries must be bounded to (0, 60] seconds')
    start = time.monotonic()
    delay = .01
    retries = 0
    while True:
        try:
            os.replace(source, destination)
            if event and retries:
                event({'path': str(destination), 'retries': retries,
                       'elapsed_seconds': time.monotonic() - start})
            return retries
        except OSError as exception:
            remaining = retry_seconds - (time.monotonic() - start)
            if getattr(exception, 'winerror', None) not in RETRY_WINERRORS or remaining <= 0:
                raise
            retries += 1
            time.sleep(min(delay, remaining))
            delay = min(.5, delay * 2)


def temporary_path(destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=destination.name + '.atomic-v1-',
                                        suffix='.tmp', dir=destination.parent)
    os.close(descriptor)
    return Path(name)


def write_json(path, value, *, retry_seconds=DEFAULT_RETRY_SECONDS, event=None):
    path = Path(path)
    temporary = temporary_path(path)
    try:
        with temporary.open('w', encoding='utf-8', newline='\n') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        retry_replace(temporary, path, retry_seconds=retry_seconds, event=event)
    except BaseException:
        # Never delete the existing destination or discard a failed commit.
        raise


def commit_writer(path, writer, *, retry_seconds=DEFAULT_RETRY_SECONDS, event=None):
    """Call an unchanged serializer at a private path, then commit atomically."""
    path = Path(path)
    temporary = temporary_path(path)
    try:
        writer(temporary)
    except OSError as exception:
        # The unchanged trainer serializer itself first commits path+'.tmp'.
        # Recover only that exact, already-serialized private-file replace.
        nested = Path(str(temporary) + '.tmp')
        if (getattr(exception, 'winerror', None) in RETRY_WINERRORS
                and getattr(exception, 'filename', None) == str(nested)
                and getattr(exception, 'filename2', None) == str(temporary)
                and nested.is_file()):
            retry_replace(nested, temporary, retry_seconds=retry_seconds, event=event)
        else:
            raise
    with temporary.open('r+b') as stream:
        os.fsync(stream.fileno())
    retry_replace(temporary, path, retry_seconds=retry_seconds, event=event)
