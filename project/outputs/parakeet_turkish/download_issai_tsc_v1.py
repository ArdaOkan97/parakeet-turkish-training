"""Resumable single-stream pinned TSC acquisition; no extraction/training."""
from __future__ import annotations
import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time
import urllib.parse
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[2]
URL = 'https://huggingface.co/datasets/issai/Turkish_Speech_Corpus/resolve/d79ea01392882322e8f46657cb635236af25b6d4/ISSAI_TSC_218.tar.gz'
REVISION = 'd79ea01392882322e8f46657cb635236af25b6d4'
TOTAL = 21366175820
EXPECTED_SHA256 = '3feaba1f911542b587622d49c968bef971cfd8a4b3908d73090b8b914d2b694f'
BLOCK = 1024 * 1024
RATE = 8 * BLOCK
REHASH_RATE = 64 * BLOCK
ATTEMPTS = 8
PREFIX_BYTES = 4 * BLOCK
PREFIX_SHA256 = 'c2200040fb5e42d0de13ff5a796902101bb5d0e40ac14c72579d6b2f95a454af'


def require(ok, message):
    if not ok: raise ValueError(message)
def now(): return datetime.now(timezone.utc).isoformat()
def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8')) if Path(path).exists() else {}
def file_sha(path):
    with Path(path).open('rb') as source: return hashlib.file_digest(source, 'sha256').hexdigest()
def write_progress(path, value):
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('x', encoding='utf-8') as target: json.dump(value, target, indent=2)
    for attempt in range(5):
        try: os.replace(temporary, path); return
        except OSError as error:
            if getattr(error, 'winerror', None) not in (5, 32, 33) or attempt == 4: raise
            time.sleep(.1 * (attempt + 1))


def public_url(value):
    parsed = urllib.parse.urlsplit(value); host = (parsed.hostname or '').lower()
    require(parsed.scheme == 'https' and not parsed.username and not parsed.password
        and parsed.port in (None, 443)
        and any(host == item or host.endswith('.' + item) for item in ('huggingface.co', 'hf.co')),
        'Only HTTPS public HF asset hosts without user credentials allowed')
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, '', ''))


class Redirects(urllib.request.HTTPRedirectHandler):
    max_redirections = 3
    max_repeats = 2
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        public_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def validate_response(response, offset, total):
    require(response.headers.get('Content-Encoding', 'identity').lower() == 'identity', 'Unexpected HTTP content encoding')
    if response.status == 206:
        require(response.headers.get('Content-Range') == f'bytes {offset}-{total - 1}/{total}',
            'Content-Range must exactly match requested offset/end/pinned total')
    else: require(offset == 0 and response.status == 200, 'Resumed request requires HTTP206')
    require(response.headers.get('Content-Length') == str(total - offset), 'Response size differs from exact remaining bytes')
    require(callable(getattr(response, 'read1', None)), 'Available-chunk bounded HTTP reader required')
    return public_url(response.geturl())


def priority():
    if os.name != 'nt': return {'requested': 'below_normal', 'applied': False, 'scope': 'Windows own process only'}
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    kernel.SetPriorityClass.restype = ctypes.c_int
    applied = bool(kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x4000))
    return {'requested': 'below_normal', 'applied': applied, 'winerror': None if applied else ctypes.get_last_error(), 'pid': os.getpid()}


class Throttle:
    def __init__(self, rate=RATE): self.next_time = time.monotonic(); self.rate = rate
    def add(self, count):
        self.next_time = max(time.monotonic(), self.next_time) + count / self.rate
        delay = self.next_time - time.monotonic()
        if delay > 0: time.sleep(delay)


def verify_existing_part(part, trusted_prefix, previous, emit, throttle):
    """Restore SHA with bounded RAM; verify saved partial hash + actual probe."""
    size = part.stat().st_size if part.exists() else 0
    require(size <= TOTAL, 'Part exceeds pinned total')
    digest = hashlib.sha256(); prefix_digest = hashlib.sha256()
    checked = 0
    committed = previous.get('hashed_prefix_bytes', 0)
    require(isinstance(committed, int) and 0 <= committed <= size, 'Recorded partial hash exceeds actual part')
    require(not size or previous, 'Existing partial file requires this downloader\'s progress identity')
    expected_partial = previous.get('partial_prefix_sha256', hashlib.sha256().hexdigest())
    if committed == 0: require(expected_partial == digest.hexdigest(), 'Invalid empty partial hash')
    with trusted_prefix.open('rb') as trusted:
        if part.exists():
            with part.open('rb') as source:
                while checked < size:
                    limit = min(BLOCK, size - checked)
                    if checked < committed: limit = min(limit, committed - checked)
                    if checked < PREFIX_BYTES: limit = min(limit, PREFIX_BYTES - checked)
                    block = source.read(limit); require(block, 'Part changed/truncated during hashing')
                    require(block == trusted.read(len(block)) if checked < PREFIX_BYTES else True,
                        'Existing part does not match the verified archive prefix')
                    # Split at4MiB so the trusted comparison never crosses its EOF.
                    digest.update(block); prefix_digest.update(block[:max(0, min(len(block), PREFIX_BYTES - checked))])
                    checked += len(block); throttle.add(len(block))
                    if checked == committed: require(digest.hexdigest() == expected_partial, 'Saved partial SHA mismatch; never append')
                    emit('hashing_existing_part', size, committed, expected_partial, hashed_bytes_so_far=checked)
    require(not size or part.stat().st_size == size, 'Part size changed during validation')
    if size >= PREFIX_BYTES: require(prefix_digest.hexdigest() == PREFIX_SHA256, 'Verified prefix SHA differs')
    return digest, size


def copy_response(response, target, digest, offset, total, emit, throttle, prefix_hasher=None):
    """No read exceeds remaining bytes; partial transport is retried by caller."""
    position = offset
    while position < total:
        block = response.read1(min(BLOCK, total - position))
        require(block and len(block) <= min(BLOCK, total - position), 'Truncated or oversized response chunk')
        written = target.write(block); require(written == len(block), 'Partial disk write; preserve file for inspection')
        digest.update(block)
        old_position = position; position += len(block)
        emit('downloading', position, position, digest.hexdigest())
        if prefix_hasher is not None and old_position < PREFIX_BYTES:
            prefix_hasher.update(block[:min(len(block), PREFIX_BYTES - old_position)])
            if position >= PREFIX_BYTES:
                require(prefix_hasher.hexdigest() == PREFIX_SHA256, 'Downloaded source prefix differs; stop immediately')
        throttle.add(len(block))
    return position


def finalize_verified(part, final, digest, total=TOTAL, expected=EXPECTED_SHA256):
    require(part.stat().st_size == total and digest.hexdigest() == expected, 'Complete archive size or SHA mismatch; retain part')
    require(not final.exists(), 'Never overwrite an existing archive')
    os.rename(part, final) # Same directory; Windows fails if destination exists.


def main(args):
    require(args.prepare_only != args.network_download_authorized, 'Choose offline preflight or authorized public acquisition')
    folder = args.output_dir.resolve()
    require(folder.is_relative_to(ROOT / 'work/data'), 'Acquisition must stay in workspace work/data')
    prefix_report_path = args.prefix_report.resolve()
    require(prefix_report_path.is_relative_to(ROOT), 'Prefix evidence must be local workspace data')
    prefix_report = read(prefix_report_path)
    require(prefix_report.get('passed') is True and prefix_report.get('actual_network_executed') is True
        and prefix_report.get('source_url') == URL and prefix_report.get('source_revision') == REVISION
        and prefix_report.get('publisher_complete_archive_sha256') == EXPECTED_SHA256
        and prefix_report['prefix']['bytes'] == PREFIX_BYTES and prefix_report['prefix']['sha256'] == PREFIX_SHA256,
        'Actual fixed4MiB public archive prefix evidence required')
    trusted_prefix = Path(prefix_report['prefix']['path']).resolve()
    require(trusted_prefix.is_relative_to(ROOT) and trusted_prefix.stat().st_size == PREFIX_BYTES
        and file_sha(trusted_prefix) == PREFIX_SHA256, 'Trusted prefix file changed')
    part, final = folder / 'ISSAI_TSC_218.tar.gz.part', folder / 'ISSAI_TSC_218.tar.gz'
    identity = {'schema': 1, 'kind': 'pinned_ISSAI_TSC_single_stream_acquisition_v1', 'url': URL,
        'revision': REVISION, 'total_bytes': TOTAL, 'full_archive_sha256': EXPECTED_SHA256,
        'prefix_sha256': PREFIX_SHA256, 'prefix_report_sha256': file_sha(prefix_report_path),
        'downloader_sha256': file_sha(__file__), 'block_bytes': BLOCK, 'network_limit_bytes_per_second': RATE,
        'existing_part_rehash_limit_bytes_per_second': REHASH_RATE,
        'request_attempts_max': ATTEMPTS, 'socket_timeout_seconds': 30, 'progress_interval_seconds': 5,
        'archive_extraction': False, 'training_eligibility_claimed': False}
    if folder.exists():
        require(read(folder / 'download_identity.json') == identity and not final.exists(),
            'Only identical unfinished acquisition may resume; never overwrite a final archive')
        require(not part.is_symlink(), 'Symlink part rejected')
    if args.prepare_only:
        print(json.dumps({'passed': True, 'status': 'offline_prepared', 'identity': identity,
            'output_dir': str(folder), 'actual_network_executed': False})); return 0
    folder.mkdir(parents=True, exist_ok=True)
    identity_path = folder / 'download_identity.json'
    if not identity_path.exists():
        with identity_path.open('x', encoding='utf-8') as stream: json.dump(identity, stream, indent=2)
    progress_path = folder / 'download_progress.json'
    previous = read(progress_path)
    state = {**previous, 'schema': 1, 'kind': 'ISSAI_TSC_download_progress_v1', 'source_identity': identity,
        'status': 'starting', 'total_bytes': TOTAL, 'actual_network_executed': False, 'priority': priority(),
        'started_at_utc': now(), 'complete_archive_sha256_verified': False}
    started = time.monotonic(); last_emit = 0.0; start_bytes = part.stat().st_size if part.exists() else 0
    def emit(status, downloaded, hashed, partial_sha, force=False, **extra):
        nonlocal last_emit
        moment = time.monotonic()
        if not force and moment - last_emit < 5: return
        state.update(status=status, downloaded_bytes=downloaded, hashed_prefix_bytes=hashed,
            partial_prefix_sha256=partial_sha, updated_at_utc=now(),
            average_new_download_bytes_per_second=max(0, downloaded - start_bytes) / max(.001, moment - started), **extra)
        write_progress(progress_path, state); last_emit = moment
        if force: print(json.dumps({key: state[key] for key in ('status', 'downloaded_bytes', 'total_bytes')}), flush=True)
    digest = None; position = start_bytes
    try:
        digest, position = verify_existing_part(part, trusted_prefix, previous, emit, Throttle(REHASH_RATE))
        emit('ready_to_download', position, position, digest.hexdigest(), force=True)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), Redirects())
        # Preserve a partial-prefix hasher for an acquisition resumed before4MiB.
        prefix_hasher = hashlib.sha256()
        if position < PREFIX_BYTES and part.exists():
            with part.open('rb') as source:
                while block := source.read(BLOCK): prefix_hasher.update(block)
        throttle = Throttle()
        for attempt in range(ATTEMPTS):
            if position == TOTAL: break
            state['request_attempt'] = attempt + 1
            require(part.stat().st_size == position if part.exists() else position == 0, 'Part changed before resume request')
            request = urllib.request.Request(URL, headers={'Range': f'bytes={position}-{TOTAL - 1}',
                'Accept-Encoding': 'identity', 'User-Agent': 'pinned-single-stream-public-TSC-acquisition/1.0'})
            state['actual_network_executed'] = True
            try:
                with opener.open(request, timeout=30) as response:
                    state['final_public_url_without_query'] = validate_response(response, position, TOTAL)
                    with part.open('ab', buffering=0) as target:
                        def step_emit(status, count, hashed, partial_sha):
                            nonlocal position
                            position = count; emit(status, count, hashed, partial_sha)
                        position = copy_response(response, target, digest, position, TOTAL, step_emit, throttle,
                            prefix_hasher if position < PREFIX_BYTES else None)
                        target.flush(); os.fsync(target.fileno())
                break
            except Exception as error:
                state['last_error_type'] = type(error).__name__ # Never print signed URLs/exceptions containing them.
                require(not part.exists() or part.stat().st_size == position, 'Partial write/state mismatch: stop; never append blindly')
                if isinstance(error, ValueError): raise # Source/range/prefix policy failures are not transient.
                emit('retry_wait' if attempt + 1 < ATTEMPTS else 'failed', position, position, digest.hexdigest(), force=True)
                if attempt + 1 == ATTEMPTS: raise
                time.sleep(min(30, 2 ** attempt))
        require(position == TOTAL, 'Attempts ended before pinned total')
        emit('verifying_complete_stream_sha256', position, position, digest.hexdigest(), force=True)
        finalize_verified(part, final, digest)
        state.update(complete_archive_sha256_verified=True, archive_path=str(final), full_archive_sha256=digest.hexdigest())
        emit('complete', position, position, digest.hexdigest(), force=True)
        return 0
    except BaseException as error:
        state['last_error_type'] = type(error).__name__
        if digest is not None:
            emit('interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                part.stat().st_size if part.exists() else position, position, digest.hexdigest(), force=True)
        else:
            state.update(status='failed_before_append', updated_at_utc=now()); write_progress(progress_path, state)
        if isinstance(error, KeyboardInterrupt): raise
        raise RuntimeError('Pinned public acquisition failed: ' + type(error).__name__) from None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--prefix-report', type=Path, default=ROOT / 'work/research/ISSAI_archive_prefix_actual_01/probe_report.json')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--network-download-authorized', action='store_true')
    raise SystemExit(main(parser.parse_args()))
