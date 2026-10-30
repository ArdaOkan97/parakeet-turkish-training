"""INERT bounded public archive-prefix probe; never extract or decode audio."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib

ROOT = Path(__file__).resolve().parents[2]
URL = 'https://huggingface.co/datasets/issai/Turkish_Speech_Corpus/resolve/d79ea01392882322e8f46657cb635236af25b6d4/ISSAI_TSC_218.tar.gz'
REVISION = 'd79ea01392882322e8f46657cb635236af25b6d4'
PUBLISHER_ARCHIVE_SHA256 = '3feaba1f911542b587622d49c968bef971cfd8a4b3908d73090b8b914d2b694f'
COMPRESSED_LIMIT = 4 * 1024 * 1024
DECOMPRESSED_LIMIT = 16 * 1024 * 1024
POLICY = {'compressed_bytes_total_max': COMPRESSED_LIMIT, 'decompressed_bytes_max': DECOMPRESSED_LIMIT,
    'tar_members_max': 256, 'document_bytes_each_max': 8192, 'document_bytes_total_max': 32768,
    'request_attempts_max': 2, 'redirects_max': 3, 'request_read_timeout_seconds': 10,
    'body_loop_deadline_seconds': 30, 'deadline_checked_between_blocking_operations': True,
    'hard_wall_deadline_guaranteed': False,
    'timeout_scope': 'Socket operations; OS DNS resolution is not a guaranteed Python wall-time bound',
    'retry_after_any_body_bytes': False, 'archive_extraction': False, 'audio_decode': False,
    'authentication_headers_or_cookies': False, 'environment_proxy_credentials': False}


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as source: return hashlib.file_digest(source, 'sha256').hexdigest()


def public_url(value):
    parsed = urllib.parse.urlsplit(value)
    host = (parsed.hostname or '').lower()
    require(parsed.scheme == 'https' and not parsed.username and not parsed.password
        and parsed.port in (None, 443)
        and any(host == domain or host.endswith('.' + domain) for domain in ('huggingface.co', 'hf.co')),
        'Only public HTTPS Hugging Face asset redirects without user credentials are allowed')
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, '', ''))


class PublicRedirects(urllib.request.HTTPRedirectHandler):
    max_redirections = 3
    max_repeats = 2
    def __init__(self): self.hosts = []
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        public_url(newurl)
        require(len(self.hosts) < POLICY['redirects_max'], 'Redirect limit reached')
        self.hosts.append(urllib.parse.urlsplit(newurl).hostname)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_prefix(destination, *, opener=None, byte_limit=COMPRESSED_LIMIT):
    """Body budget is shared across attempts, including ignored HTTP Range."""
    require(0 < byte_limit <= COMPRESSED_LIMIT, 'Compressed limit exceeds4MiB')
    redirects = PublicRedirects()
    if opener is None:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), redirects)
    result = {'compressed_bytes_read': 0, 'attempts': 0, 'status': 'open_failed',
        'body_read_limit': byte_limit, 'body_read_ended_at_limit': False, 'response_eof_observed': False}
    deadline = time.monotonic() + POLICY['body_loop_deadline_seconds']
    with Path(destination).open('xb') as target:
        for attempt in range(POLICY['request_attempts_max']):
            result['attempts'] = attempt + 1
            request = urllib.request.Request(URL, headers={'Range': f'bytes=0-{byte_limit - 1}',
                'Accept-Encoding': 'identity', 'User-Agent': 'bounded-public-archive-prefix/1.0'})
            try:
                require(time.monotonic() < deadline, 'Read deadline reached')
                with opener.open(request, timeout=POLICY['request_read_timeout_seconds']) as response:
                    require(response.status in (200, 206), 'Expected HTTP200 or206')
                    require(response.headers.get('Content-Encoding', 'identity').lower() == 'identity',
                        'Unexpected HTTP content encoding rejected')
                    require(response.status != 206 or re.fullmatch(r'bytes 0-\d+/\d+', response.headers.get('Content-Range', '')),
                        'Partial response must start at byte0')
                    result.update(http_status=response.status, final_public_url_without_query=public_url(response.geturl()),
                        requested_range_honored=response.status == 206,
                        response_headers={key: response.headers.get(key) for key in
                            ('Content-Length', 'Content-Range', 'Content-Type', 'Accept-Ranges')})
                    require(callable(getattr(response, 'read1', None)), 'Bounded available-chunk HTTP reader required')
                    while result['compressed_bytes_read'] < byte_limit:
                        if time.monotonic() >= deadline:
                            result['status'] = 'read_deadline_reached'; break
                        count = min(65536, byte_limit - result['compressed_bytes_read'])
                        try: block = response.read1(count)
                        except Exception as error:
                            # HTTPResponse may expose partial bytes on a truncated body.
                            partial = getattr(error, 'partial', b'')
                            require(isinstance(partial, bytes) and len(partial) <= count, 'Unbounded partial response rejected')
                            result['compressed_bytes_read'] += len(partial); target.write(partial)
                            result.update(status='partial_read_error', error_type=type(error).__name__); break
                        require(len(block) <= count, 'Transport violated requested read bound')
                        if not block:
                            result.update(status='response_eof', response_eof_observed=True); break
                        result['compressed_bytes_read'] += len(block); target.write(block)
                    else:
                        result.update(status='prefix_limit_reached', body_read_ended_at_limit=True)
                break
            except Exception as error:
                result.update(status='open_or_protocol_failed', error_type=type(error).__name__)
                if result['compressed_bytes_read'] or time.monotonic() >= deadline: break
    result['redirect_hosts'] = redirects.hosts
    result['prefix_sha256'] = sha(destination)
    result['complete_archive_sha256_verified'] = False
    return result


class BoundedGzipReader:
    """Never inflate beyond the limit, including read-ahead and gzip bombs."""
    def __init__(self, source, limit=DECOMPRESSED_LIMIT):
        require(0 < limit <= DECOMPRESSED_LIMIT, 'Decompression bound exceeds16MiB')
        self.source, self.limit = source, limit
        self.decoder = zlib.decompressobj(31)
        self.pending, self.buffer = b'', b''
        self.generated = self.delivered = self.compressed_consumed = 0
        self.input_exhausted = self.limit_reached = False
    def read(self, count):
        require(count >= 0, 'Unbounded reads are forbidden')
        result = bytearray()
        while len(result) < count:
            if self.buffer:
                take = min(count - len(result), len(self.buffer))
                result.extend(self.buffer[:take]); self.buffer = self.buffer[take:]
                continue
            if self.generated >= self.limit:
                self.limit_reached = True; break
            if self.decoder.eof or self.input_exhausted: break
            if not self.pending:
                self.pending = self.source.read(65536)
                self.compressed_consumed += len(self.pending)
                require(self.compressed_consumed <= COMPRESSED_LIMIT, 'Compressed input exceeds4MiB')
                if not self.pending:
                    self.input_exhausted = True; break
            self.buffer = self.decoder.decompress(self.pending, min(65536, self.limit - self.generated))
            self.pending = self.decoder.unconsumed_tail
            self.generated += len(self.buffer)
        self.delivered += len(result)
        return bytes(result)
    def stats(self):
        return {'decompressed_bytes_generated': self.generated, 'decompressed_bytes_delivered': self.delivered,
            'decompression_limit_reached': self.limit_reached or self.generated == self.limit,
            'gzip_stream_complete': self.decoder.eof,
            'gzip_prefix_truncated': self.input_exhausted and not self.decoder.eof,
            'compressed_input_bytes_consumed': self.compressed_consumed}


def safe_member(name):
    clean = name.replace('\\', '/')
    path = PurePosixPath(clean)
    return bool(name) and not path.is_absolute() and '..' not in path.parts and ':' not in clean


def tar_number(field):
    # Support POSIX octal and positive GNU base256 sizes, without extraction.
    if field[0] & 128:
        require(not field[0] & 64, 'Negative GNU base256 field rejected')
        return int.from_bytes(bytes([field[0] & 127]) + field[1:], 'big')
    return int(field.rstrip(b'\0 ').lstrip(b' ') or b'0', 8)


def inspect_prefix(path, *, decompressed_limit=DECOMPRESSED_LIMIT):
    result = {'members': [], 'documents': [], 'archive_end_marker_seen': False,
        'partial_archive': True, 'extended_PAX_GNU_names_resolved': False, 'complete_archive_sha256_verified': False}
    require(Path(path).stat().st_size <= COMPRESSED_LIMIT, 'Input prefix exceeds4MiB')
    with Path(path).open('rb') as source:
        header = source.read(10); source.seek(0)
        result['gzip_header_recognized'] = (len(header) == 10 and header[:3] == b'\x1f\x8b\x08'
            and not header[3] & 224)
        reader = BoundedGzipReader(source, decompressed_limit)
        document_total = 0
        try:
            for index in range(POLICY['tar_members_max']):
                header = reader.read(512)
                if len(header) != 512:
                    result['stop_reason'] = 'partial_tar_header_or_decompression_bound'; break
                if header == bytes(512):
                    result['archive_end_marker_seen'] = True
                    result['stop_reason'] = 'tar_zero_block_seen'; break
                expected = tar_number(header[148:156])
                require(sum(header[:148]) + 8 * 32 + sum(header[156:]) == expected, 'Invalid tar header checksum')
                name = header[:100].split(b'\0', 1)[0].decode('utf-8', errors='replace')
                prefix = header[345:500].split(b'\0', 1)[0].decode('utf-8', errors='replace') if header[257:263].startswith(b'ustar') else ''
                if prefix: name = prefix + '/' + name
                size = tar_number(header[124:136]); kind = header[156:157].decode('ascii', errors='replace') or '0'
                require(size >= 0, 'Negative tar member size rejected')
                safe = safe_member(name)
                result['members'].append({'name': name, 'size': size, 'typeflag': kind,
                    'safe_relative_name': safe, 'header_checksum_valid': True, 'extracted': False})
                doc = safe and kind in ('0', '\0') and any(word in PurePosixPath(name).name.lower()
                    for word in ('license', 'readme', 'copying', 'copyright'))
                kept = min(size, POLICY['document_bytes_each_max'], POLICY['document_bytes_total_max'] - document_total) if doc else 0
                if kept:
                    text = reader.read(kept); document_total += len(text)
                    result['documents'].append({'name': name, 'bytes_read': len(text), 'member_size': size,
                        'text_utf8_replacement': text.decode('utf-8', errors='replace'), 'text_truncated': len(text) < size})
                    if len(text) != kept:
                        result['stop_reason'] = 'partial_document_or_decompression_bound'; break
                remaining = size - kept + (-size % 512)
                while remaining:
                    block = reader.read(min(65536, remaining)); remaining -= len(block)
                    if not block: break
                if remaining:
                    result['stop_reason'] = 'partial_member_or_decompression_bound'; break
            else: result['stop_reason'] = 'member_count_limit_reached'
        except (ValueError, zlib.error) as error:
            result.update(stop_reason='invalid_or_truncated_archive', error_type=type(error).__name__)
        result.update(reader.stats(), document_bytes_total=document_total)
    result['valid_tar_headers_parsed'] = len(result['members'])
    result['archive_prefix_structure_recognized'] = (result['gzip_header_recognized']
        and result['valid_tar_headers_parsed'] > 0
        and result.get('stop_reason') != 'invalid_or_truncated_archive')
    result['acceptable_prefix_truncation'] = (result['archive_prefix_structure_recognized']
        and result.get('stop_reason', '').startswith('partial_'))
    # Even a zero block/gzip EOF inside this prefix is not a whole-file proof.
    return result


def probe_succeeded(transport, inspection):
    """Successful bounded inspection requires valid format, not merely HTTP200."""
    return (0 < transport['compressed_bytes_read'] <= COMPRESSED_LIMIT
        and transport['status'] in ('prefix_limit_reached', 'response_eof')
        and inspection['decompressed_bytes_generated'] <= DECOMPRESSED_LIMIT
        and inspection.get('archive_prefix_structure_recognized') is True)


def main(args):
    require(args.prepare_only != args.network_probe_authorized, 'Choose offline prepare OR root-authorized network probe')
    folder = args.output_dir.resolve()
    require(folder.is_relative_to(ROOT) and not folder.exists(), 'NEW workspace probe directory required')
    folder.mkdir(parents=True)
    report = {'schema': 1, 'kind': 'ISSAI_Turkish_public_archive_bounded_prefix_probe_v1',
        'status': 'offline_prepared', 'passed': True, 'source_url': URL, 'source_revision': REVISION,
        'publisher_complete_archive_sha256': PUBLISHER_ARCHIVE_SHA256,
        'complete_archive_sha256_verified': False, 'policy': POLICY, 'script_sha256': sha(__file__),
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'actual_network_executed': False,
        'license_source_labels': {'publisher': {'label': 'CC-BY-4.0', 'url': 'https://issai.nu.edu.kz/turkic-asr/',
            'doi': '10.48342/0xes-sf45'}, 'HF_card': {'label': 'MIT', 'scope': 'Separate HF card label; not assumed to override publisher license'}},
        'scope': 'Partial archive-prefix/layout inspection only; no training eligibility, whole-file hash, extraction, or audio verification.'}
    if args.network_probe_authorized:
        report.update(status='network_probe_running', actual_network_executed=True)
        prefix = folder / 'compressed_archive_prefix.bin'
        try:
            report['transport'] = fetch_prefix(prefix)
            report['prefix'] = {'path': str(prefix), 'bytes': prefix.stat().st_size, 'sha256': sha(prefix), 'checksum_scope': 'Only downloaded compressed prefix'}
            report['inspection'] = inspect_prefix(prefix)
            report['passed'] = probe_succeeded(report['transport'], report['inspection'])
            report['status'] = 'bounded_valid_archive_prefix_inspected' if report['passed'] else 'partial_or_failed_probe'
        except Exception as error:
            report.update(status='failed', passed=False, error_type=type(error).__name__)
    with (folder / 'probe_report.json').open('x', encoding='utf-8') as target:
        json.dump(report, target, ensure_ascii=False, indent=2)
    print(json.dumps({'status': report['status'], 'passed': report['passed'],
        'report': str(folder / 'probe_report.json'), 'actual_network_executed': report['actual_network_executed']}))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--network-probe-authorized', action='store_true')
    raise SystemExit(main(parser.parse_args()))
