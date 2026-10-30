"""Small stdlib fixtures: ignored Range, expansion limit, partial archive."""
import gzip
import io
import json
from pathlib import Path
import sys
import tarfile
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'outputs/parakeet_turkish'))
import probe_issai_archive_prefix_v1 as probe

scratch = ROOT / 'work/research' / ('issai_prefix_fixtures_' + uuid.uuid4().hex)
scratch.mkdir()
checks = []
def check(ok, name):
    if not ok: raise AssertionError(name)
    checks.append(name)

class Response:
    status = 200
    headers = {'Content-Length': str(10 * 1024 * 1024)}
    closed = False
    total_returned = 0
    requests = []
    def __enter__(self): return self
    def __exit__(self, *args): self.closed = True
    def geturl(self): return probe.URL + '?signed_fixture_query_must_not_be_logged=1'
    def read(self, count):
        self.requests.append(count); self.total_returned += count
        return b'x' * count # Server ignores Range and would keep sending.
    read1 = read
class Opener:
    def __init__(self, response): self.response = response; self.calls = 0
    def open(self, request, timeout):
        self.calls += 1
        check(request.get_header('Range') == 'bytes=0-4194303', 'request_fixed_prefix_Range')
        check(not request.has_header('Authorization'), 'no_authentication_header')
        return self.response
response = Response(); opener = Opener(response)
path = scratch / 'ignored_range_prefix.bin'
transport = probe.fetch_prefix(path, opener=opener)
check(transport['compressed_bytes_read'] == path.stat().st_size == response.total_returned == probe.COMPRESSED_LIMIT,
    'ignored_Range_never_reads_more_than4MiB')
check(response.closed and max(response.requests) <= 65536 and opener.calls == 1,
    'response_closed_at_cap_without_extra_read_or_retry')
check(transport['requested_range_honored'] is False and transport['complete_archive_sha256_verified'] is False,
    'ignored_Range_and_partial_checksum_honest')
check('?' not in transport['final_public_url_without_query'], 'signed_redirect_query_not_recorded')
class FailedOpen:
    calls = 0
    def open(self, request, timeout):
        self.calls += 1
        raise TimeoutError('Fixture only; no network connection')
failed = FailedOpen()
failed_result = probe.fetch_prefix(scratch / 'failed_open_prefix.bin', opener=failed)
check(failed.calls == failed_result['attempts'] == 2 and failed_result['compressed_bytes_read'] == 0,
    'request_open_timeout_attempts_bounded_to2')

reader = probe.BoundedGzipReader(io.BytesIO(gzip.compress(b'a' * 65536)), limit=4096)
inflated = reader.read(100000)
check(len(inflated) == reader.generated == 4096 and reader.read(1) == b'', 'gzip_expansion_stops_exactly_at_output_cap')
check(reader.stats()['decompression_limit_reached'] is True, 'expansion_limit_reported')

tar = io.BytesIO()
with tarfile.open(fileobj=tar, mode='w') as writer:
    info = tarfile.TarInfo('ISSAI_TSC_218/README.txt'); text = b'License fixture only.\n' * 30
    info.size = len(text); writer.addfile(info, io.BytesIO(text))
valid = scratch / 'complete_small_fixture.gz'; valid.write_bytes(gzip.compress(tar.getvalue()))
inspection = probe.inspect_prefix(valid)
check(inspection['members'][0]['safe_relative_name'] and inspection['documents'][0]['text_utf8_replacement'] == text.decode(),
    'safe_header_and_limited_readme_text_only')
check(inspection['archive_end_marker_seen'] and inspection['partial_archive']
    and not inspection['complete_archive_sha256_verified'], 'even_small_complete_fixture_never_claims_publisher_archive_hash')
truncated_gzip = scratch / 'truncated_gzip_fixture.gz'; truncated_gzip.write_bytes(valid.read_bytes()[:-8])
partial = probe.inspect_prefix(truncated_gzip)
check(not partial['gzip_stream_complete'] and partial['partial_archive'], 'missing_gzip_trailer_is_not_complete_file_proof')
truncated_tar = scratch / 'truncated_tar_fixture.gz'; truncated_tar.write_bytes(gzip.compress(tar.getvalue()[:700]))
partial_tar = probe.inspect_prefix(truncated_tar)
check(partial_tar['stop_reason'].startswith('partial_') and not partial_tar['archive_end_marker_seen'],
    'truncated_tar_payload_reported_without_extraction')
http_ok = {'compressed_bytes_read': 700, 'status': 'response_eof'}
check(partial_tar['archive_prefix_structure_recognized'] and partial_tar['acceptable_prefix_truncation']
    and probe.probe_succeeded(http_ok, partial_tar), 'prefix_truncation_accepted_only_after_valid_gzip_and_tar_header')
html = scratch / 'HTTP200_HTML_instead_of_archive.bin'; html.write_bytes(b'<!doctype html><html>Fixture error page</html>')
invalid = probe.inspect_prefix(html)
check(invalid['stop_reason'] == 'invalid_or_truncated_archive'
    and invalid['valid_tar_headers_parsed'] == 0
    and not probe.probe_succeeded({'compressed_bytes_read': html.stat().st_size, 'status': 'response_eof'}, invalid),
    'HTTP200_HTML_non_gzip_body_never_counts_as_success')
gzip_non_tar = scratch / 'valid_gzip_non_tar_payload.gz'; gzip_non_tar.write_bytes(gzip.compress(b'not a tar archive\n' * 40))
invalid_tar = probe.inspect_prefix(gzip_non_tar)
check(invalid_tar['gzip_header_recognized'] and not invalid_tar['archive_prefix_structure_recognized']
    and not probe.probe_succeeded(http_ok, invalid_tar), 'gzip_alone_without_valid_tar_header_never_counts_as_success')
check(not probe.safe_member('../outside.wav') and not probe.safe_member('C:\\outside.wav')
    and not probe.safe_member('/outside.wav'), 'unsafe_archive_names_never_eligible_for_document_reads')
check('torch' not in sys.modules and 'nemo' not in sys.modules, 'no_Torch_NeMo_CUDA_network_in_fixture_process')
result = {'schema': 1, 'kind': 'ISSAI_bounded_prefix_stdlib_fixture_verification_v1', 'passed': True,
    'check_count': len(checks), 'checks': checks, 'probe_script_sha256': probe.sha(probe.__file__),
    'fixture_script_sha256': probe.sha(__file__), 'scratch': str(scratch),
    'actual_network_executed': False, 'actual_archive_layout_inspected': False,
    'archive_extraction': False, 'audio_decode': False}
report = ROOT / 'outputs/parakeet_turkish/ISSAI_archive_prefix_CPU_verification_v1.json'
report.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({'passed': True, 'check_count': len(checks), 'report': str(report)}))
