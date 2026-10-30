"""Tiny stdlib-only range/resume/hash tests; no network or actual archive."""
import hashlib
import io
import json
from pathlib import Path
import sys
import types
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'outputs/parakeet_turkish'))
import download_issai_tsc_v1 as download
scratch = ROOT / 'work/research' / ('TSC_download_fixtures_' + uuid.uuid4().hex)
scratch.mkdir()
checks = []
def check(ok, name):
    if not ok: raise AssertionError(name)
    checks.append(name)
def rejects(call, name):
    try: call()
    except ValueError: checks.append(name); return
    raise AssertionError(name)

class Response:
    def __init__(self, data, offset=0, status=206):
        self.status = status; self.total = len(data); self.offset = offset
        self.headers = {'Content-Length': str(len(data) - offset),
            'Content-Range': f'bytes {offset}-{len(data) - 1}/{len(data)}'}
        self.data = io.BytesIO(data[offset:]); self.requests = []
    def geturl(self): return 'https://us.aws.cdn.hf.co/public/object?signed_secret_fixture=1'
    def read1(self, count): self.requests.append(count); return self.data.read(min(7, count))
payload = b'bounded archive fixture: ' * 13
response = Response(payload, offset=19)
check('?' not in download.validate_response(response, 19, len(payload)), 'exact_resumed_range_without_signed_query_log')
rejects(lambda: download.validate_response(Response(payload, 19, 200), 19, len(payload)), 'ignored_HTTPRange_on_resume_rejected')
rejects(lambda: download.validate_response(response, 20, len(payload)), 'wrong_range_start_rejected')
rejects(lambda: download.validate_response(response, 19, len(payload) + 1), 'wrong_pinned_total_rejected')
part = scratch / 'small_archive.part'; part.write_bytes(payload[:19])
digest = hashlib.sha256(payload[:19]); updates = []
class NoSleep:
    def add(self, count): pass
with part.open('ab', buffering=0) as target:
    position = download.copy_response(response, target, digest, 19, len(payload),
        lambda *args: updates.append(args), NoSleep())
check(position == len(payload) and part.read_bytes() == payload and digest.hexdigest() == hashlib.sha256(payload).hexdigest(),
    'resumed_append_has_no_duplicate_or_skipped_bytes_and_correct_full_hash')
check(max(response.requests) <= download.BLOCK and response.data.tell() == len(payload) - 19,
    'reads_bounded_and_stop_at_exact_total')
final = scratch / 'small_archive.tar.gz'
rejects(lambda: download.finalize_verified(part, final, digest, len(payload), 'wrong'), 'checksum_failure_retains_part_and_never_publishes')
check(part.exists() and not final.exists(), 'failed_archive_not_renamed')
download.finalize_verified(part, final, digest, len(payload), digest.hexdigest())
check(final.read_bytes() == payload and not part.exists(), 'matching_full_hash_atomically_publishes')
part.write_bytes(payload)
rejects(lambda: download.finalize_verified(part, final, digest, len(payload), digest.hexdigest()), 'existing_final_never_overwritten')

# Small nonaligned prefix fixture exercises checkpoint/prefix boundary splits.
saved = download.BLOCK, download.PREFIX_BYTES, download.PREFIX_SHA256, download.TOTAL
try:
    download.BLOCK = 11; download.PREFIX_BYTES = 32; download.TOTAL = 100
    prefix = scratch / 'trusted_prefix'; prefix.write_bytes(bytes(range(32)))
    download.PREFIX_SHA256 = hashlib.sha256(prefix.read_bytes()).hexdigest()
    resumed = scratch / 'resumed.part'; resumed.write_bytes(bytes(range(79)))
    previous = {'hashed_prefix_bytes': 17, 'partial_prefix_sha256': hashlib.sha256(bytes(range(17))).hexdigest()}
    hasher, length = download.verify_existing_part(resumed, prefix, previous, lambda *args, **kwargs: None, NoSleep())
    check(length == 79 and hasher.hexdigest() == hashlib.sha256(resumed.read_bytes()).hexdigest(),
        'nonaligned_partial_hash_and_trusted_prefix_rehash_exact')
    bad = {**previous, 'partial_prefix_sha256': 'wrong'}
    rejects(lambda: download.verify_existing_part(resumed, prefix, bad, lambda *args, **kwargs: None, NoSleep()),
        'existing_partial_SHA_corruption_blocks_append')
    content = bytearray(resumed.read_bytes()); content[0] ^= 1; resumed.write_bytes(content)
    rejects(lambda: download.verify_existing_part(resumed, prefix, previous, lambda *args, **kwargs: None, NoSleep()),
        'verified_prefix_mismatch_blocks_append')
finally: download.BLOCK, download.PREFIX_BYTES, download.PREFIX_SHA256, download.TOTAL = saved
check('torch' not in sys.modules and 'nemo' not in sys.modules, 'stdlib_only_no_Torch_NeMo_GPU_network')
report = {'schema': 1, 'kind': 'ISSAI_resumable_downloader_stdlib_fixture_verification_v1', 'passed': True,
    'check_count': len(checks), 'checks': checks, 'downloader_sha256': download.file_sha(download.__file__),
    'fixture_sha256': download.file_sha(__file__), 'scratch': str(scratch), 'actual_network_executed': False,
    'actual_full_archive_acquired': False, 'archive_extraction': False}
destination = ROOT / 'outputs/parakeet_turkish/ISSAI_downloader_CPU_verification_v1.json'
destination.write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps({'passed': True, 'check_count': len(checks), 'report': str(destination)}))
