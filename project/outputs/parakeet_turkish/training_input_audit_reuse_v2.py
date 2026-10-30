"""Explicit same-data audit reuse; header checking remains enabled by default.

Stat-only reuse is opt-in. A changed fingerprint fails and requires a full
audit; this helper never silently falls back or trusts known mutated audio.
"""
from pathlib import Path
import hashlib, json, time, re

WORKER_SHA = 'cebb01f64826ad88a05e3577552d049fe0b374d744a64ea16bb7bd96ecf08695'
POLICY_SHA = '08eee3896bf8571c9e97dae1be688ecef7766816c019741dd7631d70f640ce41'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_reuse_receipt(path, *, check_audio_headers=True):
    if type(check_audio_headers) is not bool:
        raise TypeError('check_audio_headers must be an explicit boolean')
    path = Path(path)
    receipt = json.loads(path.read_text(encoding='utf-8'))
    if not (receipt.get('passed') is True
            and receipt.get('kind') == 'observed_full_input_audit_reuse_receipt_v1'
            and receipt.get('observed_completed_full_audit') is True
            and receipt.get('original_returned_audit_JSON_separately_available') is False
            and receipt.get('new_PCM_byte_hashes_computed') is False):
        raise ValueError('Explicit observed full-audit receipt required')
    for evidence in receipt['source_snapshots'].values():
        if sha(evidence['path']) != evidence['sha256']:
            raise ValueError('Immutable completion snapshot changed')
    identity = json.loads(Path(receipt['source_snapshots']['owned_identity']['path']).read_text(encoding='utf-8'))
    io = json.loads(Path(receipt['source_snapshots']['IO_provenance']['path']).read_text(encoding='utf-8'))
    log = Path(receipt['source_snapshots']['owned_worker_log']['path']).read_text(encoding='utf-8')
    values = list(io['arguments'])
    values = values[1:] if values[:1] == ['--'] else values
    argument_sha = hashlib.sha256(json.dumps(values, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    if (identity['expected_run_id'] != receipt['source_run_id']
            or identity['pid'] != receipt['actual_owned_worker_pid']
            or identity['frozen_worker_sha256'] != WORKER_SHA
            or io['worker_sha256'] != WORKER_SHA or receipt['worker_sha256'] != WORKER_SHA
            or identity['data_policy_sha256'] != POLICY_SHA or receipt['data_policy_sha256'] != POLICY_SHA
            or argument_sha != identity['worker_arguments_sha256']
            or argument_sha != receipt['worker_arguments_sha256']
            or 'public v3 PCM/input verification 70000/74883' not in log
            or not re.search(r'acoustic cache \d+/74883', log)):
        raise ValueError('Pinned observed full-audit-to-cache evidence differs')
    expected_audio = set()
    for source, expected in receipt['input_files_sha256'].items():
        if sha(source) != expected:
            raise ValueError('Source/heldout manifest or proof changed; run full PCM audit')
        if Path(source).suffix == '.jsonl':
            with Path(source).open(encoding='utf-8') as stream:
                for line in stream:
                    expected_audio.add(str(Path(json.loads(line)['audio_filepath']).resolve()))
    fingerprints = Path(receipt['fingerprints_path'])
    if sha(fingerprints) != receipt['fingerprints_sha256']:
        raise ValueError('Fingerprint snapshot changed')
    seen = set()
    count = 0
    if check_audio_headers:
        import soundfile as sf
    with fingerprints.open(encoding='utf-8') as stream:
        for line in stream:
            item = json.loads(line)
            audio = Path(item['path']).resolve()
            if str(audio) in seen:
                raise ValueError('Duplicate WAV fingerprint')
            seen.add(str(audio))
            try:
                stat = audio.stat()
            except OSError as error:
                raise ValueError('WAV unavailable; run full PCM audit: ' + str(audio)) from error
            if stat.st_size != item['size'] or stat.st_mtime_ns != item['mtime_ns']:
                raise ValueError('WAV fingerprint changed; run full PCM audit: ' + str(audio))
            if check_audio_headers:
                info = sf.info(audio)
                if (info.channels != 1 or info.samplerate != item['sample_rate']
                        or info.frames / info.samplerate != item['duration']):
                    raise ValueError('Fresh WAV header differs from audited manifest')
            count += 1
            if count % 10000 == 0:
                label = 'fresh WAV/header verification' if check_audio_headers else 'fresh WAV fingerprint verification'
                print('Reused full-audit ' + label + ' ' + str(count), flush=True)
    if count != receipt['unique_TRAIN_replay_heldout_WAV_files'] or seen != expected_audio:
        raise ValueError('Complete exact TRAIN/replay/heldout WAV fingerprint set required')
    return receipt, count


def reuse_verification_fields(receipt, count, *, check_audio_headers=True):
    """Evidence distinguishes fresh headers from headers reused from the receipt."""
    if type(check_audio_headers) is not bool:
        raise TypeError('check_audio_headers must be an explicit boolean')
    mode = ('observed_frozen_full_audit_receipt_plus_fresh_all_WAV_fingerprints_and_headers'
            if check_audio_headers else
            'observed_frozen_full_audit_receipt_plus_fresh_all_WAV_fingerprints_reused_headers')
    return dict(
        audio_verification_mode=mode,
        fresh_WAV_fingerprint_files=count,
        fresh_WAV_header_files=count if check_audio_headers else 0,
        fresh_WAV_fingerprint_and_header_files=count if check_audio_headers else 0,
        reused_audited_WAV_header_files=0 if check_audio_headers else count,
        WAV_headers_checked_this_audit=check_audio_headers,
        original_returned_audit_JSON_separately_available=False,
        actual_WAVs_reread_in_this_audit=False,
        audio_PCM_payloads_read_this_audit=False,
        new_PCM_byte_hashes_computed=False,
        fingerprint_change_requires_full_audit=True,
        audio_mutation_since_reused_audit_not_claimed_checked=True,
        reuse_scope_limitation=receipt['fingerprint_limitation'],
    )


def audit_public_training_inputs_reused(manifest, replay_manifest, preparation_proof, *,
                                        train_limit=None, supplement=None, reuse_receipt=None,
                                        check_audio_headers=True):
    if type(check_audio_headers) is not bool:
        raise TypeError('check_audio_headers must be an explicit boolean')
    if reuse_receipt is None and not check_audio_headers:
        raise ValueError('Stat-only audit reuse requires an explicit prior audit receipt')
    from public_turkish_data_policy_v3 import audit_public_training_inputs, OUTPUT
    if reuse_receipt is None:
        return audit_public_training_inputs(manifest, replay_manifest, preparation_proof,
                                            train_limit=train_limit, supplement=supplement,
                                            verify_actual_audio=True)
    started = time.perf_counter()
    receipt, count = validate_reuse_receipt(reuse_receipt, check_audio_headers=check_audio_headers)
    current_inputs = [manifest, replay_manifest, preparation_proof, OUTPUT / 'tr_public_v3_tokenizer_acceptance.json']
    for source in current_inputs:
        if receipt['input_files_sha256'].get(str(Path(source).resolve())) != sha(source):
            raise ValueError('Receipt does not bind requested exact inputs')
    for source_name, key in [('train_lora_batched_data_v4.py', 'worker_sha256'),
                             ('public_turkish_data_policy_v3.py', 'data_policy_sha256')]:
        if sha(OUTPUT / source_name) != receipt[key]:
            raise ValueError('Observed original source changed')
    # All source/role/heldout/duplicate and pinned byte-proof checks remain.
    audit = audit_public_training_inputs(manifest, replay_manifest, preparation_proof,
                                         train_limit=train_limit, supplement=supplement,
                                         verify_actual_audio=False)
    audit.update(reuse_verification_fields(receipt, count, check_audio_headers=check_audio_headers))
    audit.update(reused_observed_full_audit_receipt_path=str(Path(reuse_receipt).resolve()),
                 reused_observed_full_audit_receipt_sha256=sha(reuse_receipt),
                 elapsed_seconds=time.perf_counter() - started)
    return audit
