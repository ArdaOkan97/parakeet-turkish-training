"""Explicit eligibility for the separately prepared public Turkish v3 pool.

Distinct-audio recordings sharing TRAIN text are allowed. Fixed DEV/TEST text,
global FLEURS sentence IDs and source/audio identities are excluded. CV validated
records keep their honest source split and require an explicit training role.
No GPU, predictions, original data writes or sampler retrofit occurs here.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time
import soundfile as sf
from metrics import normalize_text

OUTPUT = Path(__file__).resolve().parent
DATA = OUTPUT.parents[1] / 'work/data'
CV_REPO = 'ysdede/commonvoice_17_tr_fixed'
CV_REV = '9d8a025f1ae7c5a2f5c43298c7c4910b03995a4b'
FL_REPO = 'google/fleurs'
FL_REV = '70bb2e84b976b7e960aa89f1c648e09c59f894dd'
PREPARATION_SHA = '5c220d7fa4fa554b741afa7c565afe3f1500b3de8bfa99de0b1040a450cf4980'
LABEL_PROOF_SHA = '42aa606057ba78232ef930f7317675ef318dddb034bc302e4de2247eedd04e3a'
V2_FULL_AUDIO_AUDIT_SHA = '90ad0408ea8a2b77ed2b0eecc80a1f2ae46562a29d478cf762474a778afde9ca'
TOKENIZER_SHA = 'b91d500326aca290ed709d629dab2bee56f298ea7323b94855b9f5ef0caeac75'
POLICY = {'version': 3, 'tokenizer_decode_group_tokens': False, 'validated_split_training_role_required': True,
          'repeated_training_text_with_distinct_audio_allowed': True,
          'heldout_reference_and_global_sentence_exclusion': True,
          'duration_bounds_seconds': [.5, 30.], 'maximum_target_tokens': 180}


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read_rows(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def ids_sha(rows):
    return hashlib.sha256('\n'.join(sorted(row['source_id'] for row in rows)).encode()).hexdigest()


def source_isolation_v3(tr, replay, heldout):
    ids = {row['source_id'] for row in heldout}
    sentences = {row['sentence_id'] for row in heldout if row.get('dataset') == FL_REPO and row.get('sentence_id') is not None}
    audios = {row[key] for row in heldout for key in ('audio_sha256', 'wav_sha256', 'canonical_pcm_sha256') if row.get(key)}
    references = {(row['language'], normalize_text(row['text'], row['language'])) for row in heldout}
    tr_references = {normalize_text(row['text'], lang) for row in heldout for lang in (row['language'], 'tr')}
    all_rows = tr + replay
    if len({row['source_id'] for row in all_rows}) != len(all_rows):
        raise ValueError('Duplicate source IDs in training pools')
    for row in all_rows:
        repo, revision, split = row.get('dataset'), row.get('dataset_revision'), row.get('dataset_split')
        if repo == CV_REPO:
            valid = (revision == CV_REV and row.get('language') == 'tr' and row.get('dataset_config') == 'default'
                     and ((split == 'train' and row.get('dataset_file') == 'data/train-00000-of-00001.parquet')
                          or (split == 'validated' and row.get('training_role') == 'train'
                              and row.get('dataset_file') == 'data/validated-00000-of-00001.parquet')))
        elif repo == FL_REPO:
            configuration = {'en': 'en_us', 'es': 'es_419', 'it': 'it_it', 'fr': 'fr_fr', 'de': 'de_de', 'tr': 'tr_tr'}.get(row.get('language'))
            valid = (revision == FL_REV and split == 'train' and configuration is not None
                     and row.get('dataset_config') == configuration
                     and row.get('dataset_file') == f'parquet-data/{configuration}/train-00000-of-00001.parquet')
        else:
            valid = False
        if not valid or not row['source_id'].startswith(f'{repo}:{row.get("dataset_config")}:{split}:'):
            raise ValueError('Unvetted source/split/training role: ' + row['source_id'])
        if (row['source_id'] in ids
            or (repo == FL_REPO and row.get('sentence_id') is not None and row['sentence_id'] in sentences)
            or any(row.get(key) in audios for key in ('audio_sha256', 'wav_sha256', 'canonical_pcm_sha256') if row.get(key))
            or (row['language'], normalize_text(row['text'], row['language'])) in references
            or (row['language'] == 'tr' and normalize_text(row['text'], 'tr') in tr_references)):
            raise ValueError('Fixed heldout source/reference/audio overlap: ' + row['source_id'])


def audio_evidence(row, expected_pcm=None, *, training=True):
    path = Path(row['audio_filepath'])
    if sha256(path) != row['wav_sha256']:
        raise ValueError('Training WAV changed: ' + row['source_id'])
    value = hashlib.sha256()
    with sf.SoundFile(path) as stream:
        if stream.samplerate != 16000 or stream.channels != 1 or stream.subtype != 'PCM_16':
            raise ValueError('Mono16kPCM16 training audio required')
        duration = stream.frames / stream.samplerate
        while True:
            block = stream.read(65536, dtype='int16')
            if not block.size: break
            value.update(block.astype('<i2', copy=False).tobytes())
    if ((training and not .5 <= duration <= 30.) or duration <= 0 or duration != row['duration']
        or (expected_pcm is not None and value.hexdigest() != expected_pcm)):
        raise ValueError('Actual PCM duration/hash differs: ' + row['source_id'])
    return value.hexdigest()


def audit_public_training_inputs(manifest, replay_manifest, proof_path, *, train_limit=None, supplement=None, verify_actual_audio=True):
    started = time.perf_counter()
    if train_limit is not None or supplement is not None:
        raise ValueError('This version binds the full vetted v3 pool; no implicit subset or supplemental source')
    proof_path = Path(proof_path)
    if sha256(proof_path) != PREPARATION_SHA:
        raise ValueError('Pinned completed v3 preparation proof differs')
    proof = json.loads(proof_path.read_text(encoding='utf-8'))
    if Path(replay_manifest).resolve() != (DATA/'replay_train.jsonl').resolve():
        raise ValueError('Existing fixed replay manifest is required')
    manifest = Path(manifest).resolve()
    if manifest != Path(proof['CPU_tokenizer_accepted_manifest_path']).resolve() or sha256(manifest) != proof['CPU_tokenizer_accepted_manifest_sha256']:
        raise ValueError('Exact vetted v3 accepted manifest required')
    label_path = OUTPUT / 'tr_public_v3_tokenizer_acceptance.json'
    if sha256(label_path) != LABEL_PROOF_SHA:
        raise ValueError('Exact CPU tokenizer/audio proof differs')
    labels = json.loads(label_path.read_text(encoding='utf-8'))
    if not labels['passed'] or labels['tokenizer_semantic_sha256'] != TOKENIZER_SHA or not labels['existing_actual_trainer_source_set_preserved'] or labels.get('group_tokens') is not False:
        raise ValueError('Original tokenizer/acceptance verification differs')
    label_rows = labels['accepted_source_labels_and_audio']
    by_id = {row['source_id']: row for row in label_rows}
    tr, replay = read_rows(manifest), read_rows(replay_manifest)
    if len(by_id) != len(label_rows) or set(by_id) != {row['source_id'] for row in tr}:
        raise ValueError('Tokenizer source set is not the actual training manifest')
    if proof.get('tokenizer_decode_group_tokens') is not False or proof.get('vocabulary_changed') is not False:
        raise ValueError('Explicit unchanged-vocabulary ungrouped TDT policy required')
    previous_audit_path = Path(proof['reused_full_v2_audio_audit_path'])
    if sha256(previous_audit_path) != V2_FULL_AUDIO_AUDIT_SHA or proof['reused_full_v2_audio_audit_sha256'] != V2_FULL_AUDIO_AUDIT_SHA:
        raise ValueError('Pinned completed v2 full actual-PCM audit differs')
    previous = json.loads(previous_audit_path.read_text(encoding='utf-8'))
    if not previous.get('passed') or previous['audit']['identity']['accepted_clips'] != 71684:
        raise ValueError('Complete previous actual-PCM audit required')
    previous_manifest = Path(proof['prior_v2_manifest_path'])
    if sha256(previous_manifest) != proof['prior_v2_manifest_sha256'] or previous['audit']['identity']['accepted_manifest_sha256'] != proof['prior_v2_manifest_sha256']:
        raise ValueError('Reused actual-audio manifest differs')
    old_rows = read_rows(previous_manifest)
    old_by_id = {row['source_id']: row for row in old_rows}
    tr_by_id = {row['source_id']: row for row in tr}
    if any(tr_by_id.get(source_id) != row for source_id, row in old_by_id.items()):
        raise ValueError('Previously verified audio rows must remain exact')
    recovery = proof['newly_recovered_actual_audio_evidence']
    recovery_by_id = {row['source_id']: row for row in recovery}
    if len(recovery_by_id) != len(recovery) or set(recovery_by_id) != set(tr_by_id)-set(old_by_id):
        raise ValueError('Newly recovered actual-audio evidence must cover the exact added set')
    for source_id, audio in recovery_by_id.items():
        if audio['wav_sha256'] != tr_by_id[source_id]['wav_sha256'] or audio['canonical_pcm_sha256'] != by_id[source_id]['canonical_pcm_sha256']:
            raise ValueError('Recovered WAV/canonical proof differs')
    if len(old_by_id) != proof['reused_verified_audio_clips'] or len(recovery) != proof['newly_recovered_clips']:
        raise ValueError('Actual-audio proof partition counts differ')
    heldout = []
    for path, expected in proof['heldout_manifests_sha256'].items():
        if sha256(path) != expected: raise ValueError('Fixed heldout manifest changed')
        heldout.extend(read_rows(path))
    source_isolation_v3(tr, replay, heldout)
    heldout_pcm = ({audio_evidence(row, training=False) for row in {r['wav_sha256']:r for r in heldout}.values()} if verify_actual_audio else set())
    for path, expected in proof['protected_original_files_sha256'].items():
        if sha256(path) != expected: raise ValueError('Protected original source/data changed: ' + path)
    for source in proof['sources']:
        if sha256(source['path']) != source['sha256']: raise ValueError('Pinned public parquet changed')
    seen_audio = set(); seen_pcm = set()
    for index, row in enumerate(tr):
        label = by_id[row['source_id']]
        if not 1 <= label['target_tokens'] <= 180 or not 0 <= label['target_max_id'] < 8199:
            raise ValueError('Invalid vetted tokenizer target')
        canonical = audio_evidence(row, label['canonical_pcm_sha256']) if verify_actual_audio else label['canonical_pcm_sha256']
        if not .5 <= row['duration'] <= 30. or not Path(row['audio_filepath']).is_file():
            raise ValueError('Vetted duration or audio file unavailable')
        if canonical in heldout_pcm:
            raise ValueError('Canonical training PCM overlaps fixed heldout audio')
        if row['audio_sha256'] in seen_audio or canonical in seen_pcm:
            raise ValueError('Duplicate exact embedded/canonical training audio')
        seen_audio.add(row['audio_sha256']); seen_pcm.add(canonical)
        if (index + 1) % 5000 == 0: print(f'public v3 PCM/input verification {index + 1}/{len(tr)}', flush=True)
    hours = math.fsum(row['duration'] for row in tr) / 3600
    if len(tr) != proof['CPU_tokenizer_accepted_clips'] or abs(hours - proof['CPU_tokenizer_accepted_hours']) > 1e-10:
        raise ValueError('Full v3 accepted counts/actual duration differ')
    sources = dict(Counter(row.get('data_expansion_role', 'preserved_existing') for row in tr))
    if sources != proof['CPU_tokenizer_accepted_source_counts']: raise ValueError('Source-role counts differ')
    identity = {'policy': POLICY, 'policy_source_sha256': sha256(__file__), 'preparation_proof_sha256': PREPARATION_SHA,
                'label_proof_sha256': LABEL_PROOF_SHA, 'accepted_manifest_sha256': sha256(manifest),
                'accepted_source_ids_sha256': ids_sha(tr), 'tokenizer_semantic_sha256': TOKENIZER_SHA,
                'accepted_clips': len(tr), 'accepted_actual_PCM_hours': hours, 'source_role_counts': sources}
    return {'schema': 1, 'passed': True, 'kind': 'public_turkish_v3_trainer_input_audit', 'identity': identity,
            'accepted_source_ids_sha256': identity['accepted_source_ids_sha256'],
            'elapsed_seconds': time.perf_counter() - started,
            'actual_WAVs_reread_in_this_audit': verify_actual_audio,
            'audio_verification_mode': 'full_actual_PCM_now' if verify_actual_audio else 'hash_bound_completed_v2_full_audio_plus_v3_recovered_actual_audio',
            'audio_mutation_since_reused_audit_not_claimed_checked': not verify_actual_audio,
            'reused_actual_audio_audit_sha256': V2_FULL_AUDIO_AUDIT_SHA,
            'acoustic_cache_acceptance_pending': True, 'optimizer_exposure_claimed': False,
            'no_heldout_predictions_read': True, 'repeated_training_text_retained_for_distinct_audio': True}


def confirm_acoustic_cache_acceptance(rows, audit):
    if ids_sha(rows) != audit['accepted_source_ids_sha256'] or len(rows) != audit['identity']['accepted_clips']:
        raise ValueError('Actual tokenizer/acoustic cache did not retain the complete vetted public v3 pool')
    audit['acoustic_cache_acceptance_pending'] = False
    audit['actual_acoustic_cache_accepted_clips'] = len(rows)
    audit['actual_acoustic_cache_accepted_hours'] = math.fsum(row['duration'] for row in rows) / 3600
