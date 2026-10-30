"""Partial held-out safeguard for staged candidates; no TEST predictions or admission."""
import collections
import datetime
import hashlib
import json
import pathlib
import sqlite3
import sys

from yodas_compressed_audio_v1 import ROOT, check_policy, write_json

PROJECT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'outputs/parakeet_turkish'))
from metrics import normalize_text


def main():
    check_policy()
    state = json.loads((ROOT / 'analysis/alania_yodas_preparation_v1/status.json').read_text(encoding='utf-8-sig'))
    shards = state['pilot_shards']
    if len(shards) != 3 or not state.get('pilot_staging_actual_CLOSED0'):
        raise ValueError('Pinned compressed pilot is not actually staged')
    # Read only these existing manifests. Never open inference/evaluation results.
    files = sorted(p for p in (PROJECT / 'work/data').glob('*.jsonl')
                   if p.name.startswith(('validation', 'test')))
    if not files:
        raise ValueError('No existing held-out manifest bindings')
    references = collections.defaultdict(set)
    source_ids = set()
    bindings = []
    unique_heldout = {}
    for path in files:
        data = path.read_bytes()
        records = [json.loads(line) for line in data.decode('utf-8-sig').splitlines() if line.strip()]
        for row in records:
            if row.get('dataset_split', row.get('split')) not in {'validation', 'test'}:
                raise ValueError('Unexpected held-out split')
            if 'hypothesis' in row or 'prediction' in row:
                raise ValueError('Inference outputs are outside this exclusion check')
            sid = row['source_id']
            source_ids.add(sid)
            unique_heldout[sid] = row['language']
            references[normalize_text(row['reference'], 'tr')].add(sid)
        bindings.append({'path': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
                         'rows': len(records)})
    db = sqlite3.connect('file:' + (ROOT / 'indexes/alania_metadata_candidates_v1.sqlite').as_posix() + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA cache_size=-8192')
    candidates = collisions = 0
    candidate_seconds = excluded_seconds = 0.0
    collision_path = ROOT / 'analysis/alania_yodas_preparation_v1/pilot_heldout_text_collisions_v1.jsonl'
    with collision_path.open('w', encoding='utf-8') as output:
        for raw in db.execute('SELECT segment_id,recording_id,shard,duration_s,transcript FROM candidates WHERE eligible=1 AND shard IN (?,?,?) ORDER BY shard,pilot_rank', shards):
            row = dict(raw)
            candidates += 1
            candidate_seconds += row['duration_s']
            matches = references.get(normalize_text(row['transcript'], 'tr'), set())
            if matches:
                collisions += 1
                excluded_seconds += row['duration_s']
                # Only exclusion identities are retained; never use TEST predictions.
                row['excluded_reason'] = 'exact_official_normalized_heldout_reference'
                row['matching_heldout_source_ids'] = sorted(matches)
                output.write(json.dumps(row, ensure_ascii=False) + '\n')
    db.close()
    result = {'kind': 'staged_Alania_partial_heldout_text_exclusion_v1',
        'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'passed_declared_check': True, 'candidate_shards': shards, 'candidate_clips': candidates,
        'candidate_hours': candidate_seconds / 3600, 'exact_text_collision_clips': collisions,
        'exact_text_collision_hours': excluded_seconds / 3600,
        'bound_heldout_manifests': bindings, 'unique_heldout_source_ids': len(source_ids),
        'unique_heldout_languages': dict(collections.Counter(unique_heldout.values())),
        'normalizer_source_sha256': hashlib.sha256((PROJECT / 'outputs/parakeet_turkish/metrics.py').read_bytes()).hexdigest(),
        'collision_manifest': str(collision_path), 'collision_manifest_sha256': hashlib.sha256(collision_path.read_bytes()).hexdigest(),
        'scope': 'Exact normalized transcript exclusion against the explicitly bound existing manifests only',
        'limitations': ['Not a cross-source audio fingerprint or semantic overlap check.',
            'Different transcript text can describe the same audio.',
            'Complete all-seven-language held-out source/audio coverage still requires verification before admission.',
            'Independent Turkish language, transcript and alignment QC are pending.'],
        'TEST_predictions_read': False, 'GPU_or_model_used': False,
        'official_scorer_or_gates_changed': False, 'accepted_TRAIN_hours': 0,
        'complete_heldout_exclusion_claimed': False}
    report = ROOT / 'analysis/alania_yodas_preparation_v1/pilot_heldout_text_exclusion_v1.json'
    write_json(report, result)
    print(json.dumps({'report': str(report), 'sha256': hashlib.sha256(report.read_bytes()).hexdigest(),
        'candidate_clips': candidates, 'exact_text_collisions': collisions, 'accepted_TRAIN_hours': 0}), flush=True)


if __name__ == '__main__':
    main()
