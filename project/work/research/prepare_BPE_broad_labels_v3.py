"""Incremental bounded8711-token target sidecar; no audio/model/quality filtering."""
import array
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import time

import psutil
from tokenizers import Tokenizer
from cpu_preparation_guard_v2 import memory, now, scoped, write_json
from ASR18_continuous_data_guard_v4 import DataIOGuard
IO=DataIOGuard('BPE_labels',heavy=False,maximum_bytes=20_000_000_000)

ROOT = Path(r'E:\gpu_training_scratchpad')
AREA = ROOT / 'analysis/BPE_weak_then_clean_v1'
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'outputs/parakeet_turkish'))


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '-1'
    assert os.environ['TOKENIZERS_PARALLELISM'] == 'false'
    for key in ('TEMP', 'TMP', 'HF_HOME', 'TORCH_EXTENSIONS_DIR'):
        scoped(os.environ[key])
    process = psutil.Process()
    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    from tsc_bpe_acquisition_transition_v1 import candidate_assets
    candidate, _ = candidate_assets()
    tokenizer = Tokenizer.from_str(json.dumps(candidate, ensure_ascii=False))
    assert tokenizer.get_vocab_size() == 8711
    source_path = ROOT / 'indexes/BPE_broad_weak_labels_v1.sqlite'
    db_path = ROOT / 'indexes/BPE_broad_labels_8711_v1.sqlite'
    db = sqlite3.connect(db_path)
    db.execute('PRAGMA cache_size=-4096')
    db.execute('PRAGMA temp_store=MEMORY')
    db.execute('CREATE TABLE IF NOT EXISTS labels (ordinal INTEGER PRIMARY KEY, target_ids_u32 BLOB, '
               'target_tokens INTEGER, text_sha256 TEXT, conservative_estimated_joint_bytes INTEGER)')
    db.execute('CREATE TABLE IF NOT EXISTS progress (id INTEGER PRIMARY KEY CHECK(id=1), '
               'frontier INTEGER, rows INTEGER, tokens INTEGER, large_geometry_review_rows INTEGER)')
    db.execute('INSERT OR IGNORE INTO progress VALUES (1,0,0,0,0)')
    db.commit()
    frontier, count, tokens, large = db.execute('SELECT frontier,rows,tokens,large_geometry_review_rows '
                                               'FROM progress WHERE id=1').fetchone()
    state = {'owner': '/root/cpu_data_preparation', 'task': 'ASR-18', 'pid': os.getpid(), 'started_at_utc': now(),
             'GPU_used': False, 'audio_decoded': False, 'quality_filters': [], 'language_filters': [],
             'tokenizer_source': str(PROJECT / 'work/hf_turkish_bpe_candidate_512_v1/tokenizer.json'),
             'tokenizer_sha256': hashlib.sha256((PROJECT / 'work/hf_turkish_bpe_candidate_512_v1/tokenizer.json').read_bytes()).hexdigest(),
             'vocabulary': 8711, 'blank_id': 8192, 'duration_rows': [8711, 8716],
             'TRAIN_admitted': False, 'joint_limit_bytes': 1700000000,
             'oversized_geometry_policy': 'retain_labels_flag_for_bounded_training_path_no_quality_exclusion'}
    last = 0
    while True:
        IO.wait()
        m = memory()
        private = process.memory_info().private
        assert private < 2_000_000_000 and m['commit_ratio'] < .95 and m['commit_headroom_bytes'] >= 2_000_000_000
        assert shutil.disk_usage(ROOT).free >= 100_000_000_000
        if m['commit_ratio'] >= .90:
            state['status'] = 'yielding_for_memory'
            rows = []
        elif source_path.exists():
            source = None
            try:
                source = sqlite3.connect(source_path.as_uri() + '?mode=ro', uri=True, timeout=1)
                source.execute('PRAGMA cache_size=-2048')
                rows = source.execute('SELECT ordinal,text,duration FROM clips WHERE ordinal>? ORDER BY ordinal LIMIT 256',
                                      (frontier,)).fetchall()
            except sqlite3.OperationalError as error:
                if 'locked' not in str(error).lower() and 'busy' not in str(error).lower():
                    raise
                rows = []
            finally:
                if source is not None:
                    source.close()
        else:
            rows = []
        if rows:
            IO.pace(sum(len(r[1].encode('utf-8'))+24 for r in rows))
            encoded = tokenizer.encode_batch([r[1] for r in rows], add_special_tokens=True)
            for (ordinal, text, duration), encoding in zip(rows, encoded):
                ids = encoding.ids
                assert ids and all(0 <= x < 8711 and x != 8192 for x in ids)
                estimated_frames = math.ceil(duration * 12.5) + 2
                estimated_joint = estimated_frames * (len(ids) + 1) * 8716 * 4
                # This is an estimate for loader planning, not an exclusion or exact encoder geometry.
                large += int(estimated_joint > 1700000000)
                packed = array.array('I', ids)
                assert packed.itemsize == 4 and sys.byteorder == 'little'
                db.execute('INSERT INTO labels VALUES (?,?,?,?,?)',
                           (ordinal, packed.tobytes(), len(ids), hashlib.sha256(text.encode()).hexdigest(), estimated_joint))
                frontier = ordinal
                count += 1
                tokens += len(ids)
            db.execute('UPDATE progress SET frontier=?,rows=?,tokens=?,large_geometry_review_rows=? WHERE id=1',
                       (frontier, count, tokens, large))
            db.commit()
            state['status'] = 'tokenizing_committed_source_labels'
        else:
            if m['commit_ratio'] < .90:
                state['status'] = 'waiting_for_next_metadata_commit'
            marker = AREA / 'broad_metadata_actual_CLOSED0_v3.json'
            if marker.exists() and m['commit_ratio'] < .90:
                done = json.loads(marker.read_text())
                assert done['actual_exit_code'] == 0 and count == done['metadata_retained_rows']
                state.update(status='targets_complete_finalizing_continuously',targets_complete=True,complete=False)
        state.update(updated_at_utc=now(), frontier=frontier, label_rows=count, total_target_tokens=tokens,
                     population_mean_U=tokens / count if count else None,
                     conservative_geometry_review_rows=large, own_private_bytes=private, memory=m)
        if state['status']=='targets_complete_finalizing_continuously':
            write_json(AREA/'BPE_label_preparation_status_v3.json',state)
            db.close();IO.heavy=True;IO.role='labels_finalize';IO.state['role']='labels_finalize';IO.wait()
            digest=IO.sha(db_path)
            state.update(status='completed',actual_exit_code=0,complete=True,updated_at_utc=now())
            write_json(AREA/'BPE_label_preparation_status_v3.json',state)
            write_json(AREA/'BPE_labels_actual_CLOSED0_v3.json',{**state,'catalog_path':str(db_path),'catalog_sha256':digest})
            return
        if time.monotonic()-last>10:
            assert 'torch' not in sys.modules and 'transformers' not in sys.modules
            state['no_CPU_model_imports']=True
            write_json(AREA/'BPE_label_preparation_status_v3.json',state);last=time.monotonic()
        if not rows:time.sleep(5)


if __name__ == '__main__':
    main()
