"""All1914 genuine cache/full-label audit; no audio/features materialization."""
import argparse, collections, hashlib, json, os, struct, sys, time
from pathlib import Path
P = Path(__file__).resolve().parents[2]; S = P/'outputs/parakeet_turkish'; sys.path.insert(0, str(S))
from atomic_io_retry_v1 import write_json
from bpe_broad_audio_v1 import record, require
from bpe_replay_cache_labels_v1 import full_replay_labels
ROOT = Path(r'E:\gpu_training_scratchpad'); A = ROOT/'analysis/ASR18_memory_diagnosis_v1'


def main(folder):
    require(folder.resolve().is_relative_to(A.resolve()) and not folder.exists(), 'Fresh E-only audit namespace')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '-1', 'CPU-only audit')
    source = S/'runs/adaptive_20261014_TR_WorldSpeech_expanded_01/training_pool.json'; pool_binding = record(source)
    require(pool_binding['sha256'] == '00b839c717762aba74326e9867969d85b7ffaaabd7ead7ddbe732fe6be8c6cf4', 'Entire original1914 replay source')
    rows = json.loads(source.read_bytes())['replay']; require(len(rows) == 1914 and len({row['source_id'] for row in rows}) == 1914, 'All unique original rows')
    total = 0; signatures = {}
    for row in rows:
        path = Path(row['cache_path']).resolve()
        require(path.is_relative_to(P.resolve()) or path.is_relative_to(ROOT.resolve()), 'Existing project/E cache inputs only')
        size = path.stat().st_size
        require(0 < size <= 128*1024**2, 'Bounded genuine cache; no omission of unsupported row')
        total += size; signatures[str(path)] = (size, path.stat().st_mtime_ns)
    require(total <= 8_000_000_000, 'Predeclared8GB total cache-size limit; stop whole audit before hashing, never filter')
    folder.mkdir(); began = time.monotonic(); groups = {}; entries = []; metadata_bytes = 0
    for index, row in enumerate(rows):
        path = Path(row['cache_path']).resolve(); expected = signatures[str(path)]
        require((path.stat().st_size,path.stat().st_mtime_ns) == expected, 'Original replay file changed after complete preflight')
        cache = record(path); ids, proof = full_replay_labels(row, cache)
        require(len(ids) == row['target_tokens'], 'Every original target required')
        metadata_bytes += proof['label_storage_bytes_decoded']
        entry = {'source_id': row['source_id'], 'language': row['language'], 'cache': cache,
                 'full_target_tokens': len(ids), 'full_target_ids_u32_sha256': hashlib.sha256(b''.join(struct.pack('<I',v) for v in ids)).hexdigest(),
                 'teacher_feature_shape': proof['teacher_feature_shape'], 'original_cache_signature': proof['cache_signature']}
        entries.append(entry); group = groups.setdefault(row['language'], {'rows':0,'target_tokens':0,'MAX_U':0,'MAX_T':0,'count_U_above75':0})
        group['rows'] += 1; group['target_tokens'] += len(ids); group['MAX_U'] = max(group['MAX_U'],len(ids))
        group['MAX_T'] = max(group['MAX_T'],proof['teacher_feature_shape'][0]); group['count_U_above75'] += len(ids)>75
        if index%64 == 0: write_json(folder/'status.json', {'completed':index+1,'expected':1914,'opaque_cache_SHA_bytes_read':2*sum(v['cache']['bytes'] for v in entries),'actual_GPU_updates':0})
    require(record(source) == pool_binding and set(groups) == {'en','es','it','fr','de','pt'}, 'All six original languages and source unchanged')
    require(sum(g['rows'] for g in groups.values()) == 1914 and sum(g['target_tokens'] for g in groups.values()) == 93978
            and max(g['MAX_U'] for g in groups.values()) == 126 and max(g['MAX_T'] for g in groups.values()) == 326
            and sum(g['count_U_above75'] for g in groups.values()) == 116, 'Whole original metadata geometry must agree without filtering')
    require(not any(name in sys.modules for name in ('torch','numpy','scipy','transformers','pyarrow')), 'No numerical/model/audio imports')
    write_json(folder/'report.json', {'kind':'ASR18_whole_original1914_replay_cache_full_labels_CPU_audit_v1','passed':True,
        'source':record(__file__),'label_reader':record(S/'bpe_replay_cache_labels_v1.py'),'pool':pool_binding,
        'rows':1914,'groups':groups,'entries':entries,'total_cache_bytes':total,'opaque_full_cache_SHA_bytes_read':2*total,
        'full_label_storage_bytes_decoded':metadata_bytes,'wall_seconds':time.monotonic()-began,
        'all_original_rows_retained':True,'teacher_feature_storage_decoded':False,'waveforms_decoded':False,
        'actual_native_or_GPU_updates':0,'actual_original_audio_or_feature_values_validated':False,
        'quality_or_retention_gain_claimed':False,'production_enabled':False})
    print(json.dumps({'passed':True,'rows':1914,'total_cache_bytes':total,'wall_seconds':time.monotonic()-began,'actual_GPU_updates':0}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--folder',required=True)
    main(Path(parser.parse_args().folder).resolve())
