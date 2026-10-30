"""Select actual accepted TRAIN rows for a bounded native memory stress pilot."""
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/parakeet_turkish'
pool_path=OUT/'runs/adaptive_20261011_native_data_01_public_rank8_control/training_pool.json'
config_path=pool_path.with_name('run_config.json')
def sha(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
pool=json.loads(pool_path.read_text(encoding='utf-8'))
cost=lambda row:row['encoder_frames']*(row['target_tokens']+1)*8204*4
order=lambda row:(-cost(row),row['source_id'])
tr=sorted(pool['turkish'],key=order)[:8]
replay_sorted=sorted(pool['replay'],key=order)
replay=[next(r for r in replay_sorted if r['language']==lang) for lang in ('en','es','it','fr','de')]
chosen={r['source_id'] for r in replay}
replay += [r for r in replay_sorted if r['source_id'] not in chosen][:3]
replay=sorted(replay,key=order)
all_rows=tr+replay
assert len({r['source_id'] for r in all_rows})==16
assert len([r for r in tr if cost(r)>1_600_000_000])==4
assert all(cost(r)<=1_700_000_000 for r in all_rows)
assert all(r['dataset_split']=='train' for r in all_rows)
for row in all_rows:
 assert sha(Path(row['audio_filepath']))==row['wav_sha256']
 if row['language']=='tr':
  assert len(row['tr_metadata_labels'])==row['target_tokens'] and row['training_input_type']=='TR_frame_target_metadata_v1'
record={'schema':1,'kind':'native_long_TRAIN_resource_pilot_selection_v1','status':'CPU_prepared',
 'CPU_only':True,'no_GPU_or_predictions':True,'source_run_id':'adaptive_20261011_native_data_01_public_rank8_control',
 'pool_path':str(pool_path),'pool_sha256':sha(pool_path),'config_path':str(config_path),'config_sha256':sha(config_path),
 'runtime':{'batch_size':8,'accumulation':2,'effective_batch':16,'replay_fraction':.5,
  'max_joint_bytes':1_700_000_000,'joint_vocabulary_width':8204,'joint_logits_bytes':4},
 'selection_method':'TR eight highest exact T*(U+1)*8204*4 costs including all four rows above1.6GB; replay highest cost per five old languages plus three next highest remaining costs. Source ID breaks ties.',
 'batches':[tr,replay],'source_ids':[r['source_id'] for r in all_rows],
 'four_largest_TR':[{'source_id':r['source_id'],'T':r['encoder_frames'],'U':r['target_tokens'],'native_joint_bytes':cost(r),'duration_seconds':r['duration']} for r in tr[:4]],
 'audio':{'turkish_seconds':math.fsum(r['duration'] for r in tr),'replay_seconds':math.fsum(r['duration'] for r in replay)},
 'scope':'Resource stress selection only, not throughput-representative sampling or data selection; no TRAIN row is removed from production. GPU feasibility remains pending.'}
target=OUT/'native_long_resource_pilot_selection_v1.json'
with target.open('x',encoding='utf-8') as f:json.dump(record,f,ensure_ascii=False,indent=2)
print(json.dumps({'path':str(target),'sha256':sha(target),'four_largest_TR':record['four_largest_TR']}))
