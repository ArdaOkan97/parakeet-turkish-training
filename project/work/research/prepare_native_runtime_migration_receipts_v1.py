"""Declare only the native cache-helper source/provenance transition."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/parakeet_turkish'
LOOP=OUT/'runs/adaptive_20261011_native_memory_fix_01'
def sha(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
fork_path=LOOP/'fork_snapshot/fork_manifest.json'
fork=json.loads(fork_path.read_text(encoding='utf-8'))
assert sha(fork_path)=='d5bf3d509b3ca043e086df183a763e44e8ec4f6009b0609faa092bdd2a6b3527'
config=json.loads(Path(fork['source_run_config']['path']).read_text(encoding='utf-8'))
oldready=config['native_backend_provenance']
newready=LOOP/(fork['target_run_id']+'_native_backend.json')
names={'train_lora_batched_data_v7.py':'train_lora_batched_data_v8.py',
 'native_tdt_checkpointed_v1.py':'native_tdt_checkpointed_v2.py',
 'launch_native_training_v1.py':'launch_native_training_v2.py'}
changes={old:{'old_sha256':sha(OUT/old),'new_filename':new,'new_sha256':sha(OUT/new)} for old,new in names.items()}
for cap in (6,7):
 record={'kind':'native_runtime_state_transition_v1','status':'CPU_prepared','passed':True,
  'source_run_id':fork['source_run_id'],'target_run_id':fork['target_run_id'],'source_step':63,'inherited_selected_step':0,
  'parent_latest':fork['source_checkpoint']['latest'],'parent_best':fork['source_checkpoint']['best'],
  'fork_latest':fork['fork_checkpoint']['latest'],'fork_best':fork['fork_checkpoint']['best'],
  'fork_manifest':{'path':str(fork_path),'sha256':sha(fork_path)},
  'source_run_config':fork['source_run_config'],'source_config_identity_sha256':fork['source_config_identity_sha256'],
  'approved_source_changes':changes,
  'native_backend_transition':{'old_path':oldready,'old_sha256':sha(Path(oldready)),'new_path':str(newready),
   'expected_new_kind':'native_tdt_training_backend_v2'},
  'declared_allocator_policy':{'cap_gb':cap,'end_cache_gb':4,'native_inactive_threshold_bytes':750000000},
  'optimizer_sampler_RNG_history_unchanged_in_fork':True,'training_math_change_claimed':False,
  'GPU_resource_readiness_pending':True,'full_NeMo_trainer_or_numerical_equivalence_claimed':False,
  'prior_best_step0_preserved':True,'latest63_not_qualified_winner':True}
 target=LOOP/f'runtime_migration_cap{cap}.json'
 with target.open('x',encoding='utf-8') as f:json.dump(record,f,ensure_ascii=False,indent=2)
 print(json.dumps({'path':str(target),'sha256':sha(target),'cap_gb':cap}))
