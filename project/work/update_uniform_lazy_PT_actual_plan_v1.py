from pathlib import Path
from datetime import datetime, timezone
import json

root = Path(__file__).resolve().parents[1]
out = root / 'outputs/parakeet_turkish'
path = out / 'experiment_plan.json'
plan = json.loads(path.read_text(encoding='utf-8'))
baseline = json.loads((out/'Portuguese_retention_baselines_v1.json').read_text(encoding='utf-8'))
profile = json.loads((out/'lazy_training_rows_actual_CPU_profile_v1.json').read_text(encoding='utf-8'))
assert baseline['Portuguese_gate']['passed'] is False
assert profile['passed'] and profile['actual_rows_staged'] == 253798
item = next(x for x in plan['ordered_experiments'] if x['id']=='04')
item.update(status='preparing_full_data_priority', actual_GPU_training_started=False,
    run_id='adaptive_20261012_TR_uniform_full_PT_01',
    source_receipt='TSC_uniform_full_PT_recovery_ROOT_source_v1.json',
    source_original_five_qualified=True, source_Portuguese_qualified=False,
    source_PT_word_errors=96, source_PT_reference_words=1095,
    Portuguese_retention_max_word_errors=66, Portuguese_recovery_grace_new_updates=1000,
    initial_all_six_champion_claimed=False,
    selection_policy='Select only all-six-retention-qualified checkpoints. Preserve original old-five champion as immutable lineage; it is not an all-six winner.',
    fullpass_early_stop_policy='No plateau, patience, or target-WER stop before actual253798 new-pass unique clips. Resource/nonfinite, original-five retention and post-grace all-six retention guards remain active.',
    indexed_metadata_actual_CPU_profile='lazy_training_rows_actual_CPU_profile_v1.json',
    indexed_metadata_actual_rows=253798, indexed_metadata_CPU_stage_seconds=profile['stage_seconds'],
    indexed_metadata_observed_private_peak_bytes=profile['own_private_peak_bytes_observed'],
    indexed_metadata_cache_rows=256, actual_GPU_memory_fit_claimed=False,
    pending=['Final controller/worker independent review', 'Actual streamed training admission', 'Controlled first measurement of changed indexed runtime under unchanged live memory guards'])
plan['retention_max_fixed_DEV_word_errors']['pt'] = 66
plan['Portuguese_gate_status'] = 'Actual baseline60/1095, source6500 has96/1095 and fails strict limit66. Explicitly repair with Portuguese TRAIN replay; strict all-six winner selection remains unchanged.'
policy = plan['automatic_execution_policy']
policy['Portuguese'] = 'Actual original-v3 PT baseline and BEST6500 PT50 measurement complete. Source passes oldfive but fails newPT; full run explicitly addsPT replay with1000 new-update recovery grace. Oldfive always guarded and every new BEST must pass allsix. Aftergrace two consecutive allsix failures require review.'
policy['loading'] = 'Audio remains batch loaded; Turkish metadata is indexed on disk with mmap columns and256-row decoded cache. Actual profile30.6MB/253798 rows is CPU metadata only, not trainer memory-fit proof. Streamed eligibility/heldout/WAV-stat admission is mandatory.'
policy['fullpass_stop'] = item['fullpass_early_stop_policy']
policy['pagefile'] = 'Pagefile proposal remains unapplied and unapproved. User requests lower-memory loading; do not treat explanatory questions as administrator approval.'
plan['status_authority'] += ' The proposed full-run ID has no optimization steps yet; sole Trackio mirror V10 is healthy.'
plan['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
path.write_text(json.dumps(plan,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({'updated':str(path),'run_id':item['run_id'],'actual_training_started':False,'PT_gate':66,'metadata_private_peak_bytes':profile['own_private_peak_bytes_observed']}))
