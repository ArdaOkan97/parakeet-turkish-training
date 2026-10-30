"""Independent read-only integration review: no model load, GPU or inference."""
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
OUTPUT=ROOT/'outputs/parakeet_turkish'
sys.path.insert(0,str(OUTPUT))
controller=importlib.import_module('run_experiment_loop_recovery')
worker=importlib.import_module('benchmark_strong_tr_test_v2')
source=(OUTPUT/'run_experiment_loop_recovery.py').read_text(encoding='utf-8')
checks=[]
def check(name,value):
    if not value:
        raise AssertionError(name)
    checks.append(name)

original=controller.read(OUTPUT/'runs/adaptive_20261010_01/loop_status.json')
check('original_deadline_23_41_22UTC', original['deadline_at_utc']=='2026-10-10T23:41:22.889692+00:00')
check('controller_retains_absolute_original_deadline', "original_deadline = datetime.fromisoformat(original['deadline_at_utc'])" in source and "'deadline_at_utc': original['deadline_at_utc']" in source)
check('training_reserves_finalization_and_stop_grace', 'training_deadline = deadline - reserve - stop_grace' in source)
check('shared_post_deadline_bounded_by_original', 'post_deadline = min(deadline, time.monotonic() + finalization_cap)' in source)
check('all_post_stages_use_same_deadline', source.count('timeout = post_deadline - time.monotonic()')==3)
defaults=controller.parser().parse_args([])
check('worker_v2_default_matches',defaults.strong_test_worker=='benchmark_strong_tr_test_v2.py')
check('new_recovery_auditor_default',defaults.audit_worker=='verify_recovery_results_v1.py')
check('shared_final_reserve30min',defaults.finalization_reserve_seconds==defaults.finalization_cap_seconds==1800)
check('stop_grace180seconds',defaults.stop_grace_seconds==180)
check('old_controller_not_modified',controller.sha(OUTPUT/'run_experiment_loop.py')==original['controller_sha256'])
check('carried_reuse_branch_present', "if best['run_id'] == carried_id:" in source and "validate_reused_finalization(carried_id, best['selected_step'])" in source)
check('fresh_only_finalizer_adapter_check',"if success and best['run_id'] != carried_id:" in source)
check('new_strong_output_not_exists', not (OUTPUT/'results'/worker.RESULT_TAG).exists())
check('new_comparison_not_exists',not (OUTPUT/'adaptive_strong_baseline_test_comparison_v2.json').exists())
check('comparison_v2_path_matches',"comparison_path = OUTPUT / 'adaptive_strong_baseline_test_comparison_v2.json'" in source)
check('auditor_receives_v2_comparison',"'--strong-test-comparison', str(OUTPUT / 'adaptive_strong_baseline_test_comparison_v2.json')" in source)
check('fresh_loop_prevents_overwriting_previous_evidence',"require(not status_path.exists(), 'Recovery ID already exists; preserve its evidence.')" in source)
check('fresh_trial_prevents_existing_run_reuse',"require(not (OUTPUT / 'runs' / run_id).exists() and not stop_file.exists()" in source)
examples,input_audit=worker.verified_test_rows(ROOT/'work/data/test_tr.jsonl')
check('same100fixed_TEST_sources',len(examples)==100 and input_audit['short_form_count']==99 and input_audit['long_form_count']==1)
check('fixed_TEST_manifest_identity',input_audit['manifest_sha256']=='d48bd9cb3ef387c1928b9b2f2f0eaa73faebf285c59e96596569cdb3b2ee7e0c')
protected=worker.protected_code()
check('original_DEV_worker_unchanged',controller.sha(OUTPUT/'benchmark_strong_tr.py')=='46bf78a57112d1adc7c11ec019be5c99356ea6d30f68b61bb49abc3cc6bca475')
final,reuse=controller.validate_reused_finalization('adaptive_20261010_01_screen_B',3204)
check('actual_carried_export_and600TEST_hashes_validated',final['heldout_tests_evaluated'] is True and reuse['selected_step']==3204)
check('carried_policy_no_reinference_or_rewrite','no original B artifact rewritten or inference repeated' in reuse['policy'])
report={'schema':1,'passed':True,'cpu_only':True,'gpu_used':False,'model_loaded':False,'created_at_utc':datetime.now(timezone.utc).isoformat(),
    'checks':checks,'checks_passed':len(checks),'controller_sha256':controller.sha(OUTPUT/'run_experiment_loop_recovery.py'),
    'new_strong_worker_sha256':controller.sha(OUTPUT/'benchmark_strong_tr_test_v2.py'),'original_deadline_at_utc':original['deadline_at_utc'],
    'input_audit':input_audit,'protected_code':protected,'carried_export_reuse':reuse,
    'scope':'CPU hashes/source integration and real carried-export reuse validation. No TEST hypotheses parsed for selection, no ASR inference, no source or old report mutations.',
    'file_overwrite_notes':['Original carried B uses existing export/results with hash validation. Fresh winners finalize only their new run/result tag.',
        'Current versioned strong output/comparison do not exist. Partial same-v2 attempts would require explicit preserved-attempt handling on any future rerun.',
        'Existing top-level latest selection/report pointers are archived by the recovery controller; immutable historical run/result folders remain the provenance source.']}
worker.write_json(OUTPUT/'recovery_strong_integration_cpu_review.json',report)
print(json.dumps({'passed':True,'checks_passed':len(checks),'controller_sha256':report['controller_sha256'],'path':str(OUTPUT/'recovery_strong_integration_cpu_review.json')}),flush=True)
