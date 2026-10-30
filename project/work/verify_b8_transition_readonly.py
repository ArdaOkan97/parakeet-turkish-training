"""Independent CPU-only file/history review of the intentional B8 transition."""
import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/parakeet_turkish'
checks = []

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def check(name, condition):
    checks.append({'name': name, 'passed': bool(condition)})
    if not condition:
        raise AssertionError(name)

manifest_path = OUT / 'runs/adaptive_20261010_recovery_01/b8_transition.json'
m = read(manifest_path)
snapshot = read(m['prior_status_snapshot_path'])
original = read(OUT / 'runs/adaptive_20261010_01/loop_status.json')
check('explicit user-authorized intentional terminal transition', all(m[k] is True for k in ['user_authorized_throughput_priority', 'intentional_transition', 'prior_controller_exited']))
check('unchanged terminal status snapshot', sha(m['prior_status_snapshot_path']) == m['prior_status_snapshot_sha256'] == sha(OUT / 'runs/adaptive_20261010_recovery_01/loop_status.json'))
check('intentional interlock and released awake request', snapshot['status'] == 'failed' and snapshot['failure_message'] == 'Trial ID/stop marker already exists.' and not snapshot.get('active_run_id') and snapshot['awake_restoration']['enabled'] is False)
check('original absolute deadline retained', m['original_deadline_at_utc'] == snapshot['deadline_at_utc'] == original['deadline_at_utc'] == '2026-10-10T23:41:22.889692+00:00')
check('deadline after recorded transition', datetime.fromisoformat(m['created_at_utc']) < datetime.fromisoformat(m['original_deadline_at_utc']))
check('frozen prior controller', sha(OUT / 'run_experiment_loop_recovery.py') == m['prior_controller_sha256'] == '5712954946ee3f68ebfc7e25075ecec2e718432f2393bf327c51d6c4905e406c')
check('B8 effective16 authorized', m['target_configuration'] == {'batch_size':8,'accumulation':2,'effective_batch_size':16,'warmup_steps':25})
c = m['completed_C']
c_dir = OUT / 'runs' / c['run_id']
for stem, expected in [('run_config',c_dir/'run_config.json'),('training_history',c_dir/'training_history.json'),('selection',c_dir/'selection.json'),('best_checkpoint',ROOT/'work'/(c['run_id']+'_best.pt')),('latest_checkpoint',ROOT/'work'/(c['run_id']+'_latest.pt'))]:
    check('exact C '+stem+' path/hash', Path(c[stem+'_path']).resolve() == expected.resolve() and sha(expected) == c[stem+'_sha256'])
cs = read(c['selection_path'])
history = read(c['training_history_path'])
check('191 actual sequential completed C updates', len(history) == c['completed_steps'] == cs['completed_steps'] == 191 and [r['step'] for r in history] == list(range(1,192)))
check('cooperative C stop and selected state', cs['run_id'] == c['run_id'] and cs['accepted'] is True and cs['selected_step'] == c['selected_step'] == 191 and cs['stop_early_reason'] == 'controller_stop_file')
check('final C DEV evaluation is recorded', history[-1].get('final_development_evaluation') is True and history[-1].get('development_selection_only') is True)
b = read(OUT / 'runs/adaptive_20261010_01_screen_B/selection.json')
check('preserved B is stronger by DEV only', b['accepted'] is True and b['selected_step'] == 3204 and b['turkish_validation_wer'] < cs['turkish_validation_wer'])
expected_markers = {(ROOT/'work'/('adaptive_20261010_recovery_01_'+suffix+'.stop')).resolve() for suffix in ['screen_C','screen_B2','extend_1']}
check('all three owned markers preserved', {Path(item['path']).resolve() for item in m['owned_guard_markers']} == expected_markers)
for item in m['owned_guard_markers']:
    check('owned marker hash '+Path(item['path']).name, sha(item['path']) == item['sha256'])

controller_path = OUT / 'run_experiment_loop_recovery_v2.py'
source = controller_path.read_text(encoding='utf-8')
tree = ast.parse(source)
functions = {n.name: ast.get_source_segment(source,n) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
check('v2 controller syntax valid', 'main' in functions and 'validate_transition' in functions)
check('controller revalidates original deadline and all C checkpoint hashes', "manifest['original_deadline_at_utc']" in functions['validate_transition'] and 'latest_checkpoint' in functions['validate_transition'] and "sha(artifact) == completed[stem+'_sha256']" in functions['validate_transition'])
check('original B publication reused without rerunning finalizer', "if best['run_id'] == carried_id:" in functions['main'] and "validate_reused_finalization(carried_id, best['selected_step'])" in functions['main'])
check('new B8 is classified as quality trial', "configuration=quality_configuration,experimental=True" in functions['main'] and "'batch_numerical_equivalence_proven': False" in functions['main'])
check('backend difference documented', "'trainer_fresh_defaults':{'cudnn_allow_tf32':True" in source and "'B8_profile':{'cudnn_allow_tf32':False" in source)
check('strong v2 and auditor v2 defaults', "default='benchmark_strong_tr_test_v2.py'" in functions['parser'] and "default='verify_recovery_results_v2.py'" in functions['parser'])
check('strong CLI frozen-selection and run ID bound', "'--frozen-selection', str(loop_dir / 'frozen_final_selection.json')" in source and "'--selected-run-id', best['run_id']" in source)
check('shared postselection original deadline cap', 'post_deadline = min(deadline, time.monotonic() + finalization_cap)' in source and source.count('timeout = post_deadline - time.monotonic()') == 3)
profile = read(OUT/'batching_speed_comparison.json')['selected_configuration']
runtime = {**profile,'batch_size':8,'accumulation':2}
check('actual profiled B8 runtime configuration', runtime == {'batch_size':8,'accumulation':2,'tdt_implementation':'v2','chunk_size':16,'encoder_checkpointing':True})
check('warmup presentations preserved', 25 * runtime['batch_size'] * runtime['accumulation'] == 100*4 == 400)
check('no existing new strong TEST artifacts to overwrite', not (OUT/'results/strong_whisper_large_v3_greedy_test_v2').exists() and not (OUT/'adaptive_strong_baseline_test_comparison_v2.json').exists())
check('frozen original and new Whisper worker bytes', sha(OUT/'benchmark_strong_tr_test.py') == 'f25e3e4cf61134711389157d0f184a9bac8e1df4c86b77392ace8692af4776eb' and sha(OUT/'benchmark_strong_tr_test_v2.py') == '91012f94ba80bd1c0bbee7f06855d3120d5c982e461184b30968c9fc0a4c9580')
preflight = read(OUT/'recovery_controller_v2_preflight.json')
check('completed controller preflight binds current controller', preflight['passed'] is True and preflight['controller_sha256'] == sha(controller_path) and preflight['transition_manifest_sha256'] == sha(manifest_path))
report = {'schema':1,'passed':all(c['passed'] for c in checks),'checked_at_utc':datetime.now(timezone.utc).isoformat(),'scope':'CPU read-only JSON/source/streaming file hash review; no model load, GPU, predictions, database or original file edits','check_count':len(checks),'checks':checks,'transition_manifest_sha256':sha(manifest_path),'controller_sha256':sha(controller_path),'auditor_sha256':sha(OUT/'verify_recovery_results_v2.py'),'selected_warm_start':{'run_id':b['run_id'],'step':b['selected_step'],'turkish_DEV_wer':b['turkish_validation_wer']},'preserved_C':{'run_id':cs['run_id'],'step':191,'turkish_DEV_wer':cs['turkish_validation_wer']},'runtime_configuration':runtime,'effective_batch_size':16,'original_deadline_at_utc':m['original_deadline_at_utc'],'limitations':['Process exit is bound to the parent-owned terminal status and intentional transition manifest; this file review does not independently inspect live process inventory.','Source integration checks are read-only review evidence, not executed GPU or final TEST results.','The B8 quality trial changes the optimizer batch from4 to16 and uses a DEV-selected checkpoint; all strict BF16 numerical gates remain failed and the benchmark cudnnTF32 flag differs from fresh trainer default.','No guarantee that sampled typical/long benchmark timing or memory generalizes to the whole expanded pool.']}
destination=OUT/'b8_transition_independent_verification.json'
destination.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'passed':report['passed'],'checks':len(checks),'output':str(destination),'controller_sha256':report['controller_sha256']}))
