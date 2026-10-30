"""Read-only independent arithmetic/provenance audit, no Torch or GPU import."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/parakeet_turkish'
path = OUT / 'nemo_native_loss_linker_v4_verification.json'
link_path = OUT / 'nemo_native_loss_linker_v4_verification_linker_provenance.json'
proof, linker = (json.loads(p.read_text(encoding='utf-8')) for p in (path, link_path))
checks = 0


def check(value, label):
    global checks
    checks += 1
    if not value:
        raise AssertionError(label)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


check(proof['passed'] and proof['native_kernel_executed'], 'Actual native proof completion')
check(proof['code_sha256'] == sha(OUT / 'verify_nemo_native_loss_v3.py'), 'Frozen verifier hash')
check(proof['input_sha256'] == '8c3481e742f7df332102f718bc632314cd8869896ef292b28b2e0654bbfe1259', 'Pinned actual TRAIN capture')
check(proof['input_shape'] == [2, 128, 29, 8204], 'Unchanged full TRAIN input shape')
check(len(proof['cases']) == 3, 'All configured cases')
check([(x['sigma'], x['omega']) for x in proof['cases']] == [(0.0, 0.0), (0.02, 0.0), (0.02, 1.0)], 'Exact scientific cases')
for case in proof['cases']:
    delta = max(abs(a - b) for a, b in zip(case['full_width_official_reference_loss'], case['native_loss']))
    check(len(case['expected_loss']) == len(case['native_loss']) == 2, 'Two full TRAIN rows')
    check(math.isclose(delta, case['maximum_raw_loss_delta'], rel_tol=1e-12, abs_tol=1e-12), 'Raw loss delta arithmetic')
    gate = case['thresholds']
    check(gate == {'raw_loss_absolute': 1e-4, 'gradient_relative_l2': 2e-4, 'gradient_min_cosine': 0.999999}, 'Strict gates unchanged')
    check(delta < gate['raw_loss_absolute'], 'Raw loss gate')
    gradient = case['gradient']
    check(gradient['finite'], 'Finite native full-logit gradient')
    check(all(math.isfinite(gradient[k]) for k in ('maximum_absolute_delta', 'relative_l2', 'cosine')), 'Finite gradient statistics')
    check(0 <= gradient['relative_l2'] < gate['gradient_relative_l2'] and gate['gradient_min_cosine'] < gradient['cosine'] <= 1, 'Native gradient gate')
    check(case['passed'], 'Truthful per-case completion')
    check(case['compressed_reference_raw_loss_delta'] < 1e-4, 'Reference contraction vs unchanged full loss')
    check(case['reference_compression']['original_token_classes'] == 8199, 'Full token width')
    if case['omega'] == 0:
        check(case['custom_v2_raw_loss_delta'] < 1e-4, 'Current custom TDT raw loss gate')
        check(case['custom_v2_gradient']['finite'] and case['custom_v2_gradient']['relative_l2'] < 2e-4
              and case['custom_v2_gradient']['cosine'] > 0.999999, 'Current custom gradient gate')
    else:
        check(case['duration_gradient_max_abs'] == 0.0, 'RNNT branch duration gradient zero')
check(linker['passed'] and linker['status'] == 'complete', 'Coherent linker proof completed')
check(linker['launcher_sha256'] == sha(OUT / 'launch_nemo_linker_v4.py'), 'Frozen successful launcher')
check(linker['verifier_sha256'] == proof['code_sha256'] and linker['input_sha256'] == proof['input_sha256'], 'Launcher exact verifier/input binding')
for key in ('actual_binding_linker', 'actual_binding_linker_after_proof'):
    check(linker[key]['version'] == [12, 9], 'Actual library version before/after proof')
    check(linker[key]['sha256'] == '998494a8eff0d8382b41ba443d668e3eaea7b67852256ca2982c841cd5bed47a', 'Exact 12.9 DLL before/after proof')
    check(Path(linker[key]['path']).resolve() == (ROOT / 'work/nemo-native-env-v2/Lib/site-packages/nvidia/nvjitlink/bin/nvJitLink_120_0.dll').resolve(), 'Isolated linker path')
check(linker['cuda_initialized_before_proof'] is False, 'Preload itself did not initialize CUDA')
report = {'schema': 1, 'passed': True, 'checks': checks,
          'proof_path': str(path), 'proof_sha256': sha(path),
          'linker_provenance_path': str(link_path), 'linker_provenance_sha256': sha(link_path),
          'script_sha256': sha(__file__), 'maximum_native_raw_loss_delta': max(x['maximum_raw_loss_delta'] for x in proof['cases']),
          'maximum_native_gradient_relative_l2': max(x['gradient']['relative_l2'] for x in proof['cases']),
          'current_custom_gradient_relative_l2_max': max(x['custom_v2_gradient']['relative_l2'] for x in proof['cases'] if x['omega'] == 0),
          'scope': 'Independently reopened JSON, recomputed raw loss deltas, strict statistical gate arithmetic and exact input/source/linker binding. Full gradient tensors were not saved by the proof and were not independently recomputed here. No training-throughput or whole-model equivalence claim.',
          'CUDA_executed': False, 'Torch_imported': False}
output = OUT / 'nemo_native_loss_linker_v4_independent_review.json'
if output.exists():
    raise ValueError('Preserve existing independent review')
output.write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps({'passed': True, 'checks': checks, 'maximum_native_gradient_relative_l2': report['maximum_native_gradient_relative_l2']}))
