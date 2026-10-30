"""CPU fixture checks for the new profile-selection auditor, no inference."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'outputs/parakeet_turkish'
SOURCE=OUTPUT/'verify_batched_profiles.py'
spec=importlib.util.spec_from_file_location('profile_auditor',SOURCE)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
profiles=json.loads((OUTPUT/'batching_selection_independent_audit.json').read_text())['profiles']
baseline=json.loads((OUTPUT/'batching_speed_comparison.json').read_text())
fixtures=[
    ('throughput multiplier changed',lambda x:x.__setitem__('throughput_ratio',4.0)),
    ('throughput percent changed',lambda x:x.__setitem__('throughput_increase_percent',300.0)),
    ('selected physical batch changed',lambda x:x['selected_configuration'].__setitem__('batch_size',2)),
    ('long quartet identity changed',lambda x:x['long_clip_safety']['source_ids'].__setitem__(0,'wrong TRAIN recording')),
    ('long timing changed',lambda x:x['long_clip_safety']['timing'].__setitem__('median_seconds_per_update',1.0)),
    ('input hash changed',lambda x:x['input_sha256'].__setitem__('batch_profile_v2_b1_high.json','0'*64)),
    ('historical context changed',lambda x:x.__setitem__('historical_full_run_train_audio_min_per_wall_min',10.0)),
    ('unsupported expanded-run claim',lambda x:x.__setitem__('expanded_run_rate','expanded-run average speed verified')),
]
checks=[]
with tempfile.TemporaryDirectory(dir=ROOT/'work',prefix='profile_selection_checks_') as folder:
    path=Path(folder)/'selection.json'
    path.write_text(json.dumps(baseline))
    module.audit_selection(path,profiles,module.Checks())
    checks.append({'name':'real completed selection accepted','passed':True})
    for name,mutate in fixtures:
        candidate=copy.deepcopy(baseline);mutate(candidate)
        path.write_text(json.dumps(candidate))
        try:
            module.audit_selection(path,profiles,module.Checks())
        except AssertionError as error:
            checks.append({'name':name,'passed':True,'rejected_reason':str(error)})
        else:
            checks.append({'name':name,'passed':False})
strict=json.loads((OUTPUT/'batch_profile_b2_strict_fp32_independent_audit.json').read_text())
checks.append({'name':'actual B2 FP32 rejected despite coarse profiler passed flag',
               'passed':strict['passed'] is False and 'strict FP32 oracle' in strict['error']
               and strict['auditor_sha256']==hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
               'rejected_reason':strict['error']})
result={'passed':all(x['passed'] for x in checks),'gpu_used':False,
        'auditor_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'fixture_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'Real CPU audit selection plus eight one-field corruptions; actual full-model B2 FP32 report strict rejection.',
        'checks':checks}
target=OUTPUT/'batching_selection_guard_verification.json'
target.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'passed':result['passed'],'checks':len(checks),'output':str(target)}))
raise SystemExit(0 if result['passed'] else 1)
