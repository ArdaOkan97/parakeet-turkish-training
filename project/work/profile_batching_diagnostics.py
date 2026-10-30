"""Sequential real-GPU batching oracle; one GPU child at a time."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'outputs/parakeet_turkish'
PROFILE = OUT / 'profile_batched_training.py'
env = dict(os.environ, PYTHONUTF8='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')

def run(name, flags):
    command = [sys.executable, '-X', 'utf8', str(PROFILE), '--parity-only', '--output', name+'.json', *flags]
    print(json.dumps({'starting':name}), flush=True)
    with (ROOT/'work'/f'{name}.log').open('w', encoding='utf-8') as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    result = json.loads((OUT/f'{name}.json').read_text(encoding='utf-8'))
    print(json.dumps({'finished':name, 'parity':{k:v for k,v in result['parity'].items() if k not in ('per_parameter',)}}), flush=True)
    return result

fp32 = ROOT/'work/profile_gradients_median_fp32_reference.pt'
run('batch_profile_original_b1_fp32_diagnostic', ['--reference','--batch-size','1','--tdt','original','--precision','fp32'])
result = run('batch_profile_v2_b2_fp32_diagnostic', ['--batch-size','2','--tdt','v2','--precision','fp32','--reference-path',str(fp32)])
p = result['parity']
strict = (p['gradient_relative_l2_difference'] < .002 and p['gradient_cosine_similarity'] > .99999
          and all(g['relative_l2_difference'] < .002 and g['cosine_similarity'] > .99999 for g in p['gradient_groups'].values())
          and max(p['component_absolute_deltas']) < 1e-4
          and max(p['encoder_feature_relative_l2_differences']) < 1e-4)
if not strict:
    raise RuntimeError('Full-model FP32 batching oracle did not pass strict semantic checks.')
same_shape = ROOT/'work/profile_gradients_median_bf16_b2_reference.pt'
run('batch_profile_original_b2_bf16_diagnostic', ['--reference-batched','--batch-size','2','--tdt','original','--reference-path',str(same_shape)])
run('batch_profile_v2_b2_bf16_same_shape_diagnostic', ['--batch-size','2','--tdt','v2','--reference-path',str(same_shape)])
print(json.dumps({'strict_fp32_semantics_passed':strict,'same_shape_bf16_diagnostics_complete':True}), flush=True)
