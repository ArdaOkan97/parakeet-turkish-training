"""Read-only CPU comparison of the actual profile quartet's audio frontend."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/parakeet_turkish'
sys.path.insert(0, str(OUT))
import torch
from transformers import AutoProcessor
from runtime import read_audio, inputs_for
from profile_batched_training import examples
from batched_lora import collate_training_batch
from types import SimpleNamespace

torch.set_num_threads(4)
processor = AutoProcessor.from_pretrained((ROOT/'work/base_model_path.txt').read_text().strip(), local_files_only=True)
rows, items = examples(SimpleNamespace(quartet='median'))
before = torch.get_rng_state().clone()
singletons = [dict(processor(audio=read_audio(row), sampling_rate=16000,
                              return_tensors='pt')) for row in rows]
cases = []
for name, groups in [('B1', [[0],[1],[2],[3]]), ('B2_homogeneous', [[0,2],[1,3]]), ('B4_mixed', [[0,1,2,3]])]:
    clips = []
    for group in groups:
        batch = collate_training_batch(processor, [rows[i] for i in group], [items[i] for i in group],
                                       blank_id=8192, old_size=8193, device='cpu')
        for position, index in enumerate(group):
            a = singletons[index]
            n = int(a['attention_mask'].sum())
            x = a['input_features'][0,:n].float()
            y = batch.inputs['input_features'][position,:n].float()
            delta = y-x
            clips.append({'source_id': rows[index]['source_id'], 'valid_mel_frames':n,
                          'singleton_shape':list(a['input_features'].shape),
                          'batch_shape':list(batch.inputs['input_features'].shape),
                          'valid_mask_equal':bool(torch.equal(a['attention_mask'][0,:n],batch.inputs['attention_mask'][position,:n])),
                          'valid_frame_count_equal':int(batch.inputs['attention_mask'][position].sum())==n,
                          'valid_features_exact_equal':bool(torch.equal(x,y)),
                          'max_abs_difference':float(delta.abs().max()),
                          'relative_l2_difference':float(delta.norm()/x.norm()),
                          'padded_features_zero':bool((batch.inputs['input_features'][position,n:]==0).all())})
    cases.append({'layout':name,'clips':clips})
rng_equal = bool(torch.equal(before, torch.get_rng_state()))
fe_path = Path(__import__('transformers.models.parakeet.feature_extraction_parakeet',fromlist=['x']).__file__)
model_path = Path(__import__('transformers.models.parakeet.modeling_parakeet',fromlist=['x']).__file__)
text = fe_path.read_text()
result = {'schema':1,'gpu_used':False,'scope':'Actual median profiling quartet and pinned processor, CPU only; no encoder or GPU arithmetic tested',
          'torch_version':torch.__version__,'cpu_threads':4,'processor_rng_unchanged':rng_equal,
          'feature_extractor_source_sha256':hashlib.sha256(fe_path.read_bytes()).hexdigest(),
          'modeling_source_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),
          'frontend_dither_present': 'dither' in text,
          'stft_padding_mode':'constant', 'cases':cases,
          'maximum_valid_feature_abs_difference':max(c['max_abs_difference'] for case in cases for c in case['clips']),
          'maximum_valid_feature_relative_l2_difference':max(c['relative_l2_difference'] for case in cases for c in case['clips']),
          'passed':rng_equal and all(c['valid_mask_equal'] and c['valid_frame_count_equal'] and c['padded_features_zero'] and c['max_abs_difference']<1e-5 for case in cases for c in case['clips']),
          'interpretation':'Any sub-1e-5 CPU frontend difference does not explain or validate real BF16 encoder batching; GPU parity remains a separate gate.'}
(OUT/'batched_frontend_cpu_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:result[k] for k in ['passed','processor_rng_unchanged','maximum_valid_feature_abs_difference','maximum_valid_feature_relative_l2_difference']}))
