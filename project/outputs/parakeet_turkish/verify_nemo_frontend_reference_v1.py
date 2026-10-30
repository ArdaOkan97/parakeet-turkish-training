"""CPU-only pinned NeMo/HF mel feature comparison on fixed DEV audio.

Extracts the unmodified FilterbankFeatures class and normalize_batch function.
Only the CUDA-availability error-check guard is changed to x.is_cuda so that
CPU inputs never initialize a CUDA context. No feature computation is replaced.
"""
import ast
import hashlib
import json
import math
from pathlib import Path
import random
from types import SimpleNamespace

import librosa
import numpy as np
import soundfile as sf
import torch
from torch import nn
from transformers import AutoProcessor
import transformers.models.parakeet.feature_extraction_parakeet as hf_features

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / 'work/research/nemo_reference/pinned_e4a34fd25f89/nemo/collections/asr/parts/preprocessing/features.py'
PIN = '5cbc7b5eff0c2e57c4d9e548e5abc33b8c20078e48661faba9f5872c123d4321'
COMMIT = 'e4a34fd25f895ecd7f64f0e56bc288165e078da9'


def native_class():
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != PIN:
        raise ValueError('Pinned NVIDIA source changed')
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    nodes = [node for node in tree.body if getattr(node, 'name', None) in ('normalize_batch', 'FilterbankFeatures')]
    edits = []
    class CPUGuard(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if ast.unparse(node.func) == 'torch.cuda.is_available':
                edits.append({'line': node.lineno, 'before': ast.unparse(node), 'after': 'x.is_cuda'})
                return ast.copy_location(ast.Attribute(ast.Name(id='x', ctx=ast.Load()), attr='is_cuda', ctx=ast.Load()), node)
            return node
    nodes = [CPUGuard().visit(node) for node in nodes]
    if len(edits) != 1:
        raise ValueError('Unexpected official CUDA guard changed')
    nop = lambda *args, **kwargs: None
    scope = {'math': math, 'random': random, 'librosa': librosa, 'np': np, 'torch': torch,
             'nn': nn, 'CONSTANT': 1e-5, 'PackedEncoderActivations': object,
             'logging': SimpleNamespace(debug=nop, info=nop, warning=nop)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(SOURCE), 'exec'), scope)
    return scope['FilterbankFeatures'], edits


def run_case(name, arrays, identities, feature_extractor, native):
    tensor = nn.utils.rnn.pad_sequence([torch.from_numpy(x) for x in arrays], batch_first=True)
    lengths = torch.tensor([len(x) for x in arrays])
    rng = torch.get_rng_state().clone()
    native_features, native_lengths = native(tensor.clone(), lengths)
    hf = feature_extractor(arrays, sampling_rate=16000, return_tensors='pt')
    values = []
    native_features = native_features.transpose(1, 2)
    hf_lengths = hf.attention_mask.sum(-1)
    for index, identity in enumerate(identities):
        length = int(native_lengths[index])
        left = native_features[index, :length].double()
        right = hf.input_features[index, :length].double()
        difference = left - right
        values.append({'identity': identity, 'samples': len(arrays[index]), 'valid_frames': length,
                       'finite': bool(torch.isfinite(left).all() and torch.isfinite(right).all()),
                       'maximum_absolute_difference': float(difference.abs().max()),
                       'relative_l2_difference': float(difference.norm() / left.norm().clamp_min(1e-30)),
                       'native_maximum_absolute_value': float(left.abs().max()),
                       'hf_maximum_absolute_value': float(right.abs().max()),
                       'native_zero_padding': bool(native_features[index, length:].count_nonzero() == 0),
                       'hf_zero_padding': bool(hf.input_features[index, length:].count_nonzero() == 0)})
    return {'name': name, 'frame_lengths_equal': bool(torch.equal(native_lengths, hf_lengths)),
            'native_shape': list(native_features.shape), 'hf_shape': list(hf.input_features.shape),
            'RNG_unchanged': bool(torch.equal(rng, torch.get_rng_state())), 'rows': values}


def main():
    torch.set_num_threads(2)
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU-only audit must not initialize CUDA')
    processor_path = Path((ROOT / 'work/base_model_path.txt').read_text().strip())
    feature_extractor = AutoProcessor.from_pretrained(processor_path, local_files_only=True).feature_extractor
    cls, edits = native_class()
    native = cls(sample_rate=16000, n_window_size=400, n_window_stride=160, n_fft=512,
                 nfilt=128, normalize='per_feature', preemph=.97, dither=0., pad_to=0,
                 log_zero_guard_type='add', log_zero_guard_value=2**-24, use_grads=False).eval()
    audio, identities = [], []
    for language in ('en', 'tr'):
        rows = [json.loads(line) for line in (ROOT / f'work/data/validation_{language}.jsonl').read_text(encoding='utf-8').splitlines()[:2]]
        for row in rows:
            values, sr = sf.read(row['audio_filepath'], dtype='float32')
            if sr != 16000 or values.ndim != 1:
                raise ValueError('Expected fixed mono16kHz DEV audio')
            audio.append(values)
            identities.append({'source_id': row['source_id'], 'language': language,
                               'audio_filepath': row['audio_filepath'],
                               'wav_sha256': hashlib.sha256(Path(row['audio_filepath']).read_bytes()).hexdigest()})
    cases = [run_case(f'real_singleton_{index}', [values], [identity], feature_extractor, native)
             for index, (values, identity) in enumerate(zip(audio, identities))]
    cases.append(run_case('real_padded_B4', audio, identities, feature_extractor, native))
    cases.append(run_case('synthetic_silence_B2', [np.zeros(16000, dtype='float32'), np.zeros(67360, dtype='float32')],
                          ['all_zero_1s', 'all_zero_4.21s'], feature_extractor, native))
    actual = [row for case in cases[:-1] for row in case['rows']]
    passed = all(case['frame_lengths_equal'] and case['RNG_unchanged'] for case in cases)
    passed &= all(row['finite'] and row['native_zero_padding'] and row['hf_zero_padding']
                  and row['relative_l2_difference'] < 1e-4 for row in actual)
    report = {'schema_version': 1, 'passed_real_audio_matched_settings': passed,
              'scope': 'Pinned dense NeMo FilterbankFeatures extracted without NeMo installation, CPU FP32, dither0/eval with same128mel settings as HF; no encoder or native end-to-end model proof',
              'official_commit': COMMIT, 'official_source_path': str(SOURCE), 'official_source_sha256': PIN,
              'official_source_url': f'https://github.com/NVIDIA-NeMo/Speech/blob/{COMMIT}/nemo/collections/asr/parts/preprocessing/features.py',
              'cpu_guard_edits': edits, 'HF_feature_extractor_code_sha256': hashlib.sha256(Path(hf_features.__file__).read_bytes()).hexdigest(),
              'feature_extractor_configuration': feature_extractor.to_dict(),
              'cases': cases, 'cuda_initialized': torch.cuda.is_initialized(),
              'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    OUT.joinpath('nemo_frontend_reference_cpu_verification_v1.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'real_audio_passed': passed, 'maximum_real_relative_l2': max(row['relative_l2_difference'] for row in actual),
                      'silence': cases[-1]['rows'], 'cuda_initialized': report['cuda_initialized']}))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
