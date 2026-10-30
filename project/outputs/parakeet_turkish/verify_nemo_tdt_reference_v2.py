"""Pinned-source successor to the captured-main CPU-only NeMo recurrence audit."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import torch

import verify_nemo_tdt_reference_v1 as prior

COMMIT = 'e4a34fd25f895ecd7f64f0e56bc288165e078da9'
DIRECTORY = prior.ROOT / 'work/research/nemo_reference/pinned_e4a34fd25f89'
SOURCE = DIRECTORY / 'nemo/collections/asr/losses/rnnt_pytorch.py'
PIN = '486c2395dd5d782da99228a48d727196efbd6d48a3182e15ad73241fbedb4d82'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('nemo_tdt_reference_cpu_verification_v2.json'))
    args = parser.parse_args()
    torch.set_num_threads(2)
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU-only verification must not initialize CUDA')
    receipt_path = DIRECTORY / 'source_receipt.json'
    receipt = json.loads(receipt_path.read_text())
    if receipt['commit'] != COMMIT or hashlib.sha256(SOURCE.read_bytes()).hexdigest() != PIN:
        raise ValueError('Pinned NVIDIA source or receipt changed')
    source_checks = {}
    for item in receipt['files']:
        path = Path(item['path'])
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != item['sha256']:
            raise ValueError('Captured source hash changed: ' + item['repo_path'])
        source_checks[item['repo_path']] = actual
    def node(path):
        return next(x for x in ast.parse(path.read_text(encoding='utf-8')).body
                    if isinstance(x, ast.ClassDef) and x.name == 'TDTLossPytorch')
    identical = ast.dump(node(prior.SOURCE), include_attributes=False) == ast.dump(node(SOURCE), include_attributes=False)
    prior_source = prior.SOURCE
    prior.SOURCE = SOURCE  # Changes only the standalone audit module's input, never training code.
    cls, edits, source_sha = prior.reference_class()
    # reference_class hashes canonical read_text bytes; file hash is recorded separately.
    report = {'schema_version': 2, 'scope': 'Pinned official NeMo PyTorch TDT class with two allocation-only CPU portability edits; synthetic CPU FP32, not native Numba/GPU or full model training parity',
              'official_commit': COMMIT, 'official_source_url': f'https://github.com/NVIDIA-NeMo/Speech/blob/{COMMIT}/nemo/collections/asr/losses/rnnt_pytorch.py',
              'official_source_snapshot_path': str(SOURCE), 'official_source_file_sha256': PIN,
              'official_source_text_sha256': source_sha, 'source_receipt_path': str(receipt_path),
              'source_receipt_sha256': hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
              'captured_primary_files_sha256': source_checks, 'allocation_only_edits': edits,
              'TDT_class_AST_equal_to_previous_main_capture': identical,
              'previous_main_capture_path': str(prior_source),
              'previous_main_capture_sha256': hashlib.sha256(prior_source.read_bytes()).hexdigest(),
              'audit_implementation_v1_sha256': hashlib.sha256(Path(prior.__file__).read_bytes()).hexdigest(),
              'unsupported_reference_features': ['native NeMo module/typecheck integration', 'Numba CUDA kernel', 'omega RNNT/TDT stochastic mixing', 'FastEmit', 'full encoder/frontend/decoder end-to-end parity'],
              'cases': []}
    cases = [
        ('B1_base_none', 1, 8, 4, 41, 40, 0., 'none', (0,1,2,3,4)),
        ('B2_padded_none', 2, 9, 4, 41, 40, 0., 'none', (0,1,2,3,4)),
        ('B4_padded_mean', 4, 10, 5, 41, 40, 0., 'mean', (0,1,2,3,4)),
        ('B2_sigma002_mean_volume', 2, 7, 3, 41, 40, .02, 'mean_volume', (0,1,2,3,4)),
        ('B2_sigma01_sum', 2, 8, 3, 41, 40, .1, 'sum', (0,1,2,3,4)),
        ('B2_sigma002_mean_batch', 2, 7, 3, 41, 40, .02, 'mean_batch', (0,1,2,3,4)),
        ('B2_nonlast_blank_appended_TR', 2, 6, 3, 8199, 8192, 0., 'none', (0,1,2,3,4)),
        ('B2_reordered_durations', 2, 7, 3, 41, 40, .02, 'mean', (4,0,2,1,3)),
        ('B1_large_duration', 1, 7, 3, 41, 40, 0., 'none', (0,1,3,8)),
    ]
    for values in cases:
        report['cases'].append(prior.case(cls, *values))
    report['passed'] = all(item['passed'] for item in report['cases']) and identical
    report['cuda_initialized'] = torch.cuda.is_initialized()
    report['code_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['protected_source_sha256'] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ('efficient_tdt.py', 'efficient_tdt_v2.py', 'runtime.py', 'batched_lora.py')}
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'passed': report['passed'], 'cases': len(report['cases']), 'class_AST_identical': identical,
                      'official_commit': COMMIT, 'cuda_initialized': report['cuda_initialized']}))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
