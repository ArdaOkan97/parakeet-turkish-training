"""Small stdlib policy fixture; no predictions, checkpoints or ML are opened."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'outputs/parakeet_turkish'
SOURCE = HERE / 'analyze_TSC_bpe_lora_saved_DEV_v2.py'


def main(args):
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    names = {'OLD_OPTIMIZER_KIND', 'LOW_OPTIMIZER_KIND', 'OLD_UTILITY_SHA', 'LOW_UTILITY_SHA'}
    nodes = [n for n in tree.body if isinstance(n, ast.Assign) and
        isinstance(n.targets[0], ast.Name) and n.targets[0].id in names]
    nodes += [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'approved_lora_policy']
    def require(ok, message):
        if not ok: raise ValueError(message)
    env = {'require': require}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), env)
    run = HERE / 'runs'
    configs = [json.loads((run / name / 'run_config.json').read_text(encoding='utf-8')) for name in
        ['adaptive_20261011_hf_TSC_BPE512_lora_01', 'adaptive_20261011_hf_TSC_BPE512_low_lora_lr_01_smoke']]
    checks = []
    for c, lr in zip(configs, [3e-4, 3e-5]):
        assert env['approved_lora_policy'](c)['active_lora_lr'] == lr
        checks.append('actual_small_config_admitted_' + str(lr))
    reject = []
    for label, base, mutate in [
        ('unknown_kind', 1, lambda x: x['adaptation_optimizers'].update(kind='invented')),
        ('wrong_utility', 1, lambda x: x['adaptation_optimizers'].update(utility_sha256='changed')),
        ('wrong_source_LR', 1, lambda x: x['training_arguments'].update(lora_lr=1e-4)),
        ('moments_reset', 1, lambda x: x['adaptation_optimizers'].update(source_LoRA_moments_reset=True)),
        ('source_provenance_changed', 1, lambda x: x['adaptation_optimizers'].update(source_original_LoRA_lr=1e-4)),
        ('active_group_old_LR', 1, lambda x: x['adaptation_optimizers']['lora_group'].update(lr=3e-4)),
        ('old_identity_low_LR', 0, lambda x: x['adaptation_phase'].update(lora_lr=3e-5)),
        ('source_hash_changed', 1, lambda x: x['source_sha256'].update(bpe_newrows_lora_adaptation_low_lr_v1_py='changed')),
    ]:
        c = copy.deepcopy(configs[base]); mutate(c['config_identity'])
        if label == 'source_hash_changed':
            c['config_identity']['source_sha256']['bpe_newrows_lora_adaptation_low_lr_v1.py'] = 'changed'
        try: env['approved_lora_policy'](c)
        except ValueError: reject.append(label)
        else: raise AssertionError(label)
    import sys
    assert 'torch' not in sys.modules
    report = {'kind': 'TSC_BPE_LoRA_analyzer_v2_policy_stdlib_verification_v1', 'passed': True,
        'checks': checks, 'rejections': reject, 'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'scope': 'Actual policy AST on two small real JSON configs plus eight explicit corruptions; no full analyzer invocation, predictions, checkpoint, ML or GPU. Actual smoke rescoring is a separate report.'}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2); stream.write('\n')
    print(json.dumps({'report': str(args.output), 'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
        'checks': len(checks), 'rejections': len(reject)}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
