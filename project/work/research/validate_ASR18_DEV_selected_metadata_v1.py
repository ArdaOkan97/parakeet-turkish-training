"""Actual saved descriptor-selection controls; never open tensor payloads."""
import argparse
from copy import deepcopy
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'outputs/parakeet_turkish'))


class NoNumericalRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy', 'scipy', 'transformers', 'pyarrow', 'soundfile', 'av'}:
            raise ImportError('Metadata-only selection forbids numerical/audio import: ' + fullname)


def record(path):
    path = Path(path).resolve(); raw = path.read_bytes()
    assert path.suffix in ('.json', '.py'), 'No tensor/audio input'
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def main(output):
    sys.meta_path.insert(0, NoNumericalRuntime())
    import asr18_DEV_weight_only_mapper_v1 as transport
    from bpe_streaming_checkpoint_semantics_v2 import select
    output = Path(output).resolve()
    assert output.is_relative_to(Path(r'E:\gpu_training_scratchpad').resolve()) and not output.exists()
    source = Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1\feature_window_native20_01\after_update20.state.json')
    binding = record(source)
    assert binding['sha256'] == 'a2d9c869f23d93b776b93b290cb432ea18e1138c901594ad614b837b3d42d318'
    document = json.loads(source.read_text(encoding='utf-8'))
    plan = transport.selection_plan(document); checks = []
    def check(value, name):
        assert value, name
        checks.append(name)
    check(plan['selected_weight_count'] == 207, 'All207 actual named weight descriptors selected')
    check(len(plan['value']['weights']) == len(set(plan['value']['weights'].values())) == 207,
          'Distinct full named weight indices, no partial weights')
    check(plan['selected_tensor_count'] == 207 + 1 + len(plan['value']['cuda_rng']), 'Only207weights plus Torch and all CUDA RNG tensors selected')
    check(plan['excluded_optimizer_tensor_count'] > 600 and plan['selected_optimizer_tensor_indices'] == [],
          'Every saved Adam tensor descriptor excluded from selection')
    check(set(plan['value']['optimizer']) == {'updates', 'clocks'} and plan['value']['optimizer']['updates'] == 20
          and plan['value']['optimizer']['clocks'] == [64270, 2020, 62270], 'Optimizer scalars only, exact actual clocks')
    check(plan['format_view']['diagnostic_only'] is True and plan['full_run_checkpoint_admitted'] is False,
          'Historical transport format compatibility never admits diagnostic20 for fullrun DEV')
    check(plan['actual_tensors_loaded'] is False, 'Metadata plan constructs no tensor views')
    for name, index in plan['value']['weights'].items():
        descriptor = plan['selected_entries'][index]
        check(descriptor == document['tensors'][index] and descriptor['dtype'] == 'float32'
              and len(descriptor['sha256']) == 64, 'Actual named shape/byte/hash preserved: ' + name)
    check(plan['value']['python_rng'][0] == 3 and len(plan['value']['numpy_rng'][1]) == 624,
          'Exact saved Python/NumPy scalar RNG metadata retained without RNG advance')
    check(plan['value']['sampler'] and record(source) == binding, 'Actual sampler metadata and immutable checkpoint remain unchanged')
    def rejected(mutate, name):
        changed = deepcopy(document); mutate(changed)
        try: transport.selection_plan(changed)
        except (ValueError, KeyError, TypeError): checks.append(name); return
        raise AssertionError('Unexpected acceptance: ' + name)
    first = next(iter(plan['value']['weights'].values()))
    rejected(lambda d: d['tensors'][first].update(dtype='float16'), 'Wrong weight dtype rejected')
    rejected(lambda d: d['tensors'][first].update(bytes=d['tensors'][first]['bytes'] - 4), 'Wrong weight extent rejected')
    rejected(lambda d: select(d['tree'], 'optimizer', 'updates').__setitem__(1, True),
             'Boolean optimizer scalar update cannot pass selection')
    def overlap(d):
        optimizer = select(d['tree'], 'optimizer')
        def visit(node):
            if node[0] == 'tensor': node[1] = first; return True
            if node[0] in ('tuple', 'list'):
                return any(visit(child) for child in node[1])
            if node[0] == 'dict':
                return any(visit(child) for pair in node[1] for child in pair)
            return False
        assert visit(optimizer)
    rejected(overlap, 'Optimizer/weight tensor index alias rejected')
    check(not any(name.split('.')[0] in {'numpy', 'torch', 'scipy', 'transformers', 'soundfile', 'av'} for name in sys.modules),
          'Complete metadata fixture imports no ML/audio runtime')
    output.mkdir()
    report = {'kind': 'ASR18_DEV_actual_saved_descriptor_selection_controls_v1', 'passed': True,
              'check_count': len(checks), 'checks': checks, 'source': record(__file__), 'mapper': record(transport.__file__),
              'actual_saved_metadata': binding, 'selected_weight_count': 207,
              'selected_tensor_count': plan['selected_tensor_count'], 'excluded_optimizer_tensor_count': plan['excluded_optimizer_tensor_count'],
              'actual_payloads_opened': False, 'actual_tensor_views_or_optimizer_objects_constructed': 0,
              'actual_GPU_audio_DEV_or_model_job': False, 'full_run_DEV_or_runtime_ready': False}
    (output / 'report.json').write_text(json.dumps(report, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'passed': True, 'checks': len(checks), 'metadata_only': True}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True)
    main(parser.parse_args().output)
