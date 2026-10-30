"""One synthetic CPU test; no actual ASR/source checkpoint or GPU is read."""
from __future__ import annotations
import argparse
import copy
import ctypes
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
UTILITY = ROOT / 'outputs/parakeet_turkish/bpe_newrows_acquisition_v1.py'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise ValueError('Use a fresh CPU receipt path')
    started = time.monotonic()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    priority = {'platform': os.name, 'requested': 'BELOW_NORMAL_PRIORITY_CLASS'}
    if os.name == 'nt':
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        kernel.SetPriorityClass.restype = ctypes.c_int
        kernel.GetPriorityClass.argtypes = [ctypes.c_void_p]
        kernel.GetPriorityClass.restype = ctypes.c_uint
        handle = kernel.GetCurrentProcess()
        priority.update(applied=bool(kernel.SetPriorityClass(handle, 0x4000)),
                        observed_class=int(kernel.GetPriorityClass(handle)))
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(101)
    spec = importlib.util.spec_from_file_location('bpe_rows_cpu_fixture', UTILITY)
    utility = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utility)
    checks, rejected = [], []

    def check(condition, name):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    def reject(state, name, stage):
        before = {n: raw(p) for n, p in stage.model.named_parameters()}
        try:
            stage.load_phase_state(state)
        except (ValueError, TypeError, KeyError):
            check(all(torch.equal(before[n], raw(p)) for n, p in stage.model.named_parameters()),
                  name + '_no_weight_copy')
            rejected.append(name)
        else:
            raise AssertionError('Corruption accepted: ' + name)

    def raw(tensor):
        return tensor.detach().contiguous().view(torch.uint8).reshape(tuple(tensor.shape) + (tensor.element_size(),)).clone()

    class SyntheticModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = torch.nn.Module()
            self.decoder.embedding = torch.nn.Embedding(8711, 3)
            self.decoder.shared = torch.nn.Linear(3, 3)
            self.joint = torch.nn.Module()
            self.joint.head = torch.nn.Linear(3, 8716)
            self.encoder = torch.nn.Linear(3, 3)
            self.learned_frozen = torch.nn.ParameterList([
                torch.nn.Parameter(torch.tensor([float(i + 1)])) for i in range(200)])
            self.config = SimpleNamespace(vocab_size=8711, blank_token_id=8192, durations=(0, 1, 2, 3, 4))
            with torch.no_grad():
                self.joint.head.bias[3] = -0.0
                self.joint.head.bias[8712] = -0.0

    def backward(model, stage):
        stage.zero_grad()
        with torch.no_grad():
            encoded = model.encoder(torch.ones(5, 3))
        ids = torch.tensor([4, 8195, 8199, 8300, 8710])
        hidden = model.decoder.shared(encoded + model.decoder.embedding(ids))
        # Synthetic dense table gradients, not a native ASR/TDT/quality proof.
        loss = model.joint.head(hidden).sum() + hidden.square().sum()
        loss.backward()
        return loss

    with tempfile.TemporaryDirectory(prefix='bpe_rows_cpu_', dir=ROOT / 'work') as temp:
        parked = Path(temp) / 'opaque_source_artifact.bin'
        parked.write_bytes(b'CPU fixture only: opaque source207 artifact; never deserialized.\n')
        parked_record = utility.artifact_record(parked)
        metadata = {'rank': 8, 'alpha': 16, 'synthetic_names_not_real_LoRA': True}
        model = SyntheticModel()
        names = tuple(n for n, _ in model.named_parameters())
        check(len(names) == 207, 'synthetic_explicit207_name_contract')
        stage = utility.BPEOnlyAcquisition(model, names, parked_record,
                source_lora_metadata=metadata, torch_module=torch)
        check(not stage.optimizer.state, 'fresh_Adam_has_no_source_moments')
        initial = stage.state_dict(runtime_state={'scope': 'opaque synthetic runtime'})
        stage.load_phase_state(initial)
        check(stage.optimizer_updates_completed == 0 and not stage.optimizer.state, 'zero_update_phase_restore')
        original = {n: raw(p) for n, p in model.named_parameters()}
        loss = backward(model, stage)
        check(bool(torch.isfinite(loss)), 'synthetic_forward_backward_finite')
        check(bool(torch.count_nonzero(model.decoder.embedding.weight.grad[:8199])), 'old_embedding_grad_exists_before_mask')
        check(bool(torch.count_nonzero(model.joint.head.weight.grad[8711:])), 'duration_head_grad_exists_before_mask')
        check(bool(torch.count_nonzero(model.joint.head.bias.grad[8711:])), 'duration_bias_grad_exists_before_mask')
        check(bool(torch.count_nonzero(model.decoder.embedding.weight.grad[8199:8711])), 'new_embeddings_differentiate_through_frozen_decoder')
        check(model.decoder.shared.weight.grad is None and model.encoder.weight.grad is None, 'frozen_encoder_decoder_weights_have_no_grad')
        expected = torch.linalg.vector_norm(torch.cat([p.grad[8199:8711].flatten() for p in stage.tables]))
        returned = stage.step(max_norm=1.0)
        check(torch.allclose(returned, expected, rtol=1e-6, atol=1e-6), 'norm_formed_only_after_protected_gradient_mask')
        check(float(torch.linalg.vector_norm(torch.cat([p.grad.flatten() for p in stage.tables]))) <= 1.00001,
              'masked_gradients_clipped')
        for name, parameter in zip(utility.TABLE_NAMES, stage.tables):
            check(torch.equal(original[name][:8199], raw(parameter)[:8199]), name + '_protected_old_raw_bytes')
            check(torch.equal(original[name][8711:], raw(parameter)[8711:]), name + '_protected_duration_raw_bytes')
        check(all(torch.equal(original[n], raw(p)) for n, p in model.named_parameters() if n not in utility.TABLE_NAMES),
              'all204_frozen_learned_tensor_raw_bytes_preserved')
        check(any(not torch.equal(original[n][8199:8711], raw(p)[8199:8711]) for n, p in zip(utility.TABLE_NAMES, stage.tables)),
              'actual_new_row_parameter_update')
        check(len(stage.optimizer.state) == 3 and stage.optimizer_updates_completed == 1, 'only3_Adam_states_one_completed_update')
        saved = stage.state_dict(runtime_state={'sampler': 'opaque fixture cursor', 'counter': 1})
        check(set(saved['weights']) == set(names) and len(saved['weights']) == 207, 'serializer_retains_all207_independent_requires_grad')
        check(saved['parked_source_optimizer']['loaded_into_acquisition_optimizer'] is False,
              'parked_sourceAdam_never_loaded')
        check(utility.artifact_record(parked) == parked_record, 'opaque_parked_source_bytes_unchanged')
        buffer = io.BytesIO()
        torch.save(saved, buffer)
        buffer.seek(0)
        restored = torch.load(buffer, map_location='cpu', weights_only=True)
        restored_model = SyntheticModel()
        resumed = utility.BPEOnlyAcquisition(restored_model, names, parked_record,
                source_lora_metadata=metadata, torch_module=torch)
        runtime = resumed.load_phase_state(restored)
        check(runtime == saved['runtime_state'] and resumed.optimizer_updates_completed == 1, 'serialized_phase_runtime_and_count_restore')
        check(all(torch.equal(raw(restored_model.get_parameter(n)), raw(saved['weights'][n])) for n in names),
              'actual_CPU_serialized_all207_bytes_roundtrip')
        first_id = restored['acquisition_optimizer']['param_groups'][0]['params'][0]
        input_step = restored['acquisition_optimizer']['state'][first_id]['step'].clone()
        backward(restored_model, resumed)
        resumed.step()
        check(torch.equal(input_step, restored['acquisition_optimizer']['state'][first_id]['step']),
              'restored_input_CPU_Adam_clock_not_aliased')
        bad = copy.deepcopy(saved); bad['schema'] = 2
        reject(bad, 'legacy_source_exact_schema', resumed)
        bad = copy.deepcopy(saved); bad['weights'].pop(names[-1])
        reject(bad, 'missing_frozen_learned_weight', resumed)
        bad = copy.deepcopy(saved); bad['acquisition_optimizer']['param_groups'][0]['lr'] *= 2
        reject(bad, 'same_phase_LR_change', resumed)
        bad = copy.deepcopy(saved); bad['acquisition_optimizer']['param_groups'][0]['eps'] *= 2
        reject(bad, 'same_phase_Adam_hyperparameter_change', resumed)
        bad = copy.deepcopy(saved); bad['acquisition_optimizer']['state'].pop(first_id)
        reject(bad, 'missing_Adam_state_after_update', resumed)
        bad = copy.deepcopy(saved); bad['acquisition_optimizer']['state'][first_id]['step'].add_(1)
        reject(bad, 'Adam_clock_mismatch', resumed)
        bad = copy.deepcopy(saved); bad['acquisition_optimizer']['state'][first_id]['exp_avg_sq'][8199, 0] = -1
        reject(bad, 'negative_Adam_variance', resumed)
        bad = copy.deepcopy(saved); bad['acquisition_optimizer']['state'][first_id]['exp_avg'][8192, 0] = 1
        reject(bad, 'protected_row_source_moments', resumed)
        try:
            resumed.unfreeze()
        except ValueError:
            rejected.append('undeclared_unfreeze')
        else:
            raise AssertionError('Unfreeze allowed')
    check(not torch.cuda.is_initialized(), 'CUDA_never_initialized')
    check(os.environ['CUDA_VISIBLE_DEVICES'] == '' and torch.get_num_threads() == 1, 'CPU_only_one_thread')
    check(priority.get('applied') and priority.get('observed_class') == 0x4000 if os.name == 'nt' else True,
          'Windows_below_normal_priority_applied')
    report = {'schema': 1, 'kind': 'BPE_newrows_acquisition_synthetic_CPU_verification_v1', 'passed': True,
              'utility_sha256': hashlib.sha256(UTILITY.read_bytes()).hexdigest(),
              'fixture_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'checks': checks, 'check_count': len(checks), 'rejected_corruptions': rejected,
              'rejected_count': len(rejected), 'priority': priority, 'elapsed_seconds': time.monotonic() - started,
              'torch_version': torch.__version__, 'CUDA_initialized': False,
              'actual_ASR_model_or_source_checkpoint_loaded': False,
              'source207_Adam_contract': 'opaque synthetic artifact only; no actual source state verification',
              'scope': 'Synthetic CPU tables/gradients and phase serialization only; actual nativeTDT/KD, source admission, retention, memory and GPU performance unmeasured'}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
    print(json.dumps({'passed': True, 'checks': len(checks), 'rejected': len(rejected), 'report': str(output)}))


if __name__ == '__main__':
    main()
