"""Tiny real CPU checkpoint/GraphTask controls; never full294/native evidence.

The actual cache3 hooks use an explicitly disclosed CPU frozen-parameter clone.
CPU2.7.1 cannot prove a CUDA engine callback thread; that remains the sole GPU
owner's exact differential. No arrays, model assets or runtime files are changed.
"""
import argparse
import os
from pathlib import Path
import random
import sys
import threading
import time

PROJECT = Path(__file__).resolve().parents[2]
ROOT = Path(r'E:\gpu_training_scratchpad')
sys.path.insert(0, str(PROJECT / 'outputs/parakeet_turkish'))
sys.path.insert(0, str(PROJECT / 'work/research'))
import test_ASR18_frozen_encoder_cache_hooks_v3 as base
import asr18_frozen_encoder_cuda_cache_v3 as cache


def tiny():
    model, linear, entry, state, before, after = base.toy(cache)
    model._ASR18_frozen_cuda_cache = state
    linear.register_forward_pre_hook(before)
    linear.register_forward_hook(after, always_call=True)
    return model, linear, entry, state


def root(linear):
    from torch.utils.checkpoint import checkpoint
    x = base.torch.arange(16, dtype=base.torch.float32).reshape(2, 8).to(base.torch.bfloat16).requires_grad_()
    y = checkpoint(linear, x, use_reentrant=False)
    return x, y, y.sum()


def rejected(name, call):
    try:
        call()
    except (ValueError, RuntimeError) as error:
        base.check(name, True, actual_exception=type(error).__name__)
    else:
        raise ValueError('Expected explicit fail-closed rejection: ' + name)


def clean_failure(state, entry, linear):
    return (state.poisoned and state.backward is None and not entry['busy']
            and linear.weight is entry['cpu_parameter'] and state.metrics['inflight_matrices'] == 0
            and state.model._diagnostic_frozen_metrics['logical_current_matrix_bytes'] == 0)


def positive():
    torch, np = base.torch, base.np
    baseline = base.arm()
    model, linear, entry, state = tiny()
    random.seed(812); np.random.seed(813); torch.manual_seed(814)
    before_rng = base.rng()
    x, y, loss = root(linear)
    pointer, version = x.data_ptr(), x._version
    seen, gradients = [], []
    def observe(current, inputs, output):
        seen.append({'graph': torch._C._current_graph_task_id(),
            'thread': threading.get_ident(), 'output_None': output is None,
            'exception': type(sys.exc_info()[1]).__name__,
            'original_CPU_parameter_restored': current.weight is entry['cpu_parameter']})
    observer = linear.register_forward_hook(observe, always_call=True)
    # Surround the candidate root hook to prove it returns the same object.
    first = loss.register_hook(lambda gradient: gradients.append(id(gradient)) or gradient)
    original_backward = torch.Tensor.backward
    with cache.backward_scope(model, loss, torch=torch):
        second = loss.register_hook(lambda gradient: gradients.append(id(gradient)) or gradient)
        original_backward(loss)
    first.remove(); second.remove(); observer.remove()
    base.check('Actual CPU nonreentrant recompute returns exact full BF16 output/inputgradient',
        torch.equal(y.detach().view(torch.uint8), baseline['output'].view(torch.uint8))
        and torch.equal(x.grad.detach().view(torch.uint8), baseline['gradient'].view(torch.uint8)))
    base.check('Root Tensor hook returns identical gradient object', len(gradients) == 2 and gradients[0] == gradients[1])
    base.check('Actual root GraphTask/callback-thread owns all recompute hooks', seen and all(
        v['graph'] == state.last_backward['graph'] >= 0
        and v['thread'] == state.last_backward['callback_thread'] for v in seen))
    base.check('Actual checkpoint early-stop remains accepted with CPU restoration', not state.poisoned
        and any(v['output_None'] and v['exception'] == '_StopRecomputationError' for v in seen)
        and all(v['original_CPU_parameter_restored'] for v in seen))
    base.check('Scope ends with no temporary root hook/inflight matrix or changed input',
        state.backward is None and state.metrics['backward_scopes_completed'] == 1
        and not loss._backward_hooks and not entry['busy'] and linear.weight is entry['cpu_parameter']
        and state.metrics['inflight_matrices'] == 0 and x.data_ptr() == pointer and x._version == version)
    base.check('All available Python NumPy Torch emptyCUDA RNG exact', before_rng == base.rng())
    # A second actual graph gets a new explicitly bound ID; state stays bounded.
    x2, y2, loss2 = root(linear)
    old_graph = state.last_backward['graph']
    with cache.backward_scope(model, loss2, torch=torch):
        original_backward(loss2)
    base.check('Consecutive scopes bind distinct actual GraphTasks without history growth',
        state.metrics['backward_scopes_completed'] == 2 and state.last_backward['graph'] != old_graph
        and state.backward is None and not state.poisoned)
    return {'actual_CPU_recompute_events': seen, 'last_scope': state.last_backward,
            'root_gradient_identity': True, 'CUDA_worker_thread_observed': False}


def negatives():
    torch = base.torch
    original_backward = torch.Tensor.backward
    model, linear, entry, state = tiny()
    _, _, loss = root(linear)
    rejected('Actual autograd recompute without owner scope rejected', lambda: original_backward(loss))
    base.check('Missing scope poisons and restores CPU parameter', clean_failure(state, entry, linear))
    model, linear, entry, state = tiny()
    _, _, loss = root(linear)
    def no_backward():
        with cache.backward_scope(model, loss, torch=torch):
            pass
    rejected('Scope without actual Tensor.backward cannot complete', no_backward)
    base.check('No-backward failure removes temporary root hook', clean_failure(state, entry, linear) and not loss._backward_hooks)
    model, linear, entry, state = tiny()
    _, _, loss = root(linear)
    def nested():
        with cache.backward_scope(model, loss, torch=torch):
            with cache.backward_scope(model, loss, torch=torch):
                original_backward(loss)
    rejected('Nested owner backward scope rejected', nested)
    base.check('Nested failure preserves fail-stop and CPU identity', clean_failure(state, entry, linear))
    model, linear, entry, state = tiny()
    _, _, loss = root(linear)
    def main_forward_during_backward():
        with cache.backward_scope(model, loss, torch=torch):
            linear(torch.ones(2, 8, dtype=torch.bfloat16))
    rejected('Main forward inside backward scope is not an autograd permission', main_forward_during_backward)
    base.check('Main-forward misuse poisons without borrowing a matrix', clean_failure(state, entry, linear))
    model, linear, entry, state = tiny()
    errors = []
    def foreign_forward():
        try: linear(torch.ones(2, 8, dtype=torch.bfloat16))
        except (ValueError, RuntimeError) as error: errors.append(type(error).__name__)
    foreign = threading.Thread(target=foreign_forward, name='Dummy-explicitly-untrusted')
    foreign.start(); foreign.join(timeout=2)
    base.check('Named Dummy foreign thread is rejected rather than trusted', not foreign.is_alive() and errors
               and clean_failure(state, entry, linear))
    model, linear, entry, state = tiny()
    _, _, owner_loss = root(linear)
    _, _, other_loss = root(linear)
    foreign_errors, foreign_graphs = [], []
    # Real unrelated autograd graph while the actual owning root is bound.
    def other_root(gradient):
        foreign_graphs.append(torch._C._current_graph_task_id())
        return gradient
    other_handle = other_loss.register_hook(other_root)
    def start_foreign_after_owning_root(gradient):
        def run():
            try: original_backward(other_loss)
            except (ValueError, RuntimeError) as error: foreign_errors.append(type(error).__name__)
        thread = threading.Thread(target=run)
        thread.start(); thread.join(timeout=2)
        base.require(not thread.is_alive(), 'Tiny foreign-graph test must actually finish')
        return gradient
    def wrong_graph():
        with cache.backward_scope(model, owner_loss, torch=torch):
            probe = owner_loss.register_hook(start_foreign_after_owning_root)
            try: original_backward(owner_loss)
            finally: probe.remove()
    rejected('Real foreign autograd graph cannot use owning root permission', wrong_graph)
    other_handle.remove()
    base.check('Foreign real graph has actual nonnegative distinct GraphTask and rejects',
        foreign_errors and foreign_graphs and all(type(g) is int and g >= 0 for g in foreign_graphs)
        and clean_failure(state, entry, linear))
    model, linear, entry, state = tiny()
    _, _, loss = root(linear)
    forward = linear.forward
    def true_recompute_failure(x):
        if torch._C._current_graph_task_id() >= 0:
            raise RuntimeError('PRIVATE true recompute failure')
        return forward(x)
    linear.forward = true_recompute_failure
    def fail_backward():
        with cache.backward_scope(model, loss, torch=torch): original_backward(loss)
    rejected('True recompute exception still poisons rather than early-stop exemption', fail_backward)
    base.check('True recompute failure restores matrix/counts and removes scope hook',
        clean_failure(state, entry, linear) and not loss._backward_hooks)
    model, linear, entry, state = tiny()
    _, _, loss = root(linear)
    state.pid = -1
    rejected('Process identity mismatch rejected before backward', lambda: state._hook_owner(torch))
    base.check('Process mismatch never borrows CPU Parameter', linear.weight is entry['cpu_parameter'] and not entry['busy'])
    return {'real_foreign_GraphTask_ids': foreign_graphs, 'no_thread_name_exemption': True}


def main(output):
    output = Path(output).resolve()
    base.require(output.is_relative_to(ROOT) and not output.exists(), 'Fresh E scope fixture')
    base.require(os.environ['CUDA_VISIBLE_DEVICES'] == '-1', 'CUDA masked')
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        base.require(os.environ[name] == '1', 'Four thread caps1')
    admissions = []
    for _ in range(3):
        value = base.guard(); s = value['system_memory']
        base.require(s['commit_used_bytes'] + 576_000_000 < .9 * s['commit_limit_bytes']
                     and s['commit_available_bytes'] - 576_000_000 >= 2_000_000_000, 'Whole576MB fresh admission')
        admissions.append(value); time.sleep(.1)
    sources = {'candidate': base.pin(cache.__file__), 'fixture': base.pin(__file__), 'CPU_helper': base.pin(base.__file__)}
    sys.path.insert(0, str(base.NUMPY_SITE))
    import numpy as np
    import torch
    base.torch, base.np = torch, np
    base.require(torch.__version__ == '2.7.1+cpu' and not torch.cuda.is_available()
                 and not torch.cuda.is_initialized() and Path(torch.__file__).resolve().is_relative_to(base.CPU_SITE)
                 and Path(np.__file__).resolve().is_relative_to(base.NUMPY_SITE), 'Exact existing E CPU Torch/NumPy; no native CUDA')
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    output.mkdir()
    positive_details = positive()
    negative_details = negatives()
    base.check('Candidate/source helpers unchanged', base.pin(cache.__file__) == sources['candidate']
               and base.pin(base.__file__) == sources['CPU_helper'])
    base.check('CPU fixture never initialized CUDA', not torch.cuda.is_initialized() and base.rng()['CUDA'] == [])
    report = {'kind': 'ASR18_cache3_CPU_root_owned_GraphTask_protocol_controls_v1', 'passed': True,
        'producer': base.identity(), 'worker_source': sources['fixture'],
        'parent_source': base.pin(PROJECT / 'work/research/ASR18_bounded_CPU_phase_v3.py'),
        'sources': sources, 'checks': base.checks, 'check_count': len(base.checks),
        'positive': positive_details, 'negative': negative_details,
        'CPU_standin_for_CUDA_weight_storage': True, 'native_GPU_callback_thread_validated': False,
        'full294_install_or_all207_equivalence_claimed': False, 'production_enabled': False,
        'performance_claimed': False, 'torch_version': torch.__version__, 'numpy_version': np.__version__,
        'private_peak_bytes': base.peak, 'fresh_three_admissions': admissions,
        'graph_API_sources': [base.pin(base.CPU_SITE / 'torch/utils/checkpoint.py'),
                              base.pin(base.CPU_SITE / 'torch/autograd/graph.py'),
                              base.pin(base.CPU_SITE / 'torch/include/torch/csrc/autograd/graph_task.h')]}
    report['wall_seconds'] = time.monotonic() - started
    base.write_json(output / 'report.json', report)


if __name__ == '__main__':
    started = time.monotonic()
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True)
    main(parser.parse_args().output)
