"""INERT stock RNNTLoss activation-checkpoint hook; no native math replacement.

NeMo typed forward calls use named arguments. Importing this file is stdlib-only. Installation requires a root-owned runtime
and is not authorized for the current active trial. GPU loss/gradient, Python
RNG, reduction-state and whole-update memory/throughput proof precede adoption.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import hashlib
from pathlib import Path
import random
import types


@contextmanager
def replay_python_rng(state):
    """Replay native omega sampling without consuming ambient RNG twice."""
    ambient = random.getstate()
    random.setstate(state)
    try:
        yield
    finally:
        random.setstate(ambient)


def install_stock_nemo_loss_checkpointing(model, torch_module=None):
    """Wrap only model.loss.forward; return removable hook plus provenance.

    Checkpoint the stock loss before its FP32 cast. No joint, model training_step,
    normalization, vocab, optimizer, or official CUDA kernel implementation is
    changed. This preserves stochastic omega selection during recomputation.
    Torch compile and concurrent calls sharing this loss module are unsupported.
    """
    if torch_module is None:
        import torch as torch_module
    from torch.utils.checkpoint import checkpoint
    loss = model.loss
    backend = loss._loss
    if type(backend).__name__ != "TDTLossNumba" or not hasattr(backend, "omega"):
        raise ValueError("Expected the official TDTLossNumba backend")
    if getattr(loss, "_stock_checkpoint_hook_v2", None) is not None:
        raise ValueError("Checkpoint hook is already installed")
    if torch_module.compiler.is_compiling():
        raise ValueError("This eager native-loss hook has not been validated with compile")
    original = loss.forward
    statistics = {"checkpointed_forward_calls": 0, "recomputed_forward_calls": 0,
                  "uncheckpointed_calls": 0}

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        if not torch_module.is_grad_enabled() or not log_probs.requires_grad:
            statistics["uncheckpointed_calls"] += 1
            return original(log_probs=log_probs, targets=targets, input_lengths=input_lengths, target_lengths=target_lengths)
        python_state = random.getstate()
        reduction = self.reduction
        backend_reduction = self._loss.reduction
        phase = {"recompute": False}

        @contextmanager
        def recompute_context():
            phase["recompute"] = True
            try:
                with replay_python_rng(python_state):
                    yield
            finally:
                phase["recompute"] = False

        def evaluate(acts, labels, frame_lengths, label_lengths):
            # Fused RNNTJoint changes reduction toNone during each forward;
            # backward occurs after it restores mean_volume. Capture that exact
            # invocation's state rather than reading mutable state on replay.
            ambient_outer = self.reduction
            ambient_inner = self._loss.reduction
            self.reduction, self._loss.reduction = reduction, backend_reduction
            if phase["recompute"]:
                statistics["recomputed_forward_calls"] += 1
            try:
                return original(log_probs=acts, targets=labels, input_lengths=frame_lengths, target_lengths=label_lengths)
            finally:
                # Nonreentrant early-stop can interrupt the official function
                # after save_for_backward, before its normal state restoration.
                self.reduction, self._loss.reduction = ambient_outer, ambient_inner

        statistics["checkpointed_forward_calls"] += 1
        return checkpoint(evaluate, log_probs, targets, input_lengths, target_lengths,
                          use_reentrant=False, preserve_rng_state=True,
                          determinism_check="default",
                          context_fn=lambda: (nullcontext(), recompute_context()))

    loss.forward = types.MethodType(forward, loss)
    loss._stock_checkpoint_hook_v2 = original

    def remove():
        if getattr(loss, "_stock_checkpoint_hook_v2", None) is not original:
            raise ValueError("Loss hook ownership changed")
        loss.forward = original
        del loss._stock_checkpoint_hook_v2

    identity = {"kind": "stock_nemo_RNNTLoss_nonreentrant_checkpoint_v2",
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "original_loss_class": type(loss).__module__ + "." + type(loss).__name__,
                "original_backend_class": type(backend).__module__ + "." + type(backend).__name__,
                "sigma": backend.sigma, "omega": backend.omega,
                "durations": list(backend.durations), "blank": backend.blank,
                "native_math_or_training_step_replaced": False,
                "checkpoint_boundary": "stock RNNTLoss.forward before FP32 cast",
                "preserve_Torch_CPU_CUDA_rng": True,
                "replay_Python_rng_and_restore_ambient": True,
                "restore_outer_and_backend_reduction_even_early_stop": True,
                "GPU_or_whole_update_validation_complete": False,
                "statistics": statistics}
    return {"identity": identity, "remove": remove}
