"""Tiny CPU semantic prototype; no ASR model, candidate or source state is read.

Fresh AdamW owns full embedding/classifier tensors. Gradients for old lexical,
blank, six glyph and duration rows are masked BEFORE global clipping. A frozen
shared decoder remains differentiable with respect to new embedding inputs.
This does not define the eventual unfreeze/source-optimizer policy.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = ROOT / "outputs/parakeet_turkish/BPE_newrows_fresh_Adam_CPU_prototype_v1.json"


def below_normal_priority():
    if os.name != "nt":
        return {"platform": os.name, "applied": False, "reason": "Windows-only priority API"}
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
    kernel.GetPriorityClass.argtypes = (ctypes.c_void_p,)
    kernel.GetPriorityClass.restype = ctypes.c_uint32
    process = kernel.GetCurrentProcess()
    if not kernel.SetPriorityClass(process, 0x4000):
        raise ctypes.WinError(ctypes.get_last_error())
    actual = kernel.GetPriorityClass(process)
    if actual != 0x4000:
        raise RuntimeError("BelowNormal priority was not applied")
    return {"platform": "nt", "applied": True, "actual_priority_class": actual,
            "name": "BELOW_NORMAL_PRIORITY_CLASS"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        raise ValueError("Use a NEW report path within this workspace")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    priority = below_normal_priority()  # Before the only ML import.
    started = time.perf_counter()
    import torch
    import torch.nn.functional as F

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261011)
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU-only prototype must not initialize CUDA")

    # Tiny analogue of the real prefix/interior blank/six glyph/new BPE/duration geometry.
    old, vocab, dim, blank, duration_count = 11, 14, 8, 4, 5
    embedding = torch.nn.Embedding(vocab, dim, padding_idx=blank, dtype=torch.float64)
    shared_decoder = torch.nn.Linear(dim, dim, dtype=torch.float64).requires_grad_(False)
    classifier = torch.nn.Linear(dim, vocab + duration_count, dtype=torch.float64)
    with torch.no_grad():
        # Toy donor averages and the same relative -2 bias convention; no real rows.
        for index, donors in zip(range(old, vocab), ([0, 1], [1, 2], [2, 3])):
            embedding.weight[index].copy_(embedding.weight[donors].mean(0))
            classifier.weight[index].copy_(classifier.weight[donors].mean(0))
            classifier.bias[index].copy_(classifier.bias[donors].mean() - 2.)
    parameters = [embedding.weight, classifier.weight, classifier.bias]
    optimizer = torch.optim.AdamW(parameters, lr=.03, weight_decay=0., betas=(.9, .999))
    assert len(optimizer.state) == 0
    before = [p.detach().clone() for p in parameters]
    shared_before = {name: p.detach().clone() for name, p in shared_decoder.named_parameters()}
    fixed_old_inputs = torch.tensor([0, 1, blank, 5, 10], dtype=torch.long)
    new_inputs = torch.tensor([11, 12, 13, 11, 12, 13], dtype=torch.long)
    targets = torch.tensor([11, 12, 13, 12, 13, 11], dtype=torch.long)

    def logits(indices):
        return classifier(torch.tanh(shared_decoder(embedding(indices))))

    def frozen_equal():
        return (torch.equal(embedding.weight[:old], before[0][:old])
                and torch.equal(classifier.weight[:old], before[1][:old])
                and torch.equal(classifier.bias[:old], before[2][:old])
                and torch.equal(classifier.weight[vocab:], before[1][vocab:])
                and torch.equal(classifier.bias[vocab:], before[2][vocab:])
                and all(torch.equal(p, shared_before[name])
                        for name, p in shared_decoder.named_parameters()))

    with torch.no_grad():
        initial_fixed_logits = logits(fixed_old_inputs).clone()
        teacher_old = initial_fixed_logits[:, :old].softmax(-1)
        teacher_full = torch.cat([teacher_old, teacher_old.new_zeros(len(fixed_old_inputs), vocab-old)], -1)
        initial_new_mass = initial_fixed_logits[:, :vocab].softmax(-1)[:, old:].sum(-1)

    updates = []
    peak_parameter_elements = sum(p.numel() for p in parameters)
    for step in range(1, 21):
        optimizer.zero_grad(set_to_none=True)
        training_logits = logits(new_inputs)[:, :vocab]
        fixed_logits = logits(fixed_old_inputs)
        # The teacher target is explicitly full student-vocabulary, with zero new mass.
        teacher_kl = F.kl_div(fixed_logits[:, :vocab].log_softmax(-1), teacher_full, reduction="batchmean")
        loss = F.cross_entropy(training_logits, targets) + .05 * teacher_kl
        assert torch.isfinite(loss) and torch.isfinite(teacher_kl)
        loss.backward()
        unmasked_frozen_gradient_norm = float(torch.cat([
            embedding.weight.grad[:old].flatten(), classifier.weight.grad[:old].flatten(),
            classifier.bias.grad[:old].flatten(), classifier.weight.grad[vocab:].flatten(),
            classifier.bias.grad[vocab:].flatten()]).norm())
        new_embedding_grad_norm = float(embedding.weight.grad[old:vocab].norm())
        assert new_embedding_grad_norm > 0.  # Through the frozen decoder, not a detached cache.
        assert all(p.grad is None for p in shared_decoder.parameters())
        assert torch.equal(fixed_logits[:, :old], initial_fixed_logits[:, :old])
        assert torch.equal(fixed_logits[:, vocab:], initial_fixed_logits[:, vocab:])
        with torch.no_grad():
            embedding.weight.grad[:old].zero_()
            classifier.weight.grad[:old].zero_(); classifier.bias.grad[:old].zero_()
            classifier.weight.grad[vocab:].zero_(); classifier.bias.grad[vocab:].zero_()
        # Mask precedes clipping, so protected gradients do not consume the norm budget.
        norm_before_clip = torch.nn.utils.clip_grad_norm_(parameters, .15, error_if_nonfinite=True)
        masked_old_gradients_zero = (torch.count_nonzero(embedding.weight.grad[:old]) == 0
            and torch.count_nonzero(classifier.weight.grad[:old]) == 0
            and torch.count_nonzero(classifier.bias.grad[:old]) == 0
            and torch.count_nonzero(classifier.weight.grad[vocab:]) == 0
            and torch.count_nonzero(classifier.bias.grad[vocab:]) == 0)
        assert masked_old_gradients_zero
        optimizer.step()
        assert frozen_equal()
        assert all(torch.isfinite(p).all() for p in parameters)
        with torch.no_grad():
            now = logits(fixed_old_inputs)
            assert torch.equal(now[:, :old], initial_fixed_logits[:, :old])
            assert torch.equal(now[:, vocab:], initial_fixed_logits[:, vocab:])
            student = now[:, :vocab].softmax(-1)
            new_mass = student[:, old:].sum(-1)
            expected_old_probs = teacher_old * (1-new_mass).unsqueeze(-1)
            denominator_identity_error = float((student[:, :old]-expected_old_probs).abs().max())
            conditional_old_error = float((now[:, :old].softmax(-1)-teacher_old).abs().max())
            assert denominator_identity_error < 1e-12 and conditional_old_error == 0.
        updates.append({"step": step, "toy_cross_entropy_plus_KL": float(loss.detach()),
            "teacher_full_vocab_KL": float(teacher_kl.detach()),
            "unmasked_frozen_row_gradient_norm": unmasked_frozen_gradient_norm,
            "new_embedding_gradient_norm_through_frozen_decoder": new_embedding_grad_norm,
            "global_norm_after_mask_before_clip": float(norm_before_clip),
            "mask_before_clip": True, "frozen_parameters_bitwise_same": True,
            "fixed_old_prefix_raw_logits_bitwise_same": True,
            "old_prefix_probability_denominator_identity_max_error": denominator_identity_error,
            "conditional_old_prefix_probability_max_error": conditional_old_error,
            "mean_new_softmax_mass_on_fixed_old_prefix": float(new_mass.mean())})

    new_deltas = {name: float((p.detach()[old:vocab]-base[old:vocab]).norm())
                  for name, p, base in zip(("embedding", "classifier_weight", "classifier_bias"), parameters, before)}
    assert all(value > 0 for value in new_deltas.values())
    for p in parameters:
        state = optimizer.state[p]
        assert state["step"].item() == 20
        for moment in ("exp_avg", "exp_avg_sq"):
            assert torch.count_nonzero(state[moment][:old]) == 0
            if p is not embedding.weight:
                assert torch.count_nonzero(state[moment][vocab:]) == 0

    # Negative control: zero gradients do NOT freeze rows with existing Adam moments.
    control = torch.nn.Parameter(torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float64))
    control_adam = torch.optim.AdamW([control], lr=.03, weight_decay=0.)
    control.grad = torch.tensor([[1., -1.], [0., 0.]], dtype=torch.float64)
    control_adam.step()
    after_nonzero_step = control.detach().clone()
    seeded_moment = control_adam.state[control]["exp_avg"].detach().clone()
    control.grad = torch.zeros_like(control)
    control_adam.step()
    old_drift = float((control[0].detach()-after_nonzero_step[0]).abs().max())
    assert old_drift > 0 and torch.equal(control[1], after_nonzero_step[1])
    assert torch.count_nonzero(seeded_moment[0]) > 0
    assert not torch.cuda.is_initialized()

    result = {"schema": 1, "kind": "tiny_fresh_AdamW_new_vocab_rows_CPU_semantic_prototype_v1",
        "passed": True, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Tiny CPU toy only; no real ASR/tokenizer/model/checkpoint/data/runtime reads or activation",
        "prototype_path": str(Path(__file__).resolve()),
        "prototype_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {"CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
            "CUDA_initialized": False, "device": "cpu", "torch_version": torch.__version__,
            "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
            "priority": priority, "elapsed_seconds_including_Torch_import": time.perf_counter()-started},
        "toy_geometry": {"old_ordinary_lexical_rows": [0,4], "blank_row": blank,
            "old_six_glyph_rows": [5,11], "new_rows": [11,14], "duration_rows": [14,19],
            "embedding_shape": list(embedding.weight.shape), "classifier_weight_shape": list(classifier.weight.shape),
            "classifier_bias_shape": list(classifier.bias.shape), "optimized_parameter_elements": peak_parameter_elements,
            "dtype": "float64", "updates": 20},
        "optimizer": {"class": "AdamW", "fresh_initial_state_empty": True, "full_tensor_parameters": 3,
            "weight_decay": 0., "lr": .03, "max_grad_norm": .15, "source_optimizer_loaded": False,
            "protected_row_moments_remain_exact_zero": True},
        "verification": {"frozen_parameters_bitwise_same_all20_updates": True,
            "new_embeddings_receive_nonzero_gradients_through_frozen_shared_decoder": True,
            "new_rows_all_three_tensors_have_nonzero_finite_deltas": True,
            "mask_applied_before_clipping_all_updates": True,
            "fixed_old_prefix_old_logits_and_duration_logits_bitwise_same": True,
            "old_prefix_score_change_explained_only_by_new_softmax_mass": True,
            "old_conditional_prefix_distribution_exact": True,
            "all_teacher_full_vocab_KL_values_finite": True},
        "new_row_delta_norms": new_deltas,
        "initial_mean_new_softmax_mass_on_fixed_old_prefix": float(initial_new_mass.mean()),
        "final_mean_new_softmax_mass_on_fixed_old_prefix": updates[-1]["mean_new_softmax_mass_on_fixed_old_prefix"],
        "negative_control": {"actual_AdamW_nonzero_moment_then_zero_gradient_step": True,
            "weight_decay": 0., "old_row_max_drift_with_zero_gradient": old_drift,
            "untouched_zero_moment_row_bitwise_same": True,
            "conclusion": "Source Adam moments must remain separate/dormant; masking zero gradients alone cannot freeze loaded old moments"},
        "updates": updates,
        "limits": ["Not evidence of real ASR/nativeTDT learning or candidate readiness",
            "Old-prefix denominator-only assertion uses fixed OLD input IDs; contexts containing newly learned embeddings may change old logits",
            "Fresh optimizer and weight_decay0 are necessary premises; eventual source-optimizer merge/unfreeze policy is not solved here",
            "No GPU device query, candidate activation, saved fake run or live source changes"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=True, indent=2); stream.write("\n")
    print(json.dumps({"path": str(output), "passed": True, "updates": 20,
        "new_row_delta_norms": new_deltas, "negative_control_old_row_drift": old_drift,
        "elapsed_seconds": result["environment"]["elapsed_seconds_including_Torch_import"]}))


if __name__ == "__main__":
    main()
