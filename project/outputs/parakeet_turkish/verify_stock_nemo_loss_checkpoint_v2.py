"""Root-only bounded GPU comparison of stock NeMo loss versus its checkpoint hook.

Consumes the two saved actual TRAIN joint tensors; no model, audio, CPU loss
recurrence, crop, or altered transcription. --validate-only is stdlib-only.
The exact native class permutation moves HF's interior blank to lexical end.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
INPUT_SHA = "8c3481e742f7df332102f718bc632314cd8869896ef292b28b2e0654bbfe1259"
PINS = {
    "preload_nemo_linker_v1.py": "f8787b0f3346c9f526903ff0c06de4c266f940ca3b0f41c55907cc795e7cc2ba",
    "stock_nemo_loss_checkpoint_v2.py": "f114beb1379af360fa37b9b4e2b3965bf5ba4fcb74707fecd708c7dc33706d74",
    "resource_watchdog_v2.py": "64d1393e1e45cf65dac14f78ac955331c91b4a3f0eafbf7535196e2b61778641",
    "paired_windows_commit_telemetry_v1.py": "2ae9f2cdfc388a294914e000c60b62fe09cddac147700c5762851d626a95a683",
}
NEMO_PINS = {
    "collections/asr/losses/rnnt.py": "afa39c95bf2b4f5d586bc2e169727fd708d0037a61207b2ff16e1ceb89635620",
    "collections/asr/parts/numba/rnnt_loss/rnnt_pytorch.py": "7b5ff7fe789c23a36068d20e2e3fce3d2931be6e55bcebff7efc4d86c399918e",
    "collections/asr/parts/numba/rnnt_loss/utils/cuda_utils/gpu_rnnt.py": "df40a679bbabaeb9bc05f92bb3fbc436b80ef2f4c311564a00391934087c985c",
    "collections/asr/parts/numba/rnnt_loss/utils/cuda_utils/gpu_rnnt_kernel.py": "385de70753b6c1b2a2e5b402a689d816ece82faa846368b3d52cca7e47bb9045",
}
PERMUTATION = list(range(8192)) + list(range(8193, 8199)) + [8192] + list(range(8199, 8204))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def mapping_checks():
    require(len(PERMUTATION) == 8204 and sorted(PERMUTATION) == list(range(8204)), "Class mapping is not bijective")
    inverse = [None] * 8204
    for new, old in enumerate(PERMUTATION):
        inverse[old] = new
    require(inverse[8192] == 8198 and inverse[8193:8199] == list(range(8192, 8198)), "Native blank/TR mapping changed")
    require(PERMUTATION[8199:] == list(range(8199, 8204)), "Duration IDs changed")
    require(all(PERMUTATION[inverse[i]] == i for i in range(8204)), "Inverse mapping failed")
    require(random.Random(31).uniform(0, 1) < .1 <= random.Random(0).uniform(0, 1), "Seeds do not exercise both native omega branches")
    return inverse


def preflight(args):
    require(args.input.resolve().is_relative_to(ROOT) and args.input.is_file(), "Actual workspace input required")
    require(args.input.stat().st_size <= 330_000_000, "Saved actual input exceeds the320MB tensor bound plus metadata")
    require(args.input_sha256 == INPUT_SHA and sha(args.input) == INPUT_SHA, "Actual TRAIN capture hash changed")
    require(args.output.resolve().is_relative_to(ROOT) and not args.output.exists(), "New workspace output required")
    require(0 < args.max_wall_seconds <= 600, "Bounded wall-time required")
    for name, expected in PINS.items():
        require(sha(HERE / name) == expected, "Frozen source changed: " + name)
    native_site = ROOT / "work/nemo-native-env-v1/Lib/site-packages/nemo"
    for name, expected in NEMO_PINS.items():
        require(sha(native_site / name) == expected, "Official installed source changed: " + name)
    mapping_checks()
    return {"code_sha256": sha(__file__), "input_path": str(args.input.resolve()),
            "input_sha256": INPUT_SHA, "input_file_bytes": args.input.stat().st_size,
            "source_sha256": dict(PINS), "official_loss_source_sha256": dict(NEMO_PINS), "permutation": {
                "kind": "bijective_class_order_only_no_audio_or_target_content_change",
                "old_blank": 8192, "native_blank": 8198,
                "old_TR_ids": list(range(8193, 8199)), "native_TR_ids": list(range(8192, 8198)),
                "duration_ids_unchanged": list(range(8199, 8204)),
                "new_index_to_old_index_sha256": hashlib.sha256(json.dumps(PERMUTATION).encode()).hexdigest()},
            "cap_bytes": 8_000_000_000, "native_kernel_executed": False,
            "scope": "Stock loss checkpoint boundary only; no whole-update speed/memory, model, tokenizer or quality equivalence claim"}


def validate_data(data, torch):
    require(data.get("schema") == 1 and data.get("kind") == "actual_fleurs_training_joint_logits", "Actual captured TRAIN input required")
    acts, labels, lengths, targets = (data[k] for k in ("joint_logits", "labels", "frame_lengths", "target_lengths"))
    require(acts.device.type == "cpu" and acts.dtype == torch.float32 and acts.is_contiguous()
            and list(acts.shape) == [2, 128, 29, 8204] and acts.numel() * acts.element_size() <= 320_000_000,
            "Expected exact full captured tensors, not cropped/synthetic logits")
    require(bool(torch.isfinite(acts).all()), "Nonfinite actual logits")
    require(labels.device.type == lengths.device.type == targets.device.type == "cpu"
            and labels.dtype == lengths.dtype == targets.dtype == torch.long
            and list(labels.shape) == [2, 28] and list(lengths.shape) == list(targets.shape) == [2], "Actual label/length dimensions differ")
    require(data["blank_id"] == 8192 and data["durations"] == [0, 1, 2, 3, 4], "Native source geometry differs")
    require(int(lengths.max()) == 128 and int(targets.max()) == 28
            and int(lengths.min()) > 0 and int(targets.min()) > 0, "No padding length/crop changes allowed")
    for i in range(2):
        valid = labels[i, :int(targets[i])]
        require(bool(((valid >= 0) & (valid < 8199) & (valid != 8192)).all()), "Invalid actual lexical target")
    require(bool(((labels >= 0) & (labels < 8199)).all()), "Padded labels outside vocabulary")
    provenance = data["provenance"]
    require(provenance.get("split") == "train" and len(set(provenance["source_ids"])) == 2
            and provenance.get("model_training") is False and provenance.get("dropout_disabled") is True, "Captured actual deterministic TRAIN provenance missing")
    for name in ("checkpoint_sha256", "tokenizer_sha256"):
        require(isinstance(provenance.get(name), str) and len(provenance[name]) == 64, "Capture provenance hash missing")
    return acts, labels, lengths, targets


def gradient_stats(expected_cpu, actual_gpu, torch):
    """One retained reference gradient, streamed1M CPU values for comparison."""
    require(expected_cpu.shape == actual_gpu.shape and expected_cpu.dtype == actual_gpu.dtype, "Joint gradient shape/dtype changed")
    a, b = expected_cpu.flatten(), actual_gpu.detach().flatten()
    norm_a = norm_b = delta2 = dot = maximum = duration2 = 0.
    equal = finite = True
    for start in range(0, a.numel(), 1_000_000):
        left = a[start:start + 1_000_000]
        right = b[start:start + 1_000_000].cpu()
        equal = equal and torch.equal(left, right)
        x, y = left.double(), right.double()
        finite = finite and bool(torch.isfinite(x).all() and torch.isfinite(y).all())
        difference = x - y
        norm_a += float(x.square().sum()); norm_b += float(y.square().sum())
        delta2 += float(difference.square().sum()); dot += float((x * y).sum())
        maximum = max(maximum, float(difference.abs().max()))
    for i in range(actual_gpu.shape[0]):
        duration2 += float(actual_gpu[i, ..., 8199:].float().square().sum())
    return {"all_joint_values": a.numel(), "finite": finite, "bitwise_equal": equal,
            "maximum_absolute_delta": maximum, "relative_l2": math.sqrt(delta2 / max(norm_a, 1e-60)),
            "cosine": dot / max(math.sqrt(norm_a * norm_b), 1e-60),
            "baseline_l2": math.sqrt(norm_a), "duration_gradient_l2": math.sqrt(duration2),
            "comparison_memory_scope": "one CPU reference gradient;1M streamed float64 chunks; no full CPU DP or full float64 copies"}


def gpu_run(args, report):
    # Coherent compiler/linker library loading MUST precede Torch/NeMo import.
    from preload_nemo_linker_v1 import preload
    report["preload"] = preload()
    import torch
    import nemo
    from nemo.collections.asr.losses.rnnt import RNNTLoss
    from stock_nemo_loss_checkpoint_v2 import install_stock_nemo_loss_checkpointing
    from resource_watchdog_v2 import ResourceGuard, sample_memory, worker_process_commit_reason, POLICY
    torch.set_num_threads(2)
    for name, expected in NEMO_PINS.items():
        require(sha(Path(nemo.__file__).resolve().parent / name) == expected, "Actual imported official source differs: " + name)
    require(importlib.metadata.version("nemo-toolkit") == "3.0.0", "Expected measured isolated NeMo3.0")
    device = torch.device("cuda:0")
    total = torch.cuda.get_device_properties(device).total_memory
    require(15_000_000_000 <= total <= 18_000_000_000, "Expected local16GB-class GPU")
    torch.cuda.set_per_process_memory_fraction(report["cap_bytes"] / total, device)
    report["versions"] = {key: importlib.metadata.version(key) for key in ("torch", "nemo-toolkit", "numba", "numba-cuda")}
    report["guard_policy"] = dict(POLICY)
    report["memory_events"] = []
    report["cases"] = []
    guard, started = ResourceGuard(), time.monotonic()

    def memory(phase):
        torch.cuda.synchronize(device)
        sample = sample_memory(os.getpid(), phase)
        reason, evidence = guard.observe(sample["system_memory"], sample["worker_private_memory"],
                                         observed_monotonic=sample["observed_monotonic"])
        reason = reason or worker_process_commit_reason(sample["calling_process_commit_memory"], os.getpid())
        sample.update(resource_reason=reason, resource_guard=evidence,
                      cuda_allocated_bytes=torch.cuda.memory_allocated(device),
                      cuda_reserved_bytes=torch.cuda.memory_reserved(device))
        report["memory_events"].append(sample)
        require(reason is None, "Unchanged Windows resource guard: " + str(reason))
        require(time.monotonic() - started < args.max_wall_seconds, "Bounded GPU wall-time exceeded")

    memory("before_actual_capture_load")
    data = torch.load(args.input, map_location="cpu", weights_only=True)
    acts, labels, lengths, targets = validate_data(data, torch)
    report.update(input_provenance=data["provenance"], input_shape=list(acts.shape),
                  actual_frame_lengths=lengths.tolist(), actual_target_lengths=targets.tolist(),
                  full_input_tensor_bytes=acts.numel() * acts.element_size())
    native = acts.index_select(-1, torch.tensor(PERMUTATION)).contiguous()
    inverse = torch.tensor(mapping_checks())
    # Compare by individual class rather than retain a second full inverse tensor.
    require(all(torch.equal(native[..., int(inverse[old])], acts[..., old]) for old in range(8204)), "Class permutation changed a logit")
    native_labels = inverse[labels]
    require(torch.equal(torch.tensor(PERMUTATION)[native_labels], labels), "Target permutation is not reversible")
    del acts, data
    labels_gpu, lengths_gpu, targets_gpu = (value.to(device) for value in (native_labels, lengths, targets))
    ambient_python = random.getstate()
    ambient_cpu = torch.get_rng_state().clone()
    ambient_cuda = torch.cuda.get_rng_state(device).clone()
    try:
        for dtype_name, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
            for branch, seed in (("TDT", 0), ("RNNT", 31)):
                raw_native = None
                for outer in (None, "mean_volume"):
                    case = {"dtype": dtype_name, "omega": .1, "sigma": .02, "branch": branch, "seed": seed,
                            "first_random_uniform": random.Random(seed).uniform(0, 1), "invocation_outer_reduction": outer,
                            "outer_before_backward": "mean_volume", "inner_reduction": "none"}
                    results = []
                    baseline_gradient = None
                    for wrapped in (False, True):
                        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
                        memory("case_start_" + dtype_name + "_" + branch + "_" + str(outer) + "_" + str(wrapped))
                        loss = RNNTLoss(num_classes=8198, reduction=outer, loss_name="tdt",
                            loss_kwargs={"durations": [0, 1, 2, 3, 4], "sigma": .02, "omega": .1,
                                         "fastemit_lambda": 0., "clamp": -1.})
                        require(type(loss._loss).__name__ == "TDTLossNumba" and loss._loss.reduction == "none", "Official loss resolver/state changed")
                        hook = install_stock_nemo_loss_checkpointing(SimpleNamespace(loss=loss), torch_module=torch) if wrapped else None
                        x = native.to(device=device, dtype=dtype).requires_grad_()
                        random.seed(seed); torch.set_rng_state(ambient_cpu); torch.cuda.set_rng_state(ambient_cuda, device)
                        expected_rng = random.Random(seed); expected_rng.uniform(0, 1)
                        before_cpu, before_cuda = torch.get_rng_state().clone(), torch.cuda.get_rng_state(device).clone()
                        output = loss(log_probs=x, targets=labels_gpu, input_lengths=lengths_gpu, target_lengths=targets_gpu)
                        torch.cuda.synchronize(device)
                        report["native_kernel_executed"] = True
                        forward_rng = random.getstate()
                        require(forward_rng == expected_rng.getstate(), "Native omega did not consume exactly the expected draw")
                        require(loss.reduction == outer and loss._loss.reduction == "none", "Forward changed reduction state")
                        output_cpu = output.detach().cpu().clone()
                        # Real fused-joint behavior: restore outer state before backward.
                        loss.reduction = "mean_volume"
                        objective = loss.reduce(output, targets_gpu) if outer is None else output
                        objective_value = float(objective.detach())
                        if outer is None and not wrapped:
                            raw_native = output_cpu.tolist()
                        if outer == "mean_volume":
                            require(raw_native is not None and abs(objective_value - sum(raw_native) / int(targets.sum())) <= 2e-5,
                                    "Stock outer mean_volume differs from independently reduced raw native losses")
                        objective.backward()
                        torch.cuda.synchronize(device)
                        rng_ok = random.getstate() == forward_rng
                        torch_rng_ok = torch.equal(torch.get_rng_state(), before_cpu) and torch.equal(torch.cuda.get_rng_state(device), before_cuda)
                        require(rng_ok and torch_rng_ok, "Checkpoint/native backward consumed extra RNG")
                        require(loss.reduction == "mean_volume" and loss._loss.reduction == "none", "Backward/recompute failed to restore both reduction states")
                        require(x.grad is not None, "Joint gradient missing")
                        row = {"wrapped": wrapped, "output": output_cpu.tolist(), "objective": objective_value,
                               "Python_rng_advanced_once_forward_unchanged_backward": rng_ok,
                               "Torch_CPU_CUDA_rng_unchanged": torch_rng_ok,
                               "outer_after_backward": loss.reduction, "inner_after_backward": loss._loss.reduction,
                               "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                               "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
                        if not wrapped:
                            baseline_gradient = x.grad.detach().cpu()
                        else:
                            row["hook_identity"] = hook["identity"]
                            require(hook["identity"]["statistics"]["checkpointed_forward_calls"] == 1
                                    and hook["identity"]["statistics"]["recomputed_forward_calls"] >= 1, "Actual checkpoint recomputation was not exercised")
                            case["joint_gradient"] = gradient_stats(baseline_gradient, x.grad, torch)
                            require(case["joint_gradient"]["finite"], "Joint gradients are not finite")
                            require((case["joint_gradient"]["duration_gradient_l2"] == 0.) == (branch == "RNNT"), "Actual native branch does not match seed/duration gradients")
                            hook["remove"]()
                        memory("case_end_" + dtype_name + "_" + branch + "_" + str(outer) + "_" + str(wrapped))
                        results.append(row)
                        del output, objective, x, loss, hook
                        torch.cuda.empty_cache()
                    case["arms"] = results
                    first, second = results
                    first_values = first["output"] if isinstance(first["output"], list) else [first["output"]]
                    second_values = second["output"] if isinstance(second["output"], list) else [second["output"]]
                    require(all(math.isfinite(v) for v in first_values + second_values), "Native output nonfinite")
                    case["maximum_absolute_output_delta"] = max(abs(a - b) for a, b in zip(first_values, second_values))
                    case["passed"] = first_values == second_values and first["objective"] == second["objective"] and case["joint_gradient"]["bitwise_equal"]
                    report["cases"].append(case)
                    print(json.dumps({"case": [dtype_name, branch, outer], "passed": case["passed"],
                                      "gradient": case["joint_gradient"]}), flush=True)
                    del baseline_gradient
                    require(case["passed"], "Strict bitwise stock-versus-checkpoint loss/full-joint-gradient gate failed")
        report.update(native_kernel_executed=True, passed=True, cases_completed=len(report["cases"]),
                      wall_seconds=time.monotonic() - started,
                      strict_gate="bitwise-equal stock/wrapped native loss and every joint gradient; exact RNG/reduction restoration")
        for name, expected in PINS.items():
            require(sha(HERE / name) == expected, "Frozen runtime source changed during proof: " + name)
    finally:
        random.setstate(ambient_python); torch.set_rng_state(ambient_cpu); torch.cuda.set_rng_state(ambient_cuda, device)
        report["ambient_rng_restored_finally"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "work/nemo_actual_train_logits_v1.pt")
    parser.add_argument("--input-sha256", default=INPUT_SHA)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--gpu-window-authorized", action="store_true")
    parser.add_argument("--max-wall-seconds", type=float, default=300.)
    args = parser.parse_args()
    report = {"schema": 1, "kind": "actual_TRAIN_stock_NeMo_loss_checkpoint_oracle_v2", "passed": False}
    try:
        report.update(preflight(args))
        if args.validate_only:
            require("torch" not in sys.modules and "nemo" not in sys.modules, "Stdlib preflight imported heavy runtime")
            report.update(passed=True, status="CPU_prepared_GPU_unmeasured", heavy_runtime_imported=False)
        else:
            require(args.gpu_window_authorized, "Root-owned sole GPU authorization required")
            gpu_run(args, report)
            report["status"] = "complete"
    except Exception as error:
        report.update(passed=False, status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if args.output.resolve().is_relative_to(ROOT) and not args.output.exists():
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")


if __name__ == "__main__":
    main()
