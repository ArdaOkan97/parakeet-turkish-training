"""INACTIVE ROOT-only synthetic OTHER versus official full-vocabulary native4.

Import and --source-only are stdlib-only. GPU execution requires a later sole
ROOT-owned window and coherent native-env-v2 preload. No model, audio, dataset,
checkpoint or optimizer is loaded. FP32 projection only: not a BF16/full-ASR
adoption gate. The bounded watchdog can terminate THIS process on its deadline;
ROOT must still observe the owned process exit before any following GPU task.
"""
from __future__ import annotations

import argparse
import ast
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import threading
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NATIVE_PYTHON = ROOT / "work/nemo-native-env-v2/Scripts/python.exe"
PINS = {
    "native_tdt_other_bucket_v1.py": "083a8e4288a4475ef2965d42f0d1ef1c9700f931e44bae02935c34ed1d2fa270",
    "native_tdt_checkpointed_v4.py": "24b9cf5b8f3e11d5b043c22d0940da3735a46f8f70b6f5810b97f17d545cd90e",
    "preload_nemo_linker_v1.py": "f8787b0f3346c9f526903ff0c06de4c266f940ca3b0f41c55907cc795e7cc2ba",
    "launch_nemo_linker_v4.py": "bdd7078dff81adc961c17588ed7fd84b152da71c619c2ff114994fa68d308af0",
    "resource_watchdog_v2.py": "64d1393e1e45cf65dac14f78ac955331c91b4a3f0eafbf7535196e2b61778641",
    "paired_windows_commit_telemetry_v1.py": "2ae9f2cdfc388a294914e000c60b62fe09cddac147700c5762851d626a95a683",
}
OFFICIAL = {
    "collections/asr/parts/numba/rnnt_loss/rnnt_pytorch.py": "7b5ff7fe789c23a36068d20e2e3fce3d2931be6e55bcebff7efc4d86c399918e",
    "collections/asr/parts/numba/rnnt_loss/utils/cuda_utils/gpu_rnnt.py": "df40a679bbabaeb9bc05f92bb3fbc436b80ef2f4c311564a00391934087c985c",
    "collections/asr/parts/numba/rnnt_loss/utils/cuda_utils/gpu_rnnt_kernel.py": "385de70753b6c1b2a2e5b402a689d816ece82faa846368b3d52cca7e47bb9045",
}
TOLERANCES = {"raw_loss_atol": 2e-4, "raw_loss_rtol": 2e-5,
              "gradient_atol": 1e-5, "gradient_rtol": 5e-4,
              "gradient_relative_l2": 5e-4,
              "scalar_loss_atol": 2e-4, "scalar_gradient_atol": 2e-4,
              "scalar_gradient_rtol": 3e-3}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--source-only", action="store_true")
    result.add_argument("--gpu-window-authorized", action="store_true")
    result.add_argument("--max-seconds", type=float, default=180.)
    result.add_argument("--stress-frames", type=int, default=96)
    result.add_argument("--stress-targets", type=int, default=32)
    return result


def validate_arguments(args):
    args.output = args.output.resolve()
    require(args.output.is_relative_to(ROOT) and not args.output.exists(), "Fresh workspace report required")
    require(math.isfinite(args.max_seconds) and 15 <= args.max_seconds <= 180,
            "Own process deadline must be between15 and180 seconds")
    require(1 <= args.stress_frames <= 128 and 1 <= args.stress_targets <= 64,
            "Bounded synthetic stress dimensions required; no source clips")
    require(args.source_only or args.gpu_window_authorized, "ROOT-owned solo GPU window must be explicit")
    require(not (args.source_only and args.gpu_window_authorized), "Source-only and GPU mode are distinct")
    return args


def source_records():
    records = {}
    for name, expected in PINS.items():
        path = HERE / name
        require(sha(path) == expected, "Pinned source changed: " + name)
        records[name] = {"path": str(path), "sha256": expected}
    package = ROOT / "work/nemo-native-env-v1/Lib/site-packages/nemo"
    for relative, expected in OFFICIAL.items():
        path = package / relative
        require(sha(path) == expected, "Official native source changed: " + relative)
        records[relative] = {"path": str(path), "sha256": expected}
    return records


def case_specs(args):
    return [
        {"name": "repeated_new_targets_nonlast_blank", "V": 8711, "blank": 8192, "T": 5,
         "labels": [8199, 8199, 8200, 0], "vocab_chunk": 1024, "time_chunk": 2},
        {"name": "dominant_unused_new_OTHER", "V": 8711, "blank": 8192, "T": 4,
         "labels": [8193, 1024, 8193], "vocab_chunk": 1024, "time_chunk": 2, "dominant_id": 8709},
        {"name": "all_retained_empty_OTHER", "V": 4, "blank": 1, "T": 4,
         "labels": [0, 2, 3, 2], "vocab_chunk": 2, "time_chunk": 2},
        {"name": "tiny_scalar_reference", "V": 7, "blank": 3, "T": 3,
         "labels": [6, 6], "vocab_chunk": 2, "time_chunk": 2, "scalar_reference": True},
        {"name": "bounded_lattice_memory", "V": 8711, "blank": 8192, "T": args.stress_frames,
         "labels": ([0, 1024, 8193, 8199, 8201, 8710, 8199, 3] * 8)[:args.stress_targets],
         "vocab_chunk": 1024, "time_chunk": 16, "memory_stress_only": True},
    ]


def atomic_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + str(os.getpid()) + ".partial")
    payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    os.replace(temporary, path)


class Monitor:
    """Bounded stdlib self/system sampler; never calls CUDA or other processes."""
    def __init__(self, report, path, max_seconds):
        import resource_watchdog_v2 as resource
        self.resource, self.guard = resource, resource.ResourceGuard()
        self.report, self.path, self.max_seconds = report, path, max_seconds
        self.start, self.phase = time.monotonic(), "before_preload"
        self.samples, self.lock = deque(maxlen=128), threading.RLock()
        self.stop, self.done = threading.Event(), threading.Event()
        self.reason = None
        self._snapshot = json.loads(json.dumps(report, allow_nan=False))
        self.thread = threading.Thread(target=self._loop, name="owned_OTHER_comparison_memory", daemon=True)

    def publish(self):
        with self.lock:
            self._snapshot = json.loads(json.dumps(self.report, allow_nan=False))
            atomic_report(self.path, self._snapshot)

    def observation(self):
        with self.lock:
            sample = self.resource.sample_memory(os.getpid(), self.phase)
            system = sample["system_memory"]
            reason, guard = self.guard.observe(system, sample["worker_private_memory"],
                observed_monotonic=sample["observed_monotonic"])
            process = sample["calling_process_commit_memory"]
            if not process.get("available"):
                reason = reason or "owned_process_commit_telemetry_unavailable"
            reason = reason or self.resource.worker_process_commit_reason(process, os.getpid())
            sample["guard"] = guard
            self.samples.append(sample)
            if reason and not self.reason:
                self.reason = reason
                self.stop.set()
        return sample

    def _loop(self):
        while not self.done.wait(.1):
            if time.monotonic() - self.start >= self.max_seconds:
                with self.lock:
                    self.reason = self.reason or "own_process_wall_deadline"
                    snapshot = json.loads(json.dumps(self._snapshot))
                    snapshot.update(status="stopped_at_own_deadline", passed=False,
                        stop_reason=self.reason, sampled_memory=list(self.samples),
                        own_hard_exit_without_finally=True, finished_at_utc=timestamp())
                    atomic_report(self.path, snapshot)
                os._exit(3)
            if not self.samples or time.monotonic() - self.samples[-1]["observed_monotonic"] >= 5.:
                try:
                    self.observation()
                except BaseException as error:
                    with self.lock:
                        self.reason = "owned_memory_sampler_error:" + type(error).__name__
                        self.stop.set()

    def check(self, phase):
        self.phase = phase
        self.observation()
        if self.stop.is_set():
            raise RuntimeError("Resource stop: " + str(self.reason))

    def finish(self):
        self.done.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=2.)
        self.report.update(sampled_memory=list(self.samples), memory_sampler_thread_joined=not self.thread.is_alive(),
            stop_reason=self.reason, wall_seconds_including_preload_JIT=time.monotonic()-self.start,
            resource_policy=dict(self.resource.POLICY), own_PID=os.getpid(), own_PPID=os.getppid())


def rng_fingerprint(torch):
    digest = lambda value: hashlib.sha256(bytes(value.cpu().tolist())).hexdigest()
    return {"Python": hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
            "CPU": digest(torch.get_rng_state()),
            "CUDA_all": [digest(value) for value in torch.cuda.get_rng_state_all()]}


def tensor_difference(torch, reference, candidate):
    require(reference.shape == candidate.shape, "Compared gradient shape differs")
    finite = bool(torch.isfinite(reference).all() and torch.isfinite(candidate).all())
    left, right = reference.double().flatten(), candidate.double().flatten()
    delta = (left-right).abs()
    relative = float((left-right).norm()/left.norm().clamp_min(1e-30))
    return {"shape": list(reference.shape), "elements": reference.numel(), "finite": finite,
            "maximum_absolute_delta": float(delta.max()) if delta.numel() else 0., "relative_l2": relative,
            "allclose": bool(torch.allclose(reference, candidate,
                atol=TOLERANCES["gradient_atol"], rtol=TOLERANCES["gradient_rtol"])),
            "passed": finite and relative <= TOLERANCES["gradient_relative_l2"] and
                bool(torch.allclose(reference, candidate, atol=TOLERANCES["gradient_atol"],
                                    rtol=TOLERANCES["gradient_rtol"]))}


def _logadd(a, b):
    if a == -math.inf: return b
    if b == -math.inf: return a
    maximum = max(a, b)
    return maximum + math.log(math.exp(a-maximum)+math.exp(b-maximum))


def scalar_loss(values, case, prototype):
    """Independent Float64 source-DAG oracle for a tiny synthetic projection."""
    ep, dp, weight, bias = (values[key] for key in ("EP", "DP", "weight", "bias"))
    T, U, V = case["T"], len(case["labels"]), case["V"]
    alpha = [[-math.inf] * (U+1) for _ in range(T)]
    alpha[0][0] = 0.
    terminal = -math.inf
    for t in range(T):
        for u in range(U+1):
            hidden = [math.tanh(a+b) for a, b in zip(ep[0][t], dp[0][u])]
            logits = [sum(a*b for a, b in zip(row, hidden))+offset for row, offset in zip(weight, bias)]
            token_normalizer = prototype.scalar_logsumexp(logits[:V])
            duration_normalizer = prototype.scalar_logsumexp(logits[V:])
            token = [value-token_normalizer for value in logits[:V]]
            duration = [value-duration_normalizer for value in logits[V:]]
            for d in range(5):
                dest = t+d
                base = alpha[t][u]+duration[d]
                if d and dest < T:
                    alpha[dest][u] = _logadd(alpha[dest][u], base+token[case["blank"]])
                if d and dest == T and u == U:
                    terminal = _logadd(terminal, base+token[case["blank"]])
                if u < U and dest < T:
                    alpha[dest][u+1] = _logadd(alpha[dest][u+1], base+token[case["labels"][u]])
    return -terminal


def scalar_reference(base, case, native, prototype):
    values = {name: value.tolist() for name, value in base.items()}
    loss = scalar_loss(values, case, prototype)
    coordinates = [("EP", (0, 1, 2)), ("DP", (0, 1, 1)), ("weight", (0, 2)),
                   ("weight", (6, 3)), ("weight", (3, 0)), ("weight", (9, 1)),
                   ("bias", (0,)), ("bias", (6,)), ("bias", (3,)), ("bias", (10,))]
    checks = []
    epsilon = 1e-4
    for name, coordinate in coordinates:
        parent = values[name]
        for index in coordinate[:-1]: parent = parent[index]
        index = coordinate[-1]
        original = parent[index]
        parent[index] = original+epsilon
        plus = scalar_loss(values, case, prototype)
        parent[index] = original-epsilon
        minus = scalar_loss(values, case, prototype)
        parent[index] = original
        expected = (plus-minus)/(2*epsilon)
        actual = float(native["gradients"][name][coordinate])
        delta = abs(actual-expected)
        checks.append({"parameter": name, "coordinate": list(coordinate), "scalar_FD": expected,
            "native_gradient": actual, "absolute_delta": delta,
            "passed": delta <= TOLERANCES["scalar_gradient_atol"] + TOLERANCES["scalar_gradient_rtol"]*abs(expected)})
    return {"method": "Float64 scalar source-DAG central differences, epsilon1e-4; no full-model oracle",
        "raw_loss": loss, "raw_loss_native_delta": abs(loss-native["loss"]),
        "coordinates": checks, "passed": abs(loss-native["loss"]) <= TOLERANCES["scalar_loss_atol"] and
            all(check["passed"] for check in checks)}


def run_arm(torch, native4, prototype, checkpoint, monitor, base, case, mode, timed_scope):
    monitor.check(case["name"]+":"+mode+":"+timed_scope)
    device = torch.device("cuda:0")
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    before = {"allocated_bytes": torch.cuda.memory_allocated(device), "reserved_bytes": torch.cuda.memory_reserved(device)}
    values = {name: value.to(device).detach().requires_grad_() for name, value in base.items()}
    ep, dp, weight, bias = (values[key] for key in ("EP", "DP", "weight", "bias"))
    labels = torch.tensor([case["labels"]], device=device, dtype=torch.long)
    native4.reset_cache_cleanup_evidence()
    rng_before = rng_fingerprint(torch)
    torch.cuda.synchronize(device)
    began = time.perf_counter()
    details = None
    if mode == "full_native4":
        def joint(e, d):
            return torch.nn.functional.linear(torch.tanh(e+d), weight, bias)
        value = native4.chunked_tdt_loss(joint, ep, dp, labels, case["V"], case["blank"], tuple(range(5)),
            sigma=0., omega=0., reduction="none", recompute=True, max_joint_bytes=1_850_000_000)
    else:
        def callback(acts, durations, targets, lengths, target_lengths, **policy):
            backend = native4._native_loss(policy["blank_id"], policy["durations"], policy["sigma"])
            require(backend.clamp == 0. and backend.omega == backend.fastemit_lambda == backend.sigma == 0.,
                    "Actual native effective policy differs")
            return backend.loss(acts, durations, targets, lengths, target_lengths, backend.blank,
                backend.durations, backend.reduction, backend.fastemit_lambda, backend.clamp, backend.sigma, backend.omega)
        value, details = prototype.compressed_clip_loss(ep, dp, weight, bias, case["labels"],
            vocab_size=case["V"], blank_id=case["blank"], activation=torch.tanh,
            torch_module=torch, checkpoint_fn=checkpoint, native_callback=callback,
            native_policy={"sigma": 0., "omega": 0., "fastemit_lambda": 0., "effective_native_clamp": 0.,
                           "reduction": "none", "source_files_sha256": OFFICIAL},
            time_chunk_size=case["time_chunk"], vocab_chunk_size=case["vocab_chunk"], recompute=True)
    value.sum().backward()
    torch.cuda.synchronize(device)
    seconds = time.perf_counter()-began
    after = {"allocated_bytes": torch.cuda.memory_allocated(device), "reserved_bytes": torch.cuda.memory_reserved(device),
             "peak_allocated_bytes": torch.cuda.max_memory_allocated(device), "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
    rng_after = rng_fingerprint(torch)
    require(bool(torch.isfinite(value).all()), "Nonfinite raw loss")
    require(all(tensor.grad is not None and bool(torch.isfinite(tensor.grad).all()) for tensor in values.values()),
            "Every EP/DP/full head weight/bias gradient must exist and be finite")
    gradients = {name: tensor.grad.detach().cpu() for name, tensor in values.items()}
    result = {"mode": mode, "timing_scope": timed_scope,
        "seconds_forward_backward_sync_excluding_input_H2D_and_CPU_gradient_capture": seconds,
        "loss": float(value.detach().cpu()[0]), "rng_before": rng_before, "rng_after": rng_after,
        "RNG_unchanged": rng_before == rng_after, "memory_before_inputs": before, "CUDA_memory": after,
        "gradient_elements": {name: grad.numel() for name, grad in gradients.items()},
        "cleanup": native4.cache_cleanup_evidence(), "prototype": details, "gradients": gradients}
    del values, ep, dp, weight, bias, value, labels
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    monitor.check(case["name"]+":"+mode+":complete")
    return result


def compare(torch, full, other, case):
    gradients = {name: tensor_difference(torch, full["gradients"][name], other["gradients"][name])
                 for name in full["gradients"]}
    retained = sorted(set(case["labels"]) | {case["blank"]})
    remaining = [index for index in range(case["V"]) if index not in set(retained)]
    unused_new = [index for index in range(8199, case["V"]) if index not in set(case["labels"])]
    groups = {}
    for label, rows in (("OTHER_all_rows", remaining), ("unused_new512_rows", unused_new),
                        ("retained_target_and_blank_rows", retained), ("duration5_rows", list(range(case["V"], case["V"]+5)))):
        if not rows:
            groups[label] = {"rows": 0, "scope": "empty", "passed": True}
            continue
        groups[label] = {"rows": len(rows), "weight": tensor_difference(torch, full["gradients"]["weight"][rows], other["gradients"]["weight"][rows]),
            "bias": tensor_difference(torch, full["gradients"]["bias"][rows], other["gradients"]["bias"][rows]),
            "full_nonzero_bias_rows": int((full["gradients"]["bias"][rows] != 0).sum())}
        groups[label]["passed"] = groups[label]["weight"]["passed"] and groups[label]["bias"]["passed"]
    loss_delta = abs(full["loss"]-other["loss"])
    loss_ok = loss_delta <= TOLERANCES["raw_loss_atol"]+TOLERANCES["raw_loss_rtol"]*abs(full["loss"])
    rng_ok = full["RNG_unchanged"] and other["RNG_unchanged"] and full["rng_before"] == other["rng_before"]
    return {"raw_loss_absolute_delta": loss_delta, "raw_loss_allclose": loss_ok,
        "gradients": gradients, "explicit_row_groups": groups, "same_RNG_prefix_and_preserved": rng_ok,
        "passed": loss_ok and rng_ok and all(value["passed"] for value in gradients.values()) and
            all(value["passed"] for value in groups.values())}


def main(args):
    args = validate_arguments(args)
    records = source_records()
    specs = case_specs(args)
    report = {"schema": 1, "kind": "ROOT_owned_synthetic_native4_OTHER_comparison_v1",
        "status": "prepared", "passed": False, "script_sha256": sha(__file__), "created_at_utc": timestamp(),
        "sources": records, "case_specs": specs, "tolerances": TOLERANCES,
        "full_vocabulary_gradient_scope": "Every EP/DP/full head weight/bias element including all OTHER and unused new512 rows",
        "no_model_dataset_audio_checkpoint_optimizer": True, "native_kernel_executed": False,
        "FP32_projection_only": True, "policy": {"sigma": 0., "omega": 0., "FastEmit": 0., "effective_clamp": 0.,
        "raw_NLL": True, "whole_ordered_repeated_targets": True, "cap_decimal_GB": 8.},
        "limits": ["Synthetic numerical gate only; no full-model/BF16/ASR adoption proof.",
            "Sampled host commit can miss sub-five-second peaks; CUDA allocator peaks are separate from private/system commitment.",
            "First JIT cold timings and warm measurements are separate; one sample is not a training-throughput claim.",
            "No target-only renormalization: every remaining class stays in OTHER.",
            "Own deadline can hard-exit this process; ROOT must observe actual owned exit before a next GPU launch."]}
    if args.source_only:
        require(not any(name in sys.modules for name in ("torch", "nemo", "numba.cuda", "transformers")),
                "Source-only path imported ML")
        report.update(status="source_only_prepared_GPU_unexecuted", passed_source_checks=True,
            ML_imported=False, actual_numerical_or_memory_pass=False, module_AST_parsed=True)
        ast.parse(Path(__file__).read_text(encoding="utf-8"))
        atomic_report(args.output, report)
        print(json.dumps({"output": str(args.output), "status": report["status"], "script_sha256": report["script_sha256"]}))
        return 0
    require(Path(sys.executable).resolve() == NATIVE_PYTHON.resolve(), "Use native-env-v2 Python for coherent12.9 preload")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    monitor = Monitor(report, args.output, args.max_seconds)
    report["cases"] = []
    try:
        monitor.check("before_preload")
        monitor.thread.start()
        import preload_nemo_linker_v1 as preloader
        preload = preloader.preload()
        report["preload"] = preload
        require(preload["actual_binding_linker"]["version"] == [12, 9] and not preload["cuda_initialized"],
                "Actual coherent12.9 pre-Torch preload evidence differs")
        import torch
        from torch.utils.checkpoint import checkpoint
        import native_tdt_checkpointed_v4 as native4
        import native_tdt_other_bucket_v1 as prototype
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = torch.device("cuda:0")
        properties = torch.cuda.get_device_properties(device)
        require(properties.total_memory >= 8_000_000_000, "Expected device can accommodate declared cap8")
        torch.cuda.set_per_process_memory_fraction(8_000_000_000/properties.total_memory, device)
        torch.manual_seed(952211)
        random.seed(952211)
        report.update(status="running", native_identity=native4.configuration_identity(),
            device={"name": properties.name, "total_bytes": properties.total_memory, "cap_bytes": 8_000_000_000},
            torch_version=torch.__version__, dtype="float32", hidden_size=4,
            native_effective_policy_verified=False)
        for number, case in enumerate(specs):
            monitor.check("create_synthetic:"+case["name"])
            generator = torch.Generator(device="cpu").manual_seed(1550+number)
            V, T, U = case["V"], case["T"], len(case["labels"])
            base = {"EP": torch.randn((1, T, 4), generator=generator)*.2,
                    "DP": torch.randn((1, U+1, 4), generator=generator)*.2,
                    "weight": torch.randn((V+5, 4), generator=generator)*.15,
                    "bias": torch.randn((V+5,), generator=generator)*.1}
            if "dominant_id" in case:
                base["bias"][case["dominant_id"]] = 12.
            backend = native4._native_loss(case["blank"], tuple(range(5)), 0.)
            require(backend.clamp == 0. and backend.sigma == backend.omega == backend.fastemit_lambda == 0.,
                    "Actual full native effectiveclamp/sigma/omega/FastEmit policy differs")
            report["native_effective_policy_verified"] = True
            item = {"spec": case, "cold": [], "warm": [], "status": "running"}
            report["cases"].append(item)
            # Cold pair records class-specialization/JIT and launch startup; each
            # warm arm is fresh identical synthetic tensors, not updated weights.
            for scope in ("cold_including_possible_first_JIT", "warm_after_exact_case_cold_pair"):
                scope_key = "cold" if scope.startswith("cold") else "warm"
                item[scope_key] = {}
                full = run_arm(torch, native4, prototype, checkpoint, monitor, base, case, "full_native4", scope)
                report["native_kernel_executed"] = True
                item[scope_key]["full"] = {key: value for key, value in full.items() if key != "gradients"}
                monitor.publish()
                other = run_arm(torch, native4, prototype, checkpoint, monitor, base, case, "OTHER", scope)
                item[scope_key]["OTHER"] = {key: value for key, value in other.items() if key != "gradients"}
                comparison = compare(torch, full, other, case)
                if case.get("scalar_reference"):
                    comparison["scalar_reference"] = scalar_reference(base, case, full, prototype)
                    comparison["passed"] &= comparison["scalar_reference"]["passed"]
                item[scope_key]["comparison"] = comparison
                require(comparison["passed"], "Loss/gradient/RNG gate failed: "+case["name"]+":"+scope)
                del full, other
            item.update(status="complete", passed=True)
            monitor.publish()
        for record in records.values():
            require(sha(record["path"]) == record["sha256"], "Source changed during actual comparison")
        monitor.check("all_cases_completed")
        report.update(status="complete", passed=True, native_kernel_executed=True,
                      actual_numerical_or_memory_pass=True, all_sources_unchanged=True)
    except BaseException as error:
        report.update(status="failed", passed=False, failure_type=type(error).__name__, failure=str(error),
            traceback=traceback.format_exc(limit=6))
        if "torch" in locals() and torch.cuda.is_initialized():
            report["failure_CUDA_memory"] = {"allocated_bytes": torch.cuda.memory_allocated(),
                "reserved_bytes": torch.cuda.memory_reserved(), "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "phase": monitor.phase}
    finally:
        monitor.finish()
        report["finished_at_utc"] = timestamp()
        atomic_report(args.output, report)
    print(json.dumps({"output": str(args.output), "status": report["status"], "passed": report["passed"],
        "completed_cases": sum(case.get("passed", False) for case in report["cases"]), "stop_reason": monitor.reason}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main(parser().parse_args()))
