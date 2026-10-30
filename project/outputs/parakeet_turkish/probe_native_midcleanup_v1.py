"""INACTIVE native3/native4 critical-lattice GPU oracle, prepared without ML.

ROOT alone may execute after the current owned training parent is CLOSED.
Synthetic per-clip kernel proof is not an ASR/B8/all-pool/phase2 admission.
The full case materializes the actual321x144x8204 BF16 joint and native FP32
lexical gradients. No audio/model/checkpoint is loaded. Runtime hooks are
observational and restored; frozen source files are never modified.
"""
from __future__ import annotations

import argparse
import ast
import ctypes
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PINS = {
    "native_tdt_checkpointed_v3.py": "839194dddf83c3b08bf608d01cb540b2f160feee426a91373d14cfc426ab7494",
    "native_tdt_checkpointed_v4.py": "24b9cf5b8f3e11d5b043c22d0940da3735a46f8f70b6f5810b97f17d545cd90e",
    "preload_nemo_linker_v1.py": "f8787b0f3346c9f526903ff0c06de4c266f940ca3b0f41c55907cc795e7cc2ba",
    "launch_nemo_linker_v4.py": "bdd7078dff81adc961c17588ed7fd84b152da71c619c2ff114994fa68d308af0",
    "resource_watchdog_v2.py": "64d1393e1e45cf65dac14f78ac955331c91b4a3f0eafbf7535196e2b61778641",
    "paired_windows_commit_telemetry_v1.py": "2ae9f2cdfc388a294914e000c60b62fe09cddac147700c5762851d626a95a683",
}
CASES = (("small", 8, 3), ("critical_original_shape", 321, 143))
MID_PHASE = "after_split_before_native_gradient"


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_preflight():
    for name, digest in PINS.items():
        require(sha(HERE / name) == digest, "Frozen probe source changed: " + name)
    trees = [ast.parse((HERE / name).read_text(encoding="utf-8"))
             for name in ("native_tdt_checkpointed_v3.py", "native_tdt_checkpointed_v4.py")]
    clips = [next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_clip_loss")
             for tree in trees]
    new_clip = ast.unparse(clips[1])
    require(new_clip.index("del packed") < new_clip.index(repr(MID_PHASE)) < new_clip.index("backend.loss("),
            "Split/delete/inactive-reclaim/native allocation ordering differs")
    old, new = ast.unparse(clips[0]), new_clip
    call = next(node for node in clips[1].body if isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "reclaim_inactive_cuda_cache")
    clips[1].body.remove(call)
    require(ast.dump(clips[0], include_attributes=False) == ast.dump(clips[1], include_attributes=False),
            "Native4 arithmetic/RNG body differs after removing the sole extra cache call")
    require("label_acts" in old and "backend.loss(" in old, "Actual full native kernel path required")
    t, u, v = 321, 143, 8199
    size = {"BF16_packed_bytes": t*(u+1)*(v+5)*2, "FP32_lexical_bytes": t*(u+1)*v*4,
            "FP32_lexical_gradient_bytes": t*(u+1)*v*4,
            "FP32_duration_bytes_each": t*(u+1)*5*4, "native_workspace_bytes": (3*t*(u+1)+2)*4}
    require(size["BF16_packed_bytes"] == 758443392 and size["FP32_lexical_gradient_bytes"] == 1515962304,
            "The hardware case no longer forces the actual critical lattice")
    return {"source_sha256": dict(PINS), "critical_shape": {"T": t, "U": u, "vocabulary": v,
            "joint_width": v+5, "blank": 8192}, "nominal_buffers": size,
            "allocator_hard_cap_bytes": 8000000000, "synthetic_joint_hidden_dimension": 4,
            "actual_GPU_or_ML_execution": False, "checks": ["all_frozen_source_hashes",
            "delete_then_midcleanup_then_backend", "same_clip_arithmetic_and_RNG_AST", "actual_758MB_packed_and_1516MB_gradient_shape"]}


def run_gpu(args, prepared):
    started = time.monotonic()
    require(args.gpu_probe_authorized and args.parent_closed_confirmed,
            "ROOT must authorize and observe actual training parent CLOSED before any Torch import")
    require(Path(sys.executable).resolve() == (ROOT / "work/nemo-native-env-v2/Scripts/python.exe").resolve(),
            "Use the isolated coherent native environment")
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        kernel.SetPriorityClass.restype = ctypes.c_int
        priority = bool(kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x4000))
    else: priority = None
    sys.path.insert(0, str(HERE))
    require("torch" not in sys.modules, "Preload must precede Torch")
    from preload_nemo_linker_v1 import preload
    print("Coherent12.9 preload before Torch; synthetic kernel probe only", flush=True)
    preload_evidence = preload()
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    require(torch.cuda.is_available(), "Actual CUDA required")
    total = int(torch.cuda.get_device_properties(0).total_memory)
    require(total > 8000000000, "8GB process allocator cap cannot fit this device")
    torch.cuda.set_per_process_memory_fraction(8000000000 / total, 0)
    from resource_watchdog_v2 import ResourceGuard, POLICY, sample_memory, worker_process_commit_reason
    import native_tdt_checkpointed_v3 as old
    import native_tdt_checkpointed_v4 as new
    guard, lock, stop = ResourceGuard(), threading.RLock(), threading.Event()
    observations, pressure, cleanup = [], [], []
    current = {"case": "startup", "adapter": None, "phase": "native_backend_initialization"}
    stopped_reason = [None]
    def owned_memory(phase):
        reading = sample_memory(os.getpid(), phase=phase)
        with lock:
            reason, policy_evidence = guard.observe(reading["system_memory"], reading["worker_private_memory"],
                                                    observed_monotonic=reading["observed_monotonic"])
            reason = reason or worker_process_commit_reason(reading, expected_worker_pid=os.getpid())
            item = {"case": current["case"], "adapter": current["adapter"], "phase": phase,
                    "memory": reading, "resource_guard": policy_evidence, "stop_reason": reason}
            if reason and stopped_reason[0] is None: stopped_reason[0] = reason
            if time.monotonic()-started > args.max_wall_seconds and stopped_reason[0] is None:
                stopped_reason[0] = "probe_cooperative_wall_budget_exceeded"
            pressure.append(item)
            require(len(pressure) <= 256, "Bounded pressure telemetry capacity exhausted")
        return reading
    def monitor():
        while not stop.wait(5.):
            try: owned_memory("owned_five_second_resource_poll")
            except BaseException as error:
                with lock: stopped_reason[0] = stopped_reason[0] or "resource_sampler_failed:" + type(error).__name__
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    def snapshot(phase):
        current["phase"] = phase
        torch.cuda.synchronize()
        memory = owned_memory(phase)
        item = {"case": current["case"], "adapter": current["adapter"], "phase": phase,
                "elapsed_seconds": time.monotonic()-started,
                "allocated_bytes": int(torch.cuda.memory_allocated()), "reserved_bytes": int(torch.cuda.memory_reserved()),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()), "memory": memory}
        observations.append(item)
        require(len(observations) <= 192, "Bounded operation telemetry capacity exhausted")
        require(stopped_reason[0] is None, "Resource/wall guard stopped probe: " + str(stopped_reason[0]))
        return item
    def bytes_hash(value):
        return hashlib.sha256(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
    def rng_state(): return (random.getstate(), torch.get_rng_state().clone(), [x.clone() for x in torch.cuda.get_rng_state_all()])
    def restore_rng(state):
        random.setstate(state[0]); torch.set_rng_state(state[1]); torch.cuda.set_rng_state_all(state[2])
    def rng_equal(a, b):
        return a[0] == b[0] and torch.equal(a[1], b[1]) and len(a[2]) == len(b[2]) and all(torch.equal(x,y) for x,y in zip(a[2],b[2]))
    backend = None
    results, errors = [], []
    try:
        snapshot("before_official_backend_import")
        backend = old._native_loss(8192, (0,1,2,3,4), 0.)
        snapshot("after_official_backend_import")
        for case_index, (case, t, u) in enumerate(CASES):
            generator = torch.Generator().manual_seed(317 + case_index)
            data = {"ep": torch.randn((1,t,4), generator=generator), "dp": torch.randn((1,u+1,4), generator=generator),
                    "weight": torch.randn((8204,4), generator=generator)*.02,
                    "bias": torch.randn((8204,), generator=generator)*.01,
                    "labels": torch.randint(0,8199,(1,u),generator=generator)}
            data["labels"][data["labels"] == 8192] = 0
            reference_rng = rng_state()
            arms = []
            for name, module in (("native3", old), ("native4", new)):
                current.update(case=case, adapter=name)
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); module.reset_cache_cleanup_evidence()
                restore_rng(reference_rng)
                snapshot("arm_baseline")
                class Joint(torch.nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.weight = torch.nn.Parameter(data["weight"].cuda())
                        self.bias = torch.nn.Parameter(data["bias"].cuda())
                    def forward(self, e, d):
                        snapshot("before_BF16_packed_joint")
                        value = torch.nn.functional.linear(torch.tanh(e+d), self.weight, self.bias)
                        require(value.dtype == torch.bfloat16 and tuple(value.shape) == (1,t,u+1,8204),
                                "Actual BF16 full critical packed lattice required")
                        snapshot("after_BF16_packed_joint")
                        return value
                joint = Joint(); ep = data["ep"].cuda().requires_grad_(); dp = data["dp"].cuda().requires_grad_()
                labels = data["labels"].cuda(); frames = torch.tensor([t]); targets = torch.tensor([u])
                active_native = [False]
                class Backend:
                    blank=backend.blank; durations=backend.durations; reduction=backend.reduction
                    fastemit_lambda=backend.fastemit_lambda; clamp=backend.clamp; sigma=backend.sigma; omega=backend.omega
                    def loss(self, *operands):
                        snapshot("before_official_native_loss")
                        active_native[0] = True
                        try: value = backend.loss(*operands)
                        finally: active_native[0] = False
                        snapshot("after_official_native_loss")
                        return value
                original_zeros, original_reclaim = torch.zeros_like, module.reclaim_inactive_cuda_cache
                def observe_zeros(value, *positional, **kwargs):
                    relevant = active_native[0] and value.ndim == 4 and tuple(value.shape[:3]) == (1,t,u+1)
                    phase = "lexical" if value.shape[-1] == 8199 else "duration"
                    if relevant: snapshot("before_native_" + phase + "_gradient_allocation")
                    result = original_zeros(value, *positional, **kwargs)
                    if relevant: snapshot("after_native_" + phase + "_gradient_allocation")
                    return result
                def observe_reclaim(device, phase):
                    before = snapshot("before_reclaim:" + phase)
                    value = original_reclaim(device, phase)
                    after = snapshot("after_reclaim:" + phase)
                    cleanup.append({"case":case,"adapter":name,"phase":phase,
                                    "allocated_unchanged":before["allocated_bytes"] == after["allocated_bytes"],
                                    "reserved_reclaimed_bytes":before["reserved_bytes"]-after["reserved_bytes"]})
                    return value
                torch.zeros_like = observe_zeros; module.reclaim_inactive_cuda_cache = observe_reclaim
                try:
                    before_rng = rng_state()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        value = module._loss_with_backend(joint,ep,dp,labels,8199,8192,(0,1,2,3,4),frames,targets,
                            "none",True,1700000000,Backend())
                    snapshot("after_checkpointed_forward")
                    value.sum().backward()
                    snapshot("after_complete_backward")
                    gradients = {"ep":ep.grad.detach().cpu().clone(),"dp":dp.grad.detach().cpu().clone(),
                                 "head_weight":joint.weight.grad.detach().cpu().clone(),"head_bias":joint.bias.grad.detach().cpu().clone()}
                    require(bool(torch.isfinite(value).all()) and all(bool(torch.isfinite(g).all()) for g in gradients.values()),
                            "Complete finite NLL/EP/DP/head gradients required")
                    rng_preserved = rng_equal(before_rng,rng_state())
                    require(rng_preserved, "Native clip/recompute changed Python/CPU/allCUDA RNG")
                    evidence = module.cache_cleanup_evidence()
                    summary = {"adapter":name,"raw_NLL":value.detach().cpu().tolist(),"gradient_sha256":{k:bytes_hash(g) for k,g in gradients.items()},
                               "RNG_exact":rng_preserved,"peak_allocated_bytes":int(torch.cuda.max_memory_allocated()),
                               "peak_reserved_bytes":int(torch.cuda.max_memory_reserved()),"native_cleanup":evidence}
                    arms.append((summary,value.detach().cpu().clone(),gradients))
                finally:
                    torch.zeros_like = original_zeros; module.reclaim_inactive_cuda_cache = original_reclaim
                del value,joint,ep,dp,labels,frames,targets
                torch.cuda.empty_cache()
            same_loss = torch.equal(arms[0][1],arms[1][1])
            same_grads = {key:torch.equal(arms[0][2][key],arms[1][2][key]) for key in arms[0][2]}
            require(same_loss and all(same_grads.values()), "Native3/native4 loss or complete EP/DP/head gradient differs")
            results.append({"case":case,"T":t,"U":u,"synthetic_hidden":4,"raw_NLL_bitexact":same_loss,
                            "complete_gradients_bitexact":same_grads,"arms":[arm[0] for arm in arms]})
            print(json.dumps({"case":case,"loss_and_gradients_bitexact":True}),flush=True)
            del arms,gradients,data
    except BaseException as error:
        errors.append({"type":type(error).__name__,"message":str(error),"case":dict(current)})
    finally:
        stop.set(); thread.join(timeout=2.)
        torch.cuda.empty_cache()
    critical_mid = [event for event in cleanup if event["case"] == "critical_original_shape" and event["adapter"] == "native4" and event["phase"] == MID_PHASE]
    reclaimed = sum(max(0,event["reserved_reclaimed_bytes"]) for event in critical_mid)
    passed = len(results) == len(CASES) and not errors and stopped_reason[0] is None
    return {"status":"complete" if passed else "stopped_or_failed", "passed":passed,
        "actual_GPU_or_ML_execution":True,"actual_probe_pid":os.getpid(),"executable":sys.executable,
        "actual_GPU_forward_backward_executed":bool(results),"all_requested_cases_completed":len(results)==len(CASES),
        "actual_ASR_all_pool_B8_or_phase2_admission":False,"synthetic_joint_hidden":4,
        "midpoint_inactive_storage_reclaim_observed":reclaimed>0,"critical_mid_reclaimed_reserved_bytes_sum":reclaimed,
        "memory_improvement_guaranteed":False,"preload":preload_evidence,"BELOW_NORMAL_priority_applied":priority,
        "resource_policy":dict(POLICY),"resource_stop_reason":stopped_reason[0],"elapsed_seconds":time.monotonic()-started,
        "cases":results,"operation_observations":observations,"resource_observations":pressure,
        "cleanup_observations":cleanup,"errors":errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--gpu-probe-authorized",action="store_true")
    parser.add_argument("--parent-closed-confirmed",action="store_true")
    parser.add_argument("--max-wall-seconds",type=float,default=180.)
    parser.add_argument("--report",type=Path,required=True)
    args = parser.parse_args()
    require(not args.report.exists(),"Fresh exclusive probe report required")
    require(0 < args.max_wall_seconds <= 180.,"Probe bounded to at most3 minutes")
    require(args.report.resolve().is_relative_to(ROOT),"Report must remain in this workspace")
    prepared = source_preflight()
    if args.prepare_only:
        require(not args.gpu_probe_authorized and not args.parent_closed_confirmed,"Offline preparation is a separate route")
        require("torch" not in sys.modules,"No ML execution in preparation")
        result = {"status":"CPU_stdlib_prepared_GPU_unmeasured","passed":True,"actual_GPU_forward_backward_executed":False}
    else: result = run_gpu(args,prepared)
    unchanged = all(sha(HERE/name)==digest for name,digest in PINS.items())
    report = {"schema":1,"kind":"native_mid_split_cache_actual_GPU_comparison_v1",**prepared,**result,
              "probe_source_sha256":sha(__file__),"all_source_files_unchanged":unchanged}
    report["passed"] = report["passed"] and unchanged
    args.report.parent.mkdir(parents=True,exist_ok=True)
    with args.report.open("x",encoding="utf-8") as stream:
        json.dump(report,stream,indent=2,allow_nan=False);stream.write("\n")
    print(json.dumps({"status":report["status"],"passed":report["passed"],"report":str(args.report)}),flush=True)
    if not report["passed"]: raise SystemExit(1)


if __name__ == "__main__": main()
