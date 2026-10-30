"""One CPU-only first-checkpoint observation; no patches or model state."""
import argparse
import ctypes
from ctypes import wintypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

P = Path(__file__).resolve().parents[2]
E = Path(r"E:\gpu_training_scratchpad")
sys.path.insert(0, str(P / "outputs/parakeet_turkish"))
from atomic_io_retry_v1 import read_json, write_json
from bpe_broad_production_state_v1 import require
from resource_watchdog_v2 import sample_memory, PROCESS_MEMORY_COUNTERS_EX

spec = importlib.util.spec_from_file_location("observed_guard", P / "work/research/run_ASR18_frozen_CNN_lifetime_pair_v8.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    args = parser.parse_args()
    folder = Path(args.folder).resolve()
    helper.storage()
    require(folder.is_relative_to(E) and not folder.exists(), "Fresh E observation only")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "CPU-only")
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    folder.mkdir()
    watch = helper.Watch(folder, 1_800_000_000, worker=True)
    code = 1
    began = time.monotonic()
    stages = []
    modules = ("torch._dynamo", "sympy", "torch._inductor", "torch._functorch", "numba")

    def snap(label):
        watch.phase = label
        require(time.monotonic() - began < 60, "Finite CPU observation")
        value = watch.check()
        counters = PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        api = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        api.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX), wintypes.DWORD]
        require(api.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb), "OS lifetime read")
        require(counters.PrivateUsage < 1_800_000_000 and counters.PeakPagefileUsage < 1_800_000_000, "Current and lifetime leaf cap")
        stages.append({"phase": label, "sample": value, "private_bytes": int(counters.PrivateUsage),
                       "lifetime_peak_private_bytes": int(counters.PeakPagefileUsage),
                       "loaded": {name: name in sys.modules for name in modules}})
        write_json(folder / "phase_evidence.json", stages)

    try:
        admissions = []
        for _ in range(3):
            value = watch.check()
            system = value["system_memory"]
            passed = system["commit_used_bytes"] + 1_992_000_000 < .90 * system["commit_limit_bytes"]
            passed = passed and system["commit_available_bytes"] - 1_992_000_000 > 2_000_000_000
            admissions.append({"sample": value, "passed": passed})
            time.sleep(.1)
        write_json(folder / "admission.json", admissions)
        require(all(item["passed"] for item in admissions), "Whole CPU family forecast denied")
        snap("before_Torch")
        import torch
        torch.set_num_threads(1)
        require(not torch.cuda.is_initialized(), "No CUDA context")
        snap("after_Torch")
        from torch.utils.checkpoint import checkpoint
        snap("after_checkpoint_import")
        call_results = []
        for index in range(2):
            x = torch.arange(8, dtype=torch.float32).div(8).requires_grad_()
            rng = torch.get_rng_state().clone()
            snap("call_" + str(index + 1) + "_before_forward")
            output = checkpoint(lambda v: v.sin().square(), x, use_reentrant=False)
            snap("call_" + str(index + 1) + "_after_forward")
            output.sum().backward()
            snap("call_" + str(index + 1) + "_after_backward")
            require(torch.equal(rng, torch.get_rng_state()), "Torch RNG unchanged")
            require(torch.isfinite(output).all().item() and torch.isfinite(x.grad).all().item(), "Finite original checkpoint")
            call_results.append({"output": output.detach().tolist(), "gradient": x.grad.tolist()})
            require(not torch.cuda.is_initialized(), "CPU-only throughout")
        require(call_results[0] == call_results[1], "Identical repeat")
        report = {"kind": "ASR18_original_CPU_checkpoint_cold_observation_v1", "pid": os.getpid(),
                  "script": helper.record(__file__), "torch_version": torch.__version__, "stages": stages,
                  "call_results": call_results, "checkpoint_wrapped": hasattr(checkpoint, "__wrapped__"),
                  "checkpoint_code_sha256": hashlib.sha256(checkpoint.__code__.co_code).hexdigest(),
                  "actual_updates": 0, "no_CUDA_context": True, "installed_files_read_or_modified": False,
                  "wall_seconds": time.monotonic() - began, "candidate_bypass_used": False}
        write_json(folder / "worker_report.json", report)
        code = 0
    finally:
        watch.close(code)


if __name__ == "__main__":
    main()
