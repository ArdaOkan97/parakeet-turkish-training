"""One fresh CPU-only diagnostic import arm; never constructs a model or patches files."""
import argparse
import ctypes
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
ROOT = Path(r"E:\gpu_training_scratchpad")
sys.path.insert(0, str(PROJECT / "outputs/parakeet_turkish"))
from resource_watchdog_v2 import sample_memory

OPTIONAL = (
    "is_librosa_available", "is_soundfile_available", "is_torchaudio_available",
    "is_torchvision_available", "is_torchvision_v2_available", "is_vision_available",
    "is_av_available", "is_cv2_available", "is_decord_available",
    "is_torchcodec_available", "is_yt_dlp_available",
)
PACKAGES = ("librosa", "soxr", "soundfile", "torchaudio", "torchvision", "PIL", "av", "cv2", "decord", "torchcodec", "yt_dlp")


def record(path):
    path = Path(path).resolve()
    assert path.is_relative_to(PROJECT) or path.is_relative_to(ROOT)
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("original", "native_only"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    assert output.is_relative_to(ROOT)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1"
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        assert os.environ[key] == "1"
    assert shutil.disk_usage(ROOT).free >= 100_000_000_000
    began = time.perf_counter()
    stages = []

    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
            (name, ctypes.c_size_t) for name in (
                "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage", "PrivateUsage")]

    def snap(label):
        value = sample_memory(os.getpid(), label)
        worker, system = value["worker_private_memory"], value["system_memory"]
        assert worker["available"] and system["available"] and value["paired_sampling_within_one_second"]
        assert worker["private_bytes"] < 2_000_000_000
        assert system["commit_used_bytes"] / system["commit_limit_bytes"] < .90
        assert system["commit_limit_bytes"] - system["commit_used_bytes"] >= 2_000_000_000
        count = Counters()
        count.cb = ctypes.sizeof(count)
        get_current = ctypes.windll.kernel32.GetCurrentProcess
        get_current.restype = ctypes.c_void_p
        get_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_info.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
        assert get_info(get_current(), ctypes.byref(count), count.cb)
        assert count.PeakPagefileUsage < 2_000_000_000
        stages.append({"label": label, "wall_seconds": time.perf_counter() - began,
                       "private_bytes": count.PrivateUsage, "lifetime_peak_private_commit_bytes": count.PeakPagefileUsage,
                       "paired_sample": value, "optional_loaded": {p: any(x == p or x.startswith(p + ".") for x in sys.modules) for p in PACKAGES}})

    snap("before_Torch")
    import torch
    assert not torch.cuda.is_initialized()
    snap("after_Torch")
    import transformers
    import transformers.utils as utils
    import transformers.utils.import_utils as availability
    snap("after_Transformers_package_before_policy")
    changed = []
    if args.arm == "native_only":
        # Installed dev version exposes cached functions, not *_available flags.
        # Replace only process-local aliases of these exact optional availability
        # callables, including BACKENDS_MAPPING references made during package import.
        # Tensor model classes, Torch/scipy/NumPy and all installed files stay intact.
        originals = {name: getattr(availability, name) for name in OPTIONAL}
        for name, value in originals.items():
            assert callable(value), name
        identities = {id(value): name for name, value in originals.items()}
        def unavailable():
            return False
        for module_name, module in list(sys.modules.items()):
            if module is None or not (module_name == "transformers" or module_name.startswith("transformers.")):
                continue
            for attribute, value in list(vars(module).items()):
                if id(value) in identities:
                    setattr(module, attribute, unavailable)
                    changed.append(module_name + "." + attribute)
        for backend, (check, error) in list(availability.BACKENDS_MAPPING.items()):
            if id(check) in identities:
                availability.BACKENDS_MAPPING[backend] = (unavailable, error)
                changed.append("BACKENDS_MAPPING." + backend)
        assert all(getattr(availability, name)() is False for name in OPTIONAL)
        assert availability.is_torch_available()
    snap("before_Parakeet_class_import")
    from transformers.models.parakeet.modeling_parakeet import ParakeetForTDT, ParakeetEncoderSubsamplingConv2D
    assert not torch.cuda.is_initialized()
    snap("after_Parakeet_class_import")
    class_records = {cls.__name__: {"module": cls.__module__, "qualname": cls.__qualname__,
                                  "source": record(inspect.getsourcefile(cls)),
                                  "class_source_sha256": hashlib.sha256(inspect.getsource(cls).encode()).hexdigest()}
                     for cls in (ParakeetForTDT, ParakeetEncoderSubsamplingConv2D)}
    result = {"kind": "ASR18_CPU_optional_multimedia_import_arm_v1", "arm": args.arm,
              "pid": os.getpid(), "ppid": os.getppid(), "executable": record(sys.executable),
              "script": record(__file__), "torch_version": torch.__version__, "transformers_version": transformers.__version__,
              "availability_source": record(availability.__file__), "policy_function_names": list(OPTIONAL),
              "changed_process_local_aliases": changed, "stages": stages, "class_records": class_records,
              "no_model_instantiated": True, "no_audio_loaded": True, "no_CUDA_context": True,
              "installed_files_modified": False, "diagnostic_only": True,
              "limitation": "Native tensor-only import candidate; audio/image processor APIs intentionally unavailable in native_only arm. No model math/state/quality or integrated memory forecast proof."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"arm": args.arm, "pid": os.getpid(), "after_private": stages[-1]["private_bytes"],
                      "wall_seconds": stages[-1]["wall_seconds"], "output": str(output)}))


if __name__ == "__main__":
    main()
