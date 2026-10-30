"""Fresh bounded CPU-only job guard. Does not inspect/change any other process."""
from __future__ import annotations
import ctypes
import json
import os
from pathlib import Path
import shutil
import threading
import time

ROOT = Path(r"E:\gpu_training_scratchpad")


class Guard:
    def __init__(self, *, wall_seconds=90):
        import psutil
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise RuntimeError("CUDA must be masked for this CPU-only job")
        self.process = psutil.Process()
        self.process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        self.started = time.monotonic()
        self.wall_seconds = wall_seconds
        self.peak_private = 0
        self.peak_system_commit = 0
        self.samples = 0
        self.failure = None
        self.done = threading.Event()
        self.check()
        self.monitor = threading.Thread(target=self._monitor, daemon=True)
        self.monitor.start()

    def sample(self):
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        state = MEMORYSTATUSEX()
        state.dwLength = ctypes.sizeof(state)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
            raise ctypes.WinError()
        own = self.process.memory_info().private
        commit = 1 - state.ullAvailPageFile / state.ullTotalPageFile
        self.peak_private = max(self.peak_private, own)
        self.peak_system_commit = max(self.peak_system_commit, commit)
        self.samples += 1
        return own, commit, state.ullAvailPageFile

    def _monitor(self):
        while not self.done.wait(0.05):
            try:
                own, commit, headroom = self.sample()
                if own > 2 * 1024**3 or commit >= 0.95 or headroom < 2 * 1024**3:
                    self.failure = "hard_private_or_system_or_headroom_guard"
                    # This process owns only its tiny job. Kill before an uninterruptible
                    # native transform can exceed the hard cap; never kill another owner.
                    os._exit(95)
                if time.monotonic() - self.started > self.wall_seconds:
                    self.failure = "bounded_CPU_wall_timeout"
                    os._exit(75)
            except BaseException:
                os._exit(95)

    def check(self):
        if self.failure:
            raise RuntimeError(self.failure)
        own, commit, headroom = self.sample()
        if own > 2 * 1024**3 or commit >= 0.95 or headroom < 2 * 1024**3:
            raise RuntimeError("CPU private/system/headroom guard")
        if shutil.disk_usage(ROOT).free < 100_000_000_000:
            raise RuntimeError("E reserve below 100 GB")
        # Yield without repeatedly retrying inference/computation under pressure.
        deadline = time.monotonic() + 15
        while commit >= 0.90:
            if time.monotonic() >= deadline:
                raise RuntimeError("CPU yielded at system commit >=90%; no unchanged retry")
            time.sleep(0.25)
            own, commit, headroom = self.sample()
            if own > 2 * 1024**3 or commit >= 0.95 or headroom < 2 * 1024**3:
                raise RuntimeError("CPU private/system/headroom guard while yielding")

    def report(self):
        self.check()
        return {"pid": os.getpid(), "peak_private_bytes": self.peak_private,
                "peak_system_commit_ratio": self.peak_system_commit, "memory_samples": self.samples,
                "wall_seconds": time.monotonic() - self.started, "CUDA_masked": True,
                "priority": "BelowNormal", "hard_private_bytes": 2 * 1024**3,
                "system_yield_ratio": 0.90, "system_critical_ratio": 0.95,
                "commit_headroom_bytes": 2 * 1024**3, "E_reserve_bytes": 100_000_000_000}

    def close(self):
        self.done.set()


def output_path(value):
    path = Path(value).resolve()
    if not path.is_relative_to(ROOT.resolve(strict=True)):
        raise ValueError("New artifacts must stay on designated E storage")
    path.mkdir(exist_ok=True, parents=True)
    return path
