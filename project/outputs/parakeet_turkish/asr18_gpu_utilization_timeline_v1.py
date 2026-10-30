"""Bounded, sole-owner nvidia-smi observation; this module never starts work.

The GPU owner starts/stops the actual Popen handle and charges its PID to the
existing shim allowance. Poll cadence is not the driver's measurement window.
Global GPU memory includes other clients; no per-process attribution is made.
"""
import csv
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import threading
import time

ROOT = Path(r'E:\gpu_training_scratchpad')
FIELDS = ('timestamp', 'index', 'uuid', 'name', 'pstate', 'utilization.gpu',
          'utilization.memory', 'memory.used', 'memory.total', 'power.draw',
          'clocks.current.sm', 'clocks.current.memory')


def require(value, message):
    if not value:
        raise ValueError(message)


def record(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def creation_filetime(process):
    """Use the owned Windows handle, never reopen or terminate a guessed PID."""
    require(os.name == 'nt', 'This observer is bound to the Windows GPU host')
    api = ctypes.WinDLL('kernel32', use_last_error=True).GetProcessTimes
    api.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    api.restype = wintypes.BOOL
    values = [wintypes.FILETIME() for _ in range(4)]
    require(bool(api(wintypes.HANDLE(int(process._handle)),
                     *(ctypes.byref(value) for value in values))),
            'Actual owned observer creation time must be available')
    return (values[0].dwHighDateTime << 32) | values[0].dwLowDateTime


class GPUUtilizationTimeline:
    MAX_BYTES = 16 * 1024**2
    MAX_LINE = 4096
    MAX_SECONDS = 1800

    def __init__(self, folder, guard):
        self.folder = Path(folder).resolve()
        require(self.folder.is_relative_to(ROOT.resolve()) and self.folder.is_dir(),
                'Existing explicitly owned E observer folder required')
        for part in (self.folder, *self.folder.parents):
            info = part.lstat()
            require(not part.is_symlink() and not getattr(info, 'st_file_attributes', 0)
                    & stat.FILE_ATTRIBUTE_REPARSE_POINT, 'No observer reparse escape')
            if part == ROOT.resolve():
                break
        require(callable(guard), 'Owner guard callback required')
        self.guard = guard
        self.path = self.folder / 'gpu_utilization_timeline.jsonl'
        self.completion = self.folder / 'gpu_utilization_completion.json'
        require(not self.path.exists() and not self.completion.exists(), 'Fresh observer outputs only')
        self.process = self.reader = self.stream = None
        self.lock = threading.Lock()
        self.error = None
        self.samples = self.bytes = 0
        self.last_sample_monotonic = None
        self.reader_eof = False
        self.first_sample = threading.Event()
        self.stopping = False
        self.finished = False

    def _write(self, kind, values):
        item = {'kind': kind, 'UTC': datetime.now(timezone.utc).isoformat(),
                'monotonic': time.monotonic(), **values}
        raw = (json.dumps(item, separators=(',', ':'), ensure_ascii=False,
                          allow_nan=False) + '\n').encode('utf-8')
        with self.lock:
            require(len(raw) <= 16384 and self.bytes + len(raw) <= self.MAX_BYTES,
                    'Bounded actual observer trace')
            require(self.stream.write(raw) == len(raw), 'Complete observer row write')
            self.stream.flush()
            self.bytes += len(raw)

    def _read(self):
        try:
            while True:
                raw = self.process.stdout.readline(self.MAX_LINE + 1)
                if not raw:
                    break
                if self.stopping and len(raw) <= self.MAX_LINE and not raw.endswith(b'\n'):
                    self._write('intentional_stop_partial_non_sample',
                                {'raw_hex': raw.hex(), 'bytes': len(raw)})
                    break
                require(len(raw) <= self.MAX_LINE and raw.endswith(b'\n'),
                        'Bounded complete nvidia-smi line')
                row = next(csv.reader(io.StringIO(raw.decode('utf-8', errors='strict'))))
                require(len(row) == len(FIELDS), 'Unsupported query or malformed nvidia-smi CSV')
                values = {key: value.strip() for key, value in zip(FIELDS, row)}
                require(values['uuid'].startswith('GPU-') and values['index'].isdigit(),
                        'Actual NVIDIA identity row required')
                self._write('gpu_global_sample', {'values': values})
                self.samples += 1
                self.last_sample_monotonic = time.monotonic()
                self.first_sample.set()
        except BaseException as error:
            self.error = type(error).__name__ + ': ' + str(error)
            self.first_sample.set()
        finally:
            self.reader_eof = True

    def start(self):
        require(self.process is None and not self.finished, 'One observer start')
        self.guard()
        self.stream = self.path.open('xb')
        self.began = time.monotonic()
        argv = ['nvidia-smi', '--query-gpu=' + ','.join(FIELDS),
                '--format=csv,noheader,nounits', '--loop-ms=200']
        try:
            self.process = subprocess.Popen(argv, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS)
            self.creation = creation_filetime(self.process)
            self.producer = {'pid': self.process.pid, 'ppid': os.getpid(),
                'create_time': self.creation / 10_000_000 - 11_644_473_600,
                'creation_FILETIME_100ns': self.creation}
            self._write('observer_started', {'producer': self.producer, 'argv': argv,
                'poll_ms': 200, 'driver_measurement_window_unknown': True,
                'global_memory_is_not_per_process_memory': True})
            self.reader = threading.Thread(target=self._read, name='owned_nvidia_smi_reader', daemon=True)
            self.reader.start()
            deadline = time.monotonic() + 10
            while not self.first_sample.wait(.05):
                self.guard()
                require(time.monotonic() < deadline and self.process.poll() is None,
                        'Live observer must provide its first actual sample')
            self.poll()
            require(self.samples > 0, 'An actual valid GPU sample is required')
            return dict(self.producer)
        except BaseException:
            self.stop(check_guard=False)
            raise

    def poll(self):
        self.guard()
        require(self.process is not None and not self.finished, 'Live started observer required')
        require(self.error is None, 'Observer reader failed: ' + str(self.error))
        require(time.monotonic() - self.began <= self.MAX_SECONDS, 'Finite observer lifetime')
        require(self.process.poll() is None and creation_filetime(self.process) == self.creation,
                'Same actual observer handle must remain live')
        require(not self.reader_eof and self.last_sample_monotonic is not None
                and time.monotonic() - self.last_sample_monotonic <= 2,
                'Fresh actual observer samples, never a stale last row')
        return {'producer': dict(self.producer), 'samples': self.samples,
                'trace_bytes': self.bytes, 'trace': str(self.path)}

    def mark(self, label, **extra):
        self.poll()
        require(type(label) is str and 0 < len(label) <= 128
                and len(json.dumps(extra).encode('utf-8')) <= 4096, 'Bounded owner phase marker')
        self._write('owner_phase_marker', {'label': label, 'extra': extra})

    def stop(self, *, check_guard=True):
        require(not self.finished, 'One actual observer closure')
        self.stopping = True
        terminated = False
        if self.process is not None:
            if self.process.poll() is None:
                # Intentional stop of this exact monitor handle, not a failed
                # training process or a claim that the monitor exited normally.
                self.process.terminate()
                terminated = True
            try:
                code = self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                code = self.process.wait(timeout=5)
            if self.reader is not None:
                self.reader.join(timeout=5)
                require(not self.reader.is_alive(), 'Observer reader must actually close')
            self.process.stdout.close()
        else:
            code = None
        if self.stream is not None:
            self.stream.flush()
            os.fsync(self.stream.fileno())
            self.stream.close()
        self.finished = True
        value = {'kind': 'ASR18_actual_owned_GPU_utilization_observer_closure_v1',
            'source': record(__file__), 'producer': getattr(self, 'producer', None),
            'actual_wait_exit_code': code, 'intentional_monitor_termination': terminated,
            'actual_handle_exit_observed': self.process is not None and self.process.returncode is not None,
            'reader_joined': self.reader is None or not self.reader.is_alive(),
            'samples': self.samples, 'trace': record(self.path) if self.path.exists() else None,
            'reader_error': self.error, 'poll_ms': 200,
            'driver_measurement_window_unknown': True,
            'normal_training_process_exit_claimed': False}
        with self.completion.open('x', encoding='utf-8', newline='\n') as target:
            json.dump(value, target, indent=2, ensure_ascii=False, allow_nan=False)
            target.write('\n')
        if check_guard:
            self.guard()
            require(self.error is None and self.samples > 0, 'Complete valid observer trace required')
        return record(self.completion)
