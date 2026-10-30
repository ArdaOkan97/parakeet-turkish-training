"""Inactive Windows stop check: one real filesystem query per original call.

No polling interval, cached answer, numerical import or process launch. Only
ERROR_FILE_NOT_FOUND means absent; every other error stops the owned worker.
File/directory/reparse entries all count as an actual stop request.
"""
import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import stat


class OwnedStopFlag:
    def __init__(self, folder):
        if os.name != "nt":
            raise RuntimeError("Windows stop flag requires Windows")
        given = Path(folder).absolute()
        actual = given.resolve(strict=True)
        root = Path(r"E:\gpu_training_scratchpad").resolve(strict=True)
        if (not actual.is_relative_to(root)
                or os.path.normcase(str(given)) != os.path.normcase(str(actual))):
            raise ValueError("Exact existing owned E directory required")
        info = actual.lstat()
        if (not stat.S_ISDIR(info.st_mode)
                or info.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT):
            raise ValueError("Ordinary owned stop directory required")
        self.path = str(actual / "stop.json")
        if len(self.path) >= 248:
            self.path = "\\\\?\\" + self.path
        self.pid = os.getpid()
        self._kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self._query = self._kernel.GetFileAttributesW
        self._query.argtypes = [wintypes.LPCWSTR]
        self._query.restype = wintypes.DWORD

    def exists(self):
        if os.getpid() != self.pid:
            raise RuntimeError("Stop flag belongs to a different process")
        value = self._query(self.path)
        if value != 0xFFFFFFFF:
            return True
        error = ctypes.get_last_error()
        if error == 2:  # ERROR_FILE_NOT_FOUND, never PATH_NOT_FOUND or access errors.
            return False
        raise ctypes.WinError(error)

    def check(self):
        if self.exists():
            raise ValueError("Actual owned stop request; preserve partial failure")
