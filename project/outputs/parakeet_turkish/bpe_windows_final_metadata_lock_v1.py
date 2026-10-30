"""Hold final E metadata immutable while its validated readonly view is used."""
import ctypes,os
from ctypes import wintypes
from pathlib import Path
from bpe_broad_production_state_v1 import require

ROOT=Path(r'E:\gpu_training_scratchpad')

class FinalMetadataReadLocks:
    """Windows share-read-only handles: reject open writers and replacement.

    Acquire only after genuine finalizer closure. These handles do not pause,
    close or change another process; a conflicting handle makes admission fail.
    """
    def __init__(self,paths):
        require(os.name=='nt','This final metadata lifetime policy is Windows-specific')
        self.handles=[];self.paths=[Path(p).resolve() for p in paths]
        require(self.paths and len(set(self.paths))==len(self.paths)
            and all(p.is_relative_to(ROOT.resolve()) and p.is_file() for p in self.paths),'Unique existing E-only final metadata files')
        self.api=ctypes.WinDLL('kernel32',use_last_error=True)
        self.api.CreateFileW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,ctypes.c_void_p,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
        self.api.CreateFileW.restype=wintypes.HANDLE
        self.api.CloseHandle.argtypes=[wintypes.HANDLE];self.api.CloseHandle.restype=wintypes.BOOL
        invalid=ctypes.c_void_p(-1).value
        try:
            for path in self.paths:
                handle=self.api.CreateFileW(str(path),0x80000000,0x1,None,3,0x80,None) # GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING
                if handle==invalid:raise ctypes.WinError(ctypes.get_last_error())
                self.handles.append(handle)
        except BaseException:self.close();raise

    def close(self):
        while self.handles:
            handle=self.handles.pop()
            if not self.api.CloseHandle(handle):raise ctypes.WinError(ctypes.get_last_error())

    def __enter__(self):return self
    def __exit__(self,*exc):self.close()
