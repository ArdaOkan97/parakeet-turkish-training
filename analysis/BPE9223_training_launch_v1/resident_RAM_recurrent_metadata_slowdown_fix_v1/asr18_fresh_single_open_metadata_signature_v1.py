"""Inactive narrow Windows metadata signature candidate.

Every fast call opens the CURRENT pathname, obtains fresh canonical name and
file metadata on that same handle, and closes it. Nothing caches a successful
check, size or mtime. Unproved path/API cases execute the ORIGINAL function.
"""
import ctypes,os
from ctypes import wintypes
from pathlib import Path
from bpe_declared_geometry_index_v1 import signature as ORIGINAL_SIGNATURE

FILE_READ_ATTRIBUTES=0x80
FILE_SHARE_READ_WRITE_DELETE=0x7
OPEN_EXISTING=3
FILE_ATTRIBUTE_NORMAL=0x80
FILE_ATTRIBUTE_DIRECTORY=0x10
FILE_ATTRIBUTE_REPARSE_POINT=0x400
INVALID_HANDLE_VALUE=ctypes.c_void_p(-1).value
FILETIME_UNIX_EPOCH=116444736000000000
MAX_FINAL_PATH_CHARS=32768

class FILETIME(ctypes.Structure):
    _fields_=[('dwLowDateTime',wintypes.DWORD),('dwHighDateTime',wintypes.DWORD)]

class BY_HANDLE_FILE_INFORMATION(ctypes.Structure):
    _fields_=[('dwFileAttributes',wintypes.DWORD),('ftCreationTime',FILETIME),
        ('ftLastAccessTime',FILETIME),('ftLastWriteTime',FILETIME),
        ('dwVolumeSerialNumber',wintypes.DWORD),('nFileSizeHigh',wintypes.DWORD),
        ('nFileSizeLow',wintypes.DWORD),('nNumberOfLinks',wintypes.DWORD),
        ('nFileIndexHigh',wintypes.DWORD),('nFileIndexLow',wintypes.DWORD)]

class UnsupportedFastSignature(Exception):pass
class SignatureHandleCloseFailure(RuntimeError):pass

def filetime_ns(value):
    return (((int(value.dwHighDateTime)<<32)|int(value.dwLowDateTime))-FILETIME_UNIX_EPOCH)*100

class FreshSingleOpenSignature:
    def __init__(self,expected_signatures,*,original=ORIGINAL_SIGNATURE):
        if os.name!='nt':raise RuntimeError('Windows metadata candidate only')
        if original is not ORIGINAL_SIGNATURE:raise ValueError('Exact original signature function required')
        if type(expected_signatures)is not list or len(expected_signatures)!=3:
            raise ValueError('Only the original three continuously locked metadata files')
        self.original=original;self.allowed={}
        for expected in expected_signatures:
            if type(expected)is not dict or set(expected)!={'path','bytes','mtime_ns'}:
                raise ValueError('Original exact three-field metadata signature')
            actual=original(expected['path'])
            if actual!=expected:raise ValueError('Original metadata signature changed before candidate scope')
            canonical=actual['path'];p=Path(canonical)
            # Only canonical NORMAL E drive names already produced by ORIGINAL.
            # UNC/device/extended input and all other paths always use ORIGINAL.
            if (type(canonical)is str and p.is_absolute() and p.drive=='E:' and p.root=='\\'
                and str(p)==canonical and not canonical.startswith('\\\\')
                and '/'not in canonical and '\x00'not in canonical and len(canonical)<MAX_FINAL_PATH_CHARS-5):
                self.allowed[canonical]=canonical
        if len(set(v['path']for v in expected_signatures))!=3:raise ValueError('Distinct original metadata files')
        self.api=ctypes.WinDLL('kernel32',use_last_error=True)
        self.api.CreateFileW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,ctypes.c_void_p,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
        self.api.CreateFileW.restype=wintypes.HANDLE
        self.api.GetFinalPathNameByHandleW.argtypes=[wintypes.HANDLE,wintypes.LPWSTR,wintypes.DWORD,wintypes.DWORD]
        self.api.GetFinalPathNameByHandleW.restype=wintypes.DWORD
        self.api.GetFileInformationByHandle.argtypes=[wintypes.HANDLE,ctypes.POINTER(BY_HANDLE_FILE_INFORMATION)]
        self.api.GetFileInformationByHandle.restype=wintypes.BOOL
        self.api.CloseHandle.argtypes=[wintypes.HANDLE];self.api.CloseHandle.restype=wintypes.BOOL

    def _fresh_query(self,key):
        handle=self.api.CreateFileW(key,FILE_READ_ATTRIBUTES,FILE_SHARE_READ_WRITE_DELETE,None,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,None)
        if handle==INVALID_HANDLE_VALUE:raise ctypes.WinError(ctypes.get_last_error())
        try:
            # Flags0: FILE_NAME_NORMALIZED | VOLUME_NAME_DOS. No reparse-point-open flag.
            needed=self.api.GetFinalPathNameByHandleW(handle,None,0,0)
            if not needed:raise ctypes.WinError(ctypes.get_last_error())
            if needed>=MAX_FINAL_PATH_CHARS:raise UnsupportedFastSignature('Bounded normal final pathname')
            buffer=ctypes.create_unicode_buffer(needed+1)
            count=self.api.GetFinalPathNameByHandleW(handle,buffer,len(buffer),0)
            if not count:raise ctypes.WinError(ctypes.get_last_error())
            if count>=len(buffer):raise UnsupportedFastSignature('Final pathname changed length')
            final=buffer.value
            # This is NOT generic prefix removal. The fresh handle opened key,
            # and its normalized DOS final name must match that exact ORIGINAL
            # canonical normal spelling. Any case/alias/parent-retarget/format
            # mismatch delegates the full ORIGINAL resolve+stat semantics.
            if final!='\\\\?\\'+key:raise UnsupportedFastSignature('Canonical spelling/identity requires original path resolution')
            value=BY_HANDLE_FILE_INFORMATION()
            if not self.api.GetFileInformationByHandle(handle,ctypes.byref(value)):
                raise ctypes.WinError(ctypes.get_last_error())
            if value.dwFileAttributes&(FILE_ATTRIBUTE_DIRECTORY|FILE_ATTRIBUTE_REPARSE_POINT):
                raise UnsupportedFastSignature('Original regular metadata target only')
            return {'path':key,'bytes':(int(value.nFileSizeHigh)<<32)|int(value.nFileSizeLow),
                'mtime_ns':filetime_ns(value.ftLastWriteTime)}
        finally:
            if not self.api.CloseHandle(handle):
                # Never turn a leaked/failed owned-handle close into success
                # through fallback. This is a new explicit cleanup failure.
                raise SignatureHandleCloseFailure(str(ctypes.WinError(ctypes.get_last_error())))

    def signature(self,path):
        try:key=os.fspath(path)
        except TypeError:return self.original(path)
        if type(key)is not str or key not in self.allowed:return self.original(path)
        try:return self._fresh_query(key)
        except (OSError,UnsupportedFastSignature):
            # Errors are raised by the ORIGINAL call itself; no cached success,
            # generic tolerance, changed caller messages or caught guard errors.
            return self.original(path)
