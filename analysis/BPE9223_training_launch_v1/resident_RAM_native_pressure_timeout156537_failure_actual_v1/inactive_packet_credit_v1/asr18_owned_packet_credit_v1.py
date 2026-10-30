"""INACTIVE owned Win32 logical-packet credit; no physical-copy/fit/saving claim.

API provenance: Microsoft Learn CreateSemaphoreW, ReleaseSemaphore,
CreateEventW, OpenSemaphoreW, OpenEventW, WaitForMultipleObjects, CloseHandle.
One new600s transport deadline, distinct from unchanged300s admission waits.
"""
import ctypes
from ctypes import wintypes as W
import hashlib,json,os,time,uuid
from pathlib import Path

MAXIMUM_WAIT_SECONDS=600
POLL_MILLISECONDS=250
SYNCHRONIZE=0x00100000
QUERY_LIMITED_INFORMATION=0x1000
WAIT_TIMEOUT=258
ERROR_ALREADY_EXISTS=183

def _kernel():
    if os.name!='nt':raise RuntimeError('Owned Windows packet credit only')
    k=ctypes.WinDLL('kernel32',use_last_error=True)
    declarations={
      'CreateSemaphoreW':([ctypes.c_void_p,W.LONG,W.LONG,W.LPCWSTR],W.HANDLE),
      'OpenSemaphoreW':([W.DWORD,W.BOOL,W.LPCWSTR],W.HANDLE),
      'ReleaseSemaphore':([W.HANDLE,W.LONG,ctypes.POINTER(W.LONG)],W.BOOL),
      'CreateEventW':([ctypes.c_void_p,W.BOOL,W.BOOL,W.LPCWSTR],W.HANDLE),
      'OpenEventW':([W.DWORD,W.BOOL,W.LPCWSTR],W.HANDLE),
      'SetEvent':([W.HANDLE],W.BOOL),'CloseHandle':([W.HANDLE],W.BOOL),
      'WaitForSingleObject':([W.HANDLE,W.DWORD],W.DWORD),
      'WaitForMultipleObjects':([W.DWORD,ctypes.POINTER(W.HANDLE),W.BOOL,W.DWORD],W.DWORD),
      'OpenProcess':([W.DWORD,W.BOOL,W.DWORD],W.HANDLE),
      'GetProcessTimes':([W.HANDLE,ctypes.POINTER(W.FILETIME),ctypes.POINTER(W.FILETIME),ctypes.POINTER(W.FILETIME),ctypes.POINTER(W.FILETIME)],W.BOOL)}
    for name,(args,result) in declarations.items():
        fn=getattr(k,name);fn.argtypes=args;fn.restype=result
    return k

def _ok(value):
    if not value:raise ctypes.WinError(ctypes.get_last_error())
    return value

def source_pin():
    p=Path(__file__).resolve();b=p.read_bytes()
    return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}

def _ctime(k,handle):
    created,exited,kernel,user=W.FILETIME(),W.FILETIME(),W.FILETIME(),W.FILETIME()
    _ok(k.GetProcessTimes(handle,ctypes.byref(created),ctypes.byref(exited),ctypes.byref(kernel),ctypes.byref(user)))
    return ((created.dwHighDateTime<<32)|created.dwLowDateTime)/10000000.-11644473600.

class _Handles:
    def __init__(self):self.kernel=_kernel();self.handles={};self.closed=False
    def _keep(self,name,value):self.handles[name]=_ok(value);return value
    def close(self):
        # Never call while another thread is waiting on these same handles.
        errors=[]
        for name,handle in list(self.handles.items()):
            if self.kernel.CloseHandle(handle):del self.handles[name]
            else:errors.append((name,ctypes.get_last_error()))
        self.closed=not self.handles
        if errors:raise OSError('Actual packet-credit handle closure failed',errors)
        return {'all_owned_kernel_handles_closed':self.closed,'remaining_owned_handle_count':len(self.handles)}

class PacketCreditOwner(_Handles):
    def __init__(self):
        super().__init__();self.posts=0;self.cancelled=False
        nonce=uuid.uuid4().hex
        prefix='Local\\ASR18_packet_'+str(os.getpid())+'_'+nonce
        self.descriptor={'kind':'ASR18_owned_single_packet_credit_v1','semaphore_name':prefix+'_credit',
            'cancel_name':prefix+'_cancel','nonce':nonce,'logical_packet_maximum':1,
            'credit_wait_deadline_seconds':MAXIMUM_WAIT_SECONDS,'helper_source':source_pin()}
        try:
            own=self._keep('owner_process',self.kernel.OpenProcess(SYNCHRONIZE|QUERY_LIMITED_INFORMATION,False,os.getpid()))
            self.descriptor['owner']={'pid':os.getpid(),'create_time':_ctime(self.kernel,own)}
            ctypes.set_last_error(0)
            self._keep('credit',self.kernel.CreateSemaphoreW(None,1,1,self.descriptor['semaphore_name']))
            if ctypes.get_last_error()==ERROR_ALREADY_EXISTS:raise ValueError('Never adopt a pre-existing packet credit')
            ctypes.set_last_error(0)
            self._keep('cancel',self.kernel.CreateEventW(None,True,False,self.descriptor['cancel_name']))
            if ctypes.get_last_error()==ERROR_ALREADY_EXISTS:raise ValueError('Never adopt a pre-existing packet cancellation event')
        except BaseException:
            self.close();raise
    def cancel(self):
        if not self.closed:_ok(self.kernel.SetEvent(self.handles['cancel']));self.cancelled=True
    def return_credit(self):
        if self.closed or self.cancelled:raise RuntimeError('Never post cancelled or closed packet credit')
        previous=W.LONG()
        _ok(self.kernel.ReleaseSemaphore(self.handles['credit'],1,ctypes.byref(previous)))
        if previous.value!=0:raise RuntimeError('Kernel credit did not transition exactly0 to1')
        self.posts+=1
    def receipt(self):
        return {'descriptor':self.descriptor,'successful_release_posts':self.posts,'cancelled':self.cancelled,
                'all_owned_kernel_handles_closed':self.closed,'remaining_owned_handle_count':len(self.handles)}

class PacketCreditConsumer(_Handles):
    def __init__(self,descriptor):
        super().__init__();self.descriptor=descriptor;self.acquires=0;self.last_admission=None
        if (set(descriptor)!={'kind','semaphore_name','cancel_name','nonce','logical_packet_maximum',
              'credit_wait_deadline_seconds','helper_source','owner'} or descriptor['kind']!='ASR18_owned_single_packet_credit_v1'
            or descriptor['logical_packet_maximum']!=1 or descriptor['credit_wait_deadline_seconds']!=600
            or descriptor['helper_source']!=source_pin() or len(descriptor['nonce'])!=32):
            raise ValueError('Exact immutable owned packet-credit descriptor')
        prefix='Local\\ASR18_packet_'+str(descriptor['owner']['pid'])+'_'+descriptor['nonce']
        if descriptor['semaphore_name']!=prefix+'_credit' or descriptor['cancel_name']!=prefix+'_cancel':
            raise ValueError('Exact private owner/nonce kernel object names')
        try:
            self._keep('credit',self.kernel.OpenSemaphoreW(SYNCHRONIZE,False,descriptor['semaphore_name']))
            self._keep('cancel',self.kernel.OpenEventW(SYNCHRONIZE,False,descriptor['cancel_name']))
            parent=self._keep('owner_process',self.kernel.OpenProcess(SYNCHRONIZE|QUERY_LIMITED_INFORMATION,False,descriptor['owner']['pid']))
            if abs(_ctime(self.kernel,parent)-descriptor['owner']['create_time'])>=.001:
                raise ValueError('Exact actual packet-credit owner identity; no PID reuse')
        except BaseException:
            self.close();raise
    def acquire(self,*,phase,token,check,clock=time.monotonic):
        began=clock();observations=0;last=None
        handles=(W.HANDLE*3)(self.handles['cancel'],self.handles['owner_process'],self.handles['credit'])
        while True:
            last=check();observations+=1;elapsed=clock()-began
            if elapsed>=MAXIMUM_WAIT_SECONDS:
                error=TimeoutError('Bounded owned packet-credit wait exhausted; preserve failure, no unchanged retry')
                error.packet_credit_timeout={'kind':'ASR18_actual_owned_packet_credit_timeout_v1',
                    'phase':phase,'token':token,'observed_wait_seconds':elapsed,'maximum_wait_seconds':600,
                    'observations':observations,'last_guard_observation':last,'owner':self.descriptor['owner']}
                raise error
            wait=min(POLL_MILLISECONDS,max(1,int((MAXIMUM_WAIT_SECONDS-elapsed)*1000)))
            code=self.kernel.WaitForMultipleObjects(3,handles,False,wait)
            if code==WAIT_TIMEOUT:continue
            if code==0:raise RuntimeError('Owned packet-credit preparation cancelled; no normal exit claimed')
            if code==1:raise RuntimeError('Exact packet-credit owner exited; no orphan preparation')
            if code!=2:raise ctypes.WinError(ctypes.get_last_error())
            # Cancellation/owner loss takes precedence even if it races acquisition.
            pending=self.kernel.WaitForMultipleObjects(2,handles,False,0)
            if pending==0:raise RuntimeError('Owned packet-credit preparation cancelled after acquire')
            if pending==1:raise RuntimeError('Exact packet-credit owner exited after acquire')
            if pending!=WAIT_TIMEOUT:raise ctypes.WinError(ctypes.get_last_error())
            check();self.acquires+=1
            self.last_admission={'phase':phase,'token':token,'observations':observations,'wait_seconds':clock()-began}
            return self.last_admission
    def receipt(self):
        return {'descriptor':self.descriptor,'successful_preparation_acquires':self.acquires,
                'last_credit_admission':self.last_admission,'all_owned_kernel_handles_closed':self.closed,
                'remaining_owned_handle_count':len(self.handles)}
