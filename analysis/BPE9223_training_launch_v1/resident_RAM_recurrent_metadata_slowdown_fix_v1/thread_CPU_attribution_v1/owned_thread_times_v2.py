"""Read-only thread CPU counters. No thread suspend, signal, injection or GC call."""
import ctypes, re, threading, time
from ctypes import wintypes

QUERY=0x0800
_k=ctypes.WinDLL('kernel32',use_last_error=True)
_k.OpenThread.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];_k.OpenThread.restype=wintypes.HANDLE
_k.GetProcessIdOfThread.argtypes=[wintypes.HANDLE];_k.GetProcessIdOfThread.restype=wintypes.DWORD
_k.GetThreadTimes.argtypes=[wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4;_k.GetThreadTimes.restype=wintypes.BOOL
_k.CloseHandle.argtypes=[wintypes.HANDLE];_k.CloseHandle.restype=wintypes.BOOL

def require(v,m):
 if not v:raise ValueError(m)
def ft(v):return (int(v.dwHighDateTime)<<32)|int(v.dwLowDateTime)
def raw_times(handle):
 c,e,k,u=(wintypes.FILETIME()for _ in range(4))
 if not _k.GetThreadTimes(handle,ctypes.byref(c),ctypes.byref(e),ctypes.byref(k),ctypes.byref(u)):raise ctypes.WinError(ctypes.get_last_error())
 return {'creation_filetime_100ns':ft(c),'exit_filetime_100ns':ft(e),'kernel_100ns':ft(k),'user_100ns':ft(u)}

class ThreadQuery:
 opened=0;closed_count=0
 def __init__(self,pid,tid,expected_creation=None):
  require(type(pid)is int and pid>0 and type(tid)is int and tid>0,'Positive actual PID/nativeTID')
  self.pid=pid;self.tid=tid;self.lock=threading.RLock();self.closed=False;self.last=None
  self.handle=_k.OpenThread(QUERY,False,tid)
  if not self.handle:raise ctypes.WinError(ctypes.get_last_error())
  ThreadQuery.opened+=1
  try:
   actual_pid=int(_k.GetProcessIdOfThread(self.handle))
   if not actual_pid:raise ctypes.WinError(ctypes.get_last_error())
   require(actual_pid==pid,'Actual retained thread handle belongs to exact expected process')
   value=raw_times(self.handle);self.creation=value['creation_filetime_100ns']
   require(expected_creation is None or(type(expected_creation)is int and expected_creation==self.creation),'Exact nativeTID creation-time identity, no reuse normalization')
   self.last=value
  except BaseException:
   self.close();raise
 def sample(self):
  with self.lock:
   require(not self.closed,'Owned thread query is closed')
   value=raw_times(self.handle)
   require(value['creation_filetime_100ns']==self.creation,'Same retained thread handle creation identity')
   require(value['user_100ns']>=self.last['user_100ns']and value['kernel_100ns']>=self.last['kernel_100ns'],'Same-handle CPU counters monotonic')
   if self.last['exit_filetime_100ns']:require(value==self.last,'Exited retained thread counters remain stable')
   self.last=value
   return {'pid':self.pid,'native_tid':self.tid,**value}
 def close(self):
  with self.lock:
   require(not self.closed,'No double-close of owned handle')
   if not _k.CloseHandle(self.handle):raise ctypes.WinError(ctypes.get_last_error())
   self.closed=True;ThreadQuery.closed_count+=1

class ThreadSampler:
 """Fixed limits; no frames retained. Caller supplies trusted target identity checks."""
 def __init__(self,pid,*,max_threads=128,max_identities=256,max_polls=700,max_records=80000):
  require(1<=max_threads<=128 and 1<=max_identities<=256 and 1<=max_polls<=700 and 1<=max_records<=80000,'Frozen bounded sampler limits')
  self.pid=pid;self.limits=(max_threads,max_identities,max_polls,max_records)
  self.active={};self.records=[];self.identities={};self.polls=0;self.closed=False;self.failures=[];self.poll_brackets=[]
 def poll(self,tids):
  require(not self.closed,'Sampler not closed');self.polls+=1
  require(self.polls<=self.limits[2]and len(tids)<=self.limits[0]and len(set(tids))==len(tids),'Bounded unique current nativeTID list/polls')
  now=time.monotonic_ns();observed=[]
  # Retained handles are sampled even after a thread disappears from enumeration.
  for tid,q in list(self.active.items()):
   value=q.sample();observed.append(value)
   if value['exit_filetime_100ns']:
    q.close();del self.active[tid]
  for tid in tids:
   if tid in self.active:continue
   try:q=ThreadQuery(self.pid,tid)
   except OSError as e:
    # Enumeration/open races remain explicit, never a zero-CPU observation.
    self.failures.append({'poll':self.polls,'native_tid':tid,'error':type(e).__name__,'winerror':e.winerror,'message':str(e)})
    require(len(self.failures)<=256,'Bounded terminated/open-race records');continue
   key=(tid,q.creation)
   if key in self.identities:
    q.close();self.failures.append({'poll':self.polls,'native_tid':tid,'error':'AlreadyObservedExitedIdentity','message':'Enumeration included prior exited identity; no duplicate CPU work'})
    require(len(self.failures)<=256,'Bounded terminated/open-race records');continue
   if len(self.identities)>=self.limits[1]:
    q.close();raise ValueError('Bounded total thread identities')
   self.identities[key]=True;self.active[tid]=q;observed.append(q.sample())
  require(len(self.records)+len(observed)<=self.limits[3],'Bounded scalar counter records')
  self.records.extend({'poll':self.polls,'monotonic_ns':now,**v}for v in observed)
  self.poll_brackets.append({'poll':self.polls,'start_monotonic_ns':now,'end_monotonic_ns':time.monotonic_ns(),'thread_records':len(observed)})
  return observed
 def close(self):
  require(not self.closed,'No duplicate sampler closure');errors=[]
  for tid,q in list(self.active.items()):
   try:q.close()
   except BaseException as e:errors.append({'tid':tid,'type':type(e).__name__,'message':str(e)})
  self.active.clear();self.closed=True
  require(not errors,'Every owned query handle closed: '+repr(errors))
 def summary(self):
  by={}
  for v in self.records:
   key=(v['native_tid'],v['creation_filetime_100ns']);p=by.setdefault(key,{'first':v,'last':v,'samples':0});p['last']=v;p['samples']+=1
  return [{'native_tid':key[0],'creation_filetime_100ns':key[1],'samples':v['samples'],
   'first_monotonic_ns':v['first']['monotonic_ns'],'last_monotonic_ns':v['last']['monotonic_ns'],
   'user_seconds_delta':(v['last']['user_100ns']-v['first']['user_100ns'])/1e7,
   'kernel_seconds_delta':(v['last']['kernel_100ns']-v['first']['kernel_100ns'])/1e7,
   'thread_exited_observed':bool(v['last']['exit_filetime_100ns'])}for key,v in sorted(by.items())]

def profile_role_mapping(profiles,thread_summary,*,role_specs):
 """No PID=TID assumption. Exact profile nativeTID and unique retained identity."""
 mapped=[];unknown=[]
 for p in profiles:
  name=p.get('name','');m=re.fullmatch(r'Thread ([1-9][0-9]*)(?: "([^"]*)")?',name)
  if not m:unknown.append({'name':name,'reason':'Unsupported profile nativeTID label'});continue
  tid=int(m.group(1));matches=[s for s in thread_summary if s['native_tid']==tid]
  if len(matches)!=1:unknown.append({'name':name,'reason':'Missing/reused/ambiguous nativeTID creation identity','identities':len(matches)});continue
  roles=[]
  for role,spec in role_specs.items():
   name_ok=spec.get('profile_name')is not None and m.group(2)==spec['profile_name']
   frame_ok=any(str(fr.get('file','')).replace('\\','/').lower()==spec.get('file','').replace('\\','/').lower()and fr.get('name')in spec.get('functions',[])for fr in p.get('sampled_frames',[]))
   if name_ok and(not spec.get('functions')or frame_ok):roles.append(role)
  if len(roles)!=1:unknown.append({'name':name,'reason':'Role missing or ambiguous','roles':roles});continue
  mapped.append({'profile_name':name,'role':roles[0],**matches[0]})
 for role in role_specs:
  if sum(x['role']==role for x in mapped)>1:
   raise ValueError('Ambiguous multiple native threads for one requested role')
 return {'mapped':mapped,'unknown':unknown,'CPU_times_not_exclusive_stack_phase_times':True}
