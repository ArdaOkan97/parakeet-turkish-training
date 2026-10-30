"""Small controller/metadata writer only. No process, thread, profiler, GC or model API."""
import json,math
from pathlib import Path
def require(v,m):
 if not v:raise ValueError(m)
class FixedCounterWindow:
 def __init__(self,start):
  require(type(start)in(int,float)and math.isfinite(start),'Finite explicit counter-start clock')
  self.start=start;self.deadline=start+60.;self.next_due=start;self.polls=0;self.last=start
 def phase(self,now,profiler_returncode):
  require(type(now)in(int,float)and math.isfinite(now)and now>=self.last,'Finite monotonic controller observation');self.last=now
  require(profiler_returncode is None or profiler_returncode==0,'Profiler failure is not a complete observation')
  if now<self.deadline:return 'counter_window'
  return 'wait_profiler_only'if profiler_returncode is None else'done'
 def due(self,now):return now<self.deadline and now>=self.next_due
 def recorded(self,query_start):
  require(self.due(query_start)and self.polls<700,'Original700 cap and60s counter-start deadline')
  self.polls+=1;self.next_due=self.start+self.polls*.1
 def sleep_seconds(self,now):return max(0.,min(.1,self.deadline-now,self.next_due-now))

class BoundedScalarFiles:
 opened=0;closed_count=0
 NAMES={'preconditions':('counter_poll_preconditions.jsonl',1048576),'counters':('thread_CPU_counters.jsonl',33554432),'brackets':('counter_poll_brackets.jsonl',1048576)}
 def __init__(self,folder):
  self.folder=Path(folder);self.handles={};self.sizes={};self.rows={};self.closed=False
  try:
   for key,(name,cap)in self.NAMES.items():
    self.handles[key]=(self.folder/name).open('xb');BoundedScalarFiles.opened+=1;self.sizes[key]=0;self.rows[key]=0
  except BaseException:
   for f in self.handles.values():
    if not f.closed:f.close();BoundedScalarFiles.closed_count+=1
   self.closed=True;raise
 def append(self,key,rows):
  require(not self.closed and key in self.NAMES,'Owned scalar stream open/key exact')
  require(type(rows)is list and all(type(r)is dict and all(type(v)in(int,float,bool,str,type(None))for v in r.values())for r in rows),'Only bounded flat scalar records, no frames/payload')
  data=b''.join((json.dumps(row,separators=(',',':'))+'\n').encode('utf-8')for row in rows)
  require(self.sizes[key]+len(data)<=self.NAMES[key][1],'Original32MiB counters/new1MiB bounded diagnostic scalar stream cap')
  self.handles[key].write(data);self.handles[key].flush();self.sizes[key]+=len(data);self.rows[key]+=len(rows)
 def close(self):
  require(not self.closed,'No double scalar-file close');errors=[]
  for key,f in self.handles.items():
   try:
    if not f.closed:f.close();BoundedScalarFiles.closed_count+=1
   except BaseException as e:errors.append((key,type(e).__name__,str(e)))
  self.closed=True;require(not errors,'All owned scalar files closed: '+repr(errors))
