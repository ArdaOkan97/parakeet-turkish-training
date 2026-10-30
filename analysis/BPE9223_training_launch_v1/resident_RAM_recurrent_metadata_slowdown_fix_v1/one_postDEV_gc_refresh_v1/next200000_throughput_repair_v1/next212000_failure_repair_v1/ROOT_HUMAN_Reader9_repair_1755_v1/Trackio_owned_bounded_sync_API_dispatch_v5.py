"""Reader-owned dispatch. No package edits. Import has no server/DB/thread side effect."""
import asyncio
import concurrent.futures
import hashlib
import inspect
import json
from pathlib import Path
import queue
import threading
import time
import urllib.request

def _pin(path):
    raw=Path(path).read_bytes()
    return {"path":str(Path(path).resolve()),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()}

class _OwnedSlot:
    def __init__(self,name):
        self.queue=queue.Queue(maxsize=1)
        self.gate=asyncio.Semaphore(1)
        self.thread=threading.Thread(target=self._work,name=name,daemon=False)
        self.closed=False
        self.lifecycle={"stage":"idle","owned_job_drops":0,"publication_returns":0}
    def start(self):
        if self.closed:raise RuntimeError("Owned slot is closed")
        self.thread.start()
    def _work(self):
        while True:
            job=self.queue.get()
            if job is None:
                self.queue.task_done();del job
                return
            future,call=job
            result=None;failure=None
            self.lifecycle["stage"]="running"
            try:
                try:result=call()
                except BaseException as error:failure=error
                finally:
                    # Drop owned request-bearing closures BEFORE callbacks can
                    # resume an awaiting/cancelled request on the event loop.
                    del call,job
                    self.queue.task_done()
                    self.lifecycle["owned_job_drops"]+=1
                    self.lifecycle["stage"]="owned_refs_dropped"
                if failure is None:future.set_result(result)
                else:future.set_exception(failure)
                self.lifecycle["publication_returns"]+=1
            finally:
                # Keep the original exception object/traceback for its caller;
                # release only this worker's local references, on every path.
                del future,result,failure
                self.lifecycle["stage"]="idle"
    async def invoke(self,call):
        await self.gate.acquire()
        if self.closed:
            self.gate.release()
            raise RuntimeError("Owned API slot already closed")
        future=concurrent.futures.Future()
        self.queue.put_nowait((future,call))
        wrapped=asyncio.wrap_future(future)
        released=False
        def release(done):
            # A cancelled HTTP request does not free capacity before real work ends.
            try:
                if not done.cancelled():done.exception()
            finally:self.gate.release()
        try:
            result=await asyncio.shield(wrapped)
        except asyncio.CancelledError:
            wrapped.add_done_callback(release);released=True
            raise
        finally:
            if not released:self.gate.release()
            del future,call
        if inspect.isawaitable(result):return await result
        return result
    def close(self,timeout):
        self.closed=True
        if self.thread.ident is None:return
        self.queue.put(None,timeout=timeout)
        self.thread.join(timeout)
        if self.thread.is_alive():raise RuntimeError("Owned API worker did not join")

class OwnedDispatch:
    """Two bounded lanes keep synchronous API work off the HTTP event loop."""
    def __init__(self,asgi,server,expected_sources,guard):
        self.asgi=asgi;self.original=asgi._invoke_handler;self.guard=guard
        for module in (asgi,server):
            expected=expected_sources[module.__name__]
            actual=_pin(module.__file__)
            if actual!=expected:raise ValueError("Official source pin mismatch")
        self.targets={server.get_logs_batch:"read",server.bulk_log:"write"}
        self.slots={}
        self.stats={"entered_read":0,"entered_write":0,"completed":0,"failed":0}
        self.closed=False
        def invoke(fn,request,args=None,kwargs=None):
            lane=self.targets.get(fn)
            if lane is None:
                if inspect.iscoroutinefunction(fn):
                    return self.original(fn,request,args=args,kwargs=kwargs)
                lane="write" if fn.__name__.startswith(("delete","rename","set_","create","update","bulk_","log","clear","upload")) else "read"
            async def awaited():
                self.guard()
                self.stats["entered_"+lane]+=1
                try:
                    result=await self.slots[lane].invoke(
                        lambda:self.original(fn,request,args=args,kwargs=kwargs))
                    self.stats["completed"]+=1
                    return result
                except BaseException:
                    self.stats["failed"]+=1
                    raise
            return awaited()
        self.hook=invoke
        # ALL potentially failing work after first thread start is inside this try.
        try:
            for name in ("read","write"):
                slot=_OwnedSlot("Trackio-owned-"+name)
                self.slots[name]=slot
                slot.start()
            asgi._invoke_handler=invoke
        except BaseException as primary:
            cleanup_errors=[]
            try:
                if asgi._invoke_handler is self.hook:asgi._invoke_handler=self.original
            except BaseException as cleanup:
                cleanup_errors.append(type(cleanup).__name__+": "+str(cleanup)[:500])
            for slot in reversed(list(self.slots.values())):
                try:slot.close(30)
                except BaseException as cleanup:cleanup_errors.append(type(cleanup).__name__+": "+str(cleanup)[:500])
            if cleanup_errors:
                try:primary.add_note("Owned API constructor cleanup: "+repr(cleanup_errors))
                except BaseException:pass
            raise
    def close(self):
        if self.closed:return
        if self.asgi._invoke_handler is not self.hook:
            raise RuntimeError("Owned API hook replaced unexpectedly")
        self.asgi._invoke_handler=self.original
        errors=[]
        for slot in self.slots.values():
            try:slot.close(30)
            except BaseException as error:errors.append(repr(error)[:500])
        self.closed=True
        if errors:raise RuntimeError("; ".join(errors))

def wait_http_ready(base,guard,should_exit,owned_started,total_seconds=60):
    """Existing GET/version only. No synthetic health log."""
    end=time.monotonic()+total_seconds
    while not should_exit():
        guard()
        remaining=end-time.monotonic()
        if remaining<=0:raise TimeoutError("Reader HTTP readiness deadline")
        if not owned_started():
            time.sleep(min(.25,remaining));continue
        try:
            with urllib.request.urlopen(base+"/version",timeout=min(20,remaining)) as response:
                if response.status==200:
                    json.load(response)
                    return
        except (OSError,ValueError):
            if time.monotonic()>=end:raise
        if time.monotonic()>=end:raise TimeoutError("Reader HTTP readiness deadline")
        time.sleep(.25)
    raise RuntimeError("Reader stopped before HTTP readiness")
