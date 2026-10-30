"""Inactive scalar observation, not a GC policy: exact original call passthrough.

Install only AFTER the original once-only startup freeze; remove BEFORE its
original teardown unfreeze/fullcollect. No retained frames, exception objects,
per-call records or payloads. Timer failures cannot replace the original result
or exception. No extra collect/freeze/sync/threshold/frequency operation.
"""
import gc,threading,time

ORIGINAL_COLLECT=gc.collect
ORIGINAL_CLOCK=time.perf_counter_ns

class CollectScalarObservation:
    def __init__(self,original,*,clock=ORIGINAL_CLOCK):
        self.original=original;self.clock=clock;self.lock=threading.RLock()
        self.calls=0;self.raised=0;self.elapsed_ns=0;self.max_elapsed_ns=0;self.clock_errors=0

    def _start(self):
        try:return self.clock()
        except BaseException:return None

    def _record(self,start,raised):
        # Observation cannot replace an original return/exception. Keep only
        # bounded scalar counters; never save sys.exc_info or an exception.
        try:
            end=self.clock() if start is not None else None
            valid=type(start)is int and type(end)is int and end>=start
            elapsed=end-start if valid else 0
            with self.lock:
                self.calls+=1;self.raised+=int(raised);self.elapsed_ns+=elapsed
                self.max_elapsed_ns=max(self.max_elapsed_ns,elapsed);self.clock_errors+=int(not valid)
        except BaseException:
            # Even a timing/counter failure leaves the original operation's
            # exact outcome authoritative. No new exception normalization.
            try:self.clock_errors+=1
            except BaseException:pass

    def __call__(self,*args,**kwargs):
        start=self._start()
        try:value=self.original(*args,**kwargs)
        except BaseException:
            self._record(start,True)
            raise
        self._record(start,False)
        return value

    def take_window(self):
        with self.lock:
            result={'calls':self.calls,'raised_calls':self.raised,'elapsed_ns':self.elapsed_ns,
                'max_elapsed_ns':self.max_elapsed_ns,'clock_errors':self.clock_errors,
                'call_wall_durations_may_overlap_for_reentrant_or_concurrent_calls':True,
                'scope':'All original explicit gc.collect calls in this window; sampled timing only, no GC policy change'}
            self.calls=0;self.raised=0;self.elapsed_ns=0;self.max_elapsed_ns=0;self.clock_errors=0
        return result

class ObserveOriginalCollectScope:
    def __init__(self):
        if threading.current_thread()is not threading.main_thread():raise ValueError('Original main-thread observation authority')
        if gc.collect is not ORIGINAL_COLLECT:raise ValueError('Exact unchanged original builtin before observation')
        self.observer=CollectScalarObservation(ORIGINAL_COLLECT);self.active=False;self.restored=False

    def __enter__(self):
        if self.active or self.restored:raise ValueError('Original collect observation scope is single-use')
        if gc.collect is not ORIGINAL_COLLECT:raise ValueError('Original builtin before installation')
        gc.collect=self.observer;self.active=True
        return self.observer

    def __exit__(self,exc_type,exc_value,traceback):
        if gc.collect is self.observer:
            gc.collect=ORIGINAL_COLLECT;self.active=False;self.restored=True
        elif exc_type is None:raise ValueError('Unexpected original collect alias replacement')
        return False
