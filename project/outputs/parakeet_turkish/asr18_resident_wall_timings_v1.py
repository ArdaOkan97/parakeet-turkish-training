"""Inactive bounded monotonic wall counters; never add a CUDA barrier or affect work.

Timers include host work and any synchronization already present in the wrapped
operation. They are not CUDA-event/kernel timings. Diagnostic errors are recorded
without replacing the original operation's result or exception.
"""
from contextlib import contextmanager
import math,time

NAMES=(
 'packet_wait','packet_metadata_and_transaction_validation','native_update_total',
 'after_native_augmentation_validation','transaction_commit_total','after_commit_sampler_validation',
 'transaction_validation','transaction_state_and_native_receipt','transaction_journal_append_total',
 'native_ingress_routes_and_initial_state','native_group_guard_stage','native_collation_transfer',
 'native_encoder_and_stem','native_encoder_mask_validation','native_row_objectives_and_scalar_reads',
 'native_loss_stack','native_backward_enqueue','native_group_existing_sync_and_before_snapshot',
 'native_group_RNG_versions_capture','native_group_gc','native_group_existing_gc_sync',
 'native_group_empty_cache','native_group_after_snapshot','native_group_invariants',
 'native_group_receipt_stage','native_payload_validation','native_optimizer_step',
 'native_optimizer_zero_grad','native_final_gc','native_final_existing_sync',
 'native_final_empty_cache','native_final_invariants','native_return_receipt')

class WallTimings:
    def __init__(self,clock=time.monotonic):
        self.clock=clock;self.seconds={name:0. for name in NAMES}
        self.counts={name:0 for name in NAMES};self.maximum={name:0. for name in NAMES}
        self.errors=0;self.first_error=None
    def _error(self,error):
        self.errors+=1
        if self.first_error is None:self.first_error=(type(error).__name__+': '+str(error))[:256]
    @contextmanager
    def measure(self,name):
        started=None
        try:
            if name not in self.seconds:raise ValueError('Unknown fixed diagnostic channel')
            started=self.clock()
        except Exception as error:self._error(error)
        try:yield
        finally:
            if started is not None:
                try:
                    elapsed=self.clock()-started
                    if not math.isfinite(elapsed) or elapsed<0:raise ValueError('Invalid monotonic interval')
                    self.seconds[name]+=elapsed;self.counts[name]+=1;self.maximum[name]=max(self.maximum[name],elapsed)
                except Exception as error:self._error(error)
    def snapshot(self):
        return {'kind':'ASR18_bounded_existing_operation_wall_timings_v1',
            'seconds':dict(self.seconds),'counts':dict(self.counts),'maximum_seconds':dict(self.maximum),
            'diagnostic_errors':self.errors,'first_diagnostic_error':self.first_error,
            'GPU_barriers_added':0,'CUDA_kernel_or_GPU_only_timing_claimed':False,
            'nested_timers_overlap_do_not_sum_them':True}

@contextmanager
def wall_time(timings,name):
    if timings is None:yield
    else:
        with timings.measure(name):yield

def wall_call(timings,name,function,*args,**kwargs):
    # Python evaluates the original arguments before entering this helper.
    with wall_time(timings,name):return function(*args,**kwargs)