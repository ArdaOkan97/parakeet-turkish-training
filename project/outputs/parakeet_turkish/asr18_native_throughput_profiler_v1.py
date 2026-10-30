"""Bounded symmetric observation of the unchanged serial207 native operation.

No numerical import, process launch, source rewrite or guard cadence change.
CUDA events are per coarse phase, never per frozen-matrix hook. Existing row
synchronization is the event drain; verification work is reported separately.
"""
from contextlib import contextmanager
from functools import wraps
import copy,inspect,sys,threading,time
from pathlib import Path
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import digest
from atomic_io_retry_v1 import write_json


class NativeProfiler:
    MAX_EVENTS=256
    MAX_OUTPUT_BYTES=16*1024**2

    def __init__(self,model,optimizer,*,torch,folder,guard):
        self.model=model;self.optimizer=optimizer;self.torch=torch
        self.folder=Path(folder);self.guard=guard;self.active=False;self.closed=False
        self.events=[];self.outputs=[];self.output_bytes=0;self.undo=[];self.phases={};self.hooks={}
        self.update=None;self.microbatch=0;self.row_records=[];self.update_records=[];self.gradient_records=[]
        self.stack=[];self.bookkeeping_seconds=0.;self.verification_seconds=0.;self.stage_seconds=0.
        self.boundary_seconds=0.;self.guard_seconds=0.;self.guard_calls=0;self.threads={}
        self.original_source=record(__file__)

    def _add(self,label,wall,*,cuda=None):
        v=self.phases.setdefault(label,{'calls':0,'host_wall_seconds':0.,'CUDA_elapsed_ms':0.})
        v['calls']+=1;v['host_wall_seconds']+=wall
        if cuda is not None:v['CUDA_elapsed_ms']+=cuda

    @contextmanager
    def phase(self,label,*,cuda=True):
        if not self.active:
            yield;return
        t=time.perf_counter();start=end=None
        if cuda:
            require(len(self.events)<self.MAX_EVENTS,'Bounded per-row coarse CUDA event pairs')
            start=self.torch.cuda.Event(enable_timing=True);end=self.torch.cuda.Event(enable_timing=True);start.record()
        self.stack.append(label);self.bookkeeping_seconds+=time.perf_counter()-t
        began=time.perf_counter()
        try:yield
        finally:
            wall=time.perf_counter()-began;t=time.perf_counter()
            if end is not None:end.record();self.events.append((label,start,end))
            require(self.stack.pop()==label,'Exact nested profiler phase stack')
            self._add(label,wall);self.bookkeeping_seconds+=time.perf_counter()-t

    def _patch(self,obj,key,function):
        original=getattr(obj,key);setattr(obj,key,function)
        self.undo.append(lambda obj=obj,key=key,original=original,function=function:
            self._restore(obj,key,original,function))

    @staticmethod
    def _restore(obj,key,original,function):
        require(getattr(obj,key) is function,'Profiled callable changed unexpectedly')
        setattr(obj,key,original)

    def _wrap(self,function,label,*,cuda=True,observe=None):
        @wraps(function)
        def wrapped(*a,**kw):
            with self.phase(label,cuda=cuda):result=function(*a,**kw)
            if self.active and observe is not None:observe(result)
            return result
        return wrapped

    def _retain(self,label,value):
        if not self.torch.is_tensor(value):return
        value=value.detach();n=value.numel()*value.element_size()
        require(value.is_contiguous() and self.output_bytes+n<=self.MAX_OUTPUT_BYTES,
            'Only bounded complete contiguous per-row observed outputs, no graph retention')
        self.outputs.append((label,value));self.output_bytes+=n

    def _encoded(self,value):
        self._retain('encoder_full_last_hidden_state',value.last_hidden_state)
        if value.attention_mask is not None:self._retain('encoder_full_attention_mask',value.attention_mask)

    def _objective(self,value):
        for name in ('total','tdt','token_kl','duration_kl','feature_distillation'):
            self._retain('objective_'+name,getattr(value,name))
        for name,value in value.per_clip.items():self._retain('per_clip_'+name,value)

    def _hook_wrap(self,function,name,kind):
        @wraps(function)
        def wrapped(*a,**kw):
            if not self.active:return function(*a,**kw)
            t=time.perf_counter();thread=threading.current_thread()
            thread_key=str(thread.ident)+':'+thread.name
            self.threads.setdefault(thread_key,{'ident':thread.ident,'name':thread.name,
                'is_main_thread':thread is threading.main_thread(),'calls':0})['calls']+=1
            phase=self.stack[-1] if self.stack else 'outside_coarse_phase'
            key=(name,kind,phase);before=time.perf_counter();self.bookkeeping_seconds+=before-t
            try:return function(*a,**kw)
            finally:
                elapsed=time.perf_counter()-before;t=time.perf_counter();e=sys.exception()
                item=self.hooks.setdefault(key,{'calls':0,'host_wall_seconds':0.,'None_output_calls':0,'unwind_exception_types':{}})
                item['calls']+=1;item['host_wall_seconds']+=elapsed
                if kind=='after' and len(a)>=3 and a[2] is None:
                    item['None_output_calls']+=1;typ=None if e is None else type(e).__module__+'.'+type(e).__qualname__
                    item['unwind_exception_types'][str(typ)]=item['unwind_exception_types'].get(str(typ),0)+1
                self.bookkeeping_seconds+=time.perf_counter()-t
        return wrapped

    def install(self):
        require(not self.active and not self.undo and not self.closed,'One symmetric profiler installation')
        import asr18_precomputed_native_v2 as native
        import asr18_narrow_objective_v1 as objective
        import gc
        pairs=[(native,'collate_precomputed_row','collate_to_GPU',None),
            (native,'pt_focused_batch_objective','full_objective',self._objective),
            (native,'bounded_tdt_loss','canonical_full_TDT_forward_internal_gradient',None),
            (objective,'sampled_preservation','replay_teacher_preservation',None),
            (self.model.encoder,'forward','encoder_forward',self._encoded),
            (self.torch.Tensor,'backward','PyTorch_backward_recompute',None),
            (self.optimizer,'step','global_clip_threeAdam_full_step',None)]
        for obj,key,label,observe in pairs:self._patch(obj,key,self._wrap(getattr(obj,key),label,observe=observe))
        for index,optimizer in enumerate(self.optimizer.optimizers):
            self._patch(optimizer,'step',self._wrap(optimizer.step,'Adam_partition_'+str(index)))
        for obj,key,label in ((gc,'collect','gc_collect'),(self.torch.cuda,'synchronize','explicit_CUDA_synchronize'),
                (self.torch.cuda,'empty_cache','CUDA_empty_cache')):
            self._patch(obj,key,self._wrap(getattr(obj,key),label,cuda=False))
        original=self.torch.nn.utils.clip_grad_norm_
        @wraps(original)
        def clipping(*a,**kw):
            if not self.active:return original(*a,**kw)
            from asr18_live_state_fingerprint_v1 import tensor_digest
            began=time.perf_counter()
            before={name:tensor_digest(p.grad,torch=self.torch,guard=self.guard)
                for name,p in sorted(self.optimizer.parameters.items())}
            self.verification_seconds+=time.perf_counter()-began
            with self.phase('original_global_clip'):result=original(*a,**kw)
            began=time.perf_counter()
            after={name:tensor_digest(p.grad,torch=self.torch,guard=self.guard)
                for name,p in sorted(self.optimizer.parameters.items())}
            require(len(before)==len(after)==207,'Exact all207 accumulated and clipped gradient fingerprints')
            self.gradient_records.append({'update':self.update,'all207_before_original_clip':before,
                'all207_after_original_clip':after,'norm':tensor_digest(result,torch=self.torch,guard=self.guard)})
            self.verification_seconds+=time.perf_counter()-began
            return result
        self._patch(self.torch.nn.utils,'clip_grad_norm_',clipping)
        ids={handle.id for handle in self.model._diagnostic_frozen_handles};seen=set()
        require(len(ids)==588,'Exact frozen294 hook pair inventory')
        for module_name,module in self.model.named_modules():
            for kind,table in (('before',module._forward_pre_hooks),('after',module._forward_hooks)):
                for key,function in list(table.items()):
                    if key not in ids:continue
                    require(key not in seen,'One frozen hook observation per exact original handle')
                    seen.add(key);wrapped=self._hook_wrap(function,module_name,kind);table[key]=wrapped
                    def restore(table=table,key=key,function=function,wrapped=wrapped):
                        require(table.get(key) is wrapped,'Profiled frozen hook changed');table[key]=function
                    self.undo.append(restore)
        require(seen==ids,'All588 hooks observed without adding/removing a hook or per-hook CUDA event/sync')
        return {'source':self.original_source,'frozen_hooks_observed':len(seen),'coarse_CUDA_event_pair_limit':self.MAX_EVENTS,
            'maximum_detached_observed_output_bytes_per_row':self.MAX_OUTPUT_BYTES,'per_hook_extra_synchronization':False}

    def checked_guard(self):
        began=time.perf_counter()
        try:return self.guard()
        finally:
            if self.active:self.guard_calls+=1;self.guard_seconds+=time.perf_counter()-began

    def feature(self,function,index):
        with self.phase('feature_reader_hash_load',cuda=False):return function(index)

    def stage(self,original,name,**extra):
        if self.active and name=='precomputed_native_microbatch':
            self.microbatch=extra['microbatch'];self.drain(label='row_'+str(self.microbatch))
            self.boundary_seconds+=extra['memory_boundary']['seconds']
        began=time.perf_counter()
        try:return original(name,**extra)
        finally:
            if self.active:self.stage_seconds+=time.perf_counter()-began

    def drain(self,*,label):
        # Caller invokes only AFTER the original synchronize. No extra CUDA
        # synchronization is introduced by a module hook or event drain.
        began=time.perf_counter()
        for name,start,end in self.events:
            require(end.query(),'Original native row drain must complete all coarse events')
            self.phases[name]['CUDA_elapsed_ms']+=start.elapsed_time(end)
        self.events.clear()
        from asr18_live_state_fingerprint_v1 import tensor_digest
        hashes={name:tensor_digest(value,torch=self.torch,guard=self.guard) for name,value in self.outputs}
        self.outputs.clear();self.output_bytes=0
        self.verification_seconds+=time.perf_counter()-began
        self.row_records.append({'update':self.update,'label':label,'outputs':hashes})

    def run_update(self,operation,*,update):
        require(not self.active and not self.events and not self.outputs and not self.stack,'Serial profiler update boundary')
        self.update=update;self.microbatch=0;self.phases={};self.hooks={};self.row_records=[]
        before={'verification':self.verification_seconds,'bookkeeping':self.bookkeeping_seconds,
            'stage':self.stage_seconds,'boundary':self.boundary_seconds,'guard':self.guard_seconds,'guard_calls':self.guard_calls,
            'loader':copy.deepcopy(self.model._diagnostic_frozen_metrics)}
        began=time.perf_counter();self.active=True
        try:result=operation()
        finally:
            elapsed=time.perf_counter()-began
            if self.events or self.outputs:self.drain(label='update_final')
            self.active=False
        record_value={'update':update,'instrumented_native_wall_seconds':elapsed,'coarse_phases':copy.deepcopy(self.phases),
            'frozen_hook_host_observations':[{'module':key[0],'kind':key[1],'phase':key[2],**v} for key,v in self.hooks.items()],
            'full_row_output_fingerprints':copy.deepcopy(self.row_records),
            'measured_verification_host_seconds':self.verification_seconds-before['verification'],
            'measured_wrapper_bookkeeping_host_seconds':self.bookkeeping_seconds-before['bookkeeping'],
            'original_stage_callback_host_seconds':self.stage_seconds-before['stage'],
            'original_graph_boundary_host_seconds':self.boundary_seconds-before['boundary'],
            'unchanged_guard_calls':self.guard_calls-before['guard_calls'],'unchanged_guard_host_seconds':self.guard_seconds-before['guard'],
            'frozen_metrics_before':before['loader'],'frozen_metrics_after':copy.deepcopy(self.model._diagnostic_frozen_metrics),
            'hook_callback_threads':copy.deepcopy(self.threads),'timings_are_nested_not_additive':True,
            'verification_and_wrapper_observations_not_an_uninstrumented_counterfactual':True}
        self.update_records.append(record_value)
        write_json(self.folder/('profile_update_'+str(update)+'.json'),record_value)
        return result

    def close(self):
        require(not self.active and not self.events and not self.outputs,'No pending profile graphs/events')
        for restore in reversed(self.undo):restore()
        self.undo.clear();self.closed=True
        out={'kind':'ASR18_actual_native_throughput_phase_profile_v1','source':self.original_source,
            'all207_gradient_fingerprints':self.gradient_records,'hook_callback_threads':self.threads,
            'measured_verification_seconds':self.verification_seconds,'measured_wrapper_bookkeeping_seconds':self.bookkeeping_seconds,
            'per_hook_CUDA_events_or_extra_synchronization':False,'numerical_or_guard_policy_changed':False,
            'source_callable_and_hook_objects_restored':True,'production_enabled':False}
        write_json(self.folder/'profile_summary.json',out);return record(self.folder/'profile_summary.json')
