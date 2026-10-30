"""Coarse observation only: unchanged verified grouped2/loss7 operation.

No tensor retention, numerical work, gradient replacement or added CUDA sync.
Node pre/post hooks return None. All event drains follow original native syncs.
"""
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
import copy,gc,threading,time
from atomic_io_retry_v1 import write_json
from bpe_broad_audio_v1 import record,require


class CoarseProfiler:
    MAX_EVENT_PAIRS=256
    def __init__(self,model,optimizer,*,torch,folder,guard):
        self.model=model;self.optimizer=optimizer;self.torch=torch;self.guard=guard;self.folder=Path(folder).resolve()
        require(self.folder.is_dir() and self.folder.is_relative_to(Path(r'E:\gpu_training_scratchpad')),'Owned E profiler output')
        self.active=False;self.closed=False;self.failed=False;self.events=[];self.handles=[];self.pending=set();self.undo=[]
        self.phases={};self.updates=[];self.serial=0;self.threads={};self.guard_calls=0;self.guard_wall=0.
        self.stream=None;self.lock=threading.Lock()

    def start(self,label,cuda=True):
        require(self.active,'Only active measured update')
        first=last=None
        if cuda:
            require(len(self.events)+len(self.pending)<self.MAX_EVENT_PAIRS,'Bounded coarse events, never per matrix/tile')
            current=int(self.torch.cuda.current_stream().cuda_stream)
            require(current==self.stream,'Coarse callbacks must use the same actual owned CUDA stream')
            first=self.torch.cuda.Event(enable_timing=True);last=self.torch.cuda.Event(enable_timing=True);first.record()
        return (label,time.perf_counter(),first,last,threading.get_ident())

    def finish(self,token):
        label,began,first,last,ident=token
        require(ident==threading.get_ident(),'A coarse phase ends on its actual starting callback thread')
        if last is not None:last.record()
        with self.lock:
            entry=self.phases.setdefault(label,{'calls':0,'host_wall_seconds':0.,'CUDA_elapsed_ms':0.})
            entry['calls']+=1;entry['host_wall_seconds']+=time.perf_counter()-began
            if first is not None:self.events.append((label,first,last))

    @contextmanager
    def phase(self,label,cuda=True):
        if not self.active:yield;return
        token=self.start(label,cuda)
        try:yield
        finally:self.finish(token)

    def patch(self,obj,key,label,*,cuda=True,after=None):
        original=getattr(obj,key)
        @wraps(original)
        def observed(*a,**kw):
            with self.phase(label,cuda):value=original(*a,**kw)
            if self.active and after is not None:after(value)
            return value
        setattr(obj,key,observed)
        def restore():
            require(getattr(obj,key) is observed,'Exactly owned observation wrapper remains installed')
            setattr(obj,key,original)
        self.undo.append(restore)

    def loss_node(self,value):
        require(value.requires_grad and value.grad_fn is not None and type(value.grad_fn).__name__=='FullNativeTiledBackward',
            'Exact reduction-none frozen loss7 custom Node, not a guessed enclosing graph')
        self.serial+=1;identity=self.serial;state={'token':None,'done':False}
        def before(grad_outputs):
            require(self.active and state['token'] is None and not state['done'],'One actual loss Node backward')
            state['token']=self.start('full_TDT_custom_backward');self.pending.add(identity)
            thread=threading.current_thread();self.threads[str(thread.ident)]={'ident':thread.ident,'name':thread.name}
            return None
        def after(grad_inputs,grad_outputs):
            require(state['token'] is not None and not state['done'] and identity in self.pending,'Matching actual loss Node posthook')
            self.finish(state['token']);self.pending.remove(identity);state['done']=True;state['token']=None
            return None
        require(len(self.handles)<32,'At most16 source losses per update; no graph history')
        self.handles.extend((value.grad_fn.register_prehook(before),value.grad_fn.register_hook(after)))

    def install(self):
        require(not self.undo and not self.closed,'One exact profiler installation after cache/GC setup')
        import asr18_grouped_precomputed_native_v2 as native
        self.patch(self.model.encoder,'forward','encoder_total_forward')
        # exact_serial_stem captures this observed original and invokes it once
        # per original B1 row; no change to the stem wrapper or physical length.
        self.patch(self.model.encoder.subsampling,'forward','serial_B1_CNN_stem')
        self.patch(self.model.encoder_projector,'forward','student_B1_encoder_projector')
        self.patch(self.model.decoder,'forward','student_B1_decoder')
        self.patch(native,'sampled_preservation','B1_replay_teacher_and_student_preservation')
        self.patch(native,'bounded_tdt_loss','full_TDT_forward_internal_DP',after=self.loss_node)
        self.patch(self.torch.Tensor,'backward','total_group_backward')
        self.patch(self.optimizer,'step','full_original_clip_and_threeAdam_step')
        for i,optimizer in enumerate(self.optimizer.optimizers):self.patch(optimizer,'step','Adam_partition_'+str(i))
        self.patch(gc,'collect','original_GC_collect',cuda=False)
        self.patch(self.torch.cuda,'synchronize','original_explicit_CUDA_sync',cuda=False)
        self.patch(self.torch.cuda,'empty_cache','original_CUDA_empty_cache',cuda=False)
        return {'source':record(__file__),'coarse_event_pair_limit_per_update':self.MAX_EVENT_PAIRS,
            'loss_Node_hooks_return_None_without_gradient_or_tensor_access':True,'extra_synchronizations':0,
            'frozen_matrix_hook_observers':0,'retained_output_tensor_bytes':0,'timings_nested_not_additive':True}

    def remove_handles(self):
        for handle in self.handles:handle.remove()
        self.handles.clear()

    def drain(self):
        require(not self.pending,'Original synchronized boundary must finish every current TDT backward')
        for label,first,last in self.events:
            require(last.query(),'Only drain after the unchanged original native synchronization')
            self.phases[label]['CUDA_elapsed_ms']+=first.elapsed_time(last)
        self.events.clear();self.remove_handles()

    def feature(self,function,index):
        with self.phase('original_feature_reader_hash_load',cuda=False):return function(index)

    def checked_guard(self):
        began=time.perf_counter()
        try:return self.guard()
        finally:
            if self.active:self.guard_calls+=1;self.guard_wall+=time.perf_counter()-began

    def observe_row(self,*args):return None

    def stage(self,original,name,**extra):
        if self.active and name=='grouped_native_graph_boundary':self.drain()
        with self.phase('original_stage_telemetry_and_write',cuda=False):return original(name,**extra)

    def run_update(self,operation,*,update):
        require(not self.active and not self.events and not self.pending and not self.handles and 1<=update<=4 and len(self.updates)<4,
            'Exact original first4 only; no retained graph/event history')
        self.phases={};self.threads={};before=(self.guard_calls,self.guard_wall);self.stream=int(self.torch.cuda.current_stream().cuda_stream)
        began=time.perf_counter();self.active=True
        try:
            value=operation();elapsed=time.perf_counter()-began;self.drain()
            record_value={'update':update,'instrumented_native_wall_seconds':elapsed,'coarse_phases':copy.deepcopy(self.phases),
                'guard_calls':self.guard_calls-before[0],'guard_host_wall_seconds':self.guard_wall-before[1],
                'loss_backward_callback_threads':copy.deepcopy(self.threads),'same_CUDA_stream':self.stream,
                'timings_nested_not_additive':True,'no_uninstrumented_speed_claim':True}
            require(self.phases['serial_B1_CNN_stem']['calls']==16 and self.phases['encoder_total_forward']['calls']==2
                and self.phases['full_TDT_custom_backward']['calls']==16,'All original16 rows/two encoder groups/loss backwards observed')
            self.updates.append(record_value);write_json(self.folder/('coarse_profile_update_'+str(update)+'.json'),record_value)
            return value
        except BaseException:
            self.failed=True;self.events.clear();self.pending.clear();self.remove_handles();raise
        finally:self.active=False

    def close(self):
        require(not self.active,'No active profiler during teardown')
        self.remove_handles()
        for restore in reversed(self.undo):restore()
        self.undo.clear();self.closed=True
        output=self.folder/'coarse_profile_summary.json'
        write_json(output,{'kind':'ASR18_verified_grouped_coarse_phase_observation_v1','source':record(__file__),'profile_enabled':True,
            'updates':self.updates,'operation_failed':self.failed,'production_enabled':False,
            'Transformer_forward_remainder':'encoder total minus nested B1 stem; includes encoder non-stem bookkeeping',
            'non_TDT_backward_remainder':'total group backward minus TDT custom backwards; includes Transformer and B1 projector/decoder gradients',
            'extra_CUDA_synchronizations':0,'retained_output_tensor_bytes':0})
        return record(output)
