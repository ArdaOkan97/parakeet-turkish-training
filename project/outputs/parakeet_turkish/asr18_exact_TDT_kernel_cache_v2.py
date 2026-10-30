"""Inactive exact six-dispatcher E-only cache experiment; no new kernel math."""
from copy import deepcopy
import hashlib
import importlib.metadata
import inspect
import os
from pathlib import Path
import sys
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_production_state_v1 import require,digest

class ExactTDTCache:
    def __init__(self,c,*,mode,torch,coherent,record,guard):
        from numba.cuda.dispatcher import CUDACacheImpl,CUDADispatcher
        from numba.cuda.core.caching import _UserProvidedCacheLocator
        from numba.cuda.core import config
        from nemo.collections.asr.parts.numba.rnnt_loss.utils.cuda_utils import reduce,gpu_rnnt_kernel
        from nemo.collections.asr.parts.numba.rnnt_loss.utils import rnnt_helper
        root=Path(c['cache_directory']).resolve();e=Path(r'E:\gpu_training_scratchpad').resolve()
        require(root.is_relative_to(e) and root.is_dir() and mode in ('seed','hit'),'Reviewed owned cache namespace')
        require(Path(os.environ['NUMBA_CACHE_DIR']).resolve()==root and Path(config.CACHE_DIR).resolve()==root,
                'Numba cache configured on exact E BEFORE imports')
        self.root=root;self.mode=mode;self.record=record;self.guard=guard;self.dispatchers={
            'reduce._reduce_rows':reduce._reduce_rows,'reduce._reduce_minus':reduce._reduce_minus,
            'gpu_rnnt_kernel.compute_tdt_alphas_kernel':gpu_rnnt_kernel.compute_tdt_alphas_kernel,
            'gpu_rnnt_kernel.compute_tdt_betas_kernel':gpu_rnnt_kernel.compute_tdt_betas_kernel,
            'gpu_rnnt_kernel.compute_tdt_grad_kernel':gpu_rnnt_kernel.compute_tdt_grad_kernel,
            'rnnt_helper.compute_costs_data':rnnt_helper.compute_costs_data}
        props=torch.cuda.get_device_properties(0);self.cc=(props.major,props.minor)
        namespace={'kind':'ASR18_exact_six_official_TDT_cache_identity_v1','source_bindings':c['source_bindings'],
            'GPU_python_binary':c['GPU_python_binary'],'torch':torch.__version__,
            'numba':importlib.metadata.version('numba'),'numba_cuda':importlib.metadata.version('numba-cuda'),
            'python':sys.version,'platform':sys.platform,'CC':list(self.cc),
            'physical_CUDA_bytes':props.total_memory,'compiler_DLLs':coherent['before_torch'],
            'actual_linker':coherent['actual_binding_linker'],'compiler_paths':coherent['pathfinder_paths'],
            'shape':c['shape'],'labels':c['labels'],'cache_math_unchanged':True}
        require(namespace['numba_cuda']=='0.30.4','Exact reviewed CUDA dispatcher runtime')
        owner=root.parent/'cache_identity.json'
        if mode=='seed':require(not owner.exists(),'Fresh cache identity');write_json(owner,namespace)
        else:require(read_json(owner)==namespace and digest(read_json(owner))==digest(namespace),'Exact CC/runtime/compiler/source cache identity')
        self.identity=namespace;self.before={};self.paths={}
        # PREVENT all fallback locators, including mkdir attempts in project/user.
        CUDACacheImpl._locator_classes=[_UserProvidedCacheLocator]
        require(CUDACacheImpl._locator_classes==[_UserProvidedCacheLocator],'User-provided-only E cache locator')
        for name,d in self.dispatchers.items():
            guard();require(isinstance(d,CUDADispatcher) and not d.overloads and not d.targetoptions.get('device',False),
                            'Uncompiled EXISTING official launch dispatcher, no redecorating')
            before=(d.py_func,d.py_func.__code__,deepcopy(d.targetoptions))
            self.before[name]=before;d.enable_caching()
            path=Path(d._cache.cache_path).resolve();require(path.is_relative_to(root),'No NumPy/Numba cache fallback outside exact E')
            self.paths[name]=str(path)
            require((d.py_func,d.py_func.__code__,d.targetoptions)==before,'No function/code/options change')
            require(record(inspect.getsourcefile(d.py_func)) in c['source_bindings'].values(),'Bound official kernel source')
            if mode=='hit':
                # Cache lookup still precedes this flag in explicit compile().
                # It also blocks post-load LCS invalidation from compiling anew.
                d._can_compile=False
                original_load=d._cache.load_overload
                def hit_only(sig,target_context,original_load=original_load,name=name):
                    kernel=original_load(sig,target_context)
                    require(kernel is not None,'Cache miss rejected BEFORE _Kernel/compile_cuda: '+name)
                    require(kernel._codelibrary._cubin_cache.get(self.cc),'Exact current-CC CUBIN required BEFORE bind/link')
                    return kernel
                d._cache.load_overload=hit_only
                require(not d._cache.is_launch_config_sensitive(),'This six-kernel experiment rejects LCS markers')
                index=d._cache._cache_file._load_index()
                require(index and all(type(key) is tuple and len(key)==4 and type(key[0]) is tuple for key in index),
                        'Exact immutable current-version seed signature index; no eval/string reconstruction or LCS keys')
                for key in index:
                    guard();d.compile(key[0])
                require(d.overloads and d._can_compile is False,'Actual cache-only signatures loaded before ordinary calls')
    def snapshot(self):
        result={}
        for name,d in self.dispatchers.items():
            self.guard();before=self.before[name]
            require((d.py_func,d.py_func.__code__,d.targetoptions)==before,'Existing function/code/options unchanged')
            result[name]={'hits':sum(d._cache_hits.values()),'misses':sum(d._cache_misses.values()),
                'signatures':[str(x) for x in d.overloads], 'cache_path':self.paths[name],
                'source':self.record(inspect.getsourcefile(d.py_func)),
                'bytecode_sha256':hashlib.sha256(d.py_func.__code__.co_code).hexdigest(),
                'compile_fallback_allowed':d._can_compile,
                'options':d.targetoptions,'every_loaded_kernel_has_current_CC_cubin':all(bool(k._codelibrary._cubin_cache.get(self.cc)) for k in d.overloads.values())}
        return result
    def complete(self):
        result=self.snapshot()
        if self.mode=='seed':require(all(x['misses']>0 and x['signatures'] for x in result.values()),'All six actual kernel signatures seeded')
        else:require(all(x['hits']>0 and x['misses']==0 and x['signatures'] and x['every_loaded_kernel_has_current_CC_cubin'] for x in result.values()),
                     'All SIX actual cache hits, zero compile misses/fallback, exact CUBIN')
        return result

def inventory(root,*,record,guard):
    root=Path(root).resolve();rows=[];total=0
    for path in sorted(root.rglob('*')):
        guard();require(path.resolve().is_relative_to(root) and not path.is_symlink(),'Exact owned E cache entries only')
        if not path.is_file():continue
        total+=path.stat().st_size;require(total<=64*1024**2 and len(rows)<128,'Bound64MiB/128cachefiles')
        rows.append(record(path))
    return {'files':rows,'bytes':total}
