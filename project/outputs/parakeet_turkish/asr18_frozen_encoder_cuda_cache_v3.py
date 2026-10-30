"""Inactive optional instance cache for the exact readonly294 BF16 matrices.

Original module operations and registered CPU Parameter objects are preserved.
One CUDA Parameter per frozen matrix is borrowed only while that module runs.
No model/optimizer/teacher construction, blanket conversion or runtime import.
Persistent CUDA bytes are additional live state until actual worker exit; no
host/private saving, speedup or numerical-equivalence claim is made here.
"""
from contextlib import contextmanager
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import threading

ROOT = Path(r'E:\gpu_training_scratchpad')
PROJECT = Path(__file__).resolve().parents[2]
LOADER = Path(__file__).with_name('stream_frozen_encoder_matrices_v1.py')
LOADER_SHA = '0dc64cbf128543d626402881ccb3a6f15cb26709a08f90d5f839ea99b4f7770c'
LORA = Path(__file__).with_name('encoder_lora.py')
LORA_SHA = 'a133c26df16cbf4d473dc5bcf7eab6d627869caf418234f16712bdec51213455'
BANK = {'path': str(ROOT/'analysis/ASR18_memory_diagnosis_v1/frozen_weight_bank_v1/report.json'),
        'bytes': 153716, 'sha256': '65a5cc493e114fca573eef911f5d0402da560199f171a0d9abaf8bf586221447'}
TUPLE_SHA = 'a3c843cdb4ec5d953d5ce67c14dc846e3f792a5f9ebbec710ea1fc2abb904a89'
COUNT, HOOKS, CACHE_BYTES, LARGEST_COPY, READBACK = 294, 588, 1217066496, 8388608, 4*1024**2


def require(value, message):
    if not value: raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode('utf8')).hexdigest()


def record(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT.resolve()) or path.is_relative_to(PROJECT), 'E/new artifacts or existing project sources only')
    before = path.stat(); hashed = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(65536): hashed.update(block)
    require((before.st_size, before.st_mtime_ns) == (path.stat().st_size, path.stat().st_mtime_ns), 'Source/bank changed during hash')
    return {'path': str(path), 'bytes': before.st_size, 'sha256': hashed.hexdigest()}


def _serial(torch):
    require(threading.current_thread() is threading.main_thread()
            and not torch.distributed.is_initialized(), 'Single-owner serial uncompiled use only')


def _active(model):
    result = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            require(parameter.device.type == 'cuda' and str(parameter.dtype) == 'torch.float32'
                    and parameter.grad is None, 'Install only after bootstrap/before first native graph, exact207 FP32 CUDA/no grads')
            result.append((name, id(parameter), parameter.data_ptr(), parameter._version, tuple(parameter.shape)))
    require(len(result) == 207, 'Exact original active207 required')
    return result


def _owner_digest(function):
    value = function()
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None,
            'Owner must hash all207/threeAdam/all4RNG and unchanged teacher state, not return a bool')
    return value


def _callback_source(function):
    return record(inspect.getsourcefile(inspect.unwrap(function)))


def _entries(model, bank, torch):
    from encoder_lora import LoRALinear
    backing = model._diagnostic_frozen_backing
    handles = model._diagnostic_frozen_handles
    require(type(backing) is dict and len(backing) == COUNT and type(handles) is list and len(handles) == HOOKS
            and len({handle.id for handle in handles}) == HOOKS, 'Exact294 mapped matrices and588 original hook handles')
    ids = {handle.id for handle in handles}; seen = set(); entries = []
    for name, cpu in backing.items():
        description = bank['matrices'][name]; module_path, parameter_name = name.rsplit('.', 1)
        module = model.get_submodule(module_path)
        if isinstance(module, LoRALinear): module = module.base_layer
        require(parameter_name == 'weight' and isinstance(module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)),
                'Only original hooked Linear/Conv matrices; LoRA wrapper keeps the same base object')
        parameter = module._parameters['weight']; size = cpu.numel()*cpu.element_size()
        require(cpu.device.type == parameter.device.type == 'cpu' and cpu.dtype == parameter.dtype == torch.bfloat16
                and not cpu.requires_grad and not parameter.requires_grad and cpu.is_contiguous()
                and parameter.data_ptr() == cpu.data_ptr() and tuple(parameter.shape) == tuple(cpu.shape) == tuple(description['shape'])
                and size == description['tensor_bytes'] and description['BF16_roundtrip_byte_exact'] is True,
                'Exact original mapped BF16 geometry/CPU Parameter/backing identity')
        require(len(module._forward_pre_hooks) == len(module._forward_hooks) == 1,
                'Install before instrumentation adds hooks; do not remove unknown hooks')
        pre_id, pre = next(iter(module._forward_pre_hooks.items()))
        post_id, post = next(iter(module._forward_hooks.items()))
        require(pre_id in ids and post_id in ids and pre_id != post_id
                and Path(pre.__code__.co_filename).resolve() == LOADER.resolve()
                and Path(post.__code__.co_filename).resolve() == LOADER.resolve()
                and pre.__name__ == 'before' and post.__name__ == 'after'
                and len(pre.__defaults__) == 3 and pre.__defaults__[0] is cpu
                and type(pre.__defaults__[1]) is int and pre.__defaults__[1] == size and pre.__defaults__[2] is parameter
                and len(post.__defaults__) == 2 and post.__defaults__[0] is parameter
                and type(post.__defaults__[1]) is int and post.__defaults__[1] == size
                and inspect.getclosurevars(pre).nonlocals['metrics'] is model._diagnostic_frozen_metrics
                and inspect.getclosurevars(post).nonlocals['metrics'] is model._diagnostic_frozen_metrics
                and pre_id not in module._forward_pre_hooks_with_kwargs
                and post_id not in module._forward_hooks_with_kwargs
                and post_id in module._forward_hooks_always_called,
                'Only exact source/defaults/metrics/always_call original loader hooks may be replaced')
        require(id(module) not in seen, 'One source matrix per original module, no shared-module ambiguity')
        seen.add(id(module))
        entries.append({'name': name, 'module': module, 'cpu': cpu, 'cpu_parameter': parameter,
            'size': size, 'cpu_version': cpu._version, 'parameter_version': parameter._version,
            'pre': pre, 'post': post, 'pre_id': pre_id, 'post_id': post_id, 'gpu_parameter': None, 'busy': False})
    require({i for entry in entries for i in (entry['pre_id'], entry['post_id'])} == ids
            and len(entries) == COUNT and sum(entry['size'] for entry in entries) == CACHE_BYTES,
            'Complete exact original294/588 coverage')
    return entries


class _InstanceCache:
    def __init__(self, model, entries):
        self.pid = os.getpid(); self.model = model; self.entries = entries; self.handles = []
        self.poisoned = False
        self.metrics = {'persistent_cuda_cache_bytes': CACHE_BYTES, 'cache_initial_upload_bytes': 0,
            'cache_verification_readback_bytes': 0, 'cache_hit_reuses': 0, 'inflight_matrices': 0}
        self.policy = None
        self.backward = None
        self.backward_sequence = 0

    def _hook_owner(self, torch):
        require(os.getpid() == self.pid and not torch.distributed.is_initialized(),
                'Same-process non-distributed cache owner required')
        graph = torch._C._current_graph_task_id()
        require(type(graph) is int, 'Exact available current GraphTask identity required')
        if self.backward is None:
            _serial(torch)
            require(graph == -1, 'Autograd cache callback outside explicit owned backward scope')
            return ('forward', threading.get_ident(), graph)
        scope = self.backward
        require(scope['owner_thread'] == threading.main_thread().ident
                and scope['pid'] == self.pid and scope['root_calls'] == 1
                and type(scope['graph']) is int and scope['graph'] >= 0
                and graph == scope['graph'] and threading.get_ident() == scope['callback_thread'],
                'Only exact root-Tensor graph and registered callback thread inside owned backward scope')
        return (scope['sequence'], threading.get_ident(), graph)

    @contextmanager
    def backward_scope(self, tensor, torch):
        _serial(torch)
        require(os.getpid() == self.pid and not self.poisoned and self.backward is None
                and torch._C._current_graph_task_id() == -1
                and self.metrics['inflight_matrices'] == 0
                and self.model._diagnostic_frozen_metrics['logical_current_matrix_bytes'] == 0
                and all(not e['busy'] and e['module']._parameters['weight'] is e['cpu_parameter'] for e in self.entries),
                'One main-owned serialized backward at an idle cache boundary')
        require(isinstance(tensor, torch.Tensor) and tensor.requires_grad and tensor.grad_fn is not None,
                'Actual Tensor.backward self with a live differentiable graph required')
        self.backward_sequence += 1
        scope = {'sequence': self.backward_sequence, 'pid': self.pid,
                 'owner_thread': threading.get_ident(), 'tensor_id': id(tensor),
                 'tensor_pointer': tensor.data_ptr(), 'tensor_version': tensor._version,
                 'graph': None, 'callback_thread': None, 'root_calls': 0}
        self.backward = scope
        handle = None
        def bind_root(gradient):
            try:
                graph = torch._C._current_graph_task_id()
                require(os.getpid() == self.pid and self.backward is scope and not self.poisoned
                        and type(graph) is int and graph >= 0 and scope['root_calls'] == 0
                        and self.metrics['inflight_matrices'] == 0
                        and id(tensor) == scope['tensor_id'] and tensor.data_ptr() == scope['tensor_pointer']
                        and tensor._version == scope['tensor_version'],
                        'Only the actual owned root Tensor may register one live GraphTask')
                scope['graph'], scope['callback_thread'], scope['root_calls'] = graph, threading.get_ident(), 1
                return gradient  # Same object: no clone/cast/math/RNG or gradient rewrite.
            except BaseException:
                self.poisoned = True
                raise
        try:
            handle = tensor.register_hook(bind_root)
            yield
            _serial(torch)
            require(self.backward is scope and not self.poisoned and scope['root_calls'] == 1
                    and self.metrics['inflight_matrices'] == 0
                    and self.model._diagnostic_frozen_metrics['logical_current_matrix_bytes'] == 0
                    and tensor.data_ptr() == scope['tensor_pointer'] and tensor._version == scope['tensor_version']
                    and all(not e['busy'] and e['module']._parameters['weight'] is e['cpu_parameter']
                            and e['cpu_parameter']._version == e['parameter_version']
                            and e['gpu_parameter']._version == e['gpu_version'] for e in self.entries),
                    'Owned backward must return once with unchanged root/weights and no in-flight cache hooks')
            self.metrics['backward_scopes_completed'] = self.metrics.get('backward_scopes_completed', 0) + 1
            self.last_backward = dict(scope)
        except BaseException:
            self.poisoned = True
            raise
        finally:
            if handle is not None:
                handle.remove()
            self.backward = None

    def _restore_failed_hook(self, entry, current):
        if entry['busy']:
            current._parameters['weight'] = entry['cpu_parameter']
            entry['busy'] = False
            self.metrics['inflight_matrices'] -= 1
            self.model._diagnostic_frozen_metrics['logical_current_matrix_bytes'] -= entry['size']

    def hooks(self, entry, torch):
        import sys
        from torch.utils.checkpoint import _StopRecomputationError
        def before(current, inputs):
            try:
                ticket = self._hook_owner(torch)
                require(not self.poisoned and os.getpid() == self.pid and not entry['busy']
                        and self.metrics['inflight_matrices'] == 0
                        and self.model._diagnostic_frozen_metrics['logical_current_matrix_bytes'] == 0
                        and current._parameters['weight'] is entry['cpu_parameter']
                        and entry['cpu_parameter']._version == entry['parameter_version']
                        and entry['gpu_parameter']._version == entry['gpu_version']
                        and not self.model.encoder.training and not hasattr(self.model, '_orig_mod'),
                        'Exact quiescent readonly CPU Parameter at cache entry')
                current._parameters['weight'] = entry['gpu_parameter']; entry['busy'] = True
                entry['hook_owner'] = ticket
                self.metrics['cache_hit_reuses'] += 1; self.metrics['inflight_matrices'] += 1
                old = self.model._diagnostic_frozen_metrics
                old['forward_materializations'] += 1
                old['logical_current_matrix_bytes'] += entry['size']
                old['logical_peak_matrix_bytes'] = max(old['logical_peak_matrix_bytes'], old['logical_current_matrix_bytes'])
            except BaseException:
                self.poisoned = True
                raise
        def after(current, inputs, output):
            # always_call also runs if a pre-hook/module throws. Restore only
            # if this cache actually entered; do not subtract on failed entry.
            # Nonreentrant checkpoint normally stops recomputation by raising
            # this exact private exception inside a saved-tensor pack. Module
            # always_call hooks then receive None before checkpoint catches it.
            # Restore CPU registration on that control flow without poisoning;
            # all real forward/prehook failures and weight mismatches still stop.
            if output is None and type(sys.exc_info()[1]) is not _StopRecomputationError:
                self.poisoned = True
            if entry['busy']:
                try:
                    valid = (self._hook_owner(torch) == entry['hook_owner']
                             and self.metrics['inflight_matrices'] == 1
                             and self.model._diagnostic_frozen_metrics['logical_current_matrix_bytes'] == entry['size']
                             and current._parameters['weight'] is entry['gpu_parameter']
                             and entry['gpu_parameter']._version == entry['gpu_version'])
                    require(valid, 'Cache owner/weight changed or overlapping hooks; CPU Parameter restored and worker must stop')
                except BaseException:
                    self.poisoned = True
                    raise
                finally:
                    self._restore_failed_hook(entry, current)
        return before, after


def install(model, *, torch, bank_binding, guard, admit_cache, owner_state_digest):
    """Future reviewed opt-in only; mandatory exact owner state and admission.

    admit_cache(request)->True must use the full unchanged worker/global/CUDA
    budgets, count the persistent bank and keep native end-cache<=3GB. There
    is no forecast deduction or implicit fallback. Any exception is fail-stop.
    """
    _serial(torch); guard()
    require(not hasattr(model, '_orig_mod') and not hasattr(model, '_ASR18_frozen_cuda_cache')
            and not model.encoder.training and str(torch.__version__).startswith('2.7.1'),
            'One opt-in installation after original bootstrap; eval encoder/uncompiled exact Torch family')
    require(record(LOADER)['sha256'] == LOADER_SHA and record(LORA)['sha256'] == LORA_SHA
            and bank_binding == BANK and record(BANK['path']) == BANK, 'Exact unchanged loader/LoRA/bank pins')
    bank = json.loads(Path(BANK['path']).read_bytes())
    rows = [[name, bank['matrices'][name]['shape'], bank['matrices'][name]['tensor_bytes'], bank['matrices'][name]['tensor']]
            for name in sorted(bank['matrices'])]
    require(bank['passed'] is True and bank['GPU_used'] is False and bank['matrix_count'] == COUNT
            and bank['mapped_BF16_weight_bytes'] == CACHE_BYTES and digest(rows) == TUPLE_SHA,
            'Exact frozen294 source tuple; no additional tensors or regenerated bank')
    mappings = model._diagnostic_frozen_mappings
    require(len(mappings) == COUNT and all(not mapping.closed for mapping in mappings)
            and model._diagnostic_frozen_metrics['forward_materializations'] == 0
            and model._diagnostic_frozen_metrics['logical_current_matrix_bytes'] == 0,
            'Before first native forward only; same open readonly mappings, no existing autograd graph')
    mapping_ids = tuple(id(mapping) for mapping in mappings)
    active = _active(model); owner_before = _owner_digest(owner_state_digest)
    entries = _entries(model, bank, torch); state = _InstanceCache(model, entries)
    callback_sources = {'owner_state_digest': _callback_source(owner_state_digest), 'admit_cache': _callback_source(admit_cache)}
    request = {'kind': 'ASR18_optional_frozen294_cache_allocation_request_v1', 'source': record(__file__), 'bank': BANK,
        'logical_new_persistent_cuda_bytes': CACHE_BYTES, 'maximum_single_H2D_matrix_bytes': LARGEST_COPY,
        'maximum_readback_staging_bytes': READBACK, 'host_private_savings_credited_bytes': 0,
        'full_worker_global_and_CUDA_admission_required': True, 'native_end_cache_cap_bytes': 3000000000,
        'old_hook_logical_current_excludes_persistent_bank': True}
    require(admit_cache(deepcopy_json(request)) is True, 'Separately reviewed full allocation admission required before any cache allocation')
    guard(); original_handles = list(model._diagnostic_frozen_handles); removed_originals = False
    try:
        # Same original CPU BF16.to(cuda) then Parameter(False), once per
        # matrix. No cast, merge, parameter mutation, RNG or operation rewrite.
        for entry in entries:
            guard()
            value = entry['cpu'].to(device='cuda', non_blocking=False)
            parameter = torch.nn.Parameter(value, requires_grad=False); del value
            require(parameter.dtype == torch.bfloat16 and parameter.device.type == 'cuda'
                    and parameter.is_contiguous() and tuple(parameter.shape) == tuple(entry['cpu'].shape), 'Exact cached matrix layout')
            entry['gpu_parameter'] = parameter; entry['gpu_version'] = parameter._version
            state.metrics['cache_initial_upload_bytes'] += entry['size']
            # Verify all bytes with bounded readback, never make a whole-bank
            # CPU clone or claim pointer/layout identity implies byte equality.
            cpu_bytes = entry['cpu'].detach().view(torch.uint8).reshape(-1)
            gpu_bytes = parameter.detach().view(torch.uint8).reshape(-1)
            for low in range(0, entry['size'], READBACK):
                guard(); high = min(entry['size'], low + READBACK)
                actual = gpu_bytes[low:high].to(device='cpu', non_blocking=False)
                require(torch.equal(actual, cpu_bytes[low:high]), 'Cached CUDA bytes differ from exact readonly BF16 source')
                state.metrics['cache_verification_readback_bytes'] += high - low
                del actual
            del cpu_bytes, gpu_bytes, parameter
        torch.cuda.synchronize(); guard()
        require(state.metrics['cache_initial_upload_bytes'] == state.metrics['cache_verification_readback_bytes'] == CACHE_BYTES,
                'All294 complete cache uploads/readbacks, never sampled')
        # No hook is removed until every cache byte has passed. A failed fill
        # leaves the original hook set installed; the caller still stops.
        removed_originals = True
        for handle in original_handles: handle.remove()
        for entry in entries:
            before, after = state.hooks(entry, torch)
            state.handles.append(entry['module'].register_forward_pre_hook(before))
            state.handles.append(entry['module'].register_forward_hook(after, always_call=True))
        require(len(state.handles) == HOOKS, 'Exactly588 cache hooks, no duplicate original hook')
        model._diagnostic_frozen_handles = state.handles
        model._ASR18_frozen_cuda_cache = state
        model._diagnostic_frozen_metrics['transferred_weight_bytes'] += CACHE_BYTES
        require(_active(model) == active and tuple(id(mapping) for mapping in model._diagnostic_frozen_mappings) == mapping_ids
                and _owner_digest(owner_state_digest) == owner_before, 'Active207/threeAdam/all4RNG/teacher or readonly mappings changed')
        assert_idle(model, torch=torch, guard=guard)
        policy = {'kind': 'ASR18_inactive_optional_exact_frozen294_CUDA_cache_v1', 'source': record(__file__),
            'loader': record(LOADER), 'LoRA': record(LORA), 'bank': BANK, 'matrix_tuple_sha256': TUPLE_SHA,
            'matrix_count': COUNT, 'hook_handles': HOOKS, 'persistent_cuda_cache_bytes': CACHE_BYTES,
            'maximum_readback_staging_bytes': READBACK, 'original_CPU_Parameter_identity_restored_between_calls': True,
            'original_readonly_mappings_retained': True, 'original_module_forward_operations_unchanged': True,
            'active207_threeAdam_all4RNG_and_teacher_owner_digest': owner_before, 'owner_callback_sources': callback_sources,
            'host_private_savings_credited_bytes': 0, 'speedup_measured': False, 'native_byte_equivalence_proved': False,
            'persistent_bank_in_all_future_CUDA_forecasts_and_end_cache_cap_required': True,
            'normal_uninstall_supported': False, 'cache_release_boundary': 'actual sole worker process exit only',
            'failure_policy': 'any install/hook failure poisons candidate; no training fallback or in-process retry',
            'production_enabled': False}
        state.policy = policy
        return policy
    except BaseException:
        state.poisoned = True
        # Restore CPU registration immediately; never close readonly mappings.
        cleanup_errors = []
        for entry in entries:
            try: entry['module']._parameters['weight'] = entry['cpu_parameter']; entry['busy'] = False
            except BaseException as error: cleanup_errors.append(type(error).__name__)
        for handle in state.handles + (original_handles if removed_originals else []):
            try: handle.remove()
            except BaseException as error: cleanup_errors.append(type(error).__name__)
        if removed_originals:
            restored = []
            for entry in entries:
                try:
                    restored.append(entry['module'].register_forward_pre_hook(entry['pre']))
                    restored.append(entry['module'].register_forward_hook(entry['post'], always_call=True))
                except BaseException as error: cleanup_errors.append(type(error).__name__)
            model._diagnostic_frozen_handles = restored
        state.failure_cleanup = {'attempted_CPU_restore_and_original_hook_rollback': True,
            'cleanup_errors': cleanup_errors, 'safe_to_continue_or_retry': False,
            'actual_worker_exit_required': True, 'readonly_mappings_closed_here': False}
        model._ASR18_frozen_cuda_cache = state
        # Keep failed state truthful and process-owned. Actual worker exit is
        # the only supported resource-release boundary, even after rollback.
        raise


def deepcopy_json(value):
    return json.loads(json.dumps(value, allow_nan=False))


def assert_idle(model, *, torch, guard):
    _serial(torch); guard(); state = model._ASR18_frozen_cuda_cache
    require(type(state) is _InstanceCache and not state.poisoned and state.pid == os.getpid() and state.backward is None
            and state.metrics['inflight_matrices'] == model._diagnostic_frozen_metrics['logical_current_matrix_bytes'] == 0,
            'No in-flight or poisoned cache; old logical_current is not a persistent-CUDA inventory')
    for entry in state.entries:
        parameter = entry['module']._parameters['weight']
        require(parameter is entry['cpu_parameter'] and parameter.device.type == 'cpu' and not parameter.requires_grad
                and parameter._version == entry['parameter_version'] and entry['cpu']._version == entry['cpu_version']
                and parameter.data_ptr() == entry['cpu'].data_ptr() and not entry['busy']
                and entry['gpu_parameter'].device.type == 'cuda' and not entry['gpu_parameter'].requires_grad
                and entry['gpu_parameter']._version == entry['gpu_version'],
                'Same original CPU Parameter/backing and persistent frozen CUDA cache')
    return {'passed': True, 'persistent_cuda_cache_bytes': CACHE_BYTES,
            'logical_current_streamed_matrix_bytes': 0, 'metrics': dict(state.metrics)}


@contextmanager
def backward_scope(model, tensor, *, torch):
    """Candidate-only wrapper around exactly one original Tensor.backward call.

    Install/assert_idle stay main-thread-only. The actual root Tensor's temporary
    hook binds its GraphTask and one callback thread; unrelated autograd or named
    foreign threads never receive a cache permission. No original math changes.
    """
    _serial(torch)
    state = model._ASR18_frozen_cuda_cache
    require(type(state) is _InstanceCache, 'Exact installed instance cache3 required')
    with state.backward_scope(tensor, torch):
        yield
