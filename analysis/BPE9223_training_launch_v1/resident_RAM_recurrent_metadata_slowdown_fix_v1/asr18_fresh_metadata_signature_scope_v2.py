"""Inactive explicit process-local alias integration; original source unchanged.

Install only after original context/hash/three denywrite handles are admitted.
Main-thread driver owns lifetime; restore only after private worker has joined.
The feature CPU is a separate interpreter and is never patched by this scope.
"""
import threading
from pathlib import Path
from bpe_windows_final_metadata_lock_v1 import FinalMetadataReadLocks
import asr18_BPE9223_source_context_v2 as context_module
import bpe_declared_geometry_index_v1 as geometry_module
import asr18_BPE9223_private_scalar_prefetch_v1 as prefetch_module
from asr18_fresh_single_open_metadata_signature_v2 import FreshSingleOpenSignature,ORIGINAL_SIGNATURE

class FreshMetadataSignatureScope:
    def __init__(self,context,*,private_worker_closed):
        if threading.current_thread()is not threading.main_thread():raise ValueError('Original main-thread integration authority')
        if type(context)is not context_module.FullRunSourceContext9223 or context.closed:
            raise ValueError('Original already-open exact full-run context')
        if not callable(private_worker_closed):raise ValueError('Callable original private-worker closure check')
        resolver=context.resolver
        if type(resolver)is not context_module.FinalBroadRows9223 or resolver.poisoned:
            raise ValueError('Original admitted broad resolver')
        expected=[*resolver.signatures,resolver.geometry.signature]
        if (type(resolver.metadata_locks)is not FinalMetadataReadLocks
            or len(resolver.metadata_locks.handles)!=3
            or resolver.metadata_locks.paths!=[Path(v['path']) for v in expected]):
            raise ValueError('All three exact original denywrite lock paths and handles remain live')
        self.reader=FreshSingleOpenSignature(expected)
        self.modules=(context_module,geometry_module,prefetch_module)
        self.private_worker_closed=private_worker_closed
        self.originals={m.__name__:m.signature for m in self.modules}
        if not all(v is ORIGINAL_SIGNATURE for v in self.originals.values()):
            raise ValueError('All three original signature aliases exact before explicit installation')
        self.installed=self.reader.signature;self.active=False;self.closed=False;self.deferred_to_abnormal_exit=False

    def __enter__(self):
        if self.active or self.closed:raise ValueError('Signature scope used exactly once')
        if threading.current_thread()is not threading.main_thread():raise ValueError('Main-thread installation')
        for module in self.modules:module.signature=self.installed
        self.active=True
        return self

    def __exit__(self,exc_type,exc_value,traceback):
        if not self.active:return False
        try:joined=self.private_worker_closed()
        except BaseException:
            self.deferred_to_abnormal_exit=True
            if exc_type is None:raise
            return False
        if joined is not True:
            self.deferred_to_abnormal_exit=True
            if exc_type is None:raise ValueError('Cannot restore aliases while original private worker remains live')
            return False
        changed=[m.__name__ for m in self.modules if m.signature is not self.installed]
        for module in self.modules:
            if module.signature is self.installed:module.signature=self.originals[module.__name__]
        self.active=False;self.closed=not changed
        if changed and exc_type is None:raise ValueError('Unexpected signature alias replacement during scoped lifetime')
        return False

    def status(self):
        return {'active':self.active,'closed':self.closed,'deferred_to_abnormal_exit':self.deferred_to_abnormal_exit,
            'exact_original_aliases_before':all(v is ORIGINAL_SIGNATURE for v in self.originals.values()),
            'canonical_fast_path_names':sorted(self.reader.allowed),
            'original_three_modules':[m.__name__ for m in self.modules],
            'fresh_size_mtime_and_path_each_call':True,'no_guard_check_throttling_or_cached_success':True}
