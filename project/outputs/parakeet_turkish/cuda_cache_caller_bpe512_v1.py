"""Isolated BPE caller binding; inherit unchanged inactive-cache operations."""
import ast
from collections import deque
from pathlib import Path
from cuda_cache_reclaim_v2 import CudaCacheReclaimer as Original, file_sha

WORKER_SHA = '9dca06b398a028d1f8fdc84c984f297f3874cb58917150a5199d6f6dfeb52f11'
HELPER_SHA = '869b9ffaac70c9b7b6e7590ba1d330363100d6b2ddb9f8f7af9b2b9865c1c7f9'


def caller_line(path, expected_sha=WORKER_SHA):
    path = Path(path).resolve()
    if file_sha(path) != expected_sha or expected_sha != WORKER_SHA:
        raise RuntimeError('BPE worker differs from reviewed caller source')
    if file_sha(Path(__file__).with_name('cuda_cache_reclaim_v2.py')) != HELPER_SHA:
        raise RuntimeError('Inherited cache implementation changed')
    tree = ast.parse(path.read_text(encoding='utf-8'))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run']
    calls = [node for function in functions for node in ast.walk(function) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute) and node.func.attr == 'synchronize'
        and ast.unparse(node.func.value) == 'torch.cuda']
    if len(functions) != 1 or len(calls) != 1 or calls[0].lineno != 588 or calls[0].args or calls[0].keywords:
        raise RuntimeError('Expected reviewed single argument-free BPE run:588 synchronize')
    return 588


class CudaCacheReclaimer(Original):
    def __init__(self, cuda, worker_path, *, threshold_gb=4., max_events=128, expected_sha=WORKER_SHA):
        if threshold_gb != 4. or not 1 <= max_events <= 1024:
            raise ValueError('Unchanged4GB cache threshold and bounded events required')
        self.cuda = cuda; self.worker_path = Path(worker_path).resolve(); self.expected_sha = expected_sha
        self.line = caller_line(self.worker_path, expected_sha)
        self.threshold_bytes = int(threshold_gb * 1e9); self.max_events = max_events
        self.events = deque(maxlen=max_events)
        self.eligible_calls = self.ignored_calls = self.reclamations = self.failed_reclamations = 0
        self.cleanup_seconds = 0.; self.original = self.wrapped = None

    def evidence(self):
        return {**super().evidence(), 'caller_binding_sha256': file_sha(__file__),
            'inherited_cache_implementation_sha256': HELPER_SHA}
