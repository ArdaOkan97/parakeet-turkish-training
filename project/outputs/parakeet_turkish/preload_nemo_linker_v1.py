"""Callable coherent CUDA12.9 preload; call before importing Torch or NeMo."""
import ctypes
import os
from pathlib import Path
import sys

import launch_nemo_linker_v4 as pinned

LAUNCHER_SHA = 'bdd7078dff81adc961c17588ed7fd84b152da71c619c2ff114994fa68d308af0'
_HANDLES = []


def preload():
    pinned.require(pinned.sha(pinned.__file__) == LAUNCHER_SHA, 'Pinned successful linker launcher changed')
    pinned.require('torch' not in sys.modules and 'numba.cuda' not in sys.modules,
                   'Coherent linker preload must precede Torch/Numba import')
    pinned.require(Path(sys.executable).resolve().parent.parent == pinned.SITE.parent.parent,
                   'Use isolated native-env-v2 interpreter')
    for path in pinned.LIBRARIES.values():
        _HANDLES.append(os.add_dll_directory(str(path.parent)))
        _HANDLES.append(ctypes.CDLL(str(path)))
    before = pinned.loaded_modules()
    pinned.pinned_loaded(before)
    import torch
    import numba.cuda
    from cuda.pathfinder import load_nvidia_dynamic_lib
    actual = pinned.binding_linker()
    pinned.require(actual['version'] == [12, 9] and
                   Path(actual['path']) == pinned.LIBRARIES['nvjitlink'].resolve(),
                   'Actual CUDA binding must use the successful 12.9 linker')
    paths = {name: str(load_nvidia_dynamic_lib('nvJitLink' if name == 'nvjitlink' else name).abs_path)
             for name in pinned.LIBRARIES}
    pinned.require(all(Path(paths[name]).resolve() == path.resolve()
                       for name, path in pinned.LIBRARIES.items()), 'CUDA compiler library path changed')
    pinned.require(not torch.cuda.is_initialized(), 'Preload must not initialize CUDA')
    return {'kind': 'coherent_native_cuda12_9_preload_v1',
            'module_sha256': pinned.sha(__file__), 'successful_launcher_sha256': LAUNCHER_SHA,
            'actual_binding_linker': actual, 'pathfinder_paths': paths,
            'cuda_initialized': False, 'before_torch': before,
            'after_torch': pinned.loaded_modules()}
