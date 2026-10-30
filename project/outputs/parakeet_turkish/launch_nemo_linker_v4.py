"""Preload one coherent CUDA 12.9 linker/compiler set before unchanged proof.

This launcher changes library loading order only. CPU preflight neither obtains
a CUDA device nor initializes CUDA. Root must authorize actual GPU proof.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / 'work/nemo-native-env-v2/Lib/site-packages'
VERIFIER = ROOT / 'outputs/parakeet_turkish/verify_nemo_native_loss_v3.py'
VERIFIER_SHA = 'e4e7653f941ddd75cb048db4d8e5d803f1906354736b031e323dcf73362a2944'
LIBRARIES = {
    'nvjitlink': SITE / 'nvidia/nvjitlink/bin/nvJitLink_120_0.dll',
    'nvvm': SITE / 'nvidia/cuda_nvcc/nvvm/bin/nvvm64_40_0.dll',
    'nvrtc': SITE / 'nvidia/cuda_nvrtc/bin/nvrtc64_120_0.dll',
}


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def loaded_modules():
    """Only enumerate this process, using Windows' read-only module API."""
    require(os.name == 'nt', 'Native Windows launcher required')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.EnumProcessModules.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                         ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong)]
    psapi.EnumProcessModules.restype = ctypes.c_int
    psapi.GetModuleFileNameExW.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                         ctypes.c_wchar_p, ctypes.c_ulong]
    psapi.GetModuleFileNameExW.restype = ctypes.c_ulong
    modules = (ctypes.c_void_p * 4096)()
    needed = ctypes.c_ulong()
    process = kernel.GetCurrentProcess()
    require(psapi.EnumProcessModules(process, modules, ctypes.sizeof(modules),
                                    ctypes.byref(needed)), 'EnumProcessModules failed')
    require(needed.value <= ctypes.sizeof(modules), 'Module inventory bound exceeded')
    result = []
    for i in range(needed.value // ctypes.sizeof(ctypes.c_void_p)):
        text = ctypes.create_unicode_buffer(32768)
        require(psapi.GetModuleFileNameExW(process, modules[i], text, len(text)),
                'GetModuleFileNameExW failed')
        name = Path(text.value).name.lower()
        if any(x in name for x in ('nvjitlink', 'nvvm', 'nvrtc', 'nvcuda', 'cudart')):
            path = Path(text.value).resolve()
            result.append({'path': str(path), 'sha256': sha(path)})
    return result


def pinned_loaded(modules):
    for name, path in LIBRARIES.items():
        same_name = [row for row in modules if Path(row['path']).name.lower() == path.name.lower()]
        require(len(same_name) == 1 and Path(same_name[0]['path']) == path.resolve(),
                f'{name}: other toolkit DLL loaded or coherent DLL unavailable')


def binding_linker():
    """Resolve the DLL owning the actual CUDA Python nvJitLink function."""
    from cuda.bindings import nvjitlink
    from cuda.bindings._internal import nvjitlink as internal
    result = nvjitlink.version()
    pointer = internal._inspect_function_pointer('__nvJitLinkVersion')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetModuleHandleExW.argtypes = [ctypes.c_ulong, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    kernel.GetModuleHandleExW.restype = ctypes.c_int
    kernel.GetModuleFileNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_ulong]
    kernel.GetModuleFileNameW.restype = ctypes.c_ulong
    handle = ctypes.c_void_p()
    require(kernel.GetModuleHandleExW(6, ctypes.c_void_p(pointer), ctypes.byref(handle)),
            'Cannot resolve linker function owner')
    name = ctypes.create_unicode_buffer(32768)
    require(kernel.GetModuleFileNameW(handle, name, len(name)), 'Cannot read linker owner path')
    path = Path(name.value).resolve()
    return {'version': [int(result[0]), int(result[1])], 'path': str(path), 'sha256': sha(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-preflight', action='store_true')
    parser.add_argument('--gpu-window-authorized', action='store_true')
    parser.add_argument('--input', type=Path)
    parser.add_argument('--input-sha256')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    require(args.cpu_preflight != args.gpu_window_authorized, 'Choose CPU preflight or explicit GPU proof')
    require(sha(VERIFIER) == VERIFIER_SHA, 'Frozen verifier changed')
    require(Path(sys.executable).resolve().parent.parent == SITE.parent.parent,
            'Use isolated native-env-v2 interpreter')
    output = args.output.resolve()
    require(output.is_relative_to(ROOT) and not output.exists(), 'New workspace output required')
    provenance = output.with_name(output.stem + '_linker_provenance.json')
    require(not provenance.exists(), 'Preserve previous provenance')
    # Keep both DLL-directory handles and DLL objects alive through the proof.
    directories = [os.add_dll_directory(str(path.parent)) for path in LIBRARIES.values()]
    libraries = {name: ctypes.CDLL(str(path)) for name, path in LIBRARIES.items()}
    major, minor = ctypes.c_uint(), ctypes.c_uint()
    version = libraries['nvjitlink'].nvJitLinkVersion
    version.argtypes = [ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint)]
    version.restype = ctypes.c_int
    require(version(ctypes.byref(major), ctypes.byref(minor)) == 0, 'nvJitLinkVersion failed')
    require((major.value, minor.value) == (12, 9), 'Expected CUDA 12.9 nvJitLink')
    evidence = {'schema': 1, 'kind': 'absolute_cuda_linker_preload_v4',
                'status': 'preparing', 'launcher_sha256': sha(__file__),
                'verifier_sha256': VERIFIER_SHA, 'pid': os.getpid(),
                'library_files': {k: {'path': str(p.resolve()), 'sha256': sha(p)} for k, p in LIBRARIES.items()},
                'nvjitlink_version': [major.value, minor.value],
                'scope': 'Library loading order only; unchanged full-logit official loss proof',
                'before_torch': loaded_modules()}
    pinned_loaded(evidence['before_torch'])
    import torch
    import numba.cuda
    from cuda.pathfinder import load_nvidia_dynamic_lib
    evidence['torch_version'] = torch.__version__
    evidence['torch_path'] = torch.__file__
    evidence['numba_cuda_path'] = numba.cuda.__file__
    evidence['pathfinder_paths'] = {k: str(load_nvidia_dynamic_lib('nvJitLink' if k == 'nvjitlink' else k).abs_path)
                                   for k in LIBRARIES}
    evidence['after_torch_and_compiler'] = loaded_modules()
    # Torch may retain its own 12.8 DLL too; what matters for the native proof
    # is the actual library selected by the CUDA bindings, checked below.
    for name, expected in LIBRARIES.items():
        require(Path(evidence['pathfinder_paths'][name]).resolve() == expected.resolve(),
                f'Pathfinder did not use preloaded {name}')
    evidence['actual_binding_linker'] = binding_linker()
    require(Path(evidence['actual_binding_linker']['path']) == LIBRARIES['nvjitlink'].resolve()
            and evidence['actual_binding_linker']['version'] == [12, 9],
            'Actual CUDA binding did not select CUDA 12.9 linker')
    require(not torch.cuda.is_initialized(), 'CPU preload unexpectedly initialized CUDA')
    evidence['cuda_initialized_before_proof'] = False
    if args.cpu_preflight:
        evidence.update(status='complete', passed=True, CUDA_executed=False)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(evidence, indent=2), encoding='utf-8')
        print(json.dumps({'passed': True, 'nvjitlink_version': evidence['nvjitlink_version'],
                          'CUDA_executed': False, 'output': str(output)}))
        return
    require(args.input is not None and args.input_sha256 is not None, 'Bounded actual TRAIN input required')
    require(sha(args.input) == args.input_sha256, 'Actual TRAIN capture changed')
    evidence['input_sha256'] = args.input_sha256
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        sys.argv = [str(VERIFIER), '--gpu-window-authorized', '--input', str(args.input),
                    '--input-sha256', args.input_sha256, '--output', str(output)]
        runpy.run_path(str(VERIFIER), run_name='__main__')
        evidence.update(status='complete', passed=True)
    except BaseException as error:
        evidence.update(status='failed', passed=False, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        evidence['after_proof'] = loaded_modules()
        evidence['actual_binding_linker_after_proof'] = binding_linker()
        evidence['coherent_libraries_after_proof'] = all(
            sum(Path(row['path']) == path.resolve() for row in evidence['after_proof']) == 1
            for path in LIBRARIES.values())
        provenance.write_text(json.dumps(evidence, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
