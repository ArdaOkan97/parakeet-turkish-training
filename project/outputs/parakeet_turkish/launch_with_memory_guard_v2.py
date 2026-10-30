"""New-process13GB allocator cap around the immutable cache/I/O launcher with6GB threshold.

The cap is per process, not a driver-wide setting, and applies only to PyTorch's
CUDA allocator. Quality/retention gates remain mandatory. CPU --validate-only
does not initialize CUDA; actual16GB-device validation occurs at GPU launch.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from atomic_io_retry_v1 import write_json
from cuda_cache_reclaim_v1 import file_sha

OUTPUT=Path(__file__).resolve().parent
ROOT=OUTPUT.parents[1]
CACHE_LAUNCHER_SHA256='db1ad326f26a89c5163a34a839cc781e457793b83d625a7c99b8366db9301459'
CACHE_HELPER_SHA256='99ec5974fcd4714fa6fa341eb5cde70dc817085c997bd564c2b465aaaf98a543'


def validate_capacity(total_bytes,cap_gb=13.):
    if not 15e9<=total_bytes<=18e9:raise ValueError('The guarded launch requires the reviewed16GB-class CUDA device')
    if not 6<cap_gb<=13:raise ValueError('Per-process cap must exceed6GB cleanup threshold and stay<=13GB')
    fraction=cap_gb*1e9/total_bytes
    if not 0<fraction<.9:raise ValueError('Conservative allocator fraction required')
    return {'physical_cuda_device_bytes':int(total_bytes),'cap_bytes':int(cap_gb*1e9),
            'per_process_allocator_fraction':fraction,'cuda_device_index':0}


def main(args):
    for name,expected in [('launch_with_cache_reclaim_v1.py',CACHE_LAUNCHER_SHA256),
                          ('cuda_cache_reclaim_v1.py',CACHE_HELPER_SHA256)]:
        if file_sha(OUTPUT/name)!=expected:raise RuntimeError('Reviewed cache source differs: '+name)
    paths=[args.provenance,args.cache_provenance,args.memory_provenance]
    if len({p.resolve() for p in paths})!=3:raise ValueError('Three distinct new provenance files required')
    for path in paths:
        if not path.resolve().is_relative_to(ROOT.resolve()):raise ValueError('Provenance must belong to the workspace')
        if path.exists():raise FileExistsError('Existing provenance is immutable: '+str(path))
    if not 6<args.cap_gb<=13:raise ValueError('Reviewed cap must exceed6GB and stay<=13GB')
    if args.threshold_gb!=6.:raise ValueError('Reviewed cleanup threshold is exactly6GB')
    import torch
    sources={'memory_launcher_sha256':file_sha(__file__),'cache_launcher_sha256':CACHE_LAUNCHER_SHA256,
             'cache_helper_sha256':CACHE_HELPER_SHA256}
    if args.validate_only:
        print({'passed':True,'CUDA_initialized':torch.cuda.is_initialized(),'device_validation_pending_GPU_launch':True,
               'cap_gb':args.cap_gb,'cleanup_threshold_gb':args.threshold_gb,**sources});return
    properties=torch.cuda.get_device_properties(0)
    cap=validate_capacity(properties.total_memory,args.cap_gb)
    evidence={'schema':1,'kind':'memory_guard_training_launcher_v2',**sources,**cap,
              'cuda_device_name':properties.name,'cap_scope':'This process only; PyTorch CUDA allocator only',
              'actual_API':'torch.cuda.set_per_process_memory_fraction(fraction, device=0)',
              'allocator_cap_may_trigger_native_reclaim_or_OOM':True,'cleanup_threshold_bytes':6_000_000_000,'global_driver_settings_changed':False,
              'training_math_changed':False,'backend_flags_changed':False,'RNG_sampler_optimizer_changed':False,
              'cache_provenance_path':str(args.cache_provenance.resolve()),'io_provenance_path':str(args.provenance.resolve()),
              'arguments':args.arguments,'started_at_utc':datetime.now(timezone.utc).isoformat(),'status':'running'}
    write_json(args.memory_provenance,evidence)
    try:
        torch.cuda.set_per_process_memory_fraction(cap['per_process_allocator_fraction'],device=0)
        evidence['allocator_cap_applied']=True
        write_json(args.memory_provenance,evidence)
        import launch_with_cache_reclaim_v1 as cache_launcher
        cache_launcher.main(SimpleNamespace(provenance=args.provenance,cache_provenance=args.cache_provenance,
            threshold_gb=args.threshold_gb,max_events=args.max_events,retry_seconds=args.retry_seconds,
            validate_only=False,arguments=args.arguments))
        evidence['status']='complete'
    except BaseException as error:
        evidence.update(status='failed',failure_type=type(error).__name__,failure_message=str(error));raise
    finally:
        evidence.update(finished_at_utc=datetime.now(timezone.utc).isoformat(),
            memory_launcher_sha256_after=file_sha(__file__),
            cache_launcher_sha256_after=file_sha(OUTPUT/'launch_with_cache_reclaim_v1.py'),
            cache_helper_sha256_after=file_sha(OUTPUT/'cuda_cache_reclaim_v1.py'))
        write_json(args.memory_provenance,evidence)


def parser():
    value=argparse.ArgumentParser(description=__doc__)
    value.add_argument('--provenance',type=Path,required=True)
    value.add_argument('--cache-provenance',type=Path,required=True)
    value.add_argument('--memory-provenance',type=Path,required=True)
    value.add_argument('--cap-gb',type=float,default=13.)
    value.add_argument('--threshold-gb',type=float,default=6.)
    value.add_argument('--max-events',type=int,default=128)
    value.add_argument('--retry-seconds',type=float,default=30.)
    value.add_argument('--validate-only',action='store_true')
    value.add_argument('arguments',nargs=argparse.REMAINDER)
    return value


if __name__=='__main__':main(parser().parse_args())
