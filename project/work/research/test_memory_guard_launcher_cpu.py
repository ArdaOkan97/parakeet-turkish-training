"""Mocked CPU guards for the new-process allocator cap wrapper."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import uuid
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2];OUTPUT=ROOT/'outputs/parakeet_turkish'
sys.path.insert(0,str(OUTPUT))
import launch_with_memory_guard_v1 as wrapper
import launch_with_cache_reclaim_v1 as cache
from atomic_io_retry_v1 import read_json,write_json
from cuda_cache_reclaim_v1 import file_sha


def main():
    checks=[]
    def require(value,name):assert value,name;checks.append(name)
    capacity=17_179_344_896
    valid=wrapper.validate_capacity(capacity)
    require(valid['cap_bytes']==13_000_000_000 and valid['per_process_allocator_fraction']==13e9/capacity,'Exact13GB decimal cap fraction')
    for memory,cap in [(8e9,13),(24e9,13),(capacity,8),(capacity,13.01)]:
        try:wrapper.validate_capacity(memory,cap)
        except ValueError:checks.append('Unreviewed capacity/cap rejected '+str((memory,cap)))
        else:raise AssertionError('capacity/cap')
    scratch=ROOT/'work/research/memory_guard_cpu'/uuid.uuid4().hex;scratch.mkdir(parents=True)
    def arguments(prefix='',extra=()):
        return wrapper.parser().parse_args(['--provenance',str(scratch/(prefix+'io.json')),
            '--cache-provenance',str(scratch/(prefix+'cache.json')),'--memory-provenance',str(scratch/(prefix+'memory.json')),
            *extra,'--','--run-id','CPU_FORK','--resume-exact','owned_fork.pt'])
    calls=[]
    def properties(device):calls.append(('get_properties',device));return SimpleNamespace(total_memory=capacity,name='CPU mock16GB')
    def cap_api(fraction,device):calls.append(('set_fraction',fraction,device))
    fake_cuda=SimpleNamespace(is_initialized=lambda:False,get_device_properties=properties,set_per_process_memory_fraction=cap_api)
    args=arguments('validate_',('--validate-only',))
    with patch.dict(sys.modules,{'torch':SimpleNamespace(cuda=fake_cuda)}):wrapper.main(args)
    require(not calls and not args.memory_provenance.exists(),'CPU validation never initializesCUDA/appliescap/writesGPUevidence')
    args=arguments()
    def nested(child):
        require(calls==[('get_properties',0),('set_fraction',13e9/capacity,0)],'Cap API called before nested trainer/cache launcher')
        require(child.arguments==args.arguments and child.threshold_gb==8 and not child.validate_only,'Workerargv/8GB cleanup policy unchanged')
    protected={path:file_sha(path) for path in [OUTPUT/'cuda_cache_reclaim_v1.py',OUTPUT/'launch_with_cache_reclaim_v1.py',
        OUTPUT/'launch_with_io_retry_v1.py',OUTPUT/'train_lora_batched.py',OUTPUT/'run_experiment_loop_recovery_v2.py']}
    with patch.dict(sys.modules,{'torch':SimpleNamespace(cuda=fake_cuda)}),patch.object(cache,'main',side_effect=nested):wrapper.main(args)
    evidence=read_json(args.memory_provenance)
    require(evidence['status']=='complete' and evidence['allocator_cap_applied'] is True,'Actual mocked cap application and completion recorded')
    require(evidence['global_driver_settings_changed'] is False and evidence['backend_flags_changed'] is False
            and evidence['RNG_sampler_optimizer_changed'] is False,'Perprocess allocator change clearly separate from global/math/backend/RNG')
    require(evidence['memory_launcher_sha256']==evidence['memory_launcher_sha256_after'],'Actual wrapper before/aftersourceSHA')
    try:wrapper.main(args)
    except FileExistsError:checks.append('Existing provenance cannot be overwritten')
    else:raise AssertionError('overwrite')
    failed=arguments('failed_');calls.clear()
    def fail_api(*args,**kwargs):raise RuntimeError('mock cap failure')
    failed_cuda=SimpleNamespace(is_initialized=lambda:False,get_device_properties=properties,set_per_process_memory_fraction=fail_api)
    with patch.dict(sys.modules,{'torch':SimpleNamespace(cuda=failed_cuda)}),patch.object(cache,'main') as child:
        try:wrapper.main(failed)
        except RuntimeError as error:require(str(error)=='mock cap failure','OriginalcapAPIerror propagates')
        else:raise AssertionError('API failure')
        require(child.call_count==0 and read_json(failed.memory_provenance)['status']=='failed','Failedcap preventsGPUtraining and retainsfailureevidence')
    for path,digest in protected.items():require(file_sha(path)==digest,'Frozen previoussource unchanged '+path.name)
    report={'passed':True,'check_count':len(checks),'checks':checks,'wrapper_sha256':file_sha(OUTPUT/'launch_with_memory_guard_v1.py'),
            'scope':'CPU mocked CUDAdevice/cap API and nested launcher only; no GPU initialized or work launched'}
    write_json(OUTPUT/'memory_guard_launcher_v1_cpu_verification.json',report)
    print(json.dumps({'passed':True,'check_count':len(checks),'wrapper_sha256':report['wrapper_sha256']}))


if __name__=='__main__':main()
