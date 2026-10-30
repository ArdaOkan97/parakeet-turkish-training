"""CPU-only measured9GB/4GB policy proof; no model load or GPU execution."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import torch
from atomic_io_retry_v1 import read_json,write_json
import run_experiment_loop_continuation_v3 as controller
import profile_cuda_cache_reclaim_v4 as probe_worker

OUTPUT=Path(__file__).resolve().parent
ROOT=OUTPUT.parents[1]

def ref(path):
    path=controller.scoped(path)
    return {'path':str(path),'sha256':controller.sha(path)}

def build(args):
    for path in (args.output,args.verification):
        controller.require(not path.exists(),'Existing policy evidence is immutable: '+str(path))
    torch.set_num_threads(2)
    controller.require(not torch.cuda.is_initialized(),'CPU validation must not initialize CUDA')
    report=read_json(args.probe,{})
    controller.require(report.get('status')=='complete' and report.get('probe_version')==4,'Actual completed v4 probe required')
    controller.require(controller.sha(probe_worker.__file__)==controller.FROZEN['profile_cuda_cache_reclaim_v4.py'],'Frozen probe worker changed')
    capture=controller.scoped(report['full_capture_path'])
    controller.require(controller.sha(capture)==report['full_capture_sha256'],'Actual measured tensor capture changed')
    values=torch.load(capture,map_location='cpu',weights_only=True)
    count=report['shared_prefix_updates'];checks=[];actual={}
    for step in sorted({1,count}):
        reference=values[('control',step)];candidate=values[('reclaim',step)]
        for category in ('gradients','weights'):
            controller.require(len(reference[category])==len(candidate[category])==207,'All207 trainable tensors required')
            controller.require(set(reference[category])==set(candidate[category]),'Matched trainable keys differ')
            for name,left in reference[category].items():
                right=candidate[category][name]
                controller.require(left.dtype==right.dtype and left.shape==right.shape and left.device.type==right.device.type=='cpu'
                  and torch.isfinite(left).all() and torch.isfinite(right).all(),'Measured tensor shape/dtype/finiteness differs: '+name)
            checks.append('All207 '+category+' tensors finite/CPU/shape matched at shared update'+str(step))
        comparison=probe_worker.compare_capture(reference,candidate)
        declared=report['comparisons'][str(step)]
        controller.require(set(comparison)==set(declared),'Measured comparison schema differs')
        for key,value in comparison.items():
            if isinstance(value,dict):
                controller.require(set(value)==set(declared[key]),'Measured tensor comparison fields differ')
                for field,number in value.items():
                    controller.require(number==declared[key][field],'Recomputed raw tensor comparison differs: '+key+'/'+field)
            else:controller.require(value==declared[key],'Recomputed RNG/Adam/sampler digest comparison differs: '+key)
        actual[str(step)]=comparison;checks.append('Raw-byte capture independently recomputes every declared tensor/RNG field at'+str(step))
    del values
    control=report['branches']['control']['updates'];reclaim=report['branches']['reclaim']['updates']
    maximum=max(abs(a['component_losses'][key]-b['component_losses'][key])
       for a,b in zip(control,reclaim) for key in a['component_losses'])
    controller.require(maximum==report['max_component_absolute_delta'],'Actual maximum objective difference differs')
    limits={'gradient_relative_l2':.0005,'gradient_min_cosine':.9999995,'component_absolute_delta':.0001,'weights_relative_l2':.000001}
    strict=maximum<limits['component_absolute_delta'] and all(v['gradients']['relative_l2']<limits['gradient_relative_l2']
      and v['gradients']['cosine']>limits['gradient_min_cosine'] and v['weights']['relative_l2']<limits['weights_relative_l2'] for v in actual.values())
    manifest={'schema':1,'kind':'memory_guard_policy_launch_validation_v1','created_at_utc':datetime.now(timezone.utc).isoformat(),
      'probe':ref(args.probe),'probe_worker':ref(probe_worker.__file__),'production_wrapper':ref(OUTPUT/'launch_with_memory_guard_v3.py'),
      'builder_source_sha256':controller.sha(__file__),'actual_raw_tensor_capture':ref(capture),
      'independent_actual_tensor_recomputation':True,
      'strict_numerical_gate':{'thresholds':limits,'passed':strict,'waived':False,
        'scope':'Cache-policy shared-prefix global gradients/trainable weights/components; not fullBF16 loss equivalence'},
      'historical_full_BF16_original_vs_v2_equivalence_gate_passed':False,
      'new9GB4GB_production_execution_prelaunch_claim':False}
    # Validate the final manifest and replay all measured guard observations.
    # Write a unique temporary sibling, validate, then publish exact bytes once.
    temporary=args.output.with_name(args.output.stem+'_unpublished.json')
    controller.require(not temporary.exists(),'Unpublished proof path already exists')
    write_json(temporary,manifest)
    config=read_json(OUTPUT/'runs'/controller.SOURCE_RUN/'run_config.json',{})
    policy=controller.validate_memory_policy(temporary,config)
    checks.append('All100 intervention counters, per-update actual self PID/critical/sustained guard replay, source/runtime identities passed')
    verification={'schema':1,'passed':True,'kind':'actual_memory_policy_CPU_verification_v1','checked_at_utc':datetime.now(timezone.utc).isoformat(),
      'checks':checks,'CUDA_initialized':torch.cuda.is_initialized(),'probe':ref(args.probe),'capture':ref(capture),
      'builder_source_sha256':controller.sha(__file__),'controller_source_sha256':controller.sha(controller.__file__),
      'actual_comparisons':actual,'strict_cache_policy_numerical_gate_passed':strict,
      'historical_full_BF16_original_vs_v2_equivalence_gate_passed':False,'policy':policy}
    write_json(args.verification,verification)
    manifest['independent_verification']=ref(args.verification)
    write_json(args.output,manifest)
    print(json.dumps({'passed':True,'CUDA_initialized':torch.cuda.is_initialized(),'manifest':str(args.output),
      'manifest_sha256':controller.sha(args.output),'strict_cache_policy_numerical_gate_passed':strict,'intervention_updates':len(reclaim)}))

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe',type=Path,default=OUTPUT/'cuda_cache_reclaim_gpu_probe_v4.json')
    p.add_argument('--output',type=Path,default=OUTPUT/'memory_policy_launch_validation_v1.json')
    p.add_argument('--verification',type=Path,default=OUTPUT/'memory_policy_actual_cpu_verification_v1.json')
    return p

if __name__=='__main__':build(parser().parse_args())
