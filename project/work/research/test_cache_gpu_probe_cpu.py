"""Small CPU guards for the bounded real-GPU probe; never initializes CUDA."""
import ast
import json
from pathlib import Path
import sys

import torch
ROOT=Path(__file__).resolve().parents[2];OUTPUT=ROOT/'outputs/parakeet_turkish'
sys.path.insert(0,str(OUTPUT))
import profile_cuda_cache_reclaim_v1 as probe
from atomic_io_retry_v1 import write_json


def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    checks=[]
    def require(value,name):assert value,name;checks.append(name)
    args=probe.parser().parse_args([])
    require((args.control_updates,args.intervention_updates,args.max_wall_seconds)==(64,100,600.),'Bounded diverse B8 defaults')
    require(not args.solo_gpu and not args.prepare,'GPU requires explicit ownership declaration')
    for commit,reserved,expected in [(.89,0,None),(.90,0,'system_commit_guard'),(.89,12_000_000_000,'allocator_reservation_guard'),
                                    (.89,11_999_999_999,None),(.95,15_000_000_000,'system_commit_guard')]:
        require(probe.safety_reason({'available':True,'commit_fraction':commit},reserved)==expected,'Exact safety boundary '+str((commit,reserved)))
    fixture={'segments':[{'total_size':100,'active_size':0,'blocks':[{'size':100,'state':'inactive','frames':[]}]},
                         {'total_size':80,'active_size':50,'blocks':[{'size':50,'state':'active_allocated','frames':[]},
                          {'size':30,'state':'inactive','frames':[{'filename':str(OUTPUT/'efficient_tdt_v2.py'),'line':200,'name':'tdt'}]}]}],
             'device_traces':[[{'action':'alloc'},{'action':'free_completed'}]]}
    summary=probe.summarize_snapshot(fixture)
    require(summary['totals']['fully_inactive_segment_bytes']==100,'Fully inactive whole segments separately totaled')
    require(summary['totals']['inactive_in_live_segment_bytes']==30,'Inactive split/live-segment bytes separately totaled')
    require(summary['totals']['reserved_bytes']==180 and summary['totals']['active_allocated_bytes']==50,'Snapshot reserved/active accounting')
    require(summary['largest_inactive_block_sites'][0]['bytes']==100,'Allocation sites ranked with unknown-frame fallback')
    require(probe.tree_sha({'step':torch.tensor(907.),'state':[torch.ones(2)]})==probe.tree_sha({'step':torch.tensor(907.),'state':[torch.ones(2)]}),
            'Full Adam scalar/tensor hashing supports0D optimizer step')
    left={'gradients':{'a':torch.tensor([1.,2.])},'weights':{'a':torch.tensor([3.,4.])},
          'sampler_sha256':'same','rng_sha256':'same','optimizer_sha256':'same'}
    right={**left,'gradients':{'a':torch.tensor([1.,2.])},'weights':{'a':torch.tensor([3.,4.])}}
    equal=probe.compare_capture(left,right)
    require(equal['all_bitwise_equal'] and equal['gradients']['relative_l2']==0,'Exact matched cache-policy capture equality')
    right['gradients']={'a':torch.tensor([1.1,2.])};different=probe.compare_capture(left,right)
    require(not different['all_bitwise_equal'] and different['gradients']['max_absolute_delta']>0,'Real numerical deltas reported rather than waived')
    right['rng_sha256']='changed';require(not probe.compare_capture(left,right)['python_cpu_cuda_rng_equal'],'Changed RNG detected')
    require(equal['quality_or_numerical_equivalence_waiver'] is False,'Probe never waives strict BF16 or DEV quality requirements')
    source=Path(probe.__file__).read_text();tree=ast.parse(source)
    require('max_entries=5000' in source and "stacks='python'" in source,'Allocation traces bounded5000 with Python callsites')
    require("allow_tf32=True" in source and "allow_tf32=False" in source,'Actual frozen B8 cudnnTF32True/matmulFalse recorded')
    require("13e9/capacity" in source and "reserved_limit=12_000_000_000" in source,'Identical13GB per-process cap and conservative12GB boundary')
    require("real_hook_target_passed=False" in source and "production_hook_validation_pending=True" in source,'Harness cannot claim frozen production-hook execution')
    require(not any(isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='evaluate' for node in ast.walk(tree)),
            'No DEV/TEST inference/selection call')
    plan=json.loads((OUTPUT/'cuda_cache_gpu_probe_v1_cpu_plan.json').read_text())
    require(plan['identity']['all_rows_train_and_heldout_identity_isolated'],'Actual exact accepted TRAIN pool isolated from heldout identities')
    require(plan['identity']['diverse_shape_count']==100 and len(plan['identity']['plans'])==100,'Actual sampler plan has100distinct shape pairs')
    require(plan['identity']['source_step']==907 and plan['identity']['optimizer_states']==207,'Actual source full trained907 checkpoint and207 Adam states')
    require(plan['identity']['selected_unique_clips']>1000,'Actual plan uses many unique clips rather than16cyclingrows')
    require(not torch.cuda.is_initialized() and plan['identity']['CUDA_initialized_during_CPU_preparation'] is False,'Preparation/guards never initialized CUDA')
    report={'passed':True,'check_count':len(checks),'checks':checks,'probe_sha256':probe.sha(probe.__file__),
            'CPU_plan_sha256':probe.sha(OUTPUT/'cuda_cache_gpu_probe_v1_cpu_plan.json'),
            'scope':'CPU mocked boundary/snapshot/numeric/hash checks plus actual TRAIN-only preparation; no GPU execution'}
    write_json(OUTPUT/'cuda_cache_gpu_probe_v1_cpu_verification.json',report)
    print(json.dumps({'passed':True,'check_count':len(checks),'probe_sha256':report['probe_sha256']}))


if __name__=='__main__':main()
