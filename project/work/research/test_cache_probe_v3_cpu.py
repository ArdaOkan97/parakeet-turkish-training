"""Actual CPU AdamW alias regression and immutable baseline fix proof."""
import copy
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[2];OUTPUT=ROOT/'outputs/parakeet_turkish'
sys.path.insert(0,str(OUTPUT))
import profile_cuda_cache_reclaim_v3 as probe
from atomic_io_retry_v1 import write_json


def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    checks=[]
    def require(value,name):assert value,name;checks.append(name)
    protected=[OUTPUT/'profile_cuda_cache_reclaim_v1.py',OUTPUT/'cuda_cache_reclaim_gpu_probe_v1.json',
               OUTPUT/'launch_with_cache_reclaim_v1.py',OUTPUT/'cuda_cache_reclaim_v1.py',OUTPUT/'train_lora_batched.py']
    before={str(path):probe.sha(path) for path in protected}
    parameter=torch.nn.Parameter(torch.tensor([1.,2.]));source=torch.optim.AdamW([parameter],lr=.001)
    parameter.grad=torch.ones(2);source.step()
    baseline=copy.deepcopy(source.state_dict());baseline_hash=probe.tree_sha(baseline)
    other=torch.nn.Parameter(parameter.detach().clone());aliased=torch.optim.AdamW([other],lr=.001)
    aliased.load_state_dict(baseline)
    require(aliased.state[other]['step'].data_ptr()==baseline['state'][0]['step'].data_ptr(),'CPU Adam step load aliases source tensor without deepcopy')
    other.grad=torch.ones(2);aliased.step()
    require(probe.tree_sha(baseline)!=baseline_hash,'Update reproduces mutable in-memory baseline corruption')
    immutable=copy.deepcopy(source.state_dict());immutable_hash=probe.tree_sha(immutable)
    for branch in range(2):
        candidate=torch.nn.Parameter(parameter.detach().clone());optimizer=torch.optim.AdamW([candidate],lr=.001)
        optimizer.load_state_dict(copy.deepcopy(immutable))
        require(probe.tree_sha(optimizer.state_dict())==immutable_hash,'Exact optimizer reset branch'+str(branch))
        require(optimizer.state[candidate]['step'].data_ptr()!=immutable['state'][0]['step'].data_ptr(),'Branch scalar step source detached'+str(branch))
        candidate.grad=torch.ones(2);optimizer.step()
        require(probe.tree_sha(immutable)==immutable_hash,'Baseline remains unchanged after update branch'+str(branch))
    text=Path(probe.__file__).read_text()
    require('optimizer.load_state_dict(copy.deepcopy(saved[\'optimizer\']))' in text,'Actual v3 uses deep copy at every optimizer restore')
    require('>6_000_000_000' in text and "'cleanup_threshold_gb':6." in text,'Exact6GB policy and report recorded')
    require('Immutable baseline optimizer was mutated' in text,'Actual v3 rechecks immutable baseline digest after branches')
    require(probe.parser().parse_args([]).output.name=='cuda_cache_reclaim_gpu_probe_v3.json','New v2 output preserves failed v1 evidence')
    for path,digest in before.items():require(probe.sha(path)==digest,'Existing evidence/source unchanged '+Path(path).name)
    require(not torch.cuda.is_initialized(),'All actual alias guards CPU-only without CUDA initialization')
    report={'passed':True,'checks':checks,'check_count':len(checks),'probe_sha256':probe.sha(probe.__file__),
            'protected_before_after_sha256':before,'scope':'Actual tiny CPU Adam alias reproduction/deep-copy regression; no GPU'}
    write_json(OUTPUT/'cuda_cache_gpu_probe_v3_cpu_verification.json',report)
    print({'passed':True,'checks':len(checks),'probe_sha256':report['probe_sha256']})


if __name__=='__main__':main()
