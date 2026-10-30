"""Small CPU guards for policy/source/reset semantics; no GPU execution."""
from pathlib import Path
import sys
import ast
import json
from datetime import datetime, timezone, timedelta
from unittest.mock import patch
import copy
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'outputs/parakeet_turkish'))
import profile_cuda_cache_reclaim_v4 as probe
import resource_watchdog_v1 as resource
torch.set_num_threads(2)
checks=[]
def check(name,value):
    if not value: raise AssertionError(name)
    checks.append(name)

stamp=datetime(2026,10,11,6,tzinfo=timezone.utc)
def memory(fraction,seconds,limit=40_000_000_000):
    used=int(limit*fraction)
    return dict(available=True,commit_fraction=used/limit,commit_used_bytes=used,
                commit_limit_bytes=limit,commit_available_bytes=limit-used,
                observed_at_utc=(stamp+timedelta(seconds=seconds)).isoformat())
def observe(guard,fraction,seconds,limit=40_000_000_000,reserved=0):
    with patch.object(resource,'_timestamp',return_value=(stamp+timedelta(seconds=seconds)).isoformat()):
        return probe.safety_reason(memory(fraction,seconds,limit),reserved,resource_guard=guard,observed_monotonic=seconds)
check('critical95-immediate',observe(resource.ResourceGuard(),.95,0)=='system_commit_at_or_above95percent')
check('headroom-below2GB-immediate',observe(resource.ResourceGuard(),.85,0,limit=10_000_000_000)=='system_commit_headroom_below2GB')
guard=resource.ResourceGuard()
check('moderate-first-not-stop',observe(guard,.91,0) is None)
check('moderate-too-early-not-stop',observe(guard,.91,4) is None and guard.moderate_streak==1)
check('moderate-second-not-stop',observe(guard,.91,5) is None)
check('moderate-third-stop',observe(guard,.91,10)=='system_commit_above90percent_three_fresh_observations')
guard=resource.ResourceGuard();observe(guard,.91,0)
check('normal-resets-streak',observe(guard,.9,5) is None and guard.moderate_streak==0)
check('reserved12-independent-stop',observe(resource.ResourceGuard(),.5,0,reserved=12_000_000_000)=='allocator_reservation_guard')
actual=probe.memory_sample('CPU_guard_self_identity')
check('actual-self-private-telemetry',actual['worker_private_memory']['available'] and actual['worker_private_memory']['pid']==__import__('os').getpid())
check('actual-system-alias',actual['system_memory']['committed_bytes']==actual['system_memory']['commit_used_bytes'])
check('CPU-no-CUDA-initialization',not torch.cuda.is_initialized())
args=probe.parser().parse_args([])
check('default-source1821-run',args.run_config.parent.name=='adaptive_20261010_continuation_02_resume_B8')
check('default-bounded10-vs100',args.control_updates==10 and args.intervention_updates==100 and args.max_wall_seconds==600.)

# Reproduce the optimizer alias hazard and verify deep-copy reset avoids it.
p=torch.nn.Parameter(torch.tensor([1.]));optimizer=torch.optim.AdamW([p],lr=.01)
p.grad=torch.ones_like(p);optimizer.step();baseline=copy.deepcopy(optimizer.state_dict())
digest=probe.tree_sha(baseline)
optimizer.load_state_dict(copy.deepcopy(baseline));p.grad=torch.ones_like(p);optimizer.step()
check('deep-copy-baseline-optimizer-immutable',probe.tree_sha(baseline)==digest)
optimizer.load_state_dict(copy.deepcopy(baseline))
check('deep-copy-reset-exact',probe.tree_sha(optimizer.state_dict())==digest)

def calls(path,target):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    return [ast.dump(node,include_attributes=False) for node in ast.walk(tree)
            if isinstance(node,ast.Call) and ast.unparse(node.func)==target]
old=ROOT/'outputs/parakeet_turkish/profile_cuda_cache_reclaim_v3.py'
new=Path(probe.__file__)
for target in ('batch_objective','scaled.backward','optimizer.step','torch.nn.utils.clip_grad_norm_'):
    check('unchanged-training-call-'+target,calls(old,target)==calls(new,target))
source=new.read_text(encoding='utf-8')
check('explicit-control13-cache6',"branch_cap_gb=13. if policy=='control' else 9." in source and '6_000_000_000 if policy' in source)
check('explicit-intervention9-cache4','else 4_000_000_000' in source)
check('no-BF16-quality-waiver',"'strict_BF16_original_vs_v2_gate':'Previously failed; no threshold change or quality waiver.'" in source)
target=ROOT/'outputs/parakeet_turkish/cache_probe_v4_cpu_verification.json'
if target.exists():raise FileExistsError(target)
target.write_text(json.dumps(dict(schema=1,passed=True,CPU_only=True,check_count=len(checks),checks=checks,
    probe_sha256=probe.sha(new),GPU_policy_verification_pending=True),indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(passed=True,checks=len(checks),probe_sha256=probe.sha(new))))
