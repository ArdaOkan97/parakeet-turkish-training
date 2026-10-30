"""Bind independently recomputed GPU A/B and real frozen-worker cache proofs."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import gzip
import json
from pathlib import Path
import torch
from atomic_io_retry_v1 import write_json
from verify_final_results import file_sha,json_load
from profile_cuda_cache_reclaim_v3 import compare_capture,summarize_snapshot

OUTPUT=Path(__file__).resolve().parent
ROOT=OUTPUT.parents[1]

def bound(path):
    path=Path(path).resolve()
    if not path.is_relative_to(ROOT.resolve()) or not path.is_file():raise ValueError('Owned evidence file required')
    return {'path':str(path),'sha256':file_sha(path)}

def main(args):
    torch.set_num_threads(2)
    if not args.output.resolve().is_relative_to(ROOT.resolve()):raise ValueError('Owned output required')
    if args.output.exists():raise FileExistsError('Preserve existing validation evidence')
    probe=json_load(args.probe)
    assert probe['passed'] and probe['status']=='complete' and probe['matched_sources_equal']
    assert probe['identity']['probe_sha256']==probe['probe_sha256_after']==file_sha(OUTPUT/'profile_cuda_cache_reclaim_v3.py')
    assert file_sha(probe['full_capture_path'])==probe['full_capture_sha256']
    captures=torch.load(probe['full_capture_path'],map_location='cpu',weights_only=True)
    for step,actual in probe['comparisons'].items():
        rebuilt=compare_capture(captures[('control',int(step))],captures[('reclaim',int(step))])
        assert rebuilt==actual,'Actual captured tensors must match reported comparisons'
    for branch in probe['branches'].values():
        for snapshot in branch['snapshots']:
            assert file_sha(snapshot['path'])==snapshot['sha256']
            with gzip.open(snapshot['path'],'rt',encoding='utf-8') as stream:raw=json.load(stream)
            assert summarize_snapshot(raw)==snapshot['summary'],'Actual allocation snapshot counters'
    thresholds={'gradient_relative_l2':.0005,'gradient_min_cosine':.9999995,
                'component_absolute_delta':.0001,'weights_relative_l2':.000001}
    maximum=max(abs(a['component_losses'][k]-b['component_losses'][k])
                for a,b in zip(probe['branches']['control']['updates'],probe['branches']['reclaim']['updates'])
                for k in a['component_losses'])
    assert maximum==probe['max_component_absolute_delta']
    passed=maximum<thresholds['component_absolute_delta'] and all(
        c['gradients']['relative_l2']<thresholds['gradient_relative_l2'] and
        c['gradients']['cosine']>thresholds['gradient_min_cosine'] and
        c['weights']['relative_l2']<thresholds['weights_relative_l2']
        for c in probe['comparisons'].values())
    smoke=Path(args.smoke_dir).resolve();history=json_load(smoke/'training_history.json')
    from fork_training_state_v1 import Checks,validate_fork
    fork_receipt=smoke/'fork_receipt_v2.json';fork=json_load(fork_receipt)
    fork_check=Checks()
    for record in fork['forks']:
        assert file_sha(record['source_path'])==record['source_sha256']
        assert file_sha(record['fork_path'])==record['fork_sha256']
        source=torch.load(record['source_path'],map_location='cpu',weights_only=True)
        target=torch.load(record['fork_path'],map_location='cpu',weights_only=True)
        validate_fork(source,target,fork['source_run_id'],fork['smoke_run_id'],fork_check)
        del source,target
    production={'initial_step':907,'completed_steps':len(history)}
    for key,path in [('cache_provenance',args.cache_provenance),('io_provenance',args.io_provenance),
                     ('memory_provenance',args.memory_provenance),('training_history',smoke/'training_history.json'),
                     ('run_config',smoke/'run_config.json')]:production[key]=bound(path)
    report={'schema':1,'kind':'cuda_cache_reclaim_launch_validation_v1','created_at_utc':datetime.now(timezone.utc).isoformat(),
            'probe':bound(args.probe),'probe_worker':bound(OUTPUT/'profile_cuda_cache_reclaim_v3.py'),
            'production_hook':production,'strict_numerical_gate':{'waived':False,'passed':passed,'thresholds':thresholds,
            'scope':'Cache A/B matched global clipped gradients, weights and component losses only; encoder features are unmeasured.',
            'historical_original_vs_v2_full_BF16_gate_passed':False},
            'smoke_fork_receipt':bound(fork_receipt),'smoke_fork_exact_CPU_checks':fork_check.count,
            'independent_actual_tensor_and_snapshot_recomputation':True,'builder_source_sha256':file_sha(__file__)}
    from run_experiment_loop_continuation_v1 import validate_cache_gpu
    cfg=json_load(OUTPUT/'runs'/probe['identity']['source_run_id']/'run_config.json')
    # Validate a new scratch manifest, then promote the reviewed immutable bytes.
    scratch=args.output.with_suffix('.pending.json')
    if scratch.exists():raise FileExistsError('New scratch evidence required')
    write_json(scratch,report)
    proof=validate_cache_gpu(scratch,cfg)
    scratch.replace(args.output)
    print(json.dumps({'passed':True,'CUDA_initialized':torch.cuda.is_initialized(),
                     'cache_AB_numeric_gate_passed':passed,'real_hook_completed_steps':len(history),
                     'manifest':str(args.output),'sha256':file_sha(args.output)}))

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('probe','smoke-dir','cache-provenance','io-provenance','memory-provenance','output'):
        p.add_argument('--'+name,type=Path,required=True)
    return p
if __name__=='__main__':main(parser().parse_args())
