"""Compare two actually CLOSED matched native controls; CPU tensors only."""
from pathlib import Path
from datetime import datetime, timezone
import ctypes, hashlib, json, math, os, statistics, time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/parakeet_turkish'
RUNS=['adaptive_20261012_TR_uniform_IO_double_01','adaptive_20261012_TR_uniform_IO_single_01']
EXIT=[{'session_id':28770,'exit_code':0,'tool_chunk':'088906'},
      {'session_id':18486,'exit_code':0,'tool_chunk':'2271b0'}]

def read(p): return json.loads(p.read_text(encoding='utf-8'))

def record(p):
    with p.open('rb') as stream: h=hashlib.file_digest(stream,'sha256').hexdigest()
    return {'path':str(p),'sha256':h,'bytes':p.stat().st_size}

status=[]
for run in RUNS:
    loop=OUT/'runs'/(run+'_loop_01');s=read(loop/'loop_status.json')
    assert s['status']=='train_window_ended' and s['child_exit_code']==0 and s['completed_steps']==25209
    assert not s['resource_review_required']
    done=read(Path(s['paths']['native']).with_suffix('.completion.json'))
    assert done['status']=='complete' and done['sources_unchanged'] and done['error'] is None
    status.append(s)
# The known actual CLOSED boundaries above precede any tensor import/read.
ctypes.windll.kernel32.GetCurrentProcess.restype=ctypes.c_void_p
ctypes.windll.kernel32.SetPriorityClass.argtypes=(ctypes.c_void_p,ctypes.c_ulong)
ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(),0x4000)
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
torch.set_num_threads(1)
assert not torch.cuda.is_initialized()
started=time.perf_counter()
states=[torch.load(ROOT/'work'/(run+'_latest.pt'),map_location='cpu',weights_only=True) for run in RUNS]
a,b=states
assert all(s['schema']==2 and s['step']==25209 and len(s['history'])==25209 for s in states)
assert len(a['weights'])==len(b['weights'])==207 and a['weights'].keys()==b['weights'].keys()
weight_diffs=[]
for name,x in a['weights'].items():
    y=b['weights'][name]
    assert x.device.type==y.device.type=='cpu' and x.shape==y.shape and x.dtype==y.dtype==torch.float32
    if not torch.equal(x,y): weight_diffs.append({'name':name,'max_abs':float((x-y).abs().max())})
assert a['optimizer']['param_groups']==b['optimizer']['param_groups']
assert a['optimizer']['state'].keys()==b['optimizer']['state'].keys() and len(a['optimizer']['state'])==207
moment_diffs=[];moments=0
for ident,x in a['optimizer']['state'].items():
    y=b['optimizer']['state'][ident];assert x.keys()==y.keys()
    for name,value in x.items():
        other=y[name]
        if isinstance(value,torch.Tensor):
            moments+=1
            assert value.device.type==other.device.type=='cpu' and value.shape==other.shape and value.dtype==other.dtype
            if not torch.equal(value,other):moment_diffs.append({'parameter_id':ident,'key':name,'max_abs':float((value-other).abs().max())})
        else: assert value==other
assert a['python_rng']==b['python_rng'] and torch.equal(a['torch_rng'],b['torch_rng'])
assert len(a['cuda_rng'])==len(b['cuda_rng']) and all(torch.equal(x,y) for x,y in zip(a['cuda_rng'],b['cuda_rng']))
suffix=[s['history'][25109:] for s in states]
ids_exact=all(x['sampled_source_ids']==y['sampled_source_ids'] for x,y in zip(*suffix))
loss_keys=['loss','tdt_loss','replay_kl','replay_duration_kl','replay_feature_distillation','gradient_norm','head_lr','lora_lr']
loss_diffs=[{'step':x['step'],'key':key,'A':x[key],'B':y[key],'abs_difference':abs(x[key]-y[key])}
            for x,y in zip(*suffix) for key in loss_keys if x[key]!=y[key]]
assert ids_exact
assert a['history'][:25109]==b['history'][:25109]
def normalize_records(value):
    if isinstance(value,dict):
        return {key:normalize_records(v) for key,v in value.items()
                if not (key=='path' and 'sha256' in value)}
    if isinstance(value,list): return [normalize_records(v) for v in value]
    if isinstance(value,tuple): return tuple(normalize_records(v) for v in value)
    return value
sampler_exact=normalize_records(a['sampler'])==normalize_records(b['sampler'])
paths_exact=0;prediction_diffs=[];scores=[]
token_keys=['token_id','duration_id','piece','start_frame','duration_frames']
identity_keys=['source_id','sentence_id','language','split','reference','dataset_revision','audio_sha256','wav_sha256','duration','hypothesis']
for language in ['tr','en','es','it','fr','de','pt']:
    paths=[OUT/'results'/(run+'_step_25209')/('validation_'+language+'.jsonl') for run in RUNS]
    with paths[0].open(encoding='utf-8') as left,paths[1].open(encoding='utf-8') as right:
        n=0
        for first,second in zip(left,right,strict=True):
            x,y=json.loads(first),json.loads(second);n+=1
            identical=all(x[key]==y[key] for key in identity_keys)
            identical &= [tuple(t[key] for key in token_keys) for t in x['confidence']['tokens']]==[tuple(t[key] for key in token_keys) for t in y['confidence']['tokens']]
            identical &= x['confidence']['blank_steps']==y['confidence']['blank_steps'] and x['confidence']['total_steps']==y['confidence']['total_steps']
            if identical: paths_exact+=1
            else: prediction_diffs.append({'language':language,'source_id':x['source_id']})
        assert n==50
measure=[]
for run,s,state in zip(RUNS,status,states):
    timing=read(OUT/'runs'/run/'persistence_timing.json')
    events=timing['events'];periodic=[e['pair_total_observed_seconds'] for e in events if e['label']=='periodic']
    first,last=state['history'][25108],state['history'][-1]
    train=last['train_wall_seconds']-first['train_wall_seconds']
    evaluation=last['evaluation_wall_seconds']-first['evaluation_wall_seconds']
    wall=(datetime.fromisoformat(s['finished_at_utc'])-datetime.fromisoformat(s['started_at_utc'])).total_seconds()
    score=read(OUT/'results'/(run+'_step_25209')/'summary.json')
    scores.append({key:value['asr'] for key,value in score.items() if key.startswith('validation_')})
    measure.append({'run_id':run,'mode':timing['mode'],'actual_exit':EXIT[len(measure)],
        'updates':100,'train_seconds':train,'additional_DEV_seconds':evaluation,'full_native_window_wall_seconds':wall,
        'pair_calls':len(events),'periodic_pair_calls':len(periodic),'all_pair_seconds':sum(e['pair_total_observed_seconds'] for e in events),
        'periodic_pair_seconds':sum(periodic),'mean_periodic_pair_seconds':statistics.mean(periodic),
        'publisher_counters':timing['publisher_counters'],'timing':record(OUT/'runs'/run/'persistence_timing.json'),
        'private_peak_bytes':s['peak_observed_owned_private_bytes'],
        'latest_checkpoint':record(ROOT/'work'/(run+'_latest.pt'))})
scores_exact=scores[0]==scores[1]
assert not torch.cuda.is_initialized()
passed=not weight_diffs and not moment_diffs and not loss_diffs and sampler_exact and paths_exact==350
report={'kind':'matched_uniform_IO_actual_native_100update_comparison_v1','passed':passed,
    'created_at_utc':datetime.now(timezone.utc).isoformat(),'source_step':25109,'matched_updates':100,
    'all207_weights_bitwise_equal':not weight_diffs,'Adam_tensor_states_checked':moments,
    'all207_Adam_moments_and_clocks_bitwise_equal':not moment_diffs,'optimizer_groups_exact':True,
    'Python_CPU_allCUDA_RNG_states_exact':True,'source25109_history_prefix_unchanged_exact':True,
    'all100_sampled_source_ID_sequences_exact':ids_exact,'all_logged_losses_gradient_norm_LRs_exact':not loss_diffs,
    'sampler_exact_after_only_file_record_path_normalization':sampler_exact,
    'all350_final_raw_transcript_reference_token_duration_blank_paths_exact':paths_exact==350,
    'all_seven_final_ASR_counts_and_scores_exact':scores_exact,'actual_ASR_scores':scores,
    'weight_differences':weight_diffs,'moment_differences':moment_diffs,
    'loss_differences':loss_diffs,'prediction_differences':prediction_diffs,'measurements':measure,
    'periodic_pair_time_reduction_fraction':1-measure[1]['mean_periodic_pair_seconds']/measure[0]['mean_periodic_pair_seconds'],
    'full_native_window_wall_reduction_fraction':1-measure[1]['full_native_window_wall_seconds']/measure[0]['full_native_window_wall_seconds'],
    'actual_GPU_runs_completed':True,'comparison_used_CPU_only':True,'CUDA_initialized_during_comparison':False,
    'CPU_comparison_seconds':time.perf_counter()-started,
    'scope':'One sequential matched native A/B, isolated same-volume two-file publications without production mirror-reader contention. Pair timers attribute this persistence operation; total wall includes source350/setup/DEV/save and can vary with caches/order.',
    'quality_recipe_improvement_claimed':False,'TEST_used':False,'goal_complete':False}
target=OUT/'matched_uniform_IO_actual_native_comparison_v1.json'
with target.open('x',encoding='utf-8') as handle:json.dump(report,handle,indent=2)
print(json.dumps({'passed':passed,'report':record(target),'measurements':measure,
    'weight_diffs':len(weight_diffs),'moment_diffs':len(moment_diffs),'loss_diffs':len(loss_diffs),'sampler_exact':sampler_exact,
    'predictions_exact':paths_exact,'CUDA_initialized':False}))
