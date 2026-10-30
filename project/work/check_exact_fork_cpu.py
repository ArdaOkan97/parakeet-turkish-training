"""CPU guard cases for the new versioned checkpoint fork, without running it."""
import copy
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/parakeet_turkish'
sys.path.insert(0,str(OUT))
import torch
import fork_training_state_v1 as module

torch.set_num_threads(2)
source_id='adaptive_20261010_recovery_b8_01_b8_quality'
target_id='adaptive_20261010_continuation_01_resume_B8'
cases=[]
assertions=0

def require(value,message):
    global assertions
    assertions+=1
    if not value:raise AssertionError(message)

protected=[OUT/'train_lora_batched.py',OUT/'atomic_io_retry_v1.py',OUT/'launch_with_io_retry_v1.py',
           ROOT/'work'/(source_id+'_best.pt'),ROOT/'work'/(source_id+'_latest.pt'),
           OUT/'runs'/source_id/'run_config.json',OUT/'runs'/source_id/'training_history.json',
           OUT/'b8_memory_slowdown_diagnosis.json',OUT/'b8_cooperative_stop_verification.json']
before={str(p):module.digest(p) for p in protected}
with (ROOT/'work'/(source_id+'_latest.pt')).open('rb') as f:
    state=torch.load(f,map_location='cpu',weights_only=True)
require(state['step']==907 and state['selected_step']==801,'actual source checkpoint identity')
require(len(state['weights'])==len(state['optimizer']['state'])==207,'actual full optimizer/weights')
original_id=state['run_id']
fork=dict(state);fork['run_id']=target_id
check=module.Checks();module.validate_fork(state,fork,source_id,target_id,check)
cases.append({'name':'actual907_inmemory_exact_fork','passed':True,'comparisons':check.count,'tensor_leaves':check.tensor_count})
with tempfile.TemporaryDirectory(prefix='exact-fork-cpu-',dir=ROOT/'work') as folder:
    path=Path(folder)/'roundtrip.pt'
    module.commit_writer(path,lambda p:torch.save(fork,p))
    with path.open('rb') as f:reloaded=torch.load(f,map_location='cpu',weights_only=True)
    check=module.Checks();module.validate_fork(state,reloaded,source_id,target_id,check)
    require(reloaded['optimizer']['param_groups']==state['optimizer']['param_groups'],'Adam groups exact after serialization')
    require(reloaded['config_identity']==state['config_identity'],'full configuration exact after serialization')
    cases.append({'name':'actual907_native_atomic_torch_roundtrip','passed':True,'comparisons':check.count,'tensor_leaves':check.tensor_count})
del reloaded
require(state['run_id']==original_id,'shallow fork changed source run ID')

# Small complete-state fixture retains real schema/container types. Mutations
# target each critical semantic branch rather than repeating the implementation.
fixture={k:v for k,v in state.items()}
fixture['weights']={'w':torch.tensor([[1.,2.],[3.,4.]],dtype=torch.float32)}
fixture['optimizer']={'state':{0:{'step':torch.tensor(907.),'exp_avg':torch.ones(2,2),
                                'exp_avg_sq':torch.ones(2,2)*2}},
                      'param_groups':[{'params':[0],'lr':.0003,'betas':(.9,.999)}]}
fixture['history']=[{'step':1,'validation':{'turkish_wer':.69},'sampled_source_ids':['tr:one','en:two']}]
fixture['sampler']={'rng':state['sampler']['rng'],'batch_size':8,'accumulation':2,
                    'tr':{'draws':7256,'cursor':22},'replay':{'en':{'draws':42}}}
fixture['config_identity']={'base_revision':'pinned','training_arguments':{'head_lr':5e-5,'replay_kl':5.,'warmup_steps':25}}
fixture['cuda_rng']=[torch.arange(10,dtype=torch.uint8)]
fixture['torch_rng']=torch.arange(10,dtype=torch.uint8)
good=copy.deepcopy(fixture);good['run_id']=target_id
check=module.Checks();module.validate_fork(fixture,good,source_id,target_id,check)
cases.append({'name':'small_complete_fixture_pass','passed':True,'comparisons':check.count})
mutations={
 'wrong_target_id':lambda x:x.update(run_id='wrong'),
 'source_id_unchanged':lambda x:x.update(run_id=source_id),
 'optimizer_step':lambda x:x['optimizer']['state'][0]['step'].add_(1),
 'optimizer_moment':lambda x:x['optimizer']['state'][0]['exp_avg'].add_(.001),
 'optimizer_second_moment':lambda x:x['optimizer']['state'][0]['exp_avg_sq'].zero_(),
 'optimizer_lr':lambda x:x['optimizer']['param_groups'][0].update(lr=1e-4),
 'optimizer_param_ids':lambda x:x['optimizer']['param_groups'][0].update(params=[1]),
 'weight_value':lambda x:x['weights']['w'].add_(.001),
 'weight_dtype':lambda x:x['weights'].update(w=x['weights']['w'].to(torch.bfloat16)),
 'weight_shape':lambda x:x['weights'].update(w=x['weights']['w'].reshape(4)),
 'torch_rng':lambda x:x['torch_rng'].fill_(3),
 'cuda_rng':lambda x:x['cuda_rng'][0].fill_(5),
 'python_rng_container':lambda x:x.update(python_rng=list(x['python_rng'])),
 'sampler_cursor':lambda x:x['sampler']['tr'].update(cursor=23),
 'sampler_draws':lambda x:x['sampler']['tr'].update(draws=7257),
 'sampler_batch':lambda x:x['sampler'].update(batch_size=1),
 'sampler_rng':lambda x:x['sampler'].update(rng=(3,tuple([0]*625),None)),
 'history_step':lambda x:x['history'][0].update(step=2),
 'history_validation':lambda x:x['history'][0]['validation'].update(turkish_wer=.68),
 'history_source_ids':lambda x:x['history'][0]['sampled_source_ids'].reverse(),
 'config_lr':lambda x:x['config_identity']['training_arguments'].update(head_lr=1e-4),
 'config_kl':lambda x:x['config_identity']['training_arguments'].update(replay_kl=.5),
 'config_warmup':lambda x:x['config_identity']['training_arguments'].update(warmup_steps=100),
 'selected_step':lambda x:x.update(selected_step=907),
 'best_score':lambda x:x.update(best_score=.70),
 'next_dev_epoch':lambda x:x.update(next_eval_epoch=1.),
 'audio_counter':lambda x:x.update(audio_seconds=0.),
 'wall_counter':lambda x:x.update(train_wall_seconds=0.),
 'stop_reason':lambda x:x.update(stop_reason=None),
 'extra_top_level_key':lambda x:x.update(new_metadata=True),
 'missing_top_level_key':lambda x:x.pop('sampler'),
}
for name,mutate in mutations.items():
    bad=copy.deepcopy(good);mutate(bad)
    try:module.validate_fork(fixture,bad,source_id,target_id,module.Checks())
    except ValueError:rejected=True
    else:rejected=False
    require(rejected,'corruption accepted: '+name)
    cases.append({'name':name,'corruption_rejected':True})
for value in ['../evil','a/b','a\\b','a.b','a:bad','']:
    try:module.checked_id(value)
    except Exception:rejected=True
    else:rejected=False
    require(rejected,'unsafe run ID accepted')
    cases.append({'name':'unsafe_id_'+repr(value),'corruption_rejected':True})
require(module.inside(ROOT/'outputs/parakeet_turkish/runs/new/snapshot',OUT/'runs'),'valid snapshot containment')
require(not module.inside(ROOT/'work',OUT/'runs'),'invalid snapshot containment')
require(not (ROOT/'work'/(target_id+'_latest.pt')).exists(),'CPU guards created live target checkpoint')
require(torch.cuda.is_initialized() is False,'CPU guards initialized CUDA')
for path,value in before.items():require(module.digest(Path(path))==value,'source artifact changed')
report={'schema':1,'passed':True,'status':'complete','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'helper_sha256':module.digest(OUT/'fork_training_state_v1.py'),
        'case_count':len(cases),'corruptions_rejected':len(mutations)+6,
        'positive_assertions':assertions,'cases':cases,'actual_source_step':907,
        'actual_inherited_selected_step':801,'protected_source_sha256':before,
        'cuda_initialized':False,'actual_fork_cli_executed':False,
        'scope':'Actual complete907 CPU checkpoint atomic roundtrip + critical semantic corruption guards; no training/optimizer reset or published artifact mutation.'}
(OUT/'exact_state_fork_cpu_verification.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({k:report[k] for k in ['passed','helper_sha256','case_count','corruptions_rejected','positive_assertions','cuda_initialized']},indent=2))
