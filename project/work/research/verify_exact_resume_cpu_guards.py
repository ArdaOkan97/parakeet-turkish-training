"""Tiny CPU-only corruption fixtures for exact-resume checkpoint equality.

No real checkpoint, GPU, process, training artifact or heldout prediction is
loaded. The test model consists solely of small scalar/vector tensors.
"""
import copy
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path.cwd()
OUTPUT = ROOT/'outputs/parakeet_turkish'
sys.path.insert(0,str(OUTPUT))
import verify_exact_resume as verify
import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(1)

paused = {'schema':2,'run_id':'toy_screen_A','step':1159,
    'weights':{'small_weight':torch.arange(9,dtype=torch.float32).reshape(3,3)},
    'optimizer':{'state':{index:{'step':torch.tensor(1159.,dtype=torch.float32),
        'exp_avg':torch.tensor([index*.001],dtype=torch.float32),
        'exp_avg_sq':torch.tensor([index*.0001],dtype=torch.float32)} for index in range(207)},
        'param_groups':[{'lr':2e-5,'params':list(range(207))},{'lr':3e-4,'params':[]}]},
    'sampler':{'tr':{'order':['clip_a','clip_b'],'position':1,'draws':2318},
        'replay':{'en':{'position':1,'draws':2318}},'rng':random.Random(9).getstate()},
    'python_rng':random.Random(10).getstate(),'torch_rng':torch.arange(16,dtype=torch.uint8),
    'cuda_rng':[torch.arange(12,dtype=torch.uint8)],'config_identity':{'batch_size':1,'accumulation':4,'code_sha256':'abc'},
    'history':[{'step':1159,'loss':1.25,'sampled_source_ids':['clip_a','clip_en'],'development_selection_only':True}],
    'best_score':.7684,'selected_step':1159,'patience':0,'retention_failures':0,'next_eval_epoch':.25,
    'train_wall_seconds':100.,'audio_seconds':1000.,'evaluation_wall_seconds':20.,
    'language_audio_seconds':{'tr':500.,'en':100.},'initial_gate':{'turkish_wer':.8242},'stop_reason':'controller_stop_file'}
resumed = copy.deepcopy(paused)
resumed['stop_reason'] = None
rows = []


def check(name,expected,actual,want_pass):
    error = None
    try:
        stats, sections = verify.compare_checkpoints(expected,actual,torch)
        passed = True
    except Exception as exception:
        passed = False
        error = str(exception)
        stats, sections = {},{}
    outcome = passed == want_pass
    rows.append({'name':name,'passed':outcome,'expected_accept':want_pass,'accepted':passed,'error':error})
    if not outcome:
        raise AssertionError(name+': '+str(error))
    return stats,sections


stats,sections = check('sole_stop_reason_reset_accepted',paused,resumed,True)
assert sections['optimizer']['tensor_leaves'] == 207*3
assert stats['tensor_leaves'] == 624
rows.append({'name':'all_207_Adam_states_and_three_tensors_counted','passed':True,
    'adam_tensors':sections['optimizer']['tensor_leaves'],'total_tensor_leaves':stats['tensor_leaves']})

mutations = [
    ('one_weight_value',lambda state:state['weights']['small_weight'].__setitem__((0,0),.001)),
    ('weight_dtype',lambda state:state['weights'].update(small_weight=state['weights']['small_weight'].double())),
    ('weight_shape',lambda state:state['weights'].update(small_weight=state['weights']['small_weight'].flatten())),
    ('missing_weight',lambda state:state['weights'].pop('small_weight')),
    ('Adam_exp_avg',lambda state:state['optimizer']['state'][17]['exp_avg'].add_(.1)),
    ('Adam_exp_avg_sq',lambda state:state['optimizer']['state'][18]['exp_avg_sq'].add_(.1)),
    ('Adam_step',lambda state:state['optimizer']['state'][19]['step'].add_(1)),
    ('Adam_LR',lambda state:state['optimizer']['param_groups'][0].update(lr=3e-5)),
    ('Adam_parameter_order',lambda state:state['optimizer']['param_groups'][0]['params'].reverse()),
    ('sampler_order',lambda state:state['sampler']['tr']['order'].reverse()),
    ('sampler_cursor',lambda state:state['sampler']['tr'].update(position=2)),
    ('sampler_draws',lambda state:state['sampler']['tr'].update(draws=2319)),
    ('CPU_RNG',lambda state:state['torch_rng'].__setitem__(0,255)),
    ('CUDA_RNG',lambda state:state['cuda_rng'][0].__setitem__(0,255)),
    ('Python_RNG',lambda state:state.update(python_rng=random.Random(11).getstate())),
    ('history_value',lambda state:state['history'][0].update(loss=2.)),
    ('history_source',lambda state:state['history'][0]['sampled_source_ids'].__setitem__(0,'other_clip')),
    ('config_identity',lambda state:state['config_identity'].update(batch_size=2)),
    ('selected_step',lambda state:state.update(selected_step=0)),
    ('best_score',lambda state:state.update(best_score=.7)),
    ('patience',lambda state:state.update(patience=1)),
    ('retention_failures',lambda state:state.update(retention_failures=1)),
    ('next_evaluation',lambda state:state.update(next_eval_epoch=.5)),
    ('training_timer',lambda state:state.update(train_wall_seconds=101.)),
    ('audio_timer',lambda state:state.update(audio_seconds=1001.)),
    ('stop_reason_not_cleared',lambda state:state.update(stop_reason='controller_stop_file')),
    ('extra_checkpoint_field',lambda state:state.update(extra=True)),
    ('integer_boolean_alias',lambda state:state.update(step=True)),
    ('RNG_tuple_list',lambda state:state.update(python_rng=list(state['python_rng']))),
]
for name,modify in mutations:
    corrupted = copy.deepcopy(resumed)
    modify(corrupted)
    check(name,paused,corrupted,False)

fewer_paused,fewer_resumed = copy.deepcopy(paused),copy.deepcopy(resumed)
fewer_paused['optimizer']['state'].pop(206)
fewer_resumed['optimizer']['state'].pop(206)
check('expected_207_states_required',fewer_paused,fewer_resumed,False)
negative_zero_paused,negative_zero_resumed = copy.deepcopy(paused),copy.deepcopy(resumed)
negative_zero_paused['best_score'],negative_zero_resumed['best_score'] = 0.,-0.
check('scalar_negative_zero_bits',negative_zero_paused,negative_zero_resumed,False)
stride_paused,stride_resumed = copy.deepcopy(paused),copy.deepcopy(resumed)
stride_paused['weights']['small_weight']=torch.zeros(3,3)
stride_resumed['weights']['small_weight']=torch.zeros(3,3).T
check('equal_values_different_tensor_stride',stride_paused,stride_resumed,False)
nan_paused,nan_resumed = copy.deepcopy(paused),copy.deepcopy(resumed)
nan_paused['weights']['small_weight'][0,0]=float('nan')
nan_resumed['weights']['small_weight'][0,0]=float('nan')
check('nonfinite_identical_NaN_rejected',nan_paused,nan_resumed,False)
meta = copy.deepcopy(resumed)
meta['weights']['small_weight']=torch.empty(3,3,device='meta')
check('non_CPU_tensor_rejected_without_GPU_allocation',paused,meta,False)

report = {'passed':all(row['passed'] for row in rows),'cpu_only':True,'checks':len(rows),
    'verifier_sha256':verify.sha(Path(verify.__file__)),'fixtures':rows,
    'scope':'Tiny CPU toy tensors only. No actual paused/resumed checkpoint, GPU, process or heldout inference was used; actual exact-resume audit is still pending.'}
verify.write_json(OUTPUT/'exact_resume_auditor_cpu_verification.json',report)
print(json.dumps({'passed':report['passed'],'checks':report['checks'],'verifier_sha256':report['verifier_sha256']}))
