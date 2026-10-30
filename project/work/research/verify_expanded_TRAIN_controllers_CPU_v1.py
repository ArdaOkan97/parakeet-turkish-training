"""Focused new completion clocks/coverage rejection and inherited journal binding checks."""
import ast,copy,hashlib,json,pathlib,sys
R=pathlib.Path(__file__).resolve().parents[2];O=R/'outputs/parakeet_turkish';sys.path.insert(0,str(O))
import run_TSC_expanded_TRAIN_v1 as c
import run_TSC_expanded_TRAIN_journal_v1 as a
checks=[];rejections=[]
def passed(name):checks.append(name)
def rejects(name,fn):
    try:fn()
    except (ValueError,KeyError):rejections.append(name);return
    raise AssertionError('Accepted invalid completion: '+name)
def node(path,name):
    return ast.dump(next(x for x in ast.parse(path.read_text()).body if getattr(x,'name',None)==name),include_attributes=False)
for name in ['WindowObserver','HistoryRouter','run_window','run_child','validate_runtime','publish_queue','validate_identity']:
    assert node(O/'run_TSC_expanded_TRAIN_journal_v1.py',name)==node(O/'run_TSC_uniform_TR_loss_journal_v2.py',name);passed('unchanged actual journal implementation '+name)
for name in ['run_worker','validate_identity','request_stop']:
    if name=='run_worker':continue
    assert node(O/'run_TSC_expanded_TRAIN_v1.py',name)==node(O/'run_TSC_uniform_TR_loss_v1.py',name);passed('unchanged owned process/stop '+name)
obj=object.__new__(a.HistoryRouter);scope={'router':obj}
exec(compile('def publish_history(label):\n    return router.label()\n',str(O/c.WORKER),'exec'),scope)
for label in ['initial','periodic','DEV','final']:assert scope['publish_history'](label)==label;passed('new exact publisher path '+label)
rejects('unknown publisher label',lambda:scope['publish_history']('unknown'))
rejects('outside actual publisher path',lambda:obj.label())
assert c.caller_line(O/c.WORKER)==810;passed('actual pinned single CUDA-sync line810 AST')
assert c.parser().parse_args(['--worker-plan','synthetic.json','--gpu-window-authorized']).worker_plan
assert a.parser().parse_args(['--worker-plan','synthetic.json','--gpu-window-authorized']).worker_plan;passed('actual core and journal native-child parser routes')
assert not any(s in (O/'run_TSC_expanded_TRAIN_journal_v1.py').read_text() for s in ['_legacy_verify_completion','_old_phase','reviewed_closure']);passed('no copied legacy phase/failed closure projection')

class History:
    def __init__(self,n,validation=True):self.n=n;self.validation=validation
    def __len__(self):return self.n
    def __getitem__(self,k):
        assert k==-1
        return {'step':self.n,**({'validation':{'turkish_wer':.4}} if self.validation else {})}

def fixture(arm,step):
    pool=352768 if arm=='expanded' else 253798;local=step-40750;unique=min(pool,14*local)
    paths={k:O/'CPU_FIXTURE'/f'{k}.json' for k in ['native','identity','memory','cache','io']}
    plan={'paths':{k:str(v) for k,v in paths.items()},'run_id':'CPU_FIXTURE','configured_steps':65948,'source_selected_step':40750,
      'resume_step':40750,'active_TR_pool_clips':pool,'data_arm':arm,'encoder_lr_schedule':'cosine','source_closure':{'fixture':'explicit'}}
    selection={'run_id':'CPU_FIXTURE','completed_steps':step,'configured_steps':65948,'heldout_tests_evaluated':False,
      'encoder_schedule_updates_completed':local,'data_expansion_updates':local,'data_arm':arm,'encoder_lr_schedule':'cosine',
      'encoder_schedule_horizon':25198,'persistence_mode':'double_json','uniform_origin_step':6500,'majority_updates':step-7621,
      'PT_focus_updates':step-8000,'full_new_uniform_pass_complete':unique==pool,
      'data_expansion_epoch':{'source_step':40750,'local_updates':local,'data_arm':arm,'draws':14*local,'pool':pool,'unique':unique,
      'repeat_visits':14*local-unique,'new_epoch_not_inherited_prefix':True,'full_pass_complete':unique==pool}}
    plan.update(quota_boundary_step=7621,PT_focus_boundary_step=8000)
    config={'trainable_parameter_tensors':207,'turkish_clips_used':pool,'replay_clips_used':1914,'data_arm':arm,'phase':'HF_TSC_8199_expanded_TRAIN_control_v1'}
    proof={'kind':'TSC_uniform_majority_before_update_source_equivalence_v1','passed':True,'actual_GPU_decode':True,
      'source_selected_step':40750,'source_receipt':plan['source_closure'],'before_first_majority_update':True,'compared_fixed_DEV_utterances':350,
      'all_raw_transcripts_and_emitted_paths_exact':True,'all_seven_ASR_counts_exact':True,'RNG_restored_after_added_evaluation':True,
      'module_training_modes_restored_after_added_evaluation':True}
    summary={'validation_'+lang:{'asr':{'utterances':50,'reference_words':words,'wer':.4}} for lang,words in {'en':1045,'es':1289,'it':1158,'fr':1072,'de':1004,'tr':825,'pt':1095}.items()}
    values={paths['identity']:{'pid':10},paths['native'].with_suffix('.completion.json'):{'kind':'TSC_uniform_majority_ptfocus_native_completion_v1',
       'status':'complete','sources_unchanged':True,'pid':10,'run_id':'CPU_FIXTURE','fresh_runtime_sha256':'fixture','actual_dispatch_source_sha256':'fixture'},
      paths['memory']:{'kind':'TSC_uniform_majority_ptfocus_memory_runtime_v1','status':'complete','sources_unchanged':True,'cap_bytes':8000000000,
       'cleanup_threshold_bytes':3000000000,'allocator_cap_applied':True,'all_phase_owned_memory_telemetry':{'thread_join_completed':True}},
      paths['cache']:{'kind':'TSC_uniform_majority_ptfocus_cache_runtime_v1','status':'complete','sources_unchanged':True,'cache_reclamation':{'eligible_update_calls':local}},
      paths['io']:{'kind':'TSC_uniform_majority_ptfocus_atomic_IO_v1','status':'complete','sources_unchanged':True},
      O/'runs/CPU_FIXTURE/training_history.json':History(step),O/'runs/CPU_FIXTURE/run_config.json':config,
      O/'runs/CPU_FIXTURE/majority_initial_source_equivalence.json':proof,
      O/'results'/f'CPU_FIXTURE_step_{step}'/'summary.json':summary}
    return plan,selection,values

original=(c.read,c.sha,c.record)
try:
    for arm in ['base','expanded']:
        for step in [41250,65948]:
            p,s,v=fixture(arm,step);c.read=lambda path,*args:v[pathlib.Path(path)];c.sha=lambda p:'fixture';c.record=lambda p:{'path':str(p),'sha256':'fixture'}
            assert c.verify_completion(p,s)['full207_frontier']==step;passed(f'actual completion function {arm} {step} completeDEV regardless retention')
    mutations=[('stale phase',lambda p,s,v:v[O/'runs/CPU_FIXTURE/run_config.json'].update(phase='HF_TSC_8199_uniform_head_LR_control_v1')),
      ('wrong local clock',lambda p,s,v:s.update(data_expansion_updates=25197)),('wrong horizon',lambda p,s,v:s.update(encoder_schedule_horizon=18129)),
      ('wrong arm',lambda p,s,v:s.update(data_arm='base')),('old TR count',lambda p,s,v:v[O/'runs/CPU_FIXTURE/run_config.json'].update(turkish_clips_used=253798)),
      ('wrong source H',lambda p,s,v:s['data_expansion_epoch'].update(source_step=25109)),('false full coverage',lambda p,s,v:s['data_expansion_epoch'].update(unique=352767)),
      ('wrong tail',lambda p,s,v:s['data_expansion_epoch'].update(repeat_visits=10)),('inherited fake epoch',lambda p,s,v:s['data_expansion_epoch'].update(new_epoch_not_inherited_prefix=False)),
      ('missing final DEV',lambda p,s,v:v.update({O/'runs/CPU_FIXTURE/training_history.json':History(65948,False)})),
      ('incomplete fixed DEV',lambda p,s,v:v[O/'results/CPU_FIXTURE_step_65948/summary.json']['validation_tr']['asr'].update(utterances=49)),
      ('wrong fixed DEV words',lambda p,s,v:v[O/'results/CPU_FIXTURE_step_65948/summary.json']['validation_pt']['asr'].update(reference_words=1094)),
      ('changed runtime sources',lambda p,s,v:v[pathlib.Path(p['paths']['native']).with_suffix('.completion.json')].update(sources_unchanged=False)),
      ('wrong cleanup hook count',lambda p,s,v:v[pathlib.Path(p['paths']['cache'])]['cache_reclamation'].update(eligible_update_calls=25197)),
      ('missing source350',lambda p,s,v:v[O/'runs/CPU_FIXTURE/majority_initial_source_equivalence.json'].update(compared_fixed_DEV_utterances=300)),
      ('source RNG not restored',lambda p,s,v:v[O/'runs/CPU_FIXTURE/majority_initial_source_equivalence.json'].update(RNG_restored_after_added_evaluation=False))]
    for name,mutate in mutations:
        p,s,v=fixture('expanded',65948);mutate(p,s,v);c.read=lambda path,*args:v[pathlib.Path(path)];rejects(name,lambda:c.verify_completion(p,s))
finally:c.read,c.sha,c.record=original
assert 'torch' not in sys.modules
def rec(path):return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
result={'kind':'ROOT_expanded_TRAIN_controller_journal_focused_CPU_verification_v1','passed':True,'checks':checks,'rejections':rejections,
 'positive_count':len(checks),'rejection_count':len(rejections),'core':rec(O/'run_TSC_expanded_TRAIN_v1.py'),
 'adapter':rec(O/'run_TSC_expanded_TRAIN_journal_v1.py'),'worker':rec(O/c.WORKER),'Torch_imported':False,'GPU_used':False,
 'real_runtime_closure_or_tensor_fit_proved':False,'fixture_state_is_synthetic_and_never_authorizes_GPU':True}
path=O/'expanded_TRAIN_controller_journal_focused_CPU_verification_v1.json';assert not path.exists();path.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result));print(json.dumps(rec(path)))
