"""Focused stdlib-only adapter fixtures; no native/model/checkpoint execution."""
import ast,copy,hashlib,importlib.util,json,pathlib,sys,tempfile
ROOT=pathlib.Path.cwd();HERE=ROOT/'outputs/parakeet_turkish';sys.path.insert(0,str(HERE))
import run_TSC_uniform_head_lr_journal_v1 as m
from append_only_history_v1 import HistoryJournal
checks=[];rejections=[]
def check(name,ok):
    if not ok:raise AssertionError(name)
    checks.append(name)
def reject(name,fn):
    try:fn()
    except (ValueError,KeyError):rejections.append(name);return
    raise AssertionError(name)

def normalize_window(source):
    tree=ast.parse(source);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_window')
    return fn
old=normalize_window((HERE/'run_TSC_uniform_head_lr_v1.py').read_text(encoding='utf-8'))
new=normalize_window(pathlib.Path(m.__file__).read_text(encoding='utf-8'))
# Remove only the declared observer initialization; map its two reads back to
# the original source. All stop, identity, resource, cleanup and closure AST stays.
text=ast.get_source_segment(pathlib.Path(m.__file__).read_text(encoding='utf-8'),new)
text=text.replace("\n    observer=WindowObserver(state['history_runtime_migration']['journal_directory'],state['resume_step'])",'')
text=text.replace('history=observer.rows()',"history=read(HERE/'runs'/state['run_id']/'training_history.json',[])")
text=text.replace('enumerate(history[1:],1)',"enumerate(history[state['resume_step']:],1)")
check('unchanged_parent_guard_stop_wait_and_closure_AST',ast.dump(ast.parse(text).body[0],include_attributes=False)==ast.dump(old,include_attributes=False))
check('stdlib_import_has_no_Torch','torch' not in sys.modules)
check('unchanged_worker_source',m.c.sha(HERE/m.c.WORKER)==m.EXPECTED_WORKER)
check('unchanged_dispatch_core_source',m.c.sha(m.c.__file__)==m.EXPECTED_CORE)

with tempfile.TemporaryDirectory(prefix='journal-adapter-',dir=ROOT/'work') as td:
    d=pathlib.Path(td);checkpoint=d/'opaque.fixture';checkpoint.write_text('not a model state')
    run_id='_CPU_journal_adapter_fixture';run=HERE/'runs'/run_id
    paths={k:str(d/(k+'.json')) for k in ('native','fresh_runtime','identity','memory','cache','io','log')}
    plan={'run_id':run_id,'resume_step':3,'source_checkpoint':m.c.record(checkpoint),'paths':paths,'continuation_mode':'same_new_run_exact_budget_continuation',
      'command':[str(ROOT/'work/nemo-native-env-v2/Scripts/python.exe'),str(pathlib.Path(m.__file__).resolve()),'--worker-plan',str(d/'plan.json'),'--gpu-window-authorized']}
    migration={'kind':m.KIND,'adapter':m.byte_record(m.__file__),'journal_helper':m.byte_record(HERE/'append_only_history_v1.py'),
        'journal_directory':str(run/'history_journal_v1'),'activation_source_step':3,'scientific_identity_changed':False,'pagefile_or_guard_changes':False}
    plan['history_runtime_migration']=migration
    original_validate=m.c.validate_plan;m.c.validate_plan=lambda value:{}
    try:
        check('extra_runtime_admission_with_stubbed_science_gate',m.validate_runtime(plan)==migration)
        bad=copy.deepcopy(plan);bad['history_runtime_migration']['scientific_identity_changed']=True
        reject('reject_scientific_identity_change',lambda:m.validate_runtime(bad))
        bad=copy.deepcopy(plan);bad['history_runtime_migration']['journal_helper']['sha256']='0'*64
        reject('reject_changed_journal_source',lambda:m.validate_runtime(bad))
        bad=copy.deepcopy(plan);bad['command'][1]=str(d/'wrong.py')
        reject('reject_wrong_actual_child_command',lambda:m.validate_runtime(bad))
        # Router/journal test uses a temp directory and explicit source-frame
        # label stub. It exercises actual row/snapshot/routing methods only.
        original_runtime=m.validate_runtime;m.validate_runtime=lambda value:value['history_runtime_migration']
        plan['history_runtime_migration']['journal_directory']=str(d/'journal')
        writes={};calls=[]
        def sink(path,value,**kwargs):
            value=copy.deepcopy(value);writes[pathlib.Path(path).resolve()]=value;calls.append(pathlib.Path(path).resolve())
        router=m.HistoryRouter(plan,sink);label=['initial'];router.label=lambda:label[0]
        rows=[{'step':i,'loss':i/10,'text':'İstanbul çığ','nested':{'source':'TRAIN','number':i}} for i in range(1,4)]
        before=copy.deepcopy(rows);router(run/'training_history.json',rows);router(HERE/'training_history.json',rows)
        check('source_history_not_mutated',rows==before)
        check('initial_both_compatible_snapshots_exact',writes[run/'training_history.json']==rows and writes[HERE/'training_history.json']==rows)
        initial_frontier=router.journal.max_step;initial_calls=calls.count(run/'training_history.json')
        label[0]='periodic';rows.append({'step':4,'loss':.125,'nested':{'source':'TRAIN','number':4}})
        router(run/'training_history.json',rows);router(HERE/'training_history.json',rows)
        check('periodic_tail_is_durable_and_no_full_snapshot',router.journal.max_step==4 and calls.count(run/'training_history.json')==initial_calls)
        check('periodic_retry_does_not_duplicate_events',router.journal.event_id==2)
        rows[-1]['validation']={'retention_gate_passed':False,'PT_errors':67};rows[-1]['selection_diagnostics']={'consecutive_retention_failures':1};label[0]='DEV'
        router(run/'training_history.json',rows);router(HERE/'training_history.json',rows)
        check('DEV_amendment_updates_same_step_once',router.journal.max_step==4 and router.journal.event_id==3)
        check('DEV_snapshots_and_reconstruction_exact',writes[run/'training_history.json']==rows and list(router.journal.records())==rows)
        unrelated=d/'training_history.json';router(unrelated,rows)
        check('same_basename_outside_scope_forwards_unchanged',writes[unrelated]==rows)
        config={'config_identity':{'unchanged':True},'phase':'old'};router(run/'run_config.json',config)
        check('config_overlay_does_not_mutate_science',config=={'config_identity':{'unchanged':True},'phase':'old'} and writes[run/'run_config.json']['config_identity']==config['config_identity'])
        ready={'kind':'fresh','launcher_sha256':m.EXPECTED_CORE};router(paths['fresh_runtime'],ready)
        check('fresh_runtime_adds_extra_provenance_only',writes[pathlib.Path(paths['fresh_runtime'])]['launcher_sha256']==m.EXPECTED_CORE and 'history_runtime_migration' in writes[pathlib.Path(paths['fresh_runtime'])])
        router(paths['native'],ready)
        check('original_native_ready_is_not_projected_as_new_provenance','history_runtime_migration' not in writes[pathlib.Path(paths['native'])])
        observer=m.WindowObserver(d/'journal',3);observed=observer.rows()
        check('parent_observer_excludes_whole_prefix_and_keeps_global_frontier',[r['step'] for r in observed]==[3,4])
        local=[{**row,'step':i} for i,row in enumerate(observed[1:],1)]
        check('progress_guard_receives_only_true_local_suffix',[r['step'] for r in local]==[1] and local[0]['loss']==rows[-1]['loss'])
        check('observer_gets_latest_DEV_amendment',observed[-1]['validation']==rows[-1]['validation'])
        empty=m.WindowObserver(d/'not_committed',3)
        check('startup_anchor_is_not_fake_gradient_progress',empty.rows()==[{'step':3}])
        label[0]='final';router(run/'training_history.json',rows);router(HERE/'training_history.json',rows)
        check('final_compatible_snapshots_complete',writes[run/'training_history.json']==rows and writes[HERE/'training_history.json']==rows)
        label[0]='periodic';bad=copy.deepcopy(rows);bad[-1]['loss']=99
        reject('reject_immutable_training_metric_change',lambda:router(run/'training_history.json',bad))
        m.validate_runtime=original_runtime
    finally:m.c.validate_plan=original_validate


core_tree=ast.parse(pathlib.Path(m.c.__file__).read_text(encoding='utf-8'))
core_run=next(x for x in core_tree.body if isinstance(x,ast.FunctionDef) and x.name=='run_worker')
mem=next(x.value for x in ast.walk(core_run) if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='memory' for t in x.targets))
lr_expr=next(v for k,v in zip(mem.keys,mem.values) if isinstance(k,ast.Constant) and k.value=='declared_head_lr')
for rate in (1e-5,3e-5):
    got=eval(compile(ast.Expression(lr_expr),'<actual_head_metadata>','eval'),{'float':float,'option':m.c.option,'argv':['--head-lr',str(rate)]})
    check('actual_head_sidecar_'+str(rate),got==rate)
for mode in ('constant','cosine'):
    for rate in (1e-5,3e-5):
        args=m.parser().parse_args(['--encoder-lr-schedule',mode,'--head-lr',str(rate)])
        check('future_parser_'+mode+'_'+str(rate),args.encoder_lr_schedule==mode and args.head_lr==rate)
check('future_choice_receipt_still_required',"encoder_policy_choice_review" in (HERE/'tsc_uniform_head_lr_transition_v1.py').read_text(encoding='utf-8'))

check('still_no_Torch_or_GPU_execution','torch' not in sys.modules)
report={'kind':'uniform_head_LR_journal_runtime_adapter_stdlib_fixtures_v1','passed':True,'checks':checks,'rejections':rejections,
    'scope':'Small explicit mocked-science/source-frame fixtures; actual journal/router/observer and source AST exercised. No actual checkpoint admission, native fit, training equality, throughput, Trackio or GPU.',
    'source_sha256':hashlib.sha256(pathlib.Path(m.__file__).read_bytes()).hexdigest(),'journal_sha256':hashlib.sha256((HERE/'append_only_history_v1.py').read_bytes()).hexdigest()}
p=ROOT/'work/uniform_head_LR_journal_runtime_adapter_CPU_verification_v1.json';p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'passed':True,'checks':len(checks),'rejections':len(rejections),'report':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))
