"""INACTIVE finite fixture component; no action at import.

Runs the ACTUAL capture/compare source under explicitly substituted metadata,
tensor/optimizer/fingerprint/source-context modules. No Torch/NumPy/GPU tensor,
scientific byte equality, production context, DB, journal or draw is exercised.
The ORIGINAL pure require_exact_live_restore AST is used without alteration.
"""
import ast,hashlib,importlib.util,json,sys,types
from pathlib import Path


def require(value,message):
    if not value:raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,
        separators=(',',':'),allow_nan=False).encode()).hexdigest()


def pin(path):
    p=Path(path).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}


class TensorMetadataDouble:
    def __init__(self,index):
        self.pointer=100000+index*8;self._version=0;self.shape=(1,)
        self.dtype='METHOD_STUB_FP32';self.device='METHOD_STUB_CPU'
        self.requires_grad=True;self.grad=None
    def data_ptr(self):return self.pointer
    def stride(self):return (1,)
    def storage_offset(self):return 0


class AdamMetadataDouble:
    def __init__(self,count,offset):
        params=[TensorMetadataDouble(offset+i)for i in range(count)]
        self.param_groups=[{'params':params,'lr':.001,'METHOD_STUB':True}]
        self.state={p:{k:TensorMetadataDouble(offset+1000+i*3+j)
            for j,k in enumerate(('step','exp_avg','exp_avg_sq'))}for i,p in enumerate(params)}


class OptimizerMetadataDouble:
    def __init__(self):
        self.parameters={'w'+str(i).zfill(3):TensorMetadataDouble(i)for i in range(207)}
        self.optimizers=[AdamMetadataDouble(n,10000+j*10000)for j,n in enumerate((17,6,192))]
        self.updates=4000;self.base_clocks=(64250,2000,62250);self.poisoned=False;self.fingerprint_marker=0


class ScheduleMetadataDouble:
    def __init__(self):self.poisoned=False;self.updates=4000
    def committed_updates(self):return self.updates
    def state_dict(self):return {'METHOD_STUB':True,'updates':self.updates,'broad_draws':self.updates*14,'replay':self.updates,'cleaner_draws':0}


def check_actual_capture_and_compare_sources():
    F=Path(__file__).parent
    P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20')
    fingerprint_path=P/'outputs/parakeet_turkish/asr18_live_state_fingerprint_v1.py'
    tree=ast.parse(fingerprint_path.read_text(encoding='utf-8'))
    pure=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='require_exact_live_restore')
    pure_globals={'digest':digest,'require':require}
    exec(compile(ast.Module(body=[pure],type_ignores=[]),str(fingerprint_path),'exec'),pure_globals)
    original_pure_compare=pure_globals['require_exact_live_restore']
    optimizer=OptimizerMetadataDouble();schedule=ScheduleMetadataDouble()
    resolver=types.SimpleNamespace(metadata_locks=types.SimpleNamespace(handles=[111,222,333]))
    context=types.SimpleNamespace(resolver=resolver)
    journal=types.SimpleNamespace(binding=lambda:{'METHOD_STUB_no_journal_read':True,'offset':0,'sha256':'c'*64})
    queued=({'METHOD_STUB_immutable_plan':True,'draws_consumed':0},pin(F/'asr18_postDEV_refresh_state_capture_v1.py'))
    modules={}
    audio=types.ModuleType('bpe_broad_audio_v1');audio.record=pin;audio.require=require;modules[audio.__name__]=audio
    state=types.ModuleType('bpe_broad_production_state_v1');state.digest=digest;modules[state.__name__]=state
    fp=types.ModuleType('asr18_live_state_fingerprint_v1')
    def fingerprint_METHOD_STUB(actual,*,torch,guard):
        require(actual is optimizer and torch is None,'Explicit metadata fingerprint double, never actual GPU claim')
        guard()
        state={'METHOD_STUB_scientific_GPU_payloads':True,'updates':actual.updates,
            'weights':{name:{'version':v._version,'fake_payload_sha256':digest([name,v._version])}
                for name,v in sorted(actual.parameters.items())},
            'Adam_partitions':[{'state':{i:{'METHOD_STUB':True}for i in range(n)}}for n in (17,6,192)],
            'four_RNG_METHOD_STUB_marker':actual.fingerprint_marker}
        return {'state':state,'state_sha256':digest(state)}
    fp.live_state_fingerprint=fingerprint_METHOD_STUB;fp.require_exact_live_restore=original_pure_compare;modules[fp.__name__]=fp
    ctx=types.ModuleType('asr18_BPE9223_source_context_v2')
    ctx.full_run_next_draw_packet=lambda s,c,**kwargs:{'METHOD_STUB_no_draw':True,'updates':s.updates}
    modules[ctx.__name__]=ctx
    missing=object();prior={k:sys.modules.get(k,missing)for k in modules}
    before_modules=set(sys.modules);results=[]
    try:
        sys.modules.update(modules)
        source=F/'asr18_postDEV_refresh_state_capture_v1.py'
        spec=importlib.util.spec_from_file_location('ACTUAL_capture_source_under_explicit_metadata_doubles',source)
        actual=importlib.util.module_from_spec(spec);spec.loader.exec_module(actual)
        def capture(pending=queued):
            return actual.capture_idle_state(optimizer=optimizer,torch=None,guard=lambda:None,schedule=schedule,
                context=context,journal=journal,lifecycle={'winner':'METHOD_STUB'},periodic={'records':1},
                queued=pending,latest={'full_state_verified':True,'committed_updates':4000},completed=1)
        before=capture();after=capture();actual.compare_idle_state(before,after)
        require(len(before['parameters'])==207 and len(before['Adam_owners_and_215_slots'])==3
            and sum(len(group['states'])for owner in before['Adam_owners_and_215_slots']for group in owner['groups'])==215,
            'Actual bridge covers207 names/three owners/all215 metadata slots')
        results.append({'case':'actual_capture_compare_source_positive_under_metadata_tensor_fingerprint_doubles','passed':True,
            'real_GPU_scientific_fingerprint':False,'captured_207_names':207,'captured_Adam_slots':215})
        def reject(name,call):
            try:call()
            except ValueError as error:
                results.append({'case':name,'passed':True,'actual_error':str(error)});error.__traceback__=None
            else:raise AssertionError(name+' unexpectedly passed')
        value=optimizer.parameters['w000'];value._version+=1
        reject('actual_bridge_rejects_changed_parameter_version_and_fake_semantic_hash',lambda:actual.compare_idle_state(before,capture()))
        value._version-=1
        optimizer.fingerprint_marker+=1
        reject('actual_original_pure_compare_AST_rejects_changed_fake_RNG_fingerprint_value',lambda:actual.compare_idle_state(before,capture()))
        optimizer.fingerprint_marker-=1
        replacement=tuple([queued[0],queued[1]])
        require(replacement is not queued and replacement==queued,'Genuine same-content new tuple identity')
        reject('actual_bridge_rejects_same_content_new_queued_tuple_identity',lambda:actual.compare_idle_state(before,capture(replacement)))
        tensor=next(iter(optimizer.optimizers[0].state.values()))['exp_avg'];tensor.pointer+=8
        reject('actual_bridge_rejects_changed_Adam_storage_pointer_without_hash_difference',lambda:actual.compare_idle_state(before,capture()))
        tensor.pointer-=8
        removed=optimizer.parameters.pop('w206')
        reject('actual_capture_rejects_206_parameter_metadata',capture)
        optimizer.parameters['w206']=removed
        owner=optimizer.optimizers[0];removed_parameter=owner.param_groups[0]['params'].pop()
        reject('actual_capture_rejects_214_Adam_slot_metadata',capture)
        owner.param_groups[0]['params'].append(removed_parameter)
        actual.compare_idle_state(before,capture())
        require(not any(k in sys.modules for k in ('torch','numpy','av','transformers')),'No scientific/audio runtime imported')
        return {'cases':results,'actual_capture_source':pin(source),'original_pure_compare_source':pin(fingerprint_path),
            'original_pure_compare_AST_sha256':hashlib.sha256(ast.dump(pure,include_attributes=False).encode()).hexdigest(),
            'maximum_capture_canonical_bytes':len(json.dumps(before,sort_keys=True,separators=(',',':')).encode()),
            'explicit_doubles':list(modules),'real_production_semantics_or_tensor_payload_read':False}
    finally:
        for name,value in prior.items():
            if value is missing:sys.modules.pop(name,None)
            else:sys.modules[name]=value
        require(all(sys.modules.get(k,missing)is v for k,v in prior.items()),'Exact preexisting module aliases restored')


def check_actual_periodic_eligibility_branch(h,driver_path):
    """Actual new branch+original terminal AST, explicit closure/engine doubles."""
    import copy
    tree=ast.parse(Path(driver_path).read_text(encoding='utf-8'))
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main')
    event=next(n for n in ast.walk(main)if isinstance(n,ast.If)
        and ast.unparse(n.test)=='not postDEV_refresh.opportunity_consumed')
    # Exact original terminal branch immediately following the periodic branch.
    loop=next(n for n in ast.walk(main)if isinstance(n,ast.While)
        and ast.unparse(n.test)=="optimizer.updates < c['planned_native_updates']")
    periodic_index=next(i for i,n in enumerate(loop.body)if isinstance(n,ast.If)
        and isinstance(n.test,ast.Name)and n.test.id=='periodic_due'
        and any(isinstance(x,ast.If)and ast.unparse(x.test)=='not postDEV_refresh.opportunity_consumed'for x in n.body))
    terminal=loop.body[periodic_index+1]
    require(isinstance(terminal,ast.If)and ast.unparse(terminal.test)=='validation_stop or budget_stop',
        'Actual following original normal-stop branch, not an isolated replacement helper')
    wrapped=ast.While(test=ast.Constant(True),body=[copy.deepcopy(event),copy.deepcopy(terminal),ast.Break()],orelse=[])
    code=compile(ast.fix_missing_locations(ast.Module(body=[wrapped],type_ignores=[])),str(driver_path),'exec')
    results=[]
    def run(phase,worker,generation,validation,budget,expect_error=False):
        receipts=[];resolver=object();machine=h.OnePostDEVRefresh(0)
        if worker=='dead':
            prefetch=types.SimpleNamespace(closed=False,thread=types.SimpleNamespace(is_alive=lambda:False),
                worker_status={'closed':False,'error':None},outstanding=None,held_result=None,original=resolver)
        else:prefetch=None
        namespace={'require':require,'postDEV_refresh':machine,'optimizer':types.SimpleNamespace(updates=4000,poisoned=False),
            'engine':types.SimpleNamespace(schedule=types.SimpleNamespace(phase=phase,poisoned=False),
                context=types.SimpleNamespace(resolver=resolver)),
            'prefetch':prefetch,'prefetch_generation':generation,'future_ticket':None,'validation_stop':validation,
            'budget_stop':budget,'stream':None,'queued':({'METHOD_STUB_plan':True},{'METHOD_STUB_pin':True}),
            'last_checkpoint':4000,'latest':{'committed_updates':4000,'full_state_verified':True},
            'CPU_quiescence':{'CPU_actual0':{'actual_Popen_wait_code':0,'transport_threads_joined':True,'pipe_handles_closed':True,
                'packet_credit':{'all_owned_kernel_handles_closed':True},'packet_credit_owner':{'all_owned_kernel_handles_closed':True}}},
            'preservation':{'path':'METHOD_STUB_no_file_read'},
            'read_json':lambda path:{'actual_phase_private':{'joined':True,'error':None},
                'all207_threeAdam_all4RNG_unchanged':True,'both_cursors_journal_and_nextdraw_unchanged':True},
            'guard':lambda:None,'invocation':Path('E:/gpu_training_scratchpad/METHOD_STUB_no_write'),
            'write_json':lambda path,value:receipts.append(copy.deepcopy(value)),'normal_reason':None}
        try:exec(code,namespace)
        except ValueError as error:
            require(expect_error,'Unexpected actual branch rejection '+str(error));message=str(error);error.__traceback__=None
            require(not receipts and not machine.opportunity_consumed,'Reject absent/dead/unknown source before policy operation')
            return {'rejected':True,'actual_error':message}
        require(not expect_error,'Actual branch negative unexpectedly passed')
        require(machine.opportunity_consumed and not machine.attempted and not machine.completed
            and not machine.operations_completed and len(receipts)==1
            and receipts[0]['actual_refresh_performed']is False and receipts[0]['scalar_closure_fabricated']is False,
            'One truthful consumed opportunity, no GC/join/recreate or fabricated closure')
        require(namespace['prefetch']is prefetch,'Original terminal/absent scalar object untouched')
        exec(code,namespace)
        require(len(receipts)==1,'Actual branch never retries at a later pass once consumed')
        if validation or budget:
            expected='first200_checkpoint_fresh_restore_validation'if validation else'normal_owner_wall_budget'
            require(namespace['normal_reason']==expected,'Exact original following normal-stop reason/path preserved')
        return {'receipt':receipts[0],'original_normal_reason':namespace['normal_reason'],'all_production_closure_evidence':'METHOD_STUB'}
    for label,args in (
        ('actual_cleaner_never_constructed_truthful_ineligible',('cleaner_data_fine_tuning','absent',0,False,False,False)),
        ('actual_coincident_budget_preserves_original_terminal_path',('weak_label_continued_training','absent',0,False,True,False)),
        ('actual_coincident_validation_preserves_original_terminal_path',('weak_label_continued_training','absent',0,True,False,False)),
        ('actual_weak_label_absent_is_error',('weak_label_continued_training','absent',0,False,False,True)),
        ('actual_weak_label_dead_is_error',('weak_label_continued_training','dead',1,False,False,True)),
        ('actual_cleaner_prior_worker_missing_is_error',('cleaner_data_fine_tuning','absent',1,False,False,True))):
        results.append({'case':label,'passed':True,'detail':run(*args)})
    return {'cases':results,'driver_source':pin(driver_path),
        'event_AST_sha256':hashlib.sha256(ast.dump(event,include_attributes=False).encode()).hexdigest(),
        'original_terminal_AST_sha256':hashlib.sha256(ast.dump(terminal,include_attributes=False).encode()).hexdigest(),
        'candidate_driver_imported':False,'real_GPU_CPU_DB_Closure_or_state_operation':False}
