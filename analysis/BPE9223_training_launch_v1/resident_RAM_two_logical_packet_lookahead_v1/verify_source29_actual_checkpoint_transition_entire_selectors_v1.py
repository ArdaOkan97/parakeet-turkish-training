"""INACTIVE actual-closure/entire-selector metadata regression for source29.
Real source/config/closure validators and real record() on compact files. Journal
prefix-view and binding are explicit in-memory doubles: NO restore/copy/model/GPU.
Separate exact ROOT metadata CPU scope is required AFTER the actual stop seal.
"""
import argparse,ast,copy,hashlib,json,os,shutil,sys,time,types
from pathlib import Path
import psutil
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish';R=P/'work/research'
sys.path[:0]=[str(S),str(R)];F=Path(__file__).resolve().parent;BEGIN=time.monotonic()
from resource_watchdog_v2 import sample_memory

def require(v,m):
    if not v:raise ValueError(m)
def pin(p):
    p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    with Path(p).open('x',encoding='utf-8')as f:json.dump(v,f,indent=2);f.write('\n')
def identity(pid=None):
    p=psutil.Process(os.getpid()if pid is None else pid);return {'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()}
def guard():
    v=sample_memory(os.getpid(),'actual_closed_source27_metadata_selector_control');m=v['system_memory']
    require(v['paired_sampling_within_one_second']and v['worker_private_memory']['private_bytes']<536870912
        and time.monotonic()-BEGIN<180 and m['commit_available_bytes']>=2268435456
        and m['physical_available_bytes']>=1768435456 and shutil.disk_usage('E:/').free>=100000000000,
        'Metadata512MiB/180s/fullprefetch/originalfloors/E100GB')
    return v

def expect(kind,call):
    try:call()
    except kind as error:return {'type':type(error).__name__,'message':str(error)}
    raise AssertionError('Expected '+kind.__name__)
def execute(statement,scope):
    exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(statement)],type_ignores=[])),
        '<EXACT_FROZEN_SOURCE_AST_METADATA_ONLY>','exec'),scope)
def expression(value,scope):
    return eval(compile(ast.fix_missing_locations(ast.Expression(copy.deepcopy(value))),
        '<EXACT_FROZEN_SOURCE_DICTIONARY>','eval'),scope)

class PrefixViewDouble:
    """EXPLICIT no-read view double: real restore hashes/restores only after admission."""
    def __init__(self,saved,guard):self.saved=copy.deepcopy(saved)
    def binding(self):return copy.deepcopy(self.saved)
class JournalBindingDouble:
    """EXPLICIT in-memory prospective binding; no journal created or copied."""
    def __init__(self,value):self.value=value
    def binding(self):return copy.deepcopy(self.value)
class ForbiddenNamespace:
    def writer(self,*args,**kwargs):raise AssertionError('Metadata control MUST NOT open a production writer')

def main(a):
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'and os.environ.get('PYTHONUTF8')=='1'
        and all(os.environ.get(k)=='1'for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'))
        and Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad')),
        'Exact new metadata CPU-only E runtime')
    review=read(a.ROOT_review)
    require(review['accepted']is True and review['reviewer']=='/root'
        and review['fixture_source']==pin(__file__),'Separate exact actual closure metadata CPU scope')
    for item in review['input_pins']:require(pin(item['path'])==item,'Exact frozen metadata input')
    out=F/'typed_selector_actual_01';require(not out.exists(),'One fresh typed metadata output');out.mkdir()
    entry={'parent':identity(),'launcher':identity(os.getppid()),'ROOT_scope':pin(a.ROOT_review)};write(out/'entry.json',entry)
    initial=guard();m=initial['system_memory'];require(m['commit_available_bytes']>=2805306368 and m['physical_available_bytes']>=2305306368,
        'Full prospective512MiB plus fullprefetch/original floors')
    results={}
    try:
        configuration_pin=review['source_configuration'];configuration=read(configuration_pin['path'])
        closure_pin=review['lookahead_resume'];pointer_pin=review['initial_checkpoint'];branch=Path(review['branch'])
        require(configuration['prior_resident_RAM_approved_lookahead_checkpoint_transition']==closure_pin
            and configuration['resident_RAM_packet_credit_policy']['actual_CPU_control_review']==review['actual_CPU_review'],
            'Only genuine completed ROOT transport review and actual closure in final immutable configuration')
        import run_ASR18_BPE9223_resident_RAM_training_v29 as driver
        import fastmss_resident_feature_stream_v11 as CPU
        from asr18_approved_lookahead_checkpoint_transition_v3 import checked_lookahead_checkpoint_resume,validate_intentional_optimizer_journal_gap
        require('torch'not in sys.modules and 'numpy'not in sys.modules,'No NumPy/Torch/model imports')
        driver.checked_native_commit_pressure_policy(configuration) # Reuse CLOSED225 proof, no execution.
        CPU.checked_resource_profile(configuration);CPU.checked_packet_credit_policy(configuration)
        actual=checked_lookahead_checkpoint_resume(closure_pin,base=configuration,folder=branch,pointer_pin=pointer_pin)
        results['real_actual_closure_and_all_policy_validators']={'passed':True,'actual_optimizer_frontier':actual['actual_optimizer_updates'],
            'published_journal_frontier':actual['published_journal_optimizer_updates'],'gap':actual['optimizer_vs_journal_frontier']}
        source=Path(driver.__file__);tree=ast.parse(source.read_text(encoding='utf-8-sig'))
        main_node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main')
        optional=['prior_process','cushion225_resume','postrestore_selector_resume','packet_credit_resume','native_headroom_resume',
            'abnormal_resume','source_transition_resume','original_interval_resume','DEV_frontend_resume','DEV_admission_resume','periodic_quiescence_resume']
        args=types.SimpleNamespace(**{k:None for k in optional},lookahead_resume=closure_pin['path'],invocation=21,start_invocation=21)
        entry_if=next(n for n in main_node.body if isinstance(n,ast.If)and isinstance(n.test,ast.Compare)
            and isinstance(n.test.left,ast.Name)and n.test.left.id=='lookahead_transition')
        execute(entry_if,{'a':args,'lookahead_transition':actual,'require':require})
        wrong=copy.deepcopy(args);wrong.invocation=20
        results['wrong_initial_invocation_rejected']=expect(ValueError,lambda:execute(entry_if,{'a':wrong,'lookahead_transition':actual,'require':require}))
        wrong=copy.deepcopy(args);wrong.postrestore_selector_resume='historical_not_selected'
        results['historical_selector_mix_rejected']=expect(ValueError,lambda:execute(entry_if,{'a':wrong,'lookahead_transition':actual,'require':require}))
        saved=read(pointer_pin['path'])['journal_prefix']
        factory_node=next(n for n in ast.walk(main_node)if isinstance(n,ast.FunctionDef)and n.name=='journal_factory')
        scope={'a':args,'source_journal':saved,'transition':actual,'lookahead_transition':actual,
            'interval_failure':None,'periodic_quiescence_failure':None,'DEV_admission_failure':None,
            'frontend_failure':None,'abnormal':None,'SavedJournalPrefixView':PrefixViewDouble,'guard':lambda:None,
            'source_namespace':ForbiddenNamespace(),'require':require,'record':driver.record}
        execute(factory_node,scope);view,before=scope['journal_factory'](saved)
        require(view.binding()==saved and before=={'kind':'ASR18_readonly_source_transition_saved_prefix_restore_v1',
            'actual_transition':closure_pin,'no_tail_replay':True,'original_modified':False},
            'ENTIRE real pre-restore metadata dictionary uses actual closure record, not record(None)')
        results['entire_real_pre_restore_factory_metadata']={'metadata':before,'explicit_no_read_view_double':True,
            'factory_AST':ast.dump(factory_node,include_attributes=False)}
        post_calls=[n for n in ast.walk(main_node)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)
            and n.func.id=='write_json'and n.args and 'actual_journal_branch.json'in ast.unparse(n.args[0])]
        require(len(post_calls)==1 and isinstance(post_calls[0].args[1],ast.Dict),'One exact complete post-restore metadata dictionary')
        prospective={'path':str(branch/'journals/commits_000021.jsonl'),'offset':saved['offset'],'sha256':saved['sha256']}
        post_scope={**scope,'preserved_tail':actual,'journal':JournalBindingDouble(prospective)}
        after=expression(post_calls[0].args[1],post_scope)
        require(after=={'kind':'ASR18_actual_saved_prefix_recovery_journal_branch_v1','predecessor':closure_pin,
            'saved_prefix':saved,'preserved_original':actual['whole_journal_including_preserved_tail'],
            'new_active_journal':prospective,'historical_unsaved_tail_updates':actual['preserved_tail_positive_updates'],
            'native_updates_or_draws_in_copy':0},
            'ENTIRE actual post-restore metadata dict keeps real predecessor and journal-only tail, binding explicitly doubled')
        results['entire_real_post_restore_dictionary']={'metadata':after,'explicit_journal_binding_double_no_copy':True,
            'dictionary_AST':ast.dump(post_calls[0].args[1],include_attributes=False)}
        wrapper=Path(review['wrapper_source']['path']);wtree=ast.parse(wrapper.read_text(encoding='utf-8-sig'))
        wm=next(n for n in wtree.body if isinstance(n,ast.FunctionDef)and n.name=='main')
        route=next(n for n in wm.body if isinstance(n,ast.If)and isinstance(n.test,ast.Compare)
            and isinstance(n.test.left,ast.Attribute)and n.test.left.attr=='prior_process')
        # This dictionary is an explicit structural review DOUBLE in memory only.
        # It is never written as ROOT acceptance or usable production configuration.
        structural_review={'lookahead_resume':closure_pin,'lookahead_transition_validator':review['lookahead_transition_validator'],
            'human_approval':configuration['resident_RAM_packet_credit_policy']['human_approval'],
            'packet_credit_policy':configuration['resident_RAM_packet_credit_policy'],
            'native_update_commit_pressure_policy':configuration['native_update_commit_pressure_policy'],'initial_owner_invocation':21}
        wrapper_scope={'a':args,'review':structural_review,'configuration':configuration,'record':driver.record,'S':S,'require':require}
        execute(route,wrapper_scope)
        wrong=copy.deepcopy(args);wrong.start_invocation=20
        results['wrapper_wrong_initial_rejected']=expect(ValueError,lambda:execute(route,{**wrapper_scope,'a':wrong}))
        missing=copy.deepcopy(args);missing.lookahead_resume=None
        results['wrapper_missing_new_lane_rejected']=expect(ValueError,lambda:execute(route,{**wrapper_scope,'a':missing}))
        loop=next(n for n in wm.body if isinstance(n,ast.While))
        forwarding=next(n for n in loop.body if isinstance(n,ast.If)and isinstance(n.test,ast.Compare)
            and isinstance(n.test.left,ast.Name)and n.test.left.id=='prior')
        forwarding_scope={'a':args,'prior':None,'invocation':21,'command':[]}
        execute(forwarding,forwarding_scope)
        require(forwarding_scope['command']==['--lookahead-resume',closure_pin['path']],
            'ENTIRE actual first-invocation wrapper forwarding selects new lane exactly once')
        results['entire_actual_wrapper_entry_and_forwarding']={'passed':True,'forwarded':forwarding_scope['command'],
            'structural_review_explicit_double':True,'entry_AST':ast.dump(route,include_attributes=False),
            'forwarding_AST':ast.dump(forwarding,include_attributes=False)}
        K=actual['last_fully_saved_optimizer_updates']
        zero={'actual_optimizer_updates':K,'error':"ValueError('Actual owner cooperative stop')",'traceback':''}
        one={'actual_optimizer_updates':K+1,'error':zero['error'],
            'traceback':'in commit_RAM_transaction\n journal.append(event)\n in append\n self.namespace.check(); self.binding()\n in check\n self.guard()'}
        results['gap0_METHOD_STUB']=validate_intentional_optimizer_journal_gap(zero,K)
        results['gap1_exact_preappend_METHOD_STUB']=validate_intentional_optimizer_journal_gap(one,K)
        for name,change in [('gap2',{'actual_optimizer_updates':K+2}),('negative_gap',{'actual_optimizer_updates':K-1}),
            ('wrong_error',{'error':'unrelated error'}),('missing_commit_trace',{'traceback':'in append self.namespace.check(); self.binding() self.guard()'}),
            ('missing_append_trace',{'traceback':'in commit_RAM_transaction self.namespace.check(); self.binding() self.guard()'}),
            ('native_cleanup_not_accepted',{'traceback':'in native_update norm=float(optimizer.step()) resident_gc_scope'})]:
            failure={**one,**change}
            results['gap_reject_'+name+'_METHOD_STUB']=expect(ValueError,lambda failure=failure:validate_intentional_optimizer_journal_gap(failure,K))
        stat=actual['closed_old_journal_STAT'];now=Path(stat['path']).stat()
        require(now.st_size==stat['bytes']and now.st_mtime_ns==stat['mtime_ns'],'Original20 STAT only remains unchanged')
        require(not(branch/'invocations/owner_000021').exists()and not(branch/'journals/commits_000021.jsonl').exists()
            and not(F.parent/'resident_RAM_training_actual_22').exists(),'No production output/lease/branch was created by metadata control')
        for item in review['input_pins']:require(pin(item['path'])==item,'Frozen input exact after typed selector proof')
        require('torch'not in sys.modules and 'numpy'not in sys.modules,'No NumPy/Torch/model activity')
        report={'kind':'ASR18_actual_source29_saved_checkpoint_lookahead_entire_selector_metadata_CPU_control_v1',
            'passed':True,'results':results,'parent':entry,'initial_guard':initial,'final_guard':guard(),
            'entire_pre_and_postrestore_metadata_dictionaries_and_wrapper_entry_forwarding_executed':True,
            'real_actual_closure_and_credit_policy_validators_executed':True,
            'prefix_view_and_journal_binding_and_structural_review_are_explicit_in_memory_doubles':True,
            'gap0_gap1_trace_controls_are_METHOD_STUBS_not_actual_training':True,
            'actual_native_updates':0,'actual_audio_decode':0,'actual_sampler_draws':0,
            'actual_restore_or_prefix_copy_or_journal_write_or_kernel_credit':False,
            'production_source_or_environment_changed':False,'input_pins':review['input_pins'],
            'fixture_source':pin(__file__),'driver_source':pin(source),'wrapper_source':pin(wrapper),
            'source_configuration':configuration_pin,'actual_transition':closure_pin}
        write(out/'report.json',report);print(json.dumps({'report':pin(out/'report.json')}),flush=True)
    except BaseException as error:
        import traceback
        write(out/'actual_failure.json',{'error':repr(error),'traceback':traceback.format_exc(),'parent':entry,'partial_results':results});raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ROOT-review',required=True);main(p.parse_args())
