"""CPU-only source/metadata controls; no worker/model/decoder/native entry."""
import ast, copy, hashlib, importlib.util, json, os, shutil, sys, time
from pathlib import Path
from types import SimpleNamespace
import psutil

P=Path(r'C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish';R=P/'work/research'
N=Path(r'E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1');O=Path(__file__).resolve().parent
sys.path[:0]=[str(S),str(R)]
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import CompletePassCursor,canonical,digest
from bpe_final_DATA_schedule_context_v2 import normalize_schedule_state
from asr18_BPE9223_private_scalar_prefetch_v1 import PrivateScalarPrefetch
from resource_watchdog_v2 import sample_memory


def main():
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1' and os.environ.get('PYTHONUTF8')=='1' and sys.flags.utf8_mode==1,
            'CPU-only process-local UTF8')
    require(all(os.environ.get(k)=='1'for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')),
            'Four thread caps1')
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    began=time.monotonic();peak=0;samples=0
    def guard():
        nonlocal peak,samples
        require(time.monotonic()-began<60 and shutil.disk_usage(Path(r'E:/gpu_training_scratchpad')).free>=100000000000,
                'Metadata-only60s/E100GB limits')
        s=sample_memory(os.getpid(),'inactive_scalar_prefetch_integrated_metadata_protocol')
        peak=max(peak,s['worker_private_memory']['private_bytes']);samples+=1
        require(peak<=512*1024**2 and s['system_memory']['commit_available_bytes']>=2000000000 and
                s['system_memory']['physical_available_bytes']>=1500000000,'Unchanged actual private/system limits')
        return s
    guard()
    out=O/'metadata_protocol_control_actual_01';require(not out.exists(),'One fresh metadata result; preserve failures');out.mkdir()
    try:
        old_path=R/'run_ASR18_BPE9223_resident_RAM_training_v4.py';new_path=R/'run_ASR18_BPE9223_resident_RAM_training_v5.py'
        old_ast=ast.parse(old_path.read_text(encoding='utf-8'));new_ast=ast.parse(new_path.read_text(encoding='utf-8'))
        dump=lambda node:ast.dump(node,include_attributes=False)
        top=lambda tree:{n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        old_top,new_top=top(old_ast),top(new_ast)
        exact_functions=[]
        for name,node in old_top.items():
            if name!='main':require(dump(node)==dump(new_top[name]),'Original untouched function/class:'+name);exact_functions.append(name)
        nested=lambda node,name:next(x for x in ast.walk(node)if isinstance(x,ast.FunctionDef)and x.name==name)
        for name in ('submit_plan','evaluate_coherent_DEV','record_periodic','stage','guard','journal_factory','admit_cache','owner_state_digest'):
            require(dump(nested(old_top['main'],name))==dump(nested(new_top['main'],name)),'Exact original nested body:'+name)
            exact_functions.append('main.'+name)
        native_loop=lambda tree:next(n for n in ast.walk(top(tree)['main'])if isinstance(n,ast.For)and
            isinstance(n.iter,ast.Call)and isinstance(n.iter.func,ast.Name)and n.iter.func.id=='enumerate'and
            ast.unparse(n.iter.args[0])=="plan['updates']")
        previous,current=native_loop(old_ast),copy.deepcopy(native_loop(new_ast))
        added=[n for n in current.body if isinstance(n,ast.If)and ast.unparse(n.test)=='future_ticket is not None']
        require(len(added)==1,'Only one post-commit metadata polling block')
        current.body=[n for n in current.body if n not in added]
        require(dump(current)==dump(previous),'Entire original native/commit/receipt/release/progress loop byteequivalent at AST level')
        checkblock=lambda tree:next(n for n in ast.walk(top(tree)['main'])if isinstance(n,ast.If)and
                                    isinstance(n.test,ast.Name)and n.test.id=='need_checkpoint')
        require(dump(checkblock(old_ast))==dump(checkblock(new_ast)),'Entire checkpoint writer/semantic/fingerprint/retirement block unchanged')
        fullpass=lambda tree:next(n for n in ast.walk(top(tree)['main'])if isinstance(n,ast.If)and
            ast.unparse(n.test)=="action['kind'] == 'checkpoint_then_serial350DEV'")
        require(dump(fullpass(old_ast))==dump(fullpass(new_ast)),'Original fullpass DEV/winner/phase body unchanged')
        loop=next(n for n in ast.walk(new_top['main'])if isinstance(n,ast.While)and
                  ast.unparse(n.test)=="optimizer.updates < c['planned_native_updates']")
        native_position=next(i for i,n in enumerate(loop.body)if n is native_loop(new_ast))
        after=loop.body[native_position+1:]
        require(isinstance(after[0],ast.If)and ast.unparse(after[0].test)=='future_ticket is not None'and
                any(isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='adopt_private_successor'and
                    any(k.arg=='wait'and isinstance(k.value,ast.Constant)and k.value.value is True for k in n.keywords)
                    for n in ast.walk(after[0])),'Mandatory final ordered adoption before queued/checkpoint/DEV decisions')
        checkpoint_position=next(i for i,n in enumerate(after)if isinstance(n,ast.Assign)and
            any(isinstance(t,ast.Name)and t.id=='need_checkpoint'for t in n.targets))
        require(checkpoint_position>2 and isinstance(after[1],ast.Expr)and
                'outstanding is None' in ast.unparse(after[1])and 'held_result is None' in ast.unparse(after[1]),
                'No outstanding metadata operation before checkpoint/DEV')
        old_wrapper_ast=ast.parse((R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v4.py').read_text(encoding='utf-8'))
        new_wrapper_ast=ast.parse((R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v5.py').read_text(encoding='utf-8'))
        require(dump(next(n for n in ast.walk(old_wrapper_ast)if isinstance(n,ast.While)))==
                dump(next(n for n in ast.walk(new_wrapper_ast)if isinstance(n,ast.While))),
                'Entire real Popen/wait0/death/all-updates-saved/autocontinuation loop unchanged')
        spec=importlib.util.spec_from_file_location('inactive_scalar_prefetch_driver5',new_path)
        driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
        h=read_json(N/'resident_RAM_private_scalar_prefetch_candidate_v1/proposal_handoff.json')
        plan_pin=h['proposed_exact_frozen_plan_pairs'][1][1]
        require(record(plan_pin['path'])==plan_pin,'Actual frozen plan from reviewed equality fixture')
        state=normalize_schedule_state(read_json(plan_pin['path'])['final_private_planning_sampler'])
        before=canonical(state);request=driver.private_successor_request(state,completed_boundaries=0,validation_stop=False,budget_stop=False)
        require(request['maximum_updates']==20 and request['before_sampler_sha256']==digest(state)and canonical(state)==before,
                'Actual ordinary state produces same bounded future request without mutation')
        for validation,budget in ((True,False),(False,True),(True,True)):
            require(driver.private_successor_request(state,completed_boundaries=0,validation_stop=validation,budget_stop=budget)is None,
                    'Known validation/budget boundary suppresses new work')
        # These are private scalar boundary descriptors, not claimed saved/native
        # frontiers. Original cardinality/seed/bijection methods supply the tail.
        cursor_state=copy.deepcopy(state['broad_cursor']);n=cursor_state['population'];u=(n+13)//14
        cursor_state['draws']=(n//14)*14
        cursor=CompletePassCursor(population=n,identity=cursor_state['identity'],seed=cursor_state['seed'],passes=2)
        cursor.load_state_dict(cursor_state);tail=cursor.preview(n-cursor.draws)
        require(1<=len(tail)<=14 and cursor.draws+len(tail)==n,'Exact original private pass-tail preview, no row drop')
        boundary_state=copy.deepcopy(state);boundary_state['broad_cursor']=cursor_state;boundary_state['carried_replay']['updates']=u-1
        tail_request=driver.private_successor_request(boundary_state,completed_boundaries=0,validation_stop=False,budget_stop=False)
        require(tail_request['maximum_updates']==1,'Only one full/tail update before realpassDEV')
        boundary_state['broad_cursor']['draws']=n;boundary_state['carried_replay']['updates']=u
        require(driver.private_successor_request(boundary_state,completed_boundaries=0,validation_stop=False,budget_stop=False)is None,
                'Unacknowledged pass boundary suppresses all future metadata')
        # Protocol exception controls only: no fictional live worker/process0.
        not_ready=SimpleNamespace(ready=lambda:False)
        require(driver.adopt_private_successor(not_ready,{},None,None,wait=False)is None,'No premature adoption or callback')
        wrong_ticket=SimpleNamespace(outstanding={'sequence':1},closed=False)
        try:PrivateScalarPrefetch.take(wrong_ticket,{'sequence':2})
        except ValueError:pass
        else:raise AssertionError('Reordered ticket accepted')
        class FailingProtocol:
            def take(self,*args,**kwargs):raise RuntimeError('injected metadata protocol failure')
        try:driver.adopt_private_successor(FailingProtocol(),{},None,None,wait=True)
        except RuntimeError as error:require(str(error)=='injected metadata protocol failure','Exact exception preserved')
        else:raise AssertionError('Metadata failure swallowed')
        # Run the actual changed admission function with actual current sample,
        # then copied scalar threshold cases. No system memory is changed.
        admission=nested(new_top['main'],'admit');tree=ast.Module(body=[copy.deepcopy(admission)],type_ignores=[]);ast.fix_missing_locations(tree)
        actual=guard();sample_box=[actual]
        ns={'guard':guard,'monitor':SimpleNamespace(_observe=lambda:sample_box[0]),'prefetch_policy':{'additional_host_reservation_bytes':268435456},'require':require}
        exec(compile(tree,str(new_path),'exec'),ns)
        admitted=ns['admit']('private_scalar_prefetch_additional_host_state',268435456)
        require(admitted['increment']==268435456 and admitted['additional_prefetch_reservation_bytes']==0,'Full constructor charge, no double-counted name')
        native=ns['admit']('native_existing_allocation',64*1024**2)
        require(native['additional_prefetch_reservation_bytes']==268435456,'Every other future allocation retains entire extra allowance')
        copied=copy.deepcopy(actual);sample_box[0]=copied
        copied['system_memory']['commit_available_bytes']=2000000000+268435456-1
        try:ns['admit']('private_scalar_prefetch_additional_host_state',268435456)
        except ValueError:pass
        else:raise AssertionError('Prospective commit headroom shortfall accepted')
        copied['system_memory']['commit_available_bytes']=actual['system_memory']['commit_available_bytes']
        copied['system_memory']['physical_available_bytes']=1500000000+268435456-1
        try:ns['admit']('private_scalar_prefetch_additional_host_state',268435456)
        except ValueError:pass
        else:raise AssertionError('Prospective physical headroom shortfall accepted')
        old_config=read_json(N/'resident_RAM_DEV_frontend_source_v1/configuration.json');new_config=read_json(O/'configuration.json')
        delta=[k for k in set(old_config)|set(new_config)if old_config.get(k)!=new_config.get(k)]
        require(delta==['private_scalar_prefetch_policy'],'One exact configuration semantic addition only')
        guard();require('torch'not in sys.modules and'numpy'not in sys.modules,'No model/audio/numerical runtime imported')
        report={'kind':'ASR18_inactive_integrated_scalar_prefetch_metadata_protocol_control_v1','passed':True,
            'source':record(__file__),'driver':record(new_path),'wrapper':record(R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v5.py'),
            'configuration':record(O/'configuration.json'),'exact_unchanged_original_functions':exact_functions,
            'entire_native_commit_progress_loop_unchanged_except_post_commit_poll':True,'entire_checkpoint_block_unchanged':True,
            'entire_fullpass_DEV_winner_block_unchanged':True,'actual_frozen_state_plan':plan_pin,'private_tail_row_count':len(tail),
            'ordered_adoption_precedes_checkpoint_and_DEV':True,'real_Popen_wait0_and_automatic_continuation_loop_unchanged':True,
            'budget_validation_pass_boundary_suppression':True,'not_ready_ordering_and_exception_propagation_controls':True,
            'actual_fresh_256MiB_admission':admitted,'synthetic_threshold_arithmetic_negatives_rejected':True,
            'configuration_changed_fields':delta,'actual_worker_model_audio_GPU_or_native_execution':False,
            'actual_sampler_RNG_journal_DB_training_child_mutation':False,'scope':'Source and metadata protocol only; actual worker/whole-plan proof reused from prior CPU control, no production overlap claim',
            'guard_samples':samples,'maximum_sampled_private_bytes':peak,'elapsed_seconds':time.monotonic()-began}
        write_json(out/'report.json',report);print(json.dumps({'passed':True,'report':record(out/'report.json')}),flush=True)
    except BaseException as error:
        import traceback
        write_json(out/'actual_failure.json',{'error':repr(error),'traceback':traceback.format_exc()});raise


if __name__=='__main__':main()
