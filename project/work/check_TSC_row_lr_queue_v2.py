"""Focused isolated stdlib-only queue-v2 owned-child cleanup checks."""
from pathlib import Path
import argparse, ast, contextlib, hashlib, importlib.util, io, json, subprocess, sys, tempfile
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'outputs/parakeet_turkish/run_TSC_row_lr_queue_v2.py'

def main(args):
    checks=[]
    def check(value,label):
        if not value: raise AssertionError(label)
        checks.append(label)
    spec=importlib.util.spec_from_file_location('queue_v2_fixture',SOURCE)
    q=importlib.util.module_from_spec(spec); spec.loader.exec_module(q)
    original={key:getattr(q,key) for key in ('HERE','ROOT','CONTROLLER','write')}
    real_popen=q.subprocess.Popen; real_sleep=q.time.sleep; real_monotonic=q.time.monotonic
    try:
      with tempfile.TemporaryDirectory(prefix='row-queue-v2-',dir=ROOT/'work') as directory:
        temp=Path(directory); q.ROOT=temp; q.HERE=temp/'outputs'; q.HERE.mkdir(); q.CONTROLLER=q.HERE/'run_TSC_bpe_row_lr_v1.py'
        run_id='fixture_owned_run'; loop=q.HERE/'runs'/(run_id+'_loop_01'); loop.mkdir(parents=True)
        stop=loop/(run_id+'_stop.txt')
        argv=[sys.executable,'-S',str(q.CONTROLLER),'--run-id',run_id,'--output-dir',str(loop),'--gpu-window-authorized']
        q.time.sleep=lambda value:None
        class Child:
          pid=321
          def __init__(self,mode): self.mode=mode; self.wait_calls=[]; self.poll_calls=0
          def poll(self):
            self.poll_calls+=1
            if self.mode=='poll_error': raise OSError('mock handle polling failure')
            if self.mode in ('normal','nonzero'): return 0 if self.mode=='normal' else 7
            return None
          def wait(self,timeout=None):
            self.wait_calls.append(timeout)
            if self.mode=='timeout': raise subprocess.TimeoutExpired(argv,timeout)
            if timeout==0: raise subprocess.TimeoutExpired(argv,timeout)
            return 7 if self.mode=='nonzero' else 0
        def case(name,mode='normal',fail_publish=None,persistence_fail=False,cpu=False):
          folder=temp/name; folder.mkdir(); state={}; publications=[]; child=Child(mode)
          calls={'popen':0}
          def popen(*a,**k): calls['popen']+=1; return child
          q.subprocess.Popen=popen
          def publish(**values):
            publications.append(values); state.update(values)
            if fail_publish==len(publications): raise RuntimeError('mock publication failure')
            if fail_publish=='interrupt' and len(publications)==2: raise KeyboardInterrupt('mock interruption')
          if stop.exists(): stop.unlink()
          if persistence_fail: q.write=lambda *a,**k:(_ for _ in ()).throw(OSError('mock persistence failure'))
          else: q.write=original['write']
          error=None
          with contextlib.redirect_stdout(io.StringIO()):
            try: q.execute_owned(name,argv if not cpu else [sys.executable,'-S','mock_creator.py'],folder,publish,state)
            except BaseException as caught: error=caught
          return state,child,error,calls,folder
        state,child,error,calls,folder=case('normal')
        check(error is None and child.wait_calls==[None] and state['owned_child_pid'] is None,'normal owned wait0 before next stage')
        check(state['last_owned_child_cleanup']['cleanup_requested'] is False and not stop.exists(),'healthy completion adds no stop')
        state,child,error,calls,folder=case('after_spawn',mode='running',fail_publish=2)
        check(isinstance(error,RuntimeError) and calls['popen']==1,'post-spawn publish error propagates without retry')
        check(stop.exists() and child.wait_calls==[0.,5.],'post-spawn failure requests whole-update stop then bounded actual wait')
        check(state['owned_child_pid'] is None and state['last_actual_waitpid']['exit_code']==0,'only observed actual wait clears owned PID')
        proof=json.loads((folder/'after_spawn_owned_closure.json').read_text())
        check(proof['closure_observed'] and proof['actual_wait_observed'] and not proof['forced_termination_used'],'durable cleanup proof distinguishes actual owned close')
        state,child,error,calls,folder=case('loop_publish',mode='running',fail_publish=3)
        check(isinstance(error,RuntimeError) and state['last_owned_child_cleanup']['closure_observed'],'loop-publication failure closes owned child before repair')
        state,child,error,calls,folder=case('poll_failure',mode='poll_error')
        check(isinstance(error,OSError) and stop.exists() and state['owned_child_pid'] is None,'poll failure reaches cooperative finally and actual wait')
        state,child,error,calls,folder=case('interrupt',mode='running',fail_publish='interrupt')
        check(isinstance(error,KeyboardInterrupt) and state['last_owned_child_cleanup']['actual_wait_observed'],'interruption follows same owned cleanup')
        state,child,error,calls,folder=case('nonzero',mode='nonzero')
        check(isinstance(error,ValueError) and state['last_actual_waitpid']['exit_code']==7 and calls['popen']==1,'actual exit7 stops without second child')
        tick=[0.]
        def mono(): tick[0]+=400.; return tick[0]
        q.time.monotonic=mono
        state,child,error,calls,folder=case('timeout',mode='timeout',fail_publish=2)
        q.time.monotonic=real_monotonic
        check(state['owned_child_pid']==321 and not state['last_owned_child_cleanup']['closure_observed'],'timeout retains owned PID and closure unknown')
        check('last_actual_waitpid' not in state and state['last_owned_child_cleanup']['exit_code'] is None,'timeout never fabricates actual wait or exit0')
        stop.write_text('existing ROOT reason\n'); before=stop.read_bytes()
        check(q.request_owned_stop(stop) and stop.read_bytes()==before,'existing ROOT stop remains byte-identical')
        missing=loop/'not_created'/'stop.txt'
        check(not q.request_owned_stop(missing) and not missing.parent.exists(),'cleanup never creates controller namespace during admission')
        state,child,error,calls,folder=case('cpu_failure',mode='running',fail_publish=2,cpu=True)
        check(state['last_owned_child_cleanup']['cooperative_stop_path'] is None and not stop.exists(),'owned CPU stage waits without touching GPU stop')
        state,child,error,calls,folder=case('persist_failure',mode='running',fail_publish=2,persistence_fail=True)
        check(state['owned_child_pid'] is None and 'cleanup_evidence_persistence_error' in state,'proof persistence failure retains actual wait in state/stdout')
        state,child,error,calls,folder=case('before_spawn',mode='running',fail_publish=1)
        check(calls['popen']==0 and not child.wait_calls,'pre-spawn publication failure launches no child')
        bad=argv.copy(); bad[bad.index('--output-dir')+1]=str(temp/'outside_loop')
        try: q.cooperative_stop_path(bad)
        except ValueError: check(True,'stop path rejects namespace escape')
        else: check(False,'stop path rejects namespace escape')
        bad=argv.copy(); bad[2]=str(q.HERE/'other.py')
        try: q.cooperative_stop_path(bad)
        except ValueError: check(True,'stop admission rejects other GPU executors')
        else: check(False,'stop admission rejects other GPU executors')
        old=ast.parse(SOURCE.with_name('run_TSC_row_lr_queue_v1.py').read_text())
        new=ast.parse(SOURCE.read_text())
        functions=lambda tree:{node.name:ast.dump(node,include_attributes=False) for node in tree.body if isinstance(node,ast.FunctionDef)}
        a,b=functions(old),functions(new)
        check(a['review_closed']==b['review_closed'],'independent fixed DEV/scoring/retention/smoke reviewer unchanged')
        check(a['backlog_note']==b['backlog_note'],'existing local Backlog integration unchanged and never executed')
        check('torch' not in sys.modules and 'nemo' not in sys.modules,'no ML imports or tensors/checkpoints/models/GPU')
    finally:
      for key,value in original.items(): setattr(q,key,value)
      q.subprocess.Popen=real_popen; q.time.sleep=real_sleep; q.time.monotonic=real_monotonic
    output=args.output.resolve(); assert not output.exists(),'Fresh fixture receipt required'
    report={'schema':1,'kind':'TSC_row_LR_queue_v2_stdlib_owned_cleanup_verification_v1','passed':True,
            'source':str(SOURCE),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            'fixture_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'checks':checks,'check_count':len(checks),'owned_Popen_wait_and_poll':'explicit mocks',
            'all_paths_isolated_under_work':True,'actual_process_or_GPU_launched':False,
            'actual_controller_stop_or_hardware_closure_measured':False,'live_v1_modified':False}
    output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'passed':True,'checks':len(checks),'output':str(output)}))

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True); main(p.parse_args())
