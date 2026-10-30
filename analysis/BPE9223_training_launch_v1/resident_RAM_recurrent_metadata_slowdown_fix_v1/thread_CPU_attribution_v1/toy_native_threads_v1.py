"""NEW CPU-only toy for nativeTID/counter lifecycle; not the old py-spy toy."""
import argparse,hashlib,json,math,os,threading,time
from pathlib import Path
import psutil
from owned_thread_times_v1 import ThreadQuery

def identity(p):return {'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()}
def thread_identity():
 q=ThreadQuery(os.getpid(),threading.get_native_id())
 try:return {'threading_ident':threading.get_ident(),**q.sample()}
 finally:q.close()
def main(a):
 out=Path(a.output);assert out.is_dir()and not(out/'toy_ready.json').exists()
 assert os.environ['CUDA_VISIBLE_DEVICES']=='-1'and all(os.environ[k]=='1'for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'))
 psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
 began=time.monotonic();ready=threading.Event();stop=threading.Event();worker={};error=[]
 def toy_scalar_worker():
  try:
   worker.update(thread_identity());ready.set();x=.25
   while time.monotonic()-began<8 and not stop.is_set():
    for i in range(3000):x=math.sin(x)+.25
    time.sleep(.001)
   worker['scalar_result']=x
  except BaseException as e:error.append({'type':type(e).__name__,'message':str(e)});ready.set()
 t=threading.Thread(target=toy_scalar_worker,name='ToyScalarWorker');t.start();assert ready.wait(5)and not error
 current=thread_identity();value={'kind':'ASR18_new_thread_CPU_counter_toy_ready_v1','producer':identity(psutil.Process()),'launcher':identity(psutil.Process().parent()),'main':current,'scalar_worker':dict(worker),'source':{'path':str(Path(__file__).resolve()),'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},'no_torch_numpy_audio_model_GPU':True}
 with(out/'toy_ready.pending').open('x',encoding='utf-8',newline='\n')as f:json.dump(value,f,indent=2);f.write('\n')
 (out/'toy_ready.pending').rename(out/'toy_ready.json')
 x=.5
 try:
  while time.monotonic()-began<10:
   for i in range(5000):x=math.cos(x)+.5
   time.sleep(.001)
 finally:stop.set();t.join(3)
 assert not t.is_alive()and not error
 with(out/'toy_completion.json').open('x',encoding='utf-8',newline='\n')as f:json.dump({'passed':True,'producer':value['producer'],'launcher':value['launcher'],'main':current,'scalar_worker':worker,'worker_joined':True,'errors':error,'scalar_result':x,'no_training_target_or_state_operation':True},f,indent=2);f.write('\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);main(p.parse_args())
