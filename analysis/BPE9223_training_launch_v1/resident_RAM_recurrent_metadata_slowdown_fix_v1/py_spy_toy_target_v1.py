"""Bounded owned standard-library toy; never imports the training parent."""
import argparse, gc, json, os, sys, threading, time
from pathlib import Path

def toy_python_phase():
    value=0
    for i in range(12000):value=(value+i*i)%10000019
    return value

def toy_gc_phase():
    a=[];a.append(a);del a
    return gc.collect()

def toy_sleep_phase():time.sleep(.004)

def toy_background(stop):
    while not stop.is_set():toy_sleep_phase()

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output).resolve()
    assert out.is_relative_to(Path(r'E:\gpu_training_scratchpad'))
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    assert all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'))
    assert not any(m=='torch' or m.startswith('torch.') or m=='numpy' or m.startswith('numpy.') for m in sys.modules)
    stop=threading.Event();worker=threading.Thread(target=toy_background,args=(stop,),name='owned-toy-sleep');worker.start()
    began=time.monotonic();cycles=0
    with (out/'toy_ready.json').open('x',encoding='utf-8') as f:
        json.dump({'pid':os.getpid(),'interpreter':sys.executable,'version':sys.version,'source':str(Path(__file__).resolve()),'monotonic':began,'standard_library_only':True},f,indent=2)
    try:
        while time.monotonic()-began<30 and not (out/'toy_stop.json').exists():
            toy_python_phase();toy_gc_phase();toy_sleep_phase();cycles+=1
    finally:
        stop.set();worker.join(timeout=5)
        assert not worker.is_alive()
        forbidden=[m for m in sys.modules if m=='torch' or m.startswith('torch.') or m=='numpy' or m.startswith('numpy.')]
        with (out/'toy_completion.json').open('x',encoding='utf-8') as f:
            json.dump({'pid':os.getpid(),'cycles':cycles,'wall_seconds':time.monotonic()-began,'background_thread_joined':True,'forbidden_modules':forbidden,'passed':cycles>0 and not forbidden},f,indent=2)
        assert cycles>0 and not forbidden

if __name__=='__main__':main()
