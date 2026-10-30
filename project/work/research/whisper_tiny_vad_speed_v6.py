"""Measure the speech detector separately, with exact state isolation checks."""
import pathlib,json,time,statistics,os
import whisper_tiny_gpu_batch_v1 as common
AREA=common.AREA;ROOT=common.ROOT
def main():
    guard=common.Guard();memory=guard.check();assert memory['commit_ratio']<.9 and memory['commit_headroom_bytes']>=8_000_000_000
    import torch
    guard.torch=torch;torch.set_num_threads(1);torch.set_num_interop_threads(1)
    pin=json.loads((AREA/'VAD_model_pin.json').read_text());path=pin['files']['src/silero_vad/data/silero_vad.jit']['path'];model=torch.jit.load(path,map_location='cpu').eval();clips,_=common.load_audio()
    print('METHODS '+json.dumps(model._c._method_names()),flush=True)
    length=max(len(a) for _,a in clips);padded=((length+511)//512)*512;buf=torch.zeros((32,padded))
    for i in range(32):a=clips[i%len(clips)][1];buf[i,:len(a)].copy_(torch.from_numpy(a))
    def python_loop(m,x):
        m.reset_states();out=[]
        for i in range(0,x.shape[-1],512):out.append(m(x[:,i:i+512],16000).flatten())
        return torch.stack(out,dim=1)
    results=[];reference=None;fast_available=hasattr(model,'audio_forward')
    with torch.inference_mode():
        reference=python_loop(model,buf[:8])
        if fast_available:
            fast=model.audio_forward(buf[:8],16000);assert fast.shape==reference.shape
            assert (fast-reference).abs().max().item()<1e-6
        for threads in [1,2,4]:
            torch.set_num_threads(threads)
            for b in [1,8,16,32]:
                fn=(lambda:model.audio_forward(buf[:b],16000)) if fast_available else (lambda:python_loop(model,buf[:b]))
                fn();times=[]
                for _ in range(3):
                    guard.check();start=time.perf_counter();out=fn();times.append(time.perf_counter()-start)
                v={'device':'cpu','threads':threads,'batch':b,'wall_s':statistics.median(times),'clips_per_s':b/statistics.median(times),'audio_forward':fast_available,'padded_audio_s_per_clip':padded/16000};results.append(v);print('VAD '+json.dumps(v),flush=True)
        torch.set_num_threads(1);gpumodel=model.to('cuda');gpu=buf.to('cuda');leasep=ROOT/'work/gpu_training_owner.json';lease=json.loads(leasep.read_text());assert not lease.get('actual_GPU_running');lease.update(actual_GPU_running=True,status='ROOT_VAD_GPU_speed_check',actual_owned_worker_pid=os.getpid());leasep.write_text(json.dumps(lease,indent=2)+'\n')
        for b in [1,8,16,32]:
            fn=(lambda:gpumodel.audio_forward(gpu[:b],16000)) if fast_available else (lambda:python_loop(gpumodel,gpu[:b]));fn();torch.cuda.synchronize();times=[]
            for _ in range(3):
                guard.check();start=time.perf_counter();out=fn();torch.cuda.synchronize();times.append(time.perf_counter()-start)
            if b==8:
                diff=(out.cpu()-reference).abs().max().item();assert diff<1e-4,diff
            v={'device':'cuda','threads':1,'batch':b,'wall_s':statistics.median(times),'clips_per_s':b/statistics.median(times),'audio_forward':fast_available,'padded_audio_s_per_clip':padded/16000};results.append(v);print('VAD '+json.dumps(v),flush=True)
    common.write('vad_v6_report.json',dict(at_utc=common.now(),results=results,memory=guard.check(),GPU_vs_CPU_probability_max_error=diff));guard.stop.set();lease.update(actual_GPU_running=False,status='ROOT_VAD_speed_check_closed',actual_owned_worker_pid=None);leasep.write_text(json.dumps(lease,indent=2)+'\n');print('COMPLETE vad_v6',flush=True)
if __name__=='__main__':
    try:main()
    except Exception as e:common.write('vad_v6_failure.json',{'error':repr(e),'at_utc':common.now(),'pid':os.getpid()});raise
