"""Optional final runtime comparison, admitted only after prior GPU closure."""
import os,sys,pathlib,json,time,hashlib,argparse,statistics,gc
import whisper_tiny_gpu_batch_v1 as common
AREA=common.AREA;ROOT=common.ROOT
RUNTIME=pathlib.Path(r'E:\gpu_training_scratchpad\scratch\CPU_validation_v2_env\Lib\site-packages')
MODEL=pathlib.Path(r'E:\gpu_training_scratchpad\models\whisper-tiny-CT2-fp16-169d4a-v1')

class Guard(common.Guard):
    def snapshot(self):
        m=super().snapshot()
        if self.torch and self.torch.cuda.is_initialized():
            free,total=self.torch.cuda.mem_get_info();used=total-free
            # Includes other display allocations: conservative total-device8GB bound.
            m['cuda_reserved_bytes']=max(m['cuda_reserved_bytes'],used);self.peak_cuda=max(self.peak_cuda,used);m['peak_cuda_reserved_bytes']=self.peak_cuda
        return m

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--convert',action='store_true');args=ap.parse_args();guard=Guard();m=guard.check();assert m['commit_ratio']<.9 and m['commit_headroom_bytes']>=8_000_000_000
    leasep=ROOT/'work/gpu_training_owner.json';lease=json.loads(leasep.read_text());assert lease['owner']=='/root' and not lease.get('actual_GPU_running'),'Existing GPU owner is active'
    if args.convert:os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1);dll_directory_handle=os.add_dll_directory(str(pathlib.Path(torch.__file__).parent/'lib'));sys.path.append(str(RUNTIME));import ctranslate2 as ct2
    pin=json.loads((AREA/'HF_download_receipt.json').read_text());folder=pin['model_directory']
    if args.convert:
        assert not MODEL.exists();started=time.perf_counter();ct2.converters.TransformersConverter(folder,load_as_float16=True,low_cpu_mem_usage=True).convert(str(MODEL),quantization='float16')
        files={p.name:dict(bytes=p.stat().st_size,SHA256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()) for p in MODEL.iterdir() if p.is_file()};guard.check();common.write('CT2_conversion_v12.json',dict(at_utc=common.now(),directory=str(MODEL),files=files,source=pin['repo'],revision=pin['revision'],conversion_wall_s=time.perf_counter()-started,CT2_version=ct2.__version__,CPU_only=True,quantization='float16,not integer',memory=guard.check()));guard.stop.set();print('CONVERSION_COMPLETE',flush=True);return
    assert (AREA/'CT2_conversion_v12.json').exists();guard.torch=torch;lease.update(status='ROOT_tiny_CT2_bounded_candidate',actual_GPU_running=True,actual_owned_worker_pid=os.getpid(),source=str(pathlib.Path(__file__)));leasep.write_text(json.dumps(lease,indent=2)+'\n');torch.cuda.init();guard.check()
    from transformers import WhisperFeatureExtractor
    import numpy as np
    extractor=WhisperFeatureExtractor.from_pretrained(folder,local_files_only=True);clips,_=common.load_audio();waves=[]
    for _,pcm in clips:
        for i in range(0,len(pcm),64000):
            if len(pcm[i:i+64000])>=4000:waves.append(pcm[i:i+64000])
    features=extractor(waves,sampling_rate=16000,return_tensors='np').input_features.astype(np.float32);reference=json.loads((AREA/'baseline_full_predictions.json').read_text());languages=list(reference[0]['probabilities']);reports=[]
    assert len(features)==len(reference)==126
    for flash in [False,True]:
        model=None
        try:
            model=ct2.models.Whisper(str(MODEL),device='cuda',compute_type='float16',inter_threads=1,intra_threads=2,flash_attention=flash);assert model.is_multilingual and model.n_mels==80
            def infer(x):return model.detect_language(ct2.StorageView.from_array(np.ascontiguousarray(x)))
            candidates=[]
            for at in range(0,len(features),8):guard.check();candidates.extend(infer(features[at:at+8]))
            probs=[{(k[2:-2] if k.startswith('<|') else k):float(v) for k,v in row} for row in candidates];maxerr=max(abs(p[k]-r['probabilities'][k]) for p,r in zip(probs,reference) for k in languages);labels=sum(max(p,key=p.get)!=max(r['probabilities'],key=r['probabilities'].get) for p,r in zip(probs,reference));results=[]
            for b in ([1,4,8,16,32,64] if flash else [1,4,8,16,32]):
                guard.check();x=features[np.arange(b)%len(features)].copy();infer(x);start=time.perf_counter();infer(x);single=time.perf_counter()-start;repeats=max(3,min(100,int(1/max(single,.0001))));times=[]
                for _ in range(3):
                    start=time.perf_counter()
                    for i in range(repeats):infer(x)
                    times.append((time.perf_counter()-start)/repeats);guard.check()
                result=dict(batch=b,windows_per_s=b/statistics.median(times),batch_latency_s=statistics.median(times),memory=guard.check());results.append(result);print('CT2_RESULT '+json.dumps(dict(flash=flash,**result)),flush=True)
                common.write('CT2_candidate_v12_partial.json',dict(completed_backends=reports,current_backend_flash_attention=flash,current_backend_results=results))
            reports.append(dict(flash_attention=flash,max_probability_error_vs_HF_FP16=maxerr,top_language_changes_vs_HF_FP16=labels,verification_windows=len(probs),results=results))
        except Exception as e:
            guard.check()
            if guard.reason or any(x in str(e).lower() for x in ['out of memory','out_of_memory','allocation failed','bad_alloc']):raise
            reports.append(dict(flash_attention=flash,supported=False,error=repr(e)));print('CT2_CANDIDATE_UNAVAILABLE '+repr(e),flush=True)
        finally:
            if model:model.unload_model()
            del model;gc.collect();torch.cuda.empty_cache()
    common.write('CT2_candidate_v12_report.json',dict(at_utc=common.now(),reports=reports,model=str(MODEL),CT2_version=ct2.__version__,GPU=torch.cuda.get_device_name(),full_encoder_frames=3000,model_source=pin['repo'],revision=pin['revision'],not_adopted_automatically=True,memory=guard.check()));guard.stop.set();lease.update(status='ROOT_tiny_CT2_candidate_closed',actual_GPU_running=False,actual_owned_worker_pid=None);leasep.write_text(json.dumps(lease,indent=2)+'\n');print('CT2_COMPLETE',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:common.write('CT2_candidate_v12_failure.json',dict(error=repr(e),pid=os.getpid(),at_utc=common.now()));raise
