"""Equivalent fused-QKV encoder and one-token decoder; benchmark against HF."""
import os,pathlib,json,time,statistics,gc,contextlib
import whisper_tiny_gpu_batch_v1 as common
AREA=common.AREA;ROOT=common.ROOT;write=common.write

class FusedLanguageModel:
    def __init__(self,model,language_ids):
        import torch
        self.torch=torch;self.model=model;self.enc=model.model.encoder;self.dec=model.model.decoder
        self.enc_qkv=[];self.ids=language_ids
        for layer in self.enc.layers:
            a=layer.self_attn;zero=torch.zeros_like(a.q_proj.bias)
            self.enc_qkv.append((torch.cat([a.q_proj.weight,a.k_proj.weight,a.v_proj.weight]),torch.cat([a.q_proj.bias,zero,a.v_proj.bias])))
        self.cross_w=torch.cat([x for layer in self.dec.layers for x in [layer.encoder_attn.k_proj.weight,layer.encoder_attn.v_proj.weight]])
        self.cross_b=torch.cat([x for layer in self.dec.layers for x in [torch.zeros_like(layer.encoder_attn.v_proj.bias),layer.encoder_attn.v_proj.bias]])
        self.start=(self.dec.embed_tokens.weight[model.generation_config.decoder_start_token_id]+self.dec.embed_positions.weight[0]).detach()
        self.lang_w=model.proj_out.weight.index_select(0,language_ids).detach()
    def __call__(self,features):
        t=self.torch;F=t.nn.functional;e=self.enc;d=self.dec;b=features.shape[0];width=e.config.d_model;heads=e.config.encoder_attention_heads;dim=width//heads
        h=F.gelu(e.conv1(features));h=F.gelu(e.conv2(h)).transpose(1,2)+e.embed_positions.weight
        for layer,(weight,bias) in zip(e.layers,self.enc_qkv):
            residual=h;z=layer.self_attn_layer_norm(h);q,k,v=F.linear(z,weight,bias).chunk(3,dim=-1)
            q=q.reshape(b,-1,heads,dim).transpose(1,2);k=k.reshape(b,-1,heads,dim).transpose(1,2);v=v.reshape(b,-1,heads,dim).transpose(1,2)
            z=F.scaled_dot_product_attention(q,k,v,dropout_p=0.0).transpose(1,2).reshape(b,-1,width)
            h=residual+layer.self_attn.out_proj(z);h=h+layer.fc2(F.gelu(layer.fc1(layer.final_layer_norm(h))))
        h=e.layer_norm(h);cross=F.linear(h,self.cross_w,self.cross_b).split(width,dim=-1)
        z=self.start.reshape(1,1,-1).expand(b,1,-1)
        for i,layer in enumerate(d.layers):
            # With exactly one start token, self-attention softmax has one element=1.
            z=z+layer.self_attn.out_proj(layer.self_attn.v_proj(layer.self_attn_layer_norm(z)))
            residual=z;query=layer.encoder_attn.q_proj(layer.encoder_attn_layer_norm(z));q=query.reshape(b,1,heads,dim).transpose(1,2)
            k=cross[2*i].reshape(b,-1,heads,dim).transpose(1,2);v=cross[2*i+1].reshape(b,-1,heads,dim).transpose(1,2)
            a=F.scaled_dot_product_attention(q,k,v,dropout_p=0.).transpose(1,2).reshape(b,1,width)
            z=residual+layer.encoder_attn.out_proj(a);z=z+layer.fc2(F.gelu(layer.fc1(layer.final_layer_norm(z))))
        return F.linear(d.layer_norm(z[:,0]),self.lang_w).float().softmax(-1)

def main():
    import numpy as np,torch
    from transformers import WhisperForConditionalGeneration,WhisperFeatureExtractor
    from torch.nn.attention import sdpa_kernel,SDPBackend
    guard=common.Guard();m=guard.check();assert m['commit_ratio']<.9 and m['commit_headroom_bytes']>8_000_000_000
    guard.torch=torch;torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    receipt=json.loads((AREA/'HF_download_receipt.json').read_text());folder=receipt['model_directory']
    model=WhisperForConditionalGeneration.from_pretrained(folder,local_files_only=True,dtype=torch.float16,attn_implementation='sdpa',low_cpu_mem_usage=True).eval().to('cuda')
    ext=WhisperFeatureExtractor.from_pretrained(folder,local_files_only=True)
    ids=torch.tensor(list(model.generation_config.lang_to_id.values()),device='cuda');sot=model.generation_config.decoder_start_token_id
    leasep=ROOT/'work/gpu_training_owner.json';lease=json.loads(leasep.read_text());assert not lease.get('actual_GPU_running');lease.update(actual_GPU_running=True,status='ROOT_tiny_fused_benchmark_v3',actual_owned_worker_pid=os.getpid());leasep.write_text(json.dumps(lease,indent=2)+'\n')
    clips,_=common.load_audio();waves=[]
    for _,a in clips:
        for i in range(0,len(a),64000):
            if len(a[i:i+64000])>=4000:waves.append(a[i:i+64000])
    features=ext(waves,sampling_rate=16000,return_tensors='pt').input_features.to('cuda',dtype=torch.float16)
    fused=FusedLanguageModel(model,ids)
    def hf(x):
        inputs=torch.full((len(x),1),sot,device='cuda',dtype=torch.long)
        h=model.model(input_features=x,decoder_input_ids=inputs,use_cache=False).last_hidden_state[:,0]
        return torch.nn.functional.linear(h,model.proj_out.weight.index_select(0,ids)).float().softmax(-1)
    def measure(fn,b):
        for _ in range(4):fn()
        torch.cuda.synchronize();times=[]
        for _ in range(3):
            guard.check();start=time.perf_counter()
            for i in range(40):out=fn()
            torch.cuda.synchronize();times.append((time.perf_counter()-start)/40)
        return {'batch':b,'latency_s':statistics.median(times),'windows_per_second':b/statistics.median(times)}
    results=[]
    with torch.inference_mode():
        reference=torch.cat([hf(features[i:i+8]) for i in range(0,len(features),8)])
        candidate=torch.cat([fused(features[i:i+8]) for i in range(0,len(features),8)])
        verification={'windows':len(features),'max_probability_error_vs_HF':(candidate-reference).abs().max().item(),'language_agreement':(candidate.argmax(-1)==reference.argmax(-1)).float().mean().item(),'TR70_category_changes':int(((candidate[:,list(model.generation_config.lang_to_id).index('<|tr|>')]>=.7)!=(reference[:,list(model.generation_config.lang_to_id).index('<|tr|>')]>=.7)).sum())}
        write('fused_v3_verification.json',verification);print('VERIFY '+json.dumps(verification),flush=True)
        assert verification['language_agreement']==1 and verification['max_probability_error_vs_HF']<.02,verification
        for backend in ['default','FLASH_ATTENTION','CUDNN_ATTENTION','EFFICIENT_ATTENTION']:
            ctx=contextlib.nullcontext if backend=='default' else lambda:sdpa_kernel(getattr(SDPBackend,backend))
            for b in [1,4,8,16]:
                x=features[torch.arange(b,device='cuda')%len(features)].contiguous()
                try:
                    with ctx():
                        v=measure(lambda:fused(x),b);p=fused(x);ref=reference[torch.arange(b,device='cuda')%len(features)]
                        v.update(backend=backend,mode='fused_eager',max_probability_error=(p-ref).abs().max().item(),language_agreement=(p.argmax(-1)==ref.argmax(-1)).float().mean().item());results.append(v)
                        if backend=='default':
                            side=torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream())
                            with torch.cuda.stream(side):
                                for _ in range(3):fused(x)
                            torch.cuda.current_stream().wait_stream(side);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
                            with torch.cuda.graph(g):out=fused(x)
                            def replay():g.replay();return out
                            w=measure(replay,b);w.update(backend=backend,mode='fused_CUDA_graph',memory=guard.check());results.append(w);del g,out;gc.collect();torch.cuda.empty_cache()
                except RuntimeError as e:
                    results.append({'backend':backend,'batch':b,'supported':False,'error':str(e)[:500]});print('BACKEND_UNAVAILABLE '+backend+' '+str(e)[:200],flush=True);break
                write('fused_v3_partial.json',{'verification':verification,'results':results});print('RESULT '+json.dumps(results[-2:] if backend=='default' else results[-1:]),flush=True)
        write('fused_v3_report.json',dict(at_utc=common.now(),verification=verification,results=results,memory=guard.check(),full_model_weights_and_3000frame_context_preserved=True))
    guard.stop.set();del model;torch.cuda.empty_cache();lease.update(actual_GPU_running=False,status='ROOT_tiny_fused_benchmark_closed',actual_owned_worker_pid=None);leasep.write_text(json.dumps(lease,indent=2)+'\n');print('COMPLETE fused_v3',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:write('fused_v3_failure.json',dict(at_utc=common.now(),error=repr(e),pid=os.getpid()));raise
