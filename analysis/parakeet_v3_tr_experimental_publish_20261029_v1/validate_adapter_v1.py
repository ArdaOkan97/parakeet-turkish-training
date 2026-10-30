"""Verify public unmerged loader and compare native merged weights/transcripts."""
import gc,hashlib,json,os,shutil,sys,threading,time,traceback
from pathlib import Path
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');R=Path(__file__).parent
F=Path(r'E:\gpu_training_scratchpad\analysis\FLEURS_full_vs_v3_20261029_v1')
A=Path(r'E:\gpu_training_scratchpad\exports\parakeet-tdt-0.6b-v3-tr-experimental-400000-20261029-v3\adapter')
sys.path.insert(0,str(P/'outputs/parakeet_turkish'));sys.path.insert(0,str(P/'work/research'));sys.path.insert(0,str(A))
from paired_windows_commit_telemetry_v1 import paired_sample
from resource_watchdog_v2 import sample_memory
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
def main():
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    out=R/'adapter_validation_v1';out.mkdir(exist_ok=False);start=time.monotonic()
    samples=[paired_sample(),paired_sample()];write(out/'admission.json',samples)
    assert all(s['global_memory_status_ex']['ullAvailPageFile_bytes']>=8066304384 and s['get_performance_info_after']['physical_available_bytes']>=5316304384 for s in samples)
    write(out/'scope.json',{'inputs':'Public adapter directory, pinned original native base, new native inference fingerprints and matched native validation predictions','tensors':'207 native LoRA/full-head tensors; no original training payload reread','CPU_runtime':[5547868928,5047868928],'GPU_runtime':[3509949440,3009949440],'private8GB':True,'CUDA8GB':True,'active_wall':180,'total_wall':1800,'test':'All 50 Turkish DEV clips plus first 10 TEST clips per tracked language; no new full original test'})
    deadline=time.monotonic()+180;last={};gpu_phase=False;stop=threading.Event();peak=0
    def guard():
        nonlocal last,peak
        last=sample_memory();s=last['system_memory'];w=last['worker_private_memory'];peak=max(peak,w['private_bytes'])
        assert time.monotonic()-start<1800 and time.monotonic()<deadline
        floor=(3509949440,3009949440) if gpu_phase else (5547868928,5047868928)
        assert s['commit_available_bytes']>=floor[0] and s['physical_available_bytes']>=floor[1] and s['commit_fraction']<.95
        assert w['private_bytes']<=8000000000 and shutil.disk_usage(R).free>=100000000000
    def watch():
        while not stop.wait(.25):
            try:guard()
            except BaseException as e:write(out/'guard_failure.json',{'error':str(e),'memory':last});os._exit(23)
    thread=threading.Thread(target=watch,daemon=True);thread.start()
    from preload_nemo_linker_v1 import preload
    preload()
    import torch
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    from load_adapter import load_model,LoRALinear
    base=P/'work/models/hf_native_cache/models--nvidia--parakeet-tdt-0.6b-v3/snapshots/541d1f99c6b0c3cd0b11a95167540bb8edefd82b/parakeet-tdt-0.6b-v3.nemo'
    model=load_model(A,base_nemo=base,cache_dir=str(out/'base_cache'))
    expected=json.loads((R/'native_validation_v8/merged_parameter_fingerprints.json').read_text());matched=[];failures=[]
    state=model.state_dict()
    with torch.no_grad():
        for name,item in expected.items():
            guard();value=state[name]
            parent,kind=name.rsplit('.',1);module=model.get_submodule(parent)
            if isinstance(module,LoRALinear) and kind=='weight':value=value+module.scaling*(module.lora_B@module.lora_A)
            digest=hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            (matched if digest==item['sha256'] else failures).append(name)
    write(out/'merged_weight_equivalence.json',{'matched':matched,'mismatched':failures,'count':len(expected),'comparison':'CPU FP32 materialized B@A exactly hashed against the actual restored merged native archive'})
    assert not failures,failures
    del state;gc.collect()
    cfg=json.loads((A/'adapter_config.json').read_text());parameters=dict(model.named_parameters())
    for name in cfg['frozen_matrix_names']:parameters[name].data=parameters[name].data.to(torch.bfloat16)
    del parameters;gc.collect();gpu_phase=True;deadline=time.monotonic()+180;torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(8000000000/torch.cuda.get_device_properties(0).total_memory)
    assert torch.cuda.mem_get_info()[0]>=2000000000
    model=model.to('cuda');guard()
    from nemo.utils import logging
    logging.set_verbosity(logging.ERROR)
    from metrics import score_transcriptions
    def infer(path):
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):result=model.transcribe([str(path)],batch_size=1,num_workers=0,verbose=False,return_hypotheses=False)
        if isinstance(result,tuple):result=result[0]
        return result[0] if isinstance(result[0],str) else result[0].text
    dev=[]
    for line in (P/'work/data/validation_tr.jsonl').read_text(encoding='utf-8').splitlines():
        deadline=time.monotonic()+180;guard();row=json.loads(line)
        dev.append({'source_id':row['source_id'],'reference':row.get('reference',row.get('text')),'hypothesis':infer(row['audio_filepath']),'language':'tr'})
    write(out/'Turkish_DEV.json',{'score':score_transcriptions(dev,language='tr'),'predictions':dev})
    comparisons=[]
    for language in ['tr','en','es','it','fr','de','pt']:
        references=[json.loads(x) for x in (R/'native_validation_v8'/(language+'_predictions.jsonl')).read_text(encoding='utf-8').splitlines()[:10]]
        for row in references:
            deadline=time.monotonic()+180;guard();actual=infer(row['audio_filepath'])
            comparisons.append({'language':language,'row_index':row['row_index'],'merged':row['hypothesis'],'adapter':actual,'identical':actual==row['hypothesis']})
    write(out/'transcript_comparison.json',comparisons)
    write(out/'complete.json',{'passed':True,'merged_tensor_hash_matches':len(matched),'merged_tensor_hash_mismatches':0,'Turkish_DEV':{k:v for k,v in score_transcriptions(dev,language='tr').items() if k!='details'},'matched_TEST_clips':len(comparisons),'identical_TEST_transcripts':sum(x['identical'] for x in comparisons),'seconds':time.monotonic()-start,'peak_private':peak,'CUDA_peak':torch.cuda.max_memory_allocated()})
    print((out/'complete.json').read_text());stop.set();thread.join()
try:main()
except BaseException as e:write(R/'adapter_validation_v1/failure.json',{'error':str(e),'traceback':traceback.format_exc()});raise
