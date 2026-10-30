"""Fresh stock NeMo restoration and full matched FLEURS inference validation."""
import gc,hashlib,json,os,shutil,sys,threading,time,traceback
from pathlib import Path
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20')
R=Path(__file__).parent;F=Path(r'E:\gpu_training_scratchpad\analysis\FLEURS_full_vs_v3_20261029_v1')
sys.path.insert(0,str(P/'outputs/parakeet_turkish'));sys.path.insert(0,str(P/'work/research'))
from paired_windows_commit_telemetry_v1 import paired_sample
from resource_watchdog_v2 import sample_memory
def write(path,value):
    p=Path(path);tmp=p.with_suffix('.writing')
    with tmp.open('w',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while b:=f.read(4*1024**2):h.update(b)
    return h.hexdigest()
def main():
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    start=time.monotonic();out=R/'native_validation_v7';out.mkdir(exist_ok=False)
    samples=[paired_sample(),paired_sample()];write(out/'admission.json',samples)
    assert all(s['global_memory_status_ex']['ullAvailPageFile_bytes']>=8066304384 and s['get_performance_info_after']['physical_available_bytes']>=5316304384 for s in samples)
    write(out/'scope.json',{'source':'Standard native EncDecRNNTBPEModel.restore_from with mmap torch.load connector; identical archive/tensors, lower temporary CPU commitment','test':'Existing identical full FLEURS test WAVs and references','private_cap':8000000000,'CUDA_allocator_cap':8000000000,'active_wall':180,'whole_wall':7200,'CPU_runtime_floors':[5547868928,5047868928],'GPU_runtime_floors':[3509949440,3009949440],'decoding':'Native saved greedy_batch; CPU FP32 smoke, GPU BF16 frozen/merged matrices and autocast; no test tuning'})
    deadline=time.monotonic()+180;last={};stop=threading.Event();gpu_phase=False;peak_private=0
    def guard():
        nonlocal last,peak_private
        last=sample_memory();s=last['system_memory'];w=last['worker_private_memory'];peak_private=max(peak_private,w['private_bytes'])
        assert time.monotonic()-start<7200 and time.monotonic()<deadline,'Finite operation/worker wall limit'
        floors=(3509949440,3009949440) if gpu_phase else (5547868928,5047868928)
        assert s['commit_available_bytes']>=floors[0] and s['physical_available_bytes']>=floors[1] and s['commit_fraction']<.95,'Original runtime floors'
        assert w['private_bytes']<=8000000000,'Existing CPU/inference private8GB cap'
        assert shutil.disk_usage(R).free>=100000000000
    def watch():
        while not stop.wait(.25):
            try:guard()
            except BaseException as e:write(out/'guard_failure.json',{'error':str(e),'last':last});os._exit(23)
    thread=threading.Thread(target=watch,daemon=True);thread.start()
    from preload_nemo_linker_v1 import preload
    preload()
    import torch
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    from nemo.collections.asr.models import EncDecRNNTBPEModel
    target=Path(r'E:\gpu_training_scratchpad\exports\parakeet-tdt-0.6b-v3-tr-experimental-400000-20261029-v3\merged\parakeet-tdt-0.6b-v3-tr-experimental.nemo')
    assert sha(target)=='5c4f51ee50ea19bb598c4e8024c6741cf8513d57d55c5cae01287ae335d4ae25'
    from nemo.core.connectors.save_restore_connector import SaveRestoreConnector
    class MemoryMappedRestore(SaveRestoreConnector):
        @staticmethod
        def _load_state_dict_from_disk(model_weights, map_location='cpu'):
            return torch.load(model_weights,map_location=map_location,mmap=True,weights_only=False)
    import tarfile
    extracted=out/'extracted_native';extracted.mkdir()
    with tarfile.open(target,'r:') as archive:
        for member in archive.getmembers():
            if member.isdir() and member.name.rstrip('/') in ('.',''):continue
            assert member.isfile() and Path(member.name).name==member.name.lstrip('./') and len(Path(member.name).parts)<=1,member.name
        archive.extractall(extracted,filter='data')
    connector=MemoryMappedRestore();connector.model_extracted_dir=str(extracted)
    model=EncDecRNNTBPEModel.restore_from(str(target),map_location='cpu',save_restore_connector=connector);model.eval();model.requires_grad_(False)
    assert type(model) is EncDecRNNTBPEModel and model.decoder.blank_idx==9222
    selected={}
    for name,tensor in model.state_dict().items():
        if 'self_attn.linear_' in name or name.startswith(('decoder.','joint.')):
            selected[name]={'shape':list(tensor.shape),'sha256':hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()}
    write(out/'merged_parameter_fingerprints.json',selected)
    # Prove stock FP32 CPU transcription before optional mixed precision placement.
    cpu_row=json.loads((P/'work/data/validation_tr.jsonl').read_text(encoding='utf-8').splitlines()[0])
    deadline=time.monotonic()+180
    with torch.inference_mode():cpu_result=model.transcribe([cpu_row['audio_filepath']],batch_size=1,num_workers=0,verbose=False,return_hypotheses=False)
    write(out/'CPU_FP32_stock_inference.json',{'result':[x if isinstance(x,str) else x.text for x in (cpu_result[0] if isinstance(cpu_result,tuple) else cpu_result)],'source_id':cpu_row['source_id'],'memory':sample_memory()})
    del cpu_result
    cfg=json.loads(Path(r'E:\gpu_training_scratchpad\exports\parakeet-tdt-0.6b-v3-tr-experimental-400000-20261029-v3\adapter\adapter_config.json').read_text())
    parameters=dict(model.named_parameters())
    for name in cfg['frozen_matrix_names']:
        value=parameters[name];value.data=value.data.to(torch.bfloat16)
    del parameters,value;gc.collect()
    gpu_phase=True;deadline=time.monotonic()+180
    torch.cuda.set_per_process_memory_fraction(8000000000/torch.cuda.get_device_properties(0).total_memory)
    assert torch.cuda.mem_get_info()[0]>=2000000000
    model=model.to('cuda');gc.collect();guard()
    write(out/'restore_passed.json',{'stock_class':str(type(model)),'model_sha256':sha(target),'blank':model.decoder.blank_idx,'torch':torch.__version__,'nemo':'3.0.0','precision':'BF16 matrices and autocast, FP32 stored archive; CPU FP32 single-clip check separately','elapsed':time.monotonic()-start,'memory':last})
    from nemo.utils import logging
    logging.set_verbosity(logging.ERROR)
    from metrics import score_transcriptions
    def infer(path):
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):result=model.transcribe([str(path)],batch_size=1,num_workers=0,verbose=False,return_hypotheses=False)
        if isinstance(result,tuple):result=result[0]
        item=result[0]
        return item if isinstance(item,str) else item.text
    dev=[]
    for line in (P/'work/data/validation_tr.jsonl').read_text(encoding='utf-8').splitlines():
        deadline=time.monotonic()+180;guard();row=json.loads(line)
        dev.append({'source_id':row['source_id'],'reference':row.get('reference',row.get('text')),'hypothesis':infer(row['audio_filepath']),'language':'tr'})
    score=score_transcriptions(dev,language='tr');write(out/'Turkish_DEV.json',{'score':score,'predictions':dev})
    print(json.dumps({'event':'native_DEV','wer':score['wer'],'errors':score['word_errors'],'words':score['reference_words']}),flush=True)
    summary={};inventory=json.loads((F/'input_inventory.json').read_text())
    for language in ['tr','en','es','it','fr','de','pt']:
        deadline=time.monotonic()+180;guard();rows=[];lang_start=time.monotonic();folder=F/'full_test_data'/language
        preparation=json.loads((folder/'complete.json').read_text());assert sha(folder/'test.jsonl')==preparation['manifest_sha256']
        with (out/(language+'_predictions.jsonl')).open('x',encoding='utf-8') as f:
            for index,line in enumerate((folder/'test.jsonl').read_text(encoding='utf-8').splitlines()):
                deadline=time.monotonic()+180;guard();row=json.loads(line);assert sha(row['audio_filepath'])==row['audio_sha256']
                row['hypothesis']=infer(row['audio_filepath']);rows.append(row);f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
                if (index+1)%50==0:
                    write(out/'status.json',{'language':language,'completed':index+1,'total':preparation['utterances'],'elapsed':time.monotonic()-start,'memory':last,'CUDA_peak':torch.cuda.max_memory_allocated()})
                    print(json.dumps({'language':language,'completed':index+1,'elapsed':time.monotonic()-start}),flush=True)
            f.flush();os.fsync(f.fileno())
        assert len(rows)==inventory['datasets'][language]['rows']
        score=score_transcriptions(rows,language=language);write(out/(language+'_score.json'),score)
        summary[language]={k:v for k,v in score.items() if k!='details'};write(out/'summary.json',summary)
        print(json.dumps({'event':'language_complete','language':language,**summary[language]}),flush=True)
        del rows;gc.collect();torch.cuda.empty_cache()
    write(out/'complete.json',{'passed':True,'all_test_utterances':sum(v['utterances'] for v in summary.values()),'summary':summary,'elapsed':time.monotonic()-start,'peak_private':peak_private,'CUDA_peak':torch.cuda.max_memory_allocated(),'source_sha256':sha(__file__)})
    stop.set();thread.join()
try:main()
except BaseException as e:
    write(R/'native_validation_v7'/'failure.json',{'error':str(e),'traceback':traceback.format_exc()});raise
