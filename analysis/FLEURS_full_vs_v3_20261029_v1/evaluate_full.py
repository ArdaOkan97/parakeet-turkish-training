"""Full pinned FLEURS TEST, unchanged greedy decoding and project WER scorer.

Inference only. No optimizer, sampler, training/controller or checkpoint writes.
Uses the existing readonly frozen base and existing unmerged LoRA/BPE structure.
Frozen BF16 matrices are materialized once on GPU for inference; their bits and
forward functions are unchanged. Every utterance is processed at full length.
"""
import argparse, gc, hashlib, io, json, math, mmap, os, shutil, sys, time, traceback
from pathlib import Path
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20')
S=P/'outputs/parakeet_turkish'
sys.path.insert(0,str(S));sys.path.insert(0,str(P/'work/research'))
R=Path(__file__).parent
from paired_windows_commit_telemetry_v1 import paired_sample
from resource_watchdog_v2 import sample_memory

def write(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.writing')
    with temporary.open('w',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while b:=f.read(4*1024**2): h.update(b)
    return h.hexdigest()

def main(args):
    began=time.monotonic();out=R/args.model;out.mkdir(exist_ok=False)
    os_priority=None
    import psutil
    process=psutil.Process();process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    samples=[paired_sample() for _ in range(3)]
    write(out/'admission.json',samples) # Numeric evidence before predicates.
    assert all(s['global_memory_status_ex']['ullAvailPageFile_bytes']>=8066304384 and s['get_performance_info_after']['physical_available_bytes']>=5316304384 for s in samples),'Original CPU/Reader prospective floors'
    assert shutil.disk_usage(R).free>=100000000000
    inventory=json.loads((R/'input_inventory.json').read_text())
    write(out/'scope.json',{'mode':args.model,'full_test_rows':{k:v['rows'] for k,v in inventory['datasets'].items()},'revision':inventory['revision'],'checkpoint_update':400000 if args.model=='adapted' else None,'whole_private_limit':5450000000,'CUDA_allocator_limit':8000000000,'minimum_commit':3509949440,'minimum_physical':3009949440,'per_active_operation_seconds':180,'whole_worker_seconds':7200,'decode':'serial greedy default generation config; no test-derived tuning','source':{'path':str(Path(__file__)),'sha256':sha(__file__)}})
    deadline=time.monotonic()+180
    last_memory=None
    def guard():
        nonlocal last_memory
        assert time.monotonic()-began<7200 and time.monotonic()<deadline,'Finite worker/active operation wall budget'
        value=sample_memory();last_memory=value
        s=value['system_memory'];w=value['worker_private_memory']
        assert s['available'] and w['available']
        assert s['commit_available_bytes']>=3509949440 and s['physical_available_bytes']>=3009949440 and s['commit_fraction']<.95,'Original runtime floors'
        assert w['private_bytes']<=5450000000,'Original optimizer-free whole-worker private forecast'
        assert shutil.disk_usage(R).free>=100000000000
    guard()
    from preload_nemo_linker_v1 import preload
    coherent=preload()
    import torch
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=False
    total=torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(8000000000/total,0)
    assert torch.cuda.mem_get_info()[0]>=2000000000
    from transformers import AutoProcessor
    from stream_frozen_encoder_matrices_v1 import load
    from streamed_generation_device_v1 import apply
    base=Path((P/'work/base_model_path.txt').read_text().strip())
    processor=AutoProcessor.from_pretrained(base,local_files_only=True)
    model,loader=load(base);logical=apply(model)
    structure=restoration=None
    if args.model=='adapted':
        from asr18_optimizer_free_DEV_structure_9223_v1 import prepare_structure,expected_names
        structure=prepare_structure(model,processor,torch=torch,guard=guard,admit=lambda name,budget:guard())
        from asr18_DEV_weight_only_mapper_9223_v1 import selection_plan
        document=json.loads((R/'checkpoint_metadata_RAW.json').read_bytes())
        assert sha(R/'checkpoint_metadata_RAW.json')==inventory['checkpoint_metadata_sha256']
        plan=selection_plan(document)
        assert plan['value']['optimizer']['updates']==400000
        weights=plan['value']['weights'];assert set(weights)==set(expected_names(model)) and len(weights)==207
        payload=Path(document['payload']['path'])
        assert payload.parent==Path(r'E:\gpu_training_scratchpad\checkpoints\BPE9223_resident_RAM_20261018_01_protected_DEV_v1\update_000400000\original')
        assert payload.stat().st_size==document['payload']['bytes'] and sha(payload)==document['payload']['sha256']
        actual=dict(model.named_parameters());verified={}
        with payload.open('rb') as f:
            mapping=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)
            try:
                with torch.no_grad():
                    for name,index in weights.items():
                        guard();entry=document['tensors'][index]
                        assert entry['dtype']=='float32' and list(actual[name].shape)==entry['shape']
                        tensor=torch.frombuffer(mapping,dtype=torch.float32,count=math.prod(entry['shape']),offset=entry['offset']).reshape(entry['shape'])
                        assert hashlib.sha256(mapping[entry['offset']:entry['offset']+entry['bytes']]).hexdigest()==entry['sha256']
                        actual[name].copy_(tensor)
                        assert torch.equal(actual[name].detach().cpu().view(torch.uint8),tensor.view(torch.uint8))
                        verified[name]=entry['sha256'];del tensor
            finally: mapping.close()
        model._ASR18_DEV_weights_restored=True
        restoration={'update':400000,'verified_weights':verified,'payload':document['payload'],'Adam_tensor_views':0,'optimizer_objects':0}
        from tdt_joint_canonical_2d_fp32_9223_v1 import apply as canonical
        canonical(model.joint)
    model.requires_grad_(False);model.eval()
    # Placement-only change avoids retransferring 1.217GB frozen weights per clip.
    # All preexisting CPU immutable backing remains alive; no quantization/merge.
    for handle in model._diagnostic_frozen_handles:handle.remove()
    resident_bytes=0
    for name,value in model._diagnostic_frozen_backing.items():
        guard();module_name,key=name.rsplit('.',1);module=model.get_submodule(module_name)
        parameter=module._parameters[key]
        assert parameter.data_ptr()==value.data_ptr() and value.dtype==torch.bfloat16
        module._parameters[key]=torch.nn.Parameter(value.to('cuda'),requires_grad=False)
        resident_bytes+=value.numel()*value.element_size()
    # The production DEV route uses this conv tiling; use the same on both sides.
    from asr18_temporal_tiled_subsampling_v2 import apply as tiled
    tiling=tiled(model,torch=torch,guard=guard)
    assert all(p.device.type=='cuda' for p in model.parameters())
    write(out/'model_ready.json',{'loader':loader,'logical_device':logical,'structure':structure,'restoration':restoration,'resident_frozen_bytes':resident_bytes,'tiling':tiling,'torch':torch.__version__,'CUDA':torch.version.cuda,'coherent':coherent,'memory':last_memory,'vocabulary':len(processor.tokenizer)})
    print(json.dumps({'event':'model_ready','model':args.model,'seconds':time.monotonic()-began}),flush=True)
    import soundfile as sf
    import numpy as np
    import pyarrow.parquet as pq
    from metrics import normalize_text,score_transcriptions
    def infer(audio,sr):
        assert sr==16000 and audio.ndim==1 and len(audio)>0
        inputs=processor(audio=[audio],sampling_rate=sr,return_tensors='pt')
        inputs={k:v.to('cuda') for k,v in inputs.items()}
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            generated=model.generate(**inputs,do_sample=False,num_beams=1)
        result=processor.decode(generated.sequences[0],skip_special_tokens=True)
        del generated,inputs
        return result
    if args.model=='adapted':
        deadline=time.monotonic()+180;dev=[]
        for line in (P/'work/data/validation_tr.jsonl').read_text(encoding='utf-8').splitlines():
            row=json.loads(line);guard()
            audio,sr=sf.read(row['audio_filepath'],dtype='float32')
            dev.append({'source_id':row['source_id'],'reference':row.get('reference',row.get('text')),'hypothesis':infer(audio,sr),'language':'tr'})
        score=score_transcriptions(dev,language='tr')
        write(out/'Turkish_DEV_loader_check.json',{'score':score,'predictions':dev,'expected_word_errors':174,'expected_reference_words':825})
        print(json.dumps({'event':'DEV_check','wer':score['wer'],'word_errors':score['word_errors'],'reference_words':score['reference_words']}),flush=True)
        assert score['word_errors']==174 and score['reference_words']==825,'Checkpoint inference must reproduce known DEV174/825 before TEST'
    summary={}
    for language,item in inventory['datasets'].items():
        rows=[];seen=set();audio_seconds=0;language_start=time.monotonic()
        destination=out/(language+'_test_predictions.jsonl')
        with destination.open('x',encoding='utf-8') as predictions:
            pf=pq.ParquetFile(item['path'])
            for index,batch in enumerate(pf.iter_batches(batch_size=1,use_threads=False)):
                deadline=time.monotonic()+180;guard();record=batch.to_pylist()[0]
                source_id=f"google/fleurs:{language}:test:{record['id']}:{record['path']}"
                assert source_id not in seen;seen.add(source_id)
                audio_bytes=record['audio']['bytes'];assert audio_bytes
                audio,sr=sf.read(io.BytesIO(audio_bytes),dtype='float32')
                assert len(audio)==record['num_samples']
                reference=record['raw_transcription'];assert isinstance(reference,str) and normalize_text(reference,language)
                hypothesis=infer(audio,sr);seconds=len(audio)/sr;audio_seconds+=seconds
                row={'source_id':source_id,'sentence_id':record['id'],'language':language,'split':'test','reference':reference,'hypothesis':hypothesis,'audio_sha256':hashlib.sha256(audio_bytes).hexdigest(),'duration_seconds':seconds,'dataset_revision':inventory['revision'],'row_index':index}
                predictions.write(json.dumps(row,ensure_ascii=False)+'\n');predictions.flush()
                rows.append(row)
                if (index+1)%25==0:
                    guard();status={'model':args.model,'language':language,'completed':index+1,'total':item['rows'],'elapsed_seconds':time.monotonic()-language_start,'total_elapsed_seconds':time.monotonic()-began,'memory':last_memory,'CUDA_allocated':torch.cuda.memory_allocated(),'CUDA_peak':torch.cuda.max_memory_allocated()}
                    write(out/'status.json',status);print(json.dumps({k:v for k,v in status.items() if k!='memory'}),flush=True)
                del batch,record,audio,audio_bytes
            predictions.flush();os.fsync(predictions.fileno())
        assert len(rows)==item['rows'] and len(seen)==item['rows']
        score=score_transcriptions(rows,language=language);details=score.pop('details')
        write(out/(language+'_test_score.json'),{**score,'details':details,'audio_seconds':audio_seconds,'prediction_sha256':sha(destination),'full_official_split':True,'parquet_rows':item['rows']})
        summary[language]={**score,'audio_seconds':audio_seconds,'seconds':time.monotonic()-language_start}
        write(out/'summary.json',summary);print(json.dumps({'event':'language_complete','model':args.model,'language':language,**summary[language]}),flush=True)
        del rows,details,pf;gc.collect();torch.cuda.empty_cache()
    write(out/'complete.json',{'summary':summary,'wall_seconds':time.monotonic()-began,'all_full_TEST_rows':sum(x['utterances'] for x in summary.values()),'source_sha256':sha(__file__),'CUDA_peak':torch.cuda.max_memory_allocated(),'maximum_wall_seconds':7200})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--model',choices=['v3','adapted'],required=True)
    args=parser.parse_args()
    try:main(args)
    except BaseException as error:
        directory=R/args.model
        if directory.exists():write(directory/'failure.json',{'exception':str(error),'traceback':traceback.format_exc()})
        raise
