"""Inference-only NeMo export of protected checkpoint400000 and unmerged207 adapter.

Original files remain read-only. NVIDIA/HF's installed conversion key map is
reused in reverse; all native model parameters must have an exact source.
"""
import ast,gc,hashlib,json,math,mmap,os,re,shutil,sys,threading,time,traceback
from pathlib import Path
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20')
R=Path(__file__).parent;S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
OUT=Path(r'E:\gpu_training_scratchpad\exports\parakeet-tdt-0.6b-v3-tr-experimental-400000-20261029-v1')
from paired_windows_commit_telemetry_v1 import paired_sample
from resource_watchdog_v2 import sample_memory
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
def main():
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    began=time.monotonic();samples=[paired_sample(),paired_sample()];write(R/'export_admission.json',samples)
    assert all(s['global_memory_status_ex']['ullAvailPageFile_bytes']>=8066304384 and s['get_performance_info_after']['physical_available_bytes']>=5316304384 for s in samples)
    write(R/'export_scope.json',{'source_checkpoint_metadata':str(R/'checkpoint_RAW.json'),'base':'Pinned original HF v3 and frozen294 bank','native_config':'Existing original NeMo config/tokenizer','outputs':str(OUT),'weights_only':True,'optimizer_state_exported':False,'private_limit':8000000000,'wall_seconds':180})
    assert not OUT.exists();OUT.mkdir();adapter=OUT/'adapter';merged=OUT/'merged';adapter.mkdir();merged.mkdir()
    last={};stopped=threading.Event()
    def guard():
        nonlocal last
        last=sample_memory();s=last['system_memory'];w=last['worker_private_memory']
        assert time.monotonic()-began<180,'Finite CPU export wall budget'
        assert s['commit_available_bytes']>=5547868928 and s['physical_available_bytes']>=5047868928,'Original CPU runtime floors'
        assert w['private_bytes']<=8000000000,'Original CPU private8GB guard'
        assert shutil.disk_usage(OUT).free>=100000000000
    def watch():
        while not stopped.wait(.25):
            try:guard()
            except BaseException as e:
                write(R/'export_guard_failure.json',{'error':str(e),'last':last});os._exit(23)
    thread=threading.Thread(target=watch,daemon=True);thread.start()
    def digest(path):
        h=hashlib.sha256()
        with Path(path).open('rb') as f:
            while block:=f.read(4*1024**2):guard();h.update(block)
        return h.hexdigest()
    from preload_nemo_linker_v1 import preload
    preload()
    import torch
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    from safetensors import safe_open
    from safetensors.torch import save_file
    from sentencepiece import sentencepiece_model_pb2 as spb
    import sentencepiece as spm
    from omegaconf import OmegaConf,open_dict
    from nemo.collections.asr.models import EncDecRNNTBPEModel
    from asr18_DEV_weight_only_mapper_9223_v1 import selection_plan
    metadata=json.loads((R/'checkpoint_RAW.json').read_bytes());plan=selection_plan(metadata)
    assert plan['value']['optimizer']['updates']==400000 and len(plan['value']['weights'])==207
    payload=Path(metadata['payload']['path']);assert digest(payload)==metadata['payload']['sha256']
    saved={}
    with payload.open('rb') as f:
        mapping=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)
        try:
            for name,index in plan['value']['weights'].items():
                guard();entry=metadata['tensors'][index];h=hashlib.sha256()
                for off in range(entry['offset'],entry['offset']+entry['bytes'],4*1024**2):h.update(mapping[off:min(off+4*1024**2,entry['offset']+entry['bytes'])])
                assert h.hexdigest()==entry['sha256'] and entry['dtype']=='float32'
                value=torch.frombuffer(mapping,dtype=torch.float32,count=math.prod(entry['shape']),offset=entry['offset']).reshape(entry['shape'])
                saved[name]=value.clone();del value
        finally:mapping.close()
    assert len(saved)==207 and not torch.count_nonzero(saved['decoder.embedding.weight'][8192]),'Original zero blank/SOS embedding'
    base=Path((P/'work/base_model_path.txt').read_text().strip())
    bank=json.loads(Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1\frozen_weight_bank_v1\report.json').read_text())
    assert digest(base/'model.safetensors')==bank['source']['sha256']
    frozen=set(bank['matrices']);assert len(frozen)==294
    tokenizer_path=Path(r'E:\gpu_training_scratchpad\analysis\Turkish_labels_BPE_circumflex_v1\corpus_driven_BPE9223_v2\tokenizer.json')
    assert digest(tokenizer_path)=='f2d80ffee7ed1b006a83b52cc77bfb28bed4ba850352b5222bb3a49d6cfbac4a'
    candidate=json.loads(tokenizer_path.read_text(encoding='utf-8'));by_id={i:p for p,i in candidate['model']['vocab'].items()}
    native_to_hf=list(range(8192))+list(range(8193,9223))+[8192]
    joint_rows=torch.tensor(native_to_hf+list(range(9223,9228)),dtype=torch.long)
    embedding_rows=torch.tensor(native_to_hf,dtype=torch.long)
    pieces=[by_id[i] for i in native_to_hf[:-1]];assert len(pieces)==9222 and len(set(pieces))==9222
    original=P/'work/research/nemo_reference/tokenizer_original_v1'
    proto=spb.ModelProto();proto.ParseFromString((original/'902810505b78491d95378fa3000d1bc6_tokenizer.model').read_bytes())
    assert len(proto.pieces)==8192 and [x.piece for x in proto.pieces]==pieces[:8192]
    lowest=min(x.score for x in proto.pieces)
    for i,piece in enumerate(pieces[8192:]):
        entry=proto.pieces.add();entry.piece=piece;entry.score=lowest-i-1;entry.type=spb.ModelProto.SentencePiece.NORMAL
    proto.trainer_spec.vocab_size=9222
    token_dir=adapter/'tokenizer';token_dir.mkdir()
    (token_dir/'tokenizer.model').write_bytes(proto.SerializeToString())
    (token_dir/'tokenizer.vocab').write_text(''.join(f'{p.piece}\t{p.score}\n' for p in proto.pieces),encoding='utf-8')
    (token_dir/'vocab.txt').write_text('\n'.join(pieces)+'\n',encoding='utf-8')
    shutil.copyfile(tokenizer_path,adapter/'hf_tokenizer.json')
    sp=spm.SentencePieceProcessor(model_file=str(token_dir/'tokenizer.model'))
    assert sp.get_piece_size()==9222 and [sp.id_to_piece(i) for i in range(9222)]==pieces
    cfg=OmegaConf.load(original/'model_config.yaml')
    with open_dict(cfg):
        cfg.tokenizer={'dir':str(token_dir),'type':'bpe','model_path':str(token_dir/'tokenizer.model'),'vocab_path':str(token_dir/'vocab.txt'),'spe_tokenizer_vocab':str(token_dir/'tokenizer.vocab')}
        cfg.decoder.vocab_size=9222;cfg.joint.num_classes=9222;cfg.joint.vocabulary=pieces
        cfg.train_ds=None;cfg.validation_ds=None;cfg.test_ds=None
        cfg.preprocessor.dither=0.0
        cfg.decoding.strategy='greedy_batch';cfg.decoding.greedy.use_cuda_graph_decoder=False
    print(json.dumps({'stage':'construct_native','elapsed':time.monotonic()-began}),flush=True)
    model=EncDecRNNTBPEModel(cfg=cfg,trainer=None);model.eval();model.requires_grad_(False)
    converter=P/'work/training-env/Lib/site-packages/transformers/models/parakeet/convert_nemo_to_hf.py'
    tree=ast.parse(converter.read_text(encoding='utf-8'));key_map={}
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('NEMO_TO_HF_WEIGHT_MAPPING','NEMO_TDT_WEIGHT_MAPPING') for t in node.targets):key_map.update(ast.literal_eval(node.value))
    def hf_key(key):
        for pattern,replacement in key_map.items():key=re.sub(pattern,replacement,key)
        return key
    def native_key(hf):
        for name in model.state_dict():
            if hf_key(name)==hf:return name
        raise KeyError(hf)
    native_adapter={};precision_names=[];copied=[];merged_count=0
    source_to_native={hf_key(k):k for k in model.state_dict()}
    for name,value in saved.items():
        if name.endswith(('.lora_A','.lora_B')):
            parent,suffix=name.rsplit('.',1);target=source_to_native[parent+'.weight'].rsplit('.',1)[0]+'.'+suffix
            native_adapter[target]=value.clone()
        else:
            target=source_to_native[name]
            if name=='decoder.embedding.weight':value=value[embedding_rows]
            elif name in ('joint.head.weight','joint.head.bias'):value=value[joint_rows]
            native_adapter[target]=value.contiguous().clone()
    assert len(native_adapter)==207
    with safe_open(str(base/'model.safetensors'),framework='pt',device='cpu') as source,torch.no_grad():
        keys=set(source.keys())
        for name,destination in model.state_dict().items():
            guard()
            if name.endswith(('featurizer.window','featurizer.fb')):continue
            hf=hf_key(name);assert hf in keys or hf in saved,('No original source',name,hf)
            if hf in saved:
                value=saved[hf]
                if hf=='decoder.embedding.weight':value=value[embedding_rows]
                elif hf in ('joint.head.weight','joint.head.bias'):value=value[joint_rows]
            else:
                value=source.get_tensor(hf)
                if hf in frozen:value=value.to(torch.bfloat16).to(torch.float32);precision_names.append(name)
                if hf.endswith('.weight') and hf[:-7]+'.lora_A' in saved:
                    value=value+2.0*(saved[hf[:-7]+'.lora_B']@saved[hf[:-7]+'.lora_A']);merged_count+=1
            assert destination.shape==value.shape,(name,destination.shape,value.shape)
            destination.copy_(value);copied.append(name);del value
    assert len(precision_names)==294 and merged_count==96
    assert set(dict(model.named_parameters())).issubset(copied)
    save_file(native_adapter,str(adapter/'adapter_model.safetensors'),metadata={'format':'nemo-unmerged-lora8-alpha16-plus-trained-decoder-joint','checkpoint':'400000'})
    adapter_config={'format':'parakeet_nemo_lora_and_full_heads_v1','base_model':'nvidia/parakeet-tdt-0.6b-v3','base_revision':'541d1f99c6b0c3cd0b11a95167540bb8edefd82b','rank':8,'alpha':16,'dropout':0,'checkpoint_update':400000,'adapter_tensor_count':192,'full_head_tensor_count':15,'native_vocabulary':9222,'native_blank_id':9222,'native_duration_ids':[9223,9224,9225,9226,9227],'original_HF_blank_id':8192,'native_to_original_HF_rows':native_to_hf,'frozen_matrix_names':precision_names,'tokenizer_encoding_note':'Same output pieces/IDs after blank relocation. SentencePiece segmentation is not a claim of exact HF TRAIN tokenization. Inference export, not an optimizer/sampler resume.'}
    write(adapter/'adapter_config.json',adapter_config)
    print(json.dumps({'stage':'save_nemo','elapsed':time.monotonic()-began,'private':last['worker_private_memory']['private_bytes']}),flush=True)
    target=merged/'parakeet-tdt-0.6b-v3-tr-experimental.nemo';model.save_to(str(target))
    guard();result={'passed':True,'source_update':400000,'source_payload_sha256':metadata['payload']['sha256'],'native_model':str(target),'native_model_bytes':target.stat().st_size,'native_model_sha256':digest(target),'native_blank':9222,'lexical_pieces':9222,'all_native_parameters_mapped':True,'copied_state_tensors':len(copied),'frozen_BF16_rounded_base_matrices':294,'LoRA_projection_merges':96,'adapter_tensors':len(native_adapter),'native_library':'nemo_toolkit3.0.0','wall_seconds':time.monotonic()-began,'last_memory':last,'inference_validation_pending':True,'public_upload_performed':False}
    write(R/'export_complete.json',result);print(json.dumps(result),flush=True);stopped.set();thread.join()
try:main()
except BaseException as e:
    path=R/'export_failure.json'
    if not path.exists():write(path,{'error':str(e),'traceback':traceback.format_exc()})
    raise
