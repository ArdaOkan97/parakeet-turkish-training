"""First-full-forward host cost attribution and exact CPU frontend comparison."""
import gc,json,time
from pathlib import Path
from functools import partial
import torch
import numpy as np
from atomic_io_retry_v1 import write_json
from bpe_broad_audio_v1 import require,record
from bpe_broad_production_v3 import assert_model,prohibit_model_transforms,read_prepared
from tdt_tiled_full_native_v6 import bounded_tdt_loss
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')

def validate(model,processor,optimizer,provenance,folder,*,stage,admit,stop_check):
    from batched_lora_tr_metadata_v1 import collate_training_batch
    from train_lora_batched_tsc_expanded_TRAIN_v1 import pt_focused_batch_objective
    source=A/'hard_U_rows_decode_01/decoded/report.json';decoded=json.loads(source.read_text())
    CPU_path=A/'CPU_frontend_actual_02/report.json';CPU=json.loads(CPU_path.read_text());require(CPU['passed'] and CPU['Torch_CUDA_runtime'] is None,'Actual CPU frontend fixture required')
    prohibit_model_transforms(model,torch);model.train();model.encoder.eval();optimizer.zero_grad();events=[];handles=[]
    from resource_watchdog_v2 import sample_memory
    import os
    def observe(phase,**extra):
        torch.cuda.synchronize();value=sample_memory(os.getpid(),phase)
        require(value['worker_private_memory']['available'],'Available owned component private memory required')
        item={'phase':phase,'private_bytes':value['worker_private_memory']['private_bytes'],'system_commit_bytes':value['system_memory']['commit_used_bytes'],
            'CUDA_allocated_bytes':torch.cuda.memory_allocated(),'CUDA_reserved_bytes':torch.cuda.memory_reserved(),**extra}
        events.append(item);write_json(folder/'component_progress.json',{'events':events});return item
    def hooks(name):
        def before(module,inputs):observe(name+'_before')
        def after(module,inputs,output):observe(name+'_after')
        return before,after
    for name,module in model.named_modules():
        if name in ('encoder','encoder_projector','decoder') or isinstance(module,(torch.nn.Conv1d,torch.nn.Conv2d,torch.nn.LSTM)):
            pre,post=hooks(name);handles.extend([module.register_forward_pre_hook(pre),module.register_forward_hook(post)])
    comparisons=[]
    try:
        for index,item in enumerate(decoded['rows']):
            stop_check();row=item['row'];require(record(item['PCM_cache']['path'])==item['PCM_cache'],'Actual full source waveform unchanged')
            require(CPU['rows'][index]['source_id']==row['source_id'] and CPU['rows'][index]['full_PCM']==item['PCM_cache'],'Independent full source binding differs')
            admit(f'component_full_source_case_{index}',1_800_000_000 if index==0 else 700_000_000)
            observe('before_original_parent_collate',case=index)
            batch=collate_training_batch(processor,[row],blank_id=8192,old_size=8193,device='cuda',audio_reader=read_prepared)
            observe('after_original_parent_collate',case=index)
            equal={}
            for name,binding in CPU['rows'][index]['inputs'].items():
                require(record(binding['file']['path'])==binding['file'],'Independent CPU frontend file changed')
                expected=torch.from_numpy(np.load(binding['file']['path'],allow_pickle=False)).to('cuda')
                actual=batch.inputs[name]
                equal[name]=actual.shape==expected.shape and actual.dtype==expected.dtype and torch.equal(actual.contiguous().view(torch.uint8),expected.contiguous().view(torch.uint8))
                del expected,actual
            comparisons.append({'source_id':row['source_id'],'all_original_frontend_inputs_byte_exact':all(equal.values()),'inputs':equal,'full_samples':round(row['duration']*16000)})
            require(all(equal.values()),'CPU-only Torch frontend must match original parent processor byte exactly')
            if index==0:
                native=partial(bounded_tdt_loss,guard=stop_check)
                observe('before_first_full_native_forward')
                result=pt_focused_batch_objective(model,{},batch,old_size=8193,replay_kl=40.,duration_kl=2.,feature_distillation=20.,
                    portuguese_token_kl_multiplier=1.,chunk_size=16,tdt_loss_fn=native,tdt_host_lengths=True)
                observe('after_first_full_native_forward');loss=float(result.total.detach());result.total.backward();observe('after_first_full_native_backward')
                require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in optimizer.parameters.values()),'All207 finite full gradients')
                del result
            del batch;optimizer.zero_grad();observe('after_graph_refs_deleted',case=index)
            objects=gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache();observe('after_explicit_graph_GC',case=index,collected_objects=objects)
            assert_model(model,optimizer,torch)
    finally:
        for handle in handles:handle.remove()
    require(optimizer.updates==0,'Observational component profile cannot advance source Adam')
    report={'passed':True,'kind':'ASR18_first_full_native_component_host_profile_v1','optimizer_updates':0,'full_native_row':'kiraat:src01792-00990',
        'full_native_T':1991,'full_native_U':574,'objective':loss,'events':events,'CPU_frontend_byte_comparisons':comparisons,
        'CPU_frontend_proof':record(CPU_path),'CPU_original_reader_proof':record(source),'original_clocks':[64250,2000,62250],
        'synchronous_hooks_observational_only':True,'training_steady_improvement_claimed':False,'production_ready':False}
    write_json(folder/'component_memory_profile.json',report);return report
