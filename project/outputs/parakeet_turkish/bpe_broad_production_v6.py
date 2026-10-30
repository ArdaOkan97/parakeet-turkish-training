"""Weak-phase native core with explicit 2D policy and measured graph-boundary GC.

Preparation/transport remain frozen. Named streaming state is the only intended
checkpoint transport; this module does not export legacy mmap-COW restore.
"""
import gc,os,time,random
from functools import partial
from bpe_broad_production_v5 import (PROJECT,ROOT,SOURCE,assert_model,prohibit_model_transforms,
    augmentation_binding,read_prepared,require,CommitJournal,CompletePassCursor,CarriedReplay,digest,
    weight_objective,record,DrawTransaction,write_json,decode_prepared,prepared_reader,cleanup_prepared)


def native_update(model,teacher,optimizer,rows,draws,reader,*,torch,population_mean_U,update,stage,stop_check,phase):
    from batched_lora_tr_metadata_v1 import collate_training_batch
    from train_lora_batched_tsc_expanded_TRAIN_v1 import pt_focused_batch_objective
    from tdt_tiled_full_native_v6 import bounded_tdt_loss
    from resource_watchdog_v2 import sample_memory
    require(len(rows)==len(draws)>0 and phase=='weak_label_continued_training','Explicit weak-phase rows/draws required')
    require(model.joint._ASR18_fp32_joint_policy['kind']=='ASR18_joint_canonical_T1_U128_FP32_v2','Explicit reviewed new2D policy')
    for row,draw in zip(rows,draws):
        require(row['source_id']==draw['source_id'] and row['language']==draw['language'] and draw['phase']==phase,'Durable ordered identity differs')
        if row['language']=='tr':require(draw['ordinal']==row['broad_original']['ordinal'],'Durable broad ordinal differs')
    native=partial(bounded_tdt_loss,guard=stop_check)
    optimizer.zero_grad();assert_model(model,optimizer,torch);model.train();model.encoder.eval();losses=[];boundaries=[]
    def snapshot(label):
        worker=sample_memory(os.getpid(),label)['worker_private_memory'];require(worker['available'],'Available owned private sample at GC boundary')
        return {'private_bytes':worker['private_bytes'],'CUDA_allocated_bytes':torch.cuda.memory_allocated(),'CUDA_reserved_bytes':torch.cuda.memory_reserved()}
    try:
        for i,row in enumerate(rows):
            stop_check()
            batch=collate_training_batch(model._broad_processor,[row],blank_id=8192,old_size=8193,device='cuda',audio_reader=reader)
            result=pt_focused_batch_objective(model,teacher,batch,old_size=8193,replay_kl=40.,duration_kl=2.,feature_distillation=20.,
                portuguese_token_kl_multiplier=1.,chunk_size=16,tdt_loss_fn=native,tdt_host_lengths=True)
            result=weight_objective(result,[row],population_mean_U,torch_module=torch)
            require(bool(torch.isfinite(result.total)),'Finite full-target native objective')
            (result.total/len(rows)).backward();losses.append(float(result.total.detach()));del result,batch
            torch.cuda.synchronize();before=snapshot('after_backward_graphdelete_beforeGC');began=time.monotonic()
            rng=(random.getstate(),torch.get_rng_state(),torch.cuda.get_rng_state_all());updates_before=optimizer.updates
            versions=[(p.data_ptr(),p._version,p.grad.data_ptr(),p.grad._version) for p in optimizer.parameters.values()]
            collected=gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache();after=snapshot('after_backward_graphdelete_GC_and_empty_cache')
            require(optimizer.updates==updates_before and versions==[(p.data_ptr(),p._version,p.grad.data_ptr(),p.grad._version) for p in optimizer.parameters.values()],
                'GC boundary changed a live parameter/gradient version, pointer or optimizer frontier')
            require(random.getstate()==rng[0] and torch.equal(torch.get_rng_state(),rng[1]) and all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),rng[2])),
                'GC boundary changed Python/Torch/CUDA RNG')
            del rng,versions
            assert_model(model,optimizer,torch)
            boundary={'microbatch':i+1,'before':before,'after':after,'collected_objects':collected,'seconds':time.monotonic()-began,
                'live_all207_parameters_gradients_Adam_preserved':True,'parameter_and_gradient_pointers_versions_unchanged':True,
                'Python_Torch_CUDA_RNG_unchanged':True,'frozen_backing_restored':True}
            boundaries.append(boundary)
            stage('canonical2D_caller_native_microbatch',update=update,microbatch=i+1,source_id=row['source_id'],language=row['language'],GC_boundary=boundary)
        payload=reader.prepared_payload();norm=float(optimizer.step());optimizer.zero_grad();gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache()
        require(torch.cuda.memory_reserved()<=3_000_000_000,'Unchanged native end-cache bound');assert_model(model,optimizer,torch)
        return {'update':update,'mean_objective':sum(losses)/len(losses),'global_grad_norm':norm,'augmentation':payload,
            'source_ids':[r['source_id'] for r in rows],'languages':[r['language'] for r in rows],
            'prepared_transport':reader.transport_binding,'new_joint_policy':model.joint._ASR18_fp32_joint_policy,
            'graph_boundary_GC_observations':boundaries,'GC_policy':'after each completed backward graph deletion, live scientific state preserved'}
    except BaseException:
        if not optimizer.poisoned:optimizer.zero_grad()
        raise
