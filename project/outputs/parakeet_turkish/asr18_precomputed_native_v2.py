"""Serial native updates from verified provisional input features; no frontend."""
import gc,json,os,random,time
from functools import partial
import numpy as np
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_v3 import assert_model
from bpe_broad_production_state_v1 import weight_objective
from asr18_precomputed_collator_v1 import collate_precomputed_row
from asr18_narrow_objective_v1 import pt_focused_batch_objective
from tdt_tiled_full_native_v7 import bounded_tdt_loss


def native_update(model,teacher,optimizer,rows,draws,*,feature_for_row,update_binding,torch,
                  population_mean_U,update,stage,stop_check,phase,feature_update_index):
    require(record(update_binding['path'])==update_binding,'Exact independently closed feature-update metadata binding required')
    from pathlib import Path
    window_update=json.loads(Path(update_binding['path']).read_text(encoding='utf-8'))
    manifest_rows=window_update['rows']
    require(phase in ('weak_label_continued_training','cleaner_data_fine_tuning') and
        type(feature_update_index) is int and 0<=feature_update_index<20 and window_update['update_index']==feature_update_index and
        len(rows)==len(draws)==len(manifest_rows) and 3<=len(rows)<=16,'Explicit phase,local feature index and full tail batch required')
    require(update==optimizer.updates+1 and all(r['language']=='tr' for r in rows[:-2]) and
        rows[-2]['language'] in ('en','es','it','fr','de') and rows[-1]['language']=='pt','Actual positive global frontier and carried1oldfive+1PT pair required')
    require(model.joint._ASR18_fp32_joint_policy['kind']=='ASR18_joint_canonical_T1_U128_FP32_v2','Explicit new2D numeric policy required')
    for row,draw,prepared in zip(rows,draws,manifest_rows):
        context={key:draw[key] for key in ('source_id','language','epoch','draw_index','phase')};context['split']='train'
        require(row['source_id']==draw['source_id']==prepared['source_id'] and row['language']==draw['language']==prepared['language']
            and prepared['context']==context and draw['phase']==phase and prepared['encoder_frames']==row['encoder_frames'],
            'Source/draw/language/full geometry differs from prepared input')
        require(prepared['role']==('student_tr' if row['language']=='tr' else 'replay'),'Explicit replay/teacher protection role differs')
        if row['language']=='tr':
            ordinal=row['broad_original']['ordinal'] if phase=='weak_label_continued_training' else row['cleaner_original_ordinal']
            require(draw['ordinal']==ordinal and prepared['full_target_ids']==row['tr_metadata_labels'],'Full phase-specific original ordinal/targets differ')
    native=partial(bounded_tdt_loss,guard=stop_check);optimizer.zero_grad();assert_model(model,optimizer,torch)
    model.train();model.encoder.eval();losses=[];boundaries=[]
    from resource_watchdog_v2 import sample_memory
    def snapshot(label):
        item=sample_memory(os.getpid(),label)['worker_private_memory'];require(item['available'],'Available owned native private sample')
        return {'private_bytes':item['private_bytes'],'CUDA_allocated_bytes':torch.cuda.memory_allocated(),'CUDA_reserved_bytes':torch.cuda.memory_reserved()}
    try:
        for i,(row,prepared) in enumerate(zip(rows,manifest_rows)):
            stop_check();inputs=feature_for_row(i)
            batch=collate_precomputed_row(row,inputs,prepared['full_target_ids'],device='cuda');del inputs
            result=pt_focused_batch_objective(model,teacher,batch,old_size=8193,replay_kl=40.,duration_kl=2.,feature_distillation=20.,
                portuguese_token_kl_multiplier=1.,chunk_size=16,tdt_loss_fn=native,tdt_host_lengths=True)
            result=weight_objective(result,[row],population_mean_U,torch_module=torch)
            require(bool(torch.isfinite(result.total)),'Finite full-target native objective')
            (result.total/len(rows)).backward();losses.append(float(result.total.detach()));del result,batch
            torch.cuda.synchronize();before=snapshot('precomputed_after_graphdelete_before_GC_and_emptycache');began=time.monotonic()
            numpy_rng=np.random.get_state();rng=(random.getstate(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
            versions=[(p.data_ptr(),p._version,p.grad.data_ptr(),p.grad._version) for p in optimizer.parameters.values()];frontier=optimizer.updates
            objects=gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache();after=snapshot('precomputed_after_graph_GC_and_emptycache')
            require(optimizer.updates==frontier and versions==[(p.data_ptr(),p._version,p.grad.data_ptr(),p.grad._version) for p in optimizer.parameters.values()],
                'GC boundary changed live parameter/gradient pointers, versions or Adam frontier')
            actual_numpy=np.random.get_state()
            require(numpy_rng[0]==actual_numpy[0] and np.array_equal(numpy_rng[1],actual_numpy[1]) and numpy_rng[2:]==actual_numpy[2:]
                and random.getstate()==rng[0] and torch.equal(torch.get_rng_state(),rng[1]) and all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),rng[2])),
                'Graph reclamation changed Python/NumPy/Torch/CUDA RNG')
            del numpy_rng,rng,versions,actual_numpy;assert_model(model,optimizer,torch)
            boundary={'microbatch':i+1,'before':before,'after':after,'collected_objects':objects,'seconds':time.monotonic()-began,
                'live207_parameter_and_gradient_pointers_versions_unchanged':True,'Python_NumPy_Torch_CUDA_RNG_unchanged':True}
            boundaries.append(boundary);stage('precomputed_native_microbatch',update=update,microbatch=i+1,source_id=row['source_id'],language=row['language'],memory_boundary=boundary)
        payload=window_update['prepared_payload'];require(payload['state']=='prepared_not_committed' and len(payload['rows'])==len(rows),'Full copied CPU preparation is not a sampler commit')
        require([r['draw']['source_id'] for r in payload['rows']]==[r['source_id'] for r in rows],'Prepared payload does not match actual native source order')
        if phase=='cleaner_data_fine_tuning':
            require(payload.get('protection_kind')=='FastMSS_explicit_clean_phase_protection_v1' and
                all(r['augmentation']['applied'] is False and r['draw']['phase']==phase for r in payload['rows']),
                'Cleaner phase requires direct clean protection; no prepared weak augmentation transport')
        norm=float(optimizer.step());optimizer.zero_grad();gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache()
        require(torch.cuda.memory_reserved()<=3_000_000_000,'Unchanged native end-cache cap');assert_model(model,optimizer,torch)
        return {'update':update,'phase':phase,'mean_objective':sum(losses)/len(losses),'global_grad_norm':norm,'augmentation':payload,
            'source_ids':[r['source_id'] for r in rows],'languages':[r['language'] for r in rows],'new_joint_policy':model.joint._ASR18_fp32_joint_policy,
            'precomputed_input_update':update_binding,'parent_frontend_or_augmentation_invoked':False,
            'graph_boundary_memory_observations':boundaries,'prepared_is_committed':False}
    except BaseException:
        if not optimizer.poisoned:optimizer.zero_grad()
        raise
