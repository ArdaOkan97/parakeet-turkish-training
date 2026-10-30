"""NEW bounded grouped Transformer; original B1 CNN, BF16 joint/FP32 full-U larger-T loss.

This changes Transformer batch kernels and gradient accumulation order. It is
not a claim of byte equivalence to the serial native trajectory. No source is
admitted until an explicit grouped-reference oracle and resource review.
"""
import gc,json,os,random,time
from contextlib import contextmanager
from functools import partial
import numpy as np
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_v3 import assert_model
from bpe_broad_production_state_v1 import weight_objective
from asr18_precomputed_collator_v1 import collate_precomputed_row
from asr18_narrow_objective_v1 import BatchObjective,sampled_preservation,preservation_row_scales
from tdt_blocked_bf16_full_native_v1 import bounded_tdt_loss
from tdt_blocked_bf16_joint_view_v1 import BlockedBF16JointView,POLICY as LOSS_POLICY
from tdt_blocked_full_native_allocation_budget_v1 import allocation_budget


def envelope(prepared):
    count=len(prepared);frames=[int(p['input_features']['shape'][1]) for p in prepared]
    targets=[len(p['full_target_ids']) for p in prepared]
    maximum=(max(frames)+7)//8
    return {'count':count,'physical_frames':frames,'padded_encoder_frames':maximum,
        'padded_encoder_positions':count*maximum,'padded_attention_cells':count*maximum*maximum,
        'physical_input_positions':count*max(frames),'unpadded_predictor_positions':sum(u+1 for u in targets),
        'full_DP_cells':sum(int(p['encoder_frames'])*(u+1) for p,u in zip(prepared,targets))}


def allowed(prepared):
    v=envelope(prepared)
    return (1<=v['count']<=8 and v['padded_encoder_positions']<=1991
        and v['padded_attention_cells']<=1991**2 and v['physical_input_positions']<=15925
        and v['unpadded_predictor_positions']<=575 and v['full_DP_cells']<=1991*575)


def groups(prepared):
    """Consecutive rows only; full hard row is supported as a singleton."""
    output=[];current=[]
    for index,p in enumerate(prepared):
        require(allowed([p]),'Full source row must independently fit the retained hard logical envelope')
        if current and not allowed([prepared[i] for i in [*current,index]]):
            output.append(current);current=[]
        current.append(index)
    if current:output.append(current)
    require([i for g in output for i in g]==list(range(len(prepared))),'Every original row exactly once in original order')
    return output


@contextmanager
def exact_serial_stem(model,physical_frames,*,torch,guard):
    """Call the validated conv-only B1 forward with each ORIGINAL full shape."""
    module=model.encoder.subsampling;original=module.forward
    require(getattr(module,'_ASR18_temporal_tile_policy',{}).get('kind')=='ASR18_proposed_frozen_CNN_conv_only_temporal_tiled_lifetime_v2',
        'The exact validated conv-only instance must already be installed')
    calls=0
    def forward(input_features,attention_mask=None):
        nonlocal calls
        guard();calls+=1
        require(calls==1 and not model.encoder.training and not input_features.requires_grad
            and input_features.shape[0]==len(physical_frames) and attention_mask is not None
            and tuple(attention_mask.shape)==tuple(input_features.shape[:2]),'One actual padded encoder group, no recomputed/foreign stem call')
        outputs=[]
        for index,frames in enumerate(physical_frames):
            guard()
            require(1<=frames<=input_features.shape[1] and not bool(attention_mask[index,frames:].any()),
                'Padding never expands or replaces the independently verified original physical input')
            outputs.append(original(input_features[index:index+1,:frames],attention_mask[index:index+1,:frames])[0])
        return torch.nn.utils.rnn.pad_sequence(outputs,batch_first=True,padding_value=0.)
    module.forward=forward
    try:
        yield
        require(calls==1,'Exactly one grouped encoder invocation')
    finally:module.forward=original


def collate_group(rows,prepared,feature_for_row,indices,*,torch):
    """Original singleton target/replay validation; pad only CPU mel inputs."""
    batches=[];features=[];masks=[];physical=[]
    for i in indices:
        inputs=feature_for_row(i)
        batch=collate_precomputed_row(rows[i],inputs,prepared[i]['full_target_ids'],device='cpu')
        require(list(batch.inputs['input_features'].shape)==prepared[i]['input_features']['shape']
            and list(batch.inputs['attention_mask'].shape)==prepared[i]['attention_mask']['shape'],
            'Actual full NPY shapes must equal the independently bound metadata')
        features.append(batch.inputs['input_features'][0]);masks.append(batch.inputs['attention_mask'][0])
        physical.append(features[-1].shape[0]);batches.append(batch);del inputs
    padded={'input_features':torch.nn.utils.rnn.pad_sequence(features,batch_first=True,padding_value=0.),
        'attention_mask':torch.nn.utils.rnn.pad_sequence(masks,batch_first=True,padding_value=False)}
    require(physical==[int(prepared[i]['input_features']['shape'][1]) for i in indices],
        'Original physical lengths remain exact, independently of valid-mask counts')
    for batch in batches:batch.inputs={}
    return [batch.to('cuda') for batch in batches],{k:v.to('cuda') for k,v in padded.items()},physical


@contextmanager
def original_head_selected_encoder(model,features,mask):
    """Reference only: original pt_focused function sees the original grouped row."""
    from types import SimpleNamespace
    encoder=model.encoder;original=encoder.forward;calls=0
    def selected(*args,**kwargs):
        nonlocal calls
        require(not args and not kwargs and calls==0,'Reference receives the empty consumed-input map exactly once')
        calls+=1
        return SimpleNamespace(last_hidden_state=features,attention_mask=mask)
    encoder.forward=selected
    try:
        yield
        require(calls==1,'Reference actually invoked original singleton batch_objective')
    finally:encoder.forward=original


def encoded_row_objective(model,teacher,batch,features,*,torch,native):
    """Original B1 decoder/teacher/weights; declared BF16 joint and full FP32 TDT."""
    require(batch.batch_size==features.shape[0]==1 and features.shape[1]==int(batch.frame_lengths[0]),
        'One full valid source feature slice, no padded frames in per-clip loss')
    ep=model.encoder_projector(features);dp=model.decoder(batch.decoder_input_ids)
    raw=native(model.joint,ep,dp,batch.labels,model.config.vocab_size,batch.blank_id,model.config.durations,
        logit_lengths=batch.frame_lengths,target_lengths=batch.target_lengths,chunk_size=16,reduction='none')
    require(raw.shape==(1,),'Unchanged raw full-target sequence loss per original clip')
    per_tdt=raw.float()/batch.target_lengths.to(features.device)
    token,duration,feature=sampled_preservation(model,teacher,batch,features,ep,dp,old_size=8193)
    scale=torch.tensor(preservation_row_scales(batch.rows),dtype=torch.float32,device=features.device)
    token_scale=scale*torch.tensor([1. if row['language']=='pt' else 1. for row in batch.rows],
        dtype=torch.float32,device=features.device)
    total=per_tdt+40.*token_scale*token+2.*scale*duration+20.*scale*feature
    return BatchObjective(total=total.mean(),tdt=per_tdt.mean(),token_kl=token.mean(),
        duration_kl=duration.mean(),feature_distillation=feature.mean(),
        per_clip={'tdt':per_tdt,'token_kl':token,'duration_kl':duration,'feature_distillation':feature,'total':total},
        batch_size=1,encoder_frames=tuple(batch.frame_lengths.tolist()),target_tokens=tuple(batch.target_lengths.tolist()))


def native_update(model,teacher,optimizer,rows,draws,*,feature_for_row,update_binding,torch,
                  population_mean_U,update,stage,stop_check,phase,feature_update_index,
                  reference_original_head=False,observe=lambda *a:None):
    require(record(update_binding['path'])==update_binding,'Exact independently closed feature-update metadata binding required')
    from pathlib import Path
    window_update=json.loads(Path(update_binding['path']).read_text(encoding='utf-8'));prepared=window_update['rows']
    require(phase in ('weak_label_continued_training','cleaner_data_fine_tuning') and
        type(feature_update_index) is int and 0<=feature_update_index<20 and window_update['update_index']==feature_update_index
        and len(rows)==len(draws)==len(prepared) and 3<=len(rows)<=16,'Full current phase and exact tail batch')
    require(update==optimizer.updates+1 and all(r['language']=='tr' for r in rows[:-2])
        and rows[-2]['language'] in ('en','es','it','fr','de') and rows[-1]['language']=='pt','Original positive frontier and carried replay pair')
    require(model.joint._ASR18_fp32_joint_policy['kind']=='ASR18_joint_canonical_T1_U128_FP32_v2','Canonical FP32 T1/U128 remains unchanged')
    for row,draw,p in zip(rows,draws,prepared):
        context={key:draw[key] for key in ('source_id','language','epoch','draw_index','phase')};context['split']='train'
        require(row['source_id']==draw['source_id']==p['source_id'] and row['language']==draw['language']==p['language']
            and p['context']==context and draw['phase']==phase and p['encoder_frames']==row['encoder_frames'],
            'Exact source/draw/language/full geometry in original order')
        require(p['role']==('student_tr' if row['language']=='tr' else 'replay'),'Original teacher protection role')
        if row['language']=='tr':
            ordinal=row['broad_original']['ordinal'] if phase=='weak_label_continued_training' else row['cleaner_original_ordinal']
            require(draw['ordinal']==ordinal and p['full_target_ids']==row['tr_metadata_labels'],'All original target IDs and ordinal')
    packing=groups(prepared)
    width=model.joint.head.in_features;head_count=sum(p.numel() for p in model.joint.head.parameters())
    routes={}
    for indices in packing:
        all_cells=sum(int(prepared[i]['encoder_frames'])*(len(prepared[i]['full_target_ids'])+1) for i in indices)
        for i in indices:
            t=int(prepared[i]['encoder_frames']);u=len(prepared[i]['full_target_ids']);current=t*(u+1)
            # Charge ALL other compact contexts, including not-yet-produced
            # future rows. No autograd scheduling/release-order credit.
            routes[i]=allocation_budget(t,u,projected_elements=(t+u+1)*width,head_elements=head_count,
                hidden_width=width,prior_compact_cells=all_cells-current)
    def routed_native(index,original_joint,*args,**kwargs):
        require(original_joint is model.joint,'Original unchanged teacher/preservation joint')
        return bounded_tdt_loss(BlockedBF16JointView(original_joint),*args,**kwargs,guard=stop_check,
            prior_compact_cells=routes[index]['prior_compact_cells'])
    optimizer.zero_grad();assert_model(model,optimizer,torch);model.train();model.encoder.eval();losses=[];boundaries=[]
    from resource_watchdog_v2 import sample_memory
    def snapshot(label):
        item=sample_memory(os.getpid(),label)['worker_private_memory'];require(item['available'],'Available owned grouped private sample')
        return {'private_bytes':item['private_bytes'],'CUDA_allocated_bytes':torch.cuda.memory_allocated(),'CUDA_reserved_bytes':torch.cuda.memory_reserved()}
    try:
        for gi,indices in enumerate(packing):
            stop_check();stage('before_larger_T_full_U_group',group=gi+1,
                prospective_joint_GPU_bytes=max(routes[i]['total_budget_bytes'] for i in indices))
            batches,inputs,physical=collate_group(rows,prepared,feature_for_row,indices,torch=torch)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                with exact_serial_stem(model,physical,torch=torch,guard=stop_check):encoded=model.encoder(**inputs)
                require(encoded.attention_mask is not None and torch.equal(encoded.attention_mask.sum(-1).detach().cpu().long(),
                    torch.tensor([rows[i]['encoder_frames'] for i in indices],dtype=torch.long)),
                    'Every grouped output mask must equal its original full valid encoder length')
                scalars=[]
                for j,(i,batch) in enumerate(zip(indices,batches)):
                    stop_check();features=encoded.last_hidden_state[j:j+1,:rows[i]['encoder_frames']]
                    native=partial(routed_native,i)
                    if reference_original_head:
                        from asr18_narrow_objective_v1 import pt_focused_batch_objective
                        with original_head_selected_encoder(model,features,encoded.attention_mask[j:j+1,:features.shape[1]]):
                            result=pt_focused_batch_objective(model,teacher,batch,old_size=8193,replay_kl=40.,duration_kl=2.,
                                feature_distillation=20.,portuguese_token_kl_multiplier=1.,chunk_size=16,tdt_loss_fn=native,tdt_host_lengths=True)
                    else:result=encoded_row_objective(model,teacher,batch,features,torch=torch,native=native)
                    result=weight_objective(result,[rows[i]],population_mean_U,torch_module=torch)
                    require(bool(torch.isfinite(result.total)),'Finite full canonical per-clip objective')
                    observe(i,features,result,encoded.attention_mask[j:j+1])
                    scalars.append(result.total);losses.append(float(result.total.detach()));del result,features
                # Each original clip has exactly 1/effective_count weight. One
                # backward traverses the shared grouped encoder graph ONCE.
                objective=torch.stack(scalars).sum()/len(rows)
            objective.backward();del objective,scalars,encoded,batches,inputs
            torch.cuda.synchronize();before=snapshot('grouped_after_graphdelete_before_GC_and_emptycache');began=time.monotonic()
            numpy_rng=np.random.get_state();rng=(random.getstate(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
            versions=[(p.data_ptr(),p._version,p.grad.data_ptr(),p.grad._version) for p in optimizer.parameters.values()];frontier=optimizer.updates
            objects=gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache();after=snapshot('grouped_after_graph_GC_and_emptycache')
            require(optimizer.updates==frontier and versions==[(p.data_ptr(),p._version,p.grad.data_ptr(),p.grad._version) for p in optimizer.parameters.values()],
                'Group GC boundary keeps all207 parameter/gradient pointers, versions and Adam frontier')
            actual_numpy=np.random.get_state()
            require(numpy_rng[0]==actual_numpy[0] and np.array_equal(numpy_rng[1],actual_numpy[1]) and numpy_rng[2:]==actual_numpy[2:]
                and random.getstate()==rng[0] and torch.equal(torch.get_rng_state(),rng[1])
                and all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),rng[2])),
                'Actual grouped graph reclamation preserves all4 RNG states')
            del numpy_rng,rng,versions,actual_numpy;assert_model(model,optimizer,torch)
            boundary={'group':gi+1,'original_row_indices':indices,'logical_envelope':envelope([prepared[i] for i in indices]),
                'before':before,'after':after,'collected_objects':objects,'seconds':time.monotonic()-began,
                'live207_parameter_and_gradient_pointers_versions_unchanged':True,'Python_NumPy_Torch_CUDA_RNG_unchanged':True}
            boundaries.append(boundary);stage('grouped_native_graph_boundary',update=update,group=gi+1,memory_boundary=boundary)
        payload=window_update['prepared_payload'];require(payload['state']=='prepared_not_committed' and len(payload['rows'])==len(rows)
            and [r['draw']['source_id'] for r in payload['rows']]==[r['source_id'] for r in rows],'Full ordered prepared payload is not a sampler commit')
        if phase=='cleaner_data_fine_tuning':
            require(payload.get('protection_kind')=='FastMSS_explicit_clean_phase_protection_v1'
                and all(not r['augmentation']['applied'] and r['draw']['phase']==phase for r in payload['rows']),
                'Unchanged clean-phase direct protection')
        norm=float(optimizer.step());optimizer.zero_grad();gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache()
        require(torch.cuda.memory_reserved()<=3_000_000_000,'Unchanged native end-cache3GB including persistent bank');assert_model(model,optimizer,torch)
        return {'update':update,'phase':phase,'mean_objective':sum(losses)/len(losses),'global_grad_norm':norm,'augmentation':payload,
            'source_ids':[r['source_id'] for r in rows],'languages':[r['language'] for r in rows],
            'new_joint_policy':LOSS_POLICY,'unchanged_preservation_joint_policy':model.joint._ASR18_fp32_joint_policy,
            'joint_work_unit_allocation_routes':routes,'all_other_group_compact_contexts_charged':True,'precomputed_input_update':update_binding,
            'parent_frontend_or_augmentation_invoked':False,'graph_boundary_memory_observations':boundaries,
            'physical_group_row_indices':packing,'Transformer_batch_kernel_trajectory_change_explicit':True,'prepared_is_committed':False}
    except BaseException:
        if not optimizer.poisoned:optimizer.zero_grad()
        raise
