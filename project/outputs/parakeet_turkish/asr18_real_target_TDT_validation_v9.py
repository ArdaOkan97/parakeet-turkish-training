"""Finite original-source all207 gradient oracle, then separately a large-row fork."""
import gc,hashlib,json,math,os,random,time
from pathlib import Path
from functools import partial
import torch
from atomic_io_retry_v1 import write_json
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_v3 import assert_model,read_prepared,decode_batch,prohibit_model_transforms,checkpoint,augmentation_binding
from tdt_tiled_full_native_v5 import bounded_tdt_loss
ROOT=Path(r'E:\gpu_training_scratchpad');AREA=ROOT/'analysis/ASR18_memory_diagnosis_v1'
SOURCE=Path(__file__).resolve().parent

def objective(model,processor,row,native):
    from batched_lora_tr_metadata_v1 import collate_training_batch
    from train_lora_batched_tsc_expanded_TRAIN_v1 import pt_focused_batch_objective
    batch=collate_training_batch(processor,[row],blank_id=8192,old_size=8193,device='cuda',audio_reader=read_prepared)
    result=pt_focused_batch_objective(model,{},batch,old_size=8193,replay_kl=40.,duration_kl=2.,feature_distillation=20.,
        portuguese_token_kl_multiplier=1.,chunk_size=16,tdt_loss_fn=native,tdt_host_lengths=True)
    return result,batch

def compare(model,processor,optimizer,provenance,folder,*,stage,admit,stop_check):
    from native_tdt_checkpointed_v4 import chunked_tdt_loss
    source=AREA/'hard_U_rows_decode_01/decoded/report.json';saved=json.loads(source.read_text());require(saved['passed'],'Actual bounded compressed-source decoder proof')
    rows=[]
    for item in saved['rows']:
        require(record(item['PCM_cache']['path'])==item['PCM_cache'],'Pinned actual full-waveform cache changed')
        rows.append(item['row'])
    declaration={'kind':'real_source_2D_all207_TDT_gradient_equivalence_v1','optimizer_updates':0,'augmentation':False,'rows':[r['source_id'] for r in rows],
        'global_gradient_relative_L2_tolerance':.03,'per_tensor_relative_L2_tolerance':.05,'near_zero_reference_norm':1e-8,'near_zero_maxabs_tolerance':1e-6,
        'raw_objective_relative_tolerance':.0001,'gradient_tiling_BF16_rounding_not_byte_equality':False,'allow_bf16_reduced_precision_reduction':True,'joint_policy':'tdt_joint_canonical_2d_fp32_v2' ,'bounded_kernel':'tdt_tiled_full_native_v5','raw_labels_and_full_audio':True,'loss_acoustic_frames':15,'acoustic_subset_numerical_oracle_only':True,'full_length_objective_fit_claimed':False,'production_admitted':False}
    write_json(folder/'declaration.json',declaration)
    prohibit_model_transforms(model,torch);model.train();model.encoder.eval();comparisons=[];all_cases_passed=True
    def subset(function):
        def call(*args,**kwargs):
            lengths=kwargs['logit_lengths'].clone()
            require(int(lengths[0])>=15,'Actual full source must exceed oracle subset')
            kwargs['logit_lengths']=lengths.clamp(max=15)
            return function(*args,**kwargs)
        return call
    dense=subset(partial(chunked_tdt_loss,sigma=0.,omega=0.,recompute=True,max_joint_bytes=1_700_000_000))
    bounded=subset(partial(bounded_tdt_loss,guard=stop_check))
    for index,row in enumerate(rows):
        stop_check();admit('real_original207_gradient_case_'+str(index),1_800_000_000 if index==0 else 400_000_000)
        optimizer.zero_grad();rng=(random.getstate(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
        captures={}
        def capture(name):
            def hook(module,inputs,output):captures[name]=output.detach().cpu().clone()
            return hook
        handles=[model.encoder_projector.register_forward_hook(capture('ep')),model.decoder.register_forward_hook(capture('dp'))]
        result,batch=objective(model,processor,row,dense);reference_loss=float(result.total.detach())
        for handle in handles:handle.remove()
        projected={n:v for n,v in captures.items()}
        fixture={'kind':'actual_trained_joint_gradient_diagnostic_fixture','diagnostic_only':True,
            'ep':projected['ep'],'dp':projected['dp'],'labels':batch.labels.cpu(),'T':15,'U':row['target_tokens'],
            'joint':{n:p.detach().cpu() for n,p in model.joint.state_dict().items()},'row':row,'original_objective':reference_loss,
            'source_records':provenance['source_records'],'hidden_activation':'relu','vocabulary':8711,'blank_id':8192,'durations':[0,1,2,3,4]}
        torch.save(fixture,folder/('actual_joint_fixture_'+str(index)+'.pt'));del fixture
        result.total.backward()
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in optimizer.parameters.values()),'Dense original all207 finite gradients')
        reference={n:p.grad.detach().cpu().clone() for n,p in optimizer.parameters.items()}
        del result,batch;optimizer.zero_grad();gc.collect();torch.cuda.empty_cache();stage('real_dense_reference_gradient_complete',case=index+1)
        random.setstate(rng[0]);torch.set_rng_state(rng[1]);torch.cuda.set_rng_state_all(rng[2])
        captures.clear();handles=[model.encoder_projector.register_forward_hook(capture('ep')),model.decoder.register_forward_hook(capture('dp'))]
        result,batch=objective(model,processor,row,bounded);actual_loss=float(result.total.detach())
        for handle in handles:handle.remove()
        projected_equal={n:torch.equal(v.contiguous().view(torch.uint8),captures[n].contiguous().view(torch.uint8)) for n,v in projected.items()}
        result.total.backward()
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in optimizer.parameters.values()),'Bounded all207 finite gradients')
        metrics=[];tensor_failures=[];sums={kind:[0.,0.] for kind in ('all207','head15','LoRA192')}
        for name,p in optimizer.parameters.items():
            actual=p.grad.detach().cpu();expected=reference[name];delta=(actual.double()-expected.double())
            d2=float((delta*delta).sum());e2=float((expected.double()**2).sum());norm=math.sqrt(e2)
            relative=math.sqrt(d2/max(e2,1e-40));maximum=float(delta.abs().max())
            if not(relative<.05 if norm>=1e-8 else maximum<1e-6):tensor_failures.append(name)
            for kind in ('all207','head15' if name in optimizer.head_names else 'LoRA192'):
                sums[kind][0]+=d2;sums[kind][1]+=e2
            metrics.append({'name':name,'relative_L2':relative,'max_abs':maximum,'reference_L2':norm});del actual,expected,delta
        aggregate={k:math.sqrt(a/max(b,1e-40)) for k,(a,b) in sums.items()}
        case_passed=(not tensor_failures and all(projected_equal.values()) and all(v<.03 for v in aggregate.values()) and
            abs(actual_loss-reference_loss)<.0001*max(1.,abs(reference_loss)))
        all_cases_passed=all_cases_passed and case_passed
        comparison={'source_id':row['source_id'],'frames':row['encoder_frames'],'tokens':row['target_tokens'],
            'dense_objective':reference_loss,'bounded_objective':actual_loss,'aggregate_gradient_relative_L2':aggregate,'all207':metrics,
            'passed':case_passed,'failed_tensor_names':tensor_failures,'projected_ep_dp_byte_equal':projected_equal,'all207_byte_equal':all(v['max_abs']==0. for v in metrics),'loss_acoustic_frames':15,'full_input_encoder_frames':row['encoder_frames'],'joint_fixture':record(folder/('actual_joint_fixture_'+str(index)+'.pt'))}
        comparisons.append(comparison);write_json(folder/'real_model_progress.json',{'completed_cases':len(comparisons),'comparisons':comparisons})
        del reference,result,batch;optimizer.zero_grad();gc.collect();torch.cuda.empty_cache();assert_model(model,optimizer,torch)
        stage('real_original207_gradient_case_complete',case=index+1,aggregate=aggregate)
    require(optimizer.updates==0,'Gradient-only oracle must not advance original Adam/clocks')
    report={'passed':all_cases_passed,'comparisons':comparisons,'source_PCM_proof':record(source),'native_math_proof':record(AREA/'canonical2D_joint_GC_01/actual_joint_precision.json'),
        'optimizer_updates':0,'original_clocks':[64250,2000,62250],'source_provenance':provenance,'production_admitted':False,'declaration':declaration}
    write_json(folder/'real_model_equivalence.json',report);return report
