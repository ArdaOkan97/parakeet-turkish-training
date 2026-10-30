"""Finite original-source all207 gradient oracle, then separately a large-row fork."""
import gc,hashlib,json,math,os,random,time
from pathlib import Path
from functools import partial
import torch
from atomic_io_retry_v1 import write_json
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_v3 import assert_model,read_prepared,decode_batch,prohibit_model_transforms,checkpoint,augmentation_binding
from tdt_tiled_full_native_v4 import bounded_tdt_loss
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
    source=AREA/'new_broad_caller_native_03/decoded/report.json';saved=json.loads(source.read_text());require(saved['passed'],'Actual bounded compressed-source decoder proof')
    rows=[]
    for item in saved['rows']:
        require(record(item['PCM_cache']['path'])==item['PCM_cache'],'Pinned actual full-waveform cache changed')
        rows.append(item['row'])
    declaration={'kind':'real_source_all207_TDT_gradient_equivalence_v1','optimizer_updates':0,'augmentation':False,'rows':[r['source_id'] for r in rows],
        'global_gradient_relative_L2_tolerance':.03,'per_tensor_relative_L2_tolerance':.05,'near_zero_reference_norm':1e-8,'near_zero_maxabs_tolerance':1e-6,
        'raw_objective_relative_tolerance':.0001,'gradient_tiling_BF16_rounding_not_byte_equality':True,'allow_bf16_reduced_precision_reduction':True,'joint_policy':'tdt_joint_canonical_fp32_v1' ,'bounded_kernel':'tdt_tiled_full_native_v4','raw_labels_and_full_audio':True,'production_admitted':False}
    write_json(folder/'declaration.json',declaration)
    prohibit_model_transforms(model,torch);model.train();model.encoder.eval();comparisons=[];all_cases_passed=True
    dense=partial(chunked_tdt_loss,sigma=0.,omega=0.,recompute=True,max_joint_bytes=1_700_000_000)
    bounded=partial(bounded_tdt_loss,guard=stop_check)
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
            'ep':projected['ep'],'dp':projected['dp'],'labels':batch.labels.cpu(),'T':row['encoder_frames'],'U':row['target_tokens'],
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
            'passed':case_passed,'failed_tensor_names':tensor_failures,'projected_ep_dp_byte_equal':projected_equal,'joint_fixture':record(folder/('actual_joint_fixture_'+str(index)+'.pt'))}
        comparisons.append(comparison);write_json(folder/'real_model_progress.json',{'completed_cases':len(comparisons),'comparisons':comparisons})
        del reference,result,batch;optimizer.zero_grad();gc.collect();torch.cuda.empty_cache();assert_model(model,optimizer,torch)
        stage('real_original207_gradient_case_complete',case=index+1,aggregate=aggregate)
    require(optimizer.updates==0,'Gradient-only oracle must not advance original Adam/clocks')
    report={'passed':all_cases_passed,'comparisons':comparisons,'source_PCM_proof':record(source),'native_math_proof':record(AREA/'actual_joint_precision_08/actual_joint_precision.json'),
        'optimizer_updates':0,'original_clocks':[64250,2000,62250],'source_provenance':provenance,'production_admitted':False,'declaration':declaration}
    write_json(folder/'real_model_equivalence.json',report);return report

def large(model,processor,optimizer,provenance,folder,*,stage,admit,stop_check):
    from bpe_broad_production_state_v1 import CommitJournal
    proof=AREA/'real_target_TDT_gradients_06/real_model_equivalence.json'
    require(json.loads(proof.read_text())['passed'],'Actual original207 gradient oracle must pass before largest-row fork')
    pinned=AREA/'broad_large_audio_actual_02/report.json';data=json.loads(pinned.read_text());require(data['passed'],'Actual full-source geometry proof')
    selected=max(data['observations'],key=lambda v:v['full_joint_bytes'])
    require(selected['row']['ordinal']==872745 and selected['encoder_frames']==1115 and selected['target_tokens']==419,'Exact retained large source fixture changed')
    decoded=decode_batch([{'ordinal':872745,'epoch':0,'draw_index':0}],folder,admit=admit,stop_check=stop_check)
    row=decoded['rows'][0]['row'];require(row['broad_original']==selected['row'],'Exact raw large label/flags/source identity')
    dense_bytes=row['encoder_frames']*(row['target_tokens']+1)*8716*4
    compact_bytes=row['encoder_frames']*(row['target_tokens']+1)*8*4
    declaration={'kind':'finite_one_full_large_row_native_TDT_update','source_id':row['source_id'],'ordinal':872745,
        'full_audio_samples':selected['decoded_samples'],'target_tokens':419,'encoder_frames':1115,'dense_joint_bytes_not_materialized':dense_bytes,
        'actual_compact_DP_operands_bytes':compact_bytes,'maximum_full_vocabulary_joint_tile_bytes':16_000_000,'actual_operand_bound_unchanged':1_700_000_000,
        'original_full_joint_path_untouched':True,'entire_T_U_DP_and_full8711_normalizer_retained':True,
        'physical_batch':1,'effective_batch':1,'replay_clips':0,'optimizer_updates_bound':1,'augmentation':False,'diagnostic_only':True,
        'long_production_or_complete_pass':False,'quality_gain_or_winner_claimed':False}
    write_json(folder/'declaration.json',declaration);prohibit_model_transforms(model,torch);model.train();model.encoder.eval()
    native=partial(bounded_tdt_loss,guard=stop_check);admit('native_update_full_large_original_row',1_800_000_000)
    optimizer.zero_grad();began=time.monotonic();result,batch=objective(model,processor,row,native)
    require(bool(torch.isfinite(result.total)),'Finite full large-row objective')
    loss=float(result.total.detach());result.total.backward()
    require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in optimizer.parameters.values()),'All207 full-large-row finite gradients')
    norm=float(optimizer.step());optimizer.zero_grad();del result,batch;gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache();assert_model(model,optimizer,torch)
    progress={'completed_updates':1,'clocks':[64251,2001,62251],'objective':loss,'global_grad_norm':norm,'seconds':time.monotonic()-began,'declaration':declaration,'diagnostic_only':True}
    write_json(folder/'native_progress.json',progress);stage('full_large_row_native_update_committed',**progress)
    # Serial guarded checkpoint after releasing native graph; no implicit save.
    admit('full_large_row_checkpoint',350_000_000)
    augmentation=augmentation_binding({'augmentation_policy':str(SOURCE/'fastmss_train_augmentation_v1.pilot.disabled.json')},stop_check)
    journal=CommitJournal(folder/'committed_updates.jsonl');journal.append(progress)
    identity={'source':record(__file__),'declaration':declaration,'equivalence_proof':record(proof)}
    saved=checkpoint(folder/'after_full_large_row_update1.pt',torch=torch,optimizer=optimizer,state={'fixture_committed_draws':1},
        provenance=provenance,identity=identity,augmentation=augmentation,journal=journal,diagnostic_only=True)
    report={'passed':True,'completed_native_updates':1,'checkpoint':saved,'progress':progress,'production_admitted':False,'full_350DEV_for_this_fork_performed':False}
    write_json(folder/'full_large_row_validation.json',report);return report
