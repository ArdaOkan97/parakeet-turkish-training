"""NEW opt-in weak-phase caller, canonical full-vocabulary native long path.

CPU decode/augmentation children close before serial GPU work. Original source
Adam/RNG is carried. Final population/ROOT caller review remain prerequisites.
The finite integration probe is diagnostic and cannot seed production.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
from functools import partial
import gc,hashlib,json,os,random,subprocess,sys,time
from pathlib import Path
from bpe_broad_production_v3 import (PROJECT,ROOT,SOURCE,assert_model,prohibit_model_transforms,
    augmentation_binding,checkpoint,read_prepared)
from bpe_broad_production_state_v1 import require,CommitJournal,CompletePassCursor,CarriedReplay,digest,weight_objective
from bpe_broad_audio_v1 import record
from bpe_broad_transaction_v1 import DrawTransaction
from bpe_broad_journal_branch_v1 import branch_checkpoint_prefix
from atomic_io_retry_v1 import write_json

def decode_prepared(draws,contexts,folder,config,*,batch_id,admit,stop_check,diagnostic_only):
    from fastmss_cpu_prepared_transport_v1 import observe_closed_child
    folder.mkdir(exist_ok=False)
    request={'rows':draws,'contexts':[asdict(c) for c in contexts],'batch_id':batch_id,
        'job_id':folder.name,'augmentation_policy':config['augmentation_policy'],'diagnostic_only':diagnostic_only,
        'heldout_PCM_binding':config['heldout_PCM_binding'],'source_inventory_binding':config.get('source_inventory_binding')}
    write_json(folder/'decode_request.json',request)
    command=[sys.executable,'-X','utf8','-B',str(PROJECT/'work/research/ASR18_bounded_CPU_phase_v3.py'),
        '--folder',str(folder/'decoder_guard'),'--seconds','300','--script',str(PROJECT/'work/research/decode_ASR18_broad_batch_v6.py'),
        '--','--request',str(folder/'decode_request.json'),'--output',str(folder/'decoded')]
    proof=config['CPU_prepare_measured_proof'];require(record(proof['path'])==proof and json.loads(Path(proof['path']).read_text())['passed'],'Actual full14 fixedgeometry CPU proof required')
    fixture=[json.loads(v) for v in Path(config['probe_manifest']).read_text().splitlines()]
    require(sorted(d['ordinal'] for d in draws)==sorted(v['ordinal'] for v in fixture),'300MB CPU admission is scoped to this exact14source diagnostic geometry')
    admit('bounded_CPU_prepare_'+folder.name,300_000_000)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    with (folder/'decoder.log').open('x',encoding='utf-8') as log:
        child=subprocess.Popen(command,cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT)
        try:
            began=time.monotonic()
            while child.poll() is None:
                stop_check();require(time.monotonic()-began<310,'Bounded CPU preparation timeout');time.sleep(.2)
            require(child.wait()==0,'CPU preparation must actually close0 before any native update')
            result=json.loads((folder/'decoded/report.json').read_text());require(result['passed'],'Complete source preparation required')
            transport=result['augmentation_transport']
            closure=observe_closed_child(child,owner_record=transport['owner'],manifest_record=transport['manifest'],
                proof_path=folder/'CPU_actual_closure.json',guard=stop_check)
            result['CPU_actual_closure']=closure
            result['decoder_report']=record(folder/'decoded/report.json')
            return result
        finally:
            if child.poll() is None:
                write_json(folder/'decoder_guard/stop.json',{'reason':'owned_GPU_caller_leaving_CPU_prepare'})
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    import psutil
                    boot=json.loads((folder/'decoder_guard/boot.json').read_text());actual=psutil.Process(boot['pid'])
                    require((boot['pid']==child.pid or boot['ppid']==child.pid) and actual.ppid()==boot['ppid'],'Exact owned CPU identity before stop')
                    actual.terminate();actual.wait(timeout=5);child.wait(timeout=5)

def prepared_reader(prepared,rows,contexts,binding,*,batch_id,stop_check):
    from fastmss_cpu_prepared_transport_v1 import VerifiedPreparedAudioReader
    require(record(prepared['decoder_report']['path'])==prepared['decoder_report'],'Independent clean decoder report changed')
    independent=json.loads(Path(prepared['decoder_report']['path']).read_text())
    require(independent['rows']==prepared['rows'],'In-memory rows differ from bound independent clean decoder report')
    return VerifiedPreparedAudioReader(binding,rows,contexts,batch_id=batch_id,
        manifest_record=prepared['augmentation_transport']['manifest'],closure_record=prepared['CPU_actual_closure'],
        protected_reader=read_prepared,record_prepared=lambda v:None,guard=stop_check,
        expected_clean_hashes=[v['reader_evidence']['PCM_f32_sha256'] for v in independent['rows']])

def cleanup_prepared(prepared,folder,*,batch_id,journal,discard,stop_check):
    from fastmss_cpu_prepared_transport_v1 import cleanup_owned_cache
    transport=prepared['augmentation_transport']
    # Keep the exact small receipt contents before deleting transient waves.
    manifest=json.loads(Path(transport['manifest']['path']).read_text());owner=json.loads(Path(transport['owner']['path']).read_text())
    durable={'kind':'discard_uncommitted_preparation' if discard else 'durable_native_boundary','batch_id':batch_id,
        'owner_sha256':transport['owner']['sha256'],'journal':journal.binding()}
    write_json(folder/'prepared_transport_durable_evidence.json',{'manifest':manifest,'owner':owner,'closure':prepared['CPU_actual_closure'],'boundary':durable})
    cleaned=cleanup_owned_cache(owner_record=transport['owner'],manifest_record=transport['manifest'],closure_record=prepared['CPU_actual_closure'],
        boundary=durable,verify_boundary=lambda v:v==durable and journal.verify(v['journal']),guard=stop_check)
    expected_parent=(folder/'decoded').resolve();removed=[]
    for item in prepared['rows']:
        path=Path(item['PCM_cache']['path']).resolve()
        require(path.parent==expected_parent and path.suffix=='.npy' and path.is_relative_to(ROOT.resolve()) and record(path)==item['PCM_cache'],
                'Only exact verified owned clean update cache can be removed')
    for item in prepared['rows']:
        path=Path(item['PCM_cache']['path']).resolve();stop_check();path.unlink();removed.append(str(path))
    write_json(folder/'cache_cleanup.json',{'passed':True,'augmented':cleaned,'clean_exact_files_removed':removed,'boundary':durable,
        'sampler_or_journal_advanced_by_cleanup':False})

def native_update(model,teacher,optimizer,rows,draws,reader,*,torch,population_mean_U,update,stage,stop_check,phase):
    from batched_lora_tr_metadata_v1 import collate_training_batch
    from train_lora_batched_tsc_expanded_TRAIN_v1 import pt_focused_batch_objective
    from tdt_tiled_full_native_v4 import bounded_tdt_loss
    from native_tdt_checkpointed_v4 import reclaim_inactive_cuda_cache
    require(len(rows)==len(draws)>0 and phase=='weak_label_continued_training','Explicit finite weak-phase rows/draws required')
    for row,draw in zip(rows,draws):
        require(row['source_id']==draw['source_id'] and row['language']==draw['language'] and draw['phase']==phase,'Durable ordered identity differs')
        if row['language']=='tr':require(draw['ordinal']==row['broad_original']['ordinal'],'Durable broad ordinal differs')
    native=partial(bounded_tdt_loss,guard=stop_check)
    optimizer.zero_grad();assert_model(model,optimizer,torch);model.train();model.encoder.eval();losses=[]
    try:
        for i,row in enumerate(rows):
            stop_check()
            batch=collate_training_batch(model._broad_processor,[row],blank_id=8192,old_size=8193,device='cuda',audio_reader=reader)
            result=pt_focused_batch_objective(model,teacher,batch,old_size=8193,replay_kl=40.,duration_kl=2.,feature_distillation=20.,
                portuguese_token_kl_multiplier=1.,chunk_size=16,tdt_loss_fn=native,tdt_host_lengths=True)
            result=weight_objective(result,[row],population_mean_U,torch_module=torch)
            require(bool(torch.isfinite(result.total)),'Finite full-target native objective')
            (result.total/len(rows)).backward();losses.append(float(result.total.detach()))
            del result,batch
            # Release only inactive allocator blocks after this graph is gone.
            # All207 gradients/weights/Adam and frozen backing remain live.
            torch.cuda.synchronize();torch.cuda.empty_cache()
            stage('canonical_caller_native_microbatch',update=update,microbatch=i+1,source_id=row['source_id'],language=row['language'])
        payload=reader.prepared_payload();norm=float(optimizer.step());optimizer.zero_grad();torch.cuda.synchronize();torch.cuda.empty_cache()
        require(torch.cuda.memory_reserved()<=3_000_000_000,'Unchanged native end-cache bound')
        assert_model(model,optimizer,torch)
        return {'update':update,'mean_objective':sum(losses)/len(losses),'global_grad_norm':norm,'augmentation':payload,
            'source_ids':[r['source_id'] for r in rows],'languages':[r['language'] for r in rows],
            'prepared_transport':reader.transport_binding,'new_joint_policy':model.joint._ASR18_fp32_joint_policy}
    except BaseException:
        # A partial optimizer commit is already poisoned by its optimizer.
        # No live retry or sampler advancement is attempted here.
        if not optimizer.poisoned:optimizer.zero_grad()
        raise

def restore_branched(path,*,torch,model,optimizer,identity,augmentation,new_journal_path,diagnostic_only,guard):
    import numpy as np
    from bpe_broad_full207_optimizer_v1 import bits_equal
    saved=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    require(saved['kind']=='ASR18_broad_caller_state_v1' and saved['identity']==identity and saved['diagnostic_only']==diagnostic_only,'Exact own checkpoint/config/qualification scope required')
    original=CommitJournal(saved['augmentation']['committed_journal']['path'])
    augmentation.verify_restore(saved['augmentation'],verify_journal_prefix=original.verify)
    journal,branch=branch_checkpoint_prefix(saved['augmentation']['committed_journal'],new_journal_path,guard=guard)
    model.load_state_dict(saved['weights'],strict=False);optimizer.load_state_dict(saved['optimizer'])
    require(all(bits_equal(torch,p,saved['weights'][n]) for n,p in optimizer.parameters.items()),'All207 restored weights byte exact')
    for actual,old in zip(optimizer.optimizers,saved['optimizer']['optimizers']):
        for index,p in enumerate(actual.param_groups[0]['params']):
            require(all(bits_equal(torch,actual.state[p][k],old['state'][index][k]) for k in ('step','exp_avg','exp_avg_sq')),'All3 Adam histories byte exact')
    random.setstate(saved['python_rng']);torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
    n=saved['numpy_rng'];np.random.set_state((n[0],np.array(n[1],dtype=np.uint32),*n[2:]))
    require(random.getstate()==saved['python_rng'] and torch.equal(torch.get_rng_state(),saved['torch_rng']) and
        all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),saved['cuda_rng'])),'Python/Torch/CUDA RNG byte exact')
    state=deepcopy(saved['sampler']);del saved;gc.collect();assert_model(model,optimizer,torch)
    return state,journal,branch

def production_preflight(config,*,guard):
    from bpe_broad_final_population_v2 import final_population
    require(config['kind']=='ASR18_opt_in_broad_production_v5' and config['production_enabled'] is True,'Explicit reviewed production enable required')
    require(config['resource_guards']=={'CUDA_allocator_cap_bytes':8_000_000_000,'full_joint_bytes':1_700_000_000,
        'native_end_cache_bytes':3_000_000_000,'system_commit_yield':.90,'system_commit_critical':.95,'fresh_observations':3,'minimum_headroom_bytes':2_000_000_000},'Exact unchanged guards')
    path=Path(config['final_population_receipt']);require(path.exists(),'Final pinned population unavailable; no partial denominator')
    result=final_population(json.loads(path.read_text()),guard=guard)
    review=Path(config['final_ROOT_review_receipt']);require(review.exists(),'Final exact caller/config ROOT review required')
    value=json.loads(review.read_text());require(value['scope']=='final_production_caller' and value['accepted'] is True,'Finite diagnostic review cannot arm production')
    require(record(__file__) in value['bindings'] and record(config['_path']) in value['bindings'],'Final review does not bind exact caller/config')
    return result
