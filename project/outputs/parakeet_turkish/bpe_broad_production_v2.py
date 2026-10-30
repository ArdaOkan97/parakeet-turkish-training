"""NEW opt-in caller core. Disabled production; finite preparation is separate.

Readonly streamed base, all207 FP32/three Adam histories, transactional draws,
unchanged native/resource bounds. No diagnostic checkpoint is a production seed.
"""
from __future__ import annotations
import gc,hashlib,json,os,random,subprocess,sys,time,threading
from pathlib import Path
from copy import deepcopy
from functools import partial

PROJECT=Path(__file__).resolve().parents[2]
ROOT=Path(r'E:\gpu_training_scratchpad')
SOURCE=Path(__file__).resolve().parent
from atomic_io_retry_v1 import write_json
from bpe_broad_production_state_v1 import CompletePassCursor,CarriedReplay,CommitJournal,canonical,digest,require,final_population,weight_objective
from bpe_broad_audio_v1 import record,BoundViolation

def prohibit_model_transforms(model,torch):
    require(hasattr(model,'_diagnostic_frozen_backing') and len(model._diagnostic_frozen_backing)==294,'Reviewed readonly streamed base required')
    require(not hasattr(model,'_orig_mod') and not torch.distributed.is_initialized(),'Compile/DDP reuse not admitted')
    require(threading.current_thread() is threading.main_thread(),'Serial main-thread caller required')
    def forbidden(*args,**kwargs):raise RuntimeError('Blanket model device/dtype transformation would invalidate readonly backing')
    # Encoder modules and temporary frozen tensor copies retain their own .to;
    # only whole-model transforms are prohibited by this caller.
    for name in ('to','cuda','cpu','half','bfloat16','float','double','_apply'):setattr(model,name,forbidden)

def frozen_parameter(model,name):
    module_path,key=name.rsplit('.',1);module=model.get_submodule(module_path)
    # install_encoder_lora keeps the original hooked Linear object here.
    from encoder_lora import LoRALinear
    if isinstance(module,LoRALinear):module=module.base_layer
    return getattr(module,key)

def assert_model(model,optimizer,torch):
    require(threading.current_thread() is threading.main_thread(),'Only serial forward/hook usage admitted')
    require(len(optimizer.parameters)==207 and all(p.device.type=='cuda' and p.dtype==torch.float32 and p.requires_grad for p in optimizer.parameters.values()),'Active207 GPU FP32 invariant')
    require(all((p:=frozen_parameter(model,n)).device.type=='cpu' and not p.requires_grad and p.data_ptr()==backing.data_ptr()
                for n,backing in model._diagnostic_frozen_backing.items()),'Frozen readonly backing must be restored after forward/backward')
    require(model._diagnostic_frozen_metrics['logical_current_matrix_bytes']==0,'In-flight frozen-matrix hook at commit')

def decode_batch(draws,folder,*,admit,stop_check,heldout_binding=None):
    request=folder/'decode_request.json';write_json(request,{'rows':draws,'heldout_PCM_binding':heldout_binding})
    command=[sys.executable,'-X','utf8','-B',str(PROJECT/'work/research/ASR18_bounded_CPU_phase_v1.py'),
        '--folder',str(folder/'decoder_guard'),'--seconds','300','--script',str(PROJECT/'work/research/decode_ASR18_broad_batch_v1.py'),
        '--','--request',str(request),'--output',str(folder/'decoded')]
    admit('bounded_CPU_decode_'+folder.name,950_000_000)
    environment=os.environ.copy();environment.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (folder/'decoder.log').open('x',encoding='utf-8') as log:
        child=subprocess.Popen(command,cwd=PROJECT,env=environment,stdout=log,stderr=subprocess.STDOUT)
        try:
            started=time.monotonic()
            while child.poll() is None:
                stop_check();require(time.monotonic()-started<310,'Bounded CPU decoder timeout');time.sleep(.2)
            require(child.wait()==0,'Original-source CPU decoder did not close0; no consumed draw')
        finally:
            if child.poll() is None:
                guard_folder=folder/'decoder_guard'
                write_json(guard_folder/'stop.json',{'reason':'owned_GPU_caller_leaving_decode_phase'})
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    import psutil
                    boot=json.loads((guard_folder/'boot.json').read_text())
                    actual=psutil.Process(boot['pid'])
                    require((boot['pid']==child.pid or boot['ppid']==child.pid) and actual.ppid()==boot['ppid'],'Exact owned CPU child identity before stop')
                    actual.terminate();actual.wait(timeout=5);child.wait(timeout=5)
    result=json.loads((folder/'decoded/report.json').read_text());require(result['passed'],'No partial decode admission')
    for item in result['rows']:require(record(item['PCM_cache']['path'])==item['PCM_cache'],'Transient PCM cache changed')
    return result

def read_prepared(row):
    if row['language']!='tr':
        from runtime import read_audio
        return read_audio(row)
    import numpy as np
    path=Path(row['audio_filepath']).resolve();require(path.is_relative_to(ROOT.resolve()),'Bounded E-only transient TR PCM cache')
    audio=np.load(path,allow_pickle=False)
    require(audio.dtype==np.float32 and audio.ndim==1 and len(audio)/16000==row['duration'],'Exact prepared full PCM identity')
    return audio

def augmentation_binding(config,guard):
    from fastmss_train_augmentation_v1 import AugmentationPolicy,TrainWaveformAugmenter
    from fastmss_ordered_draw_binding_v1 import FastMSSOrderedDrawBinding
    policy_path=config['augmentation_policy'];policy=AugmentationPolicy.from_json(policy_path)
    augmenter=TrainWaveformAugmenter(policy,guard=guard)
    return FastMSSOrderedDrawBinding(augmenter,policy_path=policy_path,adapter_path=SOURCE/'fastmss_train_augmentation_v1.py')

def checkpoint(path,*,torch,optimizer,state,provenance,identity,augmentation,journal,diagnostic_only):
    import numpy as np
    optimizer.zero_grad();numpy_state=np.random.get_state()
    value={'kind':'ASR18_broad_caller_state_v1','diagnostic_only':diagnostic_only,'identity':identity,'source_provenance':provenance,
        'weights':{n:p.detach().cpu() for n,p in optimizer.parameters.items()},'optimizer':optimizer.state_dict(),
        'sampler':deepcopy(state),'augmentation':augmentation.checkpoint_binding(committed_journal=journal.binding()),
        'python_rng':random.getstate(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
        'numpy_rng':(numpy_state[0],numpy_state[1].tolist(),*numpy_state[2:])}
    temporary=path.with_suffix('.writing.pt');torch.save(value,temporary)
    with temporary.open('r+b') as stream:os.fsync(stream.fileno())
    temporary.replace(path);del value;gc.collect();return record(path)

def restore(path,*,torch,model,optimizer,identity,augmentation,journal,diagnostic_only):
    import numpy as np
    from bpe_broad_full207_optimizer_v1 import bits_equal
    saved=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    require(saved['kind']=='ASR18_broad_caller_state_v1' and saved['identity']==identity and saved['diagnostic_only']==diagnostic_only,'Own exact caller checkpoint/config scope required; diagnostic state cannot seed production')
    augmentation.verify_restore(saved['augmentation'],verify_journal_prefix=journal.verify)
    require(journal.binding()==saved['augmentation']['committed_journal'],
        'Bounded in-place restore requires no journal tail; future production must preserve and branch any uncheckpointed tail')
    model.load_state_dict(saved['weights'],strict=False);optimizer.load_state_dict(saved['optimizer'])
    require(all(bits_equal(torch,p,saved['weights'][n]) for n,p in optimizer.parameters.items()),'All207 restored weights must be byte exact')
    for actual,old in zip(optimizer.optimizers,saved['optimizer']['optimizers']):
        for index,p in enumerate(actual.param_groups[0]['params']):
            require(all(bits_equal(torch,actual.state[p][k],old['state'][index][k]) for k in ('step','exp_avg','exp_avg_sq')),'All3 Adam clocks/moments must restore byte exact')
    random.setstate(saved['python_rng']);torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
    n=saved['numpy_rng'];np.random.set_state((n[0],np.array(n[1],dtype=np.uint32),*n[2:]))
    require(random.getstate()==saved['python_rng'] and torch.equal(torch.get_rng_state(),saved['torch_rng']) and
            all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),saved['cuda_rng'])),'Python/Torch/CUDA RNG restore must be exact')
    state=deepcopy(saved['sampler']);del saved;gc.collect();assert_model(model,optimizer,torch)
    return state

def native_update(model,teacher,optimizer,rows,draws,*,torch,population_mean_U,augmentation,journal,update,stage,stop_check,phase):
    from batched_lora_tr_metadata_v1 import collate_training_batch
    from train_lora_batched_tsc_expanded_TRAIN_v1 import pt_focused_batch_objective
    from native_tdt_checkpointed_v4 import chunked_tdt_loss,reclaim_inactive_cuda_cache
    from fastmss_ordered_draw_binding_v1 import AugmentationDrawContext
    native=partial(chunked_tdt_loss,sigma=0.,omega=0.,recompute=True,max_joint_bytes=1_700_000_000)
    require(len(rows)==len(draws) and len(rows)>0,'Exact ordered row/durable draw count required')
    require(phase in ('weak_label_continued_training','cleaner_data_fine_tuning'),'Explicit training phase required')
    for row,draw in zip(rows,draws):
        require(draw['source_id']==row['source_id'] and draw['language']==row['language'] and draw['phase']==phase,'Actual row differs from independent durable draw identity')
        if row['language']=='tr':require(draw['ordinal']==row['broad_original']['ordinal'],'Exact broad catalog ordinal/draw differs')
    contexts=tuple(AugmentationDrawContext(draw['source_id'],draw['language'],draw['epoch'],draw['draw_index'],phase=phase) for draw in draws)
    prepared=[];reader=augmentation.prepare_batch(read_prepared,contexts,batch_id=str(update),record_prepared=prepared.append)
    before=(random.getstate(),torch.get_rng_state(),torch.cuda.get_rng_state_all());losses=[]
    optimizer.zero_grad();assert_model(model,optimizer,torch);model.train();model.encoder.eval()
    try:
        for i,row in enumerate(rows):
            stop_check();joint=row['encoder_frames']*(row['target_tokens']+1)*8716*4
            if joint>1_700_000_000:raise BoundViolation('native_full_joint_bytes',row['source_id'],joint,1_700_000_000)
            batch=collate_training_batch(processor=model._broad_processor,rows=[row],blank_id=8192,old_size=8193,device='cuda',audio_reader=reader)
            result=pt_focused_batch_objective(model,teacher,batch,old_size=8193,replay_kl=40.,duration_kl=2.,feature_distillation=20.,
                portuguese_token_kl_multiplier=1.,chunk_size=16,tdt_loss_fn=native,tdt_host_lengths=True)
            result=weight_objective(result,[row],population_mean_U,torch_module=torch)
            require(bool(torch.isfinite(result.total)),'Finite native objective required')
            (result.total/len(rows)).backward();losses.append(float(result.total.detach()))
            del result,batch;reclaim_inactive_cuda_cache(torch.device('cuda'),'broad_after_serial_clip_backward')
            stage('caller_native_microbatch',update=update,microbatch=i+1,source_id=row['source_id'],language=row['language'],full_joint_bytes=joint)
        payload=reader.prepared_payload();norm=optimizer.step();optimizer.zero_grad();torch.cuda.synchronize();torch.cuda.empty_cache()
        assert_model(model,optimizer,torch)
        return {'update':update,'mean_objective':sum(losses)/len(losses),'global_grad_norm':float(norm),'augmentation':payload,
                'source_ids':[r['source_id'] for r in rows],'languages':[r['language'] for r in rows]}
    except BaseException:
        optimizer.zero_grad();random.setstate(before[0]);torch.set_rng_state(before[1]);torch.cuda.set_rng_state_all(before[2]);torch.cuda.empty_cache();raise

def prepare_probe(model,processor,teacher,optimizer,provenance,replay_snapshot,folder,config,*,torch,stage,admit,stop_check):
    """Exactly two native updates, original sources; fixed2TR+all6 replay only."""
    from tsc_bpe_full_decoder_transition_v1 import preflight
    from train_lora_batched_tsc_bpe_full_decoder_v1 import load_catalogs
    from train_lora_batched_data_v11 import BucketCycler
    from types import SimpleNamespace
    source_config=json.loads((ROOT/'runs/adaptive_20261015_TR_BPE512_full_decoder_02/run_config.json').read_text())
    context=preflight(SimpleNamespace(**source_config['arguments']));original,tr,replay=load_catalogs(context)
    carried=CarriedReplay(replay,replay_snapshot,BucketCycler);replay_before=carried.state_dict()
    expected=[r['source_id'] for r in carried.next()];carried.load_state_dict(replay_before)
    require([r['source_id'] for r in carried.next()]==expected,'Carried six queues/selector RNG failed exact next-draw restore');carried.load_state_dict(replay_before)
    evidence=json.loads(Path(config['probe_reader_proof']).read_text());require(evidence['passed'],'Actual source reader proof required')
    broad=evidence['observations'];require(len(broad)==2,'Fixed two-source fixture, never production corpus')
    draws=[{'ordinal':v['row']['ordinal'],'epoch':0,'draw_index':i} for i,v in enumerate(broad)]
    decoded=decode_batch(draws,folder,admit=admit,stop_check=stop_check)
    require(all(item['row']['broad_original']==old['row'] for item,old in zip(decoded['rows'],broad)),
        'Decoded original rows differ from pinned proof fixture')
    rows=[item['row'] for item in decoded['rows']]
    rows.extend(min((r for r in replay if r['language']==l),key=lambda r:r['encoder_frames']*(r['target_tokens']+1)) for l in ('en','es','it','fr','de','pt'))
    fixture_mean=sum(r['target_tokens'] for r in rows[:2])/2
    declaration={'kind':'bounded_new_caller_preparation','production_training':False,'updates_bound':2,'rows':8,'TR_replay':[2,6],
        'all_six_replay_languages':True,'physical_batch':1,'diagnostic_accumulation':8,'production14TR2replay_equivalence_claimed':False,
        'fixture_only_population_mean_U':fixture_mean,'final_broad_population_mean_U':None,'complete_broad_passes':0,
        'parent_original_sources':provenance['source_records'],'diagnostic20_checkpoint_used':False,'reader_proof':record(config['probe_reader_proof']),
        'ordered_fixture_draw_identity':[{'source_id':r['source_id'],'language':r['language'],
            'ordinal':r['broad_original']['ordinal'] if r['language']=='tr' else None,'phase':config['phase']} for r in rows]}
    write_json(folder/'declaration.json',declaration);identity={'config':record(config['_path']),'source':record(__file__),'declaration':declaration}
    journal=CommitJournal(folder/'committed_updates.jsonl');augmentation=augmentation_binding(config,stop_check)
    prohibit_model_transforms(model,torch);model._broad_processor=processor
    history=[];state={'fixture_committed_draws':0,'replay_source_restored':True};next_hash=None
    try:
        for update in (1,2):
            admit('native_update_new_caller_'+str(update),2_000_000_000 if update==1 else 500_000_000)
            row_draws=[{**draw,'epoch':update-1,'draw_index':state['fixture_committed_draws']+i} for i,draw in enumerate(declaration['ordered_fixture_draw_identity'])]
            item=native_update(model,teacher,optimizer,rows,row_draws,torch=torch,population_mean_U=fixture_mean,
                augmentation=augmentation,journal=journal,update=update,stage=stage,stop_check=stop_check,phase=config['phase'])
            state['fixture_committed_draws']+=8;history.append(item);journal.append({'kind':'committed_native_update','sampler':state,**item})
            stage('new_caller_update_committed',update=update,clocks=[c+optimizer.updates for c in (64250,2000,62250)])
            if update==1:
                admit('new_caller_checkpoint',300_000_000)
                from fastmss_ordered_draw_binding_v1 import AugmentationDrawContext
                next_draw=declaration['ordered_fixture_draw_identity'][0]
                context_=AugmentationDrawContext(next_draw['source_id'],next_draw['language'],1,state['fixture_committed_draws'],phase=next_draw['phase'])
                pending=augmentation.prepare_batch(read_prepared,(context_,),batch_id='resume_next_draw',record_prepared=lambda value:None)
                wave=pending(rows[0]);next_hash=hashlib.sha256(wave.tobytes()).hexdigest();del wave,pending
                path=folder/'after_update1.pt';checkpoint(path,torch=torch,optimizer=optimizer,state=state,provenance=provenance,identity=identity,augmentation=augmentation,journal=journal,diagnostic_only=True)
                admit('new_caller_exact_restore',350_000_000)
                saved_state=deepcopy(state);state=restore(path,torch=torch,model=model,optimizer=optimizer,identity=identity,augmentation=augmentation,journal=journal,diagnostic_only=True)
                require(saved_state==state and optimizer.updates==1,'Own native draw frontier/Adam restore failed')
                # Compare actual pre-save expectation with post-restore draw.
                pending=augmentation.prepare_batch(read_prepared,(context_,),batch_id='resume_next_draw',record_prepared=lambda value:None)
                wave=pending(rows[0]);require(hashlib.sha256(wave.tobytes()).hexdigest()==next_hash,'Next waveform draw changed across actual checkpoint restore');del wave,pending
        final=checkpoint(folder/'after_update2.pt',torch=torch,optimizer=optimizer,state=state,provenance=provenance,identity=identity,augmentation=augmentation,journal=journal,diagnostic_only=True)
        applied=sum(event['augmentation']['applied'] for update in history for event in update['augmentation']['rows'])
        require(applied>0,'New broad caller needs an actual transformed TR waveform; all-clean controls do not prove this hook')
        report={'passed':True,'native_updates':2,'languages':sorted(set(history[0]['languages'])),'history':history,'checkpoint':final,'actual_augmented_waveforms':applied,
            'checkpoint_all207_Adam_RNG_restore_byte_exact':True,'next_draw_waveform_sha256':next_hash,'original_replay_queues_and_RNG_restore_next_draw_exact':True,
            'final_clocks':[64252,2002,62252],'augmentation_binding':augmentation.identity,'production_ready':False,'diagnostic_only':True,'declaration':declaration}
        write_json(folder/'production_caller_probe.json',report);return report
    finally:
        if tr is not original:tr.close()
        original.close()

def production_preflight(config):
    require(config['kind']=='ASR18_opt_in_broad_production_v1','Exact production config kind')
    require(config['resource_guards']=={'CUDA_allocator_cap_bytes':8_000_000_000,'full_joint_bytes':1_700_000_000,
        'native_end_cache_bytes':3_000_000_000,'system_commit_yield':.90,'system_commit_critical':.95,
        'fresh_observations':3,'minimum_headroom_bytes':2_000_000_000},'Unchanged guards required')
    path=Path(config['final_population_receipt']);require(path.exists(),'Full pinned corpus/geometry/heldout final receipt is unavailable; partial statuses cannot arm production')
    receipt=json.loads(path.read_text());return final_population(receipt,verify_record=lambda item:record(item['path'])==item)
