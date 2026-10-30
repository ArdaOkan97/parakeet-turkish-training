"""Selected original8199/all207 recipe: finish first pass, then two more passes."""
from __future__ import annotations
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import random
import time
from functools import partial
import torch
from atomic_io_retry_v1 import read_json, write_json, retry_replace
from append_only_history_v1 import HistoryJournal
from tsc_three_epoch_transition_v1 import (
    SOURCE, ROOT, STORAGE, PHASE, SOURCE_STEP, END, UPDATES, MILESTONES,
    preflight, audit_parent_state, compare_source_development, proof_path,
    encoder_lr, declaration, record, checked, canonical, require,
)
from tsc_three_epoch_sampler_v1 import ThreeEpochSampler
from full207_three_epoch_optimizer_v1 import Full207Optimizer
from train_lora_batched_tsc_bpe_full_decoder_v1 import (
    stage, load_catalogs, evaluate, gate, diagnostic_predictions,
)
from train_lora_batched_tsc_expanded_TRAIN_v1 import pt_focused_batch_objective
from batched_lora_tr_metadata_v1 import collate_training_batch
from TR_tdt_population_weighting_v1 import apply_to_objective, detached_metrics
from native_tdt_checkpointed_v4 import chunked_tdt_loss, reclaim_inactive_cuda_cache
from encoder_lora import install_encoder_lora, encoder_lora_metadata
from extend_vocab import extend_processor_and_model
from train_lora import checkpoint_encoder_blocks
from train_lora_batched_data_v11 import BucketCycler
from runtime import load_model

STATE_KIND = "TSC_full207_three_epoch_coherent_state_v1"


def source_pins():
    names = [Path(__file__).name,"tsc_three_epoch_transition_v1.py","tsc_three_epoch_sampler_v1.py",
        "full207_three_epoch_optimizer_v1.py","train_lora_batched_tsc_bpe_full_decoder_v1.py",
        "train_lora_batched_tsc_expanded_TRAIN_v1.py","tsc_bpe_full_decoder_transition_v1.py",
        "tsc_bpe_full_decoder_sampler_v1.py","tsc_expanded_TRAIN_sampler_v1.py",
        "batched_lora_tr_metadata_v1.py","TR_tdt_population_weighting_v1.py","encoder_lora.py",
        "extend_vocab.py","train_lora.py","native_tdt_checkpointed_v4.py","append_only_history_v1.py",
        "lazy_training_rows_v1.py","expanded_training_rows_v1.py","bpe_full_decoder_optimizer_v1.py"]
    return {name:record(SOURCE/name)["sha256"] for name in names}


def native_identity(args):
    value=read_json(args.native_backend_provenance)
    require(value["passed"] and value["pid"]==os.getpid() and value["ppid"]==os.getppid()
        and value["run_id"]==args.run_id and value["source_receipt"]==record(args.training_source_receipt)
        and vars(parser().parse_args(value["worker_arguments"]))==vars(args)
        and value["worker_sha256"]==record(__file__)["sha256"]
        and value["CUDA_version"]=="12.9" and value["native_batch_cap_bytes"]==8000000000
        and value["native_end_cache_bytes"]==3000000000
        and value["maximum_full_joint_bytes"]==1700000000,"Actual fresh owned CUDA12.9 native provenance required")
    return value


def build_sampler(original,tr,replay,context):
    a=context["source_arguments"]
    return ThreeEpochSampler(original,tr,[r for r in replay if r["language"]!="pt"],
        portuguese_replay=[r for r in replay if r["language"]=="pt"],
        portuguese_seed=a["portuguese_seed"],data_arm="expanded",seed=a["seed"],replay_seed=a["replay_seed"],
        sampling_mode="uniform",batch_size=8,accumulation=2,replay_fraction=.125,
        fleurs_fraction=a["fleurs_fraction"],bucket_size=a["bucket_size"],cycler_factory=BucketCycler,
        cycler_source_sha256=record(SOURCE/"train_lora_batched_data_v11.py")["sha256"])


def validate_own(state,args,context,identity,optimizer):
    require(state["kind"]==STATE_KIND and state["run_id"]==args.run_id and state["schema"]==2
        and canonical(state["config_identity"])==canonical(identity)
        and state["source_checkpoint"]==context["source_checkpoint"]
        and SOURCE_STEP<=state["step"]<=END and len(state["history"])==state["step"]
        and set(state["weights"])==set(optimizer.names),"Own coherent full207 source/frontier/identity differs")
    require(canonical(state["history"][:SOURCE_STEP])==context["source_history_prefix_sha256"],"Original immutable history prefix changed")
    score, winner, failures = 259/825, SOURCE_STEP, 0
    for row in state["history"][SOURCE_STEP:]:
        if "validation" not in row:continue
        v=row["validation"]; passed=all(v["retention_word_errors"][k]<=n for k,n in dict(en=69,es=52,it=30,fr=74,de=70,pt=66).items())
        require(passed==v["retention_gate_passed"],"Own strict integer gate accounting changed")
        failures=0 if passed else failures+1
        if passed and v["turkish_wer"]<score:score,winner=v["turkish_wer"],row["step"]
    require(state["best_score"]==score and state["selected_step"]==winner and state["retention_failures"]==failures,
            "Own champion/monitored counter changed")
    for n,w in state["weights"].items():
        require(w.shape==optimizer.parameters[n].shape and w.dtype==torch.float32 and bool(torch.isfinite(w).all()),"Own actual207 geometry/finiteness changed")


def run(args):
    context=preflight(args); fresh_native=native_identity(args)
    run_dir=Path(args.run_directory)
    require(bool(args.resume_exact)==run_dir.is_dir(),"Fresh or exact namespace required")
    if not args.resume_exact:run_dir.mkdir(parents=True)
    memory_trace=run_dir/"stage_memory.jsonl"
    stage(memory_trace,"before_original_base_load")
    model,processor=load_model()
    teacher={n:copy.deepcopy(getattr(model,n)).eval().requires_grad_(False) for n in ("encoder_projector","decoder","joint")}
    extend_processor_and_model(processor,model)
    a=context["source_arguments"]
    layout=install_encoder_lora(model,rank=a["rank"],alpha=a["alpha"],last_n_layers=a["last_n_layers"])
    model.config.encoder_lora_tracking=dict(layout=layout,implementation_sha256=record(SOURCE/"encoder_lora.py")["sha256"],kind="unmerged")
    checkpoint_encoder_blocks(model)
    parent_path=checked(context["source_checkpoint"],checkpoint=True)
    require(record(parent_path)==context["source_checkpoint"],"Immutable original source checkpoint changed")
    source=torch.load(parent_path,map_location="cpu",weights_only=True)
    audit=audit_parent_state(model,source,context,torch_module=torch)
    model.load_state_dict(source["weights"],strict=False)
    optimizer=Full207Optimizer(model,source,context,torch)
    original,tr,replay=load_catalogs(context)
    sampler=build_sampler(original,tr,replay,context)
    source.pop("weights");source.pop("optimizer");gc.collect()
    stage(memory_trace,"after_actual207_Adam_GPU_copy_and_CPU_tensor_release")
    if args.resume_exact:
        source.clear();gc.collect()
        config=read_json(run_dir/"run_config.json");identity=config["config_identity"]
        require(record(args.resume_exact)==fresh_native["exact_checkpoint"],"Owned exact checkpoint bytes changed")
        own=torch.load(args.resume_exact,map_location="cpu",weights_only=True)
        validate_own(own,args,context,identity,optimizer)
        model.load_state_dict(own["weights"],strict=False)
        optimizer.load_state_dict(own["optimizer"],own["step"])
        migration=sampler.initialize_from_own_state(own["sampler"],context,identity["domain_sampling_identity"])
        own.pop("weights");own.pop("optimizer")
        runtime=own;equality=read_json(proof_path(args))
        require(equality["passed"] and source_pins()==identity["source_sha256"],"Exact admitted source or initial350 changed")
    else:
        migration=sampler.migrate_closed_state(source["sampler"],context)
        random.setstate(source["python_rng"]);torch.set_rng_state(source["torch_rng"]);torch.cuda.set_rng_state_all(source["cuda_rng"])
        equality=compare_source_development(model,processor,args,context,source,evaluate,torch)
        identity=dict(phase=PHASE,run_id=args.run_id,declaration=declaration(),source_checkpoint=context["source_checkpoint"],
            source_receipt=context["receipt_record"],source_sha256=source_pins(),
            domain_sampling_identity=sampler.identity,native_loss=native_identity(args),initial350=record(proof_path(args)))
        config=dict(phase=PHASE,run_id=args.run_id,source_step=SOURCE_STEP,configured_end_step=END,
            config_identity=identity,initial_source_equivalence=equality,source207_audit=audit,
            sampler_transition=migration,all207_active=True,original8199=True,retention_stop_policy="monitor_only",
            arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
        write_json(run_dir/"run_config.json",config)
        write_json(run_dir/"training_pool.json",dict(source_training_pool=context["source_training_pool"],
            accepted_TR_clips=len(tr),accepted_TR_hours=675.0964,no_new_data_added=True))
        runtime=source
    history=runtime["history"];start=runtime["step"]
    require(sampler.updates==optimizer.global_step==start,"All207/global/sampler restore clocks must agree")
    random.setstate(runtime["python_rng"]);torch.set_rng_state(runtime["torch_rng"]);torch.cuda.set_rng_state_all(runtime["cuda_rng"])
    keys=("step","history","best_score","selected_step","patience","retention_failures","next_eval_epoch",
          "train_wall_seconds","audio_seconds","language_audio_seconds","evaluation_wall_seconds","initial_gate")
    live={k:runtime[k] for k in keys}
    del runtime,source
    if args.resume_exact:del own
    gc.collect()
    journal=HistoryJournal(run_dir/"history_journal")
    if args.resume_exact:journal.reconcile_checkpoint(history,record(args.resume_exact),preserve_orphan_suffix=True)
    else:journal.seed(history,context["source_checkpoint"])
    write_json(Path(args.native_backend_provenance).parent/"exact_restore.json",dict(passed=True,run_id=args.run_id,
        restored_step=start,all207_Adam_clock=optimizer.global_step,sampler_clock=sampler.updates,
        retention_failures=live["retention_failures"],selected_step=live["selected_step"],sampler_migration=migration,
        no_weights_optimizer_RNG_history_counter_reset=True,actual_PID=os.getpid(),all207_active=True))
    def publish(label):
        journal.publish(history,publication=label)
        if label!="periodic":write_json(run_dir/"training_history.json",history)
    def save(path,step,reason=None):
        live.update(step=step,history=history,stop_reason=reason)
        weights={n:p.detach().cpu() for n,p in model.named_parameters() if n in optimizer.parameters}
        require(len(weights)==207,"Full207 checkpoint required")
        state=dict(kind=STATE_KIND,schema=2,run_id=args.run_id,weights=weights,optimizer=optimizer.state_dict(),
            sampler=sampler.state_dict(),source_checkpoint=context["source_checkpoint"],config_identity=identity,
            lora_metadata=encoder_lora_metadata(model),python_rng=random.getstate(),torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(),**live)
        tmp=Path(str(path)+".tmp");torch.save(state,tmp)
        with tmp.open("r+b") as f:os.fsync(f.fileno())
        retry_replace(tmp,path)
    best_path,latest_path=run_dir/"best_checkpoint.pt",run_dir/"latest_checkpoint.pt"
    if not args.resume_exact:
        save(best_path,start,"actual_qualified_source207");save(latest_path,start)
    publish("checkpoint")
    stage(memory_trace,"training_ready",step=start,all207_Adam_clock=optimizer.global_step,source350_passed=equality["passed"])
    native_loss=partial(chunked_tdt_loss,sigma=0.,omega=0.,recompute=True,max_joint_bytes=1700000000)
    last_dev=start
    def dev(step,final=False):
        nonlocal last_dev
        optimizer.zero_grad(set_to_none=True);torch.cuda.empty_cache()
        before=time.perf_counter();scores=evaluate(model,processor,args.run_id+"_step_"+str(step))
        live["evaluation_wall_seconds"]+=time.perf_counter()-before
        value=gate(scores)
        history[-1].update(validation=value,development_selection_only=True,final_development_evaluation=final,
            selection_diagnostics=diagnostic_predictions(args.run_id+"_step_"+str(step)))
        live["retention_failures"]=0 if value["retention_gate_passed"] else live["retention_failures"]+1
        improved=value["retention_gate_passed"] and value["turkish_wer"]<live["best_score"]
        if improved:live.update(best_score=value["turkish_wer"],selected_step=step,patience=0)
        elif step in MILESTONES:live["patience"]+=1
        publish("DEV")
        if improved:save(best_path,step,"strict_all_six_qualified_improvement")
        save(latest_path,step)
        write_json(run_dir/"selection.json",dict(run_id=args.run_id,phase=PHASE,selected_step=live["selected_step"],
            turkish_validation_wer=live["best_score"],latest_step=step,latest_gate=value,
            retention_failures=live["retention_failures"],coverage=sampler.coverage(),strict_all_six_winner_only=True,
            best_checkpoint=str(best_path),all207_Adam_clock=optimizer.global_step))
        last_dev=step
        print(f"DEV step{step} TR{value['turkish_wer']:.6f} all6={value['retention_gate_passed']} best{live['selected_step']}",flush=True)
        model.train();model.encoder.eval();torch.cuda.empty_cache()
    model.train();model.encoder.eval();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    stop_reason="configured_steps_complete"
    for step in range(start+1,END+1):
        if args.stop_file and args.stop_file.exists():stop_reason="controller_stop_file";break
        before=time.perf_counter();optimizer.zero_grad(set_to_none=True)
        totals={};ids=[];actual_TR_weights=[];sampling_target_tokens=[]
        for rows in sampler.next_batches():
            ids.extend(r["source_id"] for r in rows)
            sampling_target_tokens.extend(int(r["target_tokens"]) for r in rows)
            batch=collate_training_batch(processor,rows,blank_id=8192,old_size=8193,device="cuda")
            result=pt_focused_batch_objective(model,teacher,batch,old_size=8193,replay_kl=40,
                duration_kl=2,feature_distillation=20,portuguese_token_kl_multiplier=1,
                chunk_size=16,tdt_loss_fn=native_loss,tdt_host_lengths=True)
            result,exposure=apply_to_objective(result,rows,mode="token_population",population_mean_U=26.642061797177284,torch_module=torch)
            actual_TR_weights.extend(exposure["actual_TR_weights"])
            require(bool(torch.isfinite(result.total)),"Nonfinite objective")
            (result.total/2).backward()
            for k,v in detached_metrics(result,exposure,torch_module=torch).items():totals[k]=totals.get(k,0.)+v/2
            for row in rows:
                seconds=float(row["duration"]);live["audio_seconds"]+=seconds;live["language_audio_seconds"][row["language"]]+=seconds
            del result,batch
            reclaim_inactive_cuda_cache(torch.device("cuda"),"after_whole_microbatch_backward")
        fingerprints=optimizer.fingerprints() if step-start<=2 else None
        norm=optimizer.step()
        reclaim_inactive_cuda_cache(torch.device("cuda"),"after_adam_and_new_rows")
        torch.cuda.synchronize()  # One actual complete all207/native update.
        live["train_wall_seconds"]+=time.perf_counter()-before;live["step"]=step
        history.append(dict(step=step,phase=PHASE,head_lr=1e-5,lora_lr=encoder_lr(step),
            all207_Adam_clock=optimizer.global_step,encoder_LoRA_active=True,gradient_norm=float(norm),**totals,
            TR_tdt_weighting="token_population",TR_tdt_population_mean_U=26.642061797177284,
            actual_TR_tdt_weights=actual_TR_weights,sampling_target_tokens=sampling_target_tokens,
            source_ID_sequence_sha256=canonical(ids),coverage=sampler.coverage(),
            train_wall_seconds=live["train_wall_seconds"],evaluation_wall_seconds=live["evaluation_wall_seconds"],
            audio_seen_seconds=live["audio_seconds"],language_audio_seen_seconds=dict(live["language_audio_seconds"])))
        if fingerprints is not None:
            after=optimizer.fingerprints();changed=[n for n in fingerprints if fingerprints[n]!=after[n]]
            require(any(n in optimizer.head_names for n in changed) and any(n in optimizer.lora_names for n in changed),"Actual decoder and encoder-adapter changes required")
            stage(memory_trace,"after_actual_full207_optimizer",step=step,all207_Adam_clock=optimizer.global_step,
                changed_head_tensors=sum(n in optimizer.head_names for n in changed),
                changed_encoder_LoRA_tensors=sum(n in optimizer.lora_names for n in changed),
                actual_native_backward_optimizer_update=True,head_lr=1e-5,lora_lr=encoder_lr(step))
        if step%10==0:
            publish("periodic");print(f"optimizer step{step} continuous3pass/{END}",flush=True)
        if step%250==0 or step in MILESTONES:dev(step)
    completed=live["step"]
    if last_dev!=completed:dev(completed,final=True)
    elif completed>SOURCE_STEP:history[-1]["final_development_evaluation"]=True
    publish("end");save(latest_path,completed,stop_reason)
    write_json(run_dir/"worker_completion.json",dict(run_id=args.run_id,phase=PHASE,completed_step=completed,
        training_updates=completed-SOURCE_STEP,stop_reason=stop_reason,final350_complete=last_dev==completed,
        all207_active=True,all207_Adam_clock=optimizer.global_step,source350=record(proof_path(args)),
        sources_unchanged=source_pins()==identity["source_sha256"],selected_step=live["selected_step"],
        best_TR_WER=live["best_score"],coverage=sampler.coverage(),actual_PID=os.getpid()))
    if tr is not original:tr.close()
    original.close()


def parser():
    p=argparse.ArgumentParser()
    p.add_argument("--run-id",required=True);p.add_argument("--run-directory",type=Path,required=True)
    p.add_argument("--training-source-receipt",type=Path,required=True)
    p.add_argument("--training-updates",type=int,default=UPDATES);p.add_argument("--shared-lr",type=float,default=1e-5)
    p.add_argument("--batch-size",type=int,default=8);p.add_argument("--accumulation",type=int,default=2)
    p.add_argument("--native-max-joint-bytes",type=int,default=1700000000)
    p.add_argument("--native-backend-provenance",type=Path,required=True)
    p.add_argument("--resume-exact",type=Path);p.add_argument("--stop-file",type=Path)
    return p


def validate_arguments(args):preflight(args);native_identity(args)


if __name__=="__main__":
    args=parser().parse_args();validate_arguments(args);run(args)
