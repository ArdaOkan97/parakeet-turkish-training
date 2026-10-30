"""Inactive CLOSED TSC BEST -> row-only BPE512 acquisition boundary.

No ML import or checkpoint/audio read occurs at module import. ROOT must supply
an actual closed-source receipt before preflight; no current source is admitted
by this file's existence. Runtime tensor work is caller-injected and executes
only inside the future dispatched worker. This is a declared lexical/optimizer
transition, not SAME-science resume or an old74883-only BPE warm start.
"""
from __future__ import annotations
import copy
from functools import lru_cache
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PARENT_RUN = "adaptive_20261011_hf_TSC_headlr02_01"
PARENT_SOURCES = {
    "train_lora_batched_tsc_headlr_v1.py": "40d81ba6f28136b4a95feec371a582bc86754d0c7edbb13eac37b892ce58b8d5",
    "tsc_optimization_transition_v1.py": "23292718104bc14098f45b6d637b2abdc84916812dac1330c0f26bc646cf8669",
    "run_TSC_headlr_fork_v1.py": "aed37d34bee878e51e9ec906023d97c132af83b11bb7f6989b0ed3cf7d1a7f8d",
}
WORKER = "train_lora_batched_tsc_bpe_acquire_v1.py"
OUTER = "run_TSC_bpe_acquire_v1.py"
PROOF_NAME = "TSC_BPE_acquisition_initial_source_equivalence.json"
LANGUAGES = ("tr", "en", "es", "it", "fr", "de")
TABLES = ("decoder.embedding.weight", "joint.head.weight", "joint.head.bias")
IDENTITY_KEYS = ("source_id", "sentence_id", "language", "split", "reference",
                 "dataset_revision", "audio_sha256", "wav_sha256", "duration")
TOKEN_KEYS = ("token_id", "duration_id", "piece", "start_frame", "duration_frames")
OLD_VOCAB, NEW_VOCAB, BLANK_ID = 8199, 8711, 8192
NATIVE_MAX_BYTES = 1_850_000_000
CONVERTER_SHA = "2ea9a1e221b74dd9ed5705f7408e1bb56f39821fcfb2a303dae70c9c5121078d"
CONSUMER_SHA = "98d79657035be7ed59a9c372ac5335755391456563167b891fd883fb89b7ccf7"
SAMPLER_MIGRATION_SHA = "f43f6bd46ebbad0192bafdb66daa1a5414f2ca6b1d54f451f943c9268da3140e"


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream: return hashlib.file_digest(stream,"sha256").hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def scoped(path):
    path=Path(path).resolve(); require(path.is_relative_to(ROOT),"Evidence must stay within workspace"); return path


def read(path):
    return json.loads(scoped(path).read_text(encoding="utf-8"))


def record(path):
    path=scoped(path);return {"path":str(path),"sha256":sha(path),"bytes":path.stat().st_size}


def checked_record(value, *, checkpoint=False):
    path=scoped(value["path"])
    require(set(value)=={"path","sha256","bytes"} and re.fullmatch("[0-9a-f]{64}",value["sha256"])
        and path.stat().st_size==value["bytes"],"Invalid/changed CLOSED byte record")
    # preflight never reads checkpoint bytes; the worker hashes once at runtime.
    if not checkpoint: require(record(path)==value,"Immutable CLOSED artifact changed: "+str(path))
    return path


def acquisition_declaration():
    return {"learning_rate":1e-4,"weight_decay":0.0,"trainable_tables":list(TABLES),"trainable_rows":[8199,8711],
        "source_Adam_retained_dormant":True,"original_shared_LoRA_six_blank_duration_rows_frozen":True,
        "source_sampler_RNG_counters_preserved_with_declared_TR_target_fingerprint_transition":True,
        "same_science_resume_claimed":False,"native_max_joint_bytes":NATIVE_MAX_BYTES}


DECLARED_ACQUISITION = acquisition_declaration()
declared_acquisition = acquisition_declaration


def reviewed_resource_stop(receipt):
    """Only ROOT's recorded34624 memory stop; preserve actual exit1/status."""
    reference=receipt.get("resource_stop_review")
    require(isinstance(reference,dict),"Explicit hash-bound ROOT memory review required")
    review=read(checked_record(reference))
    require(review.get("kind")=="ROOT_headlr_memory_resource_stop_review_v1" and review.get("passed") is True
        and review.get("source_run_id")==PARENT_RUN
        and review.get("source_controller_sha256")==PARENT_SOURCES["run_TSC_headlr_fork_v1.py"]
        and review.get("allow_BPE512_row_only_acquisition") is True and review.get("guards_retained") is True,
        "Specific reviewed headLR memory source/guard approval required")
    owned=review.get("owned_actual_exit",{})
    require(owned.get("session_id")==34624 and owned.get("exit_code")==1 and owned.get("chunk_id")=="b84373",
        "Only the explicit ROOT-observed34624 exit1/b84373 memory event is admitted")
    observed=datetime.fromisoformat(str(owned.get("observed_at_utc","")).replace("Z","+00:00"))
    require(observed.tzinfo is not None and observed.astimezone(timezone.utc)==datetime(2026,10,11,22,0,3,tzinfo=timezone.utc),
        "ROOT memory-review owned-exit timestamp differs")
    artifact=receipt["artifacts"]["controller_status"]
    reviewed=review["reviewed_status_artifact"]
    controller=read(checked_record(reviewed));checked_record(artifact)
    require(reviewed["sha256"]==artifact["sha256"] and reviewed["bytes"]==artifact["bytes"]
        and scoped(reviewed["path"])==scoped(receipt.get("original_artifact_paths",{}).get("controller_status",artifact["path"])),
        "Review must bind the actual original and archived identical controller status")
    require(controller.get("status")=="resource_review_required" and controller.get("stop_category")=="resource"
        and controller.get("resource_review_required") is True and controller.get("child_exit_code")==0
        and controller.get("cooperative_stop_reason")=="system_commit_at_or_above95percent"
        and controller.get("run_id")==PARENT_RUN
        and controller.get("controller_sha256")==PARENT_SOURCES["run_TSC_headlr_fork_v1.py"]
        and controller.get("selected_step")==receipt["selected_step"]
        and controller.get("completed_steps")==receipt["completed_steps"],
        "Only the exact saved child0 system-commit critical terminal state is reviewed")
    return review


def validate_receipt(receipt, target_run_id):
    require(receipt.get("schema")==1 and receipt.get("kind")=="ROOT_owned_closed_TSC_best_for_BPE512_acquisition_v1"
        and receipt.get("status")=="actual_CLOSED_qualified_best" and receipt.get("source_run_id")==PARENT_RUN
        and receipt.get("target_run_id")==target_run_id and receipt.get("allow_BPE512_row_only_acquisition") is True
        and receipt.get("declared_acquisition")==acquisition_declaration(),"Explicit ROOT-approved CLOSED acquisition receipt required")
    require(type(receipt.get("selected_step")) is int and type(receipt.get("completed_steps")) is int
        and 0 < receipt["selected_step"] <= receipt["completed_steps"]
        and receipt.get("retention_gate_passed_at_selected_step") is True,
        "Use dynamically selected qualifying BEST, never latest-state mixture")
    exits=receipt.get("owned_actual_exits",[])
    if receipt.get("resource_stop_review") is not None:
        review=reviewed_resource_stop(receipt);owned=review["owned_actual_exit"]
        controller_exits=[item for item in exits if item.get("role")=="trainer_controller"]
        require(len(controller_exits)==1 and controller_exits[0].get("session_id")==owned["session_id"]
            and controller_exits[0].get("exit_code")==1
            and (controller_exits[0].get("chunk_id") or controller_exits[0].get("tool_chunk"))==owned["chunk_id"]
            and datetime.fromisoformat(str(controller_exits[0].get("observed_at_utc","")).replace("Z","+00:00"))
                ==datetime.fromisoformat(owned["observed_at_utc"].replace("Z","+00:00"))
            and all(item.get("exit_code")==0 and (item.get("chunk_id") or item.get("tool_chunk")) and item.get("session_id")
                for item in exits if item.get("role")!="trainer_controller"),
            "Preserve the reviewed actual controller exit1; every other owned exit must remain0")
    else:
        require(any(item.get("role")=="trainer_controller" for item in exits)
            and all(item.get("exit_code")==0 and (item.get("chunk_id") or item.get("tool_chunk")) and item.get("session_id") for item in exits),
            "ROOT-owned controller actual exit0/chunk/session required; no inferred process closure")
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{2,110}",target_run_id) is not None
        and target_run_id!=PARENT_RUN,"Safe new scientific namespace required")
    return receipt


def validate_closed_runtime(receipt,controller,completion,runtime):
    """Same frozen native completion proof, with one explicit reviewed terminal."""
    step=receipt["selected_step"];artifacts=receipt["artifacts"]
    reviewed=receipt.get("resource_stop_review") is not None
    if reviewed: reviewed_resource_stop(receipt)
    require(controller.get("status")==("resource_review_required" if reviewed else "train_window_ended")
        and controller.get("child_exit_code")==0
        and controller.get("resource_review_required") is reviewed and controller.get("run_id")==PARENT_RUN
        and controller.get("controller_sha256")==PARENT_SOURCES["run_TSC_headlr_fork_v1.py"]
        and controller.get("selected_step")==step and controller.get("completed_steps")==receipt["completed_steps"],
        "Actual parent child-close/terminal source required; resource source needs its specific ROOT review")
    require(completion.get("kind")=="TSC_headlr_fork_native_completion_v1" and completion.get("status")=="complete"
        and completion.get("run_id")==PARENT_RUN and completion.get("sources_unchanged") is True
        and completion.get("fresh_runtime_sha256")==artifacts["fresh_runtime"]["sha256"]
        and completion.get("actual_dispatch_source_sha256")==PARENT_SOURCES["run_TSC_headlr_fork_v1.py"]
        and runtime.get("pid")==completion.get("pid"),"Actual frozen native completion/freshPID source required")


def validate_layout(layout):
    require(layout.get("format")=="parakeet_encoder_attention_lora_v1" and layout.get("num_encoder_layers")==24
        and layout.get("adapted_layer_indices")==list(range(24))
        and layout.get("target_projection_names")==["q_proj","k_proj","v_proj","o_proj"]
        and layout.get("non_adapter_encoder_trainable_parameter_count")==0,"Original all24 QKVO layout required")
    targets=layout.get("targets",[])
    require(len(targets)==96 and {(t["layer_index"],t["projection"]) for t in targets}
        =={(i,p) for i in range(24) for p in ("q_proj","k_proj","v_proj","o_proj")}
        and all(t["module"]==f'encoder.layers.{t["layer_index"]}.self_attn.{t["projection"]}'
            and t["rank"]==8 and t["alpha"]==16 and t["scaling"]==2 and t["dropout"]==0
            and t["in_features"]==t["out_features"]==1024 for t in targets),"Rank8/alpha16 adapter geometry changed")


def candidate_assets():
    # Reuse only the pure donor-plan function, never the converter's old source admission.
    require(sha(HERE/"expand_trainable_bpe_checkpoint_v1.py")==CONVERTER_SHA,"Reviewed donor implementation changed")
    from expand_trainable_bpe_checkpoint_v1 import ASSET_SHA256, build_donor_plan
    from tsc_bpe512_training_inputs_v1 import BASELINE_PATH,BASELINE_FILE_SHA,CANDIDATE_PATH,validate_tokenizer_geometry
    directory=CANDIDATE_PATH.parent
    for name,digest in ASSET_SHA256.items():require(sha(directory/name)==digest,"Frozen candidate asset changed: "+name)
    require(sha(BASELINE_PATH)==BASELINE_FILE_SHA,"Original8199 byte-exact tokenizer changed")
    old,new=read(BASELINE_PATH),read(CANDIDATE_PATH);validate_tokenizer_geometry(old,new)
    freeze=read(directory/"candidate_frozen_before_measurement.json")
    require(freeze.get("fit_performed") is False and freeze.get("DEV_or_TEST_text_read") is False
        and freeze.get("merges_read_from_prior_TRAIN_only_candidate") is True,"Only frozen TRAIN-built512 pieces")
    donors=build_donor_plan(old,new,read(directory/"added_pieces.json"))
    require(len(donors)==512 and all(BLANK_ID not in item["original_leaf_donor_ids"] for item in donors),"Originalleaf donors must excludeblank")
    return new,donors


@lru_cache(maxsize=2)
def _context(receipt_path, target_run_id, receipt_digest):
    require(sha(receipt_path)==receipt_digest,"ROOT closure changed while read")
    receipt=validate_receipt(read(receipt_path),target_run_id);artifacts=receipt["artifacts"]
    for key in ("best_checkpoint","latest_checkpoint"):checked_record(artifacts[key],checkpoint=True)
    for key in ("run_config","training_history","selection","controller_status","runtime_completion","fresh_runtime",
                "DEV_summary","DEV_metadata",*("DEV_"+language for language in LANGUAGES)):checked_record(artifacts[key])
    config=read(artifacts["run_config"]["path"]);selection=read(artifacts["selection"]["path"])
    identity=config["config_identity"];step=receipt["selected_step"]
    require(canonical(identity)==receipt["scientific_config_identity_sha256"]
        and config["run_id"]==selection["run_id"]==PARENT_RUN
        and selection.get("accepted") is True and selection.get("heldout_tests_evaluated") is False
        and selection.get("selection_split")=="FLEURS validation only"
        and selection.get("selected_step")==step and selection.get("completed_steps")==receipt["completed_steps"]
        and scoped(selection["weights"])==scoped(receipt.get("original_artifact_paths",{}).get("best_checkpoint",artifacts["best_checkpoint"]["path"])),
        "Actual CLOSED selectedBEST/config required; archived original pointer must be explicitly bound")
    require(identity.get("extended_tokenizer_sha256")=="b91d500326aca290ed709d629dab2bee56f298ea7323b94855b9f5ef0caeac75",
        "Only original8199 six-glyph lineage")
    for name,digest in identity["source_sha256"].items():require(sha(HERE/name)==digest,"Parent source changed: "+name)
    require(all(identity["source_sha256"].get(name)==digest for name,digest in PARENT_SOURCES.items()),"Reviewed headLR parent source chain required")
    validate_layout(identity["lora_layout"])
    recipe=identity["training_arguments"]
    require(recipe["head_lr"]==1e-5 and recipe["lora_lr"]==3e-4 and recipe["new_token_lr_multiplier"]==6
        and recipe["rank"]==8 and recipe["alpha"]==16 and recipe["batch_size"]==8 and recipe["accumulation"]==2
        and recipe["replay_kl"]==10 and recipe["duration_kl"]==.5 and recipe["feature_distillation"]==1
        and recipe["sampling_mode"]=="domain30" and recipe["fleurs_fraction"]==.3 and recipe["tdt_implementation"]=="nemo"
        and recipe["epochs"]==1 and recipe["steps"]==44958,"Reviewed current TSC science required")
    controller=read(artifacts["controller_status"]["path"]);completion=read(artifacts["runtime_completion"]["path"])
    runtime=read(artifacts["fresh_runtime"]["path"])
    validate_closed_runtime(receipt,controller,completion,runtime)
    history=read(artifacts["training_history"]["path"])
    require(len(history)==receipt["completed_steps"] and all(row["step"]==i for i,row in enumerate(history,1))
        and history[step-1]["validation"]["retention_gate_passed"] is True
        and history[step-1]["validation"]["turkish_wer"]==selection["turkish_validation_wer"],"Saved qualifyingDEV BEST frontier required")
    summary=read(artifacts["DEV_summary"]["path"]);baseline=read(HERE/"results/base_v3/summary.json")
    require(summary["validation_tr"]["asr"]["wer"]==selection["turkish_validation_wer"]
        and all(summary["validation_"+language]["asr"]["utterances"]==50 for language in LANGUAGES),"Actual selected source300DEV required")
    for language in LANGUAGES[1:]:
        base=baseline["validation_"+language]["asr"]["wer"]
        require(summary["validation_"+language]["asr"]["wer"]<=base+max(.005,.1*base),"Selected source retention gate failed")
    require(sha(HERE/"tsc_bpe512_training_inputs_v1.py")==CONSUMER_SHA,"Reviewed wholeTSC consumer changed")
    from tsc_bpe512_training_inputs_v1 import preflight as metadata_preflight
    from bpe_target_sampler_transition_v1 import preparation_for_source
    require(sha(HERE/"bpe_target_sampler_transition_v1.py")==SAMPLER_MIGRATION_SHA,"Reviewed sampler migration changed")
    preparation=metadata_preflight();require(preparation_for_source(config)==preparation,"Whole253798 source/preparation identity differs")
    _,donors=candidate_assets()
    return {"receipt":receipt,"receipt_record":record(receipt_path),"config":config,"selection":selection,
        "parent_identity":identity,"source_selected_step":step,"source_recipe":dict(recipe),"donor_plan":donors,
        "metadata_preparation":preparation,"source_history_prefix_sha256":canonical(history[:step])}


def preflight(args):
    """ROOT/controller stdlib admission; actual checkpoint tensor reads are deferred."""
    require(getattr(args,"resume_acquisition",None) is not None
        and all(getattr(args,key,None) is None for key in ("resume_exact","resume_weights","resume_runtime_migration","resume_optimization")),
        "Only initial declared acquisition; exact/acquisition-phase continuation is not implemented")
    path=scoped(args.acquisition_source_receipt);context=_context(str(path),args.run_id,sha(path))
    require(scoped(args.resume_acquisition)==scoped(context["receipt"]["artifacts"]["best_checkpoint"]["path"]),"Use whole CLOSED selectedBEST only")
    for key,value in context["source_recipe"].items():require(getattr(args,key)==value,"Parent recipe option changed: "+key)
    require(args.acquisition_lr==1e-4 and args.native_max_joint_bytes==NATIVE_MAX_BYTES,"Explicit row-only LR/bound required")
    require(type(args.acquisition_max_updates) is int and 0<args.acquisition_max_updates<=10000
        and args.smoke_updates in (None,1) and (args.smoke_updates is None or type(args.smoke_updates) is int),
        "Explicit finite acquisition screen or separate1-update smoke required; source horizon remains44958")
    for key in ("tsc_input_proof","tsc_training_approval","source_closure","tr_manifest","replay_manifest","data_v3_proof",
                "tr_frame_metadata","tr_frame_metadata_proof","reuse_input_audit","supplement"):
        before,after=context["config"].get(key),getattr(args,key,None)
        require((before is None and after is None) or (before is not None and after is not None and scoped(before)==scoped(after)),"Original corpus/replay/source asset changed: "+key)
    from tsc_bpe512_training_inputs_v1 import METADATA_PATH,REPORT_PATH,CANDIDATE_PATH
    require(scoped(args.candidate_tr_metadata)==METADATA_PATH and scoped(args.candidate_tr_metadata_proof)==REPORT_PATH
        and scoped(args.candidate_dir)==CANDIDATE_PATH.parent,"Only frozen wholeTSC candidate assets admitted")
    return context


def closed_source_identity(args):
    return preflight(args)["parent_identity"]["closed_source"]


def validate_parent_state(state, context):
    step=context["source_selected_step"];identity=context["parent_identity"]
    require(state.get("schema")==2 and state.get("run_id")==PARENT_RUN and state.get("step")==state.get("selected_step")==step
        and state.get("config_identity")==identity,"Full BEST state/config, never latest mixture")
    history=state.get("history",[])
    require(len(history)==step and all(row["step"]==i for i,row in enumerate(history,1))
        and canonical(history)==context["source_history_prefix_sha256"]
        and history[-1]["validation"]["retention_gate_passed"] is True
        and history[-1]["validation"]["turkish_wer"]==state["best_score"]==context["selection"]["turkish_validation_wer"],"Original selected history/gate/score changed")
    optimizer=state.get("optimizer",{});groups=optimizer.get("param_groups",[])
    require(len(state.get("weights",{}))==len(optimizer.get("state",{}))==207 and len(groups)==2
        and len(groups[0]["params"])==15 and len(groups[1]["params"])==192
        and groups[0]["lr"]==1e-5 and groups[1]["lr"]==3e-4
        and set(groups[0]["params"]+groups[1]["params"])==set(optimizer["state"])
        and all({"step","exp_avg","exp_avg_sq"}.issubset(item) for item in optimizer["state"].values()),"Dormant original207 Adam states/groups required")
    require(state["lora_metadata"]==identity["lora_layout"] and state["sampler"]["identity"]==identity["domain_sampling_identity"]
        and state["sampler"]["updates"]==step,"Original BEST sampler/layout frontier required")
    for key in ("python_rng","torch_rng","cuda_rng","patience","retention_failures","next_eval_epoch","train_wall_seconds",
                "audio_seconds","language_audio_seconds","evaluation_wall_seconds","initial_gate"):
        require(key in state and state[key] is not None,"Missing inherited full-state field: "+key)
    require(state["retention_failures"]==0,"Qualifying BEST failure counter must not be reset from another point")
    return {"inherited_steps":step,"first_acquisition_update":step+1,"source_is_BEST_not_latest":True,
        "history_prefix_sha256":canonical(history),"source_sampler_sha256":canonical(state["sampler"]),
        "source_Adam_states_retained_dormant":207,"new_active_Adam_tables":3,"fresh_sampler_or_global_warmup":False}


def audit_source_optimizer(state, context, expected, *, torch_module):
    """Future injected-Torch audit of the parked original207 Adam, without mutation.

    The frozen parent constructs heads first, then adapters, using named-parameter
    order within each group. PyTorch serializes those parameter IDs as0..206.
    No source moment is installed in the fresh three-table acquisition optimizer.
    """
    torch=torch_module;optimizer=state["optimizer"];groups=optimizer["param_groups"]
    heads=[name for name in expected if not name.endswith(("lora_A","lora_B"))]
    adapters=[name for name in expected if name.endswith(("lora_A","lora_B"))]
    require(len(heads)==15 and len(adapters)==192 and len(groups)==2,
        "Original head15/adapter192 named-parameter order required")
    require(groups[0]["params"]==list(range(15)) and groups[1]["params"]==list(range(15,207))
        and set(optimizer["state"])==set(range(207)),"Original Adam serialized group/parameter IDs changed")
    recipe=context["source_recipe"];step=context["source_selected_step"]
    require(recipe["schedule"]=="constant" and step>=recipe["warmup_steps"],
        "Reviewed source Adam clock must be past constant-schedule warmup")
    group_evidence=[];mapping=[]
    for index,(names,lr,decay) in enumerate(((heads,recipe["head_lr"],.01),(adapters,recipe["lora_lr"],0.))):
        group=groups[index]
        require(group["lr"]==lr and tuple(group["betas"])==(.9,.999) and group["eps"]==1e-8
            and group["weight_decay"]==decay and group["amsgrad"] is False
            and group["maximize"] is False and group["capturable"] is False
            and group["differentiable"] is False and group["foreach"] is None and group["fused"] is None,
            "Frozen parent Adam group recipe changed")
        for name,parameter_id in zip(names,group["params"]):
            moments=optimizer["state"][parameter_id];shape=expected[name].shape
            require(set(moments)=={"step","exp_avg","exp_avg_sq"},"Standard non-AMSGrad Adam moment keys required: "+name)
            clock=moments["step"]
            require(torch.is_tensor(clock) and clock.dtype==torch.float32 and clock.ndim==0
                and bool(torch.isfinite(clock).all()) and clock.item()==step,
                "Original scalar FP32 Adam clock differs from selectedBEST: "+name)
            for key in ("exp_avg","exp_avg_sq"):
                value=moments[key]
                require(torch.is_tensor(value) and value.shape==shape and value.dtype==torch.float32
                    and bool(torch.isfinite(value).all()),"Original FP32 Adam moment shape/finiteness differs: "+name+"/"+key)
            require(bool((moments["exp_avg_sq"]>=0).all()),"Adam second moment must be nonnegative: "+name)
            mapping.append({"parameter_id":parameter_id,"name":name,"shape":list(shape)})
        group_evidence.append({"parameters":len(names),"lr":lr,"betas":[.9,.999],"eps":1e-8,"weight_decay":decay})
    return {"source_Adam_states_checked":207,"source_Adam_clock":step,"all_moments_FP32_parameter_shape_finite":True,
        "all_second_moments_nonnegative":True,"original_group_ID_order_sha256":canonical(mapping),
        "groups":group_evidence,"source_optimizer_mutated_or_loaded_into_acquisition":False}


def load_validate_expand(model, processor, state, context, *, torch_module):
    """Future worker only: load exact207 source tensors then expand/install3 tables.

    No checkpoint is opened by this helper until the actual worker calls it.
    Caller keeps its baseline processor for unchanged replay/label comparison.
    Source state,207 Adam moments and all sampler/RNG/history remain unmodified.
    """
    torch=torch_module;boundary=validate_parent_state(state,context)
    require(sha(context["receipt"]["artifacts"]["best_checkpoint"]["path"])==context["receipt"]["artifacts"]["best_checkpoint"]["sha256"],"Loaded CLOSED source bytes changed")
    expected={name:p for name,p in model.named_parameters() if p.requires_grad};weights=state["weights"]
    require(set(expected)==set(weights) and len(expected)==207,"All original207 trainable source keys required")
    for name,value in weights.items():
        require(value.shape==expected[name].shape and value.dtype==torch.float32 and bool(torch.isfinite(value).all()),"Source FP32/shape/finiteness changed: "+name)
    optimizer_evidence=audit_source_optimizer(state,context,expected,torch_module=torch)
    from encoder_lora import encoder_lora_metadata
    require(encoder_lora_metadata(model)==state["lora_metadata"],"Actual installed source LoRA layout changed")
    require(model.config.vocab_size==OLD_VOCAB and model.config.blank_token_id==BLANK_ID
        and list(model.config.durations)==[0,1,2,3,4],"Original8199/blank/durations geometry required")
    model.load_state_dict(weights,strict=False)
    old_embedding,old_head=model.decoder.embedding,model.joint.head
    require(old_embedding.num_embeddings==8199 and old_head.out_features==8204,
        "Original embedding/joint geometry changed")
    candidate_json,donors=candidate_assets()
    from tsc_bpe512_training_inputs_v1 import BASELINE_SEMANTIC_SHA,semantic
    require(semantic(json.loads(processor.tokenizer._tokenizer.to_str()))==BASELINE_SEMANTIC_SHA,"Actual original processor differs")
    rng_cpu=torch.get_rng_state().clone();rng_cuda=[value.clone() for value in torch.cuda.get_rng_state_all()]
    try:
        embedding=torch.nn.Embedding(NEW_VOCAB,old_embedding.embedding_dim,padding_idx=old_embedding.padding_idx,
            device=old_embedding.weight.device,dtype=old_embedding.weight.dtype)
        head=torch.nn.Linear(old_head.in_features,NEW_VOCAB+5,bias=True,device=old_head.weight.device,dtype=old_head.weight.dtype)
    finally:
        torch.set_rng_state(rng_cpu);torch.cuda.set_rng_state_all(rng_cuda)
    with torch.no_grad():
        embedding.weight[:OLD_VOCAB].copy_(old_embedding.weight)
        head.weight[:OLD_VOCAB].copy_(old_head.weight[:OLD_VOCAB]);head.bias[:OLD_VOCAB].copy_(old_head.bias[:OLD_VOCAB])
        head.weight[NEW_VOCAB:].copy_(old_head.weight[OLD_VOCAB:]);head.bias[NEW_VOCAB:].copy_(old_head.bias[OLD_VOCAB:])
        for donor in donors:
            index,leaves=donor["HF_id"],donor["original_leaf_donor_ids"]
            embedding.weight[index].copy_(weights[TABLES[0]][leaves].mean(0))
            head.weight[index].copy_(weights[TABLES[1]][leaves].mean(0))
            head.bias[index].copy_(weights[TABLES[2]][leaves].mean()-2.0)
    def bits_equal(left,right):
        return left.shape==right.shape and left.dtype==right.dtype and torch.equal(left.detach().cpu().contiguous().reshape(-1).view(torch.uint8),right.detach().cpu().contiguous().reshape(-1).view(torch.uint8))
    for new,old in ((embedding.weight,old_embedding.weight),(head.weight,old_head.weight),(head.bias,old_head.bias)):
        require(bits_equal(new[:OLD_VOCAB],old[:OLD_VOCAB]),"Protected original lexical/blank/six prefix changed")
    require(bits_equal(head.weight[NEW_VOCAB:],old_head.weight[OLD_VOCAB:])
        and bits_equal(head.bias[NEW_VOCAB:],old_head.bias[OLD_VOCAB:]),"Protected duration rows changed")
    for name,parameter in model.named_parameters():
        if name in weights and name not in TABLES:require(bits_equal(parameter,weights[name]),"Unrelated204 source tensor changed: "+name)
    embedding.train(old_embedding.training);head.train(old_head.training)
    model.decoder.embedding,model.joint.head=embedding,head
    model.config.vocab_size=model.joint.vocab_size=NEW_VOCAB
    if hasattr(model.decoder,"vocab_size"):model.decoder.vocab_size=NEW_VOCAB
    if hasattr(model.generation_config,"vocab_size"):model.generation_config.vocab_size=NEW_VOCAB
    suppression={}
    for field in ("suppress_tokens","begin_suppress_tokens"):
        before=getattr(model.generation_config,field,None)
        require(before is None or all(type(value) is int and 0<=value<8204 for value in before),"Unexpected original suppression IDs")
        after=None if before is None else [NEW_VOCAB+value-OLD_VOCAB if OLD_VOCAB<=value<8204 else value for value in before]
        require(not any(8199<=value<8711 for value in after or []),"New512 IDs must remain unsuppressed")
        setattr(model.generation_config,field,after);suppression[field]={"before":before,"after":after}
    from tokenizers import Tokenizer
    from extend_vocab import _make_tokenizer
    processor.tokenizer=_make_tokenizer(processor.tokenizer,Tokenizer.from_str(json.dumps(candidate_json,ensure_ascii=False)))
    require(torch.equal(torch.get_rng_state(),rng_cpu) and all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),rng_cuda)),"Expansion changed tensor RNG")
    return {"boundary":{**boundary,"source_Adam_runtime_audit":optimizer_evidence},"vocabulary":{"changed":True,"extended_tokenizer_sha256":"10b747471a6613f818445f4c5697d04bb64ca45c48eb1bde7184d33e28e68607",
            "student_vocab_size":8711,"blank_id":8192,"teacher_vocab_size":8193,"new_BPE_rows":[8199,8711],"duration_rows":[8711,8716]},
        "expansion_evidence":{"expanded_tables":list(TABLES),"unrelated_source_tensors_bitexact":204,
            "original8199_rows_bitexact":True,"duration_rows_bitexact":True,"donor_plan_sha256":canonical(donors),
            "initialization":"Trained ORIGINAL leaf donor means with multiplicity; biasmean minus2 exactlyonce",
            "generation_suppression":suppression,"source_state_not_modified":True,"CPU_allCUDA_RNG_preserved":True}}


def proof_path(args):return HERE/"runs"/args.run_id/PROOF_NAME


def compare_development(source_rows,candidate_rows,source_asr,candidate_asr,language):
    from metrics import score_transcriptions
    require(len(source_rows)==len(candidate_rows)==50 and len({row["source_id"] for row in source_rows})==50,"Complete fixed50 uniqueDEV required")
    count=0
    for before,after in zip(source_rows,candidate_rows):
        require(all(key in before and key in after and before[key]==after[key] for key in IDENTITY_KEYS)
            and before["language"]==language and before["split"]=="validation","FixedDEV IDs/order/reference/audio differ")
        require(before["hypothesis"]==after["hypothesis"],"Unsuppressed expanded model raw transcript changed")
        paths=[]
        for row,vocab in ((before,8199),(after,8711)):
            confidence=row["confidence"];require(confidence["text"]==row["hypothesis"],"Confidence decode text changed")
            tokens=[tuple(item[key] for key in TOKEN_KEYS) for item in confidence["tokens"]]
            require(all(0<=t[0]<vocab and t[0]!=8192 and 0<=t[1]<=4 and t[4]==t[1] for t in tokens),"Token/duration path geometry changed")
            paths.append(tokens)
        require(paths[0]==paths[1] and all(before["confidence"][key]==after["confidence"][key] for key in ("blank_steps","total_steps")),"Actual emitted token/duration/blank step path changed")
        count+=len(paths[0])
    scores=[{k:v for k,v in score_transcriptions(rows,language=language).items() if k!="details"} for rows in (source_rows,candidate_rows)]
    require(scores[0]==scores[1]==source_asr==candidate_asr,"Independent fixedDEV ASR counts differ")
    return {"utterances":50,"exact_raw_transcripts":50,"exact_token_duration_paths":50,"tokens_compared":count,"asr":scores[0]}


def initial_source_equivalence(model,processor,args,context,state,evaluate,torch):
    """Future actual GPU300 decode before fresh Adam; no confidence/logit equivalence claim."""
    import random
    require(not proof_path(args).exists(),"Never overwrite initial300 proof")
    boundary=validate_parent_state(state,context);artifacts=context["receipt"]["artifacts"]
    source_summary=read(checked_record(artifacts["DEV_summary"]));source_metadata=read(checked_record(artifacts["DEV_metadata"]))
    for field in ("suppress_tokens","begin_suppress_tokens"):
        require(not any(8199<=value<8711 for value in getattr(model.generation_config,field,None) or []),"New512 must be unsuppressed duringDEV")
    rng=(random.getstate(),torch.get_rng_state().clone(),[value.clone() for value in torch.cuda.get_rng_state_all()])
    require(rng[0]==state["python_rng"] and torch.equal(rng[1],state["torch_rng"])
        and len(rng[2])==len(state["cuda_rng"]) and all(torch.equal(a,b) for a,b in zip(rng[2],state["cuda_rng"])),
        "Restore exact selectedBEST Python/CPU/allCUDA RNG before added source-equality evaluation")
    modes=[(module,module.training) for module in model.modules()];step=context["source_selected_step"]
    try:candidate=evaluate(model,processor,args.run_id+"_step_"+str(step),splits=("validation",))
    finally:
        random.setstate(rng[0]);torch.set_rng_state(rng[1]);torch.cuda.set_rng_state_all(rng[2])
        for module,training in modes:module.train(training)
    cuda=torch.cuda.get_rng_state_all()
    restored=random.getstate()==rng[0] and torch.equal(torch.get_rng_state(),rng[1]) and len(cuda)==len(rng[2]) and all(torch.equal(a,b) for a,b in zip(cuda,rng[2]))
    require(restored and all(module.training==training for module,training in modes),"AddedDEV must restore Python/CPU/allCUDA RNG and module/encoder modes")
    folder=HERE/"results"/(args.run_id+"_step_"+str(step));candidate_metadata=read(folder/"metadata.json")
    require(source_metadata["manifests"]==candidate_metadata["manifests"],"FixedDEV manifests changed")
    # Tokenizer/width intentionally differ; do not require identical model fingerprint.
    comparisons={}
    for language in LANGUAGES:
        source_path=checked_record(artifacts["DEV_"+language]);candidate_path=folder/("validation_"+language+".jsonl")
        sources=[json.loads(line) for line in source_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        candidates=[json.loads(line) for line in candidate_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        comparisons[language]=compare_development(sources,candidates,source_summary["validation_"+language]["asr"],candidate["validation_"+language]["asr"],language)
    baseline=read(HERE/"results/base_v3/summary.json")
    for language in LANGUAGES[1:]:
        base=baseline["validation_"+language]["asr"]["wer"]
        require(candidate["validation_"+language]["asr"]["wer"]<=base+max(.005,.1*base),"Initial fixed retention gate failed")
    proof={"schema":1,"kind":"TSC_BPE512_acquisition_before_update_source_equivalence_v1","passed":True,
        "run_id":args.run_id,"source_receipt":context["receipt_record"],"source_selected_step":step,"state_boundary":boundary,
        "actual_GPU_decode":True,"before_optimizer_creation":True,"compared_fixed_DEV_utterances":300,
        "unsuppressed_new512_ids":True,"all_raw_transcripts_and_emitted_paths_exact":True,"all_six_ASR_counts_exact":True,
        "all_five_retention_gates_passed":True,"RNG_restored_after_added_evaluation":restored,
        "module_training_modes_restored_after_added_evaluation":True,"sampler_drawn_during_check":False,
        "source_and_candidate_fingerprints_equal_required":False,"confidence_or_loss_equality_required":False,
        "candidate_summary":record(folder/"summary.json"),"candidate_metadata":record(folder/"metadata.json"),"comparisons":comparisons}
    with proof_path(args).open("x",encoding="utf-8") as stream:json.dump(proof,stream,ensure_ascii=False,indent=2);stream.write("\n")
    return proof
