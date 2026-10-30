"""Source-only full-pass worker derivative; frozen prior files are untouched."""
import ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'outputs/parakeet_turkish'
def fn(text,name):
 n=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)
 return ''.join(text.splitlines(keepends=True)[n.lineno-1:n.end_lineno])
def setfn(text,name,body):return text.replace(fn(text,name),body.rstrip(),1)
text=(HERE/'train_lora_batched_tsc_headlr_v1.py').read_text(encoding='utf-8')
text=text.replace(text[:text.index('from __future__')],'''"""INACTIVE full253798 uniform TRAIN continuation of qualified original8199 BEST.

All207 learned weights and15-head/192-LoRA Adam moments at6500 carry. The TR
queues explicitly migrate to a fresh complete uniform pass; original five
replay queues/RNG carry and approved Portuguese TRAIN becomes a sixth replay
language. New350DEV uses original six plus separate50 Portuguese manifests.
First actual source300 preservation and Portuguese baseline/gate are mandatory.
Later SAME-ID budget windows retain full state and immutable initial identity.
"""
''',1)
text=text.replace('from tsc_optimization_transition_v1 import (closed_source_identity, preflight as optimization_preflight,\n    validate_parent_state, initial_source_equivalence, transition_identity, validate_transition_identity, proof_path)',
'''from tsc_uniform_full_transition_v1 import (preflight as uniform_preflight,
    load_validate_source, compare_source_development, validate_portuguese_inputs,
    validate_own_state, transition_identity, proof_path)''',1)
text=text.replace('from tsc_domain_sampler_v3 import TSCDomainBalancedBucketSampler',
    'from tsc_uniform_full_sampler_v1 import TSCUniformFullSampler',1)
text=text.replace('from benchmark import LANGUAGES, evaluate, read_jsonl, write_json',
    'from benchmark import LANGUAGES as ORIGINAL_LANGUAGES, evaluate as original_evaluate, read_jsonl, write_json\nLANGUAGES = (*ORIGINAL_LANGUAGES, "pt")',1)
text=text.replace('from native_tdt_checkpointed_v3 import','from native_tdt_checkpointed_v4 import')
text=text.replace('def development_gate(dev, baseline):','def development_gate(dev, baseline):',1)
# The evaluator supplies Portuguese from a separate approved directory; frozen
# benchmark DATA/manifests and original summary are never changed.
index=text.index('def validate_trainable_checkpoint')
text=text[:index]+'''def evaluate(model, processor, tag, *, splits=("validation",)):
    if splits != ("validation",): raise ValueError("This full-training controller evaluates fixed DEV only")
    import benchmark
    result = original_evaluate(model, processor, tag, languages=ORIGINAL_LANGUAGES, splits=splits)
    original_data = benchmark.DATA
    try:
        benchmark.DATA = DATA / "portuguese_retention_v1"
        portuguese = original_evaluate(model, processor, tag, languages=("pt",), splits=splits)
    finally:
        benchmark.DATA = original_data
    result.update(portuguese)
    write_json(OUTPUT / "results" / tag / "summary.json", result)
    return result


''' +text[index:]
start=text.index('def run(args):');end=text.index('    new_parameters = ',start)
startup='''def run(args):
    process_start=time.perf_counter();context=uniform_preflight(args)
    public_data_audit=audit_tsc_training_inputs(args)
    portuguese=validate_portuguese_inputs(args,context)
    public_data_audit["identity"].update(uniform_full_pass_transition=True,
        target_replay_rng_independent_of_turkish=True,target_replay_seed=args.replay_seed,
        Portuguese_replay_pool_added_explicitly=True,replay_language_balance_count=6,
        old_five_future_replay_sequence_claimed=False)
    random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    run_dir=OUTPUT/"runs"/args.run_id
    best_path=ROOT/"work"/(args.run_id+"_best.pt");latest_path=ROOT/"work"/(args.run_id+"_latest.pt")
    if args.resume_exact:
        if not run_dir.is_dir() or not best_path.is_file(): raise ValueError("Exact full-run namespace/BEST missing")
    else:
        if run_dir.exists() or best_path.exists() or latest_path.exists(): raise ValueError("Fresh full-run namespace required")
        run_dir.mkdir(parents=True)
    model,processor=load_model();baseline=context["combined_baseline"]
    old_heldout=[DATA/f"{split}_{lang}.jsonl" for lang in ORIGINAL_LANGUAGES for split in ("validation","test")]
    heldout_paths=[*old_heldout,DATA/"portuguese_retention_v1/validation_pt.jsonl",DATA/"portuguese_retention_v1/test_pt.jsonl"]
    heldout=[row for path in heldout_paths for row in read_jsonl(path)]
    tr,replay=read_jsonl(args.tr_manifest),read_jsonl(args.replay_manifest)
    source_isolation(tr,replay,heldout,public_data_audit)
    pt_rows=read_jsonl(args.portuguese_replay_manifest)
    if len(tr)!=253798 or len(replay)!=1595 or len(pt_rows)!=319: raise ValueError("Complete approved TR/originalRP/PT pools required")
    all_heldout_ids={row["source_id"] for row in heldout}
    if any(row["source_id"] in all_heldout_ids for row in pt_rows): raise ValueError("Portuguese replay heldout collision")
    del heldout,all_heldout_ids
    teacher={name:copy.deepcopy(getattr(model,name)).eval().requires_grad_(False) for name in ("encoder_projector","decoder","joint")}
    old_size=model.config.vocab_size
    if old_size!=8193:raise ValueError("Original8193 teacher required")
    vocabulary=extend_processor_and_model(processor,model,[row["text"] for row in tr])
    if args.phonetic_sh_donors:
        for new,donor in (("ş","ș"),("Ş","Ș")):
            new_id,donor_id=processor.tokenizer.convert_tokens_to_ids(new),processor.tokenizer.convert_tokens_to_ids(donor)
            with torch.no_grad():
                model.decoder.embedding.weight[new_id].copy_(model.decoder.embedding.weight[donor_id])
                model.joint.head.weight[new_id].copy_(model.joint.head.weight[donor_id])
                model.joint.head.bias[new_id].copy_(model.joint.head.bias[donor_id]-2)
    if model.config.vocab_size!=8199 or model.config.blank_token_id!=8192:raise ValueError("Original8199/sixglyph geometry required")
    counts_before={"tr":len(tr),"replay":len(replay),"Portuguese":len(pt_rows)}
    namespace=hashlib.sha256((BASE_REVISION+processor.tokenizer._tokenizer.to_str()).encode()).hexdigest()[:20]
    cache=ROOT/"work/encoder-cache"/namespace
    for parameter in model.encoder.parameters():parameter.requires_grad_(False)
    tr=annotate_tr_frame_metadata(processor,tr,args.tr_frame_metadata,args.tr_frame_metadata_proof,args.max_tokens)
    replay=annotate_cache_lengths(make_cache(model,processor,replay,cache,args.max_tokens))
    confirm_tr_metadata_acceptance(tr,public_data_audit);confirm_replay_after_cache(replay,public_data_audit)
    pt_cached=annotate_cache_lengths(make_cache(model,processor,pt_rows,cache,args.max_tokens))
    if len(pt_cached)!=319:raise ValueError("No Portuguese TRAIN crop/drop")
    for row in pt_cached:
        ids=processor.tokenizer(row["text"])["input_ids"]
        if not ids or max(ids)>=old_size:raise ValueError("Portuguese teacher labels must stay original8193")
    layout=install_encoder_lora(model,rank=args.rank,alpha=args.alpha,last_n_layers=args.last_n_layers)
    model.config.encoder_lora_tracking={"layout":layout,"implementation_sha256":file_sha(OUTPUT/"encoder_lora.py"),"kind":"unmerged"}
    if not args.no_encoder_checkpoint:checkpoint_encoder_blocks(model)
    # Source remains a full207/Adam artifact, not weights-only warm start.
    source=torch.load(args.resume_uniform_full,map_location="cpu",weights_only=True)
    load_validate_source(model,source,context,torch_module=torch)
    model.load_state_dict(source["weights"],strict=False)
    adapters=[p for name,p in model.named_parameters() if p.requires_grad and name.endswith(("lora_A","lora_B"))]
    heads=[p for name,p in model.named_parameters() if p.requires_grad and not name.endswith(("lora_A","lora_B"))]
    trainable=adapters+heads
    optimizer=torch.optim.AdamW([{"params":heads,"lr":args.head_lr,"weight_decay":.01},
                                {"params":adapters,"lr":args.lora_lr,"weight_decay":0.}])
    sampler=TSCUniformFullSampler(tr,replay,portuguese_replay=pt_cached,portuguese_seed=args.portuguese_seed,
        seed=args.seed,replay_seed=args.replay_seed,sampling_mode="uniform",batch_size=args.batch_size,
        accumulation=args.accumulation,replay_fraction=args.replay_fraction,fleurs_fraction=args.fleurs_fraction,
        bucket_size=args.bucket_size,cycler_factory=BucketCycler,cycler_source_sha256=file_sha(OUTPUT/"train_lora_batched_data_v11.py"))
    sampler_transition=sampler.migrate_source_state(source["sampler"],context["parent_identity"]["domain_sampling_identity"])
    source_step=context["source_step"];configured_steps=source_step+sampler.uniform_pass_updates
    if source_step!=6500 or configured_steps!=38225 or sampler.uniform_pass_updates!=31725:raise ValueError("One complete253798 new uniform pass required")
    all_replay=[*replay,*pt_cached]
    loss_function=native_loss_function(args,tr,all_replay)
    from native_tdt_checkpointed_v4 import reclaim_inactive_cuda_cache
    if not args.resume_exact:
        random.setstate(source["python_rng"]);torch.set_rng_state(source["torch_rng"]);torch.cuda.set_rng_state_all(source["cuda_rng"])
        equality=compare_source_development(model,processor,args,context,source,evaluate,torch)
    else:
        equality=json.loads(proof_path(args).read_text(encoding="utf-8"))
    source_sha=dict(context["parent_identity"]["source_sha256"])
    for name in (Path(__file__).name,"run_TSC_uniform_full_v1.py","tsc_uniform_full_transition_v1.py","tsc_uniform_full_sampler_v1.py",
        "native_tdt_checkpointed_v4.py","compact_training_row_objects_v1.py"):
        source_sha[name]=file_sha(OUTPUT/name)
    input_paths=[args.uniform_full_source_receipt,proof_path(args),args.tsc_input_proof,args.tsc_training_approval,args.source_closure,
        args.tr_manifest,args.replay_manifest,args.data_v3_proof,args.tr_frame_metadata,args.tr_frame_metadata_proof,
        args.portuguese_replay_manifest,args.portuguese_dev_manifest,args.portuguese_baseline_proof,args.native_backend_provenance,*heldout_paths]
    input_sha={str(path.resolve()):file_sha(path) for path in input_paths}
    arguments={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()}
    science_keys=("sampling_mode","fleurs_fraction","replay_seed","portuguese_seed","epochs","steps","batch_size","accumulation","replay_fraction",
        "bucket_size","seed","max_tokens","head_lr","lora_lr","new_token_lr_multiplier","replay_kl","duration_kl","feature_distillation",
        "rank","alpha","last_n_layers","chunk_size","tdt_implementation","warmup_steps","schedule","min_lr_ratio","eval_per_epoch",
        "eval_epoch_fraction","eval_every_updates","min_improvement","no_encoder_checkpoint","phonetic_sh_donors","train_limit",
        "early_stop_patience","max_retention_failures","target_turkish_wer")
    config_identity={"public_turkish_data_policy":public_data_audit["identity"],"base_revision":BASE_REVISION,
        "extended_tokenizer_sha256":vocabulary["extended_tokenizer_sha256"],"lora_layout":encoder_lora_metadata(model),
        "tr_pool_sha256":ids_sha(tr),"replay_pool_sha256":ids_sha(all_replay),"source_sha256":source_sha,"input_sha256":input_sha,
        "native_loss":native_loss_identity(args),"domain_sampling_identity":sampler.identity,
        "uniform_full_transition":transition_identity(args,context,sampler_transition),
        "training_arguments":{key:arguments[key] for key in science_keys}}
    if args.resume_exact:
        del source
        own=torch.load(args.resume_exact,map_location="cpu",weights_only=True)
        validate_own_state(own,args,context,config_identity,torch_module=torch)
        validate_trainable_checkpoint(model,own,encoder_lora_metadata(model));model.load_state_dict(own["weights"],strict=False)
        sampler.load_state_dict(own["sampler"]);resume=own
    else:resume=source
    optimizer.load_state_dict(resume["optimizer"])
    if len(optimizer.state)!=207:raise ValueError("All207 named Adam states required")
    history,start_step=resume["history"],resume["step"]
    best,selected_step=resume["best_score"],resume["selected_step"]
    patience,retention_failures=resume["patience"],resume["retention_failures"]
    train_wall,evaluation_wall,audio_seconds=resume["train_wall_seconds"],resume["evaluation_wall_seconds"],resume["audio_seconds"]
    language_audio={**resume["language_audio_seconds"],"pt":resume["language_audio_seconds"].get("pt",0.)}
    initial_gate=resume["initial_gate"]
    next_eval=resume["next_eval_epoch"] if args.resume_exact else next_evaluation_epoch(0.,args.eval_epoch_fraction,args.eval_per_epoch)
    random.setstate(resume["python_rng"]);torch.set_rng_state(resume["torch_rng"]);torch.cuda.set_rng_state_all(resume["cuda_rng"])
    if not args.resume_exact and not equality["Portuguese_source_retention_gate_passed"]:
        best,selected_step=math.inf,None # The old all-five champion stays parked; never call it all-six qualified.
    config={**arguments,"steps":configured_steps,"source_step":source_step,"new_uniform_pass_updates":31725,
        "method":"original8199 full207 head+encoderLoRA best recipe; new fulluniformTR and approvedPortuguese replay",
        "config_identity":config_identity,"initial_source_equivalence":equality,"public_turkish_data_audit":public_data_audit,
        "uniform_sampler_transition":sampler_transition,"Portuguese_inputs":portuguese,
        "turkish_clips_used":len(tr),"replay_clips_used":len(all_replay),"source_original_replay_clips":1595,"Portuguese_replay_clips":319,
        "sampling":sampler.coverage(),"sampling_identity":sampler.identity,"inherited_history_steps":source_step,
        "trainable_parameters":sum(p.numel() for p in trainable),"trainable_parameter_tensors":207,
        "native_loss":native_loss_identity(args),"lora_layout":layout,"encoder_adapter_parameters":sum(p.numel() for p in adapters),
        "encoder_base_frozen":True,"encoder_frozen":False,"encoder_activation_checkpointing":not args.no_encoder_checkpoint,
        "teacher":"Original8193 replay teacher; six languages with oldfive state+new Portuguese explicitly added",
        "initial_turkish_validation_wer":context["source_TR_WER"],"source_champion_parked":context["source_checkpoint"],
        "original_base_v3_summary_unchanged":True,"new_uniform_epoch_clock":True,"selection_split":"fixedFLEURS DEV only",
        "heldout_tests_evaluated":False,"training_throughput_timing":"complete update intervals; excludes startup/DEV/checkpoint serialization"}
    if args.resume_exact:
        original_config=json.loads((run_dir/"run_config.json").read_text(encoding="utf-8"))
        if canonical_identity_sha(original_config["config_identity"])!=canonical_identity_sha(config_identity):raise ValueError("Exact original identity changed")
        config=original_config
    else:
        write_json(run_dir/"run_config.json",config);write_json(run_dir/"training_pool.json",{"turkish":tr,"replay":all_replay,"input_sha256":input_sha})
        write_json(run_dir/"vocabulary_audit.json",vocabulary)
    write_json(OUTPUT/"run_config.json",config)
    def save(path,step,reason=None):
        save_training_state(path,model,optimizer,sampler,run_id=args.run_id,step=step,config_identity=config_identity,
            history=history,best_score=best,selected_step=selected_step,patience=patience,retention_failures=retention_failures,
            next_eval_epoch=next_eval,train_wall_seconds=train_wall,audio_seconds=audio_seconds,language_audio_seconds=language_audio,
            evaluation_wall_seconds=evaluation_wall,initial_gate=initial_gate,stop_reason=reason)
    if not args.resume_exact:
        if selected_step is not None:save(best_path,start_step,"fulluniform_declared_source6500")
        save(latest_path,start_step)
    write_json(run_dir/"training_history.json",history);write_json(OUTPUT/"training_history.json",history)
    del resume
    if args.resume_exact:del own
    else:del source
    from compact_training_row_objects_v1 import compact_training_rows
    compact=compact_training_rows(tr,expected_rows=253798)
    write_json(run_dir/("uniform_full_startup_from_"+str(start_step)+".json"),{"passed":True,"run_id":args.run_id,
        "source_step":source_step,"resume_step":start_step,"training_row_compaction":compact,
        "compactor_integer_intern_pool_is_8711_but_actual_model_vocab_is_8199":True,"reclaimed_memory_claimed":False})
    model.train();model.encoder.eval()
'''
text=text[:start]+startup+text[end:]
text=text.replace('"Turkish_domain_coverage":sampler.coverage(),','"Turkish_domain_coverage":sampler.coverage(),"uniform_full_updates":step-source_step,',1)
text=text.replace('"turkish_epoch_semantics":"minimum_domain_exposure", "Turkish_domain_coverage":sampler.coverage(),',
    '"turkish_epoch_semantics":"new_uniform_pass_draw_equivalent", "Turkish_domain_coverage":sampler.coverage(),\n                 "source_step":source_step,"uniform_full_updates":completed-source_step,"full_new_uniform_pass_complete":sampler.coverage()["uniform_pass"]["full_pass_complete"],',1)
text=text.replace('OUTPUT / "run_TSC_headlr_fork_v1.py"','OUTPUT / "run_TSC_uniform_full_v1.py"')
text=text.replace('"native_tdt_checkpointed_v3.py"','"native_tdt_checkpointed_v4.py"')
text=text.replace('Path(evidence.get("launcher_path", "")).resolve() != (OUTPUT / "run_TSC_uniform_full_v1.py").resolve()',
    'Path(evidence.get("launcher_path", "")).resolve() != (OUTPUT / "run_TSC_uniform_full_v1.py").resolve()',1)
# A fresh actual process token is mandatory for every exact window as well.
needle='    expected_env = ROOT / "work/nemo-native-env-v2/Scripts/python.exe"'
text=text.replace(needle,'''    fresh=json.loads(args.uniform_full_runtime_proof.read_text(encoding="utf-8"))
    import os
    if fresh.get("pid")!=os.getpid() or fresh.get("ppid")!=os.getppid() or fresh.get("status")!="ready" or fresh.get("run_id")!=args.run_id:
        raise ValueError("Fresh same-PID fullrun pre-Torch proof required")
    if fresh.get("trainer_sha256")!=file_sha(__file__) or fresh.get("launcher_sha256")!=file_sha(OUTPUT/"run_TSC_uniform_full_v1.py"):
        raise ValueError("Fresh uniform executor source pin differs")
''' +needle,1)
text=text.replace('p.add_argument("--optimization-source-receipt",type=Path,required=True)', 'p.add_argument("--optimization-source-receipt",type=Path)')
text=text.replace('p.add_argument("--sampling-mode",choices=("domain30",),default="domain30")','p.add_argument("--sampling-mode",choices=("uniform",),default="uniform")')
text=text.replace('    return p\n\n\ndef validate_arguments(args):','''    p.add_argument("--resume-uniform-full",type=Path,required=True)
    p.add_argument("--uniform-full-source-receipt",type=Path,required=True)
    p.add_argument("--uniform-full-runtime-proof",type=Path,required=True)
    p.add_argument("--portuguese-replay-manifest",type=Path,default=DATA/"portuguese_retention_v1/replay_train_pt.jsonl")
    p.add_argument("--portuguese-dev-manifest",type=Path,default=DATA/"portuguese_retention_v1/validation_pt.jsonl")
    p.add_argument("--portuguese-baseline-proof",type=Path,default=OUTPUT/"Portuguese_retention_baselines_v1.json")
    p.add_argument("--portuguese-seed",type=int,default=20261012)
    return p


def validate_arguments(args):''',1)
text=text.replace('    closed_source_identity(args)\n    return args','''    if args.resume_weights or args.resume_optimization or args.resume_runtime_migration:raise ValueError("Full207 source or own exact only")
    if args.batch_size!=8 or args.accumulation!=2 or args.head_lr!=1e-5 or args.lora_lr!=3e-4 or args.replay_kl!=10 or args.schedule!="constant":
        raise ValueError("Original qualified full207 B8x2/KD10 constant recipe required")
    if args.eval_every_updates!=250 or args.sampling_mode!="uniform":raise ValueError("Fulluniform/DEV250 required")
    uniform_preflight(args)
    return args''',1)
ast.parse(text)
target=HERE/'train_lora_batched_tsc_uniform_full_v1.py'
with target.open('x',encoding='utf-8',newline='\n') as f:f.write(text)
print(target)
