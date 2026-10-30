"""Create NEW inactive BPE v2 admission/worker siblings, standard library only."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / 'outputs/parakeet_turkish'

def save(name, text):
    ast.parse(text)
    with (HERE/name).open('x', encoding='utf-8', newline='\n') as out:
        out.write(text)

helper = (HERE/'hf_bpe512_training_v1.py').read_text(encoding='utf-8')
start = helper.index('def closed_source_identity(args):')
end = helper.index('\n\ndef install_candidate_student(', start)
admission = '''COMMON_TRAINING_KEYS = frozenset(("epochs", "steps", "batch_size", "accumulation", "replay_fraction", "bucket_size", "seed",
    "max_tokens", "head_lr", "lora_lr", "new_token_lr_multiplier", "replay_kl", "duration_kl", "feature_distillation",
    "rank", "alpha", "last_n_layers", "chunk_size", "tdt_implementation", "warmup_steps", "schedule", "min_lr_ratio",
    "eval_per_epoch", "eval_epoch_fraction", "eval_every_updates", "min_improvement", "no_encoder_checkpoint",
    "phonetic_sh_donors", "train_limit", "early_stop_patience", "max_retention_failures", "target_turkish_wer"))
DOMAIN_TRAINING_KEYS = frozenset(("sampling_mode", "fleurs_fraction", "replay_seed"))
DOMAIN_SOURCES = {"train_lora_batched_domain30_v1.py":"7242b0ce65619f8df38593e89c6ed5b7a01487d1d1174c2e28f73d00bff8ee4f",
    "turkish_domain_sampler_v2.py":"822d0ddb608d719112e4a4285ad42a76ef7f1e4ede0cf4d97211fe38f97c62fd",
    "domain30_training_identity_v1.py":"151bf1b3236e3e23c269f7be2c2e94bf12a71b1e547b64a70f26d6f77ae81f5e"}


def common_source_recipe(source, closure):
    """Admit the actual CLOSED domain recipe without hiding its RNG policy."""
    recipe = source["config_identity"]["training_arguments"]
    require(set(recipe) == COMMON_TRAINING_KEYS | DOMAIN_TRAINING_KEYS,
            "BPE v2 requires the reviewed CLOSED domain source, not an older uniform fallback")
    identity = source["config_identity"].get("domain_sampling_identity", {})
    require(recipe["sampling_mode"] == "domain30" and recipe["fleurs_fraction"] == .3
        and type(recipe["replay_seed"]) is int and identity.get("sampling_mode") == "domain30"
        and identity.get("replay_seed") == recipe["replay_seed"] and identity.get("turkish_seed") == recipe["seed"]
        and identity.get("replay_rng_independent_of_turkish") is True
        and identity.get("fleurs_fraction_numerator") == 3 and identity.get("fleurs_fraction_denominator") == 10
        and identity.get("batch_size") == 8 and identity.get("accumulation") == 2
        and identity.get("replay_per_update") == 8,
        "Actual original independent-replay domain sampling identity differs")
    require(closure.get("allow_source_sampler_transition_to_BPE_uniform") is True
        and closure.get("source_sampling_metadata") == identity,
        "ROOT must explicitly approve domain/independent-replay to BPE-uniform/shared-RNG transition")
    for name, digest in DOMAIN_SOURCES.items():
        require(source["code_sha256"].get(name) == digest and sha_file(HERE/name) == digest,
                "Closed domain source implementation differs: " + name)
    return {key: value for key, value in recipe.items() if key in COMMON_TRAINING_KEYS}


def source_sampling_transition(source, closure):
    common_source_recipe(source, closure)
    recipe = source["config_identity"]["training_arguments"]
    return {"source_mode": "domain30", "source_sampling_metadata": source["config_identity"]["domain_sampling_identity"],
        "source_fleurs_fraction": recipe["fleurs_fraction"], "source_replay_seed": recipe["replay_seed"],
        "target_mode": "BPE_uniform", "source_replay_RNG_independent": True,
        "target_replay_RNG_independent": False, "original_replay_pool_teacher_labels_unchanged": True,
        "replay_order_bitwise_identity_claimed": False,
        "scope": "Initial model-only qualifying domain BEST; NEW Adam/sampler/RNG. Later SAME-run exact windows retain this chosen BPE science."}


def closed_source_identity(args):
    """Require ROOT-confirmed CLOSED domain BEST and its NEW conversion."""
    closure = read_json(args.bpe_source_closure)
    require(closure.get("kind") == "root_owned_BPE512_closed_source_v2" and closure.get("passed") is True
        and closure.get("root_observed_owned_session_completed") is True and closure.get("child_exit_code") == 0
        and closure.get("latest_closed_qualified_source_confirmed") is True
        and isinstance(closure.get("source_choice_rationale"), str) and bool(closure["source_choice_rationale"].strip()),
        "ROOT-confirmed latest CLOSED qualifying domain source and explicit choice required")
    paths = {}
    for name in ("source_checkpoint", "source_selection", "source_run_config", "source_training_history", "source_development_summary"):
        item = closure[name]; path = Path(item["path"]).resolve()
        require(path.is_relative_to(ROOT) and sha_file(path) == item["sha256"], "Closed source changed: " + name)
        paths[name] = path
    require(paths["source_checkpoint"] == Path(args.bpe_source_checkpoint).resolve()
        and paths["source_selection"] == Path(args.bpe_source_selection).resolve(), "Explicit source paths differ")
    selection, source_config = read_json(paths["source_selection"]), read_json(paths["source_run_config"])
    require(selection.get("accepted") is True and type(selection.get("selected_step")) is int
        and 0 <= selection["selected_step"] <= selection["completed_steps"]
        and selection.get("run_id") == closure.get("source_run_id") == source_config.get("run_id")
        and selection["run_id"] != args.run_id and selection.get("heldout_tests_evaluated") is False
        and selection.get("selection_split") == "FLEURS validation only"
        and type(selection.get("turkish_validation_wer")) in (int, float)
        and math.isfinite(selection["turkish_validation_wer"])
        and Path(selection["weights"]).resolve() == paths["source_checkpoint"], "Qualifying selected BEST required")
    require(paths["source_development_summary"] == HERE/"results"/(selection["run_id"]+"_step_"+str(selection["selected_step"]))/"summary.json",
        "Actual selected source DEV folder required")
    dev = read_json(paths["source_development_summary"]); baseline = read_json(HERE/"results/base_v3/summary.json")
    require(dev["validation_tr"]["asr"]["wer"] == selection["turkish_validation_wer"]
        and all(dev["validation_"+lang]["asr"]["utterances"] == 50 for lang in ("tr","en","es","it","fr","de"))
        and all(dev["validation_"+lang]["asr"]["wer"] <= baseline["validation_"+lang]["asr"]["wer"]
                + max(.005, .1*baseline["validation_"+lang]["asr"]["wer"]) for lang in ("en","es","it","fr","de")),
        "Complete selected source DEV and all five original retention gates required")
    history = read_json(paths["source_training_history"])
    require(len(history) == selection["completed_steps"] and all(row["step"] == index for index,row in enumerate(history,1)),
        "Source completed history differs")
    if selection["selected_step"]:
        gate = history[selection["selected_step"]-1].get("validation", {})
        require(gate.get("retention_gate_passed") is True and gate.get("turkish_wer") == selection["turkish_validation_wer"],
            "Source BEST is not its recorded qualifying DEV")
    common = common_source_recipe(source_config, closure)
    require(source_config.get("base_revision") == "541d1f99c6b0c3cd0b11a95167540bb8edefd82b"
        and source_config.get("turkish_clips_used") == 74883 and source_config.get("replay_clips_used") == 1595
        and source_config["config_identity"]["extended_tokenizer_sha256"] == BASELINE_SEMANTIC_SHA
        and source_config["config_identity"]["replay_pool_sha256"] == REPLAY_IDS_SHA,
        "Original8199/74883TR/1595replay lineage required")
    for key,value in common.items():
        require(getattr(args,key) == value, "Only declared tokenizer/target/geometry/sampler changes allowed; source recipe differs: " + key)
    conversion = read_json(args.bpe_conversion_report)
    require(conversion.get("kind") == "HF_Turkish_BPE_512_weights_only_warm_start_v1"
        and conversion.get("status") == "complete" and conversion.get("passed") is True
        and conversion.get("new_optimizer_required") is True and conversion.get("exact_training_resume") is False
        and conversion.get("target_run_id") == args.run_id, "Completed NEW weights-only conversion required")
    source = conversion["source"]
    require(Path(source["path"]).resolve() == paths["source_checkpoint"]
        and source["sha256"] == closure["source_checkpoint"]["sha256"]
        and source["run_id"] == selection["run_id"] and source["selected_step"] == selection["selected_step"]
        and source["selection_sha256"] == closure["source_selection"]["sha256"], "Conversion source/selection differs")
    converted = Path(conversion["output_checkpoint"]).resolve()
    require(converted.is_relative_to(ROOT) and sha_file(converted) == conversion["output_checkpoint_sha256"], "Original converted warmweights changed")
    if args.resume_exact is None:
        require(args.resume_weights is not None and Path(args.resume_weights).resolve() == converted,
            "Initial BPE arm uses approved NEW model-only converted weights")
    else:
        require(args.resume_weights is None, "Exact BPE continuation must not reset Adam from warmweights")
    return {"closed_source":closure,"closure_sha256":sha_file(args.bpe_source_closure),
        "conversion_sha256":sha_file(args.bpe_conversion_report),"selection":selection,
        "source_scientific_arguments":common,"source_sampling_transition":source_sampling_transition(source_config,closure),
        "converted_weights_path":str(converted),"converted_weights_sha256":conversion["output_checkpoint_sha256"]}


def established_source_equivalence(run_dir, run_id, source_identity, *, trainer_path, runtime_path):
    """Exact windows reuse the immutable actual first-window GPU proof, never infer it."""
    value = read_json(Path(run_dir)/"BPE512_actual_GPU_source_equivalence.json")
    require(value.get("kind") == "HF_BPE512_actual_unsuppressed_GPU_source_equivalence_v1"
        and value.get("passed") is True and value.get("actual_GPU_decode") is True
        and value.get("before_optimizer_creation") is True and value.get("decoded_clips") == 300
        and value.get("run_id") == run_id and value.get("source_identity") == source_identity
        and value.get("student_geometry") == GEOMETRY
        and value.get("converted_weights_sha256") == source_identity["converted_weights_sha256"]
        and value.get("trainer_sha256") == sha_file(trainer_path) and value.get("candidate_runtime_sha256") == sha_file(runtime_path)
        and value.get("candidate_tokenizer_sha256") == TOKENIZER_SEMANTIC_SHA
        and not value.get("raw_transcript_changes") and not value.get("token_duration_emission_changes")
        and value.get("source_gate",{}).get("retention_gate_passed") is True
        and value.get("candidate_gate",{}).get("retention_gate_passed") is True,
        "Immutable actual pre-Adam300 source equality/gates/source identity required for exact continuation")
    config = read_json(Path(run_dir)/"run_config.json")
    require(config["config_identity"]["BPE512"]["source_equivalence_sha256"] == sha_file(Path(run_dir)/"BPE512_actual_GPU_source_equivalence.json"),
        "Initial config no longer binds its actual source equivalence report")
    return value
'''
helper = helper[:start] + admission + helper[end:]
helper = helper.replace('No optimizer state is\nreused.', 'Initial admission starts fresh Adam. Later SAME-run exact windows restore all\noptimizer/sampler/RNG/history state under the original immutable identity.')
save('hf_bpe512_training_v2.py', helper)

worker = (HERE/'train_lora_batched_bpe512_v1.py').read_text(encoding='utf-8')
worker = worker.replace('hf_bpe512_training_v1', 'hf_bpe512_training_v2').replace('train_lora_batched_bpe512_v1.py', 'train_lora_batched_bpe512_v2.py').replace('launch_native_training_bpe512_v1.py','launch_native_training_bpe512_v2.py')
worker = worker.replace('native_joint_bound, GEOMETRY)', 'native_joint_bound, established_source_equivalence, GEOMETRY)')
worker = worker.replace('    if best_path.exists() or latest_path.exists():', '    if not args.resume_exact and (best_path.exists() or latest_path.exists()):')
worker = worker.replace('    for tag in (args.run_id+"_bpe_source_preflight",args.run_id+"_step_0"):\n        if (OUTPUT/"results"/tag).exists():raise ValueError("Fresh actual source-equivalence result tags required")',
'''    if not args.resume_exact:
        for tag in (args.run_id+"_bpe_source_preflight",args.run_id+"_step_0"):
            if (OUTPUT/"results"/tag).exists():raise ValueError("Fresh actual source-equivalence result tags required")
    elif not (run_dir/"run_config.json").is_file() or not (run_dir/"initial_development.json").is_file():
        raise ValueError("Exact BPE continuation requires its established initial run/config/DEV")''')
start=worker.index('    source = torch.load(args.bpe_source_checkpoint')
end=worker.index('    # Adam is constructed ONLY after',start)
initial=worker[start:end]
# Stable original converted file is a scientific input in both modes.
worker=worker[:start]+'''    if args.resume_exact is None:
'''+''.join('    '+line+'\n' for line in initial.rstrip().splitlines())+'''    else:
        source_equivalence = established_source_equivalence(run_dir,args.run_id,source_identity,
            trainer_path=Path(__file__),runtime_path=OUTPUT/"hf_bpe512_training_v2.py")
        original_vocabulary = vocabulary
        vocabulary = install_candidate_student(processor,model,args.bpe_candidate_dir,torch_module=torch)
        vocabulary["original_six_character_audit"] = original_vocabulary
        resume_path = args.resume_exact
        resume = torch.load(resume_path,map_location="cpu",weights_only=True)
        if resume.get("schema") != 2 or resume.get("run_id") != args.run_id:
            raise ValueError("Exact BPE continuation requires SAME-run full schema2 state")
        validate_trainable_checkpoint(model,resume,encoder_lora_metadata(model))
        model.load_state_dict(resume["weights"],strict=False)
    # Initial GPU proof remains immutable; exact windows verify it without decoding.
'''+worker[end:]
worker=worker.replace('args.bpe_conversion_report,args.resume_weights,','args.bpe_conversion_report,Path(source_identity["converted_weights_path"]),')
worker=worker.replace('"initial_optimizer_sampler_RNG_history_counters"', '"initial_optimizer_sampler_RNG_history_counters"')
worker=worker.replace('"method":"bucketed true-batch all-head adaptation plus encoder attention LoRA",',
'''"method":"bucketed true-batch all-head adaptation plus encoder attention LoRA",
              "source_sampling_transition":source_identity["source_sampling_transition"],
              "initial_domain_to_BPE_model_only_admission":source_identity,''')
# This isolated family never supports runtime/scientific migration; all exact identity is equal.
start=worker.index('def exact_resume_identity_matches(')
end=worker.index('\n\ndef parser():',start)
worker=worker[:start]+'''def exact_resume_identity_matches(resume, args, current):
    if args.resume_runtime_migration is not None:
        raise ValueError("BPE same-science continuation forbids runtime/scientific migration")
    return resume.get("config_identity") == current
'''+worker[end:]
worker=worker.replace('if (args.resume_exact is not None or args.resume_weights is None or args.resume_runtime_migration is not None',
                      'if ((args.resume_exact is None) == (args.resume_weights is None) or args.resume_runtime_migration is not None')
worker=worker.replace('BPE512 v1 is fresh-optimizer whole-pool native-loss warmstart, six-character6x/BPE1x only',
    'BPE512 v2 admits a CLOSED domain model-only source or SAME-run exact fullstate, whole-pool/native-loss/six6x/BPE1x only')
save('train_lora_batched_bpe512_v2.py',worker)
print('Saved NEW BPE admission/worker v2; all original v1 files untouched.')
