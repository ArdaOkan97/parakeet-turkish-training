"""NEW TSC domain30 source admission; immutable v2 corpus/approval functions reused."""
from pathlib import Path
import hashlib
import json
from tsc_training_inputs_v2 import (policy,require,file_record,read_record,COMMON_TRAINING_KEYS,
    DOMAIN_TRAINING_KEYS,DOMAIN_SOURCES,KIND)
from tsc_domain_sampler_v3 import domain_horizon
HERE=policy.OUTPUT


def annotate_tr_frame_metadata(processor, rows, metadata_path, proof_path, max_tokens):
    """Same frozen v2 checks, streamed metadata and in-place transient row annotation.

    No manifest/proof is edited. Rows are the worker's independently loaded,
    representability-checked dictionaries, not the frozen parent's context.
    This avoids retaining a second253k metadata/augmented-row dictionary array.
    """
    proof=policy.read_json(proof_path)
    require(proof.get("kind")=="TSC_supplemental_TR_frame_target_inputs_v2" and proof.get("passed") is True
        and proof.get("helper_sha256")==policy.sha256(HERE/"tsc_training_inputs_v2.py") and max_tokens==180
        and proof["files"]["TR_metadata"]==file_record(metadata_path), "Exact frozen v2 metadata proof required")
    tokenizer_sha=hashlib.sha256(json.dumps(json.loads(processor.tokenizer._tokenizer.to_str()),
        ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    require(tokenizer_sha==policy.TOKENIZER_SEMANTIC_SHA, "Keep exact original8199 tokenizer")
    require(len(rows)==proof["clips"],"Whole approved combined pool required")
    path=Path(metadata_path);before=(path.stat().st_size,path.stat().st_mtime_ns)
    with path.open(encoding="utf-8") as stream:
        for row in rows:
            line=stream.readline();require(bool(line.strip()),"Missing/blank ordered metadata row")
            item=json.loads(line);mel=item["actual_PCM_frames"]//160;frames=(mel+7)//8
            require(row["language"]=="tr" and row["source_id"]==item["source_id"]
                and int(round(row["duration"]*16000))==item["actual_PCM_frames"]
                and item["valid_mel_frames"]==mel and item["encoder_frames"]==frames,
                "Approved ordered actual PCM/frame metadata changed")
            labels=processor.tokenizer(row["text"])["input_ids"]
            require(1<=len(labels)<=max_tokens and len(labels)==item["target_tokens"]
                and min(labels)>=0 and max(labels)<8199 and 8192 not in labels
                and hashlib.sha256(json.dumps(labels,sort_keys=True).encode()).hexdigest()==item["label_ids_sha256"],
                "Exact ungrouped supplemental target IDs changed")
            if "canonical_WAV_stat_at_preparation" in item:
                stat=Path(row["audio_filepath"]).stat()
                require(item["canonical_WAV_stat_at_preparation"]=={"size_bytes":stat.st_size,"mtime_ns":stat.st_mtime_ns},
                    "TSC canonical WAV stat changed; fresh verification required")
            row.update(encoder_frames=frames,target_tokens=len(labels),training_input_type=KIND,
                tr_metadata_labels=labels,tr_frame_metadata_sha256=proof["metadata_sha256"])
        require(not stream.read(1),"Extra metadata rows beyond approved pool")
    require(before==(path.stat().st_size,path.stat().st_mtime_ns),"Metadata changed while streamed")
    return rows


def approved_source_recipe(source,closure):
    recipe=source["config_identity"]["training_arguments"]
    require(set(recipe)==set(COMMON_TRAINING_KEYS|DOMAIN_TRAINING_KEYS),"Only the reviewed original domain30 source recipe is eligible")
    identity=source["config_identity"].get("domain_sampling_identity",{})
    require(recipe["sampling_mode"]=="domain30" and recipe["fleurs_fraction"]==.3 and type(recipe["replay_seed"]) is int
        and identity.get("sampling_mode")=="domain30" and identity.get("replay_seed")==recipe["replay_seed"]
        and identity.get("turkish_seed")==recipe["seed"] and identity.get("replay_rng_independent_of_turkish") is True
        and identity.get("fleurs_fraction_numerator")==3 and identity.get("fleurs_fraction_denominator")==10
        and identity.get("batch_size")==8 and identity.get("accumulation")==2 and identity.get("replay_per_update")==8,
        "Preserve actual30% FLEURS and independent replay source identity")
    require(closure.get("allow_source_domain30_expansion") is True and closure.get("source_sampling_metadata")==identity,
        "ROOT must explicitly approve expansion of other-domain CV to CV+TSC while preserving domain30/replay")
    groups=source.get("sampling",{}).get("initial_pool",{}).get("domain",{})
    require(groups.get("fleurs",{}).get("pool_clips")==2037 and groups.get("cv",{}).get("pool_clips")==72846,
        "Actual original source domain pool membership/counts differ")
    for name,digest in DOMAIN_SOURCES.items():
        require(source["code_sha256"].get(name)==digest and policy.sha256(HERE/name)==digest,
            "Closed original domain source changed: "+name)
    return recipe.copy()


def closed_source_identity(args):
    closure = policy.read_json(args.source_closure)
    require(closure.get("kind") == "root_owned_TSC_domain30_closed_source_v3" and closure.get("passed") is True
            and closure.get("root_observed_owned_session_completed") is True and closure.get("child_exit_code") == 0,
            "Actual CLOSED qualifying original8199 uniform/domain source is required")
    for key in ("source_checkpoint", "source_selection", "source_run_config", "source_training_history", "source_development_summary"):
        read_record(closure[key])
    source = policy.read_json(closure["source_run_config"]["path"])
    selection = policy.read_json(closure["source_selection"]["path"])
    require(source["run_id"] == selection["run_id"] == closure["source_run_id"]
            and selection.get("accepted") is True and selection.get("heldout_tests_evaluated") is False
            and selection.get("selection_split") == "FLEURS validation only",
            "Closed source selection/config/retention differ")
    dev = policy.read_json(closure["source_development_summary"]["path"])
    require(Path(closure["source_development_summary"]["path"]).resolve()
            == HERE / "results" / (source["run_id"]+"_step_"+str(selection["selected_step"])) / "summary.json",
            "Bind the actual selected source DEV result folder")
    baseline = policy.read_json(HERE / "results/base_v3/summary.json")
    require(dev["validation_tr"]["asr"]["wer"] == selection["turkish_validation_wer"]
            and all(dev["validation_"+lang]["asr"]["utterances"] == 50 for lang in policy.CONFIGS)
            and all(dev["validation_"+lang]["asr"]["wer"] <= baseline["validation_"+lang]["asr"]["wer"]
                    + max(.005, .1*baseline["validation_"+lang]["asr"]["wer"])
                    for lang in ("en", "es", "it", "fr", "de")),
            "Selected source's complete fixed DEV summary must pass all five original retention gates")
    history = policy.read_json(closure["source_training_history"]["path"])
    require(len(history) == selection["completed_steps"]
            and all(row["step"] == index for index, row in enumerate(history, 1)),
            "Closed source history/completed frontier differs")
    if selection["selected_step"] > 0:
        gate = history[selection["selected_step"]-1].get("validation", {})
        require(gate.get("retention_gate_passed") is True and gate.get("turkish_wer") == selection["turkish_validation_wer"],
                "Selected BEST is not the recorded qualifying DEV point")
    require(Path(selection["weights"]).resolve() == Path(closure["source_checkpoint"]["path"]).resolve(),
            "Use only the CLOSED selected BEST, never a nonqualifying latest")
    if args.resume_exact is None:
        require(args.resume_weights is not None and file_record(args.resume_weights) == closure["source_checkpoint"],
                "Initial data-only arm requires approved BEST weights and a fresh optimizer/sampler")
    require(args.resume_runtime_migration is None and args.train_limit is None and args.supplement is None
            and args.reuse_input_audit is None, "No legacy exact-state migration/subset/supplement/reuse policy for changed TSC pool")
    source_science = approved_source_recipe(source, closure)
    for key, value in source_science.items():
        if key not in ("epochs", "steps"):
            require(getattr(args, key) == value, "TSC initial arm changes only data/horizon; scientific argument differs: "+key)
    combined = policy.read_json(args.tsc_input_proof)
    require(source.get("turkish_clips_used") == 74883 and source.get("replay_clips_used") == 1595
            and source["config_identity"]["extended_tokenizer_sha256"] == policy.TOKENIZER_SEMANTIC_SHA
            and source["config_identity"]["tr_pool_sha256"] == combined["original_TR_source_ids_sha256"]
            and source["config_identity"]["replay_pool_sha256"] == policy.REPLAY_SOURCE_SET_SHA,
            "Source must be the original8199/74883TR/1595replay lineage, not a same-shaped alternate pool")
    require(args.epochs == 1. and args.steps == domain_horizon(2037,combined["clips"]-2037),
            "One minimum-domain pass requires exact FLEURS30/CV+TSC70 quota horizon, not the old uniform cap")
    require(args.rank == 8 and args.alpha == 16 and args.last_n_layers is None and args.batch_size == 8
            and args.accumulation == 2 and args.replay_fraction == .5 and args.replay_kl == 10
            and args.head_lr == 5e-5 and args.lora_lr == 3e-4 and args.new_token_lr_multiplier == 6
            and args.tdt_implementation == "nemo", "Keep the agreed original8199/rank8/KD10/B8x2 recipe")
    require(args.native_max_joint_bytes == source["config_identity"]["native_loss"]["max_joint_bytes"],
            "Do not silently change the measured native joint memory bound")
    return {"closure": file_record(args.source_closure), "source_run_id": closure["source_run_id"],
            "source_selected_step": selection["selected_step"], "source_checkpoint": closure["source_checkpoint"],
            "source_scientific_arguments": source_science,
            "source_sampling_transition":{"source_mode":"domain30","target_mode":"domain30",
                "FLEURS_fraction_preserved":.3,"source_replay_RNG_independent":True,"target_replay_RNG_independent":True,
                "target_other_domain":"uniform originalCV plus approved TSC", "same_replay_seed_preserved":True,
                "initial_fresh_Adam_sampler_RNG":True,"source_replay_frontier_continuation_claimed":False},
            "declared_scientific_changes": ["Append approved TSC TRAIN data to exact74883 prefix; original1595 replay unchanged",
                                             "One full CV+TSC coverage pass with30% FLEURS; exact minimum-domain quota horizon",
                                             "Preserve30% FLEURS and independent replay; expand70%other domain CV->uniformCV+TSC; initial new Adam/sampler/RNG"],
            "initial_optimizer_sampler_RNG_history_counters": "fresh", "exact_old_pool_continuation_claimed": False}
