"""Inactive metadata boundary for the human-authorized paired decoder screen.

No source is admitted until both matched data arms actually close at 65,948 and
their completed-study review binds one coherent strict-qualified saved BEST.
No ML imports, checkpoint deserialization or storage writes occur at import.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import random

SOURCE = Path(__file__).resolve().parent
ROOT = SOURCE.parents[1]
STORAGE = Path(r"E:\gpu_training_scratchpad")
RECEIPT_KIND = "ROOT_completed_matched_TRAIN_for_BPE_full_decoder_v1"
REVIEW_KIND = "ROOT_completed_matched_TRAIN_BPE_source_selection_v1"
PHASE = "HF_TSC_BPE_full_decoder_matched_screen_v1"
PROOF_NAME = "BPE_full_decoder_initial_source_equivalence.json"
HANDOFF_SHA = "533d24203d62d2d0efa6569fb80d970db57319d3f7461ca807ce5e9e52912036"
UPDATES = 2000
MEAN_U = 26.642061797177284
LANGUAGES = ("tr", "en", "es", "it", "fr", "de", "pt")
REFERENCES = dict(tr=825, en=1045, es=1289, it=1158, fr=1072, de=1004, pt=1095)
LIMITS = dict(en=69, es=52, it=30, fr=74, de=70, pt=66)


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    digest = hashlib.sha256()
    for part in json.JSONEncoder(sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                                 allow_nan=False).iterencode(value):
        digest.update(part.encode("utf-8"))
    return digest.hexdigest()


def permitted(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT) or path.is_relative_to(STORAGE.resolve()),
            "Only existing authorized project or designated E: storage access")
    return path


def sha(path):
    with permitted(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def record(path):
    path = permitted(path)
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def checked(value, *, checkpoint=False):
    require(isinstance(value, dict) and set(value) >= {"path", "sha256", "bytes"}
            and len(value["sha256"]) == 64 and type(value["bytes"]) is int,
            "Complete immutable byte record required")
    path = permitted(value["path"])
    require(path.is_file() and path.stat().st_size == value["bytes"], "Artifact stat changed: " + str(path))
    if not checkpoint:
        require(sha(path) == value["sha256"], "Artifact bytes changed: " + str(path))
    return path


def read(value):
    return json.loads(checked(value).read_text(encoding="utf-8"))


def validate_scores(summary):
    counts = {}
    for language, reference in REFERENCES.items():
        score = summary["validation_" + language]["asr"]
        require(score["utterances"] == 50 and score["reference_words"] == reference
                and type(score["word_errors"]) is int and score["word_errors"] >= 0
                and math.isfinite(score["wer"]) and score["wer"] == score["word_errors"] / reference,
                "Exact count-consistent fixed seven-language DEV required")
        counts[language] = score["word_errors"]
    return counts, all(counts[name] <= limit for name, limit in LIMITS.items())


def validate_study_review(review):
    require(review.get("kind") == REVIEW_KIND and review.get("passed") is True
            and review.get("actual_both_matched_arms_complete") is True,
            "Actual completed paired data-study source decision required; preparation is not admission")
    preserved = {}
    for arm in ("base", "expanded"):
        value = read(review["completed_arms"][arm]["preservation"])
        require(value.get("passed") is True and value.get("data_arm") == arm
                and value.get("completed_step") == 65948
                and value.get("actual_owned_exit", {}).get("exit_code") == 0
                and value.get("native_child_exit_code") == 0 and value.get("native_sources_unchanged") is True
                and value.get("stop_category") == "configured_steps_complete"
                and value.get("data_expansion_epoch", {}).get("full_pass_complete") is True,
                "Both data arms must actually finish the complete declared budget and DEV")
        preserved[arm] = value
    selected = review["selected"]
    arm = selected["data_arm"]
    require(arm in preserved and selected["run_id"] == preserved[arm]["run_id"]
            and selected["step"] == preserved[arm]["selected_qualified_BEST_step"]
            and selected["checkpoint"] == preserved[arm]["artifacts"]["best_checkpoint"]["archived"],
            "Coherent actual saved qualified BEST, never mixed weights/optimizer or raw minimum")
    require(selected["checkpoint"]["sha256"] != "" and 40750 <= selected["step"] <= 65948,
            "Actual selected source frontier required")
    scores = read(selected["DEV"]["summary"])
    counts, qualifies = validate_scores(scores)
    require(qualifies and counts["tr"] / 825 == preserved[arm]["selected_qualified_BEST_TR_WER"],
            "Chosen coherent source must pass all six original integer gates")
    require(counts["tr"] / 825 == min(p["selected_qualified_BEST_TR_WER"] for p in preserved.values()),
            "Choose the measured strict-qualified winner of the completed matched study")
    return selected, preserved


def source_context(receipt_path, run_id=None):
    path = permitted(receipt_path)
    receipt_record = record(path)
    receipt = json.loads(path.read_text(encoding="utf-8"))
    require(receipt.get("kind") == RECEIPT_KIND and receipt.get("status") == "actual_completed_study_qualified_BEST_for_decoder_screen"
            and receipt.get("decoder_arm") in ("bpe512", "original8199")
            and (run_id is None or receipt.get("target_run_id") == run_id), "Purpose-bound paired decoder source receipt required")
    handoff = read(receipt["handoff"])
    require(receipt["handoff"]["sha256"] == HANDOFF_SHA and handoff["design"]["proposed_screen_updates_per_arm"] == UPDATES,
            "Direct human BPE-next scientific handoff changed")
    review = read(receipt["completed_study_review"])
    selected, preserved = validate_study_review(review)
    require(receipt["source_run_id"] == selected["run_id"] and receipt["source_step"] == selected["step"]
            and receipt["source_checkpoint"] == selected["checkpoint"], "Receipt changed chosen source")
    checked(selected["checkpoint"], checkpoint=True)
    config = read(selected["run_config"])
    identity = config["config_identity"]
    args = identity["training_arguments"]
    require(config["run_id"] == selected["run_id"] and args["data_arm"] == selected["data_arm"]
            and args["head_lr"] == 1e-5 and args["encoder_lr_schedule"] == "cosine"
            and args["lora_lr"] == 1e-4 and args["encoder_min_lr"] == 3e-5
            and args["encoder_schedule_updates"] == 25198 and args["batch_size"] == 8
            and args["accumulation"] == 2 and args["replay_fraction"] == .125
            and args["tr_tdt_weighting"] == "token_population" and args["tr_tdt_population_mean_U"] == MEAN_U
            and args["replay_kl"] == 40 and args["duration_kl"] == 2 and args["feature_distillation"] == 20
            and args["portuguese_token_kl_multiplier"] == 1 and args["retention_stop_policy"] == "monitor_only",
            "Reviewed original8199 matched data-study recipe required")
    domain = identity["domain_sampling_identity"]
    original_purpose = read(identity["data_expansion_transition"]["source_receipt"])
    original_config = read(original_purpose["artifacts"]["run_config"])
    original_identity = original_config["config_identity"]
    require(domain["data_expansion_transition"]["parent_identity_sha256"]
            == canonical(original_identity["domain_sampling_identity"]), "Original40750 sampling lineage changed")
    sampler_context = {"source_step": 40750, "data_arm": selected["data_arm"],
        "data_expansion_updates": 25198, "configured_end_step": 65948,
        "TR_pool_clips": 352768 if selected["data_arm"] == "expanded" else 253798,
        "parent_identity": original_identity,
        "source_checkpoint": domain["data_expansion_transition"]["source_checkpoint"]}
    # The actual frozen producer stores argparse fields at the config top level.
    # Retain only consumer fields, rather than cloning the complete identity.
    source_args = {key: config[key] for key in ("indexed_tr_directory", "portuguese_baseline_proof",
        "rank", "alpha", "last_n_layers", "seed", "replay_seed", "portuguese_seed",
        "fleurs_fraction", "bucket_size", "phonetic_sh_donors")}
    baseline_path = permitted(source_args["portuguese_baseline_proof"])
    require(identity["input_sha256"].get(str(baseline_path)) == sha(baseline_path), "Pinned seven-language baseline changed")
    baseline_proof = json.loads(baseline_path.read_text(encoding="utf-8"))
    closed_controller = read(preserved[selected["data_arm"]]["artifacts"]["controller_status"]["archived"])
    reference_peak = closed_controller["peak_observed_owned_private_bytes"]
    require(type(reference_peak) is int and reference_peak > 0,
            "Actual prior complete source-policy private peak required; new decoder fit remains unmeasured")
    context = {"receipt": receipt, "receipt_record": receipt_record, "review": review,
        "source_step": selected["step"], "source_selected_step": selected["step"],
        "source_checkpoint": selected["checkpoint"], "source_run_id": selected["run_id"],
        "source_DEV": selected["DEV"], "source_summary": read(selected["DEV"]["summary"]),
        "source_training_pool": selected["training_pool"], "parent_identity": identity,
        "parent_config": config, "parent_recipe": args, "source_arguments": source_args,
        "source_sampler_context": sampler_context, "source_data_arm": selected["data_arm"],
        "TR_pool_clips": sampler_context["TR_pool_clips"], "decoder_updates": UPDATES,
        "configured_end_step": selected["step"] + UPDATES,
        "combined_baseline": baseline_proof["baseline_overlay"],
        "source_history_prefix_sha256": selected["history_prefix_sha256"],
        "source_counter": 0, "source_selected_BEST": selected["step"],
        "completed_parent_step": 65948,
        "chosen_source_original_data_epoch_draws": 14*(selected["step"]-40750),
        "chosen_source_original_data_epoch_unique": min(14*(selected["step"]-40750), sampler_context["TR_pool_clips"]),
        "reference_source_policy_private_peak": reference_peak,
        "BPE_label_preparation": receipt.get("BPE_label_preparation")}
    return context


def decoder_declaration(context, arm):
    require(arm == context["receipt"]["decoder_arm"], "Receipt/arm binding changed")
    return {"kind": "TSC_BPE_full_decoder_matched_screen_declaration_v1", "source_step": context["source_step"],
        "decoder_arm": arm, "vocabulary": 8711 if arm == "bpe512" else 8199,
        "screen_updates": UPDATES, "configured_end_step": context["configured_end_step"],
        "fork_from_coherent_qualified_BEST_not_completed_latest": True,
        "completed_parent_step_preserved": 65948,
        "updates_rewound_from_archived_completed_branch": 65948-context["source_step"],
        "chosen_source_original_data_epoch_draws": context["chosen_source_original_data_epoch_draws"],
        "chosen_source_original_data_epoch_unique": context["chosen_source_original_data_epoch_unique"],
        "archived_completed_parent_unique": context["TR_pool_clips"],
        "later_parent_history_preserved_not_relabelled_as_chosen_source_exposure": True,
        "bounded_screen_not_full_epoch": True, "encoder_and192LoRA_frozen_in_both_arms": True,
        "all15_existing_decoder_tensors_active": True, "mature_old_Adam_clock": context["source_step"],
        "frozen192_Adam_storage": "Immutable full source checkpoint, referenced by exact SHA; no advanced clock or duplicate CPU moment pool",
        "fresh512_Adam_clock": 0 if arm == "bpe512" else None,
        "mature_learning_rate": 1e-5, "fresh_learning_rate": 1e-4 if arm == "bpe512" else None,
        "six_glyph_gain": 6, "six_gain_rows_only": [8193, 8199], "weight_decay": .01,
        "TR_PT_oldfive": [14, 1, 1], "physical_batch_accumulation": [8, 2],
        "TR_population_denominator": MEAN_U,
        "same_entry_RNG_and_sampler_RNG_state_for_paired_arms": True,
        "same_future_sampling_ID_order": True,
        "identical_model_dropout_draws_after_different_target_lengths_claimed": False,
        "numerical_bitwise_trajectories_claimed": False,
        "sampling_uses_original8199_T_U": True, "replay_original8193_labels_retained": True,
        "student_token_softmax_keeps_all8711_mass": arm == "bpe512",
        "PT_KL_duration_feature": [40, 2, 20], "oldfive_multiplier": 5 / 3,
        "retention_stop_policy": "monitor_only", "strict_integer_all_six_winner_gates": LIMITS,
        "DEV_every_local_updates": 250, "renewed_grace_or_failure_reset": False,
        "initial_source_counter_is_genuinely_zero_at_qualified_BEST": True,
        "new_outputs_checkpoints_cache_scratch_root": str(STORAGE)}


def validate_replay_pool(rows):
    # The pinned old-five carrier has unequal pool sizes. Its sampler chooses
    # languages uniformly; only the Portuguese pool contains exactly 319 rows.
    expected = dict(en=312, es=316, it=328, fr=317, de=322, pt=319)
    actual = {language: sum(row["language"] == language for row in rows)
              for language in expected}
    require(len(rows) == 1914 and actual == expected,
            "Exact unchanged original six-language replay pools required")
    for row in rows:
        require(Path(row["cache_path"]).is_file() and row["encoder_frames"] > 0
                and row["target_tokens"] > 0,
                "Existing immutable replay cache required; never build a C: cache")
    return actual


def preflight(args):
    context = source_context(args.decoder_source_receipt, args.run_id)
    require(args.decoder_arm == context["receipt"]["decoder_arm"] and args.decoder_updates == UPDATES
            and args.shared_lr == 1e-5 and args.new_rows_lr == 1e-4 and args.batch_size == 8
            and args.accumulation == 2 and args.native_max_joint_bytes == 1700000000,
            "Only declared paired decoder recipe admitted")
    output = permitted(args.run_directory)
    require(output.is_relative_to(STORAGE.resolve()) and output.name == args.run_id,
            "All new run outputs/checkpoints must be on designated E: storage")
    storage = json.loads((STORAGE / "storage_config.json").read_text(encoding="utf-8"))
    require(storage["storage_root"] == str(STORAGE) and storage["allow_fallback_outside_root"] is False,
            "Exact no-fallback E: storage policy required")
    if args.decoder_arm == "bpe512":
        prep = read(context["BPE_label_preparation"])
        require(prep.get("passed") is True and prep["rows"] == context["TR_pool_clips"]
                and prep["maximum_full_joint_bytes"] <= 1700000000,
                "Complete actual same-source TRAIN BPE target preparation required")
        context["BPE_label_directory"] = str(Path(checked(prep["target_catalog"])).parent)
    pool = read(context["source_training_pool"])
    context["source_replay_counts"] = validate_replay_pool(pool["replay"])
    context["declaration"] = decoder_declaration(context, args.decoder_arm)
    return context


def audit_parent_state(model, state, context, torch_module):
    from encoder_lora import encoder_lora_metadata
    from tsc_bpe_acquisition_transition_v1 import audit_source_optimizer
    t = torch_module
    step = context["source_step"]
    require(state.get("schema") == 2 and state.get("run_id") == context["source_run_id"]
            and state.get("step") == state.get("selected_step") == step
            and state.get("retention_failures") == 0
            and state.get("config_identity") == context["parent_identity"]
            and state.get("lora_metadata") == encoder_lora_metadata(model),
            "Actual coherent all207 qualified source state/counter/layout required")
    require(canonical(state["history"]) == context["source_history_prefix_sha256"],
            "Actual source checkpoint history differs from its immutable saved prefix")
    expected = {n: p for n, p in model.named_parameters() if p.requires_grad}
    require(len(expected) == len(state["weights"]) == 207 and set(expected) == set(state["weights"]),
            "All207 source tensors required before decoder freeze or resize")
    for name, value in state["weights"].items():
        require(value.shape == expected[name].shape and value.dtype == t.float32
                and bool(t.isfinite(value).all()), "Actual source tensor geometry/finiteness differs: " + name)
    local = step - 40750
    fraction = 0 if local <= 1 else (local - 1) / (25198 - 1)
    encoder_lr = (1e-4 if local <= 1 else 3e-5 if local == 25198
                  else 3e-5 + (1e-4 - 3e-5) * .5 * (1 + math.cos(math.pi * fraction)))
    require(state["history"][-1]["lora_lr"] == encoder_lr, "Actual source applied encoder schedule rate differs")
    recipe = {**context["parent_recipe"], "lora_lr": encoder_lr}
    audit = audit_source_optimizer(state, {"source_recipe": recipe, "source_selected_step": step}, expected,
                                   torch_module=t)
    return {"passed": True, "all207_source_audited_before_resize_freeze_or_live_LR_change": True,
            "source_step": step, "source_applied_encoder_lr": encoder_lr, "source_Adam": audit}


def proof_path(args):
    return Path(args.run_directory) / PROOF_NAME


def compare_source_development(model, processor, args, context, state, evaluate_fn, torch_module):
    from tsc_bpe_acquisition_transition_v1 import compare_development
    t = torch_module
    require(not proof_path(args).exists(), "Immutable initial350 proof must never be overwritten")
    rng = (random.getstate(), t.get_rng_state().clone(), [v.clone() for v in t.cuda.get_rng_state_all()])
    modes = [(module, module.training) for module in model.modules()]
    try:
        if args.decoder_arm == "bpe512":
            require(model.config.vocab_size == 8711 and model.config.blank_token_id == 8192
                    and list(model.config.durations) == [0, 1, 2, 3, 4]
                    and all(not any(8199 <= v < 8711 for v in
                            (getattr(model.generation_config, field, None) or []))
                            for field in ("suppress_tokens", "begin_suppress_tokens")),
                    "Actual new BPE IDs must be unsuppressed before the source350 decode")
        scores = evaluate_fn(model, processor, args.run_id + "_initial_" + str(context["source_step"]))
        source_counts, source_pass = validate_scores(context["source_summary"])
        candidate_counts, candidate_pass = validate_scores(scores)
        require(source_pass and candidate_pass and source_counts == candidate_counts,
                "Actual initial350 must reproduce the genuine qualified source counts")
        folder = Path(args.run_directory).parent.parent / "results" / (args.run_id + "_initial_" + str(context["source_step"]))
        for language in LANGUAGES:
            original = [json.loads(line) for line in checked(context["source_DEV"][language]).read_text(encoding="utf-8").splitlines() if line]
            candidate = [json.loads(line) for line in (folder / ("validation_" + language + ".jsonl")).read_text(encoding="utf-8").splitlines() if line]
            compare_development(original, candidate, context["source_summary"]["validation_"+language]["asr"],
                                scores["validation_"+language]["asr"], language)
        proof = {"kind": "TSC_BPE_full_decoder_actual_initial350_source_equivalence_v1", "passed": True,
                 "source_step": context["source_step"], "source_receipt": context["receipt_record"],
                 "source_checkpoint": context["source_checkpoint"], "actual_GPU_decode": True,
                 "fixed_DEV_utterances": 350, "counts": candidate_counts, "all_six_passed": True,
                 "all_raw_transcripts_and_emitted_token_duration_paths_exact": True,
                 "decoder_arm": args.decoder_arm, "new512_unsuppressed": args.decoder_arm == "bpe512",
                 "before_first_decoder_update": True, "history_gate_or_failure_rewritten": False}
    finally:
        random.setstate(rng[0]); t.set_rng_state(rng[1]); t.cuda.set_rng_state_all(rng[2])
        for module, mode in modes:
            module.training = mode
    from atomic_io_retry_v1 import write_json
    write_json(proof_path(args), proof)
    return proof
