"""One declared BEST-state fork: head LR5e-5 ->1e-5 at global step1501.

No Torch import, state rewriting, data selection, or implicit fresh-Adam path.
The worker receives the immutable parent BEST and restores its entire state.
"""
from __future__ import annotations
import hashlib
import json
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RECEIPT_SHA = "4616f689b0ae1428fa88cf2f3d6912b0b39ed5daa73661dface4b568557bfb76"
PARENT_WORKER = "train_lora_batched_tsc_v3.py"
PARENT_WORKER_SHA = "16959e893f1d3440fce705e6ef3ccde40154d8958bf99037e4908fcf113f35bc"
WORKER = "train_lora_batched_tsc_headlr_v1.py"
OUTER = "run_TSC_headlr_fork_v1.py"
PROOF_NAME = "TSC_headlr_initial_source_equivalence.json"
LANGUAGES = ("tr", "en", "es", "it", "fr", "de")
IDENTITY_KEYS = ("source_id", "sentence_id", "language", "split", "reference",
                 "dataset_revision", "audio_sha256", "wav_sha256", "duration")
TOKEN_KEYS = ("token_id", "duration_id", "piece", "start_frame", "duration_frames")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False).encode()).hexdigest()


def scoped(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT), "Optimization evidence must stay in workspace")
    return path


def read(path):
    return json.loads(scoped(path).read_text(encoding="utf-8"))


def record(path):
    path = scoped(path)
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def checked_record(item):
    require(record(item["path"]) == item, "Immutable CLOSED source artifact changed: " + item["path"])
    return scoped(item["path"])


def validate_receipt_payload(receipt, target_run_id):
    require(receipt.get("schema") == 1 and receipt.get("kind") == "ROOT_owned_closed_TSC_best_for_declared_optimization_v1"
        and receipt.get("status") == "actual_CLOSED_qualified_best", "ROOT-owned CLOSED BEST receipt required")
    change = receipt.get("declared_next_change", {})
    require(change == {"new_run_id": target_run_id, "head_lr_from": 5e-5, "head_lr_to": 1e-5,
        "parent_state_boundary": 1500, "same_science_parent_resume_claimed": False,
        "other_scientific_changes_allowed": False}, "Only the explicitly approved head-LR state fork is admitted")
    require(receipt.get("selected_step") == receipt.get("best_sampler_boundary_step") == 1500
        and receipt.get("completed_steps") == receipt.get("latest_sampler_boundary_step") == 2000
        and receipt.get("best_and_latest_state_must_not_be_mixed") is True
        and receipt.get("retention_gate_passed_at_selected_step") is True,
        "BEST1500 and latest2000 frontiers must remain distinct")
    exits = receipt.get("owned_actual_exits", [])
    require(len(exits) == 2 and {item["role"] for item in exits} == {"trainer_controller", "supervisor"}
        and all(item.get("exit_code") == 0 and item.get("tool_chunk") for item in exits),
        "Both ROOT-owned parent processes must have actually exited")
    require(target_run_id != receipt["source_run_id"] and Path(target_run_id).name == target_run_id,
        "A new safe scientific run namespace is required")
    return receipt


@lru_cache(maxsize=2)
def _context(receipt_path, target_run_id, receipt_digest):
    require(receipt_digest == RECEIPT_SHA, "ROOT-approved branch receipt bytes differ")
    receipt = validate_receipt_payload(read(receipt_path), target_run_id)
    artifacts = receipt["artifacts"]
    config = read(checked_record(artifacts["run_config"]))
    selection = read(checked_record(artifacts["selection"]))
    identity = config["config_identity"]
    require(canonical(identity) == receipt["scientific_config_identity_sha256"], "Parent scientific identity differs")
    require(config["run_id"] == selection["run_id"] == receipt["source_run_id"]
        and selection.get("accepted") is True and selection.get("heldout_tests_evaluated") is False
        and selection.get("selected_step") == 1500 and selection.get("completed_steps") == 2000
        and scoped(selection["weights"]) == scoped(artifacts["best_checkpoint"]["path"]),
        "Actual saved qualifying BEST1500 selection required")
    require(identity["source_sha256"].get(PARENT_WORKER) == PARENT_WORKER_SHA,
        "Only the reviewed frozen TSC worker is eligible")
    for name, digest in identity["source_sha256"].items():
        require(sha(HERE / name) == digest, "Original scientific source changed: " + name)
    recipe = identity["training_arguments"]
    require(recipe["head_lr"] == 5e-5 and recipe["lora_lr"] == 3e-4 and recipe["rank"] == 8
        and recipe["alpha"] == 16 and recipe["batch_size"] == 8 and recipe["accumulation"] == 2
        and recipe["replay_kl"] == 10 and recipe["epochs"] == 1 and recipe["steps"] == 44958,
        "Actual unchanged TSC recipe/horizon required")
    return {"receipt": receipt, "receipt_record": record(receipt_path), "config": config,
            "selection": selection, "parent_identity": identity}


def preflight(args):
    """Stdlib-only admission API used by the one-file controller before dispatch."""
    path = scoped(args.optimization_source_receipt)
    ctx = _context(str(path), args.run_id, sha(path))
    require(args.resume_runtime_migration is None and args.resume_weights is None
        and bool(args.resume_optimization) != bool(args.resume_exact),
        "Choose the declared initial optimization state fork or SAME new-identity exact resume")
    recipe = ctx["parent_identity"]["training_arguments"]
    for key, value in recipe.items():
        require(getattr(args, key) == (1e-5 if key == "head_lr" else value),
                "Unapproved scientific option changed: " + key)
    parent_config = ctx["config"]
    for key in ("tsc_input_proof", "tsc_training_approval", "source_closure", "tr_manifest", "replay_manifest",
                "data_v3_proof", "tr_frame_metadata", "tr_frame_metadata_proof", "reuse_input_audit", "supplement"):
        before, after = parent_config.get(key), getattr(args, key)
        require((before is None and after is None) or (before is not None and after is not None
            and scoped(before) == scoped(after)), "Original data/source asset path changed: " + key)
    require(args.native_max_joint_bytes == parent_config["native_max_joint_bytes"], "Native bound must remain unchanged")
    if args.resume_optimization:
        require(scoped(args.resume_optimization) == scoped(ctx["receipt"]["artifacts"]["best_checkpoint"]["path"]),
                "Restore the whole parent BEST1500, never latest2000 or weights only")
        checked_record(ctx["receipt"]["artifacts"]["best_checkpoint"])
    return ctx


def closed_source_identity(args):
    # This original lineage record is retained verbatim; the new transition has
    # its own explicit config identity rather than falsely repeating fresh Adam.
    return preflight(args)["parent_identity"]["closed_source"]


def validate_parent_state(state, context):
    receipt = context["receipt"]
    require(state.get("schema") == 2 and state.get("run_id") == receipt["source_run_id"]
        and state.get("step") == state.get("selected_step") == 1500
        and state.get("config_identity") == context["parent_identity"], "Whole BEST1500 scientific state required")
    history = state.get("history", [])
    require(len(history) == 1500 and all(row["step"] == i for i, row in enumerate(history, 1)),
            "BEST history must end at1500, without later2000 rows")
    require(history[-1]["validation"]["retention_gate_passed"] is True
        and history[-1]["validation"]["turkish_wer"] == state["best_score"]
        == receipt["DEV1500_asr"]["tr"]["wer"], "Selected BEST gate/score differs")
    optimizer = state.get("optimizer", {})
    groups = optimizer.get("param_groups", [])
    require(len(groups) == 2 and len(optimizer.get("state", {})) == len(state.get("weights", {})) == 207
        and len(groups[0]["params"]) == 15 and len(groups[1]["params"]) == 192
        and groups[0]["lr"] == 5e-5 and groups[1]["lr"] == 3e-4,
        "All207 actual Adam states and original parameter groups required")
    require(all({"step", "exp_avg", "exp_avg_sq"}.issubset(item) for item in optimizer["state"].values()),
            "Missing Adam moments")
    sampler = state.get("sampler", {})
    require(sampler.get("updates") == 1500 and sampler.get("identity") == context["parent_identity"]["domain_sampling_identity"],
            "BEST1500 sampler identity/frontier required")
    for key in ("python_rng", "torch_rng", "cuda_rng", "lora_metadata", "best_score", "patience",
                "retention_failures", "next_eval_epoch", "train_wall_seconds", "audio_seconds",
                "language_audio_seconds", "evaluation_wall_seconds", "initial_gate"):
        require(key in state and state[key] is not None, "Missing preserved full-state field: " + key)
    require(state["retention_failures"] == 0, "BEST must retain its actual qualifying failure counter")
    return {"inherited_steps": 1500, "first_changed_lr_step": 1501,
            "history_prefix_sha256": canonical(history), "sampler_state_sha256": canonical(sampler),
            "adam_states": 207, "best_score": state["best_score"], "next_eval_epoch": state["next_eval_epoch"]}


def proof_path(args):
    return HERE / "runs" / args.run_id / PROOF_NAME


def transition_identity(args, context):
    proof = read(proof_path(args))
    require(proof.get("kind") == "TSC_headlr_before_update_source_equivalence_v1"
        and proof.get("passed") is True and proof.get("run_id") == args.run_id
        and proof.get("source_receipt") == context["receipt_record"]
        and proof.get("compared_fixed_DEV_utterances") == 300
        and proof.get("RNG_restored_after_added_evaluation") is True,
        "Actual300 source-equality gate must pass before Adam updates")
    return {"kind": "declared_TSC_head_LR_full_state_fork_v1", "source_receipt": context["receipt_record"],
        "source_run_id": context["receipt"]["source_run_id"], "source_checkpoint": context["receipt"]["artifacts"]["best_checkpoint"],
        "source_config_identity_sha256": context["receipt"]["scientific_config_identity_sha256"],
        "new_run_id": args.run_id, "head_lr_from": 5e-5, "head_lr_to": 1e-5,
        "inherited_completed_steps": 1500, "first_changed_lr_step": 1501,
        "initial_source_equivalence": record(proof_path(args)), "history_prefix_sha256": proof["state_boundary"]["history_prefix_sha256"],
        "source_is_BEST_not_latest": True, "fresh_optimizer_sampler_or_warmup": False,
        "inherited_history_uses_parent_recipe": True, "parent_future1501_to2000_exposure_is_revisited_from_BEST": True,
        "head_group_scope": "encoder projector, decoder, lexical/duration joint; old six row Adam-delta6 multiplier unchanged",
        "same_science_parent_resume_claimed": False}


def validate_transition_identity(parent, current, args, context):
    require(set(current) == set(parent) | {"optimization_transition"}, "Unapproved identity structure change")
    changed = {"source_sha256", "input_sha256", "native_loss", "training_arguments"}
    for key in set(parent) - changed:
        require(current[key] == parent[key], "Unapproved scientific identity changed: " + key)
    recipe = dict(parent["training_arguments"]); recipe["head_lr"] = 1e-5
    require(current["training_arguments"] == recipe, "Only head LR may change")
    extra = {WORKER, Path(__file__).name, OUTER}
    expected_sources = {**parent["source_sha256"], **{name: sha(HERE / name) for name in extra}}
    require(current["source_sha256"] == expected_sources, "Only declared worker/admission/dispatcher source additions allowed")
    expected_inputs = dict(parent["input_sha256"])
    expected_inputs.pop(str(scoped(context["config"]["native_backend_provenance"])))
    expected_inputs.update({str(scoped(args.native_backend_provenance)): sha(args.native_backend_provenance),
        str(scoped(args.optimization_source_receipt)): RECEIPT_SHA, str(proof_path(args).resolve()): sha(proof_path(args))})
    require(current["input_sha256"] == expected_inputs, "Data/closure inputs must remain byte-identical")
    native = dict(parent["native_loss"]); native["native_backend_provenance_sha256"] = sha(args.native_backend_provenance)
    require(current["native_loss"] == native, "Native objective/kernel/cache policy must remain unchanged")
    require(current["optimization_transition"] == transition_identity(args, context), "Declared transition proof differs")
    return True


def compare_development(source_rows, candidate_rows, source_asr, candidate_asr, language):
    from metrics import score_transcriptions
    require(len(source_rows) == len(candidate_rows) == 50, "Exactly50 fixed DEV recordings per language required")
    require(len({row["source_id"] for row in source_rows}) == 50, "Repeated DEV source")
    count = 0
    for before, after in zip(source_rows, candidate_rows):
        require(all(key in before and key in after and before[key] == after[key] for key in IDENTITY_KEYS)
            and before["language"] == language and before["split"] == "validation", "Fixed DEV identity/order/reference differs")
        require(before["hypothesis"] == after["hypothesis"], "Raw source/candidate transcript differs")
        paths = []
        for row in (before, after):
            confidence = row["confidence"]
            require(confidence["text"] == row["hypothesis"] and isinstance(confidence["tokens"], list), "Actual decode path missing")
            path = [tuple(token[key] for key in TOKEN_KEYS) for token in confidence["tokens"]]
            require(all(0 <= t[0] < 8199 and t[0] != 8192 and 0 <= t[1] <= 4 and t[4] == t[1] for t in path),
                    "Original8199 emitted path differs")
            paths.append(path)
        require(paths[0] == paths[1], "Token/duration path differs")
        count += len(paths[0])
    scores = [{k: v for k, v in score_transcriptions(rows, language=language).items() if k != "details"}
              for rows in (source_rows, candidate_rows)]
    require(scores[0] == scores[1] == source_asr == candidate_asr, "Independent DEV scores differ")
    return {"utterances": 50, "exact_raw_transcripts": 50, "exact_token_duration_paths": 50,
            "tokens_compared": count, "asr": scores[0]}


def initial_source_equivalence(model, processor, args, context, state, evaluate, torch):
    """Actual GPU DEV gate is executed only by ROOT's dispatched worker, before Adam."""
    import random
    require(not proof_path(args).exists(), "Never overwrite the initial equivalence proof")
    boundary = validate_parent_state(state, context)
    artifacts = context["receipt"]["artifacts"]
    source_summary = read(checked_record(artifacts["DEV1500_summary"]))
    source_metadata = read(checked_record(artifacts["DEV1500_metadata"]))
    rng = (random.getstate(), torch.get_rng_state().clone(), [value.clone() for value in torch.cuda.get_rng_state_all()])
    modes = [(module, module.training) for module in model.modules()]
    try:
        candidate = evaluate(model, processor, args.run_id + "_step_1500", splits=("validation",))
    finally:
        random.setstate(rng[0]); torch.set_rng_state(rng[1]); torch.cuda.set_rng_state_all(rng[2])
        for module, training in modes:
            module.train(training)
    restored = (random.getstate() == rng[0] and torch.equal(torch.get_rng_state(), rng[1])
        and len(torch.cuda.get_rng_state_all()) == len(rng[2])
        and all(torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), rng[2])))
    require(restored and all(module.training == training for module, training in modes),
            "Added300 evaluation must restore RNG and module training/encoder modes")
    folder = HERE / "results" / (args.run_id + "_step_1500")
    candidate_metadata = read(folder / "metadata.json")
    require(source_metadata["model_fingerprint"] == candidate_metadata["model_fingerprint"]
        and source_metadata["manifests"] == candidate_metadata["manifests"], "Actual source/model/tokenizer/generation fingerprints differ")
    comparisons = {}
    for language in LANGUAGES:
        source_path = checked_record(artifacts["DEV1500_" + language]); candidate_path = folder / ("validation_" + language + ".jsonl")
        source_rows = [json.loads(line) for line in source_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        candidate_rows = [json.loads(line) for line in candidate_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        comparisons[language] = compare_development(source_rows, candidate_rows,
            source_summary["validation_" + language]["asr"], candidate["validation_" + language]["asr"], language)
        del source_rows, candidate_rows
    baseline = read(HERE / "results/base_v3/summary.json")
    for language in LANGUAGES[1:]:
        base = baseline["validation_" + language]["asr"]["wer"]
        require(candidate["validation_" + language]["asr"]["wer"] <= base + max(.005, .1 * base), "Initial retention gate failed")
    proof = {"schema": 1, "kind": "TSC_headlr_before_update_source_equivalence_v1", "passed": True,
        "run_id": args.run_id, "source_receipt": context["receipt_record"], "state_boundary": boundary,
        "source_selected_step": 1500, "compared_fixed_DEV_utterances": 300,
        "all_raw_transcripts_and_emitted_paths_exact": True, "all_six_ASR_counts_exact": True,
        "all_five_retention_gates_passed": True, "RNG_restored_after_added_evaluation": restored,
        "module_training_modes_restored_after_added_evaluation": True,
        "optimizer_created_or_updated_before_check": False, "sampler_drawn_during_check": False,
        "candidate_summary": record(folder / "summary.json"), "candidate_metadata": record(folder / "metadata.json"),
        "comparisons": comparisons}
    with proof_path(args).open("x", encoding="utf-8") as stream:
        json.dump(proof, stream, ensure_ascii=False, indent=2)
    return proof
