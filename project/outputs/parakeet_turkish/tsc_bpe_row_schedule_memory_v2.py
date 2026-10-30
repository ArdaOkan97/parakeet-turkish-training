"""INACTIVE CLOSED acquired BPE512 state -> row-only warmup/cosine challenger.

Import and preflight are stdlib-only; checkpoint records are stat-checked there.
The future worker injects Torch and supplies the two actually deserialized,
immutable artifacts. This is a declared local warmup/cosine transition, not an
original207 exact resume. No phase1 source exists merely because this file does.
"""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import re
from types import SimpleNamespace

from bpe_newrows_scheduled_acquisition_v1 import schedule_identity

from tsc_bpe_acquisition_transition_v1 import (
    HERE, LANGUAGES, IDENTITY_KEYS, TOKEN_KEYS, NATIVE_MAX_BYTES,
    require, sha, canonical, scoped, read, record, checked_record,
)

PHASE = "HF_TSC_BPE512_new_rows_scheduled_acquisition_v1"
STATE_KIND = "HF_TSC_BPE512_new_rows_scheduled_acquisition_state_v1"
PHASE1 = "HF_TSC_BPE512_new_rows_only_acquisition_v1"
PHASE1_STATE_KIND = "HF_TSC_BPE512_new_rows_only_acquisition_state_v1"
RECEIPT_KIND = "ROOT_owned_closed_TSC_BPE_phase1_for_row_schedule_v1"
WORKER = "train_lora_batched_tsc_bpe_row_schedule_memory_v2.py"
OUTER = "run_TSC_bpe_row_schedule_memory_v2.py"
PROOF_NAME = "TSC_BPE_row_schedule_initial_source_equivalence.json"
PROOF_KIND = "TSC_BPE512_row_schedule_before_update_source_equivalence_v1"
SOURCE_PINS = {
    "bpe_newrows_scheduled_acquisition_v1.py": "f72485b7d2a0c242cf3a65c77a9be8e514c281d74c1a6ed924d3ea32eb8e0875",
    "tsc_bpe_acquisition_transition_v1.py": "0c888b2bc1cdbde8fc7c8062a7c6bd6a404d6aa09b62050b93942ec16705d038",
    "train_lora_batched_tsc_bpe_acquire_v1.py": "82faecd3c19762b182d9bff1bac9a61317bd99ceb6327db4356bba3b2587cd21",
    "run_TSC_bpe_acquire_v1.py": "78e5ab804789eb492edcc36e20faf4a6429605ac98469c4da2d39c2017e65624",
    "bpe_newrows_acquisition_v1.py": "bc26638f4874aaa5e92029e86c30bef0624a477514957d8f8d054d7e92088724",
    "bpe_newrows_continued_acquisition_v1.py": "ee25eb365a3bd578c48ca95aa66be4429146ef96319ac5614d12c8a6248049b7",
    "tsc_bpe512_training_inputs_v1.py": "98d79657035be7ed59a9c372ac5335755391456563167b891fd883fb89b7ccf7",
    "native_tdt_checkpointed_v3.py": "839194dddf83c3b08bf608d01cb540b2f160feee426a91373d14cfc426ab7494",
}


def scheduled_acquisition_declaration(peak_row_lr):
    require(peak_row_lr in (1e-3, 3e-4), "Only declared rowLR1e-3 or3e-4")
    return {"phase": PHASE, "peak_row_lr": peak_row_lr, "row_schedule": schedule_identity(peak_row_lr), "source_row_lr": 1e-4,
        "encoder_LoRA_active": False, "shared204_frozen": True, "weight_decay": 0.,
        "active_table_names": ["decoder.embedding.weight", "joint.head.weight", "joint.head.bias"],
        "effective_new_rows": [8199, 8711], "source_original207_retained_dormant": True,
        "phase1_three_table_Adam_moments_and_clock_retained": True,
        "phase1_acquired_rows_retained": True, "donor_reinitialization_used": False,
        "single_global3_clip_norm": 1., "six_character_delta_multiplier_applied": False,
        "native_max_joint_bytes": NATIVE_MAX_BYTES,
        "phase1_sampler_RNG_history_exposure_counters_preserved": True,
        "new_sampler_target_fingerprint_migration": False, "source_phase2_latest_used": False,
        "source_LR_audited_before_moment_copy": True, "original207_exact_resume_claimed": False,
        "external_phase_exact_resume_supported": False, "unfreeze_allowed": False}


def validate_receipt(receipt, target_run_id, peak_row_lr=None):
    if peak_row_lr is None:
        peak_row_lr = receipt.get("declared_row_schedule", {}).get("peak_row_lr")
    require(receipt.get("schema") == 1 and receipt.get("kind") == RECEIPT_KIND
        and receipt.get("status") == "actual_CLOSED_qualified_phase1"
        and receipt.get("target_run_id") == target_run_id
        and receipt.get("allow_scheduled_row_only_acquisition") is True
        and receipt.get("source_phase2_latest_used") is False
        and receipt.get("declared_row_schedule") == scheduled_acquisition_declaration(peak_row_lr),
        "Explicit ROOT CLOSED phase1 row-LR approval required")
    source_run = receipt.get("source_run_id")
    require(isinstance(source_run, str) and isinstance(target_run_id, str)
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{2,110}", source_run) is not None
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{2,110}", target_run_id) is not None
        and target_run_id != source_run, "Distinct safe phase1/new run IDs required")
    values = [receipt.get(key) for key in ("original_step", "acquisition_updates", "chosen_step", "completed_steps")]
    require(all(type(value) is int and value > 0 for value in values)
        and values[2] == values[0] + values[1] and values[2] <= values[3],
        "Positive acquired K and originalS+K frontier required")
    require(receipt.get("chosen_checkpoint_role") in ("latest", "best")
        and receipt.get("retention_gate_passed_at_chosen_step") is True,
        "Chosen acquired state must have its own qualifying fixedDEV")
    exits = receipt.get("owned_actual_exits", [])
    require(any(item.get("role") == "trainer_controller" for item in exits)
        and all(item.get("exit_code") == 0 and (item.get("chunk_id") or item.get("tool_chunk"))
                and item.get("session_id") and item.get("observed_at_utc") for item in exits),
        "Actual ROOT-owned CLOSED0 evidence required")
    return receipt


def original_pointer(receipt, key):
    return scoped(receipt.get("original_artifact_paths", {}).get(key, receipt["artifacts"][key]["path"]))


def validate_chosen_selection(receipt, config, selection, history, summary):
    """JSON-only paired source selection; latest need not beat the old champion."""
    source_run, step, original = receipt["source_run_id"], receipt["chosen_step"], receipt["original_step"]
    identity = config["config_identity"]
    require(config.get("run_id") == selection.get("run_id") == source_run
        and config.get("phase") == identity.get("phase") == PHASE1
        and identity.get("extended_tokenizer_sha256") == "10b747471a6613f818445f4c5697d04bb64ca45c48eb1bde7184d33e28e68607"
        and canonical(identity) == receipt["scientific_config_identity_sha256"],
        "Actual acquired8711 phase1 config identity required")
    require(selection.get("accepted") is True and selection.get("heldout_tests_evaluated") is False
        and selection.get("selection_split") == "FLEURS validation only"
        and selection.get("phase") == PHASE1 and selection.get("completed_steps") == receipt["completed_steps"]
        and selection.get("source_selected_step") == config.get("source_selected_step") == original,
        "Actual CLOSED phase1 selection/source clocks required")
    require(len(history) == receipt["completed_steps"]
        and all(row.get("step") == i for i, row in enumerate(history, 1))
        and all(row.get("acquisition_updates_completed") == row["step"] - original for row in history[original:]),
        "Complete inherited history and local acquisition clock required")
    role = receipt["chosen_checkpoint_role"]
    require((role == "latest" and step == selection["completed_steps"]
             and scoped(selection["latest_weights"]) == original_pointer(receipt, "chosen_checkpoint"))
        or (role == "best" and step == selection["selected_step"]
             and scoped(selection["weights"]) == original_pointer(receipt, "chosen_checkpoint")),
        "Chosen checkpoint role/step/original pointer differs")
    if role + "_checkpoint" in receipt["artifacts"]:
        chosen, paired = receipt["artifacts"]["chosen_checkpoint"], receipt["artifacts"][role + "_checkpoint"]
        require(chosen["sha256"] == paired["sha256"] and chosen["bytes"] == paired["bytes"],
                "Optional original latest/BEST byte record differs from archived chosen state")
    gate = history[step - 1].get("validation", {})
    require(gate.get("retention_gate_passed") is True
        and gate.get("turkish_wer") == summary["validation_tr"]["asr"]["wer"]
        and all(summary["validation_" + language]["asr"]["utterances"] == 50 for language in LANGUAGES),
        "ChosenS+K must bind its actual qualifying300 DEV, not local BEST scores")
    baseline = read(HERE / "results/base_v3/summary.json")
    for language in LANGUAGES[1:]:
        base = baseline["validation_" + language]["asr"]["wer"]
        require(summary["validation_" + language]["asr"]["wer"] <= base + max(.005, .1 * base),
                "Chosen acquired source failed original retention gate")
    return gate


@lru_cache(maxsize=2)
def _context(receipt_path, target_run_id, receipt_digest):
    require(sha(receipt_path) == receipt_digest, "ROOT phase2 receipt changed while read")
    receipt = validate_receipt(read(receipt_path), target_run_id)
    artifacts = receipt["artifacts"]
    for key in ("chosen_checkpoint", "original_source_checkpoint"):
        checked_record(artifacts[key], checkpoint=True)
    for key in ("latest_checkpoint", "best_checkpoint"):
        if key in artifacts:
            checked_record(artifacts[key], checkpoint=True)
    for key in ("original_source_receipt", "run_config", "training_history", "selection", "controller_status",
                "runtime_completion", "fresh_runtime", "owned_worker_identity", "source_initial300_proof",
                "DEV_summary", "DEV_metadata", *("DEV_" + language for language in LANGUAGES)):
        checked_record(artifacts[key])
    for name, digest in SOURCE_PINS.items():
        require(sha(HERE / name) == digest, "Reviewed phase1/phase2 dependency changed: " + name)
    config = read(artifacts["run_config"]["path"])
    identity = config["config_identity"]
    for name, digest in identity["source_sha256"].items():
        require(sha(HERE / name) == digest, "Original phase1 scientific source changed: " + name)
    require(all(identity["source_sha256"].get(name) == digest for name, digest in SOURCE_PINS.items()
                if name not in ("bpe_newrows_continued_acquisition_v1.py", "bpe_newrows_scheduled_acquisition_v1.py")), "Actual phase1 compiled source map differs")
    selection = read(artifacts["selection"]["path"])
    history = read(artifacts["training_history"]["path"])
    summary = read(artifacts["DEV_summary"]["path"])
    gate = validate_chosen_selection(receipt, config, selection, history, summary)
    controller = read(artifacts["controller_status"]["path"])
    completion = read(artifacts["runtime_completion"]["path"])
    ready = read(artifacts["fresh_runtime"]["path"])
    owned = read(artifacts["owned_worker_identity"]["path"])
    require(controller.get("kind") == "guarded_TSC_BPE512_new_rows_only_acquisition_controller_v1"
        and controller.get("status") == "train_window_ended" and controller.get("child_exit_code") == 0
        and controller.get("resource_review_required") is False and controller.get("run_id") == receipt["source_run_id"]
        and controller.get("controller_sha256") == SOURCE_PINS["run_TSC_bpe_acquire_v1.py"]
        and controller.get("smoke_updates") in (None, 0) and config.get("smoke_updates") in (None, 0)
        and controller.get("completed_steps") == receipt["completed_steps"],
        "Actual nonresource CLOSED main acquisition screen required; separate smoke is not a phase2 source")
    require(completion.get("kind") == "TSC_BPE_acquisition_native_completion_v1"
        and completion.get("status") == "complete" and completion.get("sources_unchanged") is True
        and completion.get("run_id") == receipt["source_run_id"]
        and completion.get("backend_ready_sha256") == artifacts["fresh_runtime"]["sha256"]
        and completion.get("actual_dispatch_source_sha256") == SOURCE_PINS["run_TSC_bpe_acquire_v1.py"]
        and completion.get("pid") == ready.get("pid") == owned.get("pid")
        and ready.get("kind") == "native_tdt_training_backend_v3"
        and ready.get("run_id") == owned.get("expected_run_id") == receipt["source_run_id"]
        and ready.get("trainer_sha256") == SOURCE_PINS["train_lora_batched_tsc_bpe_acquire_v1.py"]
        and ready.get("launcher_sha256") == SOURCE_PINS["run_TSC_bpe_acquire_v1.py"],
        "Actual compiled phase1 native completion/PID/ready required")
    proof = read(artifacts["source_initial300_proof"]["path"])
    require(proof.get("kind") == "TSC_BPE512_acquisition_before_update_source_equivalence_v1"
        and proof.get("run_id") == receipt["source_run_id"] and proof.get("passed") is True
        and proof.get("actual_GPU_decode") is True and proof.get("before_optimizer_creation") is True
        and proof.get("source_selected_step") == receipt["original_step"]
        and proof.get("compared_fixed_DEV_utterances") == 300
        and proof.get("all_raw_transcripts_and_emitted_paths_exact") is True
        and proof.get("all_six_ASR_counts_exact") is True and proof.get("all_five_retention_gates_passed") is True
        and proof.get("RNG_restored_after_added_evaluation") is True
        and proof.get("module_training_modes_restored_after_added_evaluation") is True
        and identity.get("initial_source_equivalence_sha256") == artifacts["source_initial300_proof"]["sha256"],
        "Actual original-source preservation gate must precede phase1 Adam")
    # Re-admit the original receipt using ITS actual phase1 target run, never the
    # new phase2 ID. An archived receipt may change path, but never its bytes.
    from tsc_bpe_acquisition_transition_v1 import preflight as acquisition_preflight
    old_receipt = config["source_receipt"]
    require(old_receipt["sha256"] == artifacts["original_source_receipt"]["sha256"]
        and old_receipt["bytes"] == artifacts["original_source_receipt"]["bytes"]
        and scoped(old_receipt["path"]) == original_pointer(receipt, "original_source_receipt"),
        "Original acquisition ROOT receipt bytes/pointer changed")
    values = dict(identity["training_arguments"])
    values.update(run_id=receipt["source_run_id"],
        acquisition_source_receipt=scoped(artifacts["original_source_receipt"]["path"]))
    original_context = acquisition_preflight(SimpleNamespace(**values))
    require(original_context["source_selected_step"] == receipt["original_step"]
        and identity["source_phase_identity"] == original_context["parent_identity"]
        and artifacts["original_source_checkpoint"] == original_context["receipt"]["artifacts"]["best_checkpoint"]
        and identity["acquisition_optimizer"]["parked_source_artifact"] == artifacts["original_source_checkpoint"]
        and canonical(history[:receipt["original_step"]]) == original_context["source_history_prefix_sha256"],
        "Original207/S artifact, lineage identity or inherited history changed")
    require(identity["geometry"] == {"student_vocab_size": 8711, "joint_width": 8716, "blank_id": 8192,
        "teacher_vocab_size": 8193, "new_BPE_rows": [8199, 8711], "protected_six_rows": [8193, 8199],
        "duration_rows": [8711, 8716]}, "Phase1 actual8711/teacher8193/blank/duration geometry differs")
    prefix = canonical(history[:receipt["chosen_step"]])
    return {"receipt": receipt, "receipt_record": record(receipt_path), "config": config,
        "selection": selection, "parent_identity": identity, "source_step": receipt["chosen_step"],
        "source_selected_step": receipt["chosen_step"], "original_step": receipt["original_step"],
        "acquisition_updates": receipt["acquisition_updates"], "source_recipe": dict(original_context["source_recipe"]),
        "history_prefixSHA": prefix, "source_history_prefix_sha256": prefix,
        "chosen_gate": gate, "original_context": original_context,
        "phase1_record": artifacts["chosen_checkpoint"],
        "metadata_preparation": original_context["metadata_preparation"]}


def preflight(args):
    require(getattr(args, "resume_row_schedule", None) is not None
        and all(getattr(args, key, None) is None for key in ("resume_exact", "resume_weights", "resume_runtime_migration",
                "resume_optimization", "resume_acquisition")), "Initial declared phase2 only; no exact or legacy resume")
    path = scoped(args.row_schedule_source_receipt)
    context = _context(str(path), args.run_id, sha(path))
    require(scoped(args.resume_row_schedule) == scoped(context["receipt"]["artifacts"]["chosen_checkpoint"]["path"]),
            "Use paired acquired chosen state, not localBEST or original207 substitution")
    for key, value in context["source_recipe"].items():
        require(getattr(args, key) == value, "Dormant original recipe/common science changed: " + key)
    require(args.peak_row_lr == context["receipt"]["declared_row_schedule"]["peak_row_lr"]
        and args.native_max_joint_bytes == NATIVE_MAX_BYTES
        and type(args.acquisition_max_updates) is int and 0 < args.acquisition_max_updates <= 1000
        and args.smoke_updates in (None, 1) and (args.smoke_updates is None or type(args.smoke_updates) is int),
        "Bounded declared phase2 screen or separate1-update smoke required")
    for key in ("tsc_input_proof", "tsc_training_approval", "source_closure", "tr_manifest", "replay_manifest",
                "data_v3_proof", "tr_frame_metadata", "tr_frame_metadata_proof", "reuse_input_audit", "supplement",
                "candidate_dir", "candidate_tr_metadata", "candidate_tr_metadata_proof"):
        before, after = context["config"].get(key), getattr(args, key, None)
        require((before is None and after is None) or (before is not None and after is not None and scoped(before) == scoped(after)),
                "Original phase1 data/candidate asset changed: " + key)
    return context


def validate_phase1_state(state, context):
    """Tensor-free runtime structure check; utility later audits actual tensors."""
    receipt, identity, step = context["receipt"], context["parent_identity"], context["source_step"]
    runtime = state.get("runtime_state", {})
    require(state.get("schema") == 1 and state.get("kind") == PHASE1_STATE_KIND
        and state.get("run_id") == runtime.get("run_id") == receipt["source_run_id"]
        and state.get("step") == runtime.get("step") == step
        and state.get("source_selected_step") == runtime.get("source_selected_step") == context["original_step"]
        and state.get("acquisition_optimizer_updates") == context["acquisition_updates"]
        and state.get("config_identity") == runtime.get("config_identity")
        and canonical(state.get("config_identity")) == canonical(identity),
        "Acquired207 weights/three-table moments/runtime must come from ONE chosen frontier")
    history = runtime.get("history", [])
    require(len(history) == step and canonical(history) == context["source_history_prefix_sha256"]
        and history[-1].get("validation") == context["chosen_gate"]
        and history[-1].get("acquisition_updates_completed") == context["acquisition_updates"],
        "Chosen acquired history/gate/local clock differs")
    require(runtime.get("selected_step") == state.get("selected_step")
        and type(runtime.get("selected_step")) is int and 0 < runtime["selected_step"] <= step,
        "LocalBEST metadata must remain distinct from chosen latest")
    require(state.get("lora_metadata") == identity["lora_layout"]
        and runtime["sampler"]["identity"] == identity["domain_sampling_identity"]
        and runtime["sampler"]["updates"] == step and runtime["retention_failures"] == 0,
        "Already-migrated candidate sampler/layout/qualifying failure frontier differs")
    for key in ("python_rng", "torch_rng", "cuda_rng", "patience", "next_eval_epoch", "train_wall_seconds",
                "audio_seconds", "language_audio_seconds", "evaluation_wall_seconds", "initial_gate", "best_score"):
        require(key in runtime and runtime[key] is not None, "Missing inherited acquired runtime: " + key)
    return {"original_step": context["original_step"], "selected_acquisition_updates": context["acquisition_updates"],
        "chosen_step": step, "chosen_checkpoint_role": receipt["chosen_checkpoint_role"],
        "phase1_localBEST_step": runtime["selected_step"], "phase1_localBEST_score": runtime["best_score"],
        "continued_row_initial_best_from_chosen_qualified_DEV": context["chosen_gate"]["turkish_wer"],
        "original_champion_artifact_unchanged": True, "chosen_own_state_not_mixed_with_other_frontier": True,
        "history_prefix_sha256": canonical(history), "source_sampler_sha256": canonical(runtime["sampler"]),
        "new_target_fingerprint_migration": False, "first_phase2_update": step + 1,
        "row_Adam_clock": context["acquisition_updates"], "original_dormant_LoRA_Adam_clock": context["original_step"],
        "encoder_LoRA_active": False}


def load_validate_phase1(model, processor, original207, phase1_state, context, torch=None, *, torch_module=None):
    torch = torch_module if torch_module is not None else torch
    require(torch is not None, "Future worker must explicitly inject Torch")
    boundary = validate_phase1_state(phase1_state, context)
    for key in ("original_source_checkpoint", "chosen_checkpoint"):
        require(record(context["receipt"]["artifacts"][key]["path"]) == context["receipt"]["artifacts"][key],
                "Actual CLOSED state bytes changed before load: " + key)
    from tsc_bpe_acquisition_transition_v1 import load_validate_expand
    expanded = load_validate_expand(model, processor, original207, context["original_context"], torch_module=torch)
    # An actual acquired state must demonstrate learning, not merely positive
    # scalar Adam clocks. These small slices are checked only at this boundary.
    tables = ("decoder.embedding.weight", "joint.head.weight", "joint.head.bias")
    initialized = dict(model.named_parameters())
    optimizer = phase1_state["acquisition_optimizer"]
    ids = optimizer["param_groups"][0]["params"]
    require(ids == list(range(3)), "Phase1 explicit three-table Adam order differs")
    learned = {}
    for identifier, name in zip(ids, tables):
        acquired = phase1_state["weights"][name][8199:8711]
        before = initialized[name].detach()[8199:8711].cpu()
        moments = optimizer["state"][identifier]
        changed = int(torch.count_nonzero(acquired - before))
        first = int(torch.count_nonzero(moments["exp_avg"][8199:8711]))
        second = int(torch.count_nonzero(moments["exp_avg_sq"][8199:8711]))
        require(changed > 0 and first > 0 and second > 0, "Actual acquired learning absent in table: " + name)
        learned[name] = {"changed_new_row_scalar_elements": changed,
            "nonzero_first_moment_elements": first, "nonzero_second_moment_elements": second}
    require(sha(HERE / "bpe_newrows_continued_acquisition_v1.py") == SOURCE_PINS["bpe_newrows_continued_acquisition_v1.py"],
            "Reviewed continued-row utility changed")
    require(sha(HERE / "bpe_newrows_scheduled_acquisition_v1.py") == SOURCE_PINS["bpe_newrows_scheduled_acquisition_v1.py"], "Reviewed scheduled-row utility changed")
    from bpe_newrows_scheduled_acquisition_v1 import BPEScheduledAcquisition
    acquisition = BPEScheduledAcquisition(model, phase1_state, original207,
        original_source_record=context["receipt"]["artifacts"]["original_source_checkpoint"],
        phase1_record=context["receipt"]["artifacts"]["chosen_checkpoint"],
        peak_row_lr=context["receipt"]["declared_row_schedule"]["peak_row_lr"], torch_module=torch)
    require(acquisition.cumulative_step == context["source_step"] and acquisition.optimizer_updates_completed == 0,
            "Construction must not advance acquired Adam/runtime")
    return {"acquisition": acquisition, "vocabulary": expanded["vocabulary"],
        "boundary": {**boundary, "original207_audit": expanded["boundary"],
            "actual_acquired_table_learning": learned, "scheduled_row_identity": acquisition.phase_identity()},
        "expansion_evidence": expanded["expansion_evidence"], "inherited_runtime": acquisition.inherited_runtime}


def proof_path(args):
    return HERE / "runs" / args.run_id / PROOF_NAME


def compare_development(source_rows, candidate_rows, source_asr, candidate_asr, language):
    from metrics import score_transcriptions
    require(len(source_rows) == len(candidate_rows) == 50
        and len({row["source_id"] for row in source_rows}) == 50, "Complete fixed50 uniqueDEV required")
    tokens_compared = 0
    for before, after in zip(source_rows, candidate_rows):
        require(all(key in before and key in after and before[key] == after[key] for key in IDENTITY_KEYS)
            and before["language"] == language and before["split"] == "validation", "FixedDEV identity/order/audio/reference changed")
        require(before["hypothesis"] == after["hypothesis"], "Raw chosen8711 transcript changed before phase2 update")
        paths = []
        for row in (before, after):
            confidence = row["confidence"]
            require(confidence["text"] == row["hypothesis"], "Confidence decode text differs")
            tokens = [tuple(item[key] for key in TOKEN_KEYS) for item in confidence["tokens"]]
            require(all(0 <= item[0] < 8711 and item[0] != 8192 and 0 <= item[1] <= 4 and item[4] == item[1]
                        for item in tokens), "Chosen/candidate8711 lexical/duration paths changed")
            paths.append(tokens)
        require(paths[0] == paths[1]
            and all(before["confidence"][key] == after["confidence"][key] for key in ("blank_steps", "total_steps")),
            "Actual chosen8711 token/duration/blank paths differ")
        tokens_compared += len(paths[0])
    scores = [{key: value for key, value in score_transcriptions(rows, language=language).items() if key != "details"}
              for rows in (source_rows, candidate_rows)]
    require(scores[0] == scores[1] == source_asr == candidate_asr, "Independent same8711 six-ASR counts differ")
    return {"utterances": 50, "exact_raw_transcripts": 50, "exact_token_duration_paths": 50,
        "tokens_compared": tokens_compared, "asr": scores[0]}


def initial_source_equivalence(model, processor, args, context, state, evaluate, torch):
    """Actual future300 same8711 decode, no phase2 update; RNG and modes restored."""
    import random
    require(not proof_path(args).exists(), "Never overwrite actual initial300 proof")
    boundary = validate_phase1_state(state, context)
    runtime = state["runtime_state"]
    artifacts = context["receipt"]["artifacts"]
    summary = read(checked_record(artifacts["DEV_summary"]))
    metadata = read(checked_record(artifacts["DEV_metadata"]))
    for field in ("suppress_tokens", "begin_suppress_tokens"):
        require(not any(8199 <= value < 8711 for value in getattr(model.generation_config, field, None) or []),
                "New512 must remain unsuppressed during same8711 DEV")
    rng = (random.getstate(), torch.get_rng_state().clone(), [value.clone() for value in torch.cuda.get_rng_state_all()])
    require(rng[0] == runtime["python_rng"] and torch.equal(rng[1], runtime["torch_rng"])
        and len(rng[2]) == len(runtime["cuda_rng"])
        and all(torch.equal(a, b) for a, b in zip(rng[2], runtime["cuda_rng"])),
        "Restore actual chosen phase1 Python/CPU/allCUDA RNG before added evaluation")
    sampler_sha = canonical(runtime["sampler"])
    modes = [(module, module.training) for module in model.modules()]
    tag = args.run_id + "_step_" + str(context["source_step"])
    try:
        candidate = evaluate(model, processor, tag, splits=("validation",))
    finally:
        random.setstate(rng[0]); torch.set_rng_state(rng[1]); torch.cuda.set_rng_state_all(rng[2])
        for module, training in modes:
            module.train(training)
    cuda = torch.cuda.get_rng_state_all()
    restored = random.getstate() == rng[0] and torch.equal(torch.get_rng_state(), rng[1])
    restored = restored and len(cuda) == len(rng[2]) and all(torch.equal(a, b) for a, b in zip(cuda, rng[2]))
    require(restored and all(module.training == training for module, training in modes)
        and canonical(runtime["sampler"]) == sampler_sha, "AddedDEV changed RNG/modes or source sampler")
    folder = HERE / "results" / tag
    candidate_metadata = read(folder / "metadata.json")
    require(metadata["manifests"] == candidate_metadata["manifests"], "Fixed300 DEV manifests differ")
    comparisons = {}
    for language in LANGUAGES:
        source_path = checked_record(artifacts["DEV_" + language])
        with source_path.open(encoding="utf-8") as stream:
            source_rows = [json.loads(line) for line in stream if line.strip()]
        with (folder / ("validation_" + language + ".jsonl")).open(encoding="utf-8") as stream:
            candidate_rows = [json.loads(line) for line in stream if line.strip()]
        comparisons[language] = compare_development(source_rows, candidate_rows,
            summary["validation_" + language]["asr"], candidate["validation_" + language]["asr"], language)
    baseline = read(HERE / "results/base_v3/summary.json")
    for language in LANGUAGES[1:]:
        base = baseline["validation_" + language]["asr"]["wer"]
        require(candidate["validation_" + language]["asr"]["wer"] <= base + max(.005, .1 * base),
                "Actual phase2 initial retention gate failed")
    proof = {"schema": 1, "kind": PROOF_KIND, "passed": True, "run_id": args.run_id,
        "source_receipt": context["receipt_record"], "declared_row_schedule": context["receipt"]["declared_row_schedule"], "source_selected_step": context["source_step"],
        "original_source_step": context["original_step"], "selected_acquisition_updates": context["acquisition_updates"],
        "chosen_checkpoint_role": context["receipt"]["chosen_checkpoint_role"], "state_boundary": boundary,
        "actual_GPU_decode": True, "before_row_schedule_updates": True, "compared_fixed_DEV_utterances": 300,
        "unsuppressed_new512_ids": True, "all_raw_transcripts_and_emitted_paths_exact": True,
        "all_six_ASR_counts_exact": True, "all_five_retention_gates_passed": True,
        "RNG_restored_after_added_evaluation": restored, "module_training_modes_restored_after_added_evaluation": True,
        "sampler_drawn_during_check": False, "second_target_fingerprint_migration": False,
        "source_and_candidate_fingerprints_equal_required": False, "confidence_or_loss_equality_required": False,
        "candidate_summary": record(folder / "summary.json"), "candidate_metadata": record(folder / "metadata.json"),
        "comparisons": comparisons}
    with proof_path(args).open("x", encoding="utf-8") as stream:
        json.dump(proof, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write("\n")
    return proof
