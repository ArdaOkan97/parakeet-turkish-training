"""INACTIVE same-ID own-phase2 exact continuation, never a fresh phase fork.

Original207 plus phase1 artifacts reconstruct the strict195-tensor utility;
its own archived phase2 latest then restores full207 weights, two Adam states,
S/K/N clocks and the returned opaque runtime. Original scientific config,
native-ready, source300 and qualifying BEST stay immutable. A separately
ROOT-approved TOTAL phase2 bound extends the stopping screen only. This file
alone neither admits an actual CLOSED source nor launches a GPU job.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch

from batched_lora_tr_metadata_v1 import batch_objective, collate_training_batch
from benchmark import LANGUAGES, evaluate, read_jsonl, write_json
from encoder_lora import install_encoder_lora, encoder_lora_metadata
from extend_vocab import extend_processor_and_model
from metrics import normalize_text
from runtime import ROOT, DATA, OUTPUT, load_model
from train import make_cache
from train_lora_batched_tsc_headlr_v1 import (BASE_REVISION, BucketCycler,
    annotate_cache_lengths, development_gate, ids_sha, next_evaluation_epoch,
    parser as source_parser, source_isolation)
from tsc_training_inputs_v2 import (audit_tsc_training_inputs,
    confirm_tr_metadata_acceptance, confirm_replay_after_cache)
from tsc_domain_sampler_v3 import TSCDomainBalancedBucketSampler
from tsc_bpe512_training_inputs_v1 import (annotate_tr_frame_metadata,
    validate_training_item, METADATA_PATH, REPORT_PATH)
from tsc_bpe_lora_transition_v2 import (preflight as adaptation_preflight,
    load_validate_phase1, initial_source_equivalence)
from bpe_newrows_lora_adaptation_v1 import BPENewRowsLoRAAdaptation
from bpe_newrows_acquisition_v1 import artifact_record
from train_lora import checkpoint_encoder_blocks
import train_lora_batched_tsc_bpe_lora_native4_v2 as initial_worker
from tsc_bpe_lora_exact_transition_v1 import (preflight as exact_preflight,
    restore_runtime, validate_runtime_proof, canonical)

PHASE = "HF_TSC_BPE512_new_rows_plus_encoder_LoRA_v1"
PROOF_NAME = "TSC_BPE_LoRA_initial_source_equivalence.json"
GEOMETRY = {"student_vocab_size": 8711, "joint_width": 8716, "blank_id": 8192,
    "teacher_vocab_size": 8193, "new_BPE_rows": [8199, 8711],
    "protected_six_rows": [8193, 8199], "duration_rows": [8711, 8716]}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def proof_path(args):
    return OUTPUT / "runs" / args.run_id / PROOF_NAME


def native_loss_identity(args):
    """Immutable original science token plus a separately fresh actual executor."""
    require(file_sha(OUTPUT / "train_lora_batched_tsc_bpe_lora_native4_v2.py") ==
        "15825a12c35b1262b28509c921a5a763ecf17696ee02c58446d5a52a33218be3",
        "Frozen original science validator changed")
    identity = initial_worker.native_loss_identity(args)
    validate_runtime_proof(args, worker_path=__file__)
    return identity


def native_loss_function(args, tr, replay):
    from functools import partial
    from native_tdt_checkpointed_v4 import chunked_tdt_loss
    native_loss_identity(args)
    maximum = {"bytes": 0}
    count = 0
    for rows in (tr, replay):
        for row in rows:
            t, u = row["encoder_frames"], row["target_tokens"]
            require(type(t) is int and type(u) is int and t > 0 and 0 < u <= 180,
                    "Positive accepted T/U required")
            size = t * (u + 1) * 8716 * 4
            require(size <= args.native_max_joint_bytes,
                    "Exact widened native joint bound exceeded; no crop/drop: " + row["source_id"])
            count += 1
            if size > maximum["bytes"]:
                maximum = {"bytes": size, "T": t, "U": u, "source_id": row["source_id"], "language": row["language"]}
    require(count == 253798 + 1595, "All accepted TR and unchanged replay must be bounded")
    return partial(chunked_tdt_loss, sigma=0.0, omega=0.0, recompute=True,
        max_joint_bytes=args.native_max_joint_bytes), {"joint_width": 8716, "clips_checked": count,
        "maximum": maximum, "declared_maximum_bytes": args.native_max_joint_bytes,
        "arithmetic_only": True, "actual_GPU_fit_proven": False}


def save_training_state(path, model, adaptation, sampler, *, run_id, step, config_identity,
                        history, best_score, selected_step, patience, retention_failures,
                        next_eval_epoch, train_wall_seconds, audio_seconds, language_audio_seconds,
                        evaluation_wall_seconds, initial_gate, source_selected_step, stop_reason=None):
    """NEW phase serializer: all207 weights and two independent active Adam states.

    No legacy requires_grad filtering and no source optimizer/history relabelling.
    Own-phase exact admission is separate; the complete two-Adam boundary is retained.
    """
    require(step == adaptation.cumulative_step and sampler.updates == step and len(history) == step,
            "Durable cumulative/sampler/history frontier differs")
    runtime_state = {"run_id": run_id, "step": step, "source_phase1_step": source_selected_step,
        "source_selected_step": adaptation.source_step,
        "original_source_selected_step": adaptation.source_step,
        "selected_acquisition_updates": adaptation.acquisition_updates,
        "adaptation_updates_completed": adaptation.phase2_updates,
        "config_identity": config_identity, "sampler": sampler.state_dict(),
        "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(), "history": history,
        "best_score": best_score, "selected_step": selected_step, "patience": patience,
        "retention_failures": retention_failures, "next_eval_epoch": next_eval_epoch,
        "train_wall_seconds": train_wall_seconds, "audio_seconds": audio_seconds,
        "language_audio_seconds": language_audio_seconds, "evaluation_wall_seconds": evaluation_wall_seconds,
        "initial_gate": initial_gate, "stop_reason": stop_reason}
    state = adaptation.state_dict(runtime_state=runtime_state)
    state.update(run_id=run_id, step=step, source_phase1_step=source_selected_step,
                 original_source_selected_step=adaptation.source_step,
                 selected_acquisition_updates=adaptation.acquisition_updates,
                 selected_step=selected_step, config_identity=config_identity)
    temporary = Path(str(path) + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def _table_bytes_sha(tensor):
    """Smoke-only actual tensor bytes; no repeated per-update full-weight hashing."""
    return hashlib.sha256(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def _protected_snapshot(adaptation):
    protected = {}
    for name in adaptation.names:
        if name in adaptation.adapter_names:
            continue
        tensor = adaptation.parameters[name]
        if name in ("decoder.embedding.weight", "joint.head.weight", "joint.head.bias"):
            protected[name] = {"old_blank_six": _table_bytes_sha(tensor[:8199]),
                               "duration": _table_bytes_sha(tensor[8711:])}
        else:
            protected[name] = _table_bytes_sha(tensor)
    return protected


def _verify_smoke_phase_reload(adaptation, saved, runtime):
    """One smoke only: actual207 weights plus independent3/192 Adam states."""
    require(set(saved["weights"]) == set(adaptation.names), "Smoke phase weight names differ")
    for name in adaptation.names:
        require(_table_bytes_sha(adaptation.parameters[name]) == _table_bytes_sha(saved["weights"][name]),
                "Smoke full207 reloaded weight bytes differ: " + name)
    for key, optimizer in (("row_optimizer", adaptation.row_optimizer),
                           ("lora_optimizer", adaptation.lora_optimizer)):
        actual, expected = optimizer.state_dict(), saved[key]
        require(actual["param_groups"] == expected["param_groups"] and set(actual["state"]) == set(expected["state"]),
                "Smoke independent Adam groups or state IDs differ")
        for identifier, state in expected["state"].items():
            require(set(actual["state"][identifier]) == set(state), "Smoke Adam state fields differ")
            for field, tensor in state.items():
                reloaded = actual["state"][identifier][field]
                require(reloaded.shape == tensor.shape and reloaded.dtype == tensor.dtype
                        and _table_bytes_sha(reloaded) == _table_bytes_sha(tensor),
                        "Smoke independent Adam moment/clock bytes differ: " + field)
    cuda = torch.cuda.get_rng_state_all()
    require(random.getstate() == runtime["python_rng"]
        and _table_bytes_sha(torch.get_rng_state()) == _table_bytes_sha(runtime["torch_rng"])
        and len(cuda) == len(runtime["cuda_rng"])
        and all(_table_bytes_sha(actual_rng) == _table_bytes_sha(saved_rng)
                for actual_rng, saved_rng in zip(cuda, runtime["cuda_rng"])),
        "Internal smoke phase reload changed actual Python/CPU/allCUDA RNG frontier")


def run(args):
    process_start = time.perf_counter()
    exact_context = exact_preflight(args)
    context = exact_context["original_context"]
    # This already-validated JSON copy is not retained beside the actual state history.
    exact_context.pop("archived_history", None)
    public_data_audit = audit_tsc_training_inputs(args)
    public_data_audit["identity"].update(target_replay_rng_independent_of_turkish=True,
        target_replay_seed=args.replay_seed, replay_sampling_stream_bitwise_identity_claimed=False,
        replay_sampling_nuisance="Original replay pool and independent RNG; acquired phase1 target identities, pending orders and cursors carry exactly. No second target fingerprint or sampler transition.")
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    run_dir = OUTPUT / "runs" / args.run_id
    best_path, latest_path = (ROOT / "work" / (args.run_id + suffix) for suffix in ("_best.pt", "_latest.pt"))
    require(run_dir.is_dir() and best_path.is_file() and latest_path.is_file(),
            "Exact continuation requires the existing same-run namespace, latest and qualifying BEST")
    archived = exact_context["exact_receipt"]["artifacts"]
    require(best_path.resolve() == Path(exact_context["archived_selection"]["weights"]).resolve()
        and latest_path.resolve() == Path(exact_context["archived_selection"]["latest_weights"]).resolve()
        and file_sha(best_path) == archived["best_checkpoint"]["sha256"]
        and best_path.stat().st_size == archived["best_checkpoint"]["bytes"]
        and file_sha(latest_path) == archived["latest_checkpoint"]["sha256"]
        and file_sha(run_dir / "run_config.json") == archived["run_config"]["sha256"],
        "Existing SAME-run BEST/latest/config must match the actually closed source")
    model, processor = load_model()
    baseline_path = OUTPUT / "results/base_v3"
    baseline = json.loads((baseline_path / "summary.json").read_text(encoding="utf-8"))
    heldout_paths = [DATA / f"{split}_{lang}.jsonl" for lang in LANGUAGES for split in ("validation", "test")]
    heldout = [row for path in heldout_paths for row in read_jsonl(path)]
    for lang in LANGUAGES:
        expected = [row["source_id"] for row in read_jsonl(DATA / f"validation_{lang}.jsonl")]
        actual = [row["source_id"] for row in read_jsonl(baseline_path / f"validation_{lang}.jsonl")]
        require(expected == actual and baseline[f"validation_{lang}"]["asr"]["utterances"] == len(expected),
                "Complete ordered original fixed baseline required")
    tr, replay = read_jsonl(args.tr_manifest), read_jsonl(args.replay_manifest)
    source_isolation(tr, replay, heldout, public_data_audit)
    del heldout
    teacher = {name: copy.deepcopy(getattr(model, name)).eval().requires_grad_(False)
               for name in ("encoder_projector", "decoder", "joint")}
    old_size = model.config.vocab_size
    require(old_size == 8193, "Original teacher vocabulary must be8193")
    original_vocabulary = extend_processor_and_model(processor, model, [row["text"] for row in tr])
    if args.phonetic_sh_donors:
        for new, donor in (("ş", "ș"), ("Ş", "Ș")):
            new_id, donor_id = processor.tokenizer.convert_tokens_to_ids(new), processor.tokenizer.convert_tokens_to_ids(donor)
            require(new_id >= old_size and 0 <= donor_id < old_size, "Original six-row donor lookup differs")
            with torch.no_grad():
                model.decoder.embedding.weight[new_id].copy_(model.decoder.embedding.weight[donor_id])
                model.joint.head.weight[new_id].copy_(model.joint.head.weight[donor_id])
                model.joint.head.bias[new_id].copy_(model.joint.head.bias[donor_id] - 2)
    require(model.config.vocab_size == 8199, "Only original six-row student source is supported")
    baseline_processor = copy.deepcopy(processor)
    namespace = hashlib.sha256((BASE_REVISION + processor.tokenizer._tokenizer.to_str()).encode()).hexdigest()[:20]
    cache = ROOT / "work/encoder-cache" / namespace
    for parameter in model.encoder.parameters(): parameter.requires_grad_(False)
    # Only original replay features/IDs are cached. Candidate TR never calls make_cache.
    replay = annotate_cache_lengths(make_cache(model, baseline_processor, replay, cache, args.max_tokens))
    confirm_replay_after_cache(replay, public_data_audit)
    layout = install_encoder_lora(model, rank=args.rank, alpha=args.alpha, last_n_layers=args.last_n_layers)
    model.config.encoder_lora_tracking = {"layout": layout, "implementation_sha256": file_sha(OUTPUT / "encoder_lora.py"), "kind": "unmerged"}
    original_record = context["original_context"]["receipt"]["artifacts"]["best_checkpoint"]
    phase1_record = context["phase1_record"]
    original = torch.load(original_record["path"], map_location="cpu", weights_only=True)
    source = torch.load(args.resume_adaptation, map_location="cpu", weights_only=True)
    expanded = load_validate_phase1(model, processor, original, source, context, torch_module=torch)
    adaptation = expanded["adaptation"]
    require(isinstance(adaptation, BPENewRowsLoRAAdaptation), "Reviewed phase2 utility required")
    runtime = expanded["inherited_runtime"]
    # All24 attention blocks retain the original nonreentrant checkpoint path.
    # Base weights remain frozen, but encoder and projector forwards carry gradients.
    checkpoint_encoder_blocks(model)
    tr, consumer_acceptance = annotate_tr_frame_metadata(baseline_processor, processor, tr,
        args.candidate_tr_metadata, args.candidate_tr_metadata_proof, args.max_tokens,
        max_joint_bytes=args.native_max_joint_bytes, return_evidence=True)
    confirm_tr_metadata_acceptance(tr, public_data_audit)
    public_data_audit.update(candidate_TRAIN_admission=consumer_acceptance)
    sampler = TSCDomainBalancedBucketSampler(tr, replay, seed=args.seed, replay_seed=args.replay_seed,
        sampling_mode=args.sampling_mode, batch_size=args.batch_size, accumulation=args.accumulation,
        replay_fraction=args.replay_fraction, fleurs_fraction=args.fleurs_fraction, bucket_size=args.bucket_size,
        cycler_factory=BucketCycler, cycler_source_sha256=file_sha(OUTPUT / "train_lora_batched_data_v11.py"))
    sampler.load_state_dict(runtime["sampler"])
    start_step = context["source_selected_step"]
    require(start_step == adaptation.cumulative_step and runtime["step"] == start_step
        and sampler.identity == context["parent_identity"]["domain_sampling_identity"],
        "Chosen acquired point/target identity/frontier differs")
    require(sampler.updates == start_step and sampler.tr_draws == start_step * sampler.tr_per_update
            and sum(stream.draws for stream in sampler.replay.values()) == start_step * sampler.replay_per_update,
            "Inherited sampler frontier/draws must match selected BEST")
    configured_steps = context["config"]["steps"]
    require(configured_steps == 44958 and start_step < configured_steps,
            "Preserve actual global minimum-domain horizon with remaining progress")
    loss_function, joint_bound = native_loss_function(args, tr, replay)
    from native_tdt_checkpointed_v4 import reclaim_inactive_cuda_cache
    # Reconstruct paired original/phase1 geometry and moments first, then release
    # their full CPU dictionaries BEFORE loading the own-phase2 latest artifact.
    del original, source, baseline_processor, runtime
    adaptation.inherited_runtime = None
    expanded.pop("inherited_runtime", None)
    saved_phase = torch.load(args.resume_phase2_exact, map_location="cpu", weights_only=True)
    runtime = restore_runtime(adaptation, sampler, saved_phase, exact_context, torch_module=torch)
    resume_step = exact_context["resume_step"]
    require(runtime["step"] == resume_step == adaptation.cumulative_step
        and adaptation.phase2_updates == exact_context["phase2_updates_at_resume"]
        and sampler.updates == resume_step and sampler.tr_draws == resume_step * sampler.tr_per_update
        and sum(stream.draws for stream in sampler.replay.values()) == resume_step * sampler.replay_per_update,
        "Restored own-phase2 S/K/N sampler and cumulative frontier differ")
    # Typed internal identity is carried from Torch; the external JSON comparison
    # is serialization-equivalent (Adam betas tuple/list), never a science rewrite.
    config_identity = saved_phase["config_identity"]
    require(canonical(config_identity) == canonical(exact_context["parent_identity"])
        and canonical(public_data_audit["identity"]) == canonical(config_identity["public_turkish_data_policy"])
        and expanded["vocabulary"]["extended_tokenizer_sha256"] == config_identity["extended_tokenizer_sha256"]
        and encoder_lora_metadata(model) == config_identity["lora_layout"]
        and ids_sha(tr) == config_identity["tr_pool_sha256"] and ids_sha(replay) == config_identity["replay_pool_sha256"],
        "Actual cold reconstruction must retain original data/tokenizer/LoRA/science identity")
    source_sha, input_sha = config_identity["source_sha256"], config_identity["input_sha256"]
    require(all(file_sha(OUTPUT / name) == digest for name, digest in source_sha.items())
        and all(file_sha(path) == digest for path, digest in input_sha.items()),
        "An immutable original scientific source or input changed")
    history = runtime["history"]
    best, selected_step = runtime["best_score"], runtime["selected_step"]
    patience, retention_failures = runtime["patience"], runtime["retention_failures"]
    next_eval, initial_gate = runtime["next_eval_epoch"], runtime["initial_gate"]
    train_wall, evaluation_wall = runtime["train_wall_seconds"], runtime["evaluation_wall_seconds"]
    audio_seconds, language_audio = runtime["audio_seconds"], runtime["language_audio_seconds"]
    require(len(history) == resume_step, "Preserve the actual entire inherited and phase2 history")
    # No new300 evaluation, initial save, BEST reset or run_config replacement.
    config = exact_context["original_config"]
    phase_limit = exact_context["approved_total_phase2_updates"]
    phase_end = min(configured_steps, start_step + phase_limit)
    execution = {"kind": "TSC_BPE_LoRA_same_phase_exact_execution_v1", "run_id": args.run_id,
        "source_phase1_step": start_step, "resume_step": resume_step,
        "phase2_updates_at_resume": adaptation.phase2_updates,
        "approved_total_phase2_updates": phase_limit, "adaptation_end_step": phase_end,
        "original_config_identity_sha256": canonical(config_identity),
        "original_native_ready": artifact_record(args.native_backend_provenance),
        "original_source300": artifact_record(proof_path(args)),
        "exact_source_receipt": exact_context["exact_receipt_record"],
        "fresh_runtime_proof": artifact_record(args.phase2_exact_runtime_proof),
        "new_executor_worker": artifact_record(__file__),
        "source300_repeated": False, "original_config_or_BEST_reinitialized": False,
        "stopping_bound_extension_is_separately_approved": phase_limit > config["adaptation_phase_max_updates"],
        "further_unfreeze_allowed": False, "goal_complete": False}
    event_path = run_dir / ("phase2_exact_execution_from_" + str(resume_step) + ".json")
    require(not event_path.exists(), "Exclusive exact-window evidence path already exists")
    write_json(event_path, execution)
    # Shared monitor views may be republished; run-specific original config and
    # initial300/BEST bytes remain unchanged.
    write_json(OUTPUT / "run_config.json", config)
    write_json(OUTPUT / "training_history.json", history)
    # The returned own-phase runtime is authoritative; phase1 inherited_runtime
    # must never be reused. Free loaded CPU state dictionaries after restoration.
    del saved_phase, runtime, expanded
    def save(path, step, reason=None):
        save_training_state(path, model, adaptation, sampler, run_id=args.run_id, step=step,
            config_identity=config_identity, history=history, best_score=best, selected_step=selected_step,
            patience=patience, retention_failures=retention_failures, next_eval_epoch=next_eval,
            train_wall_seconds=train_wall, audio_seconds=audio_seconds, language_audio_seconds=language_audio,
            evaluation_wall_seconds=evaluation_wall, initial_gate=initial_gate, source_selected_step=start_step, stop_reason=reason)
    model.train(); model.encoder.eval()
    loop_start = time.perf_counter(); last_evaluated_step = max((row["step"] for row in history if "validation" in row), default=start_step)
    stop_reason = "adaptation_screen_complete_review_required"
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    smoke_before = _protected_snapshot(adaptation) if args.smoke_updates else None
    smoke_new_before = {}
    if args.smoke_updates:
        smoke_new_before = {name: _table_bytes_sha(adaptation.parameters[name][8199:8711]) for name in
            ("decoder.embedding.weight", "joint.head.weight", "joint.head.bias")}
        smoke_lora_before = {name: _table_bytes_sha(adaptation.parameters[name]) for name in adaptation.adapter_names}
    print(f"BPE+LoRA EXACT B8x2 source{start_step}; resume{resume_step}; phase{adaptation.phase2_updates}/{phase_limit}; full global horizon{configured_steps}", flush=True)
    def evaluate_current(step, final=False):
        nonlocal best, selected_step, patience, retention_failures, next_eval, evaluation_wall, last_evaluated_step
        adaptation.zero_grad(); torch.cuda.empty_cache()
        before = time.perf_counter()
        dev = evaluate(model, processor, f"{args.run_id}_step_{step}", splits=("validation",))
        gate = development_gate(dev, baseline)
        evaluation_wall += time.perf_counter() - before
        row = history[-1]; row["validation"] = gate
        row["development_selection_only"] = True; row["final_development_evaluation"] = final
        last_evaluated_step = step
        improved = gate["retention_gate_passed"] and gate["turkish_wer"] < best and gate["turkish_wer"] < baseline["validation_tr"]["asr"]["wer"]
        meaningful = improved and best - gate["turkish_wer"] >= args.min_improvement
        if improved: best, selected_step = gate["turkish_wer"], step
        retention_failures = 0 if gate["retention_gate_passed"] else retention_failures + 1
        previous = [r["turkish_epoch"] for r in history[:-1] if "validation" in r]
        full_epoch = math.floor(sampler.turkish_epoch + 1e-10) > math.floor((previous[-1] if previous else 0.) + 1e-10)
        if meaningful: patience = 0
        elif full_epoch: patience += 1
        next_eval = next_evaluation_epoch(sampler.turkish_epoch, args.eval_epoch_fraction, args.eval_per_epoch)
        row["selection_diagnostics"] = {"improved_qualifying_best": improved, "meaningful_improvement": meaningful,
            "plateau_epoch_evaluations": patience, "consecutive_retention_failures": retention_failures}
        if improved: save(best_path, step)
        save(latest_path, step); write_json(run_dir / "training_history.json", history)
        write_json(OUTPUT / "training_history.json", history)
        print(f"dev step{step} phase{adaptation.phase2_updates} TR WER{gate['turkish_wer']:.5f} retention {gate['retention_gate_passed']} best{selected_step}", flush=True)
        model.train(); model.encoder.eval(); torch.cuda.empty_cache()
        return gate
    for step in range(resume_step + 1, phase_end + 1):
        if args.stop_file and args.stop_file.exists(): stop_reason = "controller_stop_file"; break
        if args.max_wall_seconds is not None and time.perf_counter() - process_start >= args.max_wall_seconds:
            stop_reason = "soft_max_wall_seconds"; break
        step_start = time.perf_counter(); adaptation.zero_grad()
        totals = {key: 0. for key in ("loss", "tdt_loss", "replay_kl", "replay_duration_kl", "replay_feature_distillation")}
        sampled_sources, step_audio, padding = [], 0., []
        for group in sampler.next_batches():
            for row in group:
                if row["language"] == "tr": validate_training_item(row)
            batch = collate_training_batch(processor, group, blank_id=8192, old_size=old_size, device="cuda")
            result = batch_objective(model, teacher, batch, old_size=old_size, replay_kl=args.replay_kl,
                duration_kl=args.duration_kl, feature_distillation=args.feature_distillation,
                chunk_size=args.chunk_size, tdt_loss_fn=loss_function, tdt_host_lengths=True)
            scaled = result.loss / args.accumulation
            require(bool(torch.isfinite(scaled)), "Nonfinite adaptation objective")
            scaled.backward()
            for key, value in result.detached_metrics().items(): totals[key] += value / args.accumulation
            step_audio += batch.audio_seen_seconds; padding.append(batch.lattice_padding_factor)
            for row in group:
                language_audio[row["language"]] += float(row["duration"]); sampled_sources.append(row["source_id"])
            del result, scaled, batch
            reclaim_inactive_cuda_cache(torch.device("cuda"), "after_whole_microbatch_backward")
        gradient_evidence = None
        if args.smoke_updates:
            gradient_evidence = {"tables": {}}
            for name in ("decoder.embedding.weight", "joint.head.weight", "joint.head.bias"):
                gradient = adaptation.parameters[name].grad
                require(gradient is not None and bool(torch.isfinite(gradient).all()), "Actual complete finite three-table gradients required")
                gradient_evidence["tables"][name] = {"new_rows_gradient_nonzero": bool(torch.count_nonzero(gradient[8199:8711])),
                    "protected_gradients_present_before_mask": bool(torch.count_nonzero(gradient[:8199])) or bool(torch.count_nonzero(gradient[8711:]))}
                require(gradient_evidence["tables"][name]["new_rows_gradient_nonzero"], "Actual new-row gradient absent: " + name)
            nonzero_lora = []
            for name in adaptation.adapter_names:
                gradient = adaptation.parameters[name].grad
                require(gradient is not None and bool(torch.isfinite(gradient).all()),
                        "Actual complete finite encoder-LoRA gradient required: " + name)
                if bool(torch.count_nonzero(gradient)): nonzero_lora.append(name)
            require(nonzero_lora, "Actual differentiable encoder-LoRA gradient absent")
            gradient_evidence.update(LoRA_gradients_finite=192, LoRA_nonzero_gradient_names=nonzero_lora)
        norm = adaptation.step(max_norm=1.)
        reclaim_inactive_cuda_cache(torch.device("cuda"), "after_adam_and_new_rows")
        torch.cuda.synchronize()
        train_wall += time.perf_counter() - step_start; audio_seconds += step_audio
        record = {"step": step, "adaptation_updates_completed": adaptation.phase2_updates,
            "source_phase1_step": start_step, "original_source_selected_step": adaptation.source_step,
            "selected_acquisition_updates": adaptation.acquisition_updates,
            "row_Adam_clock": adaptation.acquisition_updates + adaptation.phase2_updates,
            "LoRA_Adam_clock": adaptation.source_step + adaptation.phase2_updates,
            **totals, "gradient_norm": float(norm), "elapsed_seconds": time.perf_counter() - loop_start,
            "process_wall_seconds": time.perf_counter() - process_start, "train_wall_seconds": train_wall,
            "evaluation_wall_seconds": evaluation_wall, "audio_seen_seconds": audio_seconds,
            "audio_minutes_per_wall_minute": audio_seconds / train_wall,
            "turkish_audio_seen_seconds": language_audio["tr"], "replay_audio_seen_seconds": audio_seconds - language_audio["tr"],
            "language_audio_seen_seconds": language_audio.copy(), "step_audio_seen_seconds": step_audio,
            "sampled_source_ids": sampled_sources, "turkish_epoch": sampler.turkish_epoch,
            "turkish_clips_seen": sampler.tr_draws, "turkish_unique_clips_seen": sampler.tr_unique,
            "Turkish_domain_coverage": sampler.coverage(), "replay_clips_seen": sum(s.draws for s in sampler.replay.values()),
            "replay_language_clip_counts": sampler.replay_language_counts.copy(),
            "row_lr": adaptation.row_group["lr"], "head_lr": 0., "lora_lr": adaptation.lora_group["lr"],
            "source_dormant_head_lr": args.head_lr, "source_dormant_lora_lr": args.lora_lr,
            "new_token_update_multiplier": 1., "old_six_character_update_multiplier_applied": False,
            "active_BPE_rows": [8199, 8711], "replay_kl_coefficient": args.replay_kl,
            "duration_kl_coefficient": args.duration_kl, "feature_distillation_coefficient": args.feature_distillation,
            "mean_microbatch_lattice_padding_factor": math.fsum(padding) / len(padding),
            "peak_allocated_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
            "current_allocated_vram_gb": torch.cuda.memory_allocated() / 1e9,
            "current_reserved_vram_gb": torch.cuda.memory_reserved() / 1e9}
        history.append(record)
        if args.smoke_updates:
            require(_protected_snapshot(adaptation) == smoke_before, "Actual protected old/six/blank/duration or frozen12 bytes changed")
            for name, before in smoke_new_before.items():
                require(_table_bytes_sha(adaptation.parameters[name][8199:8711]) != before, "Actual new-row parameter delta absent: " + name)
            changed_lora = [name for name, before in smoke_lora_before.items()
                            if _table_bytes_sha(adaptation.parameters[name]) != before]
            require(changed_lora, "Actual encoder-LoRA parameter delta absent")
            require(all(float(adaptation.row_optimizer.state[p]["step"]) == adaptation.acquisition_updates + 1 for p in adaptation.tables)
                and all(float(adaptation.lora_optimizer.state[p]["step"]) == adaptation.source_step + 1 for p in adaptation.adapters),
                "Actual independent Adam clocks must advance once")
            save(latest_path, step, "adaptation_one_update_smoke")
            saved_phase = torch.load(latest_path, map_location="cpu", weights_only=True)
            runtime = adaptation.load_phase_state(saved_phase)
            require(runtime["step"] == step and runtime["sampler"] == sampler.state_dict()
                and runtime["history"] == history and runtime["config_identity"] == config_identity
                and adaptation.phase2_updates == 1, "Internal SAME-phase save/load counter/frontier differs")
            _verify_smoke_phase_reload(adaptation, saved_phase, runtime)
            require(_protected_snapshot(adaptation) == smoke_before
                and artifact_record(original_record["path"]) == original_record
                and artifact_record(phase1_record["path"]) == phase1_record,
                "Smoke phase reload/protected bytes/paired immutable sources differ")
            write_json(run_dir / "adaptation_actual_one_update_smoke.json", {"kind": "TSC_BPE_LoRA_actual_one_update_smoke_v1",
                "passed": True, "actual_native_ASR_backward_and_two_Adam": True, "run_id": args.run_id,
                "source_phase1_step": start_step, "original_source_selected_step": adaptation.source_step,
                "selected_acquisition_updates": adaptation.acquisition_updates,
                "step": step, "adaptation_updates_completed": 1,
                "gradient_evidence": gradient_evidence, "all3_new_row_deltas_nonzero": True,
                "LoRA192_gradients_finite_and_any_delta_nonzero": True, "LoRA_changed_names": changed_lora,
                "protected_old_six_blank_duration_and_frozen12_rawbytes_exact": True,
                "parked_original207_and_phase1_unchanged": True, "complete207_two_optimizer_phase_save_load_passed": True,
                "row_clock_K_plus_N": adaptation.acquisition_updates + 1,
                "lora_clock_S_plus_N": adaptation.source_step + 1,
                "all207_reloaded_weight_bytes_exact": True, "all195_Adam_moments_clocks_and_groups_exact": True,
                "saved_sampler_history_config_and_actual_RNG_frontier_exact": True,
                "external_exact_resume_supported": False, "quality_or_longrun_fit_claimed": False})
            del saved_phase, runtime
        if adaptation.phase2_updates % args.log_every == 0 or args.smoke_updates:
            print(json.dumps(record), flush=True); write_json(run_dir / "training_history.json", history)
            write_json(OUTPUT / "training_history.json", history)
        if ((args.eval_every_updates and adaptation.phase2_updates % args.eval_every_updates == 0)
                or sampler.turkish_epoch + 1e-10 >= next_eval or step == phase_end):
            gate = evaluate_current(step)
            if args.max_retention_failures and retention_failures >= args.max_retention_failures:
                stop_reason = "consecutive_retention_gate_failures"; break
            if args.early_stop_patience and patience >= args.early_stop_patience:
                stop_reason = "development_plateau"; break
            if args.target_turkish_wer is not None and gate["retention_gate_passed"] and gate["turkish_wer"] <= args.target_turkish_wer:
                stop_reason = "development_target_met"; break
    completed = history[-1]["step"]
    if completed > resume_step and completed != last_evaluated_step: evaluate_current(completed, final=True)
    save(latest_path, completed, stop_reason); write_json(run_dir / "training_history.json", history)
    write_json(OUTPUT / "training_history.json", history)
    selection = {"accepted": selected_step is not None, "run_id": args.run_id, "selected_step": selected_step,
        "completed_steps": completed, "configured_steps": configured_steps, "source_phase1_step": start_step,
        "original_source_selected_step": adaptation.source_step,
        "selected_acquisition_updates": adaptation.acquisition_updates,
        "inherited_history_steps": start_step, "adaptation_updates_completed": adaptation.phase2_updates,
        "adaptation_end_step": phase_end, "stop_early_reason": stop_reason, "phase": PHASE,
        "turkish_validation_wer": best, "initial_adaptation_turkish_validation_wer": config["initial_adaptation_turkish_wer"],
        "initial_turkish_validation_wer": initial_gate["turkish_wer"], "initial_gate_is_inherited_original_lineage": True,
        "selection_split": "FLEURS validation only", "heldout_tests_evaluated": False,
        "weights": str(best_path), "latest_weights": str(latest_path),
        "turkish_epochs_completed": sampler.turkish_epoch, "turkish_epoch_semantics": "minimum_domain_exposure",
        "Turkish_domain_coverage": sampler.coverage(), "sampling_identity": sampler.identity,
        "train_wall_seconds": train_wall, "audio_seen_seconds": audio_seconds,
        "audio_minutes_per_wall_minute": audio_seconds / train_wall if train_wall else None,
        "external_exact_resume_supported": True, "further_unfreeze_allowed": False,
        "exact_execution": execution,
        "row_Adam_clock": adaptation.acquisition_updates + adaptation.phase2_updates,
        "LoRA_Adam_clock": adaptation.source_step + adaptation.phase2_updates,
        "reason": "Strict all-five qualifying BEST in this declared phase retained; ends for review, no automatic further unfreeze or goal completion."}
    write_json(run_dir / "selection.json", selection)
    write_json(OUTPUT / "selection.json", selection)
    print(json.dumps(selection), flush=True)


def parser():
    p = initial_worker.parser()
    p.description = __doc__
    p.add_argument("--resume-phase2-exact", type=Path, required=True)
    p.add_argument("--phase2-exact-source-receipt", type=Path, required=True)
    p.add_argument("--phase2-exact-runtime-proof", type=Path, required=True)
    return p


def validate_arguments(args):
    require(args.smoke_updates is None, "Same-phase exact is not a new smoke or fresh screen")
    # Original recipe/asset/native-token checks remain the frozen initial API.
    initial_worker.validate_arguments(args)
    exact_preflight(args)
    native_loss_identity(args)
    return args


if __name__ == "__main__":
    run(validate_arguments(parser().parse_args()))
