"""INACTIVE SAME-ID own-row full-state continuation; new controller still required.

The source must be an actually CLOSED competitive qualifying own-row latest.
Its207 weights, three-table AdamK+N, localN, sampler/RNG/history and original
identity carry exactly. Initial original/phase1 anchors reconstruct geometry;
their old runtime is discarded before loading the actual own frontier. Original
source300/native tokens stay immutable; no re-evaluation, BEST reset or earlier
BEST rewind is performed. Encoder LoRA and all204 shared weights remain frozen.
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

from batched_lora_tr_metadata_v1 import (BatchObjective, collate_training_batch,
                                        sampled_preservation)
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
from tsc_bpe_row_lr_transition_v1 import load_validate_phase1
from tsc_bpe_row_exact_transition_v1 import (preflight as exact_preflight, restore_runtime,
    validate_runtime_proof, canonical)
from bpe_newrows_continued_acquisition_v1 import BPEContinuedAcquisition
from bpe_newrows_acquisition_v1 import artifact_record

PHASE = "HF_TSC_BPE512_new_rows_continued_acquisition_v1"
PROOF_NAME = "TSC_BPE_row_LR_initial_source_equivalence.json"
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


def batch_objective(model, teacher, batch, *, old_size, replay_kl, duration_kl,
                    feature_distillation, chunk_size, tdt_loss_fn, tdt_host_lengths=True):
    """Frozen metadata objective with explicit no_grad ONLY around encoder/projector.

    Frozen decoder/joint parameters remain in the autograd graph so that native
    TDT and full-vocabulary KD can differentiate candidate embeddings/head rows.
    Per-clip target-length normalization and physical-batch mean are unchanged.
    """
    require(not model.encoder.training and old_size == batch.old_size
            and model.config.blank_token_id == batch.blank_id,
            "Frozen encoder/batch/teacher identities differ")
    device = batch.labels.device
    with torch.autocast("cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            encoded = model.encoder(**batch.inputs)
            require(encoded.attention_mask is not None, "Encoder output mask required")
            maximum_t = int(batch.frame_lengths.max())
            require(encoded.last_hidden_state.shape[1] >= maximum_t,
                    "Encoder cannot represent every vetted frame")
            actual_lengths = encoded.attention_mask.sum(-1).detach().cpu().long()
            require(torch.equal(actual_lengths, batch.frame_lengths), "Exact encoder lengths changed")
            features = encoded.last_hidden_state[:, :maximum_t]
            ep = model.encoder_projector(features)
        dp = model.decoder(batch.decoder_input_ids)
        loss_frame_lengths = batch.frame_lengths if tdt_host_lengths else batch.frame_lengths.to(device)
        loss_target_lengths = batch.target_lengths if tdt_host_lengths else batch.target_lengths.to(device)
        raw_tdt = tdt_loss_fn(model.joint, ep, dp, batch.labels, model.config.vocab_size,
            batch.blank_id, model.config.durations, logit_lengths=loss_frame_lengths,
            target_lengths=loss_target_lengths, chunk_size=chunk_size, reduction="none")
        require(raw_tdt.shape == (batch.batch_size,), "Native raw per-clip reduction changed")
        per_tdt = raw_tdt.float() / batch.target_lengths.to(device)
        token, duration, feature = sampled_preservation(model, teacher, batch,
            features, ep, dp, old_size=old_size)
        total_per_clip = per_tdt + replay_kl * token + duration_kl * duration + feature_distillation * feature
    return BatchObjective(total=total_per_clip.mean(), tdt=per_tdt.mean(), token_kl=token.mean(),
        duration_kl=duration.mean(), feature_distillation=feature.mean(),
        per_clip={"tdt": per_tdt, "token_kl": token, "duration_kl": duration,
                  "feature_distillation": feature, "total": total_per_clip},
        batch_size=batch.batch_size, encoder_frames=tuple(batch.frame_lengths.tolist()),
        target_tokens=tuple(batch.target_lengths.tolist()))


def native_loss_identity(args):
    """Original immutable science token plus separate actual new executor proof."""
    from train_lora_batched_tsc_bpe_row_lr_v1 import native_loss_identity as original_identity
    original = original_identity(args)
    validate_runtime_proof(args, worker_path=__file__)
    return original


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


def save_training_state(path, model, acquisition, sampler, *, run_id, step, config_identity,
                        history, best_score, selected_step, patience, retention_failures,
                        next_eval_epoch, train_wall_seconds, audio_seconds, language_audio_seconds,
                        evaluation_wall_seconds, initial_gate, source_selected_step, stop_reason=None):
    """NEW phase serializer: all207 learned tensors, but only3 acquisition Adam states.

    No legacy requires_grad filtering and no source optimizer/history relabelling.
    Actual external phase resume is intentionally not admitted by this worker.
    """
    require(step == acquisition.cumulative_step and sampler.updates == step and len(history) == step, "Durable row continuation frontier differs")
    runtime_state = {"run_id": run_id, "step": step, "source_selected_step": source_selected_step,
        "source_acquisition_updates": acquisition.source_acquisition_updates,
        "original_source_selected_step": acquisition.source_step,
        "acquisition_updates_completed": acquisition.optimizer_updates_completed,
        "config_identity": config_identity, "sampler": sampler.state_dict(),
        "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(), "history": history,
        "best_score": best_score, "selected_step": selected_step, "patience": patience,
        "retention_failures": retention_failures, "next_eval_epoch": next_eval_epoch,
        "train_wall_seconds": train_wall_seconds, "audio_seconds": audio_seconds,
        "language_audio_seconds": language_audio_seconds, "evaluation_wall_seconds": evaluation_wall_seconds,
        "initial_gate": initial_gate, "stop_reason": stop_reason}
    state = acquisition.state_dict(runtime_state=runtime_state)
    state.update(run_id=run_id, step=step, source_selected_step=source_selected_step,
                 selected_step=selected_step, config_identity=config_identity)
    temporary = Path(str(path) + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def _table_bytes_sha(tensor):
    """Smoke-only actual tensor bytes; no repeated per-update full-weight hashing."""
    return hashlib.sha256(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def _protected_snapshot(acquisition):
    protected = {}
    for name in acquisition.names:
        tensor = acquisition.parameters[name]
        if name in ("decoder.embedding.weight", "joint.head.weight", "joint.head.bias"):
            protected[name] = {"old_blank_six": _table_bytes_sha(tensor[:8199]),
                               "duration": _table_bytes_sha(tensor[8711:])}
        else:
            protected[name] = _table_bytes_sha(tensor)
    return protected


def _verify_smoke_phase_reload(acquisition, saved, runtime):
    """One smoke only: inspect actual reloaded207 weights and three Adam states."""
    require(set(saved["weights"]) == set(acquisition.names), "Smoke phase weight names differ")
    for name in acquisition.names:
        require(_table_bytes_sha(acquisition.parameters[name]) == _table_bytes_sha(saved["weights"][name]),
                "Smoke full207 reloaded weight bytes differ: " + name)
    actual = acquisition.optimizer.state_dict()
    expected = saved["acquisition_optimizer"]
    require(actual["param_groups"] == expected["param_groups"] and set(actual["state"]) == set(expected["state"]),
            "Smoke three-table Adam groups or state IDs differ")
    for identifier, state in expected["state"].items():
        require(set(actual["state"][identifier]) == set(state), "Smoke Adam state fields differ")
        for key, tensor in state.items():
            reloaded = actual["state"][identifier][key]
            require(reloaded.shape == tensor.shape and reloaded.dtype == tensor.dtype
                    and _table_bytes_sha(reloaded) == _table_bytes_sha(tensor),
                    "Smoke three-table Adam moment/clock bytes differ: " + key)
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
    exact_context.pop("archived_history", None)
    context = exact_context["original_context"]
    public_data_audit = audit_tsc_training_inputs(args)
    public_data_audit["identity"].update(target_replay_rng_independent_of_turkish=True,
        target_replay_seed=args.replay_seed, replay_sampling_stream_bitwise_identity_claimed=False,
        replay_sampling_nuisance="Original replay pool, independent RNG, acquired phase1 pending orders/targets/cursors carry exactly; no target migration.")
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    run_dir = OUTPUT / "runs" / args.run_id
    best_path, latest_path = (ROOT / "work" / (args.run_id + suffix) for suffix in ("_best.pt", "_latest.pt"))
    require(run_dir.is_dir() and best_path.is_file(), "Exact SAME run requires original namespace/BEST")
    expected_best = exact_context["exact_receipt"]["artifacts"]["best_checkpoint"]
    require(file_sha(best_path) == expected_best["sha256"] and best_path.stat().st_size == expected_best["bytes"],
            "Preserve actual CLOSED qualified BEST before any update")
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
    original_record = context["receipt"]["artifacts"]["original_source_checkpoint"]
    original = torch.load(original_record["path"], map_location="cpu", weights_only=True)
    source = torch.load(args.resume_row_lr, map_location="cpu", weights_only=True)
    expanded = load_validate_phase1(model, processor, original, source, context, torch_module=torch)
    acquisition = expanded["acquisition"]
    runtime = expanded["inherited_runtime"]
    tr, consumer_acceptance = annotate_tr_frame_metadata(baseline_processor, processor, tr,
        args.candidate_tr_metadata, args.candidate_tr_metadata_proof, args.max_tokens,
        max_joint_bytes=args.native_max_joint_bytes, return_evidence=True)
    confirm_tr_metadata_acceptance(tr, public_data_audit)
    public_data_audit.update(candidate_TRAIN_admission=consumer_acceptance)
    sampler = TSCDomainBalancedBucketSampler(tr, replay, seed=args.seed, replay_seed=args.replay_seed,
        sampling_mode=args.sampling_mode, batch_size=args.batch_size, accumulation=args.accumulation,
        replay_fraction=args.replay_fraction, fleurs_fraction=args.fleurs_fraction, bucket_size=args.bucket_size,
        cycler_factory=BucketCycler, cycler_source_sha256=file_sha(OUTPUT / "train_lora_batched_data_v11.py"))
    sampler_transition = "none; exact acquired phase1 targets and queues"
    sampler.load_state_dict(runtime["sampler"])
    start_step = context["source_selected_step"]
    require(start_step == acquisition.cumulative_step and source["step"] == start_step, "Acquired frontier differs")
    require(sampler.updates == start_step and sampler.tr_draws == start_step * sampler.tr_per_update
            and sum(stream.draws for stream in sampler.replay.values()) == start_step * sampler.replay_per_update,
            "Inherited sampler frontier/draws must match selected BEST")
    configured_steps = context["config"]["steps"]
    require(configured_steps == 44958 and start_step < configured_steps,
            "Preserve actual global minimum-domain horizon with remaining progress")
    loss_function, joint_bound = native_loss_function(args, tr, replay)
    from native_tdt_checkpointed_v4 import reclaim_inactive_cuda_cache
    # Release paired immutable original/phase1 CPU objects BEFORE loading own
    # latest. The constructor's old phase1 runtime is never reused afterward.
    del original, source, runtime, baseline_processor
    acquisition.inherited_runtime = None
    expanded.pop("inherited_runtime", None)
    require(artifact_record(args.resume_row_exact) == exact_context["exact_receipt"]["artifacts"]["latest_checkpoint"],
            "Archived own207/3Adam bytes changed before actual state load")
    own_state = torch.load(args.resume_row_exact, map_location="cpu", weights_only=True)
    runtime = restore_runtime(acquisition, sampler, own_state, exact_context, torch_module=torch)
    resume_step = exact_context["resume_step"]
    require(resume_step == acquisition.cumulative_step == sampler.updates == runtime["step"]
        and acquisition.optimizer_updates_completed == exact_context["continued_updates_at_resume"],
        "Exact current207/3Adam/sampler localN and K+N frontier differs")
    config_identity = own_state["config_identity"]
    require(canonical(config_identity) == canonical(exact_context["parent_identity"])
        and canonical(public_data_audit["identity"]) == canonical(config_identity["public_turkish_data_policy"])
        and expanded["vocabulary"]["extended_tokenizer_sha256"] == config_identity["extended_tokenizer_sha256"]
        and encoder_lora_metadata(model) == config_identity["lora_layout"]
        and ids_sha(tr) == config_identity["tr_pool_sha256"] and ids_sha(replay) == config_identity["replay_pool_sha256"],
        "Cold reconstruction must preserve actual immutable data/tokenizer/layout science")
    source_sha, input_sha = config_identity["source_sha256"], config_identity["input_sha256"]
    require(all(file_sha(OUTPUT / name) == digest for name, digest in source_sha.items())
        and all(file_sha(path) == digest for path, digest in input_sha.items()), "Original scientific source/input changed")
    history = runtime["history"]
    best, selected_step = runtime["best_score"], runtime["selected_step"]
    patience, retention_failures = runtime["patience"], runtime["retention_failures"]
    next_eval, initial_gate = runtime["next_eval_epoch"], runtime["initial_gate"]
    train_wall, evaluation_wall = runtime["train_wall_seconds"], runtime["evaluation_wall_seconds"]
    audio_seconds, language_audio = runtime["audio_seconds"], runtime["language_audio_seconds"]
    config = exact_context["original_config"]
    phase_limit = exact_context["approved_total_continued_updates"]
    phase_end = min(configured_steps, start_step + phase_limit)
    execution = {"kind": "TSC_BPE_continued_row_same_phase_exact_execution_v1", "run_id": args.run_id,
        "source_phase1_step": start_step, "resume_step": resume_step,
        "continued_updates_at_resume": acquisition.optimizer_updates_completed,
        "source_row_Adam_clock": acquisition.source_acquisition_updates,
        "approved_total_continued_updates": phase_limit, "acquisition_end_step": phase_end,
        "original_config_identity_sha256": canonical(config_identity),
        "original_native_ready": artifact_record(args.native_backend_provenance),
        "original_source300": artifact_record(proof_path(args)),
        "exact_source_receipt": exact_context["exact_receipt_record"],
        "fresh_runtime_proof": artifact_record(args.row_exact_runtime_proof),
        "new_executor_worker": artifact_record(__file__), "source300_repeated": False,
        "original_config_or_BEST_reinitialized": False, "unfreeze_allowed": False,
        "separate_stopping_bound_extension": phase_limit > config["acquisition_phase_max_updates"], "goal_complete": False}
    event_path = run_dir / ("row_exact_execution_from_" + str(resume_step) + "_"
        + exact_context["exact_receipt_record"]["sha256"][:16] + ".json")
    require(not event_path.exists(), "Exact execution evidence may not be overwritten")
    write_json(event_path, execution)
    write_json(OUTPUT / "run_config.json", config); write_json(OUTPUT / "training_history.json", history)
    parked = context["receipt"]["artifacts"]["original_source_checkpoint"]
    del own_state, runtime, expanded
    def save(path, step, reason=None):
        save_training_state(path, model, acquisition, sampler, run_id=args.run_id, step=step,
            config_identity=config_identity, history=history, best_score=best, selected_step=selected_step,
            patience=patience, retention_failures=retention_failures, next_eval_epoch=next_eval,
            train_wall_seconds=train_wall, audio_seconds=audio_seconds, language_audio_seconds=language_audio,
            evaluation_wall_seconds=evaluation_wall, initial_gate=initial_gate, source_selected_step=start_step, stop_reason=reason)
    model.train(); model.encoder.eval()
    loop_start = time.perf_counter(); last_evaluated_step = max((r["step"] for r in history if "validation" in r), default=start_step)
    stop_reason = "acquisition_screen_complete_review_required"
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    smoke_before = _protected_snapshot(acquisition) if args.smoke_updates else None
    smoke_new_before = {}
    if args.smoke_updates:
        smoke_new_before = {name: _table_bytes_sha(acquisition.parameters[name][8199:8711]) for name in
            ("decoder.embedding.weight", "joint.head.weight", "joint.head.bias")}
    print(f"Acquisition EXACT B8x2 source{start_step}; resume{resume_step}; phase{acquisition.optimizer_updates_completed}/{phase_limit}; full global horizon{configured_steps}", flush=True)
    def evaluate_current(step, final=False):
        nonlocal best, selected_step, patience, retention_failures, next_eval, evaluation_wall, last_evaluated_step
        acquisition.zero_grad(); torch.cuda.empty_cache()
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
        print(f"dev step{step} phase{acquisition.optimizer_updates_completed} TR WER{gate['turkish_wer']:.5f} retention {gate['retention_gate_passed']} best{selected_step}", flush=True)
        model.train(); model.encoder.eval(); torch.cuda.empty_cache()
        return gate
    for step in range(resume_step + 1, phase_end + 1):
        if args.stop_file and args.stop_file.exists(): stop_reason = "controller_stop_file"; break
        if args.max_wall_seconds is not None and time.perf_counter() - process_start >= args.max_wall_seconds:
            stop_reason = "soft_max_wall_seconds"; break
        step_start = time.perf_counter(); acquisition.zero_grad()
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
            require(bool(torch.isfinite(scaled)), "Nonfinite acquisition objective")
            scaled.backward()
            for key, value in result.detached_metrics().items(): totals[key] += value / args.accumulation
            step_audio += batch.audio_seen_seconds; padding.append(batch.lattice_padding_factor)
            for row in group:
                language_audio[row["language"]] += float(row["duration"]); sampled_sources.append(row["source_id"])
            del result, scaled, batch
            reclaim_inactive_cuda_cache(torch.device("cuda"), "after_whole_microbatch_backward")
        gradient_evidence = None
        if args.smoke_updates:
            gradient_evidence = {}
            for name in ("decoder.embedding.weight", "joint.head.weight", "joint.head.bias"):
                gradient = acquisition.parameters[name].grad
                require(gradient is not None and bool(torch.isfinite(gradient).all()), "Actual complete finite three-table gradients required")
                gradient_evidence[name] = {"new_rows_gradient_nonzero": bool(torch.count_nonzero(gradient[8199:8711])),
                    "protected_gradients_present_before_mask": bool(torch.count_nonzero(gradient[:8199])) or bool(torch.count_nonzero(gradient[8711:]))}
                require(gradient_evidence[name]["new_rows_gradient_nonzero"], "Actual new-row gradient absent: " + name)
        norm = acquisition.step(max_norm=1.)
        reclaim_inactive_cuda_cache(torch.device("cuda"), "after_adam_and_new_rows")
        torch.cuda.synchronize()
        train_wall += time.perf_counter() - step_start; audio_seconds += step_audio
        record = {"step": step, "acquisition_updates_completed": acquisition.optimizer_updates_completed,
            "source_acquisition_updates": acquisition.source_acquisition_updates,
            "row_Adam_clock": acquisition.source_acquisition_updates + acquisition.optimizer_updates_completed,
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
            "acquisition_lr": args.active_row_lr, "active_row_lr": args.active_row_lr, "source_row_lr": 1e-4, "head_lr": 0., "lora_lr": 0.,
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
            require(_protected_snapshot(acquisition) == smoke_before, "Actual protected old/six/blank/duration or frozen204 bytes changed")
            for name, before in smoke_new_before.items():
                require(_table_bytes_sha(acquisition.parameters[name][8199:8711]) != before, "Actual new-row parameter delta absent: " + name)
            save(latest_path, step, "acquisition_one_update_smoke")
            saved_phase = torch.load(latest_path, map_location="cpu", weights_only=True)
            runtime = acquisition.load_phase_state(saved_phase)
            require(runtime["step"] == step and runtime["sampler"] == sampler.state_dict()
                and runtime["history"] == history and runtime["config_identity"] == config_identity
                and acquisition.optimizer_updates_completed == 1, "Internal SAME-phase save/load counter/frontier differs")
            _verify_smoke_phase_reload(acquisition, saved_phase, runtime)
            require(_protected_snapshot(acquisition) == smoke_before and artifact_record(parked["path"]) == parked,
                    "Smoke phase reload/protected bytes/parked source differs")
            write_json(run_dir / "acquisition_actual_one_update_smoke.json", {"kind": "TSC_BPE_newrows_actual_one_update_smoke_v1",
                "passed": True, "actual_native_ASR_backward_and_Adam": True, "run_id": args.run_id,
                "source_step": start_step, "step": step, "acquisition_updates_completed": 1,
                "original_source_selected_step": acquisition.source_step,
                "source_acquisition_updates": acquisition.source_acquisition_updates,
                "row_Adam_clock": acquisition.source_acquisition_updates + acquisition.optimizer_updates_completed,
                "active_row_lr": args.active_row_lr, "source_row_lr": 1e-4,
                "encoder_LoRA_frozen": True, "acquired_phase1_source_used": True,
                "gradient_evidence": gradient_evidence, "all3_new_row_deltas_nonzero": True,
                "protected_old_six_blank_duration_and_frozen204_rawbytes_exact": True,
                "parked_source_unchanged": True, "complete207_phase_save_load_passed": True,
                "all207_reloaded_weight_bytes_exact": True, "all3_Adam_moments_clocks_and_groups_exact": True,
                "saved_sampler_history_config_and_actual_RNG_frontier_exact": True,
                "external_exact_resume_supported": False, "quality_or_longrun_fit_claimed": False})
            del saved_phase, runtime
        if acquisition.optimizer_updates_completed % args.log_every == 0 or args.smoke_updates:
            print(json.dumps(record), flush=True); write_json(run_dir / "training_history.json", history)
            write_json(OUTPUT / "training_history.json", history)
        if ((args.eval_every_updates and acquisition.optimizer_updates_completed % args.eval_every_updates == 0)
                or sampler.turkish_epoch + 1e-10 >= next_eval or step == phase_end):
            gate = evaluate_current(step)
            if args.max_retention_failures and retention_failures >= args.max_retention_failures:
                stop_reason = "consecutive_retention_gate_failures"; break
            if args.early_stop_patience and patience >= args.early_stop_patience:
                stop_reason = "development_plateau"; break
            if args.target_turkish_wer is not None and gate["retention_gate_passed"] and gate["turkish_wer"] <= args.target_turkish_wer:
                stop_reason = "development_target_met"; break
    completed = history[-1]["step"]
    if completed > start_step and completed != last_evaluated_step: evaluate_current(completed, final=True)
    save(latest_path, completed, stop_reason); write_json(run_dir / "training_history.json", history)
    write_json(OUTPUT / "training_history.json", history)
    selection = {"accepted": selected_step is not None, "run_id": args.run_id, "selected_step": selected_step,
        "completed_steps": completed, "configured_steps": configured_steps, "source_selected_step": start_step,
        "inherited_history_steps": start_step, "acquisition_updates_completed": acquisition.optimizer_updates_completed,
        "original_source_selected_step": acquisition.source_step, "source_acquisition_updates": acquisition.source_acquisition_updates,
        "row_Adam_clock": acquisition.source_acquisition_updates + acquisition.optimizer_updates_completed,
        "acquisition_end_step": phase_end, "stop_early_reason": stop_reason, "phase": PHASE,
        "turkish_validation_wer": best, "initial_acquisition_turkish_validation_wer": config["initial_acquisition_turkish_wer"],
        "initial_turkish_validation_wer": initial_gate["turkish_wer"], "initial_gate_is_inherited_original_lineage": True,
        "selection_split": "FLEURS validation only", "heldout_tests_evaluated": False,
        "weights": str(best_path), "latest_weights": str(latest_path),
        "turkish_epochs_completed": sampler.turkish_epoch, "turkish_epoch_semantics": "minimum_domain_exposure",
        "Turkish_domain_coverage": sampler.coverage(), "sampling_identity": sampler.identity,
        "train_wall_seconds": train_wall, "audio_seen_seconds": audio_seconds,
        "audio_minutes_per_wall_minute": audio_seconds / train_wall if train_wall else None,
        "external_exact_resume_supported": True, "unfreeze_allowed": False, "exact_execution": execution,
        "reason": "Strict all-five qualifying BEST retained; row-only screen ends for review, never automatic unfreeze or goal completion."}
    write_json(run_dir / "selection.json", selection)
    write_json(OUTPUT / "selection.json", selection)
    print(json.dumps(selection), flush=True)


def parser():
    p = source_parser()
    p.description = __doc__
    for action in p._actions:
        if action.dest == "optimization_source_receipt": action.required = False
    p.add_argument("--resume-row-lr", type=Path, required=True)
    p.add_argument("--row-lr-source-receipt", type=Path, required=True)
    p.add_argument("--active-row-lr", type=float, choices=(1e-3, 3e-4), default=1e-3)
    p.add_argument("--acquisition-max-updates", type=int, default=1000)
    p.add_argument("--smoke-updates", type=int, choices=(1,))
    p.add_argument("--resume-row-exact", type=Path, required=True)
    p.add_argument("--row-exact-source-receipt", type=Path, required=True)
    p.add_argument("--row-exact-runtime-proof", type=Path, required=True)
    p.add_argument("--candidate-dir", type=Path, default=ROOT / "work/hf_turkish_bpe_candidate_512_v1")
    p.add_argument("--candidate-tr-metadata", type=Path, default=METADATA_PATH)
    p.add_argument("--candidate-tr-metadata-proof", type=Path, default=REPORT_PATH)
    p.set_defaults(native_max_joint_bytes=1_850_000_000)
    return p


def validate_arguments(args):
    require(args.resume_weights is None and args.resume_exact is None and args.resume_optimization is None
            and args.resume_runtime_migration is None and args.optimization_source_receipt is None,
            "Initial acquisition only; no legacy or shallow exact-phase resume is implemented")
    require(args.train_limit is None and args.supplement is None and args.reuse_input_audit is None,
            "Complete approved253798/TR1595-replay acquisition only")
    require(args.active_row_lr in (1e-3, 3e-4) and type(args.acquisition_max_updates) is int
            and 1 <= args.acquisition_max_updates <= 1000, "Retained acquired moments with declared rowLR1e-3 or3e-4 bounded phase required")
    require(args.batch_size == 8 and args.accumulation == 2 and args.sampling_mode == "domain30"
        and args.fleurs_fraction == .3 and args.replay_fraction == .5 and args.replay_kl == 10
        and args.duration_kl == .5 and args.feature_distillation == 1 and args.rank == 8 and args.alpha == 16
        and args.last_n_layers is None and args.tdt_implementation == "nemo"
        and args.native_max_joint_bytes == 1_850_000_000,
        "Preserve the approved B8x2/domain30/nativeTDT/KD10 recipe with explicit widened bound")
    require(args.eval_every_updates == 250 and args.eval_per_epoch > 0 and 0 < args.eval_epoch_fraction <= 1
        and args.log_every > 0 and args.bucket_size > 0 and args.max_tokens == 180
        and args.chunk_size > 0 and args.early_stop_patience >= 0 and args.max_retention_failures >= 0
        and args.min_improvement >= 0, "Source evaluation/stop/masking settings required")
    require(all(math.isfinite(v) for v in vars(args).values() if isinstance(v, float)), "Finite arguments required")
    require(args.max_wall_seconds is None or args.max_wall_seconds > 0, "Positive cooperative process budget required")
    require(args.run_id and Path(args.run_id).name == args.run_id and args.run_id not in (".", ".."),
            "Safe fresh single-component run ID required")
    require(args.smoke_updates is None, "Exact continuation never admits fresh smoke")
    exact_preflight(args)
    native_loss_identity(args)
    return args


if __name__ == "__main__":
    run(validate_arguments(parser().parse_args()))
