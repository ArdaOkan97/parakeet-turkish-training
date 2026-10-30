"""NEW inactive same-row-phase worker; never execute model/state code here."""
from pathlib import Path
HERE=Path(__file__).resolve().parents[2]/'outputs/parakeet_turkish'
w=(HERE/'train_lora_batched_tsc_bpe_row_lr_v1.py').read_text(encoding='utf-8')
def rep(a,b):
    global w
    if a not in w:raise ValueError('Missing predecessor:'+a[:80])
    w=w.replace(a,b)
rep('from tsc_bpe_row_lr_transition_v1 import (preflight as acquisition_preflight,\n    load_validate_phase1, initial_source_equivalence)',
    'from tsc_bpe_row_lr_transition_v1 import load_validate_phase1\nfrom tsc_bpe_row_exact_transition_v1 import (preflight as exact_preflight, restore_runtime,\n    validate_runtime_proof, canonical)')
start=w.index('def native_loss_identity(args):')
end=w.index('\n\ndef native_loss_function(',start)
w=w[:start]+'''def native_loss_identity(args):
    """Original immutable science token plus separate actual new executor proof."""
    from train_lora_batched_tsc_bpe_row_lr_v1 import native_loss_identity as original_identity
    original = original_identity(args)
    validate_runtime_proof(args, worker_path=__file__)
    return original
''' +w[end:]
rep('    context = acquisition_preflight(args)','    exact_context = exact_preflight(args)\n    exact_context.pop("archived_history", None)\n    context = exact_context["original_context"]')
rep('    require(not run_dir.exists() and not best_path.exists() and not latest_path.exists(),\n            "Initial acquisition requires an exclusive new namespace and checkpoint paths")\n    run_dir.mkdir(parents=True, exist_ok=False)',
    '    require(run_dir.is_dir() and best_path.is_file(), "Exact SAME run requires original namespace/BEST")\n    expected_best = exact_context["exact_receipt"]["artifacts"]["best_checkpoint"]\n    require(file_sha(best_path) == expected_best["sha256"] and best_path.stat().st_size == expected_best["bytes"],\n            "Preserve actual CLOSED qualified BEST before any update")')
start=w.index('    # Mandatory new unsuppressed actual300 equality')
end=w.index('    def save(path, step, reason=None):',start)
w=w[:start]+'''    # Release paired immutable original/phase1 CPU objects BEFORE loading own
    # latest. The constructor's old phase1 runtime is never reused afterward.
    del original, source, runtime, baseline_processor
    acquisition.inherited_runtime = None
    expanded.pop("inherited_runtime", None)
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
    event_path = run_dir / ("row_exact_execution_from_" + str(resume_step) + ".json")
    require(not event_path.exists(), "Exact execution evidence may not be overwritten")
    write_json(event_path, execution)
    write_json(OUTPUT / "run_config.json", config); write_json(OUTPUT / "training_history.json", history)
    parked = context["receipt"]["artifacts"]["original_source_checkpoint"]
    del own_state, runtime, expanded
''' +w[end:]
rep('    save(best_path, start_step, "actual_acquired_source300_equal_row_LR_phase0")\n    save(latest_path, start_step)\n','')
rep('    loop_start = time.perf_counter(); last_evaluated_step = start_step',
    '    loop_start = time.perf_counter(); last_evaluated_step = max((r["step"] for r in history if "validation" in r), default=start_step)')
rep('    for step in range(start_step + 1, phase_end + 1):','    for step in range(resume_step + 1, phase_end + 1):')
rep('"external_exact_resume_supported": False, "unfreeze_allowed": False,\n        "reason":',
    '"external_exact_resume_supported": True, "unfreeze_allowed": False, "exact_execution": execution,\n        "reason":')
rep('    p.add_argument("--smoke-updates", type=int, choices=(1,))',
    '    p.add_argument("--smoke-updates", type=int, choices=(1,))\n    p.add_argument("--resume-row-exact", type=Path, required=True)\n    p.add_argument("--row-exact-source-receipt", type=Path, required=True)\n    p.add_argument("--row-exact-runtime-proof", type=Path, required=True)')
rep('    acquisition_preflight(args)','    require(args.smoke_updates is None, "Exact continuation never admits fresh smoke")\n    exact_preflight(args)')
w=w.replace('"""Inactive acquired-source TSC BPE512 row-only LR challenger.',
    '"""INACTIVE SAME-ID own-row full-state continuation; new controller still required.')
p=HERE/'train_lora_batched_tsc_bpe_row_exact_v1.py'
if p.exists():raise ValueError('New exact worker exists')
p.write_text(w,encoding='utf-8')
print(str(p))
