"""Source-only derivation of the inactive own-shared-state executor."""
import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / 'outputs/parakeet_turkish'

def function(text, name):
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ''.join(text.splitlines(keepends=True)[node.lineno-1:node.end_lineno])

def replace_function(text, name, replacement):
    old = function(text, name)
    return text.replace(old, replacement.rstrip(), 1)

source = (HERE/'train_lora_batched_tsc_bpe_shared_head_memory_v2.py').read_text(encoding='utf-8')
worker = source.replace(source[:source.index('from __future__')], '''"""INACTIVE SAME-ID own-shared-state exact continuation.

The original config, source300/native tokens and independent BEST are preserved.
Only an actually CLOSED all-five-qualified latest improving the starting
champion can supply this full207/two-Adam/S-K-N/sampler/RNG/history frontier.
An approved total localN extension is recorded separately from the original
1000-update screen; budget alone never proves complete unique TRAIN coverage.
"""
''',1)
worker = worker.replace('    load_validate_phase1, initial_source_equivalence, release_admission_caches)',
    '    load_validate_phase1, release_admission_caches)\nfrom tsc_bpe_shared_head_exact_transition_v1 import (preflight as exact_preflight,\n    restore_runtime, validate_runtime_proof, canonical, full_TRAIN_coverage)')
worker = replace_function(worker, 'native_loss_identity', '''def native_loss_identity(args):
    # Scientific token remains the immutable original ready, while this fresh
    # executor must independently prove same-PID preload/allocator/source pins.
    from train_lora_batched_tsc_bpe_shared_head_memory_v2 import native_loss_identity as original_identity
    value = original_identity(args)
    validate_runtime_proof(args, worker_path=__file__)
    return value''')
worker = worker.replace('context = adaptation_preflight(args)',
    'exact_context = exact_preflight(args)\n    exact_context.pop("archived_history", None)\n    context = exact_context["original_context"]',1)
worker = worker.replace('''    require(not run_dir.exists() and not best_path.exists() and not latest_path.exists(),
            "Initial adaptation requires an exclusive new namespace and checkpoint paths")
    run_dir.mkdir(parents=True, exist_ok=False)''','''    require(run_dir.is_dir() and best_path.is_file() and latest_path.is_file(),
            "SAME-run exact continuation retains its actual durable namespace/BEST/latest")
    require(artifact_record(best_path)["sha256"] == exact_context["exact_receipt"]["artifacts"]["best_checkpoint"]["sha256"],
            "Existing independent BEST changed; no implicit rewind or startup BEST overwrite")''',1)
start = worker.index('    # Mandatory new unsuppressed actual300 equality, RNG/modes restored, before Adam.')
end = worker.index('    model.train(); model.encoder.eval()', start)
worker = worker[:start] + '''    # Release phase1/original CPU objects and their old runtime BEFORE loading
    # the current own frontier. The returned CURRENT runtime is authoritative.
    del original, source, runtime, baseline_processor
    adaptation.inherited_runtime = None
    expanded.pop("inherited_runtime", None)
    require(artifact_record(args.resume_shared_head_exact) == exact_context["exact_receipt"]["artifacts"]["latest_checkpoint"],
            "Archived own207/two-Adam latest bytes changed before actual state load")
    own_state = torch.load(args.resume_shared_head_exact, map_location="cpu", weights_only=True)
    runtime = restore_runtime(adaptation, sampler, own_state, exact_context, torch_module=torch)
    resume_step = exact_context["resume_step"]
    require(resume_step == adaptation.cumulative_step == sampler.updates == runtime["step"]
        and adaptation.phase2_updates == exact_context["shared_updates_at_resume"],
        "Exact current207/two-Adam/sampler/localN frontier differs")
    config_identity = own_state["config_identity"]
    require(canonical(config_identity) == canonical(exact_context["parent_identity"])
        and canonical(public_data_audit["identity"]) == canonical(config_identity["public_turkish_data_policy"])
        and expanded["vocabulary"]["extended_tokenizer_sha256"] == config_identity["extended_tokenizer_sha256"]
        and encoder_lora_metadata(model) == config_identity["lora_layout"]
        and ids_sha(tr) == config_identity["tr_pool_sha256"] and ids_sha(replay) == config_identity["replay_pool_sha256"],
        "Cold reconstruction must preserve immutable data/tokenizer/layout science")
    source_sha, input_sha = config_identity["source_sha256"], config_identity["input_sha256"]
    require(all(file_sha(OUTPUT/name) == digest for name,digest in source_sha.items())
        and all(file_sha(path) == digest for path,digest in input_sha.items()), "Original scientific source/input changed")
    history = runtime["history"]
    best, selected_step = runtime["best_score"], runtime["selected_step"]
    patience, retention_failures = runtime["patience"], runtime["retention_failures"]
    next_eval, initial_gate = runtime["next_eval_epoch"], runtime["initial_gate"]
    train_wall, evaluation_wall = runtime["train_wall_seconds"], runtime["evaluation_wall_seconds"]
    audio_seconds, language_audio = runtime["audio_seconds"], runtime["language_audio_seconds"]
    config = exact_context["original_config"]
    phase_limit = exact_context["approved_total_shared_head_updates"]
    phase_end = min(configured_steps, start_step + phase_limit)
    execution = {"kind":"TSC_BPE_shared_heads_same_phase_exact_execution_v1", "run_id":args.run_id,
        "source_phase1_step":start_step,"resume_step":resume_step,
        "shared_updates_at_resume":adaptation.phase2_updates,
        "row_Adam_clock_at_resume":adaptation.acquisition_updates+adaptation.phase2_updates,
        "shared_Adam_clock_at_resume":adaptation.source_step+adaptation.phase2_updates,
        "approved_total_shared_head_updates":phase_limit,"adaptation_end_step":phase_end,
        "original_config_identity_sha256":canonical(config_identity),
        "original_native_ready":artifact_record(args.native_backend_provenance),
        "original_source300":artifact_record(proof_path(args)),
        "exact_source_receipt":exact_context["exact_receipt_record"],
        "fresh_runtime_proof":artifact_record(args.shared_head_exact_runtime_proof),
        "new_executor_worker":artifact_record(__file__),
        "source300_repeated":False,"original_config_or_BEST_reinitialized":False,
        "separate_stopping_bound_extension":phase_limit>config["adaptation_phase_max_updates"],
        "full_unique_coverage_at_resume":full_TRAIN_coverage(sampler.coverage()),
        "further_unfreeze_allowed":False,"goal_complete":False}
    event_path = run_dir/("shared_head_exact_execution_from_"+str(resume_step)+"_"
        +exact_context["exact_receipt_record"]["sha256"][:16]+".json")
    require(not event_path.exists(), "Exact execution evidence may not be overwritten")
    write_json(event_path, execution)
    # Shared monitor aliases retain the original static config and current
    # genuine history. Never rewrite initial run-specific config/proof/BEST.
    write_json(OUTPUT/"run_config.json", config);write_json(OUTPUT/"training_history.json", history)
    del own_state,runtime,expanded,context
    def save(path, step, reason=None):
        save_training_state(path, model, adaptation, sampler, run_id=args.run_id, step=step,
            config_identity=config_identity, history=history, best_score=best, selected_step=selected_step,
            patience=patience, retention_failures=retention_failures, next_eval_epoch=next_eval,
            train_wall_seconds=train_wall, audio_seconds=audio_seconds, language_audio_seconds=language_audio,
            evaluation_wall_seconds=evaluation_wall, initial_gate=initial_gate, source_selected_step=start_step, stop_reason=reason)
    import gc, os
    from launch_with_memory_guard_TSC_resume_v3 import safe_sample
    before_release = safe_sample(os.getpid(), "before_exact_startup_compaction")
    training_row_compaction = compact_training_rows(tr, expected_rows=253798)
    cache_release = release_admission_caches();garbage_collected=gc.collect()
    after_release = safe_sample(os.getpid(), "after_exact_startup_compaction")
    write_json(event_path.with_name(event_path.stem+"_startup.json"), {
        "kind":"shared_heads_exact_startup_compaction_v1","passed":True,"run_id":args.run_id,
        "before_next_optimizer_update":True,"resume_step":resume_step,
        "authoritative_history_rows_retained":len(history),"training_row_compaction":training_row_compaction,
        "cache_release":cache_release,"gc_collected":garbage_collected,
        "host_memory_before":before_release,"host_memory_after":after_release,
        "original_config_BEST_source300_untouched":True,"reclaimed_memory_magnitude_claimed":False})
    del before_release,after_release
    print("completed exact cold reconstruction/compaction before resumed optimizer update",flush=True)
''' + worker[end:]
worker = worker.replace('loop_start = time.perf_counter(); last_evaluated_step = start_step',
    'loop_start = time.perf_counter(); last_evaluated_step = resume_step',1)
worker = worker.replace('stop_reason = "adaptation_screen_complete_review_required"',
    'stop_reason = "approved_shared_head_update_bound_complete_review_required"',1)
worker = worker.replace('print(f"BPE+shared12 B8x2 source{start_step}; phase0/{phase_limit}; full global horizon{configured_steps}", flush=True)',
    'print(f"BPE+shared12 EXACT B8x2 source{start_step}; resume{resume_step}; phase{adaptation.phase2_updates}/{phase_limit}; provisional global horizon{configured_steps}", flush=True)',1)
worker = worker.replace('for step in range(start_step + 1, phase_end + 1):','for step in range(resume_step + 1, phase_end + 1):',1)
worker = worker.replace('"external_exact_resume_supported": False, "further_unfreeze_allowed": False,\n        "row_Adam_clock":',
    '"external_exact_resume_supported": True, "further_unfreeze_allowed": False,\n        "exact_execution": execution, "full_unique_TRAIN_coverage": full_TRAIN_coverage(sampler.coverage()),\n        "row_Adam_clock":',1)
worker = worker.replace('    return p\n\n\ndef validate_arguments(args):',
    '    p.add_argument("--resume-shared-head-exact", type=Path, required=True)\n    p.add_argument("--shared-head-exact-source-receipt", type=Path, required=True)\n    p.add_argument("--shared-head-exact-runtime-proof", type=Path, required=True)\n    return p\n\n\ndef validate_arguments(args):',1)
worker = worker.replace('            "Initial adaptation only; no legacy or shallow exact-phase resume is implemented")',
    '            "Own shared exact mode only; no legacy or source reset")',1)
worker = worker.replace('    adaptation_preflight(args)\n    native_loss_identity(args)',
    '    require(args.smoke_updates is None, "Exact resume is not a fresh smoke/reset")\n    exact_preflight(args)\n    native_loss_identity(args)',1)
ast.parse(worker)
target=HERE/'train_lora_batched_tsc_bpe_shared_head_exact_v1.py'
with target.open('x',encoding='utf-8',newline='\n') as stream:stream.write(worker)
sync=[n.lineno for n in ast.walk(next(n for n in ast.parse(worker).body if isinstance(n,ast.FunctionDef) and n.name=='run'))
    if isinstance(n,ast.Call) and ast.unparse(n.func)=='torch.cuda.synchronize']
assert len(sync)==1
print({'worker_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'synchronize_line':sync[0]})
