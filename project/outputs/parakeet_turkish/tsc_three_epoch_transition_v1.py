"""Explicit three-pass extension of the selected original8199/full207 recipe."""
from __future__ import annotations
from copy import deepcopy
import json
import math
from pathlib import Path
from tsc_bpe_full_decoder_transition_v1 import (
    SOURCE, ROOT, STORAGE, require, record, checked, canonical, validate_scores,
    source_context as selected_context, validate_replay_pool, audit_parent_state,
)

PHASE = "TSC_original8199_full207_three_epoch_extension_v1"
ORIGIN, SOURCE_STEP, POOL, END = 40750, 62250, 352768, 116344
UPDATES = END - SOURCE_STEP
MILESTONES = (65948, 91146, 116344)
PROOF_NAME = "full207_three_epoch_initial_source350.json"
RUN = "adaptive_20261015_TR_full207_three_epochs_01"


def encoder_lr(step):
    require(type(step) is int and SOURCE_STEP <= step <= END, "Declared continuous LR clock required")
    if step >= 65948: return 3e-5
    fraction = (step - ORIGIN - 1) / (25198 - 1)
    return 3e-5 + (1e-4 - 3e-5) * .5 * (1 + math.cos(math.pi * fraction))


def declaration():
    return dict(kind=PHASE, original8199_and_blank8192=True, all207_active=True,
        accepted_unique_TR_clips=POOL, accepted_unique_TR_hours=675.0964,
        target_complete_permutation_passes_since_original40750=3,
        source_step=SOURCE_STEP, end_step=END, additional_updates=UPDATES,
        epoch_completion_steps=list(MILESTONES), source_TR_draws=301000,
        source_TR_unique=301000, target_total_TR_draws=14*(END-ORIGIN),
        declared_draws_beyond_three_pools=14*(END-ORIGIN)-3*POOL,
        selected_source_remaining_first_pass_updates=65948-SOURCE_STEP,
        no_sampler_RNG_permutation_seen_or_clock_reset=True,
        original_completed_and_failed_branches_preserved=True,
        encoder_schedule="Continue original cosine through65948, then hold its existing3e-5 floor",
        encoder_LR_at_source=encoder_lr(SOURCE_STEP), head_LR=1e-5,
        native=4, physical_batch=8, accumulation=2, TR_PT_oldfive=[14,1,1],
        PT_preservation=[40,2,20], oldfive_preservation_multiplier=5/3,
        TR_population_divisor=26.642061797177284, six_glyph_gain=6,
        retention="monitor and strict champion selection; never early-stop a declared pass",
        not_a_10000h_data_admission=True, synthetic_or_unaccepted_audio_added=False)


def preflight(args):
    path = Path(args.training_source_receipt).resolve()
    require(path.is_relative_to(STORAGE.resolve()), "New science/source receipt must be on E:")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    require(receipt["kind"] == PHASE and receipt["target_run_id"] == args.run_id == RUN
            and receipt["declaration"] == declaration(), "Only the explicit selected-recipe three-pass trial")
    review = json.loads(checked(receipt["completed_BPE_pair"]).read_text(encoding="utf-8"))
    require(review["passed"] and review["matched_2000_ID_orders_and_sampling_geometry"]
            and review["candidate"]["selected_step"] == review["control"]["selected_step"] == SOURCE_STEP,
            "Completed matched BPE family must retain the original coherent champion")
    context = selected_context(checked(receipt["original_selected_source_basis"]))
    require(context["source_step"] == SOURCE_STEP and context["source_data_arm"] == "expanded"
            and context["TR_pool_clips"] == POOL
            and context["source_checkpoint"] == receipt["source_checkpoint"],
            "Exact original full207 qualified source62250 and accepted675h pool required")
    require(args.training_updates == UPDATES and args.batch_size == 8 and args.accumulation == 2
            and args.shared_lr == 1e-5 and args.native_max_joint_bytes == 1700000000,
            "Original recipe/layout/native bound required")
    require(Path(args.run_directory).resolve() == STORAGE/"runs"/RUN, "All new run state must be on E:")
    pool = json.loads(checked(context["source_training_pool"]).read_text(encoding="utf-8"))
    context["source_replay_counts"] = validate_replay_pool(pool["replay"])
    context.update(receipt=receipt, receipt_record=record(path), configured_end_step=END,
        training_updates=UPDATES, multi_epoch_declaration=declaration())
    return context


def proof_path(args): return Path(args.run_directory)/PROOF_NAME


def compare_source_development(model, processor, args, context, state, evaluate, torch):
    from tsc_bpe_acquisition_transition_v1 import compare_development
    from atomic_io_retry_v1 import write_json
    import random
    require(not proof_path(args).exists(), "Never replace actual initial350 proof")
    rng = random.getstate(), torch.get_rng_state().clone(), [x.clone() for x in torch.cuda.get_rng_state_all()]
    modes = [(m,m.training) for m in model.modules()]
    try:
        scores = evaluate(model,processor,args.run_id+"_initial_"+str(SOURCE_STEP))
        old, passed = validate_scores(context["source_summary"])
        new, actual_passed = validate_scores(scores)
        require(passed and actual_passed and old == new, "Actual selected source350 must reproduce all seven counts")
        folder = STORAGE/"results"/(args.run_id+"_initial_"+str(SOURCE_STEP))
        for lang in ("tr","en","es","it","fr","de","pt"):
            before = [json.loads(x) for x in checked(context["source_DEV"][lang]).read_text(encoding="utf-8").splitlines() if x]
            after = [json.loads(x) for x in (folder/("validation_"+lang+".jsonl")).read_text(encoding="utf-8").splitlines() if x]
            compare_development(before,after,context["source_summary"]["validation_"+lang]["asr"],scores["validation_"+lang]["asr"],lang)
    finally:
        random.setstate(rng[0]); torch.set_rng_state(rng[1]); torch.cuda.set_rng_state_all(rng[2])
        for m,mode in modes:m.training=mode
    proof = dict(kind="actual_full207_three_epoch_initial350_v1", passed=True,
        source_checkpoint=context["source_checkpoint"], source_step=SOURCE_STEP,
        all350_raw_text_token_and_duration_paths_equal=True, counts=new, all_six_passed=actual_passed,
        RNG_and_module_modes_preserved=True, sampler_drawn=False)
    write_json(proof_path(args),proof)
    return proof
