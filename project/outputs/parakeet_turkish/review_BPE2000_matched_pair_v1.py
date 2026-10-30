"""Review only the 2,000 new journal rows; never deserialize the old history."""
from __future__ import annotations
import json
from pathlib import Path
from atomic_io_retry_v1 import read_json, write_json
from append_only_history_v1 import HistoryJournal
from tsc_bpe_full_decoder_transition_v1 import STORAGE, record, checked, require, validate_scores, canonical

ENTRY = STORAGE / "analysis/BPE_full_decoder_entry_v1"
START, END = 62250, 64250


def screen_rows(run):
    directory = STORAGE / "runs" / run / "history_journal"
    head = read_json(directory / "HEAD.json")
    # Use the frozen frame verifier without constructing an all-history index.
    view = object.__new__(HistoryJournal)
    view.directory = directory
    view.max_record_bytes = head["max_record_bytes"]
    view.amendment_keys = tuple(head["amendment_keys"])
    ref, chain = head["last_transaction"], []
    while ref:
        header = view._header(ref)
        if header["previous_max_step"] < START:
            break  # The inherited seed is deliberately not scanned/deserialized.
        require(ref["bytes"] <= 40 * 1024 * 1024, "Bounded post-source transaction required")
        chain.append((ref, header))
        ref = header["previous"]
    rows, branches = {}, []
    for ref, header in reversed(chain):
        cutoff = header.get("truncate_to")
        if cutoff is not None:
            rows = {s: r for s, r in rows.items() if s <= cutoff}
            branches.append(header["orphan_branch"])
        def admit(actual_header, step, offset, length, digest, row):
            require(actual_header == header, "Verified transaction header changed")
            if START < step <= END:
                rows[step] = row
        view._scan(ref, admit)
    require(sorted(rows) == list(range(START + 1, END + 1)) and head["max_step"] == END,
            "Exactly the complete active 2000-step trajectory is required")
    return rows, dict(head=record(directory/"HEAD.json"), bounded_transactions=len(chain),
                      new_rows_only=2000, inherited_seed_deserialized=False, preserved_branches=branches)


def arm(run):
    closure_ref = record(ENTRY/(run+"_actual_CLOSED64250_v1.json"))
    closed = read_json(closure_ref["path"])
    require(closed["passed"] and closed["observed_tool_exit_code"] == 0, "Actual owned CLOSED0 required")
    for key in ("native_completion", "worker_completion", "final_DEV", "selection", "run_config"):
        checked(closed["actual_completion"][key])
    selection = read_json(closed["actual_completion"]["selection"]["path"])
    config = read_json(closed["actual_completion"]["run_config"]["path"])
    rows, evidence = screen_rows(run)
    curve = []
    for step in range(START+250, END+1, 250):
        folder = STORAGE / "results" / (run+"_step_"+str(step))
        summary = read_json(folder/"summary.json")
        counts, passed = validate_scores(summary)
        diag = read_json(folder/"decoder_diagnostics.json")
        require(rows[step]["validation"]["turkish_wer"] == counts["tr"]/825,
                "Published DEV and complete actual summaries disagree")
        curve.append(dict(step=step, updates=step-START, official_counts=counts,
            official_TR_WER=counts["tr"]/825, all_six_passed=passed,
            spoken_TR=diag["spoken_form_TR"], official_TR_SDI=diag["official_TR_SDI"],
            new_TR_token_instances=diag["piece_emissions"]["tr"]["new_BPE_emitted_tokens"],
            new_TR_token_types=diag["piece_emissions"]["tr"]["new_BPE_unique_token_types"],
            summary=record(folder/"summary.json"), diagnostic=record(folder/"decoder_diagnostics.json")))
    return dict(run_id=run, closure=closure_ref, source_checkpoint=config["config_identity"]["source_checkpoint"],
        source350=closed["actual_completion"]["initial_source350"], curve=curve,
        selected_step=selection["selected_step"], selected_WER=selection["turkish_validation_wer"],
        journal_evidence=evidence), rows


def main():
    b, br = arm("adaptive_20261015_TR_BPE512_full_decoder_02")
    c, cr = arm("adaptive_20261015_TR_original8199_full_decoder_01")
    require(b["source_checkpoint"] == c["source_checkpoint"], "Same coherent selected207 source required")
    for step in range(START+1, END+1):
        for key in ("source_ID_sequence_sha256", "sampling_target_tokens", "coverage",
                    "language_audio_seen_seconds", "audio_seen_seconds"):
            require(br[step][key] == cr[step][key], "Matched ID/exposure mismatch at " + str(step)+": "+key)
        for row in (br[step], cr[step]):
            require(row["mature_Adam_clock"] == step and row["head_lr"] == 1e-5
                    and row["lora_lr"] == 0 and row["encoder_LoRA_active"] is False,
                    "Actual applied rates/clocks differ")
        require(br[step]["fresh_row_Adam_clock"] == step-START and br[step]["new_rows_lr"] == 1e-4
                and cr[step]["fresh_row_Adam_clock"] is None and cr[step]["new_rows_lr"] is None,
                "Actual fresh-row policy differs")
    exposure_digest = canonical([br[s]["source_ID_sequence_sha256"] for s in sorted(br)])
    result = dict(kind="actual_completed_BPE2000_matched_quality_comparison_v1", passed=True,
        candidate=b, control=c, matched_2000_ID_orders_and_sampling_geometry=True,
        matched_all_per_update_coverage_and_language_audio=True, matched_ID_order_digest=exposure_digest,
        actual_shared_LR_1e5_encoder_zero_both=True, new_rows_LR_1e4_BPE_only=True,
        source=dict(step=START, official_TR_errors=259, official_TR_references=825, official_TR_WER=259/825,
                    spoken_TR_errors=228, spoken_TR_references=851, spoken_TR_WER=228/851),
        final_official_BPE_minus_control_absolute_pp=100*(369-261)/825,
        final_spoken_BPE_minus_control_absolute_pp=100*(355-232)/851,
        conclusion="This BPE512 decoder-only adaptation recipe regressed substantially against the matched8199 control. Neither arm displaced the coherent original selected source; keep original8199/all207 source62250 for the next quality trial.",
        limits="Fixed50 Turkish DEV clips; single paired screen with a preserved/replayed crash suffix in candidate. Numerical variability limits tiny-gain inference. This does not establish that every BPE or longer adaptation strategy fails.",
        next_science="Authorized roughly3 full accepted-pool epochs with original8199/full207 adaptation, exact source62250 state and explicitly declared continuous sampler/LR extension; keep all six gates and TEST frozen. No blind extension of failed BPE.",
        TEST_used=False, official_scorer_or_gates_changed=False, overall_quality_goal_complete=False)
    require(b["curve"][-1]["official_counts"]["tr"] == 369 and c["curve"][-1]["official_counts"]["tr"] == 261
            and b["selected_step"] == c["selected_step"] == START, "Measured final/source selection changed")
    target = ENTRY / "BPE2000_completed_matched_quality_comparison_v1.json"
    require(not target.exists(), "Immutable paired comparison must never be replaced")
    write_json(target, result)
    print(json.dumps(dict(passed=True, receipt=record(target), same_ID_exposure=True,
        official_final_errors=dict(BPE=369, control=261), source_retained=START,
        control_curve=[(x["updates"],x["official_counts"]["tr"],x["all_six_passed"]) for x in c["curve"]])))


if __name__ == "__main__": main()
