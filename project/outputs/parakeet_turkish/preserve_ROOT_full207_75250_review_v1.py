"""Preserve the actual telemetry-stop closure and test current E: publication."""
import json,os,shutil,time
from pathlib import Path
from atomic_io_retry_v1 import read_json,write_json
from tsc_three_epoch_transition_v1 import STORAGE,ROOT,RUN,END,record,checked,require
import launch_with_memory_guard_TSC_resume_v3 as memory


def main():
    policy=read_json(STORAGE/"storage_config.json")
    require(policy["allow_fallback_outside_root"] is False and os.environ["GPU_TRAINING_STORAGE_ROOT"]==str(STORAGE),"E-only environment required")
    loop=STORAGE/"runs"/(RUN+"_loop_01_window_3")
    state=read_json(loop/"loop_status.json");lease=read_json(ROOT/"work/gpu_training_owner.json")
    require(lease["owner"]=="/root" and not lease["actual_GPU_running"] and lease["actual_child_exit_code"]==0,"Owned actual closure required")
    require(state["completed_step"]==75250 and state["child_exit_code"]==0 and state["stop_category"]=="review_required"
            and state["cooperative_stop_reason"]=="actual_worker_self_memory_stale" and state["actual_completion"]["passed"],"Exact actual partial closure required")
    proof=state["actual_completion"]
    for ref in proof.values():
        if isinstance(ref,dict) and "sha256" in ref:checked(ref)
    worker=read_json(proof["worker_completion"]["path"])
    require(worker["final350_complete"] and worker["all207_active"] and worker["all207_Adam_clock"]==75250
            and worker["sources_unchanged"] and worker["training_updates"]==13000,"Actual complete75250/207/DEV required")
    report=STORAGE/"analysis/three_epoch_entry_v1/telemetry_stop_75250_v1"
    sealed=STORAGE/"checkpoints/closed_sources/full207_75250_review_v1"
    require(not sealed.exists() and not report.exists(),"Never replace preserved closure evidence")
    sealed.mkdir(parents=True);report.mkdir(parents=True)
    live=report/"fresh_E_probe_live.json"
    sampler=memory.LiveTelemetrySampler(os.getpid(),lambda value:write_json(live,value,retry_seconds=1.))
    sampler.start();started=time.monotonic();preserved={}
    try:
        for name in ("latest_checkpoint","best_checkpoint","selection","run_config","worker_completion","initial_source350","final_DEV","native_completion"):
            ref=proof[name];src=Path(ref["path"]);target=sealed/(name+src.suffix)
            shutil.copyfile(src,target);observed=record(target)
            require(all(observed[k]==ref[k] for k in ("sha256","bytes")),"Sealed artifact changed: "+name)
            preserved[name]=observed
        while time.monotonic()-started<22:time.sleep(.5)
    finally:sampler.stop_and_join()
    observed=sampler.evidence();samples=[v for v in observed["samples"] if "observed_monotonic" in v]
    gaps=[b["observed_monotonic"]-a["observed_monotonic"] for a,b in zip(samples,samples[1:])]
    probe=dict(passed=len(samples)>=4 and not observed["failed_samples"] and not observed["live_publish_failures"]
               and max(gaps,default=100)<15,successful_samples=observed["successful_samples"],
               failed_samples=observed["failed_samples"],publish_failures=observed["live_publish_failures"],
               maximum_observed_gap_seconds=max(gaps,default=None),elapsed_seconds=time.monotonic()-started,
               max_commit_fraction=max(s["system_memory"]["commit_fraction"] for s in samples),
               minimum_headroom_bytes=min(s["system_memory"]["commit_available_bytes"] for s in samples),
               evidence=observed,scope="Actual ROOT CPU-self sampler writing E: while copying/hash-verifying real closed artifacts; no GPU/model operations")
    write_json(report/"fresh_E_telemetry_probe.json",probe);require(probe["passed"],"Fresh publication probe failed; do not retry GPU")
    old=read_json(loop/"memory_evidence.json")["all_phase_telemetry"]
    pairs=[dict(from_utc=a["observed_at_utc"],to_utc=b["observed_at_utc"],gap_seconds=b["observed_monotonic"]-a["observed_monotonic"])
           for a,b in zip(old["samples"],old["samples"][1:]) if b["observed_monotonic"]-a["observed_monotonic"]>15]
    stage=STORAGE/"analysis/CPU_validation_v2/additional_stage_v2_status.json";stage_value=read_json(stage)
    require(stage_value["actual_CLOSED0"] and stage_value["completed_bytes"]==58770288640,"Actual completed bulk I/O required")
    value=dict(kind="ROOT75250_reviewed_IO_workload_change_v1",passed=True,source_step=75250,configured_end_step=END,
        actual_owned_exit=dict(session=66455,exit_code=0,tool_chunk="fe378a"),previous_status=record(loop/"loop_status.json"),
        actual_partial_closure_not_final_goal=True,stop_reason=state["cooperative_stop_reason"],preserved=preserved,
        actual_sampler_gaps=pairs,sampler_failed_samples=old["failed_samples"],sampler_publish_failures=old["live_publish_failures"],
        cause="Unresolved telemetry publication/scheduling delay; concurrent E bulk staging and one CPU decode delay observed, causality not proven",
        changed_recovery_conditions="All8 bulk staging downloads and both metadata exclusion scans are actually CLOSED0; no new heavy E: I/O phase while GPU recovery is supervised; fresh E: telemetry probe passed",
        closed_CPU_bulk_stage=record(stage),fresh_E_telemetry_probe=probe,guards_unchanged=True,
        no_optimizer_RNG_history_sampler_counter_or_grace_reset=True,no_native_worker_scientific_source_edits=True,
        no_stale_limit_increase=True,no_resource_guard_relaxation=True,updated_at_utc=memory.timestamp())
    target=report/"reviewed_recovery_receipt_v1.json";write_json(target,value)
    print(json.dumps(dict(passed=True,receipt=record(target),partial_closed_step=75250,probe_samples=len(samples),maximum_probe_gap=max(gaps))),flush=True)


if __name__=="__main__":main()
