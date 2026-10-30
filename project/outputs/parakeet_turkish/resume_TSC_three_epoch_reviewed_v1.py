"""Explicit reviewed exact recovery after the real75250 telemetry stop.

The frozen worker/controller and every guard remain unchanged. This entry only
admits the preserved CLOSED0 frontier after the competing bulk-I/O phase closed.
It never relabels that review-required closure as a normal budget boundary.
"""
from pathlib import Path
import json
import sys
import run_TSC_three_epoch_v1 as base
from atomic_io_retry_v1 import read_json, write_json
from tsc_three_epoch_transition_v1 import STORAGE, ROOT, SOURCE, RUN, END, UPDATES, PHASE, record, checked, require, canonical, preflight


def reviewed_prepare(args):
    base.require_storage()
    require(.1 <= args.window_hours <= 3 and 5 <= args.poll_seconds <= 15,
            "Original bounded window and polling policy required")
    review = read_json(args.reviewed_recovery)
    require(review["passed"] and review["kind"] == "ROOT75250_reviewed_IO_workload_change_v1"
            and review["actual_owned_exit"] == {"session":66455,"exit_code":0,"tool_chunk":"fe378a"}
            and review["source_step"] == 75250 and review["configured_end_step"] == END,
            "Actual partial closure and purpose-bound reviewed recovery required")
    require(record(args.previous_status) == review["previous_status"], "Original closed status changed")
    previous = read_json(args.previous_status)
    require(previous["run_id"] == args.run_id == RUN and previous["status"] == "train_window_ended"
            and previous["stop_category"] == "review_required"
            and previous["cooperative_stop_reason"] == "actual_worker_self_memory_stale"
            and previous["child_exit_code"] == 0 and not previous["GPU_running"]
            and previous["completed_step"] == 75250 and previous["actual_completion"]["passed"],
            "Never fake a budget or final closure")
    lease = read_json(ROOT/"work/gpu_training_owner.json")
    require(lease["owner"] == "/root" and lease["target_run_id"] == RUN
            and not lease["actual_GPU_running"] and lease["actual_child_exit_code"] == 0,
            "ROOT sole owned child must be actually closed")
    for ref in review["preserved"].values(): checked(ref)
    require(record(args.closed_latest) == review["preserved"]["latest_checkpoint"], "Sealed exact checkpoint required")
    prior = previous["actual_completion"]["latest_checkpoint"]
    require(all(review["preserved"]["latest_checkpoint"][k] == prior[k] for k in ("sha256","bytes")),
            "Preservation must not alter the75250 state")
    stage = read_json(review["closed_CPU_bulk_stage"]["path"])
    require(stage["actual_CLOSED0"] and stage["phase"] == "completed_candidates_only"
            and stage["completed_bytes"] == 58770288640, "Competing bulk download must actually be closed")
    require(review["fresh_E_telemetry_probe"]["passed"] and review["guards_unchanged"],
            "Fresh telemetry and unchanged guards required")
    loop = base.output_path(args.output_dir)
    require(not loop.exists(), "Fresh recovery execution directory required")
    paths = {key:loop/name for key,name in (("native","native_backend.json"),("identity","worker_identity.json"),
        ("completion","native_completion.json"),("live","owned_live_memory.json"),("memory","memory_evidence.json"),
        ("stop","stop.txt"),("log","worker.log"))}
    values = base.worker_namespace(args, paths)
    context = preflight(values)
    pins = base.pins(context)
    require(previous["source_sha256"] == pins, "Scientific/worker/controller sources must stay byte-identical")
    argv = []
    for key,value in vars(values).items():
        if value is not None: argv.extend(("--"+key.replace("_","-"),str(value)))
    for key in ("run_config","initial_source350"): checked(previous["actual_completion"][key])
    return {"kind":base.KIND,"phase":PHASE,"run_id":RUN,"decoder_arm":"original8199",
        "source_receipt":context["receipt_record"],"source_checkpoint":context["source_checkpoint"],
        "exact_checkpoint":review["preserved"]["latest_checkpoint"],"source_sha256":pins,
        "worker_arguments":argv,"worker_arguments_sha256":canonical(argv),"source_step":context["source_step"],
        "resume_step":75250,"configured_end_step":END,"declared_additional_updates":UPDATES,
        "paths":{k:str(v) for k,v in paths.items()},"run_directory":str(values.run_directory),
        "previous_status":str(args.previous_status),"previous_observed_peak":previous["peak_observed_owned_private_bytes"],
        "reference_source_policy_private_peak":context["reference_source_policy_private_peak"],
        "command":[str(ROOT/"work/nemo-native-env-v2/Scripts/python.exe"),"-X","utf8","-B",str(SOURCE/"run_TSC_three_epoch_v1.py"),
                   "--worker-plan",str(loop/"plan.json"),"--gpu-window-authorized"],
        "window_hours":args.window_hours,"actual_runtime_mode":previous["actual_runtime_mode"],
        "retention_stop_policy":"monitor_only","full_epoch_claimed":True,"status":"prepared","GPU_launched":False,
        "created_at_utc":base.now(),"reviewed_recovery":record(args.reviewed_recovery),
        "actual_recovery_driver":record(__file__),"scientific_sources_and_all_resource_guards_unchanged":True}


def main():
    parser=base.parser(); parser.add_argument("--reviewed-recovery",type=Path,required=True)
    args=parser.parse_args(); require(not args.worker_plan,"Reviewed driver never replaces the frozen native dispatcher")
    plan=reviewed_prepare(args)
    if args.validate_only:
        if args.preflight_output:
            require(not args.preflight_output.exists(),"Never overwrite a preflight")
            write_json(args.preflight_output,{**plan,"passed":True,"GPU_launched":False,"Torch_imported":"torch" in sys.modules})
        print(json.dumps({"passed":True,"GPU_launched":False,"resume_step":75250,"end_step":END}),flush=True); return
    require(args.gpu_window_authorized,"Existing explicit ROOT sole GPU authorization required")
    original=args.output_dir; number=1
    while True:
        state=base.run_window(plan,args)
        if state["stop_category"]!="budget": return
        args.previous_status=args.output_dir/"loop_status.json"
        args.closed_latest=Path(state["actual_completion"]["latest_checkpoint"]["path"])
        number+=1; args.output_dir=original.with_name(original.name+"_window_"+str(number))
        plan=base.prepare(args)


if __name__=="__main__": main()
