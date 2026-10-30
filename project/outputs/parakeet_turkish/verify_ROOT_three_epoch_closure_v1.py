"""Actual owned exit and complete third-pass proof, CPU metadata only."""
import argparse,json
from atomic_io_retry_v1 import read_json,write_json
from tsc_three_epoch_transition_v1 import ROOT,STORAGE,RUN,END,UPDATES,record,checked,require


def main(args):
    require(args.observed_exit==0,"Actual owned exit0 required")
    lease=read_json(ROOT/"work/gpu_training_owner.json")
    require(lease["owner"]=="/root" and lease["target_run_id"]==RUN and not lease["actual_GPU_running"]
        and lease["actual_child_exit_code"]==0,"Actual sole child closure required")
    path=lease["actual_controller_loop"]+"/loop_status.json"
    state=read_json(path)
    require(state["run_id"]==RUN and state["status"]=="train_window_ended"
        and state["completed_step"]==END and state["child_exit_code"]==0
        and not state["GPU_running"] and state["stop_category"]=="configured_steps_complete"
        and state["actual_completion"]["passed"],"Actual complete three-pass controller proof required")
    proof=state["actual_completion"]
    for ref in proof.values():
        if isinstance(ref,dict) and "sha256" in ref:checked(ref)
    worker=read_json(proof["worker_completion"]["path"])
    require(worker["all207_Adam_clock"]==END and worker["all207_active"]
        and worker["training_updates"]==UPDATES and worker["final350_complete"]
        and worker["coverage"]["three_epoch_extension"]["complete_permutation_cycles"]==3
        and worker["coverage"]["three_epoch_extension"]["total_TR_visits"]==1058316,
        "Actual all207/three complete pool cycles and full end DEV required")
    target=STORAGE/"analysis/three_epoch_entry_v1/actual_CLOSED116344_v1.json"
    require(not target.exists(),"Never replace an actual full-run closure receipt")
    result=dict(passed=True,kind="ROOT_full207_three_epoch_actual_CLOSED0_v1",run_id=RUN,
        observed_owned_session=args.observed_session,observed_exit=args.observed_exit,
        observed_tool_chunk=args.observed_tool_chunk,controller_status=record(path),actual_completion=proof,
        overall_quality_goal_complete=False)
    write_json(target,result);print(json.dumps(dict(passed=True,receipt=record(target))))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--observed-session",type=int,required=True)
    p.add_argument("--observed-exit",type=int,required=True);p.add_argument("--observed-tool-chunk",required=True)
    main(p.parse_args())
