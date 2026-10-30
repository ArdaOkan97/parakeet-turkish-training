"""Bounded actual native evidence for ROOT's selected full207 three-pass run."""
import argparse
from datetime import datetime,timezone
from atomic_io_retry_v1 import read_json,write_json
from record_ROOT_BPE_positive_v1 import latest_row
from tsc_three_epoch_transition_v1 import ROOT,SOURCE,STORAGE,RUN,SOURCE_STEP,END,record,require,encoder_lr


def main(args):
    run=STORAGE/"runs"/RUN
    lease=read_json(ROOT/"work/gpu_training_owner.json")
    loop=STORAGE/"runs"/(RUN+"_loop_01")
    status=read_json(loop/"loop_status.json")
    native=read_json(loop/"worker_identity.json")
    config=read_json(run/"run_config.json")
    restored=read_json(loop/"exact_restore.json")
    source350=run/"full207_three_epoch_initial_source350.json"
    require(lease["owner"]=="/root" and lease["target_run_id"]==RUN and lease["actual_GPU_running"]
        and status["status"]=="running" and read_json(source350)["passed"]
        and config["source207_audit"]["source_Adam"]["source_Adam_states_checked"]==207
        and restored["restored_step"]==restored["all207_Adam_clock"]==restored["sampler_clock"]==SOURCE_STEP
        and restored["retention_failures"]==0 and restored["selected_step"]==SOURCE_STEP,
        "Actual sole native full207/source350/unchanged clock-counter restore required")
    row,transaction=latest_row(run/"history_journal")
    require(row["step"]>SOURCE_STEP and row["all207_Adam_clock"]==row["step"]
        and row["head_lr"]==1e-5 and row["lora_lr"]==encoder_lr(row["step"])
        and row["encoder_LoRA_active"] and row["TR_tdt_population_mean_U"]==26.642061797177284
        and len(row["actual_TR_tdt_weights"])==14,"Actual applied LR/clock/TR weights differ")
    stages=[read_json_line(x) for x in (run/"stage_memory.jsonl").read_text().splitlines()]
    updates=[x for x in stages if x["stage"]=="after_actual_full207_optimizer" and x["pid"]==native["pid"]]
    require(len(updates)>=2 and updates[0]["step"]==62251 and updates[1]["step"]==62252
        and all(x["all207_Adam_clock"]==x["step"] and x["actual_native_backward_optimizer_update"]
                and x["changed_head_tensors"]>0 and x["changed_encoder_LoRA_tensors"]>0 for x in updates[:2]),
        "Actual native decoder and encoder-adapter changes must be observed")
    target=STORAGE/"analysis/three_epoch_entry_v1/actual_native_positive_v1.json"
    require(not target.exists(),"Never replace immutable first-positive proof")
    result=dict(kind="ROOT_full207_three_epoch_actual_native_positive_v1",passed=True,
        owner="/root",run_id=RUN,actual_owned_session=args.owned_session,
        actual_native_launcher_pid=status["owned_launcher_pid"],actual_native_worker_pid=native["pid"],
        step=row["step"],positive_updates=row["step"]-SOURCE_STEP,end_step=END,
        source350=record(source350),source207_audit=config["source207_audit"],
        actual_restore=record(loop/"exact_restore.json"),first_two_actual_updates=updates[:2],
        actual_applied_LR_coverage={k:row[k] for k in ("step","all207_Adam_clock","head_lr","lora_lr",
            "TR_tdt_population_mean_U","actual_TR_tdt_weights","coverage","source_ID_sequence_sha256")},
        current_transaction=transaction,no_training_quality_gain_claimed=True,
        updated_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(target,result);ref=record(target)
    current=dict(run_id=RUN,owner="/root",family="original8199_full207_three_epoch_extension",status="actual_native_positive_RUNNING",
        source_step=SOURCE_STEP,current_frontier=row["step"],configured_end_step=END,
        new_optimizer_updates=row["step"]-SOURCE_STEP,actual_owned_session=args.owned_session,
        native_launcher_pid=status["owned_launcher_pid"],worker_pid=native["pid"],actual_GPU_running=True,
        accepted_TR_clips=352768,accepted_TR_hours=675.0964,target_complete_passes=3,
        encoder_192_adapters_active=True,all207_Adam_clock=row["step"],applied_encoder_lr=row["lora_lr"],
        applied_head_lr=1e-5,source350_pending=False,own207_restore_pending=False,retention_early_stop_active=False,
        selected_qualified_BEST_step=SOURCE_STEP,selected_qualified_BEST_TR_WER=259/825,
        last_complete_DEV_step=SOURCE_STEP,last_complete_DEV_TR_WER=259/825,last_complete_DEV_all_six_passed=True,
        actual_native_positive=ref,authoritative_live_state=str(STORAGE/"experiment_queue_state.json"),
        updated_at_utc=result["updated_at_utc"],overall_quality_goal_complete=False)
    pair=record(STORAGE/"analysis/BPE_full_decoder_entry_v1/BPE2000_completed_matched_quality_comparison_v1.json")
    for name in ("experiment_plan.json","post_epoch_experiment_protocol.json"):
        path=SOURCE/name;d=read_json(path)
        if name=="experiment_plan.json":
            for key in ("actual_current_training","current_post_epoch_trial","current_training"):d[key]=current
        d["current_GPU_owner"]="/root";d["completed_BPE_matched_pair"]=pair
        d["current_full207_three_epoch_actual_native_positive"]=ref
        d["current_authoritative_E_queue"]=str(STORAGE/"experiment_queue_state.json")
        d["next_quality_family_after_current_cycle"]="Authorized selected original8199/full207 three-pass continuation is now running"
        d["next_authorized_BPE_full_decoder_family"].update(status="both_actual_CLOSED0_pair_reviewed_no_adoption",pair_comparison_complete=True,completed_pair=pair)
        for key in ("ordered_next_work","ordered_experiments"):
            for item in d.get(key,[]):
                text=" ".join(v for v in item.values() if isinstance(v,str))
                if "BPE512" in text and "full" in text:item.update(status="completed_actual_CLOSED0_no_qualifying_improvement",completed_pair=pair,pair_comparison_complete=True)
                if item.get("backlog_id")=="ASR-15.6" or item.get("id")=="human_selected_recipe_three_epochs":
                    item.update(status="actual_native_positive_RUNNING",actual_native_positive=ref)
        write_json(path,d)
    path=SOURCE/"human_data_expansion_trial_queue_v1.json";d=read_json(path)
    def update(value):
        if isinstance(value,dict):
            if value.get("id")=="human_selected_recipe_three_epochs":value.update(status="actual_native_positive_RUNNING",actual_native_positive=ref,source_receipt=record(STORAGE/"analysis/three_epoch_entry_v1/source_receipt_v1.json"))
            for child in value.values():update(child)
        elif isinstance(value,list):
            for child in value:update(child)
    update(d);write_json(path,d)
    write_json(SOURCE/"experiment_queue_state.json",dict(kind="E_authoritative_full207_queue_pointer_v1",owner="/root",
        current_run_id=RUN,status="running",published_step=row["step"],configured_steps=END,
        authoritative_state_path=str(STORAGE/"experiment_queue_state.json"),actual_native_positive=ref,
        completed_BPE_pair=pair,updated_at_utc=result["updated_at_utc"],objective_complete=False))
    print(__import__('json').dumps(dict(passed=True,receipt=ref,step=row["step"],new_positive_updates=row["step"]-SOURCE_STEP)))


def read_json_line(value):return __import__('json').loads(value)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--owned-session",type=int,required=True);main(p.parse_args())
