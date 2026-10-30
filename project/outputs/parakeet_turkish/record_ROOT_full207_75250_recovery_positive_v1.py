"""Actual75250 exact recovery evidence; bounded current journal only."""
import argparse,json
from datetime import datetime,timezone
from atomic_io_retry_v1 import read_json,write_json
from record_ROOT_BPE_positive_v1 import latest_row
from tsc_three_epoch_transition_v1 import ROOT,SOURCE,STORAGE,RUN,END,record,require,encoder_lr


def main(args):
    loop=STORAGE/"runs"/(RUN+"_loop_telemetry_recovery_01");run=STORAGE/"runs"/RUN
    status=read_json(loop/"loop_status.json");identity=read_json(loop/"worker_identity.json")
    restored=read_json(loop/"exact_restore.json");lease=read_json(ROOT/"work/gpu_training_owner.json")
    receipt=STORAGE/"analysis/three_epoch_entry_v1/telemetry_stop_75250_v1/reviewed_recovery_receipt_v1.json"
    review=read_json(receipt);old=read_json(review["preserved"]["selection"]["path"])
    require(lease["owner"]=="/root" and lease["target_run_id"]==RUN and lease["actual_GPU_running"]
        and status["status"]=="running" and status["resume_step"]==75250
        and status["exact_checkpoint"]==review["preserved"]["latest_checkpoint"],"Actual owned reviewed restore required")
    require(restored["restored_step"]==restored["all207_Adam_clock"]==restored["sampler_clock"]==75250
        and restored["selected_step"]==old["selected_step"]
        and restored["retention_failures"]==old["retention_failures"],"No clock/winner/counter reset permitted")
    row,transaction=latest_row(run/"history_journal")
    require(row["step"]>=75260 and row["all207_Adam_clock"]==row["step"] and row["head_lr"]==1e-5
        and row["lora_lr"]==encoder_lr(row["step"])==3e-5 and row["encoder_LoRA_active"]
        and len(row["actual_TR_tdt_weights"])==14,"Fresh native clocks/rates/exposure required")
    updates=[]
    with (run/"stage_memory.jsonl").open(encoding="utf-8") as f:
        for line in f:
            v=json.loads(line)
            if v.get("stage")=="after_actual_full207_optimizer" and v.get("pid")==identity["pid"]:updates.append(v)
    require(len(updates)>=2 and [x["step"] for x in updates[:2]]==[75251,75252]
        and all(x["changed_head_tensors"]==15 and x["changed_encoder_LoRA_tensors"]==192
                and x["all207_Adam_clock"]==x["step"] and x["actual_native_backward_optimizer_update"] for x in updates[:2]),
        "Actual fresh native updates of all207 tensors required")
    target=STORAGE/"analysis/three_epoch_entry_v1/telemetry_stop_75250_v1/actual_recovery_native_positive_v1.json"
    require(not target.exists(),"Immutable recovery proof must be fresh")
    value=dict(kind="ROOT_full207_actual75250_recovery_native_positive_v1",passed=True,
        actual_owned_session=args.owned_session,launcher=status["owned_launcher_pid"],worker=identity["pid"],
        exact_restore=record(loop/"exact_restore.json"),reviewed_recovery=record(receipt),restored=restored,
        actual_first_two_native_updates=updates[:2],step=row["step"],fresh_native_updates=row["step"]-75250,
        continuous_total_updates=row["step"]-62250,declared_end_step=END,current_transaction=transaction,
        applied_rates_and_coverage={k:row[k] for k in ("step","all207_Adam_clock","head_lr","lora_lr","coverage","source_ID_sequence_sha256")},
        no_training_quality_gain_claimed=True,updated_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(target,value);ref=record(target)
    for name in ("experiment_plan.json","post_epoch_experiment_protocol.json"):
        p=SOURCE/name;j=read_json(p);j["current_full20775250_actual_recovery_positive"]=ref
        for k in ("actual_current_training","current_post_epoch_trial","current_training"):
            if isinstance(j.get(k),dict) and j[k].get("run_id")==RUN:
                j[k].update(status="actual75250_exact_recovery_native_positive_RUNNING",current_frontier=row["step"],
                    actual_owned_session=args.owned_session,native_launcher_pid=status["owned_launcher_pid"],worker_pid=identity["pid"],
                    new_optimizer_updates=row["step"]-62250,actual_GPU_running=True,all207_Adam_clock=row["step"],
                    applied_encoder_lr=3e-5,selected_qualified_BEST_step=old["selected_step"],selected_qualified_BEST_TR_WER=old["turkish_validation_wer"],
                    actual_recovery_native_positive=ref,last_complete_DEV_step=75250,last_complete_DEV_TR_WER=old["latest_gate"]["turkish_wer"],
                    last_complete_DEV_all_six_passed=old["latest_gate"]["retention_gate_passed"],updated_at_utc=value["updated_at_utc"])
        write_json(p,j)
    print(json.dumps(dict(passed=True,receipt=ref,step=row["step"],fresh_native_updates=row["step"]-75250)),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--owned-session",type=int,required=True);main(p.parse_args())
