"""Eight focused stdlib metadata bootstrap/closure cases; no process/ML/real state reads."""
import argparse,copy,json,sys,tempfile
from datetime import datetime,timezone,timedelta
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/"outputs/parakeet_turkish"
sys.path.insert(0,str(HERE))
import supervise_domain30_same_science_v3 as s

def main(args):
    checks=[]
    def check(value,name):
        assert value,name
        checks.append(name)
    def reject(call,name):
        try:call()
        except ValueError:checks.append(name)
        else:raise AssertionError(name)
    def save(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding="utf-8")
    original={"HERE":s.HERE,"source_state":s.source_state}
    old_completion=s.exact.verify_completion
    actual_order=s.controller_order_proof()
    pins=s.frozen_sources();frozen_before={n:s.sha(HERE/n) for n in pins}
    frozen_before.update({s.FIRST.name:s.sha(s.FIRST),s.HELPER.name:s.sha(s.HELPER),s.PRELOADER.name:s.sha(s.PRELOADER),"supervise_domain30_same_science_v2.py":s.sha(HERE/"supervise_domain30_same_science_v2.py")})
    try:
        with tempfile.TemporaryDirectory(dir=ROOT/"work/research",prefix="domain_exact_bootstrap_") as temporary:
            folder=Path(temporary);s.HERE=folder/"outputs";run_id="synthetic_owned_domain30"
            started=datetime.now(timezone.utc).isoformat();ready=folder/"first_ready.json";closure=folder/"closure.json"
            save(closure,{"synthetic":True});argv=["--run-id",run_id,"--native-backend-provenance",str(ready),"--source-closure",str(closure)]
            preload={"kind":"coherent_native_cuda12_9_preload_v1","module_sha256":pins["preload_nemo_linker_v1.py"],
                "actual_binding_linker":{"version":[12,9]},"cuda_initialized":False}
            fields={"memory_launcher_sha256":"launch_with_memory_guard_domain30_v1.py","cache_launcher_sha256":"launch_with_cache_reclaim_domain30_v1.py",
                "cache_helper_sha256":"cuda_cache_reclaim_v2.py","resource_helper_sha256":"resource_watchdog_v2.py",
                "resource_telemetry_sha256":"paired_windows_commit_telemetry_v1.py","frozen_worker_sha256":"train_lora_batched_domain30_v1.py","data_policy_sha256":"public_turkish_data_policy_v3.py"}
            def identity(pid,ppid,arguments):
                return {"kind":"owned_training_worker_identity_v1","expected_run_id":run_id,"pid":pid,"ppid":ppid,
                    "worker_arguments_sha256":s.domain.guards.argument_sha(arguments),"started_at_utc":started,
                    "process_creation_identity":{"available":True,"pid":pid,"creation_filetime_100ns":123,
                        "creation_unix_seconds":s.domain.guards.stamp(started).timestamp()},**{k:pins[n] for k,n in fields.items()}}
            save(ready,{"kind":"native_tdt_training_backend_v3","status":"ready","run_id":run_id,"pid":101,"ppid":100,
                "worker_arguments_sha256":s.domain.guards.argument_sha(argv),"trainer_sha256":pins["train_lora_batched_domain30_v1.py"],
                "adapter_sha256":pins["native_tdt_checkpointed_v3.py"],"launcher_sha256":pins["launch_native_training_domain30_v1.py"],
                "memory_launcher_sha256":pins["launch_with_memory_guard_domain30_v1.py"],"preload":preload,
                "declared_allocator_policy":{"cap_gb":8,"end_cache_gb":4,"native_inactive_threshold_bytes":750000000}})
            first_folder=folder/"first";save(first_folder/"identity.json",identity(101,100,argv))
            first={"kind":s.FIRST_KIND,"run_id":run_id,"status":"running","controller_sha256":s.FIRST_SHA,"owned_launcher_pid":100,
                "started_at_utc":started,"updated_at_utc":started,"source_sha256":pins,"worker_arguments":argv,
                "paths":{"native":str(ready),"identity":str(first_folder/"identity.json")}}
            exact_folder=folder/"manual_exact";exact_argv=argv+["--resume-exact",str(folder/"opaque_snapshot.pt")]
            save(exact_folder/"identity.json",identity(201,200,exact_argv))
            runtime={"kind":"domain_same_science_native_preload_resume_v1","status":"ready","run_id":run_id,"pid":201,"ppid":200,
                "worker_arguments_sha256":s.domain.guards.argument_sha(exact_argv),"launcher_sha256":s.PRELOADER_SHA,"source_sha256":pins,
                "preload":preload,"original_first_start_ready":s.record(ready),"scientific_config_identity_changed":False,
                "fresh_runtime_proof_separate_from_original_ready":True}
            runtime_path=exact_folder/"fresh_runtime.json";save(runtime_path,runtime)
            exact={"kind":s.EXACT_KIND,"run_id":run_id,"status":"running","controller_sha256":s.HELPER_SHA,"owned_launcher_pid":200,
                "started_at_utc":started,"updated_at_utc":started,"source_sha256":pins,"worker_arguments":exact_argv,
                "paths":{"identity":str(exact_folder/"identity.json"),"fresh_runtime":str(runtime_path)},
                "original_ready":s.record(ready),"new_preloader":{"sha256":s.PRELOADER_SHA}}
            science={"fixture_only":True,"training_arguments":{"epochs":1.,"steps":13009}}
            states={str(first_folder):first,str(exact_folder):exact};s.source_state=lambda path,rid:(states[str(path)],science)
            save(first_folder/"loop_status.json",first);save(exact_folder/"loop_status.json",exact)
            first_boot=s.bootstrap(first_folder,run_id,100,101)
            check(first_boot["initial_controller_kind"]==s.FIRST_KIND and first_boot["initial_controller_sha256"]==s.FIRST_SHA
                and not first_boot["initial_independent_OS_exit_observation"],"FIRST bootstrap remains valid and truthful metadata-only")
            initial=s.bootstrap(exact_folder,run_id,200,201)
            check(initial["initial_controller_kind"]==s.EXACT_KIND and initial["initial_controller_sha256"]==s.HELPER_SHA
                and initial["current_runtime_proof"]==s.record(runtime_path) and initial["original_domain_ready"]==s.record(ready),
                "manual EXACT bootstrap binds fresh PID/runtime separately from old ready")
            exact["kind"]="foreign";reject(lambda:s.bootstrap(exact_folder,run_id,200,201),"foreign initial kind rejected");exact["kind"]=s.EXACT_KIND
            exact["updated_at_utc"]=(datetime.now(timezone.utc)-timedelta(seconds=31)).isoformat()
            reject(lambda:s.bootstrap(exact_folder,run_id,200,201),"stale initial controller reading rejected");exact["updated_at_utc"]=started
            reject(lambda:s.bootstrap(exact_folder,run_id,200,101),"old first-start PID cannot masquerade as fresh exact worker")
            bad=copy.deepcopy(runtime);bad["pid"]=101;save(runtime_path,bad)
            reject(lambda:s.bootstrap(exact_folder,run_id,200,201),"stale native runtime proof rejected");save(runtime_path,runtime)
            s.exact.verify_completion=lambda plan,selection,owned:{"fixture_only":True,"actual_completion_contract_stubbed":True}
            latest=folder/"opaque_latest.bin";best=folder/"opaque_best.bin";latest.write_bytes(b"synthetic not checkpoint");best.write_bytes(b"synthetic best")
            selection_path=s.HERE/"runs"/run_id/"selection.json"
            selection={"run_id":run_id,"accepted":True,"heldout_tests_evaluated":False,"completed_steps":1600,"selected_step":1544,
                "latest_weights":str(latest),"weights":str(best),"stop_early_reason":"soft_max_wall_seconds","turkish_epochs_completed":.12}
            save(selection_path,selection)
            exact.update(status="train_window_ended",child_exit_code=0,completed_steps=1600,selected_step=1544,
                selection=s.record(selection_path),resource_review_required=False,stop_category="budget")
            save(exact_folder/"loop_status.json",exact)
            closed=s.closure(exact_folder,run_id,initial)
            check(closed["automatic_budget_handoff_allowed"] and closed["closure_observed_from_pinned_controller_evidence"]
                and closed["actual_supervisor_owned_controller_exit_code"] is None
                and not closed["supervisor_independent_initial_OS_exit_observation"],"initial manual EXACT budget closure verifies without fake OS exit")
            exact["controller_sha256"]=s.FIRST_SHA
            reject(lambda:s.closure(exact_folder,run_id,initial),"controller kind/SHA cannot change before initial closure")
    finally:
        for name,value in original.items():setattr(s,name,value)
        s.exact.verify_completion=old_completion
    assert all(s.sha(HERE/name)==value for name,value in frozen_before.items())
    assert not any(name in sys.modules for name in ("torch","nemo","transformers","numpy","soundfile"))
    assert len(checks)==8
    report={"schema":1,"kind":"domain_same_science_manual_exact_bootstrap_supervisor_CPU_verification_v3","passed":True,
        "check_count":len(checks),"checks":checks,"source":s.record(Path(s.__file__)),"fixture":s.record(Path(__file__)),
        "actual_source_order_proof":actual_order,"frozen_source_sha256_unchanged":frozen_before,
        "synthetic_source_state_and_completion_contracts_explicitly_stubbed":True,"actual_worker_metadata_validation_executed":True,
        "actual_process_or_model_GPU_launch":False,"actual_live_or_serialized_checkpoint_state_read":False,
        "created_at_utc":datetime.now(timezone.utc).isoformat()}
    target=args.output.resolve();assert target.is_relative_to(ROOT) and not target.exists()
    with target.open("x",encoding="utf-8") as stream:json.dump(report,stream,indent=2)
    print(json.dumps({"passed":True,"checks":len(checks),"source_sha256":report["source"]["sha256"],"report_sha256":s.sha(target),"report":str(target)}))

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True);main(parser.parse_args())
