"""Focused real worker_proof FIRST/EXACT metadata checks; source/closure state is synthetic."""
import argparse
import ast
import copy
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "outputs/parakeet_turkish"
sys.path.insert(0, str(HERE))
import supervise_TSC_same_science_v4 as s


def main(args):
    output = s.scoped(args.output); assert not output.exists()
    checks = []
    def check(value, name):
        assert value, name; checks.append(name)
    def rejects(call, name):
        try: call()
        except ValueError: checks.append(name)
        else: raise AssertionError(name)
    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
    # Establish that the narrow sibling preserves the existing science/runtime,
    # budget policy and owned-Popen command path; no source mutation is used.
    trees = [ast.parse((HERE/name).read_text(encoding="utf-8")) for name in
             ("supervise_TSC_same_science_v3.py", "supervise_TSC_same_science_v4.py")]
    functions = [{n.name:ast.dump(n,include_attributes=False) for n in t.body if isinstance(n,ast.FunctionDef)} for t in trees]
    unchanged = ("frozen_sources", "controller_order_proof", "source_state", "worker_proof",
                 "automatic_budget_handoff", "successor_command", "parser")
    check(all(functions[0][name] == functions[1][name] for name in unchanged),
          "Original science/native/runtime/budget/owned-successor functions AST unchanged")
    source_order = s.controller_order_proof(); pins = s.frozen_sources()
    frozen_before = {name:s.sha(HERE/name) for name in pins}
    for path in (s.FIRST,s.HELPER,s.PRELOADER,HERE/"supervise_TSC_same_science_v3.py"):
        frozen_before[path.name] = s.sha(path)
    scratch = ROOT / "work/research" / ("TSC_exact_bootstrap_v4_" + uuid.uuid4().hex); scratch.mkdir()
    original_here, original_source, original_order = s.HERE, s.source_state, s.controller_order_proof
    original_first_completion, original_exact_completion = s.domain.verify_completion, s.exact.verify_completion
    try:
        s.HERE = scratch / "outputs"
        s.controller_order_proof = lambda:source_order  # already executed on real pinned sources above
        run_id = "synthetic_TSC"
        started = datetime.now(timezone.utc).isoformat()
        original_ready = scratch / "original_ready.json"
        assets = {}
        for flag in ("tsc-input-proof","tsc-training-approval","source-closure","tr-manifest",
                     "replay-manifest","tr-frame-metadata","tr-frame-metadata-proof","data-v3-proof"):
            path = scratch / (flag+".json"); save(path,{"synthetic_asset":flag})
            assets[flag] = path
        argv = ["--run-id",run_id,"--native-backend-provenance",str(original_ready)]
        for flag,path in assets.items(): argv.extend(("--"+flag,str(path)))
        records = {flag.replace("-","_"):s.record(path) for flag,path in assets.items()}
        preload = {"kind":"coherent_native_cuda12_9_preload_v1",
            "module_sha256":pins["preload_nemo_linker_v1.py"],
            "actual_binding_linker":{"version":[12,9]},"cuda_initialized":False}
        fields = {"memory_launcher_sha256":"launch_with_memory_guard_tsc_v3.py",
            "cache_launcher_sha256":"launch_with_cache_reclaim_tsc_v3.py",
            "cache_helper_sha256":"cuda_cache_reclaim_v2.py","resource_helper_sha256":"resource_watchdog_v2.py",
            "resource_telemetry_sha256":"paired_windows_commit_telemetry_v1.py",
            "frozen_worker_sha256":"train_lora_batched_tsc_v3.py","data_policy_sha256":"public_turkish_data_policy_v3.py",
            "TSC_domain_sampler_sha256":"tsc_domain_sampler_v3.py",
            "TSC_domain_identity_sha256":"tsc_domain_training_identity_v3.py"}
        def identity(pid, ppid, arguments, exact_memory=False):
            value = {"kind":"owned_training_worker_identity_v1","expected_run_id":run_id,"pid":pid,"ppid":ppid,
                "worker_arguments_sha256":s.domain.guards.argument_sha(arguments),"started_at_utc":started,
                "TSC_explicit_asset_records":records,
                "process_creation_identity":{"available":True,"pid":pid,"creation_filetime_100ns":123,
                    "creation_unix_seconds":s.domain.guards.stamp(started).timestamp()},
                **{key:pins[name] for key,name in fields.items()}}
            if exact_memory:value["memory_launcher_sha256"] = s.preload.RUNTIME_MEMORY_SHA
            return value
        save(original_ready,{"kind":"native_tdt_training_backend_v3","status":"ready","run_id":run_id,
            "pid":101,"ppid":100,"worker_arguments_sha256":s.domain.guards.argument_sha(argv),
            "trainer_sha256":pins["train_lora_batched_tsc_v3.py"],"adapter_sha256":pins["native_tdt_checkpointed_v3.py"],
            "launcher_sha256":pins["launch_native_training_tsc_v3.py"],
            "memory_launcher_sha256":pins["launch_with_memory_guard_tsc_v3.py"],"preload":preload,
            "declared_allocator_policy":{"cap_gb":8,"end_cache_gb":4,"native_inactive_threshold_bytes":750000000}})
        first_folder = scratch / "first"; exact_folder = scratch / "exact"
        save(first_folder/"identity.json",identity(101,100,argv))
        first = {"kind":s.FIRST_KIND,"run_id":run_id,"status":"running","controller_sha256":s.FIRST_SHA,
            "owned_launcher_pid":100,"started_at_utc":started,"updated_at_utc":started,"source_sha256":pins,
            "worker_arguments":argv,"paths":{"native":str(original_ready),"identity":str(first_folder/"identity.json")},
            "configured_steps":44958}
        exact_argv = argv + ["--resume-exact",str(scratch/"opaque_snapshot.pt")]
        save(exact_folder/"identity.json",identity(201,200,exact_argv,True))
        memory = {"path":str(HERE/s.preload.RUNTIME_MEMORY_NAME),"sha256":s.preload.RUNTIME_MEMORY_SHA}
        runtime = {"kind":"TSC_same_science_native_preload_resume_v3","status":"ready","run_id":run_id,
            "pid":201,"ppid":200,"worker_arguments_sha256":s.domain.guards.argument_sha(exact_argv),
            "launcher_sha256":s.PRELOADER_SHA,"source_sha256":pins,"runtime_memory_launcher":memory,
            "preload":preload,"original_first_start_ready":s.record(original_ready),
            "scientific_config_identity_changed":False,"fresh_runtime_proof_separate_from_original_ready":True}
        runtime_path = exact_folder/"fresh_runtime.json"; save(runtime_path,runtime)
        exact = {"kind":s.EXACT_KIND,"run_id":run_id,"status":"running","controller_sha256":s.HELPER_SHA,
            "owned_launcher_pid":200,"started_at_utc":started,"updated_at_utc":started,"source_sha256":pins,
            "worker_arguments":exact_argv,"paths":{"identity":str(exact_folder/"identity.json"),"fresh_runtime":str(runtime_path)},
            "original_ready":s.record(original_ready),"new_preloader":{"sha256":s.PRELOADER_SHA},
            "runtime_memory_launcher":memory,"configured_steps":44958}
        science = {"fixture_only":True,"training_arguments":{"epochs":1.,"steps":44958}}
        states = {str(first_folder):first,str(exact_folder):exact}
        # Real source_state science validation is unchanged and intentionally
        # stubbed to avoid reading a real combined corpus/config/checkpoint.
        s.source_state = lambda folder,rid:(states[str(folder)],science)
        save(first_folder/"loop_status.json",first); save(exact_folder/"loop_status.json",exact)
        initial_first = s.bootstrap(first_folder,run_id,100,101)
        check(initial_first["initial_controller_kind"]==s.FIRST_KIND and initial_first["initial_controller_sha256"]==s.FIRST_SHA,
              "FIRST bootstrap continues to bind observed controller/hash")
        initial_exact = s.bootstrap(exact_folder,run_id,200,201)
        check(initial_exact["initial_controller_kind"]==s.EXACT_KIND and initial_exact["initial_controller_sha256"]==s.HELPER_SHA
            and initial_exact["current_runtime_proof"]==s.record(runtime_path)
            and initial_exact["original_TSC_ready"]==s.record(original_ready)
            and not initial_exact["initial_independent_OS_exit_observation"],
            "Manual EXACT bootstrap validates fresh worker/runtime separately from original ready")
        exact["kind"]="foreign"; rejects(lambda:s.bootstrap(exact_folder,run_id,200,201),"Foreign initial controller rejected"); exact["kind"]=s.EXACT_KIND
        exact["updated_at_utc"]=(datetime.now(timezone.utc)-timedelta(seconds=31)).isoformat()
        rejects(lambda:s.bootstrap(exact_folder,run_id,200,201),"Stale initial controller rejected"); exact["updated_at_utc"]=started
        rejects(lambda:s.bootstrap(exact_folder,run_id,200,101),"Old first worker PID cannot satisfy fresh exact ownership")
        bad = copy.deepcopy(runtime); bad["pid"]=101; save(runtime_path,bad)
        rejects(lambda:s.bootstrap(exact_folder,run_id,200,201),"Old runtime PID rejected"); save(runtime_path,runtime)
        s.domain.verify_completion=lambda plan,selected:{"fixture_only":True,"completion_contract_stubbed":True}
        s.exact.verify_completion=lambda plan,selected,identity:{"fixture_only":True,"completion_contract_stubbed":True}
        latest=scratch/"opaque_latest.bin"; best=scratch/"opaque_best.bin"
        latest.write_bytes(b"synthetic not checkpoint"); best.write_bytes(b"synthetic best")
        selection={"run_id":run_id,"accepted":True,"heldout_tests_evaluated":False,"completed_steps":20,"selected_step":0,
            "latest_weights":str(latest),"weights":str(best),"stop_early_reason":"soft_max_wall_seconds","turkish_epochs_completed":.0004}
        selection_path=s.HERE/"runs"/run_id/"selection.json"; save(selection_path,selection)
        for folder,state,initial in ((first_folder,first,initial_first),(exact_folder,exact,initial_exact)):
            state.update(status="quality_window_ended" if state["kind"]==s.FIRST_KIND else "train_window_ended",
                child_exit_code=0,completed_steps=20,selected_step=0,selection=s.record(selection_path),
                resource_review_required=False,stop_category="budget")
            save(folder/"loop_status.json",state)
            closed=s.closure(folder,run_id,initial)
            check(closed["automatic_budget_handoff_allowed"] and closed["actual_supervisor_owned_controller_exit_code"] is None,
                  "Initial "+state["kind"]+" closure validates without false OS exit claim")
        exact["controller_sha256"]=s.FIRST_SHA
        rejects(lambda:s.closure(exact_folder,run_id,initial_exact),"Observed initial controller SHA cannot change")
        exact["controller_sha256"]=s.HELPER_SHA
        wrong_initial=copy.deepcopy(initial_exact);wrong_initial["initial_controller_kind"]=s.FIRST_KIND
        rejects(lambda:s.closure(exact_folder,run_id,wrong_initial),"Observed initial controller kind cannot change")
        class Child:
            pid=300
            def __init__(self,code):self.code=code;self.waited=False
            def poll(self):return self.code
            def wait(self,timeout):self.waited=True;return self.code
        rejects(lambda:s.closure(exact_folder,run_id,initial_exact,Child(None)),"Future owned live child cannot be treated as closed")
        rejects(lambda:s.closure(exact_folder,run_id,initial_exact,Child(1)),"Future owned failed exit cannot permit budget successor")
        child=Child(0); closed=s.closure(exact_folder,run_id,initial_exact,child)
        check(child.waited and closed["actual_supervisor_owned_controller_exit_code"]==0,
              "Future owned poll+wait exit0 remains mandatory")
        for category,reason in (("resource","controller_stop_file"),("retention","consecutive_retention_gate_failures"),
                                ("plateau","development_plateau"),("controller_stop_file","controller_stop_file")):
            state=copy.deepcopy(exact); state["stop_category"]=category
            if category=="resource":state["resource_review_required"]=True
            check(not s.automatic_budget_handoff({"stop_early_reason":reason},state),category+" stop never auto-resumes")
        state=copy.deepcopy(exact);state["cooperative_stop_reason"]="window_budget_checkpoint_boundary"
        check(s.automatic_budget_handoff({"stop_early_reason":"controller_stop_file"},state),"Explicit scheduled boundary remains eligible")
    finally:
        s.HERE,s.source_state,s.controller_order_proof=original_here,original_source,original_order
        s.domain.verify_completion,s.exact.verify_completion=original_first_completion,original_exact_completion
    check(all(s.sha(HERE/name)==digest for name,digest in frozen_before.items()),"Frozen science/controller/runtime sources unchanged")
    check(not any(name in sys.modules for name in ("torch","nemo","transformers","numpy","soundfile")),"No ML/audio imports")
    result={"schema":1,"kind":"TSC_same_science_FIRST_or_EXACT_bootstrap_supervisor_CPU_verification_v4",
        "passed":True,"check_count":len(checks),"checks":checks,"source":s.record(s.__file__),"fixture":s.record(__file__),
        "actual_source_order_proof":source_order,"frozen_source_sha256_unchanged":frozen_before,
        "actual_worker_proof_FIRST_and_EXACT_executed":True,
        "synthetic_source_state_and_completion_contracts_explicitly_stubbed":True,
        "actual_process_or_model_GPU_launch":False,"actual_live_or_serialized_checkpoint_state_read":False,
        "scratch":str(scratch),"created_at_utc":s.now()}
    with output.open("x",encoding="utf-8") as stream:json.dump(result,stream,indent=2)
    print(json.dumps({"passed":True,"checks":len(checks),"source_sha256":result["source"]["sha256"],
                      "report":str(output),"report_sha256":s.sha(output)}))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,required=True)
    main(parser.parse_args())
