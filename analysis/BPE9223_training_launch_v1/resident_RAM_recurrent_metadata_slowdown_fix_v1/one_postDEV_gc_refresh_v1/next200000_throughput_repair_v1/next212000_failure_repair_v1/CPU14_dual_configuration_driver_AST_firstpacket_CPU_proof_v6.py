"""INACTIVE SOURCE v6. Distinct incoming stream and parent-runtime roles. REAL launch adapter/runner startup coverage; separate ROOT scope required.
No production source edit. First whole original packet only; no Adam/native/GPU/draw.
"""
import ast
import copy
from types import SimpleNamespace
import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import weakref
HERE=Path(__file__).resolve().parent
RUNNER=HERE/"CPU14_owned_startup_only_receipt_runner_v2.py"
ADAPTER=HERE/"CPU14_startup_launch_adapter_v15.py"
def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def pin(p):
    p=Path(p);raw=p.read_bytes()
    return dict(path=str(p.resolve()),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def save(p,value):
    p=Path(p);tmp=p.with_suffix(".owned_tmp")
    with tmp.open("x",encoding="utf-8")as f:json.dump(value,f,allow_nan=False,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def who(p):return dict(pid=p.pid,ctime=p.create_time(),ppid=p.ppid())
def sample():
    import psutil
    class M(ctypes.Structure):
        _fields_=[("length",ctypes.c_ulong),("load",ctypes.c_ulong)]+[(n,ctypes.c_ulonglong)for n in ("totalphysical","physical","commitlimit","commit","totalvirtual","virtual","extended")]
    m=M();m.length=ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):raise ctypes.WinError()
    return dict(commit=m.commit,physical=m.physical,private=psutil.Process().memory_info().private,E_free=shutil.disk_usage("E:/gpu_training_scratchpad").free)

def configure_from_candidate_driver_AST(scope,root,cpu,scope_path):
    driver_pin=scope["candidate_driver_source"]
    assert driver_pin in scope["input_pins"]and pin(driver_pin["path"])==driver_pin
    source=Path(driver_pin["path"]).read_text(encoding="utf8")
    tree=ast.parse(source)
    calls=[n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=="configure_startup_receipts"]
    assert len(calls)==1 and len(calls[0].args)==2 and not calls[0].keywords
    call=calls[0]
    assert ast.dump(call.args[1],include_attributes=False)==ast.dump(ast.parse("record(a.stream_configuration)",mode="eval").body,include_attributes=False)
    incoming=scope["configuration_pin"]
    assert pin(incoming["path"])==incoming
    a=SimpleNamespace(configuration=incoming["path"],stream_configuration=incoming["path"])
    candidate_base=copy.deepcopy(read(incoming["path"]))
    # Only the owned proof's authority metadata is substituted; numerical configuration remains pinned ORIGINAL input.
    candidate_base["original_CPU14_startup_reader7_recovery_policy"]["CPU_startup_policy"]=copy.deepcopy(scope["CPU_startup_policy"])
    assert scope["CPU_startup_policy"]["execution_review_path"]==str(Path(scope_path).resolve())
    c=copy.deepcopy(candidate_base)
    updates=[n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and isinstance(n.func.value,ast.Name)and n.func.value.id=="c"and n.func.attr=="update"and {k.arg for k in n.keywords}=={"worker_driver_source","parent_driver_source","resident_RAM_stream_configuration","resident_branch_namespace","checkpoint_interval_updates","source_checkpoint","source_configuration"}]
    assert len(updates)==1
    scalar_pointer=scope["saved212000_pointer_scalar_pin"]
    assert scalar_pointer==dict(path=str(Path(r"E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01/checkpoints/update_000212000/pointer.json").resolve()),bytes=16120,sha256="9bef17e4fd0ed37f34f056595ca09ec91d2a69d80786165d6ca866b270bcbe07")
    ns=dict(c=c,record=pin,__file__=driver_pin["path"],a=a,folder=root,pointer_pin=scalar_pointer)
    expr=ast.Expression(body=copy.deepcopy(updates[0]));ast.fix_missing_locations(expr)
    eval(compile(expr,driver_pin["path"],"eval"),ns)
    runtime_path=root/"candidate_parent_runtime_configuration.json";save(runtime_path,c);runtime_pin=pin(runtime_path)
    assert runtime_pin!=incoming and runtime_pin["path"]!=incoming["path"]
    assert c["resident_RAM_stream_configuration"]==incoming and c["source_configuration"]==incoming
    ns=dict(startup_adapter=cpu,candidate_base=candidate_base,record=pin,a=a,config_path=runtime_path)
    expr=ast.Expression(body=copy.deepcopy(call));ast.fix_missing_locations(expr)
    eval(compile(expr,driver_pin["path"],"eval"),ns)
    assert cpu.CONTEXT["configuration"]==incoming and cpu.CONTEXT["configuration"]!=runtime_pin
    # Exact SOURCE comparison negative: old runtime role would reject this distinct stream argv BEFORE Popen.
    actual=[str(cpu.WORKER_PYTHON),"-Xutf8","-B",str(cpu.ORIGINAL.resolve()),"--worker","--configuration",incoming["path"],"--private-limit","8000000000"]
    expected=list(actual);expected[6]=runtime_pin["path"]
    adapter_ast=ast.parse(Path(ADAPTER).read_text(encoding="utf8"))
    gates=[n for n in ast.walk(adapter_ast)if isinstance(n,ast.Assert)and ast.dump(n.test,include_attributes=False)==ast.dump(ast.parse("actual[1:3]==expected[1:3]and actual[4:]==expected[4:]",mode="eval").body,include_attributes=False)]
    assert len(gates)==1
    negative=ast.Module(body=[copy.deepcopy(gates[0])],type_ignores=[]);ast.fix_missing_locations(negative)
    rejected=False
    try:exec(compile(negative,str(ADAPTER),"exec"),dict(actual=actual,expected=expected))
    except AssertionError:rejected=True
    assert rejected
    result=dict(candidate_driver_source=driver_pin,actual_driver_configure_call_AST=ast.dump(call,include_attributes=False),incoming_worker_stream_configuration=incoming,separate_parent_runtime_configuration=runtime_pin,adapter_bound_configuration=cpu.CONTEXT["configuration"],old_wrong_role_SOURCE_gate_rejected=True,proof_only_policy_authority_substitution=True,no_pointer_body_read=True)
    save(root/"actual_candidate_driver_dual_configuration_roles.json",result)
    return result

def check_actual_dual_role_wire(early,roles,root):
    argv=early["argv"];incoming=roles["incoming_worker_stream_configuration"]
    assert argv.count("--configuration")==1 and argv[argv.index("--configuration")+1]==incoming["path"]
    assert argv.count("--expected-configuration-pin")==1 and json.loads(argv[argv.index("--expected-configuration-pin")+1])==incoming
    assert incoming!=roles["separate_parent_runtime_configuration"]
    save(root/"actual_original_worker_dual_role_wire.json",dict(incoming_stream=incoming,parent_runtime=roles["separate_parent_runtime_configuration"],actual_runner_argv=argv,actual_worker_configuration_equals_incoming=True,parent_runtime_not_used_for_worker=True))

def fixture_body(scope_path):
    import psutil
    scope=read(scope_path)
    assert scope["accepted"]is True and scope["reviewer"]=="/root"
    assert scope["kind"]=="ROOT_CPU14_startup_adapter_firstpacket_exact_ONE_CPU_scope_v1"
    # These are deliberately UNKNOWN/unread during source preparation.
    assert scope["configuration_pin"]is not None and scope["plan_pin"]is not None
    assert scope["original_plan_audio_execution_explicitly_admitted"]is True
    assert scope["next_queued_plan_unconsumed_authority_verified"]is True
    root=Path(scope["output_directory"]).resolve()
    assert root.is_relative_to(HERE)and root!=HERE and not root.exists();root.mkdir()
    limits=scope["resources"]
    # Original CPU proof minimum already has old1MiB receipt; replace1->8 and add reader API16+projector64, never count old1 twice.
    assert limits["prospective_commit"]>=8528382208
    assert limits["prospective_physical"]>=5778382208
    assert limits["packet_timeout_seconds"]==300 and limits["native_stall_seconds"]==600
    assert limits["worker_private_bytes"]==8000000000
    assert limits["external_private_bytes"]<=536870912
    assert limits["whole_proof_wall_seconds"]==limits["proof_body_wall_seconds"]
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    initial=sample();save(root/"initial_BEFORE_predicate.json",dict(sample=initial,entry=who(psutil.Process()),launcher=who(psutil.Process().parent())))
    assert initial["commit"]>=limits["prospective_commit"]and initial["physical"]>=limits["prospective_physical"]
    assert initial["private"]<=limits["external_private_bytes"]and initial["E_free"]>=100000000000
    for p in scope["input_pins"]:assert pin(p["path"])==p
    assert scope["CPU_startup_runner_source"]==pin(RUNNER)and scope["adapter_source"]==pin(ADAPTER)
    assert Path(scope["CPU_receipt_root"]).resolve()==root
    assert scope["CPU14_source"]["sha256"]=="e3d01517ac3af9e481a27b0127474c7a8fbc2d9351f7fcc6aefcf36691675d0e"
    assert scope["configuration_pin"]in scope["input_pins"]and scope["plan_pin"]in scope["input_pins"]
    # No model or audio import until every exact explicit input/initial gate passes.
    # Different venv PATHS are authoritative even though the launcher shims hash
    # identically. No package install, sys.path injection or environment edit.
    expected_parent=Path(r"C:/Users/ardao/Documents/Codex/2026-10-10/i-x20/work/nemo-native-env-v2/Scripts/python.exe").resolve()
    expected_worker=Path(r"E:/gpu_training_scratchpad/scratch/BPE_broad_CPU_preflight_env_v2/Scripts/python.exe").resolve()
    assert Path(sys.executable).resolve()==expected_parent
    assert Path(scope["fixture_body_python"]["path"]).resolve()==expected_parent
    assert Path(scope["original_worker_python"]["path"]).resolve()==expected_worker
    assert expected_parent!=expected_worker
    for executable in (scope["fixture_body_python"],scope["original_worker_python"]):
        assert executable["bytes"]==274192 and executable["sha256"]=="cac98e94324b5462d41a5c6ad1847b57508a1068ba56014cf67a3d33a493dcdf"
        assert pin(executable["path"])==executable
    save(root/"actual_parent_worker_interpreter_bindings.json",dict(
        parent=scope["fixture_body_python"],actual_parent_executable=str(Path(sys.executable).resolve()),
        worker_contract=scope["original_worker_python"],same_shim_hash_does_not_replace_path_authority=True,
        no_package_path_or_environment_mutation=True))
    spec=importlib.util.spec_from_file_location("owned_real_CPU14_startup_adapter",ADAPTER)
    cpu=importlib.util.module_from_spec(spec);spec.loader.exec_module(cpu)
    assert cpu.ResidentFeatureProducer.__init__.__globals__ is cpu._namespace
    assert type(cpu._namespace["subprocess"]).__name__=="_OwnedLaunchSubprocess"
    dual_roles=configure_from_candidate_driver_AST(scope,root,cpu,scope_path)
    save(root/"actual_adapter_private_namespace_binding.json",dict(adapter=pin(ADAPTER),runner=pin(RUNNER),
        original_CPU14=scope["CPU14_source"],actual_original_init_globals_is_private_namespace=True,
        actual_launch_subprocess_class=type(cpu._namespace["subprocess"]).__name__,proof_global_Popen_override=False))
    started=time.monotonic();known={};dead_early_receipts=[];launchers=[];stream=None;cleanup=[];primary=None
    receipts=root/"original_CPU14"/"early_startup_receipts"
    def capture():
        for launcher in launchers:
            try:
                parent=psutil.Process(launcher["pid"])
                assert parent.create_time()==launcher["ctime"]and parent.ppid()==launcher["ppid"]
                family=[parent]+parent.children(recursive=True);assert len(family)<=8
                for child in family:
                    identity=who(child)
                    if child.pid in known:assert known[child.pid]==identity
                    known[child.pid]=identity
            except psutil.NoSuchProcess:pass # Exact launcher may already be dead; receipt checked separately.
        early=receipts/"earliest_owned_worker_identity.json"
        if early.exists():
            identity=read(early)
            recorded={k:identity[k]for k in ("pid","ctime","ppid")}
            known[identity["pid"]]=recorded
            try:
                child=psutil.Process(identity["pid"])
                if child.create_time()!=identity["ctime"]:
                    if len(dead_early_receipts)<16:dead_early_receipts.append(dict(identity=recorded,observed="old exact identity dead; PID now reused"))
                    return
                assert child.ppid()==identity["ppid"]
                lineage=[who(child)]+[who(p)for p in child.parents()[:8]]
                assert any((x["pid"],x["ctime"])in {(a["pid"],a["ctime"])for a in launchers}for x in lineage) or child.ppid()==os.getpid()
            except psutil.NoSuchProcess:
                # Preserve original worker/protocol failure; receipt identity is still evidence.
                if len(dead_early_receipts)<16:dead_early_receipts.append(dict(identity=recorded,observed="exact early body already absent"))
    def guard():
        capture();s=sample()
        assert s["commit"]>=limits["runtime_commit"]and s["physical"]>=limits["runtime_physical"]
        assert s["private"]<=limits["external_private_bytes"]and s["E_free"]>=100000000000
        assert time.monotonic()-started<=limits["whole_proof_wall_seconds"]
        private=s["private"]
        for identity in known.values():
            try:
                p=psutil.Process(identity["pid"])
                if p.create_time()!=identity["ctime"]:continue
                private+=p.memory_info().private
            except psutil.NoSuchProcess:pass
        assert private<=limits["family_private_bytes"]
    # Proof-only scalar witness of the ORIGINAL parent failure path. No packet,
    # exception, traceback or frame object is retained in these bounded records.
    original_parent_fail=cpu.ResidentFeatureProducer._fail
    parent_fail_lock=threading.Lock()
    parent_fail_records=[]
    parent_fail_dropped=0
    parent_fail_receipt_errors=0
    deliberate_abort_threads=set()
    parent_fail_receipt_path=root/"parent_first_fail_scalars.json"
    def fail_scalars(error):
        item=dict(type=type(error).__name__[:128],module=type(error).__module__[:256],message="<scalar message unavailable>",traceback=[])
        try:item["message"]=str(error)[:2000]
        except BaseException:item["message"]="<error message conversion failed>"
        tb=None;frame=None
        try:
            tb=error.__traceback__
            while tb is not None and len(item["traceback"])<32:
                frame=tb.tb_frame
                item["traceback"].append(dict(file=frame.f_code.co_filename[:512],function=frame.f_code.co_name[:128],line=int(tb.tb_lineno)))
                frame=None
                tb=tb.tb_next
            item["traceback_truncated"]=tb is not None
        except BaseException:item["traceback_capture_failed"]=True
        finally:
            frame=None
            tb=None
        return item
    def write_fail_witness():
        # Called only with this proof-owned receipt lock, never the transport's
        # condition/credit lock. Any receipt failure cannot replace _fail.
        save(parent_fail_receipt_path,dict(kind="CPU14_original_parent_fail_scalar_witness_v1",
            records=parent_fail_records,dropped=parent_fail_dropped,receipt_errors=parent_fail_receipt_errors,
            first_non_deliberate=next((x["sequence"]for x in parent_fail_records if not x["deliberate_proof_abort"]),None),
            order="observer entry order; original_first_stored_error_match is authoritative where recorded",
            max_records=8,max_traceback_frames=32,no_args_packets_locals_frames_or_exception_objects_retained=True))
    def observed_parent_fail(instance,error):
        nonlocal parent_fail_dropped,parent_fail_receipt_errors
        sequence=None
        try:
            item=fail_scalars(error)
            with parent_fail_lock:
                # Only an exact error synchronously created by our original
                # abort call is labelled deliberate; concurrent receiver/sender
                # failures on other threads remain non-deliberate.
                args=error.args
                deliberate=(threading.get_ident()in deliberate_abort_threads and type(error)is RuntimeError
                    and type(args)is tuple and len(args)==1 and type(args[0])is str
                    and args[0]=="Resident producer explicitly aborted; no normal exit claimed")
                del args
                if len(parent_fail_records)<8:
                    sequence=len(parent_fail_records)
                    item.update(sequence=sequence,deliberate_proof_abort=deliberate,
                        original_call_started=False,original_call_returned=False,
                        original_first_stored_error_match="PENDING",captured_before_original_fail=True)
                    parent_fail_records.append(item)
                    write_fail_witness()
                else:parent_fail_dropped+=1
        except BaseException:
            # Observation is fail-open to the unchanged original failure path.
            parent_fail_receipt_errors+=1
        returned=False
        try:
            # EXACTLY ONE call, SAME instance/error; no catch/retry/replacement.
            result=original_parent_fail(instance,error)
            returned=True
            return result
        finally:
            try:
                with parent_fail_lock:
                    if sequence is not None:
                        parent_fail_records[sequence].update(original_call_started=True,original_call_returned=returned,
                            original_first_stored_error_match=instance.error is error)
                        write_fail_witness()
            except BaseException:parent_fail_receipt_errors+=1
    def abort_with_deliberate_label(instance):
        ident=threading.get_ident()
        with parent_fail_lock:deliberate_abort_threads.add(ident)
        try:return instance.abort() # SAME original finite abort, once.
        finally:
            with parent_fail_lock:deliberate_abort_threads.discard(ident)
    cpu.ResidentFeatureProducer._fail=observed_parent_fail
    try:
        # The REAL adapter proxy makes the sole ORIGINAL Popen call. Passive
        # identity observation occurs only AFTER that original constructor returns.
        stream=cpu.ResidentFeatureProducer(scope["configuration_pin"],root/"original_CPU14",queue_bytes=scope["original_queue_bytes"],guard=guard)
        try:
            identity=who(psutil.Process(stream.process.pid));launchers.append(identity)
            save(root/"actual_parent_Popen_identity.json",dict(identity=identity,
                argv=psutil.Process(stream.process.pid).cmdline(),real_adapter_path=True,proof_Popen_override=False))
        except psutil.NoSuchProcess:
            save(root/"actual_parent_Popen_identity.json",dict(pid=stream.process.pid,ctime="UNKNOWN_ALREADY_EXITED",
                Popen_poll=stream.process.poll(),real_adapter_path=True,proof_Popen_override=False))
        capture()
        plan_sha=stream.submit(scope["plan_pin"])
        ready=stream.wait_update_ready(plan_sha,0) # Original default300, nonconsuming.
        capture()
        early=read(receipts/"earliest_owned_worker_identity.json")
        check_actual_dual_role_wire(early,dual_roles,root)
        assert stream.producer_identity["pid"]==early["pid"]
        packet=stream.next_update(plan_sha,0) # Entire original packet; no authority transaction.
        assert packet["plan_sha256"]==plan_sha and packet["update_index"]==0
        assert len(packet["arrays"])==16 and len(packet["metadata"]["rows"])==16
        import numpy as np
        references=[]
        summary=[]
        expected=scope["exact_expected_first_packet_rows"]
        assert type(expected)is list and len(expected)==16
        fields=scope["source_reviewed_packet_fields"]
        assert fields==dict(full_targets=["full_target_ids"],source_identity=["source_id"])
        for index,(arrays,row,want)in enumerate(zip(packet["arrays"],packet["metadata"]["rows"],expected)):
            a=arrays["input_features"];mask=arrays["attention_mask"]
            assert a.dtype==np.float32 and mask.dtype==np.bool_
            assert np.isfinite(a).all()
            # Exact original prepared metadata/transport geometry, no invented channel count.
            assert list(a.shape)==row["input_features"]["shape"]
            assert list(mask.shape)==row["attention_mask"]["shape"]
            assert row["source_id"]==want["source_id"]
            targets=row["full_target_ids"]
            assert type(targets)is list and len(targets)==want["full_target_count"]
            assert all(type(v)is int and 0<=v<9223 for v in targets)
            independent_ids=False
            if want["original_stream"]=="TR":
                assert targets==want["full_target_ids_from_already_bound_plan_metadata"]
                independent_ids=True
            else:
                assert want["original_stream"]=="replay"
                assert want["original_worker_full_label_cache_checks_source_unchanged"]is True
                # Full replay IDs are NOT invented or separately read from a cache.
                # The unchanged original worker performs its original source/cache checks.
            references.extend((weakref.ref(a),weakref.ref(mask)))
            target_u32_sha=hashlib.sha256(b"".join(v.to_bytes(4,"little")for v in targets)).hexdigest()
            summary.append(dict(index=index,feature_shape=list(a.shape),mask_shape=list(mask.shape),finite=True,full_target_count=len(targets),actual_full_target_u32_sha256=target_u32_sha,independent_full_ID_equality=independent_ids,replay_full_cache_checks="unchanged original worker; no separate cache/tensor read"))
        token=packet["token"]
        del a,mask,arrays,row,want,targets,packet
        assert all(ref()is None for ref in references)
        stream.release(token) # Only after every delivered payload reference is gone.
        abort_with_deliberate_label(stream) # Same finite original abort, separately labelled.
        result=dict(first_ready=ready,first_complete_packet_rows=summary,raw_feature_refs_dead=True,original_release_after_refdrop=True,original_abort_returned=True,owner_credit=stream.credit.receipt(),sender_joined=not stream._sender.is_alive(),receiver_joined=not stream._receiver.is_alive(),pipes_closed=stream.process.stdin.closed and stream.process.stdout.closed,log_closed=stream.log.closed,child_credit_method_close="UNKNOWN under deliberate finite abort",child_Popen_code=stream.process.poll(),normal0_claimed=False,permitted_speculative_updates_before_abort="Original MAX2 credit pipeline only; release may allow a third update to begin. Future scope must explicitly admit these original rows.")
        receipt_paths=sorted(receipts.glob("*.json"));assert len(receipt_paths)<=130
        actual_receipts=[read(p)for p in receipt_paths]
        phases=[r.get("phase")for r in actual_receipts]
        for hook in ("checked_resource_profile","_startup","_header","_send"):
            assert phases.count("before_first_"+hook)==1
        receipt_bytes=sum(p.stat().st_size for p in receipts.iterdir()if p.is_file())
        assert receipt_bytes<=1048576
        result.update(real_adapter_path=True,proof_global_Popen_override=False,startup_receipt_bytes=receipt_bytes,
            startup_first_call_phases=phases,all_four_hooks_observed_once=True,
            startup_hooks_detach_before_first_original_call_SOURCE_bound=True,
            original_worker14_binary_stdout_and_FULL_transport_unchanged=True)
        save(root/"first_packet_result.json",result)
    except BaseException as error:
        primary=dict(type=type(error).__name__,message=str(error)[:2000])
        raise
    finally:
        if stream is not None:
            try:
                if not stream.log.closed:abort_with_deliberate_label(stream)
            except BaseException as error:cleanup.append(dict(action="original_abort",error=type(error).__name__+": "+str(error)[:500]))
        cpu.ResidentFeatureProducer._fail=original_parent_fail
        # Proof-only cleanup of all EARLY known fixture children if original pre-ready
        # abort had no producer_identity. No production-source patch is implied.
        for identity in reversed(list(known.values())):
            try:
                p=psutil.Process(identity["pid"])
                if p.create_time()!=identity["ctime"]:continue
                p.terminate();code=p.wait(timeout=30)
                cleanup.append(dict(action="owned_known_child_cleanup",identity=identity,observed_wait_code=code,child_method_close="UNKNOWN"))
            except psutil.NoSuchProcess:pass
            except BaseException as error:cleanup.append(dict(identity=identity,error=type(error).__name__+": "+str(error)[:500],exact_death="UNKNOWN"))
        deaths=[]
        for identity in known.values():
            try:p=psutil.Process(identity["pid"]);alive=p.is_running()and p.create_time()==identity["ctime"]
            except psutil.NoSuchProcess:alive=False
            deaths.append(dict(identity=identity,exact_dead=not alive))
        save(root/"result.json",dict(passed=primary is None and not any(x.get("exact_death")=="UNKNOWN"for x in cleanup),primary=primary,cleanup=cleanup,known_children=deaths,dead_early_receipts=dead_early_receipts,parent_first_fail_scalars=dict(path=str(parent_fail_receipt_path),exists=parent_fail_receipt_path.exists(),records=parent_fail_records,dropped=parent_fail_dropped,receipt_errors=parent_fail_receipt_errors),wall=time.monotonic()-started,CPU14_original_source_unchanged=True,original_plan_audio_execution_only_as_explicit_scope=True,model_GPU_Adam_sampler_journal_advance=False,actual212_cause_or_permanent_speed_claim=False))


def main(scope_path):
    scope=read(scope_path)
    path=HERE/"owned_CPU_startup_adapter_proof_watchdog_v3.py"
    assert scope["watchdog_source"]==pin(path)
    spec=importlib.util.spec_from_file_location("owned_CPU14_proof_watchdog",path)
    supervisor=importlib.util.module_from_spec(spec);spec.loader.exec_module(supervisor)
    return supervisor.execute(scope_path,Path(__file__).resolve(),"CPU")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--ROOT-review",required=True);p.add_argument("--proof-body",action="store_true");a=p.parse_args()
    if a.proof_body:fixture_body(a.ROOT_review)
    else:main(a.ROOT_review)
