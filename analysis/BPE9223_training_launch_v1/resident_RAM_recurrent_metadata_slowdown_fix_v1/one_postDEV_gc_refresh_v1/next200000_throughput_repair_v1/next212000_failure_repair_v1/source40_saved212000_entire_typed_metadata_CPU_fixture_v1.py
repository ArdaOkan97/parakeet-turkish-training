"""INACTIVE actual-helper/driver/wrapper AST metadata control.
All external ROOT/reader/state authorities below are explicit METHOD_STUBs.
No audio/tensor/restore/copy/GC/native/HTTP/DB/training operation.
"""
import argparse,ast,copy,ctypes,hashlib,json,os,shutil,time
from pathlib import Path as RealPath
from types import SimpleNamespace
import psutil
HERE=RealPath(__file__).resolve().parent
P=RealPath(r"C:/Users/ardao/Documents/Codex/2026-10-10/i-x20")
BRANCH=RealPath(r"E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01")
def read(p):return json.loads(RealPath(p).read_text(encoding="utf-8-sig"))
def pin(p):
    p=RealPath(p);b=p.read_bytes()
    return dict(path=str(p.resolve()),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def save(p,v):
    with RealPath(p).open("x",encoding="utf-8")as f:json.dump(v,f,ensure_ascii=False,allow_nan=False,indent=2);f.write("\n");f.flush();os.fsync(f.fileno())
def who(p):return dict(pid=p.pid,ctime=p.create_time(),ppid=p.ppid())
def memory():
    class M(ctypes.Structure):
        _fields_=[("length",ctypes.c_ulong),("load",ctypes.c_ulong)]+[(n,ctypes.c_ulonglong)for n in ("totalphysical","physical","commitlimit","commit","totalvirtual","virtual","extended")]
    m=M();m.length=ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):raise ctypes.WinError()
    return dict(commit=m.commit,physical=m.physical,private=psutil.Process().memory_info().private,E_free=shutil.disk_usage("E:/gpu_training_scratchpad").free)
def norm(p):return str(RealPath(str(p)).resolve())
def main(scope_path):
    scope=read(scope_path)
    assert scope["accepted"]is True and scope["reviewer"]=="/root"and scope["kind"]=="ROOT_source40_saved212000_entire_typed_metadata_exact_ONE_CPU_scope_v1"
    out=RealPath(scope["output_directory"]).resolve();assert out.is_relative_to(HERE)and out!=HERE and not out.exists();out.mkdir()
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    began=time.monotonic();initial=memory();limits=scope["resources"]
    save(out/"initial_BEFORE_predicate.json",dict(sample=initial,entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),limits=limits))
    assert initial["commit"]>=limits["prospective_commit"]and initial["physical"]>=limits["prospective_physical"]
    assert initial["private"]<=536870912 and initial["E_free"]>=100000000000
    for p in scope["input_pins"]:assert pin(p["path"])==p
    config=read(scope["source_configuration"]["path"])
    helper_path=HERE/"asr18_current_plan_throughput_checkpoint_transition_v5.py"
    helper_tree=ast.parse(helper_path.read_text(encoding="utf-8-sig"))
    constants={}
    for n in helper_tree.body:
        if isinstance(n,ast.Assign)and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
            try:constants[n.targets[0].id]=ast.literal_eval(n.value)
            except (ValueError,TypeError):pass
    required=constants["REQUIRED_OLD_IDENTITIES"]
    prefix_path=str(BRANCH/"METHOD_STUB_saved_J27.jsonl")
    fakeCPU=dict(path=str(out/"METHOD_STUB_CPU_ROOT.json"),bytes=17,sha256="a"*64)
    fakeReader=dict(path=str(out/"METHOD_STUB_reader_ROOT.json"),bytes=19,sha256="b"*64)
    fake_status=str(out/"METHOD_STUB_current_status.json")
    proof=dict(kind="ASR18_original_CPU14_startup_adapter_actual_CPU_v1",accepted=True,reviewer="/root",
        adapter_source=config["original_CPU14_startup_reader7_recovery_policy"]["CPU_startup_policy"]["adapter_source"],
        runner_source=config["original_CPU14_startup_reader7_recovery_policy"]["CPU_startup_policy"]["runner_source"],
        CPU14_source=config["original_CPU14_startup_reader7_recovery_policy"]["CPU_startup_policy"]["original_CPU14_source"],
        full_first_packet_passed=True,original_CPU14_worker_and_transport_unchanged=True,all_exact_known_identities_dead=True)
    rp=config["original_CPU14_startup_reader7_recovery_policy"]
    reader=dict(kind="ROOT_reader7_actual_replacement_acceptance_v1",accepted=True,reviewer="/root",
        template_source=rp["reader_template"],projection_source=rp["projection_source"],API_helper_source=rp["API_helper_source"],
        old_exact_pair_dead=True,mirror_thread_alive=True,HTTP_ready=True,reader_ready=True,error=None,
        new_reader_identity=dict(pid=991,ctime=1.0,ppid=992),new_reader_launcher_identity=dict(pid=992,ctime=2.0,ppid=993),
        status_path=fake_status,journal_path=prefix_path,journal_offset=426026870,verified_updates=212000)
    more=[dict(pid=26648,create_time=1792849617.2166405,ppid=6320),dict(pid=6320,create_time=1792849617.2091935,ppid=21892),
        dict(pid=25432,create_time=1792856348.1146228,ppid=11804),dict(pid=11804,create_time=1792856348.1062205,ppid=36956),
        dict(pid=34508,create_time=1792855690.6090496)]
    failure=dict(kind="ROOT_NEW212000_production_and_reader_failure_independent_seal_v1",accepted=True,reviewer="/root",
        actual212000_saved212000_gap0=True,production76082_consumed_ONCE_only_by_owner=True,
        native_primary_recorded_failure="TimeoutError('Resident CPU update packet timed out')",underlying_new_CPU_startup_cause="UNKNOWN",
        separate_native_CPU_wrapper_numeric_OS_exits="UNRECORDED_UNKNOWN",all_recorded_old_runtime_control_sealer_identities_exact_dead=required+more)
    saved=dict(accepted=True,reviewer="/root",updates=212000,weights=207,Adam_states=215,Adam_partitions=[17,6,192],
        all4RNG=True,named_tensor_references=854,integer_Adam_JSON_keys_only_normalized=True,semantic_identity_sampler_journal_nextdraw_equal=True)
    pointer=dict(committed_updates=212000,diagnostic_only=False,all_window_updates_committed=True,journal_prefix=dict(path=prefix_path,offset=426026870,sha256="6621855eb4191790c3a741e3e1c19014f00db87b9e471d3a4c70802cbddaaf0a"))
    status=dict(pid=991,create_time=1.0,mirror_thread_alive=True,HTTP_ready=True,reader_ready=True,error=None)
    baseline=read(constants["BASE_SOURCE"]["path"]) # SOURCE config only, no runtime/checkpoint.
    current={}
    class MissingProcess(Exception):pass
    class ProcessDouble:
        def __init__(self,pid):
            self.pid=pid
            if pid not in current["live"]:raise MissingProcess()
        def create_time(self):return current["live"][self.pid]["ctime"]
        def ppid(self):return current["live"][self.pid]["ppid"]
    class PathDouble:
        def __init__(self,p):self.p=RealPath(str(p))
        def __str__(self):return str(self.p)
        def __truediv__(self,p):return PathDouble(self.p/p)
        def resolve(self):return self
        def __eq__(self,other):return norm(self)==norm(other)
        def is_relative_to(self,other):return self.p.resolve().is_relative_to(RealPath(str(other)).resolve())
        def stat(self):
            assert norm(self)==norm(prefix_path),"METHOD_STUB only expected journal STAT, no real original file"
            return SimpleNamespace(st_size=current["stat"]["bytes"],st_mtime_ns=current["stat"]["mtime_ns"])
    def reset():
        current.clear();candidate=copy.deepcopy(config)
        repair=candidate["original_CPU14_startup_reader7_recovery_policy"]
        repair["CPU_startup_policy"]["actual_new_adapter_CPU_control_review"]=fakeCPU
        repair["actual_reader_replacement_ROOT"]=fakeReader
        repair["closed_original_journal_STAT"]=dict(kind="STAT_ONLY_original_closed_journal_v1",path=prefix_path,bytes=426026870,mtime_ns=123456789)
        records={norm(v["path"]):copy.deepcopy(v)for v in constants.values()if type(v)is dict and set(v)=={"path","bytes","sha256"}}
        cpu=repair["CPU_startup_policy"]
        for v in (cpu["adapter_source"],cpu["runner_source"],cpu["original_CPU14_source"],fakeCPU,fakeReader):records[norm(v["path"])]=copy.deepcopy(v)
        payload={norm(constants["BASE_SOURCE"]["path"]):copy.deepcopy(baseline),norm(fakeCPU["path"]):copy.deepcopy(proof),
            norm(fakeReader["path"]):copy.deepcopy(reader),norm(fake_status):copy.deepcopy(status),
            norm(constants["FAILURE_ROOT"]["path"]):copy.deepcopy(failure),norm(constants["SAVED_ROOT"]["path"]):copy.deepcopy(saved),
            norm(constants["SOURCE_DERIVED_POINTER"]["path"]):copy.deepcopy(pointer)}
        current.update(candidate=candidate,records=records,payload=payload,live={991:dict(ctime=1.0,ppid=992),992:dict(ctime=2.0,ppid=993)},stat=dict(bytes=426026870,mtime_ns=123456789))
    def record_double(path):return copy.deepcopy(current["records"][norm(path)])
    def read_double(path):return copy.deepcopy(current["payload"][norm(path)])
    def require(value,message):
        if not value:raise AssertionError(message)
    env=dict(constants,Path=PathDouble,P=P,S=P/"outputs/parakeet_turkish",ROOT=RealPath("E:/gpu_training_scratchpad"),BRANCH=BRANCH,
        W=HERE.parent,X=HERE,POLICY="authoritative_current_plan_identity_and_main_path_attribution_policy",PRIOR="prior_resident_RAM_source37_postDEV204000_throughput_transition",
        REPAIR="original_CPU14_startup_reader7_recovery_policy",FAILURE="prior_resident_RAM_source39_startup212000_failure",EXTRA_HOST_BYTES=80*1024**2,
        copy=copy,psutil=SimpleNamespace(Process=ProcessDouble,NoSuchProcess=MissingProcess),record=record_double,read_json=read_double,require=require)
    functions=[n for n in helper_tree.body if isinstance(n,ast.FunctionDef)and n.name!="checked_current_plan_throughput_policy"]
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(helper_path)+"#METHOD_STUB_authorities","exec"),env)
    def policy_double(base):
        env["legacy_source37_configuration"](base);env["checked_startup_reader_repair_policy"](base)
        return ("METHOD_STUB_identity","METHOD_STUB_view","METHOD_STUB_phase","METHOD_STUB_pressure")
    env["checked_current_plan_throughput_policy"]=policy_double
    results=[]
    def check(name,mutate=None,positive=False):
        reset()
        if mutate is not None:mutate(current)
        try:
            value=env["checked_current_plan_throughput_checkpoint_resume"](constants["FAILURE_ROOT"],base=current["candidate"],folder=BRANCH,pointer_pin=constants["SOURCE_DERIVED_POINTER"])
        except (AssertionError,KeyError,MissingProcess)as error:
            assert not positive,(name,type(error).__name__,str(error))
            results.append(dict(case=name,outcome="reject",error_type=type(error).__name__));return
        assert positive,name
        assert value["full_checkpoint_restore_and_genuine_redo_required"]is True and value["unsaved_tail_bytes"]==0 and value["honest_original_optimizer_vs_journal_gap"]==0
        results.append(dict(case=name,outcome="positive"))
    repairkey="original_CPU14_startup_reader7_recovery_policy"
    cpu=lambda c:c["candidate"][repairkey]["CPU_startup_policy"]
    repair=lambda c:c["candidate"][repairkey]
    payload=lambda c,p:c["payload"][norm(p)]
    check("whole_actual_helper_positive_METHOD_STUB",positive=True)
    check("missing_actual_adapter_review",lambda c:cpu(c).update(actual_new_adapter_CPU_control_review=None))
    check("wrong_adapter_source",lambda c:cpu(c)["adapter_source"].update(sha256="0"*64))
    check("wrong_worker_source",lambda c:cpu(c)["original_CPU14_source"].update(sha256="0"*64))
    check("startup8_reserve_reduced",lambda c:cpu(c).update(additional_host_reservation_bytes=8388607))
    check("original72_plus8_total_reduced",lambda c:repair(c).update(total_additional_native_host_bytes=80*1024**2-1))
    check("per_update_observer_requested",lambda c:cpu(c).update(no_per_update_observer=False))
    check("CPU_review_not_accepted",lambda c:payload(c,fakeCPU["path"]).update(accepted=False))
    check("CPU_full_packet_false",lambda c:payload(c,fakeCPU["path"]).update(full_first_packet_passed=False))
    check("CPU_known_deaths_false",lambda c:payload(c,fakeCPU["path"]).update(all_exact_known_identities_dead=False))
    check("reader_review_missing",lambda c:repair(c).update(actual_reader_replacement_ROOT=None))
    check("reader_mirror_dead",lambda c:payload(c,fakeReader["path"]).update(mirror_thread_alive=False))
    check("reader_HTTP_notready",lambda c:payload(c,fakeReader["path"]).update(HTTP_ready=False))
    check("reader_error",lambda c:payload(c,fakeReader["path"]).update(error="METHOD_STUB_timeout"))
    check("newreader_wrongctime",lambda c:c["live"][991].update(ctime=4.0))
    check("newlauncher_wrongppid",lambda c:c["live"][992].update(ppid=994))
    check("current_status_stale_pid",lambda c:payload(c,fake_status).update(pid=33388))
    check("current_status_mirror_dead",lambda c:payload(c,fake_status).update(mirror_thread_alive=False))
    check("current_status_unready",lambda c:payload(c,fake_status).update(reader_ready=False))
    check("old_native_missing_death",lambda c:payload(c,constants["FAILURE_ROOT"]["path"])["all_recorded_old_runtime_control_sealer_identities_exact_dead"].pop(0))
    check("old_native_stillalive",lambda c:c["live"].update({34788:dict(ctime=1792842518.865024,ppid=32332)}))
    check("false_originalgap0",lambda c:payload(c,constants["FAILURE_ROOT"]["path"]).update(actual212000_saved212000_gap0=False))
    check("fabricated_primarycause",lambda c:payload(c,constants["FAILURE_ROOT"]["path"]).update(underlying_new_CPU_startup_cause="NumPy"))
    check("fabricated_inner_OSexit",lambda c:payload(c,constants["FAILURE_ROOT"]["path"]).update(separate_native_CPU_wrapper_numeric_OS_exits=0))
    check("saved_not207",lambda c:payload(c,constants["SAVED_ROOT"]["path"]).update(weights=206))
    check("saved_RNG_missing",lambda c:payload(c,constants["SAVED_ROOT"]["path"]).update(all4RNG=False))
    check("pointer_not212000",lambda c:payload(c,constants["SOURCE_DERIVED_POINTER"]["path"]).update(committed_updates=212001))
    check("pointer_diagnostic",lambda c:payload(c,constants["SOURCE_DERIVED_POINTER"]["path"]).update(diagnostic_only=True))
    check("pointer_prefixoffset",lambda c:payload(c,constants["SOURCE_DERIVED_POINTER"]["path"])["journal_prefix"].update(offset=426026871))
    check("pointer_prefixauthority",lambda c:payload(c,constants["SOURCE_DERIVED_POINTER"]["path"])["journal_prefix"].update(sha256="0"*64))
    check("missing_ROOTbound_journalSTAT",lambda c:repair(c).update(closed_original_journal_STAT=None))
    check("journalSTAT_changedbytes",lambda c:c["stat"].update(bytes=426026871))
    check("journalSTAT_changedmtime",lambda c:c["stat"].update(mtime_ns=123456790))
    check("reader_otherjournal",lambda c:payload(c,fakeReader["path"]).update(journal_path="METHOD_STUB_OTHER"))
    check("science_LR_extra_mutation",lambda c:c["candidate"].update(METHOD_STUB_unreviewed_LR="changed"))
    # Actual generated DRIVER selector AST, no imports/model/native/restore calls.
    driver=HERE/"run_ASR18_BPE9223_resident_RAM_training_v40.py"
    wrapper=HERE/"run_ASR18_BPE9223_resident_RAM_autocontinue_v41.py"
    dt=ast.parse(driver.read_text(encoding="utf-8-sig"));wt=ast.parse(wrapper.read_text(encoding="utf-8-sig"))
    dm=next(n for n in dt.body if isinstance(n,ast.FunctionDef)and n.name=="main")
    start=next(i for i,n in enumerate(dm.body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=="throughput_active"for t in n.targets))
    finish=next(i for i,n in enumerate(dm.body[start:],start)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=="initial_updates"for t in n.targets))
    selector=ast.Module(body=dm.body[start:finish+1],type_ignores=[])
    names=("prior_process","ninth_original_resume","gc_refresh_resume","eighth_original_resume","fresh_signature_resume","canonical_archive_resume","lookahead_resume","cushion225_resume","postrestore_selector_resume","packet_credit_resume","native_headroom_resume","abnormal_resume","source_transition_resume","original_interval_resume","DEV_frontend_resume","DEV_admission_resume","periodic_quiescence_resume")
    for invocation,want in ((28,True),(27,False),(29,False)):
        reset();args=SimpleNamespace(**{k:None for k in names},throughput_resume=constants["FAILURE_ROOT"]["path"],invocation=invocation,pointer=constants["SOURCE_DERIVED_POINTER"]["path"])
        lane=dict(env,a=args,candidate_base=current["candidate"],folder=BRANCH)
        try:exec(compile(selector,str(driver)+"#actual_SELECTOR_METHOD_STUB","exec"),lane)
        except AssertionError:assert not want
        else:assert want and lane["initial_updates"]==212000
        results.append(dict(case="actual_driver_selector_invocation"+str(invocation),outcome="positive"if want else"reject"))
    # Actual wrapper command AST for initial abnormal lane and original normal continuation.
    wm=next(n for n in wt.body if isinstance(n,ast.FunctionDef)and n.name=="main")
    loop=next(n for n in wm.body if isinstance(n,ast.While))
    ci=next(i for i,n in enumerate(loop.body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=="command"for t in n.targets))
    commands=ast.Module(body=loop.body[ci:ci+2],type_ignores=[])
    for prior in (None,RealPath("E:/gpu_training_scratchpad/METHOD_STUB_prior.json")):
        args=SimpleNamespace(**{k:None for k in names},throughput_resume=constants["FAILURE_ROOT"]["path"],
            configuration="METHOD_STUB_same_config",latest_failure="METHOD_STUB_ORIGINAL_ACK260",queue_bytes=268435456,start_invocation=28)
        ns=dict(a=args,P=P,driver=driver,branch=BRANCH,pointer=RealPath(constants["SOURCE_DERIVED_POINTER"]["path"]),invocation=28 if prior is None else 29,prior=prior)
        exec(compile(commands,str(wrapper)+"#actual_COMMAND_METHOD_STUB","exec"),ns)
        argv=ns["command"];assert all(type(v)is str for v in argv)and None not in argv
        assert argv.count("--throughput-resume")==(1 if prior is None else 0)
        assert argv.count("--prior-process")==(0 if prior is None else 1)
        results.append(dict(case="actual_wrapper_initial_argv"if prior is None else"actual_wrapper_normal_continuation_argv",outcome="positive",argv=argv))
    base=ast.parse((P/"work/research/run_ASR18_BPE9223_resident_RAM_training_v39.py").read_text(encoding="utf-8-sig"))
    def digest_calls(tree):
        return [ast.dump(n,include_attributes=False)for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=="digest"and len(n.args)==1 and isinstance(n.args[0],ast.Name)and n.args[0].id=="plan"]
    assert digest_calls(base)==digest_calls(dt)and len(digest_calls(dt))==3
    assert "CPU14_startup_launch_adapter_v15"in driver.read_text()and "record(S/'fastmss_resident_feature_stream_v14.py')"in driver.read_text()
    assert 72*1024**2+8*1024**2==80*1024**2
    results.append(dict(case="all_three_original_fresh_digest_call_ASTs_exact_and_dynamic80",outcome="positive"))
    assert time.monotonic()-began<100
    final=memory();assert final["commit"]>=limits["runtime_commit"]and final["physical"]>=limits["runtime_physical"]and final["private"]<=536870912
    save(out/"result.json",dict(passed=True,results=results,count=len(results),scope="METHOD_STUB metadata authority only",
        actual_generated_helper_selector_wrapper_AST=True,all_original_science_restore_audio_GPU_draw_journal_control=False,
        old_identity_performance_not_rerun=True,real_adapter_full_firstpacket_proof_separate=True,
        entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),initial=initial,final=final,wall=time.monotonic()-began))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--ROOT-review",required=True);main(RealPath(p.parse_args().ROOT_review).resolve())
