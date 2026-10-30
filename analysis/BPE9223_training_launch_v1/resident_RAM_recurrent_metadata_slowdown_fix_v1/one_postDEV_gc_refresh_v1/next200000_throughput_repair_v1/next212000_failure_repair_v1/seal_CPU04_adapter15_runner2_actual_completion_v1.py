import ctypes,hashlib,json,os,shutil,time
from pathlib import Path
import psutil
X=Path(r"E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1/resident_RAM_recurrent_metadata_slowdown_fix_v1/one_postDEV_gc_refresh_v1/next200000_throughput_repair_v1/next212000_failure_repair_v1")
def who(p):return dict(pid=p.pid,ctime=p.create_time(),ppid=p.ppid())
def pin_bytes(p,b):return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def save(p,v):
    raw=(json.dumps(v,ensure_ascii=False,allow_nan=False,indent=2)+"\n").encode()
    with p.open("xb")as f:f.write(raw);f.flush();os.fsync(f.fileno())
    return pin_bytes(p,raw)
def sample():
    class M(ctypes.Structure):
        _fields_=[("length",ctypes.c_ulong),("load",ctypes.c_ulong)]+[(n,ctypes.c_ulonglong)for n in ("totalphysical","physical","commitlimit","commit","totalvirtual","virtual","extended")]
    m=M();m.length=ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):raise ctypes.WinError()
    return dict(commit=m.commit,physical=m.physical,private=psutil.Process().memory_info().private,E_free=shutil.disk_usage("E:/gpu_training_scratchpad").free)
psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
began=time.monotonic();initial=sample()
ip=save(X/"CPU04_adapter15_runner2_actual_completion_initial_BEFORE_predicate_v1.json",dict(entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),sample=initial,required=dict(commit=9065255808,physical=6315255808,self_private=536870912,E_free=100000000000)))
assert initial["commit"]>=9065255808 and initial["physical"]>=6315255808 and initial["private"]<=536870912 and initial["E_free"]>=100000000000
tool=dict(tool="exec_command",chunk_id="dcbab7",wall_time_seconds=7.2395777,exit_code=0,original_token_count=0,output="",session_id=None,
    interpretation="Actual TOOL return only. Direct top Python numeric code is separately persisted; do not infer worker/native/production OS status.")
toolpin=save(X/"CPU04_adapter15_runner2_actual_owned_TOOL_result_v1.json",tool)
roots=[X/"CPU04_adapter15_runner2_firstpacket_actual_v1",X/"CPU04_adapter15_runner2_external_actual_v1"]
artifacts=[];values={};identities={};refs={}
def walk(v,path):
    if isinstance(v,dict):
        if type(v.get("pid"))is int and (type(v.get("ctime"))in (float,int)or type(v.get("create_time"))in (float,int)):
            ct=v.get("ctime",v.get("create_time"));q=dict(pid=v["pid"],ctime=ct)
            if type(v.get("ppid"))is int:q["ppid"]=v["ppid"]
            key=(q["pid"],q["ctime"])
            if key in identities and "ppid"in q and "ppid"in identities[key]:assert q["ppid"]==identities[key]["ppid"]
            identities.setdefault(key,q);refs.setdefault(key,[]).append(path)
        for k,x in v.items():walk(x,path+"."+str(k))
    elif isinstance(v,list):
        for i,x in enumerate(v):walk(x,path+"["+str(i)+"]")
for root in roots:
    files=sorted(root.rglob("*"))
    assert len(files)<=128
    for p in files:
        if not p.is_file():continue
        rel=str(p.relative_to(X))
        assert p.suffix.lower()in (".json",".log"),"Only NEW known scalar artifacts, no tensors/audio/data"
        b=p.read_bytes();assert len(b)<=262144
        artifact=dict(pin=pin_bytes(p,b),relative=rel)
        if p.suffix==".json":
            v=json.loads(b);values[rel]=v;artifact["value"]=v;walk(v,rel)
        else:artifact["text_utf8"]=b.decode("utf-8",errors="replace")
        artifacts.append(artifact)
np=X/"CPU04_adapter15_runner2_direct_numeric_exit_v1.json";nb=np.read_bytes();numeric=json.loads(nb)
ep=str(roots[1].relative_to(X)/"external_result.json")
bp=str(roots[0].relative_to(X)/"result.json")
fp=str(roots[0].relative_to(X)/"first_packet_result.json")
external=values[ep];body=values[bp];first=values[fp]
deaths=[]
for key,q in identities.items():
    try:
        p=psutil.Process(q["pid"]);same=abs(p.create_time()-q["ctime"])<.001
        record=dict(identity=q,exact_dead=not same)
        if same:record["current"]=who(p);record["recorded_PPID_matches"]="ppid"not in q or p.ppid()==q["ppid"]
    except psutil.NoSuchProcess:record=dict(identity=q,exact_dead=True)
    except BaseException as e:record=dict(identity=q,exact_dead="UNKNOWN",error=type(e).__name__+":"+str(e)[:500])
    record["source_artifact_references"]=refs[key];deaths.append(record)
summary=dict(top_TOOL_exit=0,direct_top_Python_numeric_OS_exit=numeric["direct_Python_numeric_OS_exit"],
    external_Popen_returncode=external.get("Popen_returncode"),external_guard_error=external.get("guard_error"),external_cleanup=external.get("cleanup"),
    body_passed=body.get("passed"),body_primary=body.get("primary"),body_cleanup=body.get("cleanup"),
    first_packet_rows=len(first["first_complete_packet_rows"]),actual_real_adapter_path=first.get("real_adapter_path"),
    proof_global_Popen_override=first.get("proof_global_Popen_override"),startup_hooks_once=first.get("all_four_hooks_observed_once"),
    receipt_bytes=first.get("startup_receipt_bytes"),raw_feature_refs_dead=first.get("raw_feature_refs_dead"),
    original_release_after_refdrop=first.get("original_release_after_refdrop"),owner_credit=first.get("owner_credit"),
    sender_joined=first.get("sender_joined"),receiver_joined=first.get("receiver_joined"),pipes_closed=first.get("pipes_closed"),log_closed=first.get("log_closed"),
    child_Popen_code=first.get("child_Popen_code"),child_credit_method_close=first.get("child_credit_method_close"),normal0_claimed=first.get("normal0_claimed"),
    all_known_original_job_identities_exact_dead=all(r["exact_dead"]is True for r in deaths))
cp=save(X/"CPU04_adapter15_runner2_actual_completion_v1.json",dict(kind="owned_CPU04_actual_completion_metadata_v1",scope_path=str(X/"ROOT_CPU04_adapter15_runner2_exact_ONE_CPU_scope_v1.json"),
    tool=toolpin,numeric=dict(pin=pin_bytes(np,nb),value=numeric),summary=summary,new_output_pins=[a["pin"]for a in artifacts],new_scalar_artifacts=artifacts,
    all_known_original_job_identity_deaths=deaths,sealer=who(psutil.Process()),sealer_launcher=who(psutil.Process().parent()),initial=ip,
    no_original_plan_configuration_audio_journal_checkpoint_or_closed_artifacts_reread=True,
    honest_limits=dict(actual_finite_abort_is_not_normal0=True,no_inner_OS_code_inference_from_TOOLS_or_deaths=True,production212_underlying_cause="UNKNOWN_UNCHANGED",no_production_reader_training_restore_or_speed_fix_claim=True),
    final_sample=sample(),elapsed=time.monotonic()-began))
print(json.dumps(dict(completion=cp,summary=summary,new_artifact_count=len(artifacts),known_original_identity_count=len(deaths),
    sealer=who(psutil.Process()),sealer_launcher=who(psutil.Process().parent()),initial=ip),indent=2))

