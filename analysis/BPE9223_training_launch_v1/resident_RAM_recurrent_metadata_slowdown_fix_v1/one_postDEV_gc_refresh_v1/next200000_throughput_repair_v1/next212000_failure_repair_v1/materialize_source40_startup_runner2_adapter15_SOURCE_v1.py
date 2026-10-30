import ast,ctypes,hashlib,json,os,pathlib,shutil,time
import psutil
P=pathlib.Path(r'C:/Users/ardao/Documents/Codex/2026-10-10/i-x20')
D=pathlib.Path(r'E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1/resident_RAM_recurrent_metadata_slowdown_fix_v1/one_postDEV_gc_refresh_v1/next200000_throughput_repair_v1/next212000_failure_repair_v1')
psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
class MS(ctypes.Structure):
    _fields_=[("length",ctypes.c_ulong),("load",ctypes.c_ulong)]+[(n,ctypes.c_ulonglong)for n in ("totalphysical","physical","commitlimit","commit","totalvirtual","virtual","extended")]
def sample():
    m=MS();m.length=ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):raise ctypes.WinError()
    return dict(commit=m.commit,physical=m.physical,private=psutil.Process().memory_info().private,E_free=shutil.disk_usage('E:/gpu_training_scratchpad').free)
def who(p):return dict(pid=p.pid,ctime=p.create_time(),ppid=p.ppid())
def sha(b):return hashlib.sha256(b).hexdigest()
def pin_bytes(p,b):return dict(path=str(p),bytes=len(b),sha256=sha(b))
def pin(p):return pin_bytes(p,p.read_bytes())
def astsha(b):return sha(ast.dump(ast.parse(b.decode('utf-8-sig')),include_attributes=False).encode())
def write_source(p,b):
    ast.parse(b.decode('utf-8-sig'))
    with p.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
    return pin_bytes(p,b)
def write_json(p,obj):
    with p.open('x',encoding='utf-8',newline='\n')as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    return pin(p)
t0=time.monotonic();s=sample()
initial=write_json(D/'source40_startup_runner_adapter_SOURCE_initial_v1.json',dict(kind='SOURCE_preparation_initial_BEFORE_predicate',entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),sample=s,required=dict(commit=5399239296,physical=2649239296,private_max=536870912,E_free_min=100000000000),TORCH_EXTENSIONS_DIR=os.environ.get('TORCH_EXTENSIONS_DIR'),candidate_import_execution=False))
assert s['commit']>=5399239296 and s['physical']>=2649239296 and s['private']<=536870912 and s['E_free']>=100000000000

import difflib
def raw_diff(old,new,source_path,target_path,labels):
    work=old;records=[];shift=0
    for tag,a,b,c,d in difflib.SequenceMatcher(None,old,new,autojunk=False).get_opcodes():
        if tag=='equal':continue
        at=a+shift;before=work
        assert work[at:at+(b-a)]==old[a:b]
        work=work[:at]+new[c:d]+work[at+(b-a):]
        records.append(dict(offset_before=at,old_utf8=old[a:b].decode(),new_utf8=new[c:d].decode(),before_sha256=sha(before),after_sha256=sha(work)))
        shift+=(d-c)-(b-a)
    assert work==new
    for e in reversed(records):
        at=e['offset_before'];x=e['new_utf8'].encode();y=e['old_utf8'].encode()
        assert work[at:at+len(x)]==x
        work=work[:at]+y+work[at+len(x):]
    assert work==old
    return dict(base=pin_bytes(source_path,old),candidate=pin_bytes(target_path,new),ordered_edits=records,whole_byte_inverse_exact=True,semantic_changes=labels,base_AST_sha256=astsha(old),candidate_AST_sha256=astsha(new))

runner_code="\"\"\"INACTIVE original-CPU14 startup-only receipt entry; separately ROOT-admitted use only.\nBinary stdout and the original worker remain unchanged. No per-update observer.\n\"\"\"\nimport argparse\nimport functools\nimport hashlib\nimport json\nimport os\nfrom pathlib import Path\nimport runpy\nimport sys\nimport threading\nimport time\n\nROOT=Path(r\"E:/gpu_training_scratchpad\").resolve()\nORIGINAL_SHA=\"e3d01517ac3af9e481a27b0127474c7a8fbc2d9351f7fcc6aefcf36691675d0e\"\nMAXIMUM_RECEIPT_BYTES=1048576\ndef read(p):return json.loads(Path(p).read_text(encoding=\"utf-8-sig\"))\ndef pin(p):\n    p=Path(p);b=p.read_bytes()\n    return dict(path=str(p.resolve()),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())\ndef run(args):\n    import psutil\n    scope=read(args.ROOT_startup_scope)\n    assert scope[\"accepted\"]is True and scope[\"reviewer\"]==\"/root\"\n    assert scope[\"kind\"]in (\n        \"ROOT_CPU14_startup_adapter_firstpacket_exact_ONE_CPU_scope_v1\",\n        \"ROOT_ASR18_source40_saved212000_training_exact_ONE_scope_v1\")\n    source=Path(args.original_source).resolve()\n    assert pin(source)==scope[\"CPU14_source\"]and pin(source)[\"sha256\"]==ORIGINAL_SHA\n    assert pin(__file__)==scope[\"CPU_startup_runner_source\"]\n    assert pin(args.configuration)==json.loads(args.expected_configuration_pin)\n    root=Path(args.receipt_directory).resolve()\n    allowed=Path(scope[\"CPU_receipt_root\"]).resolve()\n    assert root.is_relative_to(allowed)and root!=allowed and root.is_relative_to(ROOT)\n    root.mkdir(exist_ok=False)\n    me=psutil.Process();lock=threading.Lock()\n    counter=[0];first_error=[False];write_error=[False];written_bytes=[0]\n    observed=set();wrappers={};namespace=None\n    def atomic(name,value):\n        raw=json.dumps(value,ensure_ascii=False,allow_nan=False).encode()\n        # Include a full maximum temporary record ABOVE all already saved bytes.\n        if len(raw)>65536 or written_bytes[0]+len(raw)+65536>MAXIMUM_RECEIPT_BYTES:\n            raise ValueError(\"Startup receipts plus owned temporary exceed1MiB\")\n        temporary=root/(name+\".owned_tmp\")\n        with temporary.open(\"xb\")as f:f.write(raw);f.flush();os.fsync(f.fileno())\n        os.replace(temporary,root/name);written_bytes[0]+=len(raw)\n    atomic(\"earliest_owned_worker_identity.json\",dict(pid=me.pid,ctime=me.create_time(),ppid=me.ppid(),\n        argv=list(sys.argv),original_source=pin(source),runner_source=pin(__file__),\n        CUDA_VISIBLE_DEVICES=os.environ.get(\"CUDA_VISIBLE_DEVICES\"),\n        TORCH_EXTENSIONS_DIR=os.environ.get(\"TORCH_EXTENSIONS_DIR\")))\n    started=time.monotonic()\n    def record(phase,**scalars):\n        try:\n            with lock:\n                if counter[0]>=128:return\n                counter[0]+=1\n                atomic(str(counter[0]).zfill(3)+\".json\",dict(\n                    index=counter[0],phase=phase,elapsed=time.monotonic()-started,**scalars))\n        except BaseException:\n            write_error[0]=True\n            try:sys.stderr.write(\"OWNED_CPU_STARTUP_RECEIPT_WRITE_FAILED\\n\");sys.stderr.flush()\n            except BaseException:pass\n    def describe(error):\n        try:message=str(error)[:2048]\n        except BaseException:message=\"UNPRINTABLE_ORIGINAL_EXCEPTION\"\n        return dict(exception_type=type(error).__name__,message=message)\n    def first_exception(error,phase):\n        if first_error[0]:return\n        first_error[0]=True;record(\"first_original_exception\",at=phase,**describe(error))\n    def owned_timeout_stack():\n        frames=None;frame=None\n        try:\n            frames=sys._current_frames()\n            if len(frames)>128:\n                record(\"startup_timeout_stack_unavailable\",reason=\"Own thread-count bound\");return\n            frame=frames.get(threading.main_thread().ident);rows=[]\n            while frame is not None and len(rows)<64:\n                rows.append(dict(file=frame.f_code.co_filename[:512],function=frame.f_code.co_name[:128],line=frame.f_lineno))\n                frame=frame.f_back\n            record(\"own_startup_main_thread_timeout_stack\",frames=rows)\n        except BaseException as error:record(\"startup_timeout_stack_failed\",**describe(error))\n        finally:del frame,frames\n    timer=threading.Timer(295,owned_timeout_stack)\n    timer.name=\"owned-CPU14-startup-timeout-receipt\";timer.daemon=True;timer.start()\n    try:\n        record(\"before_original_module_definitions\")\n        exported=runpy.run_path(str(source),run_name=\"CPU14_owned_startup_definitions\")\n        namespace=exported[\"worker_main\"].__globals__\n        record(\"after_original_module_definitions\")\n        for name in (\"checked_resource_profile\",\"_startup\",\"_header\",\"_send\"):\n            original=namespace[name]\n            def make_wrapper(original,name):\n                @functools.wraps(original)\n                def observed_once(*positional,**keywords):\n                    # Detach BEFORE the first original call; future calls have\n                    # zero wrapper/clock/receipt overhead, even on failure.\n                    if namespace.get(name)is wrappers.get(name):namespace[name]=original\n                    wrappers.pop(name,None)\n                    record(\"before_first_\"+name)\n                    try:result=original(*positional,**keywords)\n                    except BaseException as error:\n                        first_exception(error,name);raise\n                    finally:\n                        observed.add(name)\n                        if len(observed)==4:timer.cancel()\n                    record(\"after_first_\"+name)\n                    return result\n                return observed_once\n            wrappers[name]=make_wrapper(original,name);namespace[name]=wrappers[name]\n        record(\"before_original_worker_main\")\n        code=exported[\"worker_main\"](args.configuration,args.private_limit)\n        record(\"after_original_worker_main\",returned_code=code,receipt_write_failed=write_error[0])\n        return code\n    except BaseException as error:\n        first_exception(error,\"module_or_worker_main\");raise\n    finally:\n        timer.cancel();timer.join(timeout=2)\n        if timer.is_alive():\n            try:sys.stderr.write(\"OWNED_STARTUP_TIMEOUT_THREAD_JOIN_UNKNOWN\\n\")\n            except BaseException:pass\n        if namespace is not None:\n            for name,wrapper in list(wrappers.items()):\n                if namespace.get(name)is wrapper:namespace[name]=wrapper.__wrapped__\n            wrappers.clear()\n        record(\"runner_finally\",receipt_write_failed=write_error[0],\n            startup_hooks_observed_once=sorted(observed),startup_wrappers_remaining=len(wrappers))\nif __name__==\"__main__\":\n    p=argparse.ArgumentParser()\n    p.add_argument(\"--ROOT-startup-scope\",required=True)\n    p.add_argument(\"--original-source\",required=True)\n    p.add_argument(\"--receipt-directory\",required=True)\n    p.add_argument(\"--expected-configuration-pin\",required=True)\n    p.add_argument(\"--worker\",action=\"store_true\",required=True)\n    p.add_argument(\"--configuration\",required=True)\n    p.add_argument(\"--private-limit\",type=int,default=8000000000)\n    raise SystemExit(run(p.parse_args()))\n"
adapter_code="\"\"\"INACTIVE E-only launch adapter; ORIGINAL CPU14 methods and binary transport.\nOnly a private runpy namespace gets a subprocess proxy; no installed/global patch.\n\"\"\"\nimport hashlib\nimport json\nfrom pathlib import Path\nimport runpy\n\nORIGINAL=Path(r\"C:/Users/ardao/Documents/Codex/2026-10-10/i-x20/outputs/parakeet_turkish/fastmss_resident_feature_stream_v14.py\")\nORIGINAL_SHA=\"e3d01517ac3af9e481a27b0127474c7a8fbc2d9351f7fcc6aefcf36691675d0e\"\nROOT=Path(r\"E:/gpu_training_scratchpad\").resolve()\nWORKER_PYTHON=Path(r\"E:/gpu_training_scratchpad/scratch/BPE_broad_CPU_preflight_env_v2/Scripts/python.exe\").resolve()\nCONTEXT=None\ndef pin(p):\n    p=Path(p);b=p.read_bytes()\n    return dict(path=str(p.resolve()),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())\nassert pin(ORIGINAL)[\"sha256\"]==ORIGINAL_SHA\n_exports=runpy.run_path(str(ORIGINAL),run_name=\"CPU14_private_launch_adapter_definitions\")\nResidentFeatureProducer=_exports[\"ResidentFeatureProducer\"]\n_namespace=ResidentFeatureProducer.__init__.__globals__\n_original_subprocess=_namespace[\"subprocess\"]\ndef configure_startup_receipts(policy,configuration_pin,*,startup_scope_path=None):\n    global CONTEXT\n    assert CONTEXT is None,\"One immutable per-parent launch context\"\n    assert policy[\"kind\"]==\"ASR18_original_CPU14_startup_receipts_launch_adapter_v1\"\n    assert policy[\"adapter_source\"]==pin(__file__)\n    assert policy[\"original_CPU14_source\"]==pin(ORIGINAL)\n    assert policy[\"worker_python_path\"]==str(WORKER_PYTHON)\n    assert policy[\"additional_host_reservation_bytes\"]==8388608\n    assert policy[\"maximum_receipt_bytes\"]==1048576 and policy[\"four_coarse_hooks_once\"]is True\n    assert policy[\"no_per_update_observer\"]is True\n    assert pin(configuration_pin[\"path\"])==configuration_pin\n    runner=Path(policy[\"runner_source\"][\"path\"]).resolve()\n    assert runner.is_relative_to(ROOT)and pin(runner)==policy[\"runner_source\"]\n    scope_path=Path(startup_scope_path or policy[\"execution_review_path\"]).resolve()\n    assert scope_path.is_relative_to(ROOT)\n    CONTEXT=dict(policy=dict(policy),configuration=dict(configuration_pin),runner=runner,scope_path=scope_path)\nclass _OwnedLaunchSubprocess:\n    def __getattr__(self,name):return getattr(_original_subprocess,name)\n    def Popen(self,argv,*positional,**keywords):\n        assert CONTEXT is not None,\"Typed startup policy must be configured before CPU launch\"\n        context=CONTEXT;configuration=context[\"configuration\"]\n        expected=[str(WORKER_PYTHON),\"-Xutf8\",\"-B\",str(ORIGINAL.resolve()),\n            \"--worker\",\"--configuration\",configuration[\"path\"],\"--private-limit\",\"8000000000\"]\n        actual=list(argv)\n        assert len(actual)==len(expected)\n        assert Path(actual[0]).resolve()==WORKER_PYTHON and Path(actual[3]).resolve()==ORIGINAL.resolve()\n        assert actual[1:3]==expected[1:3]and actual[4:]==expected[4:]\n        assert pin(configuration[\"path\"])==configuration\n        assert not positional and set(keywords)=={\"stdin\",\"stdout\",\"stderr\",\"env\",\"bufsize\",\"creationflags\"}\n        assert keywords[\"stdin\"]==_original_subprocess.PIPE and keywords[\"stdout\"]==_original_subprocess.PIPE\n        assert keywords[\"env\"][\"CUDA_VISIBLE_DEVICES\"]==\"-1\"and keywords[\"bufsize\"]==0\n        folder=Path(keywords[\"stderr\"].name).resolve().parent\n        assert folder.is_relative_to(ROOT)and folder.is_dir()\n        receipt=folder/\"early_startup_receipts\";assert not receipt.exists()\n        command=[actual[0],\"-Xutf8\",\"-B\",str(context[\"runner\"]),\n            \"--ROOT-startup-scope\",str(context[\"scope_path\"]),\"--original-source\",str(ORIGINAL.resolve()),\n            \"--receipt-directory\",str(receipt),\"--expected-configuration-pin\",\n            json.dumps(configuration,sort_keys=True,separators=(\",\",\":\")), *actual[4:]]\n        # Original Popen called EXACTLY ONCE; same pipes/env/private limit/credit.\n        return _original_subprocess.Popen(command,**keywords)\n_namespace[\"subprocess\"]=_OwnedLaunchSubprocess()\n"
scopepath=D/'ROOT_source40_reader7_saved212000_INACTIVE_SOURCE_preparation_scope_v2.json'
scope=json.loads(scopepath.read_text(encoding='utf-8-sig'))
assert pin(scopepath)['sha256']=='dc9fdafe5c07c222f3d92cdf03b379501a267fbef1c62539bdab27acee8eb695'
assert scope['actual_execution_admitted']is False
oldpath=pathlib.Path(scope['tested_early_runner']['path']);old=oldpath.read_bytes()
assert pin_bytes(oldpath,old)['sha256']==scope['tested_early_runner']['sha256']and len(old)==6296
originalpath=pathlib.Path(scope['base_CPU14']['path']);original=originalpath.read_bytes()
assert sha(original)==scope['base_CPU14']['sha256']and len(original)==36718
runnerpath=D/'CPU14_owned_startup_only_receipt_runner_v2.py'
adapterpath=D/'CPU14_startup_launch_adapter_v15.py'
rawpath=D/'source40_CPU_startup_runner2_adapter15_full_RAW_AST_v1.json'
handoffpath=D/'source40_CPU_startup_runner2_adapter15_SOURCE_handoff_v1.json'
resultpath=D/'source40_CPU_startup_runner2_adapter15_SOURCE_result_v1.json'
for p in (runnerpath,adapterpath,rawpath,handoffpath,resultpath):assert not p.exists(),str(p)
def guard():
    q=sample()
    assert q['commit']>=2612368384 and q['physical']>=2112368384 and q['private']<=536870912 and q['E_free']>=100000000000
    assert time.monotonic()-t0<180
    return q
guard()
runnerpin=write_source(runnerpath,runner_code.encode())
adapterpin=write_source(adapterpath,adapter_code.encode())
raw=raw_diff(old,runner_code.encode(),oldpath,runnerpath,[
'New typed actualCPU-adapter or production ROOT scope kinds, exact runtime configuration-pin operand and approvedCPUreceiptroot; original14 source pin unchanged.',
'Original four coarse startup wrappers detach BEFORE their FIRST original call; afterfirstcall zero per-update wrapper overhead. Original arguments/results/exception objects unchanged.',
'Fixed1MiB TOTAL serializedreceipt+ownedtemporary byte bound, not claimed Python-memory bound.',
'Original own-child startup timeout295/firsterror/earliest-before-import/normalbinary stdout remain; timer cancelled after4coarse first calls orfinally.',
'No numerical/native/transport/time/private/packet/GC change; newentrypath actualCPU proof remainsUNKNOWN.'])
oldtree=ast.parse(original.decode())
originalmethods={n.name:sha(ast.dump(n,include_attributes=False).encode())for n in ast.walk(oldtree)if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
rawobj=dict(kind='source40_CPU_startup_runner2_adapter15_RAW_AST_SOURCE_ONLY_v1',scope=pin(scopepath),
runner=raw,adapter_new_source=dict(base='NEW owned E-only adapter, no prior adapter source',candidate=adapterpin,
full_source=adapter_code,candidate_AST_sha256=astsha(adapter_code.encode()),whole_inverse_to_empty_declared_source=True),
original_CPU14=pin_bytes(originalpath,original),original_CPU14_method_AST_hashes=originalmethods,
original_CPU14_source_UNCHANGED=True,adapter_private_namespace_only=True,no_global_or_installed_subprocess_patch=True)
rawpin=write_json(rawpath,rawobj)
hostextra=8388608;originalextra=75497472
policy=dict(kind='ASR18_original_CPU14_startup_receipts_launch_adapter_v1',adapter_source=adapterpin,runner_source=runnerpin,
original_CPU14_source=scope['base_CPU14'],worker_python_path=str(pathlib.Path('E:/gpu_training_scratchpad/scratch/BPE_broad_CPU_preflight_env_v2/Scripts/python.exe').resolve()),
additional_host_reservation_bytes=hostextra,maximum_receipt_bytes=1048576,four_coarse_hooks_once=True,no_per_update_observer=True,
execution_review_path=str(D/'ROOT_source40_saved212000_final_ONE_training_execution_review_v1.json'),
actual_new_adapter_CPU_control_review=None,ROOT_original_faithful_CPU03=scope['ROOT_faithful_CPU03'])
handoff=dict(kind='source40_CPU_startup_runner2_adapter15_INACTIVE_SOURCE_piece_v1',actual_execution_admitted=False,
scope=pin(scopepath),initial=initial,runner=runnerpin,adapter=adapterpin,RAW=rawpin,original_CPU14=scope['base_CPU14'],
policy_template=policy,pending_policy_field='actual_new_adapter_CPU_control_review is UNKNOWN until separatelyROOTadmitted meaningful actualCPU+typed proof; production mustfailclosed whileNone.',
coverage=dict(accepted_CPU03='OriginalP native-parent interpreter/original Eworker/current CPU14/firstpacket16rows+2speculative inputs proved; closedROOT a188 only.',
unchanged='Original CPU14 definitions/methods including resource/profile/300wait/600credit/8GB/private/MAX2/LIVE1/_send/_receive/close are executed unchanged from exactS14 source; same numerical worker.',
new_UNPROVED=['Private runpy PARENTnamespace plus subprocess proxy intercept','Typed production authority/configpin operand/receiptroot checks','Startup-only single-use wrappers/timer lifetime/total receiptbudget','Full driver/helper/wrapper integration not yet generated orproved'],
worker_entry='Original Epython -Xutf8-B runner2 then original runpy CPU14.worker_main with same args/env/pipes/credit; Popencalledonce. No extra child process beyond original shim/body.',
parent_interpreter='Original P/work/nemo-native-env-v2/Scripts/python.exe remains required; no package/path mutation.',
diagnostic_limit='Four coarse hooks only beforefirstcall; _send firstready/_header firstcommand cover startup, notperupdate firstpacket profiling. Underlyingproduction212 startupcause remainsUNKNOWN.'),
resources=dict(original_source39_bytes=originalextra,proposed_startup_bookkeeping_bytes=hostextra,total_additional_native_bytes=originalextra+hostextra,
hard_serialized_receipt_and_temp_bytes=1048576,
proposal_not_actual_fit='8MiB conservatively holds receipt1MiB+runpy/adapter Pythonbookkeeping; no deterministic allocatorfitclaim. Originalfamily/privatecaps unchanged and actualproofrequired.',
native_full_pressure_commit=4518435456+originalextra+hostextra,native_full_pressure_physical=1768435456+originalextra+hostextra,
prospective_commit=5055306368+originalextra+hostextra,prospective_physical=2305306368+originalextra+hostextra,
Monitor_commit=2000000000+originalextra+hostextra,Monitor_physical=1500000000+originalextra+hostextra,
CPU_private_limit=8000000000,packet_seconds=300,credit_stall_seconds=600,
all_original225_FULL_packetqueue256_prefetch256_preserved=True,
remaining_integration='Apply total80MiB at ALL existing parentMonitor/admit/CPUstartup/DEV/nativefullpressure gates, no newcadence or privatecap. Originalworker independent guards preserved; wholefamilyreview mustbind coupledparentguards.'),
future_scope_fields=dict(CPU14_source=scope['base_CPU14'],CPU_startup_runner_source=runnerpin,CPU_receipt_root=None,
kind_choices=['ROOT_CPU14_startup_adapter_firstpacket_exact_ONE_CPU_scope_v1','ROOT_ASR18_source40_saved212000_training_exact_ONE_scope_v1'],
configuration_pin='Explicit argv operand mustexactlymatch actualruntimeconfig pin, originally authorized by typedparent/sourceconfig/full7runtimebindings.'),
source_only_proof_plan=['Exactoriginalclass methodASTs +privateproxyno globalpatch; ownedmetadataPopen double checksonecall/argv/env/pipe/config/root negatives.',
'Separately scoped faithful actualCPU through productionPparent+newadapter15+runner2+originalEworker/current SAMEplan361 all20 metadata+upto3originalspeculative packets48audioinputs; no original audio or plan read now.',
'Failclosure startup/constructor/earliestdeadchild/receiptwriteerror/canceljoin under original full guards; no retry/secondbound.'],
later_outputs='Driver40/wrapper41/typedcoherent212000/config/readerreplacement actor and typed fixture SOURCE remain inprogress, no activation from this piece.',
preparer=dict(entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),owned_children=[]),
final_sample=guard(),elapsed=time.monotonic()-t0)
handoffpin=write_json(handoffpath,handoff)
result=dict(kind='source40_startup_runner2_adapter15_SOURCE_result_v1',prepared=True,actual_execution=False,
runner=runnerpin,adapter=adapterpin,RAW=rawpin,handoff=handoffpin,initial=initial,
entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),final_sample=guard(),elapsed=time.monotonic()-t0)
resultpin=write_json(resultpath,result)
print(json.dumps(dict(runner=runnerpin,adapter=adapterpin,RAW=rawpin,handoff=handoffpin,result=resultpin,initial=initial,
entry=result['entry'],launcher=result['launcher'],ordered_runner_RAW=len(raw['ordered_edits'])),ensure_ascii=False))

