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
initial=write_json(D/'source40_adapter_CPU_proof_SOURCE_initial_v1.json',dict(kind='SOURCE_preparation_initial_BEFORE_predicate',entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),sample=s,required=dict(commit=5399239296,physical=2649239296,private_max=536870912,E_free_min=100000000000),TORCH_EXTENSIONS_DIR=os.environ.get('TORCH_EXTENSIONS_DIR'),candidate_import_execution=False))
assert s['commit']>=5399239296 and s['physical']>=2649239296 and s['private']<=536870912 and s['E_free']>=100000000000

import difflib
def raw_diff(old,new,source_path,target_path,labels):
    work=old;records=[];shift=0
    for tag,a,b,c,d in difflib.SequenceMatcher(None,old,new,autojunk=True).get_opcodes():
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


scope_path=D/'ROOT_source40_reader7_saved212000_INACTIVE_SOURCE_preparation_scope_v2.json'
scope=json.loads(scope_path.read_text(encoding='utf-8'));assert scope['accepted']and not scope['actual_execution_admitted']
base=D/'CPU14_original_startup_firstpacket_CPU_proof_v4.py';old=base.read_bytes()
assert len(old)==19717 and sha(old)=='5c786ef91f133ff353ac6eca8c9491f284056331b4c1c7c992c39f0c741e6de3'
target=D/'CPU14_startup_adapter_firstpacket_CPU_proof_v5.py'
watchbase=D/'owned_CPU_parent_typed_proof_watchdog_v2.py';watchold=watchbase.read_bytes();watchtarget=D/'owned_CPU_startup_adapter_proof_watchdog_v3.py'
names=dict(raw='source40_adapter_CPU_proof_full_RAW_AST_v1.json',handoff='source40_adapter_CPU_proof_SOURCE_handoff_v1.json',result='source40_adapter_CPU_proof_SOURCE_result_v1.json')
assert not target.exists()and not watchtarget.exists()and not any((D/n).exists()for n in names.values())
text=old.decode('utf-8-sig');edits=[]
def replace(old,new,n=1):
    global text
    if text.count(old)==0 and '\n'in old and text.count(old.replace('\n','\r\n'))==n:old=old.replace('\n','\r\n')
    assert text.count(old)==n,(old,text.count(old),n)
    text=text.replace(old,new);edits.append(dict(old=old,new=new,count=n))
replace('INACTIVE SOURCE v3. New parent-failure scalar coverage; separate ROOT scope required.','INACTIVE SOURCE v5. REAL launch adapter/runner startup coverage; separate ROOT scope required.')
replace('RUNNER=HERE/"CPU14_owned_early_startup_receipt_runner_v1.py"','RUNNER=HERE/"CPU14_owned_startup_only_receipt_runner_v2.py"\nADAPTER=HERE/"CPU14_startup_launch_adapter_v15.py"')
replace('ROOT_CPU14_original_startup_firstpacket_exact_ONE_CPU_scope_v1','ROOT_CPU14_startup_adapter_firstpacket_exact_ONE_CPU_scope_v1')
replace('# Original native225/FULL+72+reader256+CPU2.5GB+external512+receipt1MiB.','# Original CPU proof minimum already has old1MiB receipt; replace1->8 and add reader API16+projector64, never count old1 twice.')
replace('>=8437158784','>=8528382208')
replace('>=5687158784','>=5778382208')
replace('    assert scope["runner"]==pin(RUNNER)','    assert scope["CPU_startup_runner_source"]==pin(RUNNER)and scope["adapter_source"]==pin(ADAPTER)\n    assert Path(scope["CPU_receipt_root"]).resolve()==root')
replace('    spec=importlib.util.spec_from_file_location("owned_original_CPU14_parent",scope["CPU14_source"]["path"])\n    cpu=importlib.util.module_from_spec(spec);spec.loader.exec_module(cpu)',
'''    spec=importlib.util.spec_from_file_location("owned_real_CPU14_startup_adapter",ADAPTER)
    cpu=importlib.util.module_from_spec(spec);spec.loader.exec_module(cpu)
    assert cpu.ResidentFeatureProducer.__init__.__globals__ is cpu._namespace
    assert type(cpu._namespace["subprocess"]).__name__=="_OwnedLaunchSubprocess"
    cpu.configure_startup_receipts(scope["CPU_startup_policy"],scope["configuration_pin"],startup_scope_path=scope_path)
    save(root/"actual_adapter_private_namespace_binding.json",dict(adapter=pin(ADAPTER),runner=pin(RUNNER),
        original_CPU14=scope["CPU14_source"],actual_original_init_globals_is_private_namespace=True,
        actual_launch_subprocess_class=type(cpu._namespace["subprocess"]).__name__,proof_global_Popen_override=False))''')
replace('    receipts=root/"worker_receipts"','    receipts=root/"original_CPU14"/"early_startup_receipts"')
# Remove the ENTIRE old proof-only Popen rerouter; do not bypass candidate adapter.
start=text.index('    original_popen=cpu.subprocess.Popen')
end=text.index('    # Proof-only scalar witness',start)
removed=text[start:end];text=text[:start]+text[end:];edits.append(dict(old=removed,new='',count=1))
replace('''        cpu.subprocess.Popen=owned_entry
        try:stream=cpu.ResidentFeatureProducer(scope["configuration_pin"],root/"original_CPU14",queue_bytes=scope["original_queue_bytes"],guard=guard)
        finally:cpu.subprocess.Popen=original_popen
        assert len(launchers)==1''',
'''        # The REAL adapter proxy makes the sole ORIGINAL Popen call. Passive
        # identity observation occurs only AFTER that original constructor returns.
        stream=cpu.ResidentFeatureProducer(scope["configuration_pin"],root/"original_CPU14",queue_bytes=scope["original_queue_bytes"],guard=guard)
        try:
            identity=who(psutil.Process(stream.process.pid));launchers.append(identity)
            save(root/"actual_parent_Popen_identity.json",dict(identity=identity,
                argv=psutil.Process(stream.process.pid).cmdline(),real_adapter_path=True,proof_Popen_override=False))
        except psutil.NoSuchProcess:
            save(root/"actual_parent_Popen_identity.json",dict(pid=stream.process.pid,ctime="UNKNOWN_ALREADY_EXITED",
                Popen_poll=stream.process.poll(),real_adapter_path=True,proof_Popen_override=False))
        capture()''')
replace('        cpu.subprocess.Popen=original_popen\n','')
# Lifetime capture ancestry allows an already-dead exact launcher without inventing its identity.
replace('''                assert any((x["pid"],x["ctime"])in {(a["pid"],a["ctime"])for a in launchers}for x in lineage)''',
'''                assert any((x["pid"],x["ctime"])in {(a["pid"],a["ctime"])for a in launchers}for x in lineage) or child.ppid()==os.getpid()''')
replace('''        save(root/"first_packet_result.json",result)''',
'''        receipt_paths=sorted(receipts.glob("*.json"));assert len(receipt_paths)<=130
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
        save(root/"first_packet_result.json",result)''')
replace('owned_CPU_parent_typed_proof_watchdog_v2.py','owned_CPU_startup_adapter_proof_watchdog_v3.py')
new=text.encode()
# Independent watchdog remains the same algorithm/owned cleanup/family/private/time.
wt=watchold.decode('utf-8-sig');watch_changes=[
('ROOT_CPU14_original_startup_firstpacket_exact_ONE_CPU_scope_v1','ROOT_CPU14_startup_adapter_firstpacket_exact_ONE_CPU_scope_v1'),
('"prospective_commit":8437158784','"prospective_commit":8528382208'),('"prospective_physical":5687158784','"prospective_physical":5778382208'),
('"runtime_commit":5918723328','"runtime_commit":6009946752'),('"runtime_physical":5418723328','"runtime_physical":5509946752')]
for a,b in watch_changes:assert wt.count(a)==1;wt=wt.replace(a,b)
wn=wt.encode()
tp=write_source(target,new);wp=write_source(watchtarget,wn)
raw=write_json(D/names['raw'],dict(kind='ASR18_startup_adapter_actual_path_CPU_SOURCE_full_RAW_AST_v1',
    records=[raw_diff(old,new,base,target,edits),raw_diff(watchold,wn,watchbase,watchtarget,watch_changes)],whole_byte_inverse_exact=True,
    original_fail_scalar_method_and_submit_ready_next_release_abort_unchanged=True,
    no_proof_global_subprocess_or_Popen_rerouter=True))
config=json.loads((D/'source40_saved212000_INACTIVE_configuration_v1.json').read_text())
policy=config['original_CPU14_startup_reader7_recovery_policy']['CPU_startup_policy']
# Future actual scope authorizes original inputs. No plan/audio/configuration BODY
# is read or fabricated by this SOURCE preparation.
proposal=dict(kind='ROOT_CPU14_startup_adapter_firstpacket_exact_ONE_CPU_scope_v1',accepted=False,reviewer='/root',
    fixture_source=tp,watchdog_source=wp,adapter_source=pin(D/'CPU14_startup_launch_adapter_v15.py'),
    CPU_startup_runner_source=pin(D/'CPU14_owned_startup_only_receipt_runner_v2.py'),CPU14_source=pin(P/'outputs/parakeet_turkish/fastmss_resident_feature_stream_v14.py'),
    configuration_pin=None,plan_pin=None,exact_expected_first_packet_rows=None,CPU_receipt_root=None,CPU_startup_policy=policy,
    output_directory=None,external_output_directory=None,original_plan_audio_execution_explicitly_admitted=False)
resource=dict(original_CPUproof4_minimum=dict(prospective_commit=8437158784,prospective_physical=5687158784,runtime_commit=5918723328,runtime_physical=5418723328),
    original_receipt_already_in_minimum=1048576,replacement_startup_reservation=8388608,receipt_delta=7340032,
    reader_cache256_already_original=True,reader_API16_plus_projector64_additive=83886080,total_addition_above_old_minimum=91226112,
    new_minimum=dict(prospective_commit=8528382208,prospective_physical=5778382208,runtime_commit=6009946752,runtime_physical=5509946752),
    independent_ROOT_metadata_review_if_concurrent_additional=536870912,
    unchanged_caps=dict(worker_private=8000000000,family_private=8000000000,parent_private=536870912,watchdog_private=536870912,
        reader_private=1500000000,packet_timeout=300,native_stall=600,body_wall=360,external_wall=420),
    no_reservation_or_private_cap_freed_or_raised=True,actual_fit='UNKNOWN until separately admitted actual full-family run')
handoff=write_json(D/names['handoff'],dict(kind='ASR18_startup_adapter_CPU_proof_INACTIVE_SOURCE_handoff_v1',SOURCE_only=True,actual_execution_admitted=False,
    proof=tp,watchdog=wp,RAW=raw,proposed_future_scope_template=proposal,
    original_CPU03_ROOT=scope['ROOT_faithful_CPU03'],new_actual_path_proof_required=True,
    exact_audio_input_execution_coverage='Original up to3 speculative updates/48 rows, first complete16 packet; no audio/data read here',
    resources=resource,delta='Remove old proof-only Popen path entirely; actual adapter private namespace -> ORIGINAL Popen -> runner2 -> ORIGINAL CPU14 worker',
    original_environment_and_P_parent_E_worker_paths_unchanged=True,all_original_fulltargets_credit_refdrop_abort_pipe_handle_checks_preserved=True,
    unknowns=['new actual proof acceptance','actual firstpacket/lifetime outcomes','exact future runtime configuration/plan/input bindings'],
    initial=initial,entry=who(psutil.Process()),launcher=who(psutil.Process().parent())))
assert time.monotonic()-t0<180
result=write_json(D/names['result'],dict(passed=True,SOURCE_only=True,proof=tp,watchdog=wp,RAW=raw,handoff=handoff,initial=initial,
    entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),elapsed=time.monotonic()-t0,final=sample()))
print(json.dumps(dict(proof=tp,watchdog=wp,RAW=raw,handoff=handoff,result=result,initial=initial,entry=who(psutil.Process()),launcher=who(psutil.Process().parent())),indent=2))

