import os,json,hashlib,time,ctypes,shutil,psutil,ast,difflib,base64
from pathlib import Path
p=psutil.Process();p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
root=Path("E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1/resident_RAM_recurrent_metadata_slowdown_fix_v1/one_postDEV_gc_refresh_v1/next200000_throughput_repair_v1/next212000_failure_repair_v1");out=root/'ROOT_HUMAN_Recovery43_SOURCE_build_1820_v1';out.mkdir(exist_ok=False)
class M(ctypes.Structure):
 _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong)for n in ('totalphysical','physical','commitlimit','commit','totalvirtual','virtual','extended')]
start=time.monotonic()
def sample():
 m=M();m.length=ctypes.sizeof(m);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m));return dict(time=time.time(),pid=p.pid,ctime=p.create_time(),ppid=p.ppid(),commit=m.commit,physical=m.physical,private=p.memory_info().private,E_free=shutil.disk_usage('E:/gpu_training_scratchpad').free,elapsed=time.monotonic()-start)
def save(path,v):
 b=json.dumps(v,ensure_ascii=False,indent=2).encode();f=path.open('xb');f.write(b);f.flush();os.fsync(f.fileno());f.close();return dict(path=str(path),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
first_sample=sample()
initial=save(out/'initial_BEFORE_predicate.json',dict(sample=first_sample,limits=dict(prospective_commit=6028384896,prospective_physical=3278384896,runtime_commit=3509949440,runtime_physical=3009949440,self_private=536870912,wall=180)))
s=sample()
admission=save(out/'fresh_gate_numeric_BEFORE_predicate.json',dict(sample=s,initial=initial))
print(json.dumps(dict(record='numeric_before_predicate',first=first_sample,fresh=s,initial=initial,admission=admission),separators=(',',':')),flush=True)
for admission_sample in (first_sample,s):assert admission_sample['commit']>=6028384896 and admission_sample['physical']>=3278384896 and admission_sample['private']<=536870912 and admission_sample['E_free']>=100000000000 and admission_sample['elapsed']<180
def guard():
 s=sample();assert s['commit']>=3509949440 and s['physical']>=3009949440 and s['private']<=536870912 and s['E_free']>=100000000000 and s['elapsed']<180;return s
def sha(b):return hashlib.sha256(b).hexdigest()
def AST(s):return ast.dump(ast.parse(s),include_attributes=False)



import urllib.request
scope=save(out/'scope_BEFORE_inputs.json',dict(kind='Human_NEW_recovery43_Source_derivation',production_started=False,payload_reads=0))
def source(q):
 raw=Path(q).read_bytes();save(out/('RAW_Source_'+str(len(list(out.glob('RAW_Source*'))))+'.json'),dict(path=str(q),bytes=len(raw),sha256=sha(raw),base64=base64.b64encode(raw).decode()))
 return raw.decode('utf-8-sig').replace(chr(13)+chr(10),chr(10))
src=root/'ROOT_HUMAN_recovery_diagnosis_sources_1740_v1'
driver=source(src/'r8_resume_driver_v42.py').replace('asr18_reader8_normal259800_transition_v7','asr18_reader9_recovery276000_transition_v8')
assert 'a.invocation==29 and all' in driver
driver=driver.replace('a.invocation==29 and all','a.invocation==32 and all').replace('Exact saved212000 full prefix; no unsaved positives or tail promotion','Exact saved276000 prefix; original180 positives preserved').replace('ASR18_readonly_saved212000_prefix_after_source39_startup_failure_v1','ASR18_readonly_saved276000_prefix_after_I31_unknown_exit_v1')
helper=source(src/'asr18_reader8_normal259800_transition_v7.py')
start=helper.index('def checked_startup_reader_repair_policy(');end=helper.index('BASE_SOURCE=',start)
helper=helper[:start]+"\ndef checked_startup_reader_repair_policy(base):\n    import urllib.request,time\n    p=base[REPAIR];cpu=p[\"CPU_startup_policy\"]\n    require(p[\"kind\"]==\"ASR18_source43_CPU14_Reader9_abnormal276000_recovery_v1\"\n        and cpu[\"adapter_source\"]==record(X/\"CPU14_startup_launch_adapter_v15.py\")\n        and cpu[\"runner_source\"]==record(X/\"CPU14_owned_startup_only_receipt_runner_v2.py\")\n        and cpu[\"original_CPU14_source\"]==record(S/\"fastmss_resident_feature_stream_v14.py\")\n        and cpu[\"additional_host_reservation_bytes\"]==8388608\n        and p[\"total_additional_native_host_bytes\"]==EXTRA_HOST_BYTES,\"Unchanged CPU transport/science/reservations\")\n    proof_pin=cpu[\"actual_new_adapter_CPU_control_review\"];require(record(proof_pin[\"path\"])==proof_pin,\"Exact CPU09 proof\")\n    proof=read_json(proof_pin[\"path\"])\n    require(proof[\"accepted\"]is True and proof[\"reviewer\"]==\"/root\"\n        and proof[\"driver_source\"]==record(X/\"r9_recovery_driver_v43.py\")\n        and proof[\"adapter_source\"]==cpu[\"adapter_source\"]and proof[\"runner_source\"]==cpu[\"runner_source\"]\n        and proof[\"CPU14_source\"]==cpu[\"original_CPU14_source\"]\n        and proof[\"full_first_packet_passed\"]is True and proof[\"all_exact_known_identities_dead\"]is True,\"Actual current full20x16 transport and normal closure\")\n    reader_pin=p[\"actual_reader_replacement_ROOT\"];require(record(reader_pin[\"path\"])==reader_pin,\"Current Reader9 acceptance\")\n    reader=read_json(reader_pin[\"path\"])\n    require(reader[\"accepted\"]is True and reader[\"HTTP_200\"]is True\n        and reader[\"template_source\"]==record(p[\"reader_template\"][\"path\"])\n        and reader[\"API_helper_source\"]==record(p[\"API_helper_source\"][\"path\"]),\"Actual Reader9 HTTP and source\")\n    for who in(reader[\"new_reader_identity\"],reader[\"new_reader_launcher_identity\"]):\n        observed=psutil.Process(who[\"pid\"]);require(abs(observed.create_time()-who[\"ctime\"])<.001 and observed.ppid()==who[\"ppid\"],\"Exact current Reader9 identity\")\n    status=read_json(reader[\"status_path\"])\n    require(status[\"pid\"]==reader[\"new_reader_identity\"][\"pid\"]and status[\"mirror_thread_alive\"]is True\n        and status[\"reader_ready\"]is True and status[\"error\"]is None and time.time()-status[\"updated_unix\"]<30,\"Fresh Reader9 status\")\n    with urllib.request.urlopen(\"http://127.0.0.1:7860/version\",timeout=2)as response:require(response.status==200,\"Actual HTTP gate\")\n    return p\n\n"+helper[end:]
start=helper.index('def checked_current_plan_throughput_checkpoint_resume(');helper=helper[:start]+"\ndef checked_current_plan_throughput_checkpoint_resume(pin,*,base,folder,pointer_pin):\n    require(Path(folder).resolve()==BRANCH.resolve(),\"Same branch\")\n    require(pin==base[REPAIR][\"abnormal276000_recovery\"]and record(pin[\"path\"])==pin,\"Exact NEW recovery authority\")\n    value=read_json(pin[\"path\"])\n    require(value[\"accepted\"]is True and value[\"reviewer\"]==\"/root\"\n        and value[\"kind\"]==\"ROOT_actual_I31_unknown_exit_276180_saved276000_recovery_v1\"\n        and value[\"numeric_OS_exit\"]is None and value[\"normal_completion_claimed\"]is False\n        and value[\"checkpoint_pointer\"]==pointer_pin and value[\"last_positive\"]==276180\n        and value[\"symbolic_fullstate_PRE_SAVED_equal\"]is True\n        and value[\"preserved_tail_positive_updates\"]==list(range(276001,276181)),\"Honest unknown exit, full-state checkpoint, retained180 positives\")\n    for who in value[\"exact_dead_identities\"]:require(exact_dead(who),\"All recorded identities dead\")\n    for pid in value[\"absent_PID_only\"]:require(not psutil.pid_exists(pid),\"Recorded PID absence; past ctime unknown\")\n    pointer=read_json(pointer_pin[\"path\"])\n    require(pointer[\"committed_updates\"]==276000 and pointer[\"all_window_updates_committed\"]is True\n        and pointer[\"diagnostic_only\"]is False and pointer[\"journal_prefix\"]==value[\"saved_journal_prefix\"],\"Exact saved276000\")\n    whole=value[\"whole_journal_including_preserved_tail\"];observed=Path(whole[\"path\"]).stat()\n    require(observed.st_size==whole[\"bytes\"]and observed.st_mtime_ns==whole[\"mtime_ns\"],\"Original failed journal unchanged\")\n    return dict(saved_journal_prefix=pointer[\"journal_prefix\"],whole_journal_including_preserved_tail=whole,\n        preserved_tail_positive_updates=value[\"preserved_tail_positive_updates\"],unsaved_tail_bytes=353826,\n        actual_transition=pin,full_checkpoint_restore_and_genuine_redo_required=True)\n"
wrapper=source(src/'r8_resume_owner_v44.py').replace('r8_resume_driver_v42','r9_recovery_driver_v43').replace('asr18_reader8_normal259800_transition_v7','asr18_reader9_recovery276000_transition_v8')
wrapper=wrapper.replace('a.prior_process is not None and a.start_invocation>=31','a.throughput_resume is not None and a.prior_process is None and a.start_invocation==32').replace('Reader8_Source42_normal_saved259800_continuation','Reader9_Source43_abnormal276000_recovery').replace('a.start_invocation==review[\'initial_owner_invocation\']==29','a.start_invocation==review[\'initial_owner_invocation\']==32')
wrapper=wrapper.replace('            code=child.wait()','            code=child.wait()\n            write_json(control/("native_"+str(invocation)+"_Popen_exit.json"),dict(actual_Popen_wait_code=code,invocation=invocation,time=time.time()))')
proof=source(root/'ROOT_HUMAN_Recovery43_plan_source_inputs_1815_v1'/'proof08_Source_COPY.py')
proof=proof.replace('cpu08_final259800_actual01','cpu09_recovery276000_actual01').replace('cpu08_final259800_external_actual01','cpu09_recovery276000_external_actual01').replace('CPU08','CPU09')
proof=proof.replace('scope["owner"]=="/root/memory_diagnosis"','scope["owner"]=="/root"').replace('NEW_final259800_full20x16_dual_configuration_original_receiver_NumPy','NEW276000_recovery43_full20x16_transport')
proof=proof.replace('range(259781,259801)','range(276181,276201)').replace('==259781+i','==276181+i')
proof=proof.replace('plan_sha=="13488c660c652514de01122fd72a9d3335765e70fce87e52fe95c43704ef8acb"','plan_sha==scope["canonical_plan_sha256"]')
proof=proof.replace('need(actual_stable==expected_stable,"Every original scalar/fulltarget/context/geometry/PCM/feature/mask hash")','need(metadata["global_update"]==plan["updates"][i]["global_update"]and metadata["phase"]==plan["updates"][i]["phase"],"Exact new plan update/phase; full array transport integrity follows")')
proof=proof.replace('expected_array=expected["metadata"]["rows"][row_index][name]','expected_array=metadata["rows"][row_index][name]')
sources=[]
for q,text in[(root/'r9_recovery_driver_v43.py',driver),(root/'asr18_reader9_recovery276000_transition_v8.py',helper),(root/'r9_recovery_owner_v45.py',wrapper),(root/'cpu09_recovery276000_proof_v3.py',proof)]:
 ast.parse(text);raw=text.encode();f=q.open('xb');f.write(raw);f.flush();os.fsync(f.fileno());f.close();sources.append(dict(path=str(q),bytes=len(raw),sha256=sha(raw)))
def filepin(q):raw=Path(q).read_bytes();return dict(path=str(Path(q).resolve()),bytes=len(raw),sha256=sha(raw))
raw=(root/'ROOT_HUMAN_276000_full_fingerprint_recovery_inputs_1805_v1'/'configuration_COPY.json').read_bytes()
save(out/'RAW_cfg_BEFORE_decode.json',dict(bytes=len(raw),sha256=sha(raw),base64=base64.b64encode(raw).decode()));cfg=json.loads(raw)
p=cfg["original_CPU14_startup_reader7_recovery_policy"];p["kind"]="ASR18_source43_CPU14_Reader9_abnormal276000_recovery_v1"
reader_source=root/'ROOT_HUMAN_Reader9_repair_1755_v1'/'serve_BPE9223_Trackio_live_v9.py';api_source=reader_source.parent/'Trackio_owned_bounded_sync_API_dispatch_v5.py'
p["reader_template"]=filepin(reader_source);p["API_helper_source"]=filepin(api_source)
p["CPU_startup_policy"]["execution_review_path"]=str(root/'ROOT_HUMAN_Recovery43_FINAL_ONE_v1.json')
status=json.loads(Path('E:/gpu_training_scratchpad/analysis/BPE9223_Trackio_live_v1/current_status.json').read_text())
who=psutil.Process(status["pid"]);launcher=psutil.Process(who.ppid())
with urllib.request.urlopen('http://127.0.0.1:7860/version',timeout=2)as response:assert response.status==200
reader_receipt=save(out/'Reader9_actual_acceptance.json',dict(accepted=True,reviewer='/root',HTTP_200=True,template_source=p["reader_template"],API_helper_source=p["API_helper_source"],status_path='E:/gpu_training_scratchpad/analysis/BPE9223_Trackio_live_v1/current_status.json',new_reader_identity=dict(pid=who.pid,ctime=who.create_time(),ppid=who.ppid()),new_reader_launcher_identity=dict(pid=launcher.pid,ctime=launcher.create_time(),ppid=launcher.ppid())))
p["actual_reader_replacement_ROOT"]=reader_receipt
candidate=save(out/'candidate_configuration.json',cfg)
planpath=root/'ROOT_HUMAN_Recovery43_plan_source_inputs_1815_v1'/'plan820_COPY.json';plan=json.loads(planpath.read_bytes());planpin=filepin(planpath)
reference=save(out/'plan820_expected_updates.json',[dict(metadata=dict(global_update=e['global_update']))for e in plan['updates']])
pointer=dict(path='E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01/checkpoints/update_000276000/pointer.json',bytes=16232,sha256='179dbac1643d12023d231aaeb17f7469c7ddeffdd376a4410af8812d00947b59')
pointer["path"]=str(Path(pointer["path"]).resolve())
cpu_scope=dict(accepted=True,reviewer='/root',owner='/root',kind='ROOT_CPU14_startup_adapter_firstpacket_exact_ONE_CPU_scope_v1',execution_phase='NEW276000_recovery43_full20x16_transport',
 resources=dict(prospective_commit=8066304384,prospective_physical=5316304384,runtime_commit=5547868928,runtime_physical=5047868928,self_private=536870912,reader_private=1500000000,family_private=2573741824,body_seconds=160,external_seconds=180,HTTP_seconds=20,readiness_seconds=60,interval=.1,E_free=100000000000,log_bytes=16777216,descendants=16,known_identities=32),
 body_output=str((root/'cpu09_recovery276000_actual01').resolve()),external_output=str((root/'cpu09_recovery276000_external_actual01').resolve()),proof_source=sources[3],
 required_exact_dead=[dict(pid=38136,ctime=1792984574.6114764,ppid=4032),dict(pid=4032,ctime=1792984574.6040332,ppid=21120),dict(pid=21544,ctime=1793015282.40108,ppid=25072)],
 native_full_reserve_remains_charged=True,no_production_control=True,
 public_entry=[str(Path('E:/gpu_training_scratchpad/scratch/BPE_broad_CPU_preflight_env_v2/Scripts/python.exe').resolve()),'-X','utf8','-B',str(Path(sources[3]['path']).resolve()),'--ROOT-review',str((root/'ROOT_HUMAN_CPU09_scope.json').resolve())],
 configuration_Root_copy=candidate,plan_Root_copy=planpin,CPU20_expected_Root_copy=reference,driver_source=sources[0],canonical_plan_sha256=sha(json.dumps(plan,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()),
 final_pointer=pointer,CPU_startup_adapter_source=p["CPU_startup_policy"]["adapter_source"],CPU_startup_runner_source=p["CPU_startup_policy"]["runner_source"],
 CPU14_source=filepin(Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20/outputs/parakeet_turkish/fastmss_resident_feature_stream_v14.py')),
 parent_numpy_runtime_source=filepin(Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20/outputs/parakeet_turkish/fastmss_BPE9223_cpu_common_v12.py')))
scopepin=save(root/'ROOT_HUMAN_CPU09_scope.json',cpu_scope)
print(json.dumps(dict(sources=sources,candidate=candidate,reader=reader_receipt,CPU_scope=scopepin,final=guard())),flush=True)

