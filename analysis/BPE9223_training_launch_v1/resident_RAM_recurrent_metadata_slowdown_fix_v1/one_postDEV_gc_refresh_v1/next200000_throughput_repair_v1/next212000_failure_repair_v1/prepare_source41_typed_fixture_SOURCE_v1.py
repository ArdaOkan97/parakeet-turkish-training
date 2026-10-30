"""SOURCE-only typed fixture derivation. Does not import/execute candidate code."""
import ast,ctypes,difflib,hashlib,json,os,time
from pathlib import Path
import psutil
X=Path(__file__).resolve().parent
START=time.monotonic()
def pin(p):
    p=Path(p);b=p.read_bytes();return dict(path=str(p.resolve()),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def put(p,v):
    b=(json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode()
    with p.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
    return pin(p)
def sample():
    class M(ctypes.Structure):
        _fields_=[('a',ctypes.c_ulong),('b',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong)for n in('tp','p','tc','c','tv','v','extra')]
    m=M();m.a=ctypes.sizeof(m);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return dict(commit=m.c,physical=m.p,private=psutil.Process().memory_info().private,E_free=psutil.disk_usage(str(X)).free,wall=time.monotonic()-START)
ids=[dict(pid=p.pid,create_time=p.create_time(),ppid=p.ppid())for p in(psutil.Process(),psutil.Process().parent())]
s=sample();initial=put(X/'source41_typed_SOURCE_initial_BEFORE_predicate_v1.json',dict(sample=s,identities=ids,required_commit=6028384896,required_physical=3278384896,TORCH_EXTENSIONS_DIR=os.environ.get('TORCH_EXTENSIONS_DIR')))
assert s['commit']>=6028384896 and s['physical']>=3278384896 and s['private']<=536870912 and s['E_free']>=100000000000
assert os.environ['CUDA_VISIBLE_DEVICES']=='-1'and os.environ['TORCH_EXTENSIONS_DIR']==r'E:\gpu_training_scratchpad\cache\torch_extensions'
def change(s,a,b,count=1):
    assert s.count(a)==count,(a,s.count(a));return s.replace(a,b)
def block(s,a,b,replacement):
    assert s.count(a)==s.count(b)==1
    i=s.index(a);j=s.index(b,i);return s[:i]+replacement+s[j:]
def freeze(base,p,text):
    old=base.read_bytes();new=text.encode();ast.parse(text);assert not p.exists()
    ol=old.splitlines(keepends=True);nl=new.splitlines(keepends=True);oo=[0];nn=[0]
    for line in ol:oo.append(oo[-1]+len(line))
    for line in nl:nn.append(nn[-1]+len(line))
    ops=[]
    for tag,i,j,a,b in difflib.SequenceMatcher(None,ol,nl,autojunk=False).get_opcodes():
        if tag!='equal':ops.append(dict(old_start=oo[i],old_end=oo[j],new_start=nn[a],new_end=nn[b],old_hex=old[oo[i]:oo[j]].hex(),new_hex=new[nn[a]:nn[b]].hex(),old_literal=old[oo[i]:oo[j]].decode(),new_literal=new[nn[a]:nn[b]].decode()))
    forward=old
    for o in reversed(ops):forward=forward[:o['old_start']]+bytes.fromhex(o['new_hex'])+forward[o['old_end']:]
    inverse=new
    for o in reversed(ops):inverse=inverse[:o['new_start']]+bytes.fromhex(o['old_hex'])+inverse[o['new_end']:]
    assert forward==new and inverse==old
    with p.open('xb')as f:f.write(new);f.flush();os.fsync(f.fileno())
    return dict(base=pin(base),successor=pin(p),ordered_RAW=ops,whole_forward_exact=True,whole_inverse_exact=True,base_AST=ast.dump(ast.parse(old.decode()),include_attributes=False),successor_AST=ast.dump(ast.parse(text),include_attributes=False))

base=X/'source40_saved212000_entire_typed_metadata_CPU_fixture_v1.py';text=base.read_bytes().decode()
text=change(text,'asr18_current_plan_throughput_checkpoint_transition_v5.py','asr18_current_plan_throughput_checkpoint_transition_v6_dual_config_v2.py')
text=change(text,'required=constants["REQUIRED_OLD_IDENTITIES"]','required=constants["REQUIRED_FAILURE_IDENTITIES"]\n    failed_path=str(BRANCH/"METHOD_STUB_failed_J28.jsonl")\n    candidate_driver_pin=pin(HERE/"run_ASR18_BPE9223_resident_RAM_training_v41_dual_config_v2.py")')
text=change(text,'full_first_packet_passed=True,original_CPU14_worker_and_transport_unchanged=True,all_exact_known_identities_dead=True)','full_first_packet_passed=True,original_CPU14_worker_and_transport_unchanged=True,all_exact_known_identities_dead=True,\n        dual_distinct_config_roles_and_actual_driver_call_AST_passed=True,actual_worker_wire_configuration_equals_incoming_stream_pin=True,driver_source=candidate_driver_pin)')
failure='''    failure=dict(kind="ROOT_source40_I28_actual_adapter_gate_failure_independent_seal_v1",accepted=True,reviewer="/root",
        owner50280_consumed_ONCE_only_by_owner=True,owned_TOOL_reported_exit=1,actual_direct_wrapper_numeric_OS_exit=1,
        separate_native_numeric_OS_exit="UNRECORDED_UNKNOWN",actual_primary_error="AssertionError()",compound_assert_false_operands="UNRECORDED_UNKNOWN",
        actual_optimizer_updates=212000,published_journal_update=212000,saved_update=212000,gap=0,new_positive_or_unsaved_tail_updates=0,
        all4known_native_and_sealer_exactdead=[dict(identity=i,exact_dead=True)for i in required],
        wrapper_PID_only=dict(pid=26912,observed_absent=True,ctime="UNRECORDED_UNKNOWN"),
        actual_full_restore=dict(passed=True,all207_threeAdam_all4RNG_match_actual_pre_save=True,next_draw_packet_match=True,restore_occurred_only_at_startup_before_native=True),
        actual_journal_branch=dict(historical_unsaved_tail_updates=0,native_updates_or_draws_in_copy=0,
            saved_prefix=dict(path=prefix_path,offset=426026870,sha256="6621855eb4191790c3a741e3e1c19014f00db87b9e471d3a4c70802cbddaaf0a"),
            new_active_journal=dict(path=failed_path,offset=426026870,sha256="6621855eb4191790c3a741e3e1c19014f00db87b9e471d3a4c70802cbddaaf0a")),
        NEW_J28_STAT_ONLY=dict(path=failed_path,bytes=426026870,mtime_ns=223456789,body_read=False),
        current_reader_health=dict(pid=991,create_time=1.0,journal_path=failed_path))
'''
text=block(text,'    more=[dict(pid=26648','    saved=dict(',failure)
text=change(text,'status=dict(pid=991,create_time=1.0,mirror_thread_alive=True,HTTP_ready=True,reader_ready=True,error=None)','status=dict(pid=991,create_time=1.0,mirror_thread_alive=True,HTTP_ready=True,reader_ready=True,error=None,journal_path=failed_path,journal_offset=426026870,last_verified_checkpoint_update=212000)')
text=change(text,'assert norm(self)==norm(prefix_path),"METHOD_STUB only expected journal STAT, no real original file"\n            return SimpleNamespace(st_size=current["stat"]["bytes"],st_mtime_ns=current["stat"]["mtime_ns"])','assert norm(self)in (norm(prefix_path),norm(failed_path)),"METHOD_STUB only expected two STATs, no real files"\n            value=current["stat"]if norm(self)==norm(prefix_path)else current["failed_stat"]\n            return SimpleNamespace(st_size=value["bytes"],st_mtime_ns=value["mtime_ns"])')
text=change(text,'        repair["actual_reader_replacement_ROOT"]=fakeReader','        repair["actual_reader_replacement_ROOT"]=fakeReader\n        repair["actual_reader_current_continuation_ROOT"]=constants["FAILURE_ROOT"]\n        repair["closed_failed_I28_journal_STAT"]=copy.deepcopy(failure["NEW_J28_STAT_ONLY"])')
text=change(text,'cpu["adapter_source"],cpu["runner_source"],cpu["original_CPU14_source"],fakeCPU,fakeReader','cpu["adapter_source"],cpu["runner_source"],cpu["original_CPU14_source"],fakeCPU,fakeReader,candidate_driver_pin')
text=change(text,'stat=dict(bytes=426026870,mtime_ns=123456789))','stat=dict(bytes=426026870,mtime_ns=123456789),failed_stat=dict(bytes=426026870,mtime_ns=223456789))')
text=change(text,'FAILURE="prior_resident_RAM_source39_startup212000_failure"','FAILURE="prior_resident_RAM_source40_adapter212000_failure"')
text=change(text,'["all_recorded_old_runtime_control_sealer_identities_exact_dead"].pop(0)','["all4known_native_and_sealer_exactdead"].pop(0)')
text=change(text,'{34788:dict(ctime=1792842518.865024,ppid=32332)}','{25176:dict(ctime=1792873894.854458,ppid=32972)}')
text=change(text,'update(actual212000_saved212000_gap0=False)','update(gap=1)')
text=change(text,'update(underlying_new_CPU_startup_cause="NumPy")','update(compound_assert_false_operands="fabricated_primary_cause")')
text=change(text,'update(separate_native_CPU_wrapper_numeric_OS_exits=0)','update(separate_native_numeric_OS_exit=0)')
text=change(text,'payload(c,fakeReader["path"]).update(journal_path="METHOD_STUB_OTHER")','payload(c,fake_status).update(journal_path="METHOD_STUB_OTHER")')
text=change(text,'run_ASR18_BPE9223_resident_RAM_training_v40.py','run_ASR18_BPE9223_resident_RAM_training_v41_dual_config_v2.py')
text=change(text,'run_ASR18_BPE9223_resident_RAM_autocontinue_v41.py','run_ASR18_BPE9223_resident_RAM_autocontinue_v42_dual_config_v2.py')
text=change(text,'((28,True),(27,False),(29,False))','((29,True),(28,False),(30,False))')
text=change(text,'start_invocation=28)','start_invocation=29)')
text=change(text,'invocation=28 if prior is None else 29','invocation=29 if prior is None else 30')
extra='''    check("dual_config_CPU_review_missing_distinct_roles",lambda c:payload(c,fakeCPU["path"]).update(dual_distinct_config_roles_and_actual_driver_call_AST_passed=False))
    check("dual_config_CPU_review_wire_false",lambda c:payload(c,fakeCPU["path"]).update(actual_worker_wire_configuration_equals_incoming_stream_pin=False))
    check("dual_config_CPU_review_wrong_driver",lambda c:payload(c,fakeCPU["path"])["driver_source"].update(sha256="0"*64))
    check("missing_current_reader_J28_authority",lambda c:repair(c).update(actual_reader_current_continuation_ROOT=None))
    check("failed_J28_STAT_changedbytes",lambda c:c["failed_stat"].update(bytes=426026871))
    check("failed_J28_STAT_changedmtime",lambda c:c["failed_stat"].update(mtime_ns=223456790))
    check("false_fullrestore",lambda c:payload(c,constants["FAILURE_ROOT"]["path"])["actual_full_restore"].update(passed=False))
    check("false_wrapper_exactctime",lambda c:payload(c,constants["FAILURE_ROOT"]["path"])["wrapper_PID_only"].update(ctime=1.0))
'''
text=change(text,'    # Actual generated DRIVER selector AST, no imports/model/native/restore calls.',extra+'    # Actual generated DRIVER selector AST, no imports/model/native/restore calls.')
fixture=X/'source41_saved212000_entire_typed_metadata_CPU_fixture_v1.py'
raws=[freeze(base,fixture,text)]
externalbase=X/'run_source40_saved212000_typed_metadata_CPU_external_v1.py'
external=externalbase.read_bytes().decode().replace('typed SOURCE40','typed SOURCE41 dual-role')
externalpath=X/'run_source41_saved212000_typed_metadata_CPU_external_v1.py'
raws.append(freeze(externalbase,externalpath,external))
raw=put(X/'source41_typed_fixture_external_full_RAW_AST_v1.json',dict(initial=initial,files=raws,all_authorities_METHOD_STUB=True,no_candidate_execution=True))
handoff=put(X/'source41_typed_fixture_SOURCE_handoff_v1.json',dict(kind='source41_typed_fixture_SOURCE_handoff_v1',initial=initial,identities=ids,fixture=pin(fixture),external=pin(externalpath),RAW=raw,source_bundle_handoff=pin(X/'source41_dual_configuration_INACTIVE_handoff_v3.json'),candidate_driver=pin(X/'run_ASR18_BPE9223_resident_RAM_training_v41_dual_config_v2.py'),candidate_wrapper=pin(X/'run_ASR18_BPE9223_resident_RAM_autocontinue_v42_dual_config_v2.py'),candidate_helper=pin(X/'asr18_current_plan_throughput_checkpoint_transition_v6_dual_config_v2.py'),configuration=pin(X/'source41_saved212000_candidate_configuration_INACTIVE_v2.json'),future_scope_kind='ROOT_source40_saved212000_entire_typed_metadata_exact_ONE_CPU_scope_v1',future_proposed_command=[r'E:\gpu_training_scratchpad\scratch\BPE_broad_CPU_preflight_env_v2\Scripts\python.exe','-X','utf8','-B',str(externalpath),'--ROOT-review','UNKNOWN_SEPARATE_ROOT_SCOPE_REQUIRED'],initial_owner_positive=29,strict_owner_rejects=[28,30],method_stub_authorities_only=True,no_actual_pointer_journal_checkpoint_payload_HTTP_DB_audio_or_control=True,proof_resources=dict(prospective_commit=6565255808,prospective_physical=3815255808,runtime_commit=4046820352,runtime_physical=3546820352,family_private=536870912,fixture_seconds=100,external_seconds=120),all_original_science_and_native80MiB_unchanged=True))
end=sample();assert end['commit']>=3509949440 and end['physical']>=3009949440 and end['private']<=536870912 and end['wall']<=180
result=put(X/'source41_typed_fixture_SOURCE_result_v1.json',dict(passed=True,initial=initial,identities=ids,fixture=pin(fixture),external=pin(externalpath),RAW=raw,handoff=handoff,final=end))
print(json.dumps(dict(result=result,initial=initial,identities=ids,fixture=pin(fixture),external=pin(externalpath),RAW=raw,handoff=handoff)))
