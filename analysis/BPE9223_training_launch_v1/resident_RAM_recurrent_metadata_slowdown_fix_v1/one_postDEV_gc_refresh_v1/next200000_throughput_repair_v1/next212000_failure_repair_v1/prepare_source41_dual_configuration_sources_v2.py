"""Metadata/SOURCE materialization only. Never imports or executes candidates."""
import ast
import copy
import ctypes
import difflib
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import psutil

X=Path(__file__).resolve().parent
START=time.monotonic()
def pin(p):
    p=Path(p);b=p.read_bytes()
    return dict(path=str(p.resolve()),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def write(p,v):
    b=(json.dumps(v,ensure_ascii=False,indent=2)+"\n").encode()
    with p.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
    return pin(p)
def identity(p):return dict(pid=p.pid,create_time=p.create_time(),ppid=p.ppid())
class Memory(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong),('totalphysical',ctypes.c_ulonglong),('physical',ctypes.c_ulonglong),('totalcommit',ctypes.c_ulonglong),('commit',ctypes.c_ulonglong),('totalvirtual',ctypes.c_ulonglong),('virtual',ctypes.c_ulonglong),('extra',ctypes.c_ulonglong)]
def sample():
    m=Memory();m.length=ctypes.sizeof(m);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return dict(commit=m.commit,physical=m.physical,private=psutil.Process().memory_info().private,E_free=psutil.disk_usage(str(X)).free,elapsed=time.monotonic()-START)
me=psutil.Process();ids=[identity(me),identity(me.parent())]
s=sample()
initial=write(X/'source41_dual_configuration_SOURCE_initial_BEFORE_predicate_v2.json',dict(sample=s,identities=ids,required_commit=6028384896,required_physical=3278384896,TORCH_EXTENSIONS_DIR=os.environ.get('TORCH_EXTENSIONS_DIR')))
assert s['commit']>=6028384896 and s['physical']>=3278384896 and s['private']<=536870912 and s['E_free']>=100000000000
assert os.environ['CUDA_VISIBLE_DEVICES']=='-1'and os.environ['TORCH_EXTENSIONS_DIR']==r'E:\gpu_training_scratchpad\cache\torch_extensions'

def literal(text,old,new,count=1):
    assert text.count(old)==count,(old,text.count(old),count)
    return text.replace(old,new)
def replace_function(text,name,new):
    node=next(n for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))and n.name==name)
    lines=text.splitlines(keepends=True)
    return ''.join(lines[:node.lineno-1])+new.rstrip()+'\n'+''.join(lines[node.end_lineno:])
def raw_bundle(name,oldpath,newpath,newtext):
    old=Path(oldpath).read_bytes();new=newtext.encode()
    if newpath.exists():assert newpath.read_bytes()==new,"Only byteidentical previously frozen helper may be reused"
    tree=ast.parse(newtext);ops=[]
    oldlines=old.splitlines(keepends=True);newlines=new.splitlines(keepends=True)
    oldoff=[0];newoff=[0]
    for line in oldlines:oldoff.append(oldoff[-1]+len(line))
    for line in newlines:newoff.append(newoff[-1]+len(line))
    for tag,li,lj,la,lb in difflib.SequenceMatcher(None,oldlines,newlines,autojunk=False).get_opcodes():
        i,j,a,b=oldoff[li],oldoff[lj],newoff[la],newoff[lb]
        if tag!='equal':ops.append(dict(operation=tag,old_start=i,old_end=j,new_start=a,new_end=b,old_hex=old[i:j].hex(),new_hex=new[a:b].hex(),old_literal=old[i:j].decode('utf8',errors='replace'),new_literal=new[a:b].decode('utf8',errors='replace')))
    forward=old
    for op in reversed(ops):forward=forward[:op['old_start']]+bytes.fromhex(op['new_hex'])+forward[op['old_end']:]
    inverse=new
    for op in reversed(ops):inverse=inverse[:op['new_start']]+bytes.fromhex(op['old_hex'])+inverse[op['new_end']:]
    assert forward==new and inverse==old
    if not newpath.exists():
        with newpath.open('xb')as f:f.write(new);f.flush();os.fsync(f.fileno())
    original_ast=ast.parse(old.decode())
    def nodes(t):return [dict(type=type(n).__name__,name=getattr(n,'name',None),AST=ast.dump(n,include_attributes=False))for n in t.body]
    return dict(name=name,base=pin(oldpath),successor=pin(newpath),ordered_RAW=ops,forward_exact=True,inverse_exact=True,whole_inverse_sha256=hashlib.sha256(inverse).hexdigest(),base_AST=nodes(original_ast),successor_AST=nodes(tree))

scopepin=pin(X/'ROOT_source41_dual_configuration_CPU_wire_saved212000_INACTIVE_SOURCE_scope_v1.json')
assert scopepin['sha256']=='f2d070ff9c9fc4219c191e5135f488c50bdaa38344612f44674b93a9de0065ef'
scope=json.loads(Path(scopepin['path']).read_bytes())
context=json.loads((X/'source41_SOURCE_context_and_two_config_roles_v1.json').read_bytes())
failure=context['NEW_failure_ROOT'];failurepin=scope['NEW_source40_failure_ROOT']
assert failure['kind']=='ROOT_source40_I28_actual_adapter_gate_failure_independent_seal_v1'and failure['accepted']is True
outputs=[];raw=[]
try:
    helperbase=X/'asr18_current_plan_throughput_checkpoint_transition_v5.py'
    helper=helperbase.read_bytes().decode()
    helper=literal(helper,'INACTIVE coherent source39 saved212000 startup/reader recovery.','INACTIVE coherent source40 adapter-gate saved212000 recovery; distinct CPU configuration roles.')
    helper=literal(helper,'FAILURE="prior_resident_RAM_source39_startup212000_failure"','FAILURE="prior_resident_RAM_source40_adapter212000_failure"')
    helper=literal(helper,'ASR18_source40_original_CPU14_startup_reader7_recovery_v1','ASR18_source41_original_CPU14_dual_configuration_reader7_recovery_v1')
    helper=literal(helper,'and proof["full_first_packet_passed"]is True','and proof["full_first_packet_passed"]is True\n        and proof["dual_distinct_config_roles_and_actual_driver_call_AST_passed"]is True\n        and proof["actual_worker_wire_configuration_equals_incoming_stream_pin"]is True\n        and proof["driver_source"]==record(X/"run_ASR18_BPE9223_resident_RAM_training_v41.py")')
    oldline=next(line for line in helper.splitlines()if line.startswith('FAILURE_ROOT='))
    helper=literal(helper,oldline,'FAILURE_ROOT='+repr(failurepin))
    resume='''def checked_current_plan_throughput_checkpoint_resume(pin,*,base,folder,pointer_pin):
    require(Path(folder).resolve()==BRANCH.resolve()and pin==base[FAILURE]==FAILURE_ROOT
        and record(pin["path"])==pin,"Exact accepted source40 I28 adapter-gate failure authority")
    checked_current_plan_throughput_policy(base)
    value=read_json(pin["path"])
    require(value["kind"]=="ROOT_source40_I28_actual_adapter_gate_failure_independent_seal_v1"
        and value["accepted"]is True and value["reviewer"]=="/root"
        and value["owner50280_consumed_ONCE_only_by_owner"]is True
        and value["owned_TOOL_reported_exit"]==1 and value["actual_direct_wrapper_numeric_OS_exit"]==1
        and value["separate_native_numeric_OS_exit"]=="UNRECORDED_UNKNOWN"
        and value["actual_primary_error"]=="AssertionError()"
        and value["compound_assert_false_operands"]=="UNRECORDED_UNKNOWN"
        and value["actual_optimizer_updates"]==value["published_journal_update"]==value["saved_update"]==212000
        and value["gap"]==0 and value["new_positive_or_unsaved_tail_updates"]==0,
        "Honest NEW adapter failure/direct wrapper1/native exit and operand UNKNOWN; coherent212000 zero-gap")
    deaths=value["all4known_native_and_sealer_exactdead"]
    require(len(deaths)==4 and all(d["exact_dead"]is True and exact_dead(d["identity"])for d in deaths)
        and all(any(d["identity"]==i for d in deaths)for i in REQUIRED_FAILURE_IDENTITIES),
        "All recorded NEW exact native/sealer identities dead, no invented wrapper identity")
    wrapper=value["wrapper_PID_only"]
    require(wrapper["pid"]==26912 and wrapper["observed_absent"]is True
        and wrapper["ctime"]=="UNRECORDED_UNKNOWN",
        "Historical wrapper ctime/exact death remains UNKNOWN; accepted PIDabsence is not exactdeath")
    restore=value["actual_full_restore"];branch=value["actual_journal_branch"]
    require(restore["passed"]is True and restore["all207_threeAdam_all4RNG_match_actual_pre_save"]is True
        and restore["next_draw_packet_match"]is True and restore["restore_occurred_only_at_startup_before_native"]is True
        and branch["historical_unsaved_tail_updates"]==0 and branch["native_updates_or_draws_in_copy"]==0,
        "Accepted full restore-before-savedprefix-copy/zero draw, not a successful native update")
    require(record(SAVED_ROOT["path"])==SAVED_ROOT,"Exact closed ROOT full854/PRE-SAVED authority")
    saved=read_json(SAVED_ROOT["path"])
    require(saved["accepted"]is True and saved["reviewer"]=="/root"and saved["updates"]==212000
        and saved["weights"]==207 and saved["Adam_states"]==215 and saved["Adam_partitions"]==[17,6,192]
        and saved["all4RNG"]is True and saved["named_tensor_references"]==854
        and saved["integer_Adam_JSON_keys_only_normalized"]is True
        and saved["semantic_identity_sampler_journal_nextdraw_equal"]is True,
        "All original full-state geometry/scalars/samplers/nextdraw, no new payloadhash or GPUrestore claim")
    require(pointer_pin==SOURCE_DERIVED_POINTER and record(pointer_pin["path"])==pointer_pin,
        "Exact saved212000 pointer bytes/hash required before future restore")
    pointer=read_json(pointer_pin["path"])
    require(pointer["committed_updates"]==212000 and pointer["diagnostic_only"]is False
        and pointer["all_window_updates_committed"]is True,"Full coherent212000 pointer")
    prefix=pointer["journal_prefix"]
    require(prefix==branch["saved_prefix"]and prefix["offset"]==426026870
        and prefix["sha256"]=="6621855eb4191790c3a741e3e1c19014f00db87b9e471d3a4c70802cbddaaf0a"
        and Path(prefix["path"]).resolve().is_relative_to(BRANCH.resolve()),"Original published J27 savedprefix only")
    expected=base[REPAIR]["closed_original_journal_STAT"]
    require(type(expected)is dict and expected["kind"]=="STAT_ONLY_original_closed_journal_v1"
        and expected["path"]==prefix["path"]and expected["bytes"]==prefix["offset"],"Original J27 STAT binding")
    actual=Path(expected["path"]).stat()
    require(actual.st_size==expected["bytes"]and actual.st_mtime_ns==expected["mtime_ns"],"Original J27 STAT only")
    failed=base[REPAIR]["closed_failed_I28_journal_STAT"]
    require(failed==value["NEW_J28_STAT_ONLY"]and failed["body_read"]is False
        and failed["path"]==branch["new_active_journal"]["path"]and failed["bytes"]==prefix["offset"],
        "Separately preserved failed J28 copiedprefix STAT, no body/hash/tail promotion")
    failed_actual=Path(failed["path"]).stat()
    require(failed_actual.st_size==failed["bytes"]and failed_actual.st_mtime_ns==failed["mtime_ns"],"Failed J28 STAT unchanged")
    require(base[REPAIR]["actual_reader_current_continuation_ROOT"]==FAILURE_ROOT,
        "Current same-reader J28 continuation authority must be separately exact ROOT-bound")
    status=read_json(read_json(base[REPAIR]["actual_reader_replacement_ROOT"]["path"])["status_path"])
    reader=value["current_reader_health"]
    require(status["pid"]==reader["pid"]and status["create_time"]==reader["create_time"]
        and status["journal_path"]==reader["journal_path"]==failed["path"]
        and status["journal_offset"]==prefix["offset"]and status["last_verified_checkpoint_update"]>=212000
        and status["mirror_thread_alive"]is True and status["HTTP_ready"]is True
        and status["reader_ready"]is True and status["error"]is None,
        "Healthy existing same reader on failed J28 prefix; no restart or stale J27-current claim")
    return dict(saved_journal_prefix=prefix,whole_journal_including_preserved_tail=expected,
        preserved_closed_failed_journal_STAT=failed,preserved_tail_positive_updates=0,unsaved_tail_bytes=0,
        actual_transition=pin,full_checkpoint_restore_and_genuine_redo_required=True,
        honest_original_optimizer_vs_journal_gap=0,no_unsaved_positive_updates_after212000=True)
'''
    helper=replace_function(helper,'checked_current_plan_throughput_checkpoint_resume',resume)
    helper+='\nREQUIRED_FAILURE_IDENTITIES='+repr([d['identity']for d in failure['all4known_native_and_sealer_exactdead']])+'\n'
    helperpath=X/'asr18_current_plan_throughput_checkpoint_transition_v6.py'
    raw.append(raw_bundle('helper6',helperbase,helperpath,helper));outputs.append(pin(helperpath))

    driverbase=X/'run_ASR18_BPE9223_resident_RAM_training_v40.py';driver=driverbase.read_bytes().decode()
    driver=literal(driver,'asr18_current_plan_throughput_checkpoint_transition_v5','asr18_current_plan_throughput_checkpoint_transition_v6',driver.count('asr18_current_plan_throughput_checkpoint_transition_v5'))
    driver=literal(driver,"startup_adapter.configure_startup_receipts(candidate_base['original_CPU14_startup_reader7_recovery_policy']['CPU_startup_policy'],record(config_path))","startup_adapter.configure_startup_receipts(candidate_base['original_CPU14_startup_reader7_recovery_policy']['CPU_startup_policy'],record(a.stream_configuration))")
    driver=literal(driver,"'startup_only_no_per_update_observer':True", "'startup_only_no_per_update_observer':True,'worker_stream_configuration':record(a.stream_configuration),'separate_parent_runtime_configuration':record(config_path)")
    driverpath=X/'run_ASR18_BPE9223_resident_RAM_training_v41.py'
    raw.append(raw_bundle('driver41',driverbase,driverpath,driver));outputs.append(pin(driverpath))
    wrapperbase=X/'run_ASR18_BPE9223_resident_RAM_autocontinue_v41.py';wrapper=wrapperbase.read_bytes().decode()
    for old,new in [('run_ASR18_BPE9223_resident_RAM_training_v40','run_ASR18_BPE9223_resident_RAM_training_v41'),('asr18_current_plan_throughput_checkpoint_transition_v5','asr18_current_plan_throughput_checkpoint_transition_v6')]:
        wrapper=literal(wrapper,old,new,wrapper.count(old))
    wrapperpath=X/'run_ASR18_BPE9223_resident_RAM_autocontinue_v42.py'
    raw.append(raw_bundle('wrapper42',wrapperbase,wrapperpath,wrapper));outputs.append(pin(wrapperpath))

    basecfgpath=X/'source40_saved212000_final_configuration_v1/configuration.json'
    cfg=json.loads(basecfgpath.read_bytes());before=copy.deepcopy(cfg)
    repair=cfg['original_CPU14_startup_reader7_recovery_policy']
    repair['kind']='ASR18_source41_original_CPU14_dual_configuration_reader7_recovery_v1'
    repair['CPU_startup_policy']['actual_new_adapter_CPU_control_review']=None
    repair['CPU_startup_policy']['execution_review_path']=str(X/'ROOT_source41_saved212000_final_ONE_training_execution_review_v1.json')
    repair['actual_reader_current_continuation_ROOT']=failurepin
    repair['closed_failed_I28_journal_STAT']=failure['NEW_J28_STAT_ONLY']
    del cfg['prior_resident_RAM_source39_startup212000_failure']
    cfg['prior_resident_RAM_source40_adapter212000_failure']=failurepin
    for key in set(before)-{'original_CPU14_startup_reader7_recovery_policy','prior_resident_RAM_source39_startup212000_failure'}:assert cfg[key]==before[key],key
    cfgpin=write(X/'source41_saved212000_candidate_configuration_INACTIVE_v1.json',cfg);outputs.append(cfgpin)

    proofbase=X/'CPU14_startup_adapter_firstpacket_CPU_proof_v5.py';proof=proofbase.read_bytes().decode()
    proof=literal(proof,'INACTIVE SOURCE v5.','INACTIVE SOURCE v6. Distinct incoming stream and parent-runtime roles.')
    proof=literal(proof,'import argparse','import ast\nimport copy\nfrom types import SimpleNamespace\nimport argparse')
    proof=literal(proof,'    cpu.configure_startup_receipts(scope["CPU_startup_policy"],scope["configuration_pin"],startup_scope_path=scope_path)','    dual_roles=configure_from_candidate_driver_AST(scope,root,cpu,scope_path)')
    proof=literal(proof,'        early=read(receipts/"earliest_owned_worker_identity.json")','        early=read(receipts/"earliest_owned_worker_identity.json")\n        check_actual_dual_role_wire(early,dual_roles,root)')
    functions='''def configure_from_candidate_driver_AST(scope,root,cpu,scope_path):
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

'''
    proof=literal(proof,'def fixture_body(scope_path):',functions+'def fixture_body(scope_path):')
    proofpath=X/'CPU14_dual_configuration_driver_AST_firstpacket_CPU_proof_v6.py'
    raw.append(raw_bundle('dual_config_CPUproof6',proofbase,proofpath,proof));outputs.append(pin(proofpath))
    rawpin=write(X/'source41_dual_configuration_full_ordered_RAW_AST_v2.json',dict(kind='source41_dual_configuration_full_ordered_RAW_AST_v2',initial=initial,files=raw))
    handoff=dict(kind='source41_dual_configuration_recovery_INACTIVE_SOURCE_handoff_v1',scope=scopepin,initial=initial,source_context=pin(X/'source41_SOURCE_context_and_two_config_roles_v1.json'),sources=outputs,RAW=rawpin,NEW_failure_ROOT=failurepin,
        unchanged_sources=[pin(X/n)for n in ['CPU14_startup_launch_adapter_v15.py','CPU14_owned_startup_only_receipt_runner_v2.py','owned_CPU_startup_adapter_proof_watchdog_v3.py']],
        configuration_delta=dict(base=pin(basecfgpath),candidate=cfgpin,changed_only_operational_two_fields=True,actual_new_dual_config_CPU_control_review=None,reader_replacement_preserved=scope['actual_reader_replacement_ROOT'],current_reader_continuation=scope['NEW_source40_failure_ROOT']),
        recorded_false_assert_operands='UNRECORDED_UNKNOWN',source_derived_roles=dict(before='configure record(config_path) versus all3 Producer record(a.stream_configuration)',after='configure SAME record(a.stream_configuration); parent runtime retained separately'),
        future_CPU_scope_requirements=dict(kind='ROOT_CPU14_startup_adapter_firstpacket_exact_ONE_CPU_scope_v1',proof_source=pin(proofpath),candidate_driver_source=pin(driverpath),configuration_pin=cfgpin,saved212000_pointer_scalar_pin='FUTURE_ROOT_BOUND_RETAINED_9bef_METADATA_ONLY',input_pins='ROOT_EXACT_REVIEW_REQUIRED',original_plan361_first3inputs='UNEXECUTED_THIS_STAGE; retained original48rows/full16 labels/replay checks require separate exact input scope',CPU_startup_policy_execution_review_path='FUTURE_EXACT_CPU_SCOPE',prospective_commit=9065255808,prospective_physical=6315255808,runtime_commit=6546820352,runtime_physical=6046820352,parent_private=536870912,watchdog_private=536870912,worker_private=8000000000,family_private=8000000000,body_seconds=360,external_seconds=420,packet_seconds=300,stall_seconds=600),
        proposed_training_command=[str(Path(r'C:/Users/ardao/Documents/Codex/2026-10-10/i-x20/work/nemo-native-env-v2/Scripts/python.exe')),'-X','utf8','-B',str(wrapperpath),'--review',str(X/'ROOT_source41_saved212000_final_ONE_training_execution_review_v1.json'),'--configuration',cfgpin['path'],'--pointer',str(Path(r'E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01/checkpoints/update_000212000/pointer.json')),'--latest-failure',str(Path(r'E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1/pipeline_overlap_RUN07_actual_01/actual_control_namespace_failure_TOOL1_v1.json')),'--branch',str(Path(r'E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01')),'--output',str(Path(r'E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1/resident_RAM_training_actual_31')),'--queue-bytes','268435456','--start-invocation','29','--throughput-resume',failurepin['path']],
        original_ACK260_latest_failure_unchanged=True,runner_training_scope_kind_compatibility='ROOT_ASR18_source40_saved212000_training_exact_ONE_scope_v1 retained intentionally; actual new pins/authority/final scope separately ROOT-bound',new_training_review_and_CPU_gate='UNKNOWN_NOT_ADMITTED',typed_fixture_SOURCE='PENDING_NEXT_SOURCE_ONLY_PIECE',all_science_private_time_and_80MiB_reserves_unchanged=True,no_import_execution_control=True,metadata_identities=ids,failed_builder1_provenance=dict(builder=pin(X/'prepare_source41_dual_configuration_sources_v1.py'),initial=pin(X/'source41_dual_configuration_SOURCE_initial_BEFORE_predicate_v1.json'),owned_session=57319,terminal_numeric=-1,reason='Owner stopped expensive byte-level RAW diff before180s; helper6 only, no candidate execution'))
    handoffpin=write(X/'source41_dual_configuration_INACTIVE_handoff_v2.json',handoff)
    end=sample();assert end['commit']>=3509949440 and end['physical']>=3009949440 and end['private']<=536870912 and end['elapsed']<=180 and end['E_free']>=100000000000
    result=write(X/'source41_dual_configuration_SOURCE_preparation_result_v2.json',dict(passed=True,initial=initial,identities=ids,outputs=outputs,RAW=rawpin,handoff=handoffpin,final=end))
    print(json.dumps(dict(result=result,initial=initial,identities=ids,outputs=outputs,RAW=rawpin,handoff=handoffpin)))
except BaseException as error:
    write(X/'source41_dual_configuration_SOURCE_preparation_failure_v2.json',dict(error=repr(error),traceback=traceback.format_exc(),initial=initial,identities=ids,outputs=outputs))
    raise
