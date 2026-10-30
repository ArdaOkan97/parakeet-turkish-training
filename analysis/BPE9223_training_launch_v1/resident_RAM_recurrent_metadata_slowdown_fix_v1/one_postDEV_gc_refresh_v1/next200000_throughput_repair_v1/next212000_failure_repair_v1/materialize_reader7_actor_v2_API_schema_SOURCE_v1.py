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
initial=write_json(D/'reader7_actor_v2_API_schema_SOURCE_initial_v1.json',dict(kind='SOURCE_preparation_initial_BEFORE_predicate',entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),sample=s,required=dict(commit=5399239296,physical=2649239296,private_max=536870912,E_free_min=100000000000),TORCH_EXTENSIONS_DIR=os.environ.get('TORCH_EXTENSIONS_DIR'),candidate_import_execution=False))
assert s['commit']>=5399239296 and s['physical']>=2649239296 and s['private']<=536870912 and s['E_free']>=100000000000


scopepath=D/'ROOT_reader7_actor_API_schema_successor_INACTIVE_SOURCE_scope_v1.json'
scope=json.loads(scopepath.read_text());assert scope['accepted']is True and scope['actual_execution_admitted']is False
bindingpath=pathlib.Path(scope['API09_SOURCE_schema_binding']['path']);bb=bindingpath.read_bytes()
assert pin_bytes(bindingpath,bb)==scope['API09_SOURCE_schema_binding']
binding=json.loads(bb);selected=binding['API_selected'];expected=scope['required_semantics']
assert all(selected[k]==v for k,v in expected.items())
base=pathlib.Path(scope['base_actor']['path']);old=base.read_bytes()
assert pin_bytes(base,old)==scope['base_actor']
before='        assert pin(authority["path"])==authority and read(authority["path"])["accepted"]is True'
after='''        assert pin(authority["path"])==authority
        API_acceptance=read(authority["path"])
        assert API_acceptance["kind"]=="ROOT_reader5_helper4_projector2_both_original500000_real_API_shapes_acceptance_v1"
        assert API_acceptance["reviewer"]=="/root"and API_acceptance["accepted_both_shapes"]is True and API_acceptance["accepted_all_original_lifetime"]is True
        assert API_acceptance["candidate_reader"]==s["reader_template"]
        assert API_acceptance["projection_source"]==pin(HERE/"incremental_scalar_dashboard_projection_v2.py")'''
assert old.count(before.encode())==1
offset=old.index(before.encode());new=old[:offset]+after.encode()+old[offset+len(before.encode()):]
restored=new[:offset]+before.encode()+new[offset+len(after.encode()):];assert restored==old
target=D/'replace_failed_reader6_with_tested_reader7_SOURCE_v2.py';assert not target.exists()
tp=write_source(target,new)
oldt=ast.parse(old.decode());newt=ast.parse(new.decode())
original_nodes={getattr(n,'name',str(i)):sha(ast.dump(n,include_attributes=False).encode())for i,n in enumerate(oldt.body)}
candidate_nodes={getattr(n,'name',str(i)):sha(ast.dump(n,include_attributes=False).encode())for i,n in enumerate(newt.body)}
assert [k for k in original_nodes if original_nodes[k]!=candidate_nodes[k]]==['run']
raw=write_json(D/'reader7_actor_v2_API_schema_only_full_ordered_RAW_AST_v1.json',dict(base=scope['base_actor'],candidate=tp,
    ordered_edits=[dict(offset_before=offset,old_utf8=before,new_utf8=after,before_sha256=sha(old),after_sha256=sha(new))],
    whole_byte_inverse_exact=True,whole_forward_exact=True,base_AST_sha256=astsha(old),candidate_AST_sha256=astsha(new),
    base_top_AST=original_nodes,candidate_top_AST=candidate_nodes,changed_top_definitions=['run'],
    semantic_delta='Only replace unsupported accepted field with actual frozen API09 schema and exact reader/projection candidate comparisons',
    target_resource_environment_single_stop_copy_launch_cleanup_unchanged=True))
handoff=write_json(D/'reader7_actor_v2_API_schema_SOURCE_handoff_v1.json',dict(SOURCE_only=True,actual_execution_admitted=False,source=tp,RAW=raw,scope=pin(scopepath),
    NEW_SOURCE_schema_binding=scope['API09_SOURCE_schema_binding'],exact_operational_API09_schema=expected,
    original_API09_actual_or_capture_reread=False,
    unchanged=dict(failed_pair=[dict(pid=33388,ctime=1792855546.8493547,ppid=38980),dict(pid=38980,ctime=1792855546.824607,ppid=26648)],
        exact_API09_root_SHA='970f937ef2904938980742b8a132e67de758bbd7ed927f4dd5997ad29d0aaa2f',all_targets_guards_caps_resources_wait15_wall180=True,
        one_targeted_stop_fresh_sourcecopy_one_hidden_launch=True,no_second_launch_retry_retarget_or_newreader_kill=True),
    prior_actor01=dict(completion=dict(path=str(D/'reader7_actor01_authority_schema_failure_actual_completion_v1.json'),bytes=15906,
        sha256='db881f051b951f1b5d77f78f1ef33c6c43935150ea1ec33768f957b11a1380a3'),
        TOOL=1,direct_numeric=1,primary="KeyError('accepted')",old_targets=[],source_copied=False,new_launch_attempted=False),
    proposed_future_command=[str(P/'work/nemo-native-env-v2/Scripts/python.exe'),'-X','utf8','-B',str(target),'--ROOT-review',None],
    proposed_fresh_actual_output='reader7_failed6_replacement_actual02_v1',
    exact_future_current_identity_DB_STATE_port_bindings='UNKNOWN until separate ROOT exactONE scope; never auto-rebind',
    initial=initial,entry=who(psutil.Process()),launcher=who(psutil.Process().parent())))
assert time.monotonic()-t0<180
result=write_json(D/'reader7_actor_v2_API_schema_SOURCE_result_v1.json',dict(passed=True,SOURCE_only=True,source=tp,RAW=raw,handoff=handoff,initial=initial,
    no_candidate_import_pycompile_HTTP_DB_control_P_S_or_training=True,entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),elapsed=time.monotonic()-t0,final_sample=sample()))
print(json.dumps(dict(source=tp,RAW=raw,handoff=handoff,result=result,initial=initial,entry=who(psutil.Process()),launcher=who(psutil.Process().parent())),indent=2))

