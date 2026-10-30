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
initial=write_json(D/'source40_resource_metadata_correction_initial_v1.json',dict(kind='SOURCE_preparation_initial_BEFORE_predicate',entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),sample=s,required=dict(commit=5399239296,physical=2649239296,private_max=536870912,E_free_min=100000000000),TORCH_EXTENSIONS_DIR=os.environ.get('TORCH_EXTENSIONS_DIR'),candidate_import_execution=False))
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


previous_path=D/'source40_adapter_CPU_proof_SOURCE_handoff_v1.json';previous_raw=previous_path.read_bytes();h=json.loads(previous_raw)
assert h['SOURCE_only']is True and h['actual_execution_admitted']is False
old=h['resources']['new_minimum'];correct=dict(prospective_commit=8528384896,prospective_physical=5778384896,runtime_commit=6009949440,runtime_physical=5509949440)
original=h['resources']['original_CPUproof4_minimum'];addition=8*1024**2-1024**2+16*1024**2+64*1024**2
assert addition==91226112 and all(original[k]+addition==v for k,v in correct.items())
h['resources']['new_minimum']=correct
h['resources']['SOURCE_lower_bound_assertions_only']=dict(frozen_proof_watchdog_lower_bounds=old,
    operative_guards='Every initial/body/watchdog predicate uses future scope.resources; ROOT binds corrected stronger values, no candidate byte change',
    original_private_family_time_caps_unchanged=True)
h['metadata_successor_provenance']=dict(previous_handoff=pin(previous_path),correction='ONLY four arithmetic totals; frozen proof/watchdog bytes and algorithms unchanged',difference_bytes=2688)
h['initial_metadata_correction']=initial
hp=write_json(D/'source40_adapter_CPU_proof_SOURCE_handoff_v2.json',h)
initial_names=['source40_startup_runner_adapter_SOURCE_initial_v1.json','source40_recovery_driver_wrapper_helper_SOURCE_initial_v1.json','source40_recovery_driver_wrapper_helper_SOURCE_initial_v2.json','source40_adapter_CPU_proof_SOURCE_initial_v1.json']
numbers=['source40_recovery_driver_wrapper_helper_SOURCE_numeric_exit_v1.json','source40_recovery_driver_wrapper_helper_SOURCE_numeric_exit_v2.json','source40_adapter_CPU_proof_SOURCE_numeric_exit_v1.json']
closure=[]
for name in initial_names:
    p=D/name;value=json.loads(p.read_text());identities=[value['entry'],value['launcher']]
    rows=[]
    for q in identities:
        try:
            p2=psutil.Process(q['pid']);dead=abs(p2.create_time()-q['ctime'])>=.001
            current=None if dead else who(p2)
        except psutil.NoSuchProcess:dead=True;current=None
        rows.append(dict(identity=q,exact_dead=dead,current_exact_identity=current))
    closure.append(dict(initial=pin(p),identities=rows))
supplement=write_json(D/'source40_SOURCE_preparation_resource_correction_and_known_deaths_v1.json',
    dict(SOURCE_only=True,corrected_handoff=hp,old_hand_off=pin(previous_path),difference_only=['resources.new_minimum four totals','SOURCE assertion-vs-operative guard explanation','immutable provenance/initial'],
    failed_builder1=dict(initial=pin(D/'source40_recovery_driver_wrapper_helper_SOURCE_initial_v1.json'),numeric=pin(D/numbers[0]),actual='AssertionError CRLF slice before writes; no candidate/proof failure'),
    exact_known_SOURCE_prep_deaths=closure,numeric_receipts=[dict(pin=pin(D/n),value=json.loads((D/n).read_text()))for n in numbers],
    current_entry=who(psutil.Process()),current_launcher=who(psutil.Process().parent()),initial=initial,all_actual_proof_execution=False,
    original_startup_piece_numeric_OS_exit='UNRECORDED_UNKNOWN; original TOOL0 known separately'))
print(json.dumps(dict(handoff=hp,supplement=supplement,initial=initial,entry=who(psutil.Process()),launcher=who(psutil.Process().parent())),indent=2))

