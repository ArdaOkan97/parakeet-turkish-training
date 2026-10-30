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
initial=write_json(D/'source40_recovery_SOURCE_inspection_initial_v1.json',dict(kind='SOURCE_preparation_initial_BEFORE_predicate',entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),sample=s,required=dict(commit=5399239296,physical=2649239296,private_max=536870912,E_free_min=100000000000),TORCH_EXTENSIONS_DIR=os.environ.get('TORCH_EXTENSIONS_DIR'),candidate_import_execution=False))
assert s['commit']>=5399239296 and s['physical']>=2649239296 and s['private']<=536870912 and s['E_free']>=100000000000

scopepath=D/'ROOT_source40_reader7_saved212000_INACTIVE_SOURCE_preparation_scope_v2.json'
scope=json.loads(scopepath.read_text(encoding='utf-8-sig'))
assert pin(scopepath)['sha256']=='dc9fdafe5c07c222f3d92cdf03b379501a267fbef1c62539bdab27acee8eb695'
out=[]
for key in ('base_driver','base_wrapper','base_CPU14','tested_early_runner','original_source_configuration','original_CPU_template','original_CPU_numerical_contract'):
    p=pathlib.Path(scope[key]['path']);b=p.read_bytes()
    assert len(b)==scope[key]['bytes']and sha(b)==scope[key]['sha256']
    source=b.decode('utf-8-sig')
    if p.suffix=='.py':
        tree=ast.parse(source)
        symbols=[dict(name=n.name,lineno=n.lineno,end_lineno=n.end_lineno)for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
        if key in ('base_driver','base_wrapper','base_CPU14'):
            lines=source.splitlines()
            match=[i for i,l in enumerate(lines)if any(s in l for s in ('throughput','c59f522b','ResidentFeature','Popen(','command =','command=','--worker','python_path','baseconfig','source_cfg','source_config'))]
            selected=set()
            for i in match:
                for j in range(max(0,i-3),min(len(lines),i+6)):selected.add(j)
            snippets=[dict(line=i+1,text=lines[i])for i in sorted(selected)]
        else:snippets=source
        out.append(dict(key=key,pin=pin_bytes(p,b),symbols=symbols,SOURCE=snippets))
    else:
        value=json.loads(source)
        out.append(dict(key=key,pin=pin_bytes(p,b),top_keys=list(value),selected_fields={k:v for k,v in value.items()if any(w in k.lower()for w in ('throughput','restore','resume','review','python','stream','source','gc','guard','resource'))}))
for key in ('ROOT_saved212000','ROOT_failure212000'):
    p=pathlib.Path(scope[key]['path']);b=p.read_bytes()
    assert len(b)==scope[key]['bytes']and sha(b)==scope[key]['sha256']
    out.append(dict(key=key,pin=pin_bytes(p,b),ROOT_receipt=json.loads(b)))
q=sample();assert q['commit']>=2612368384 and q['physical']>=2112368384 and q['private']<=536870912 and q['E_free']>=100000000000 and time.monotonic()-t0<180
result=dict(kind='source40_exact_BASE_SOURCE_inspection_v1',scope=pin(scopepath),initial=initial,entry=who(psutil.Process()),launcher=who(psutil.Process().parent()),sources=out,final_sample=q,source_only=True)
p=D/'source40_recovery_exact_BASE_SOURCE_inspection_v1.json';write_json(p,result)
print(json.dumps(dict(result=pin(p),sources=out),ensure_ascii=False))

