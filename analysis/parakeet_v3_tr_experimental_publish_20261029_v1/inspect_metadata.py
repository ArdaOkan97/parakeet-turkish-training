import sys,json,os,re,time,hashlib
from pathlib import Path
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');R=Path(__file__).parent
sys.path.insert(0,str(P/'outputs/parakeet_turkish'))
from paired_windows_commit_telemetry_v1 import paired_sample
from resource_watchdog_v2 import sample_memory
def save(name,value):
    with (R/name).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
samples=[paired_sample(),paired_sample()];save('inspection_admission.json',samples)
assert all(s['global_memory_status_ex']['ullAvailPageFile_bytes']>=6028384896 and s['get_performance_info_after']['physical_available_bytes']>=3278384896 for s in samples)
files={'plan':Path(r'E:\gpu_training_scratchpad\analysis\BPE_weak_then_clean_v1\preparation_plan_v1.json'),'sources':Path(r'E:\gpu_training_scratchpad\analysis\BPE_weak_then_clean_v1\final_source_identity_inventory_v1.json'),'checkpoint':Path(r'E:\gpu_training_scratchpad\analysis\FLEURS_full_vs_v3_20261029_v1\checkpoint_metadata_RAW.json')}
save('inspection_scope.json',{k:str(p) for k,p in files.items()})
data={}
for name,path in files.items():
    assert 0<path.stat().st_size<16777216
    raw=path.read_bytes()
    with (R/(name+'_RAW.json')).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    data[name]=json.loads(raw)
from bpe_streaming_checkpoint_semantics_v2 import decode_scalar_tree,select
provenance=decode_scalar_tree(select(data['checkpoint']['tree'],'source_provenance'))
identity=decode_scalar_tree(select(data['checkpoint']['tree'],'identity'))
augmentation=decode_scalar_tree(select(data['checkpoint']['tree'],'augmentation'))
for n,v in [('provenance',provenance),('identity',identity),('augmentation',augmentation)]:save(n+'.json',v)
matches=[]
def visit(v,path=''):
    if isinstance(v,dict):
        for k,x in v.items():visit(x,path+'/'+str(k))
    elif isinstance(v,list):
        for i,x in enumerate(v):visit(x,path+'/'+str(i))
    elif isinstance(v,str) and (any(w in path.lower() for w in ('licens','repository','repo_id','registry','source_manifest')) or re.match(r'https?://',v)):
        matches.append({'field':path,'value':v})
for n,v in [('sources',data['sources']),('provenance',provenance),('identity',identity),('augmentation',augmentation)]:visit(v,n)
save('publication_metadata_projection.json',{'sources':data['plan']['sources'],'cleaner':data['plan']['cleaner_phase'],'matches':matches})
print(json.dumps({'sources':data['plan']['sources'],'matches':matches},ensure_ascii=False))
