"""Portable NeMo config repair; tensor bytes are streamed unchanged."""
import hashlib,io,json,os,shutil,sys,tarfile,time
from pathlib import Path
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');R=Path(__file__).parent
sys.path.insert(0,str(P/'outputs/parakeet_turkish'))
from paired_windows_commit_telemetry_v1 import paired_sample
from resource_watchdog_v2 import sample_memory
def write(path,data):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(data,f,indent=2);f.flush();os.fsync(f.fileno())
start=time.monotonic();samples=[paired_sample(),paired_sample()];write(R/'portable_archive_v3_admission.json',samples)
assert all(x['global_memory_status_ex']['ullAvailPageFile_bytes']>=6028384896 and x['get_performance_info_after']['physical_available_bytes']>=3278384896 for x in samples)
old=Path(r'E:\gpu_training_scratchpad\exports\parakeet-tdt-0.6b-v3-tr-experimental-400000-20261029-v2')
new=old.with_name(old.name[:-2]+'v3')
write(R/'portable_archive_v3_scope.json',{'source':'Newly exported native .nemo, config only transformation plus tensor byte-copy','output':str(new),'changes':['validation_ds null -> empty mapping, required by NeMo transcribe','tokenizer.dir absolute local path -> dot; embedded nemo: assets preserved'],'tensor_decode':False,'original_training_inputs':False,'self512MiB':True,'wall180':True})
assert not new.exists();new.mkdir()
def guard():
    m=sample_memory();assert time.monotonic()-start<180
    assert m['system_memory']['commit_available_bytes']>=3509949440 and m['system_memory']['physical_available_bytes']>=3009949440
    assert m['worker_private_memory']['private_bytes']<512*1024**2 and shutil.disk_usage(new).free>=100000000000
guard();shutil.copytree(old/'adapter',new/'adapter');(new/'merged').mkdir()
for f in (old/'merged').iterdir():
    if f.suffix!='.nemo':shutil.copyfile(f,new/'merged'/f.name)
source=old/'merged/parakeet-tdt-0.6b-v3-tr-experimental.nemo';target=new/'merged'/source.name
members=[]
class Reader:
    def __init__(self,file):self.file=file;self.h=hashlib.sha256()
    def read(self,n=-1):
        guard();data=self.file.read(n);self.h.update(data);return data
with tarfile.open(source,'r:') as src,tarfile.open(target,'w:') as dst:
    for member in src.getmembers():
        guard();assert member.name in ('.','./') or (member.isfile() and len(Path(member.name).parts)==1)
        if member.isdir():dst.addfile(member);continue
        file=src.extractfile(member)
        if Path(member.name).name=='model_config.yaml':
            raw=file.read();(R/'portable_archive_v3_source_config_RAW.yaml').write_bytes(raw)
            import yaml
            cfg=yaml.safe_load(raw);assert cfg['validation_ds'] is None
            cfg['validation_ds']={};cfg['tokenizer']['dir']='.'
            data=yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False).encode('utf-8')
            assert b'E:\\' not in data and b'C:\\' not in data
            member.size=len(data);dst.addfile(member,io.BytesIO(data));members.append({'name':member.name,'changed':'config only','sha256':hashlib.sha256(data).hexdigest()})
        else:
            reader=Reader(file);dst.addfile(member,reader);members.append({'name':member.name,'bytes':member.size,'sha256':reader.h.hexdigest(),'byte_copy':True})
        file.close()
h=hashlib.sha256()
with target.open('rb') as f:
    while b:=f.read(4*1024**2):guard();h.update(b)
write(R/'portable_archive_v3_complete.json',{'passed':True,'output':str(new),'native_sha256':h.hexdigest(),'native_bytes':target.stat().st_size,'members':members,'seconds':time.monotonic()-start})
print((R/'portable_archive_v3_complete.json').read_text())
