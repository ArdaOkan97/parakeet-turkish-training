"""One source-only recovery-timeout key correction; preserve every predecessor."""
import ast,hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parents[2];R=P/'work/research';S=P/'outputs/parakeet_turkish';A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def pin(p):
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return {'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':h}
def write(p,d):
 assert not p.exists()
 p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
old=R/'run_ASR18_hard_effective16_v2.py';new=R/'run_ASR18_hard_effective16_v3.py'
text=old.read_text(encoding='utf-8-sig');assert text.count("config['maximum_final_recovery_seconds']")==1
text=text.replace("config['maximum_final_recovery_seconds']","config['maximum_recovery_phase_seconds']")
assert not new.exists();new.write_text(text,encoding='utf-8',newline='\r\n');ast.parse(text)
t=read(S/'asr18_hard_effective16.disabled.v1.json')
t.update(kind='ASR18_hard_effective16_config_TEMPLATE_NOT_LAUNCHABLE_v2',parent_driver_source=pin(new),worker_driver_source=str(new.resolve()))
assert t['maximum_recovery_phase_seconds']==180 and 'maximum_final_recovery_seconds' not in t
tp=S/'asr18_hard_effective16.disabled.v2.json';write(tp,t)
mp=A/'hard_effective16_full_caller_source_review_manifest_v1.json';m=read(mp)
bindings={p['path']:p for p in m['bindings']}
for p in (new,tp,Path(__file__)):bindings[str(p.resolve())]=pin(p)
for p in bindings.values():assert pin(Path(p['path']))==p
m.update(kind='ASR18_hard_effective16_CPU8_full_caller_source_review_manifest_v2',bindings=sorted(bindings.values(),key=lambda x:x['path']),
 configuration_template=pin(tp),direct_sources=[pin(new) if p['path']==str(old.resolve()) else p for p in m['direct_sources']],
 previous_hard_source_manifest=pin(mp),source_delta=m['source_delta']+'; parent3 only uses established maximum_recovery_phase_seconds180 key')
out=A/'hard_effective16_full_caller_source_review_manifest_v2.json';write(out,m)
print(json.dumps({'parent':pin(new),'template':pin(tp),'manifest':pin(out),'bindings':len(bindings)}))
