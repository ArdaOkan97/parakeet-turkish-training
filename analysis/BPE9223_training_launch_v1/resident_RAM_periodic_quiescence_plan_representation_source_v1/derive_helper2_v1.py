"""Preserve helper1; derive only original RNG representation normalization."""
import ast,difflib,hashlib,json
from pathlib import Path
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish';D=Path(__file__).parent
old=(S/'asr18_resident_periodic_CPU_quiescence_v1.py').read_text(encoding='utf-8-sig');new=old;changes=[]
def replace(a,b,label):
 global new
 assert new.count(a)==1,(label,new.count(a));new=new.replace(a,b);changes.append({'label':label,'old':a,'new':b})
replace('from atomic_io_retry_v1 import read_json','from copy import deepcopy\nfrom atomic_io_retry_v1 import read_json\nfrom bpe_final_DATA_schedule_context_v2 import normalize_schedule_state','Original loader representation helper')
replace("        require(record(plan_pin['path']) == plan_pin and read_json(plan_pin['path']) == plan,\n                'Keep the exact previously submitted private plan')", "        stored = read_json(plan_pin['path'])\n        represented = deepcopy(stored)\n        represented['final_private_planning_sampler'] = normalize_schedule_state(\n            represented['final_private_planning_sampler'])\n        require(record(plan_pin['path']) == plan_pin and represented == plan\n                and digest(stored) == digest(plan),\n                'Exact wire plan and original two carried RNG tuple representations')",'Normalize only final sampler two RNG paths; exact full wire digest and all other fields')
replace("            'full_prefetch_and_DEV_reservations_unchanged': True}","            'full_prefetch_and_DEV_reservations_unchanged': True,\n            'only_original_two_carried_RNG_representation_paths_normalized': True}",'Explicit comparison provenance')
ast.parse(new);inverse=new
for c in reversed(changes):assert inverse.count(c['new'])==1;inverse=inverse.replace(c['new'],c['old'])
assert inverse==old
out=S/'asr18_resident_periodic_CPU_quiescence_v2.py';assert not out.exists();out.write_text(new,encoding='utf-8')
(D/'helper1_to2.diff.txt').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='helper1',tofile='helper2')),encoding='utf-8')
(D/'helper2_declared_replacements.json').write_text(json.dumps(changes,indent=2)+'\n',encoding='utf-8')
def pin(p):
 p=Path(p);raw=p.read_bytes();return {'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
print(json.dumps({'source':pin(out),'diff':pin(D/'helper1_to2.diff.txt'),'full_inverse_source_equality':True}))
