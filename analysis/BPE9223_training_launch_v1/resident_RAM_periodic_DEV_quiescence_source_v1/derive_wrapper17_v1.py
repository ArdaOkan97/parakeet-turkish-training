"""Derive inactive wrapper17 with a typed saved136000 DEV-admission predecessor."""
import ast,difflib,hashlib,json
from pathlib import Path
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');R=P/'work/research';S=P/'outputs/parakeet_turkish';D=Path(__file__).parent
old=(R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v16.py').read_text(encoding='utf-8-sig');new=old;changes=[]
def pin(p):
 p=Path(p);raw=p.read_bytes();return {'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def req(ok,message):
 if not ok:raise ValueError(message)
def replace(a,b,label,count=1):
 global new
 req(new.count(a)==count,label+' exact occurrence '+str(new.count(a)));new=new.replace(a,b);changes.append({'label':label,'old':a,'new':b,'count':count})
replace('run_ASR18_BPE9223_resident_RAM_training_v16','run_ASR18_BPE9223_resident_RAM_training_v17','exact new driver/imports',3)
replace("    if a.prior_process is not None:\n", "    require(review['periodic_DEV_CPU_quiescence_policy']==configuration['periodic_DEV_CPU_quiescence_policy']\n        and configuration['periodic_DEV_CPU_quiescence_policy']['source']==record(S/'asr18_resident_periodic_CPU_quiescence_v1.py'),\n        'Exact reviewed periodic CPU phase scheduling; unchanged DEV and full reservations')\n    if a.prior_process is not None:\n",'review exact phase policy')
replace("a.DEV_frontend_resume is None,'New normal-boundary entry", "a.DEV_frontend_resume is None and a.DEV_admission_resume is None,'New normal-boundary entry",'normal0 remains strict')
replace("    elif a.abnormal_resume is not None:\n", "    elif a.DEV_admission_resume is not None:\n        require(a.abnormal_resume is None and a.original_interval_resume is None and a.source_transition_resume is None and a.DEV_frontend_resume is None\n            and review['DEV_admission_resume']==record(a.DEV_admission_resume)\n            and a.start_invocation==review['initial_owner_invocation']==13,\n            'Exact saved136000 prospective DEV failure is not normal0')\n    elif a.abnormal_resume is not None:\n",'one typed DEV admission predecessor')
replace("    require(control.is_relative_to(Path(r'E:\\gpu_training_scratchpad'))", "    if a.DEV_admission_resume is not None:\n        from run_ASR18_BPE9223_resident_RAM_training_v17 import checked_DEV_admission_resume\n        checked_DEV_admission_resume(record(a.DEV_admission_resume),base=configuration,folder=branch,pointer_pin=record(a.pointer))\n    require(control.is_relative_to(Path(r'E:\\gpu_training_scratchpad'))",'actual typed predecessor metadata gate before output/GPU')
replace("            if a.abnormal_resume is not None:command+=['--abnormal-resume',a.abnormal_resume]", "            if a.DEV_admission_resume is not None:command+=['--DEV-admission-resume',a.DEV_admission_resume]\n            elif a.abnormal_resume is not None:command+=['--abnormal-resume',a.abnormal_resume]",'forward exact typed flag')
replace("p.add_argument('--prior-process');p.add_argument('--original-interval-resume')", "p.add_argument('--prior-process');p.add_argument('--DEV-admission-resume');p.add_argument('--original-interval-resume')",'explicit CLI flag')
ast.parse(new);inverse=new
for c in reversed(changes):
 req(inverse.count(c['new'])==c['count'],'Inverse '+c['label']);inverse=inverse.replace(c['new'],c['old'])
req(inverse==old,'Exact full wrapper inverse')
out=R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v17.py';req(not out.exists(),'Preserve source');out.write_text(new,encoding='utf-8')
(D/'wrapper16_to17.diff.txt').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='wrapper16',tofile='wrapper17')),encoding='utf-8')
(D/'wrapper17_declared_replacements.json').write_text(json.dumps(changes,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'wrapper':pin(out),'diff':pin(D/'wrapper16_to17.diff.txt'),'full_inverse_source_equality':True}))
