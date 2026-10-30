"""Inactive wrapper18 only changes typed predecessor and helper pin."""
import ast,difflib,hashlib,json
from pathlib import Path
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');R=P/'work/research';D=Path(__file__).parent
old=(R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v17.py').read_text(encoding='utf-8-sig');new=old;changes=[]
def rep(a,b,label,count=1):
 global new
 assert new.count(a)==count,(label,new.count(a));new=new.replace(a,b);changes.append({'label':label,'old':a,'new':b,'count':count})
rep('run_ASR18_BPE9223_resident_RAM_training_v17','run_ASR18_BPE9223_resident_RAM_training_v18','new exact driver/imports',4)
rep('asr18_resident_periodic_CPU_quiescence_v1.py','asr18_resident_periodic_CPU_quiescence_v2.py','new exact helper pin')
rep("a.DEV_admission_resume is None,'New normal-boundary entry", "a.DEV_admission_resume is None and a.periodic_quiescence_resume is None,'New normal-boundary entry",'genuine normal0 stays strict')
rep('    elif a.DEV_admission_resume is not None:\n', "    elif a.periodic_quiescence_resume is not None:\n        require(a.abnormal_resume is None and a.original_interval_resume is None and a.source_transition_resume is None and a.DEV_frontend_resume is None and a.DEV_admission_resume is None\n            and review['periodic_quiescence_resume']==record(a.periodic_quiescence_resume)\n            and a.start_invocation==review['initial_owner_invocation']==14,\n            'Exact saved140000 helper1 failure is not normal0')\n    elif a.DEV_admission_resume is not None:\n",'exact typed140k predecessor')
rep("    require(control.is_relative_to(Path(r'E:\\gpu_training_scratchpad'))", "    if a.periodic_quiescence_resume is not None:\n        from run_ASR18_BPE9223_resident_RAM_training_v18 import checked_periodic_quiescence_resume\n        checked_periodic_quiescence_resume(record(a.periodic_quiescence_resume),base=configuration,folder=branch,pointer_pin=record(a.pointer))\n    require(control.is_relative_to(Path(r'E:\\gpu_training_scratchpad'))",'typed metadata gate before output/GPU')
rep("            if a.DEV_admission_resume is not None:command+=['--DEV-admission-resume',a.DEV_admission_resume]", "            if a.periodic_quiescence_resume is not None:command+=['--periodic-quiescence-resume',a.periodic_quiescence_resume]\n            elif a.DEV_admission_resume is not None:command+=['--DEV-admission-resume',a.DEV_admission_resume]",'forward exact typed flag')
rep("p.add_argument('--prior-process');p.add_argument('--DEV-admission-resume')", "p.add_argument('--prior-process');p.add_argument('--periodic-quiescence-resume');p.add_argument('--DEV-admission-resume')",'explicit flag')
ast.parse(new);inverse=new
for c in reversed(changes):assert inverse.count(c['new'])==c['count'];inverse=inverse.replace(c['new'],c['old'])
assert inverse==old;out=R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v18.py';assert not out.exists();out.write_text(new,encoding='utf-8');(D/'wrapper17_to18.diff.txt').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='wrapper17',tofile='wrapper18')),encoding='utf-8');(D/'wrapper18_declared_replacements.json').write_text(json.dumps(changes,indent=2)+'\n',encoding='utf-8')
def pin(p):
 p=Path(p);b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
print(json.dumps({'source':pin(out),'diff':pin(D/'wrapper17_to18.diff.txt'),'full_inverse_source_equality':True}))
