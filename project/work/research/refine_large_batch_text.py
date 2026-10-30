"""Polish new planning prose; update task description via the official CLI."""
import ast
import subprocess
from pathlib import Path

root=Path(__file__).resolve().parents[2]
out=root/'outputs/parakeet_turkish'
changes={'about3GB':'about 3 GB','16GB GPU':'16 GB GPU','full24-layer':'full 24-layer',
         'around1.3e-6 to1.5e-6':'around 1.3e-6 to 1.5e-6','about0.29%':'about 0.29%',
         'and1-1.6%':'and 1-1.6%','same8 Turkish and8 replay':'same 8 Turkish and 8 replay',
         '16TR+16replay':'16 TR + 16 replay','batch4':'batch 4','proposed45minutes':'proposed 45 minutes',
         'B1/v2/accumulation4':'B1/v2, accumulation 4','same8TR+8replay':'same 8 TR + 8 replay',
         'Separate16TR+16replay':'Separate 16 TR + 16 replay','expanded25,628-clip':'expanded 25,628-clip',
         'separate4-byte':'separate 4-byte','RTX5060Ti has16GB':'RTX 5060 Ti has 16 GB',
         'peak~3.05GB':'peak around 3.05 GB','at30s':'at 30 seconds','Approximately376':'Approximately 376',
         'final B2 encoder relativeL2~1.3e-6 to1.5e-6':'final B2 encoder relative L2 around 1.3e-6 to 1.5e-6',
         '~0.29% subsampling and1-1.6%':'approximately 0.29% subsampling and 1-1.6%',
         'batch16':'batch 16','batch32':'batch 32','frozen12-hour':'frozen 12-hour',
         'current12-hour':'current 12-hour','matched20updates':'matched 20 updates'}
for path in (root/'work/research/plan_large_batches.py',root/'work/research/create_large_batch_task.py'):
    text=path.read_text(encoding='utf-8')
    for old,new in changes.items():text=text.replace(old,new)
    path.write_text(text,encoding='utf-8')
tree=ast.parse((root/'work/research/create_large_batch_task.py').read_text())
node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='description' for t in n.targets))
description=ast.literal_eval(node.value)
subprocess.run([str(out/'tools/backlog.exe'),'task','edit','ASR-16','--description',description],cwd=out,check=True,timeout=60)
