"""NEW exact callback keyword fix; pure source AST, no target imports."""
from pathlib import Path
import ast,hashlib
P=Path(__file__).resolve().parents[2]
for old,new,before,after in (
 (P/'outputs/parakeet_turkish/asr18_full_run_parent_controller_v4.py',P/'outputs/parakeet_turkish/asr18_full_run_parent_controller_v5.py',
  '},callbacks=methods)','},functions=methods)'),
 (P/'work/research/run_ASR18_full_run_owner_v6.py',P/'work/research/run_ASR18_full_run_owner_v7.py',
  'from asr18_full_run_parent_controller_v4 import run','from asr18_full_run_parent_controller_v5 import run')):
    assert not new.exists();text=old.read_text(encoding='utf-8-sig');assert text.count(before)==1
    text=text.replace(before,after);ast.parse(text);compile(text,str(new),'exec');new.write_text(text,encoding='utf-8',newline='\n')
    raw=new.read_bytes();print({'path':str(new),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'AST_only':True})
