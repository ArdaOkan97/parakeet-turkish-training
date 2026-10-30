"""Exact top-level durable-head field repair; preserve prior frozen sources."""
from pathlib import Path
import ast,hashlib,json
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';R=P/'work/research'
outputs=[]
for old,new in ((S/'asr18_full_run_parent_controller_v12.py',S/'asr18_full_run_parent_controller_v13.py'),
                (R/'run_ASR18_full_run_owner_v14.py',R/'run_ASR18_full_run_owner_v15.py'),
                (R/'prepare_ASR18_full_owner_configuration_v9.py',R/'prepare_ASR18_full_owner_configuration_v10.py')):
    text=old.read_text(encoding='utf-8-sig')
    if old.parent==S:
        assert text.count("head['lifecycle']['last_recovery']")==2
        text=text.replace("head['lifecycle']['last_recovery']","head['last_recovery']")
    else:
        assert 'asr18_full_run_parent_controller_v12' in text
        text=text.replace('asr18_full_run_parent_controller_v12','asr18_full_run_parent_controller_v13')
    ast.parse(text);assert not new.exists();new.write_text(text,encoding='utf-8',newline='\n')
    raw=new.read_bytes();outputs.append({'path':str(new),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
print(json.dumps(outputs))
