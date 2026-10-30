import ast,hashlib,json
from pathlib import Path
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');R=P/'work/research';S=P/'outputs/parakeet_turkish'
helper=S/'asr18_completed_hard_checkpoint_evidence_v1.py';sha=hashlib.sha256(helper.read_bytes()).hexdigest()
text=(R/'retire_ASR18_completed_hard_features_v1.py').read_text()
text=text.replace("CPU_REPORT=N/'hard_CPU_window_actual_02/report.json'", "CPU_REPORT=N/'hard_CPU_window_actual_02/report.json'\nrequire(record(finite.__file__)['sha256']=='"+sha+"','Exact frozen hard checkpoint evidence helper')")
text=text.replace("'source':record(__file__),'finite_packet':record(packet_path)","'source':record(__file__),'evidence_source':record(finite.__file__),'finite_packet':record(packet_path)")
text=text.replace('Exact20/16row/32array/4JSON','Exact1/16row/32array/4JSON')
target=R/'retire_ASR18_completed_hard_features_v2.py';assert not target.exists();ast.parse(text);target.write_text(text,encoding='utf-8',newline='\n')
print(json.dumps({'path':str(target),'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'evidence_sha256':sha}))
