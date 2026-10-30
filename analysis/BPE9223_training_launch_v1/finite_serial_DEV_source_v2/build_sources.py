"""NEW canonical PRE transport fix; old sources remain byte-for-byte intact."""
import ast,hashlib,json
from pathlib import Path
S=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20\outputs\parakeet_turkish')
helper=(S/'asr18_finite_BPE9223_checkpoint_evidence_v1.py').read_text()
old="    require(same(metadata_fingerprint(metadata),pre), 'All854 saved descriptors differ from independent actual PRE hashes')"
new="""    actual_pre = metadata_fingerprint(metadata)
    require(actual_pre['state_sha256'] == digest(actual_pre['state'])
        == pre['state_sha256'] == digest(pre['state']),
        'All854 saved descriptors differ from independent actual PRE canonical hashes')"""
assert helper.count(old)==1
helper=helper.replace(old,new)
serial=(S/'asr18_finite_BPE9223_serial_DEV_v1.py').read_text().replace('asr18_finite_BPE9223_checkpoint_evidence_v1','asr18_finite_BPE9223_checkpoint_evidence_v2').replace('finite_BPE9223_DEV_runtime_v1','finite_BPE9223_DEV_runtime_v2')
runtime=(S/'finite_BPE9223_DEV_runtime_v1.py').read_text().replace('asr18_finite_BPE9223_serial_DEV_v1','asr18_finite_BPE9223_serial_DEV_v2').replace('finite_BPE9223_DEV_runtime_v1','finite_BPE9223_DEV_runtime_v2')
for name,value in [('asr18_finite_BPE9223_checkpoint_evidence_v2.py',helper),('asr18_finite_BPE9223_serial_DEV_v2.py',serial),('finite_BPE9223_DEV_runtime_v2.py',runtime)]:
    ast.parse(value);path=S/name;assert not path.exists();path.write_text(value,encoding='utf-8',newline='\n')
    print(json.dumps({'path':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}))
