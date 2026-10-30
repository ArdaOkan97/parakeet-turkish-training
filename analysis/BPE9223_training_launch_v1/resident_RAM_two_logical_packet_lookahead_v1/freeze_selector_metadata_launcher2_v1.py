import ast,hashlib,json
from pathlib import Path
F=Path('E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1/resident_RAM_two_logical_packet_lookahead_v1')
old=F/'launch_one_reviewed_source29_selector_metadata_CPU_v1.py';new=F/'launch_one_reviewed_source29_selector_metadata_CPU_v2.py'
raw=old.read_bytes();a=b",'--output',str(output),'--ROOT-review'";b=b",'--ROOT-review'";assert raw.count(a)==1
changed=raw.replace(a,b);ast.parse(changed);assert changed.count(b)==1 and changed.replace(b,a)==raw
assert not new.exists();new.write_bytes(changed)
def pin(p):
 p=Path(p).resolve();v=p.read_bytes();return {'path':str(p),'bytes':len(v),'sha256':hashlib.sha256(v).hexdigest()}
ledger={'kind':'ASR18_UNEXECUTED_source29_metadata_launcher_CLI2_exact_raw_inverse_v1','old_source':pin(old),'new_source':pin(new),
 'raw_byte_replacement':{'old_literal':a.decode(),'new_literal':b.decode(),'expected_count':1},'full_raw_inverse_exact':True,
 'only_change':'Remove unused --output pair because frozen fixture owns exact fixed fresh path; all fresh-output checks/resource/identity/guard behavior unchanged',
 'old_launcher_executed':False,'new_launcher_executed':False,'fixture_and_source29_changed':False}
path=F/'selector_metadata_launcher1_to2_exact_raw_inverse_v1.json'
with path.open('x',encoding='utf-8')as f:json.dump(ledger,f,indent=2);f.write('\n')
print(json.dumps({'launcher2':pin(new),'ledger':pin(path),'source_only':True}))
