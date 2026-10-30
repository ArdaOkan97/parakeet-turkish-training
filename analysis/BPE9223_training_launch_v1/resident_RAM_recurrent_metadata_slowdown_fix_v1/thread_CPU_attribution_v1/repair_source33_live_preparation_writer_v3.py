"""Metadata-only assembler repair; generated observers remain unexecuted."""
import ast,hashlib,json
from pathlib import Path
F=Path(__file__).resolve().parent
old=F/'prepare_current_source33_fixed60_live_successor_v2.py'
new=F/'prepare_current_source33_fixed60_live_successor_v3.py'
before=" ('health()','health(review)',2),"
after=" ('before_health=health()','before_health=health(review)',1),\n ('after_health=health()','after_health=health(review)',1),"
raw=old.read_bytes();value=raw.decode();assert value.count(before)==1
value=value.replace(before,after);ast.parse(value)
assert value.count(after)==1 and value.replace(after,before).encode()==raw
assert not new.exists()
for name in ('observe_existing_source33_thread_CPU_and_stacks_v1.py','launch_existing_source33_thread_CPU_external_v1.py'):
 assert not(F/name).exists(),'No observer/external was materialized by rejected writer2'
new.write_bytes(value.encode())
def pin(p):
 b=p.read_bytes();return {'path':str(p.resolve()),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
receipt=F/'source33_live_writer2_preparation_only_rejection_and_writer3_RAW_repair_v1.json'
assert not receipt.exists()
v={'actual_TOOL_exit':1,'actual_TOOL_chunk':'6b9782','actual_error':'Inverse replacement count rejected health(review) call substring also matching function declaration',
 'observer_control_or_attach_executed':False,'no_successor_files_written_by_failed_writer2':True,
 'old_writer':pin(old),'new_writer':pin(new),'exact_replacements':[{'old_literal':before,'new_literal':after,'expected_count':1}],
 'entire_writer_RAW_inverse_exact':True,'repair':'Use exact before_health and after_health call contexts; no observer algorithm change'}
receipt.write_bytes((json.dumps(v,indent=2)+'\n').encode())
print(json.dumps({'writer3':pin(new),'receipt':pin(receipt)}),flush=True)
