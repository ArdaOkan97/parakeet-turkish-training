"""Metadata assembly only; preserve frozen observer while finishing external ledger."""
import ast,hashlib,json
from pathlib import Path
F=Path(__file__).resolve().parent
old=F/'prepare_current_source33_fixed60_live_successor_v3.py';new=F/'prepare_current_source33_fixed60_live_successor_v4.py'
raw=old.read_bytes();value=raw.decode();edits=[]
def edit(a,b):
 global value
 assert value.count(a)==1
 value=value.replace(a,b);edits.append({'old_literal':a,'new_literal':b,'expected_count':1})
edit(" assert inverse.encode()==raw and not new.exists();new.write_bytes(value.encode())",
 " assert inverse.encode()==raw\n if new.exists():assert new.read_bytes()==value.encode(),'Frozen already-materialized source must match EXACT bytes'\n else:new.write_bytes(value.encode())")
edit(" ('(2456,38964,26892,41672,34044,40012,31044)','PROTECTED',2),",
 " (\"popen_identity['pid'] not in (2456,38964,26892,41672,34044,40012,31044)\",\"popen_identity['pid'] not in PROTECTED\",1),\n (\"who['pid'] not in (2456,38964,26892,41672,34044,40012,31044)\",\"who['pid'] not in PROTECTED\",1),")
ast.parse(value);inverse=value
for e in reversed(edits):
 assert inverse.count(e['new_literal'])==1
 inverse=inverse.replace(e['new_literal'],e['old_literal'])
assert inverse.encode()==raw and not new.exists();new.write_bytes(value.encode())
def pin(p):
 b=p.read_bytes();return {'path':str(p.resolve()),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
receipt=F/'source33_live_writer3_preparation_partial_freeze_and_writer4_RAW_repair_v1.json'
assert not receipt.exists()
v={'actual_TOOL_exit':1,'actual_TOOL_chunk':'a1180e','actual_error':'External inverse broad PROTECTED literal also matched new declaration and scope update',
 'observer_control_or_attach_executed':False,'frozen_observer_preserved':pin(F/'observe_existing_source33_thread_CPU_and_stacks_v1.py'),
 'external_and_final_handoff_not_written_by_failed_writer3':True,'old_writer':pin(old),'new_writer':pin(new),
 'exact_replacements':edits,'entire_writer_RAW_inverse_exact':True,
 'repair':'Exact cleanup-expression contexts; validate byte-identical already-frozen observer instead of rewriting'}
receipt.write_bytes((json.dumps(v,indent=2)+'\n').encode())
print(json.dumps({'writer4':pin(new),'receipt':pin(receipt)}),flush=True)
