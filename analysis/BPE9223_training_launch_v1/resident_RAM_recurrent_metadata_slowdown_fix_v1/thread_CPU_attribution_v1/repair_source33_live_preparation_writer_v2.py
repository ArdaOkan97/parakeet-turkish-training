"""Metadata-only preparation successor; never executes generated live sources."""
import ast,hashlib,json
from pathlib import Path
F=Path(__file__).resolve().parent;old=F/'prepare_current_source33_fixed60_live_successor_v1.py';new=F/'prepare_current_source33_fixed60_live_successor_v2.py'
value=old.read_text();edits=[]
def edit(a,b,count=1):
 global value
 assert value.count(a)==count,(a[:100],value.count(a),count)
 value=value.replace(a,b);edits.append({'old_literal':a,'new_literal':b,'expected_count':count})
edit("  before=before.replace('\\n','\\r\\n');after=after.replace('\\n','\\r\\n')\n  actual=value.count(before);assert actual>0 and(count is None or actual==count),(before[:140],actual,count)",
 "  actual=value.count(before)\n  if actual==0 and '\\n'in before:\n   before=before.replace('\\n','\\r\\n');after=after.replace('\\n','\\r\\n');actual=value.count(before)\n  assert actual>0 and(count is None or actual==count),(before[:140],actual,count)")
edit('PROTECTED={3652,31268,30732,40192,24268,26892}','PROTECTED={3652,31268,30732,26892}',2)
health=""" cpu_pin=review['current_CPU_ready'];require(pin(cpu_pin['path'])==cpu_pin,'Exact current phase CPU-ready metadata binding')
 cpu_ready=read(cpu_pin['path']);cpu=cpu_ready['producer_identity'];cpu_launcher=identity(psutil.Process(cpu['ppid']))
 require(cpu_ready['source']['protocol']==review['CPU_source']and exact(cpu)
  and cpu['pid']in PROTECTED and cpu_launcher['pid']in PROTECTED
  and any(w['pid']==cpu['pid']and abs(w['create_time']-cpu['create_time'])<.001 for w in review['protected_production_identities'])
  and any(w['pid']==cpu_launcher['pid']and abs(w['create_time']-cpu_launcher['create_time'])<.001 for w in review['protected_production_identities']),
  'Actual current CPU13 cohort and launcher, never historical40192 after172k restart')
 reader=review['reader_identity'];require(reader['pid']==26892 and abs(reader['create_time']-1792652904.3359847)<.001 and exact(reader),'Same exact reader identity; raw compact records captured without guessed schema')
 reader_paths=review['reader_compact_paths'];reader_root=Path('E:/gpu_training_scratchpad/analysis/BPE9223_Trackio_live_v1')
 require(reader_paths=={name:str((reader_root/file).resolve())for name,file in [('current_status','current_status.json'),('mirror_state','mirror_state.json'),('owner','owner.json')]},
  'Only three exact authoritative existing reader compact paths; no restart or mutation')
 reader_raw={name:read(path,1048576)for name,path in reader_paths.items()}
 return {'progress':progress,'boot':pin(I/'boot.json'),'runtime':pin(I/'configuration.json'),
  'current_CPU_ready':cpu_pin,'current_CPU_identity':cpu,'current_CPU_launcher_identity':cpu_launcher,
  'reader_identity':reader,'reader_compact_paths':reader_paths,'reader_compact_raw_without_schema_inference':reader_raw}
"""
addition="""observer_edits += [
 ('def health():','def health(review):',1),
 ('health()','health(review)',2),
 (" return {'progress':progress,'boot':pin(I/'boot.json'),'runtime':pin(I/'configuration.json')}",HEALTH_BLOCK.rstrip(),1)]
""".replace('HEALTH_BLOCK',repr(health))
edit("ol=successor('observe_existing_source31_thread_CPU_and_stacks_v3.py'",addition+"ol=successor('observe_existing_source31_thread_CPU_and_stacks_v3.py'")
edit("'protected_production_identities_required_from_future_ROOT_scope_not_selected_here':True,",
 "'protected_production_identities_required_from_future_ROOT_scope_not_selected_here':True,\n 'future_scope_must_bind_actual_post172k_current_CPU_ready_pin_reader_identity_and_three_reader_compact_paths':True,\n 'reader_compact_paths_expected':{name:str((Path('E:/gpu_training_scratchpad/analysis/BPE9223_Trackio_live_v1')/file).resolve())for name,file in [('current_status','current_status.json'),('mirror_state','mirror_state.json'),('owner','owner.json')]},\n 'historical_CPU40192_is_not_claimed_current_or_hardcoded_in_cleanup_protection':True,")
ast.parse(value);assert not new.exists();new.write_bytes(value.encode())
def pin(p):
 b=p.read_bytes();return {'path':str(p.resolve()),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
inverse=value
for e in reversed(edits):assert inverse.count(e['new_literal'])==e['expected_count'];inverse=inverse.replace(e['new_literal'],e['old_literal'])
assert inverse==old.read_text()
receipt=F/'source33_live_writer1_preparation_only_rejection_and_writer2_RAW_repair_v1.json'
v={'actual_TOOL_exit':1,'actual_TOOL_chunk':'5892d7','actual_error':'Exact multiline source match rejected forcedCRLF against mixed-line-ending live3',
 'observer_control_or_attach_executed':False,'no_successor_files_written_by_failed_writer1':True,
 'old_writer':pin(old),'new_writer':pin(new),'exact_replacements':edits,'entire_writer_RAW_inverse_exact':True,
 'new_source_later_required':{'currentCPU':'actual post172k ready pin and exact protected cohort from future ROOT scope',
 'reader':'exact identity plus bounded existing mutable raw compact status path; not an immutable input pin'}}
assert not receipt.exists();receipt.write_bytes((json.dumps(v,indent=2)+'\n').encode())
print(json.dumps({'writer2':pin(new),'receipt':pin(receipt)}),flush=True)
