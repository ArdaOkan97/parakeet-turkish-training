"""Prepare the normal172k metadata seal; never executes seal or DEV."""
import ast,hashlib,json,os
from pathlib import Path
F=Path(__file__).resolve().parent;FIX=F.parent/'resident_RAM_recurrent_metadata_slowdown_fix_v1'
assert Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad'))
old=FIX/'seal_periodic168000_source31_CPU11_helper3_max2_saved168200_v2.py'
new=F/'seal_periodic172000_source33_CPU13_helper3_max2_saved172200_v1.py'
raw=old.read_bytes();value=raw.decode();edits=[]
def edit(a,b):
 global value
 count=value.count(a);assert count>0,a
 value=value.replace(a,b);edits.append({'old_literal':a,'new_literal':b,'expected_count':count})
for a,b in [('actual168k','actual172k'),('source31','source33'),('CPU11','CPU13'),
 ('invocations/owner_000023','invocations/owner_000024'),('resident_RAM_training_actual_24','resident_RAM_training_actual_25'),
 ('168000','172000'),('168200','172200'),('164000','168000'),('2901','65067'),('620','1620'),
 ('fastmss_resident_feature_stream_v11.py','fastmss_resident_feature_stream_v13.py'),
 ('run_ASR18_BPE9223_resident_RAM_training_v31.py','run_ASR18_BPE9223_resident_RAM_training_v33.py'),
 ('One real164k record','One real172k record'),('One previous160k quality record','One previous168k quality record')]:edit(a,b)
ast.parse(value);inverse=value
for e in reversed(edits):
 assert inverse.count(e['new_literal'])==e['expected_count'],e['new_literal']
 inverse=inverse.replace(e['new_literal'],e['old_literal'])
assert inverse.encode()==raw and not new.exists();new.write_bytes(value.encode())
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
ledger=F/'source33_periodic172000_metadata_observer_exact_RAW_inverse_v1.json'
v={'old':pin(old),'new':pin(new),'replacements_in_forward_order':edits,'entire_RAW_forward_inverse_exact':True,
 'newline_normalization':'NONE; original raw CRLF bytes unchanged outside declared literals',
 'actual_CPU_cohort':'170400 restored baseline to172000 =1600 native packets plus20 original same-plan unconsumed drain =1620 acquires and returns',
 'execute_only_after_actual172000full350_normalclose0_preservation_resumedpositive_and_standard172200_postpositive':True,
 'current_CPU_ready_path':'owner_000024/CPU_after_periodic_DEV_000172000/actual_CPU_ready.json',
 'no_new_DEV_inference_audio_model_journal_payload_or_source_change':True,'execution_admitted':False,
 'old168k_scoring_phase_journal_and_persistence_audits_stay_closed':True,
 'failed_old168k_observer_v1_preserved_not_repeated':True}
assert not ledger.exists();ledger.write_bytes((json.dumps(v,indent=2)+'\n').encode())
print(json.dumps({'source':pin(new),'ledger':pin(ledger),'execution_admitted':False}),flush=True)
