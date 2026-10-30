import hashlib,json,os,time
from pathlib import Path
import psutil
F=Path(__file__).parent
def pin(p):
    p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def ident(p):return {'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()}
assert os.environ['TORCH_EXTENSIONS_DIR'].lower().startswith('e:')
scopepath=F/'ROOT_fresh_signature_original_collect_focused_CPU_execution_scope_v1.json'
extpath=F/'fresh_signature_collect_observation_external_result_v1.json'
reportpath=F/'fresh_signature_collect_observation_actual_CPU_01/report.json'
toolpath=F/'actual_fresh_signature_collect_CPU_TOOL_v1.json'
scope=load(scopepath);e=load(extpath);r=load(reportpath);t=load(toolpath)
assert t['actual_TOOL_exit_code']==0 and t['terminal_result']['exit_code']==0 and e['Popen_returncode']==0
assert e['passed'] is True and r['passed'] is True and r['semantic_passed'] is True and r['candidate_timing_eligible'] is True
assert e['external_guard_error'] is None and e['required_external_cleanup']==[]
assert e['inputs_before_after_exact'] is True and r['input_pins_before_after_exact'] is True
assert e['input_pins_before']==scope['input_pins'] and e['input_pins_after']==scope['input_pins']
assert r['input_pins_before']==scope['input_pins'] and r['input_pins_after']==scope['input_pins']
assert e['report']==pin(reportpath) and e['ROOT_review']==pin(scopepath)
assert [pin(v['path'])for v in scope['input_pins']]==scope['input_pins']
allwho=[*e['recorded_identities'],e['external_identity'],e['launcher_identity'],r['identity'],r['launcher_identity']]
unique={(v['pid'],v['create_time']):v for v in allwho};closed=[]
for v in unique.values():
    current=None
    try:current=ident(psutil.Process(v['pid']))
    except psutil.NoSuchProcess:pass
    dead=current is None or abs(current['create_time']-v['create_time'])>=.001
    assert dead,('Exact proof process still alive',v,current)
    closed.append({'identity':v,'exact_dead':dead,'current_PID_identity_if_reused':current})
me=psutil.Process()
out={'kind':'ASR18_fresh_signature_collect_actual_CPU_completion_v1','created_unix':time.time(),
 'ROOT_scope':pin(scopepath),'actual_TOOL':pin(toolpath),'actual_TOOL_exit_code':0,'actual_Popen_returncode':0,
 'actual_report':pin(reportpath),'actual_external_result':pin(extpath),'all_exact_metadata_identities_dead':True,
 'exact_metadata_identity_closure_records':closed,'all_recorded_identity_forms':allwho,
 'input_pins_before_after_exact':True,'input_pins':scope['input_pins'],'semantic_passed':r['semantic_passed'],
 'candidate_timing_eligible_in_THIS_bounded_fixed_order':r['candidate_timing_eligible'],
 'concurrent_observation':r['concurrent_observation'],'actual_case_count':len(r['cases']),
 'external_guard_error':None,'required_external_cleanup':[],'sampled_family_private_peak_bytes':e['sampled_family_private_peak_bytes'],
 'limitations':['Original-first then candidate-second observation is bounded and warm/order-sensitive; no production speed/fit/causal guarantee.',
 'Actual constructor and injected failure controls remain explicitly labelled method stubs; reparse unsupported and no attempted denywrite writes are not promoted.',
 'Original gc.collect cadence/threshold/freeze/enabled and admitted production source30/session72166 remain unchanged.'],
 'completion_sealer_identity':ident(me),'completion_sealer_launcher_identity':ident(me.parent())}
p=F/'actual_fresh_signature_collect_CPU_completion_v1.json'
with p.open('x',encoding='utf-8',newline='\n')as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps({'completion':pin(p),'actual_TOOL':pin(toolpath),'exact_dead_identities':len(closed),'candidate_timing_eligible':r['candidate_timing_eligible']}))
