import hashlib,json,os,time
from pathlib import Path
import psutil
F=Path(__file__).parent
assert os.environ['TORCH_EXTENSIONS_DIR'].lower().startswith('e:')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def ident(p):return {'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()}
scopepath=F/'ROOT_source31_actual_closed_source30_signature_metadata_selector_CPU_scope_v1.json';extpath=F/'typed_signature_collect_selector_actual_01_external_result.json'
reportpath=F/'typed_signature_collect_selector_actual_01/report.json';toolpath=F/'actual_source31_typed_selector_CPU_TOOL_v1.json'
s=read(scopepath);e=read(extpath);r=read(reportpath);tool=read(toolpath);entry=read(e['entry']['path']);fixture_entry=read(F/'typed_signature_collect_selector_actual_01/entry.json')
assert tool['actual_TOOL_exit_code']==0 and e['actual_Popen_wait_code']==0 and e['external_guard_error']is None and e['targeted_cleanup']==[] and e['failure']is None and r['passed']is True
assert e['report']==pin(reportpath) and e['inputs_before_and_after_exact']==s['input_pins'] and [pin(v['path'])for v in s['input_pins']]==s['input_pins']
who=[entry['external'],entry['external_launcher'],fixture_entry['parent'],fixture_entry['launcher'],*[x['identity']for x in e['actual_owned_identities']]]
unique={(v['pid'],v['create_time']):v for v in who};deaths=[]
for v in unique.values():
 current=None
 try:current=ident(psutil.Process(v['pid']))
 except psutil.NoSuchProcess:pass
 dead=current is None or abs(current['create_time']-v['create_time'])>=.001
 assert dead,('Exact metadata identity still alive',v,current)
 deaths.append({'identity':v,'exact_dead':dead,'current_PID_identity_if_reused':current})
me=psutil.Process();value={'kind':'ASR18_actual_source31_typed_selector_metadata_completion_v1','created_unix':time.time(),
 'scope':pin(scopepath),'actual_TOOL':pin(toolpath),'actual_TOOL_exit_code':0,'actual_Popen_returncode':0,'actual_external_result':pin(extpath),'actual_report':pin(reportpath),
 'all_exact_metadata_identities_dead':True,'exact_metadata_identity_closure_records':deaths,'all_recorded_identity_forms':who,
 'all_inputs_before_after_now_exact':True,'input_pins':s['input_pins'],'actual_case_count':len(r['results']),
 'external_guard_error':None,'targeted_cleanup':[],'sampled_family_private_peak_bytes':e['sampled_external_plus_owned_family_peak_bytes'],
 'entire_real_prepostrestore_and_wrapper_entry_forwarding_checked':r['entire_pre_and_postrestore_metadata_dictionaries_and_wrapper_entry_forwarding_executed'],
 'explicit_metadata_doubles_not_real_restore_or_training':True,'actual_restore_copy_native_GPU_audio_operations':0,
 'sealer_identity':ident(me),'sealer_launcher_identity':ident(me.parent())}
p=F/'actual_source31_typed_selector_CPU_completion_v1.json'
with p.open('x',encoding='utf-8',newline='\n')as f:json.dump(value,f,indent=2);f.write('\n')
print(json.dumps({'completion':pin(p),'TOOL':pin(toolpath),'exact_dead_identities':len(deaths),'actual_cases':len(r['results'])}))
