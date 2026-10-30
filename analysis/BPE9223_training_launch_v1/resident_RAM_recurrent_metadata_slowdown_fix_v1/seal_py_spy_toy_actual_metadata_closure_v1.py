import hashlib,json
from pathlib import Path
import psutil

F=Path(__file__).parent;O=F/'py_spy_same_interpreter_toy_actual_01'
def pin(p):
    p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def dead(w):
    try:return abs(psutil.Process(w['pid']).create_time()-w['create_time'])>=.001
    except psutil.NoSuchProcess:return True
report=json.loads((O/'external_result.json').read_text(encoding='utf-8'))
tool=json.loads((O/'actual_TOOL.json').read_text(encoding='utf-8'))
assert report['passed']is True and report['Popen_returncode']==0 and report['toy_returncode']==0
assert report['external_guard_error']is None and not report['required_external_cleanup']
assert tool['exit_code']==0 and 'session_id'not in tool
who={}
for w in report['recorded_identities']+[report['external_identity'],report['launcher_identity'],report['toy_launcher_identity'],report['toy_native_identity'],report['sampler_identity']]:
    assert w['pid']!=40012
    who[(w['pid'],w['create_time'])]=w
closure=[{'identity':w,'exact_dead':dead(w)}for w in who.values()]
assert all(v['exact_dead']for v in closure)
out={'kind':'ASR18_py_spy_same_interpreter_toy_actual_metadata_completion_v1','passed':True,
 'actual_outer_TOOL_exit':tool['exit_code'],'actual_outer_TOOL_chunk':tool['chunk_id'],
 'actual_sampler_Popen_returncode':report['Popen_returncode'],'actual_toy_launcher_Popen_returncode':report['toy_returncode'],
 'actual_toy_native_completion_return':'Toy completion passed true and actualPID matched; separate OS exit inferred only via original Popen0, not invented as an independent recorded numeric exit',
 'actual_TOOL':pin(O/'actual_TOOL.json'),'external_result':pin(O/'external_result.json'),
 'ROOT_scope':report['ROOT_review'],'profile':report['outputs']['profile.speedscope.json'],
 'recorded_owned_identity_closure':closure,'all_exact_metadata_identities_dead':True,
 'toy_launcher':report['toy_launcher_identity'],'actual_toy_native':report['toy_native_identity'],
 'sampled_actual_toy_PID':report['exact_sampler_arguments'][3],
 'toy_ready':report['toy_ready'],'profile_summary':report['profile_summary'],
 'input_pins_before_after_exact':report['input_pins_before_after_exact'],
 'sampler_stdout':report['outputs']['sampler.stdout.log'],'sampler_stderr':report['outputs']['sampler.stderr.log'],
 'external_guard_error':None,'required_external_cleanup':[],
 'existing_native40012_not_attached_or_mutated':True,'live_sampling_admitted_by_this_completion':False,
 'closure_sealer_source':pin(__file__)}
p=O/'actual_metadata_completion.json'
with p.open('x',encoding='utf-8',newline='\n')as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps({'completion':pin(p),'external_result':out['external_result'],'profile':out['profile'],'identities':closure,'sampled_PID':out['sampled_actual_toy_PID']}))
