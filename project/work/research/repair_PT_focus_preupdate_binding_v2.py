from pathlib import Path
import ast,datetime,hashlib,json,shutil
r=Path(__file__).resolve().parents[2];h=r/'outputs/parakeet_turkish'
def rec(p):return {'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
d=r/'work/closed-sources/PT_focus01_preupdate_binding_failure_v1';d.mkdir()
names=['train_lora_batched_tsc_uniform_majority_ptfocus_v1.py','run_TSC_uniform_majority_ptfocus_v1.py']
for name in names:shutil.copy2(h/name,d/name)
loop=h/'runs/adaptive_20261012_TR_uniform_majority_PT_focus_01_loop_01'
report={'schema':1,'kind':'owned_PT_focus_preupdate_code_failure_closure_v1','actual_owned_session':92124,'actual_exit_code':1,'actual_tool_chunk':'77b922',
    'observed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'known_child_PIDs_absent':[26292,32252],
    'failure':'KeyError lora_boundary_step','source_step8000_preserved':True,'optimizer_updates':0,'source350_started':False,
    'runtime_completion':rec(loop/'adaptive_20261012_TR_uniform_majority_PT_focus_01_native_backend.completion.json'),
    'controller_status':rec(loop/'loop_status.json'),'frozen_failed_sources':{n:rec(d/n) for n in names},
    'repair_scope':'newv2worker reads checked parentconfig.source_step7750; no scientific/state changes'}
p=h/'PT_focus01_actual_preupdate_binding_failure_closure_v1.json'
with p.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
w=(h/names[0]).read_text(encoding='utf-8')
assert w.count('context["lora_boundary_step"]')==1
w=w.replace('context["lora_boundary_step"]','context["config"]["source_step"]')
w=w.replace('run_TSC_uniform_majority_ptfocus_v1.py','run_TSC_uniform_majority_ptfocus_v2.py')
q=h/'train_lora_batched_tsc_uniform_majority_ptfocus_v2.py';assert not q.exists();q.write_text(w,encoding='utf-8',newline='\n')
c=(h/names[1]).read_text(encoding='utf-8').replace(names[0],q.name)
c=c.replace("WORKER_SHA='61f30fb8e75be479abf630b2919a3f6be3f7153655927b2939127d09b2ab725d'","WORKER_SHA='"+rec(q)['sha256']+"'")
z=h/'run_TSC_uniform_majority_ptfocus_v2.py';assert not z.exists();z.write_text(c,encoding='utf-8',newline='\n')
ast.parse(w);ast.parse(c)
lease=r/'work/gpu_training_owner.json';v=json.loads(lease.read_text())
v.update(status='actual_preupdate_binding_failure_closed_repaired_v2',actual_controller_session_id=92124,actual_exit_code=1,
    actual_closed_tool_chunk='77b922',actual_GPU_launched=False,updated_at_utc=report['observed_at_utc'],actual_failure_receipt=rec(p))
t=lease.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2),encoding='utf-8');t.replace(lease)
print(json.dumps({'worker':rec(q),'controller':rec(z),'failure':rec(p)}))
