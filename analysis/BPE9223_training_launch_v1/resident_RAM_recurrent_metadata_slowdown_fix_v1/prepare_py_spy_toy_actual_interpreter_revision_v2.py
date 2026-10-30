"""Inactive correction only: Windows venv launcher vs actual ready Python PID."""
import ast, copy, hashlib, json, os
from pathlib import Path
F=Path(__file__).parent.resolve()
def pin(p):
    p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def write(p,v):
    with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
assert os.environ['TORCH_EXTENSIONS_DIR'].lower().startswith('e:')
old=F/'py_spy_nonblocking_bounded_helper_v1.py';new=F/'py_spy_nonblocking_bounded_helper_v2.py'
edits=[
 ("            report['toy_identity']=remember(psutil.Process(toy.pid))\n",
  "            report['toy_identity']=remember(psutil.Process(toy.pid))\n            report['toy_launcher_identity']=report['toy_identity']\n"),
 ("            ready=read(out/'toy_ready.json');require(ready['pid']==toy.pid and Path(ready['interpreter']).resolve()==PYTHON.resolve(),'Genuine same-interpreter ready')\n            report['toy_ready']=ready;pid=toy.pid;duration=8\n",
  "            ready=read(out/'toy_ready.json')\n            require(type(ready['pid'])is int and ready['pid']>0 and ready['pid']!=TARGET['pid']\n                and Path(ready['interpreter']).resolve()==PYTHON.resolve()\n                and Path(ready['source']).resolve()==TOY.resolve() and ready['standard_library_only']is True,\n                'Genuine trusted same-interpreter/source ready, never the admitted native')\n            require(exact_alive(report['toy_launcher_identity']) and toy.poll()is None,'Original exact toy launcher still alive')\n            actual=psutil.Process(ready['pid']);actual_who=identity(actual)\n            launcher=psutil.Process(report['toy_launcher_identity']['pid'])\n            descendants=[identity(p) for p in launcher.children(recursive=True)]\n            require(actual_who==report['toy_launcher_identity'] or actual_who in descendants,\n                'Actual Python is original exact launcher or its exact live owned descendant')\n            report['toy_native_identity']=remember(actual)\n            report['toy_ready']=ready;pid=actual_who['pid'];duration=8\n"),
 ("                if toy is not None:require(toy.poll() is None,'Owned toy remains alive while sampled')\n",
  "                if toy is not None:require(toy.poll() is None and exact_alive(report['toy_launcher_identity'])\n                    and exact_alive(report['toy_native_identity']),'Both owned launcher and actual toy Python remain alive while sampled')\n"),
 ("            complete=read(out/'toy_completion.json');require(report['toy_returncode']==0 and complete['passed'] is True,'Toy clean thread/module closure')\n",
  "            complete=read(out/'toy_completion.json');require(report['toy_returncode']==0 and complete['passed'] is True\n                and complete['pid']==report['toy_native_identity']['pid'],'Actual toy Python completion plus original Popen wait0')\n")]
raw=old.read_bytes();text=raw.decode('utf-8')
for a,b in edits:
    assert text.count(a)==1
    text=text.replace(a,b,1)
assert not new.exists();new.write_bytes(text.encode('utf-8'));ast.parse(text)
inverse=new.read_bytes().decode('utf-8')
for a,b in reversed(edits):
    assert inverse.count(b)==1;inverse=inverse.replace(b,a,1)
assert inverse.encode('utf-8')==raw
ledger={'kind':'ASR18_py_spy_v2_venv_actual_Python_PID_exact_edit_ledger_v1','old':pin(old),'new':pin(new),
 'edits':[{'old_literal':a,'new_literal':b,'expected_count':1} for a,b in edits],
 'newline_normalization':'NONE: original UTF8 LF bytes preserved','full_raw_inverse_exact':True,'other_guards_call_order_policy_or_toy_source_changes':False,'launcher_edits':[]}
mapping={str(old.resolve()):new}
for stem in ('launch_py_spy_same_interpreter_toy','launch_py_spy_existing_native40012'):
    a=F/(stem+'_v1.py');b=F/(stem+'_v2.py');oldraw=a.read_bytes()
    x=b'from py_spy_nonblocking_bounded_helper_v1 import run';y=b'from py_spy_nonblocking_bounded_helper_v2 import run'
    assert oldraw.count(x)==1 and not b.exists();b.write_bytes(oldraw.replace(x,y,1));ast.parse(b.read_text(encoding='utf-8'))
    assert b.read_bytes().replace(y,x,1)==oldraw
    ledger['launcher_edits'].append({'old':pin(a),'new':pin(b),'edits':[{'old_literal':x.decode(),'new_literal':y.decode(),'expected_count':1}],'full_raw_inverse_exact':True})
    mapping[str(a.resolve())]=b
ledger_path=F/'py_spy_toy_actual_interpreter_revision_v2_exact_ledger.json';write(ledger_path,ledger)
for stem in ('py_spy_same_interpreter_toy_source_handoff','py_spy_existing_native40012_source_handoff'):
    a=F/(stem+'_v1.json');b=F/(stem+'_v2.json');h=json.loads(a.read_text(encoding='utf-8'))
    h['kind']=h['kind'].removesuffix('_v1')+'_v2'
    h['helper_source']=pin(new)
    h['launcher_source']=pin(mapping[h['launcher_source']['path']])
    h['input_pins']=[pin(mapping[v['path']]) if v['path'] in mapping else v for v in h['input_pins']]
    h['input_pins'].append(pin(ledger_path))
    req=h['required_ROOT_scope_fields'];req['helper_source']=h['helper_source'];req['launcher_source']=h['launcher_source'];req['input_pins']=h['input_pins']
    if 'required_completed_ROOT_toy_gate_fields'in h:h['required_completed_ROOT_toy_gate_fields']['helper_source']=pin(new)
    h['unexecuted_v1_preserved']=pin(a)
    h['revision_only']='Validate actual ready Python PID as original Popen launcher or its exact live descendant, retain both exact IDs, profile actual Python, check both alive, and bind actual completion PID. No toy/native/math/guards/limits changes.'
    write(b,h)
print(json.dumps({'helper':pin(new),'ledger':pin(ledger_path),
 'toy_handoff':pin(F/'py_spy_same_interpreter_toy_source_handoff_v2.json'),
 'live_handoff':pin(F/'py_spy_existing_native40012_source_handoff_v2.json'),
 'toy_launcher':pin(F/'launch_py_spy_same_interpreter_toy_v2.py'),
 'live_launcher':pin(F/'launch_py_spy_existing_native40012_v2.py'),'actual_control_execution':False}))
