import ast,copy,hashlib,json,os
from pathlib import Path
F=Path(__file__).parent
def pin(p):
    p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def write(p,v):
    with p.open('x',encoding='utf-8',newline='\n')as f:json.dump(v,f,indent=2);f.write('\n')
assert os.environ['TORCH_EXTENSIONS_DIR'].lower().startswith('e:')
launcher_edits=[
 ("    began=time.monotonic();popen=None;observed={};cleanup=[];guard_error=None;peak=0;initial=None;last=None\n",
  "    began=time.monotonic();popen=None;popen_identity=None;observed={};cleanup=[];guard_error=None;peak=0;initial=None;last=None\n"),
 ("            remember(psutil.Process(popen.pid))\n",
  "            popen_identity=remember(psutil.Process(popen.pid))\n            result['original_Popen_identity']=popen_identity\n"),
 ("                root=psutil.Process(popen.pid);owned=[*root.children(recursive=True),root]\n                for proc in owned:\n                    who=remember(proc);proc.kill();cleanup.append(who)\n",
  "                require(popen_identity is not None and same_alive(popen_identity),'Cleanup exact initial Popen identity, never PID alone')\n                require(popen_identity['pid'] not in (40012,31044,26892),'Never native, feature CPU or reader cleanup')\n                root=psutil.Process(popen_identity['pid'])\n                owned=[]\n                for proc in root.children(recursive=True):\n                    who=remember(proc);depth=len(proc.parents());owned.append((depth,who))\n                owned.sort(key=lambda item:item[0],reverse=True)\n                owned=[who for _,who in owned]+[popen_identity]\n                for who in owned:\n                    if not same_alive(who):continue\n                    require(who['pid'] not in (40012,31044,26892),'Protected production identities are never cleanup targets')\n                    require(same_alive(popen_identity),'Original owned launcher remains exact during ancestry check')\n                    proc=psutil.Process(who['pid'])\n                    require(who==popen_identity or any(p.pid==popen_identity['pid']and abs(p.create_time()-popen_identity['create_time'])<.001 for p in proc.parents()),'Fresh exact owned ancestry before cleanup')\n                    require(same_alive(who),'Recheck exact descendant create-time immediately before kill')\n                    proc.kill();cleanup.append(who)\n")]
records=[]
for oldname,newname,edits in [
 ('launch_fresh_signature_collect_observation_CPU_v1.py','launch_fresh_signature_collect_observation_CPU_v2.py',launcher_edits)]:
    a=F/oldname;b=F/newname;old=a.read_bytes();text=old.decode('utf-8')
    for x,y in edits:assert text.count(x)==1;text=text.replace(x,y,1)
    ast.parse(text);assert not b.exists();b.write_bytes(text.encode('utf-8'));inverse=text
    for x,y in reversed(edits):assert inverse.count(y)==1;inverse=inverse.replace(y,x,1)
    assert inverse.encode('utf-8')==old
    records.append({'old':pin(a),'new':pin(b),'edits':[{'old_literal':x,'new_literal':y,'expected_count':1}for x,y in edits],'full_raw_inverse_exact':True})
ledger=F/'fresh_signature_collect_focused_CPU_revision_v2_exact_ledger.json'
write(ledger,{'kind':'ASR18_fresh_signature_CPU_launcher_v2_exact_owned_cleanup_ledger_v1','records':records,'newline_normalization':'NONE','v1_launcher_preserved_unexecuted':True,'fixture_v1_unchanged':pin(F/'verify_fresh_signature_and_collect_observation_CPU_v1.py')})
old_handoff=F/'fresh_signature_collect_focused_CPU_source_handoff_v1.json';h=json.loads(old_handoff.read_text(encoding='utf-8'))
mapping={r['old']['path']:r['new']for r in records}
h['kind']=h['kind'].removesuffix('_v1')+'_v2';h['launcher_source']=records[0]['new']
h['input_pins']=[mapping.get(v['path'],v)for v in h['input_pins']]+[pin(ledger)]
h['required_ROOT_scope_fields'].update(fixture_source=h['fixture_source'],launcher_source=h['launcher_source'],input_pins=h['input_pins'])
h['unexecuted_v1_handoff']=pin(old_handoff);h['revision_ledger']=pin(ledger)
h['cleanup_identity_policy']='Record exact original Popen launcher ctime at spawn. On failure only its exact live descendants, deepest first; recheck exact create-time and live owned ancestry before each kill. Never40012/31044/26892. No PID-only cleanup.'
h['proof']['coverage_limits']='No claim of attempted writes while denywrite handles are held beyond actual executed cases. Reparse creation unsupported is an unsupported capability, not real reparse proof. Fixture v1 remains unchanged; no additional cases.'
p=F/'fresh_signature_collect_focused_CPU_source_handoff_v2.json';write(p,h)
print(json.dumps({'handoff':pin(p),'ledger':pin(ledger),'fixture':h['fixture_source'],'launcher':h['launcher_source'],'input_count':len(h['input_pins']),'control_executed':False}))
