"""Independent read-only AST/delta review; never imports numerical source."""
import ast,difflib,hashlib,importlib,json,sys
from pathlib import Path
L=Path(r'E:\gpu_training_scratchpad\analysis\Turkish_labels_BPE_circumflex_v1')
sys.path.insert(0,str(L));m=importlib.import_module('audit_and_fit_append_only_v5')
N=Path(r'E:\gpu_training_scratchpad\analysis\BPE9223_training_launch_v1')
O=N/'independent_DEV9223_source_review_v1';O.mkdir(exist_ok=True);m.AREA=O
P=m.PROJECT/'outputs/parakeet_turkish'
PAIRS=[('tdt_joint_canonical_2d_fp32_v2.py','tdt_joint_canonical_2d_fp32_9223_v1.py'),
 ('asr18_optimizer_free_DEV_structure_v2.py','asr18_optimizer_free_DEV_structure_9223_v1.py'),
 ('asr18_full_run_serial_DEV_v9.py','asr18_full_run_serial_DEV_9223_v1.py'),
 ('asr18_full_run_DEV_runtime_v4.py','asr18_full_run_DEV_runtime_9223_v1.py')]

def main():
 g=m.Guard();g.phase='read_only_four_DEV_sources_AST_and_exact_delta_review'
 findings=[];sources=[]
 for oldname,newname in PAIRS:
  old=(P/oldname).read_text();new=(P/newname).read_text();a=ast.parse(old);b=ast.parse(new)
  functions=lambda tree:{x.name:ast.dump(x,include_attributes=False) for x in tree.body if isinstance(x,(ast.FunctionDef,ast.ClassDef))}
  fa,fb=functions(a),functions(b)
  delta=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=oldname,tofile=newname))
  (O/(newname+'.read_delta.patch')).write_text(delta,encoding='utf8')
  calls=[]
  for node in ast.walk(b):
   if isinstance(node,ast.Call):
    target=ast.unparse(node.func)
    if any(x in target.lower() for x in ('adam','optimizer','sampler')):calls.append(target)
  findings.append({'old':m.record(P/oldname),'new':m.record(P/newname),
   'functions_added':sorted(set(fb)-set(fa)),'functions_removed':sorted(set(fa)-set(fb)),
   'functions_changed':sorted(k for k in fa.keys()&fb.keys() if fa[k]!=fb[k]),
   'potential_constructor_calls_by_name':calls,'exact_delta':m.record(O/(newname+'.read_delta.patch'))})
  sources.append(m.record(P/newname))
 # Canonical numerical code is an exact vocabulary-width substitution, including
 # validation and allocation sizes. No operation/reduction/order edits.
 old=(P/PAIRS[0][0]).read_text();new=(P/PAIRS[0][1]).read_text()
 assert new.replace('9223','8711').replace('9228','8716')==old
 # The other two stored ROOT patches must be exact full physical text diffs.
 for i in (2,3):
  name=PAIRS[i][1];published=N/(name.replace('.py','')+'_source_delta.patch')
  generated=O/(name+'.read_delta.patch')
  assert published.read_text()==generated.read_text(),name
 print(json.dumps({'exact_numeric_width_only':True,'findings':findings},ensure_ascii=False),flush=True)
 m.write('exact_AST_delta_inventory.json',{'at_utc':m.now(),'exact_canonical_joint_math_width_substitution_only':True,
  'all_published_runtime_and_serial_patches_equal_actual_source_diff':True,'findings':findings,
  'numerical_model_audio_GPU_runtime_executed':False})
 g.publish(passed=True)
if __name__=='__main__':main()
