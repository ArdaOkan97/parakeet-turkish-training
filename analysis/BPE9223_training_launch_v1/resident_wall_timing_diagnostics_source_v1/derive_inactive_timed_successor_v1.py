"""Mechanical inactive wall-timer insertion, with exact executable AST erasure proof."""
import ast,copy,difflib,hashlib,json
from pathlib import Path
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish';R=P/'work/research';N=Path(__file__).parent
sites=[]
def parse(s):return ast.parse(s).body
def code(s):return ast.unparse(s)
def wrap(label,body):
 if not isinstance(body,list):body=[body]
 sites.append({'channel':label,'original_operations':[code(x)for x in body]})
 return ast.With(items=[ast.withitem(context_expr=ast.Call(func=ast.Name(id='_wall_time',ctx=ast.Load()),args=[ast.Name(id='wall_timings',ctx=ast.Load()),ast.Constant(label)],keywords=[]),optional_vars=None)],body=body,type_comment=None)
def idx(body,prefix):return next(i for i,x in enumerate(body)if code(x).startswith(prefix))
def optional(fn):fn.args.kwonlyargs.append(ast.arg(arg='wall_timings'));fn.args.kw_defaults.append(ast.Constant(None))
def alias_import(include_class=False):
 return ast.ImportFrom(module='asr18_resident_wall_timings_v1',names=([ast.alias(name='WallTimings')]if include_class else[])+[ast.alias(name='wall_time',asname='_wall_time'),ast.alias(name='wall_call',asname='_wall_call')],level=0)
def pin(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def save(old,new,tree,delta):
 assert not new.exists();ast.fix_missing_locations(tree);text=ast.unparse(tree)+'\n';ast.parse(text);new.write_text(text,encoding='utf-8');d=N/delta;d.write_text(''.join(difflib.unified_diff(old.read_text().splitlines(True),text.splitlines(True),fromfile=str(old),tofile=str(new))),encoding='utf-8');return pin(new),pin(d)
# All numerical native operations remain the original AST, inside wall contexts.
old_native=S/'asr18_BPE9223_grouped_RAM_native_v1.py';native=ast.parse(old_native.read_text());native.body.insert(1,alias_import());fn=next(x for x in native.body if isinstance(x,ast.FunctionDef)and x.name=='native_update');optional(fn);ti=next(i for i,x in enumerate(fn.body)if isinstance(x,ast.Try));fn.body=[wrap('native_ingress_routes_and_initial_state',fn.body[:ti]),*fn.body[ti:]];tr=fn.body[-1];group=tr.body[0];assert isinstance(group,ast.For)
g=group.body;ac=next(i for i,x in enumerate(g)if isinstance(x,ast.With));autocast=g[ac];ab=autocast.body
assert isinstance(ab[0],ast.With) and code(ab[1]).startswith('require(')
fi=next(i for i,x in enumerate(ab)if isinstance(x,ast.For));autocast.body=[wrap('native_encoder_and_stem',ab[0]),wrap('native_encoder_mask_validation',ab[1]),*ab[2:fi],wrap('native_row_objectives_and_scalar_reads',ab[fi]),wrap('native_loss_stack',ab[fi+1:])]
back=idx(g,'objective.backward(');rng=idx(g,'numpy_rng =');objects=idx(g,'objects =');after=idx(g,'after =');boundary=idx(g,'boundary =')
assert code(g[back+1]).startswith('del ') and code(g[objects+1])=='torch.cuda.synchronize()' and code(g[objects+2])=='torch.cuda.empty_cache()'
group.body=[wrap('native_group_guard_stage',g[:ac-1]),wrap('native_collation_transfer',g[ac-1]),autocast,
 wrap('native_backward_enqueue',g[back]),wrap('native_group_existing_sync_and_before_snapshot',g[back+1:rng]),
 wrap('native_group_RNG_versions_capture',g[rng:objects]),wrap('native_group_gc',g[objects]),
 wrap('native_group_existing_gc_sync',g[objects+1]),wrap('native_group_empty_cache',g[objects+2]),
 wrap('native_group_after_snapshot',g[after]),wrap('native_group_invariants',g[after+1:boundary]),
 wrap('native_group_receipt_stage',g[boundary:])]
b=tr.body;opt=idx(b,'norm =');assert len(b[opt:])==8
tr.body=[group,wrap('native_payload_validation',b[1:opt]),wrap('native_optimizer_step',b[opt]),
 wrap('native_optimizer_zero_grad',b[opt+1]),wrap('native_final_gc',b[opt+2]),
 wrap('native_final_existing_sync',b[opt+3]),wrap('native_final_empty_cache',b[opt+4]),
 wrap('native_final_invariants',b[opt+5:opt+7]),wrap('native_return_receipt',b[opt+7])]
native_path=S/'asr18_BPE9223_grouped_RAM_native_v2.py';native_pin,native_diff=save(old_native,native_path,native,'RAM_native1_to2_wall_timers.diff.txt')
# Driver instrumentation deliberately leaves the journal implementation untouched.
old_driver=R/'run_ASR18_BPE9223_resident_RAM_training_v12.py';driver=ast.parse(old_driver.read_text());driver.body.insert(1,alias_import(True));commit=next(x for x in driver.body if isinstance(x,ast.FunctionDef)and x.name=='commit_RAM_transaction');optional(commit);ci=next(i for i,x in enumerate(commit.body)if isinstance(x,ast.Try));commit.body=[wrap('transaction_validation',commit.body[:ci]),*commit.body[ci:]];ct=commit.body[-1];ji=idx(ct.body,'journal.append(');ct.body=[wrap('transaction_state_and_native_receipt',ct.body[:ji]),wrap('transaction_journal_append_total',ct.body[ji]),*ct.body[ji+1:]]
main=next(x for x in driver.body if isinstance(x,ast.FunctionDef)and x.name=='main')
class Integrate(ast.NodeTransformer):
 def visit_ImportFrom(self,node):
  if node.module=='asr18_BPE9223_grouped_RAM_native_v1':node.module='asr18_BPE9223_grouped_RAM_native_v2'
  return node
 def visit_Constant(self,node):
  if node.value=='asr18_BPE9223_grouped_RAM_native_v1.py':node.value='asr18_BPE9223_grouped_RAM_native_v2.py'
  return node
 def visit_Assign(self,node):
  node=self.generic_visit(node)
  if len(node.targets)==1 and isinstance(node.targets[0],ast.Name)and node.targets[0].id=='window_audio_before':
   return [node,*parse('wall_timings = WallTimings()')]
  return node
 def visit_For(self,node):
  node=self.generic_visit(node)
  if isinstance(node.target,ast.Tuple)and [code(x)for x in node.target.elts]==['index','entry']:
   body=node.body;packet=idx(body,'packet =');start=idx(body,'start =')
   call=body[packet].value;assert isinstance(call,ast.Call)and code(call.func)=='stream.next_update'
   body[packet].value=ast.Call(func=ast.Name(id='_wall_call',ctx=ast.Load()),args=[ast.Name(id='wall_timings',ctx=ast.Load()),ast.Constant('packet_wait'),call.func,*call.args],keywords=call.keywords)
   node.body=[*body[:packet+1],wrap('packet_metadata_and_transaction_validation',body[packet+1:start]),*body[start:]]
   tr=next(x for x in node.body if isinstance(x,ast.Try));tb=tr.body
   assert len(tb)==5 and code(tb[0]).startswith('native =')and code(tb[3]).startswith('state =')
   tb[0].value.keywords.append(ast.keyword(arg='wall_timings',value=ast.Name(id='wall_timings',ctx=ast.Load())))
   tb[3].value.keywords.append(ast.keyword(arg='wall_timings',value=ast.Name(id='wall_timings',ctx=ast.Load())))
   tr.body=[wrap('native_update_total',tb[0]),wrap('after_native_augmentation_validation',tb[1]),tb[2],wrap('transaction_commit_total',tb[3]),wrap('after_commit_sampler_validation',tb[4])]
  return node
 def visit_Expr(self,node):
  node=self.generic_visit(node)
  if isinstance(node.value,ast.Call)and code(node.value.func)=='metadata_timing.update'and any(x.arg=='window_pipeline_seconds'for x in node.value.keywords):
   node.value.keywords.append(ast.keyword(arg='diagnostic_wall_timings',value=ast.Call(func=ast.Attribute(value=ast.Name(id='wall_timings',ctx=ast.Load()),attr='snapshot',ctx=ast.Load()),args=[],keywords=[])))
  return node
Integrate().visit(main)
# New entry is explicitly inactive until a genuine normal0/dead/all-live-saved predecessor.
policy_fn=parse('''def _checked_wall_timing_policy(base,a):
    require(a.prior_process is not None and all(getattr(a,k,None) is None for k in ('abnormal_resume','DEV_frontend_resume','source_transition_resume','original_interval_resume')),
        'Diagnostic successor requires genuine normal0; never an intentional or failure transition')
    policy=base['resident_wall_timing_policy']
    require(policy=={'kind':'ASR18_existing_operation_wall_diagnostics_v1','helper_source':record(S/'asr18_resident_wall_timings_v1.py'),
        'RAM_native_source':record(S/'asr18_BPE9223_grouped_RAM_native_v2.py'),'actual_CPU_proof':record(ROOT/'analysis/BPE9223_training_launch_v1/resident_wall_timing_diagnostics_source_v1/actual_CPU_instrumentation_proof_v1.json'),
        'per_window_fixed_channels':33,'maximum_window_record_bytes':65536,'GPU_barriers_added':0,'resource_limits_changed':False,
        'activation_requires_genuine_normal0':True,'nested_wall_timers_are_not_GPU_kernel_timings':True},'Exact source-bound inactive wall diagnostic policy')
    proof=read_json(policy['actual_CPU_proof']['path'])
    require(proof['passed'] is True and proof['native_erased_AST_equals_original'] is True
        and proof['driver_erased_AST_equals_original'] is True and proof['helper_source']==policy['helper_source']
        and proof['RAM_native_source']==policy['RAM_native_source'],'Actual bounded CPU instrumentation equivalence proof')
''')[0]
driver.body.insert(driver.body.index(main),policy_fn)
bi=next(i for i,x in enumerate(main.body)if code(x).startswith('base = read_json(a.configuration)'))
main.body.insert(bi+1,parse('_checked_wall_timing_policy(base,a)')[0])
driver_path=R/'run_ASR18_BPE9223_resident_RAM_training_v13.py';driver_pin,driver_diff=save(old_driver,driver_path,driver,'driver12_to13_wall_timers.diff.txt')
# Erase ONLY declared diagnostic wrappers/statements/keywords and the explicit new admission gate.
class Erase(ast.NodeTransformer):
 def visit_ImportFrom(self,node):
  if node.module=='asr18_resident_wall_timings_v1':return None
  if node.module=='asr18_BPE9223_grouped_RAM_native_v2':node.module='asr18_BPE9223_grouped_RAM_native_v1'
  return node
 def visit_Constant(self,node):
  if node.value=='asr18_BPE9223_grouped_RAM_native_v2.py':node.value='asr18_BPE9223_grouped_RAM_native_v1.py'
  return node
 def visit_FunctionDef(self,node):
  if node.name=='_checked_wall_timing_policy':return None
  if any(x.arg=='wall_timings'for x in node.args.kwonlyargs):
   i=next(i for i,x in enumerate(node.args.kwonlyargs)if x.arg=='wall_timings');node.args.kwonlyargs.pop(i);node.args.kw_defaults.pop(i)
  return self.generic_visit(node)
 def visit_With(self,node):
  if len(node.items)==1 and isinstance(node.items[0].context_expr,ast.Call)and code(node.items[0].context_expr.func)=='_wall_time':
   result=[]
   for value in node.body:
    changed=self.visit(value);result.extend(changed if isinstance(changed,list)else[changed])
   return result
  return self.generic_visit(node)
 def visit_Assign(self,node):
  if any(isinstance(x,ast.Name)and x.id=='wall_timings'for x in node.targets):return None
  return self.generic_visit(node)
 def visit_Expr(self,node):
  if isinstance(node.value,ast.Call)and code(node.value.func)=='_checked_wall_timing_policy':return None
  return self.generic_visit(node)
 def visit_Call(self,node):
  if isinstance(node.func,ast.Name)and node.func.id=='_wall_call':return self.visit(ast.Call(func=node.args[2],args=node.args[3:],keywords=node.keywords))
  node.keywords=[x for x in node.keywords if x.arg not in ('wall_timings','diagnostic_wall_timings')]
  return self.generic_visit(node)
results={}
for label,old,new in [('native',old_native,native_path),('driver',old_driver,driver_path)]:
 baseline=ast.dump(ast.parse(old.read_text()),include_attributes=False);erased=Erase().visit(ast.parse(new.read_text()));ast.fix_missing_locations(erased);actual=ast.dump(erased,include_attributes=False)
 assert actual==baseline,(label,'Original executable AST differs after removing declared diagnostics')
 results[label+'_erased_AST_equals_original']=True;results[label+'_original_AST_sha256']=hashlib.sha256(baseline.encode()).hexdigest()
# Wrapper successor changes only explicit source roles and strict new normal entry policy.
wold=R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v12.py';w=wold.read_text().replace('run_ASR18_BPE9223_resident_RAM_training_v12','run_ASR18_BPE9223_resident_RAM_training_v13').replace('asr18_BPE9223_grouped_RAM_native_v1.py','asr18_BPE9223_grouped_RAM_native_v2.py')
needle="    configuration=read_json(a.configuration);policy=configuration['private_scalar_prefetch_policy']"
assert w.count(needle)==1
w=w.replace(needle,needle+"\n    require(a.prior_process is not None and review['prior_process']==record(a.prior_process)\n        and review['resident_wall_timing_policy']==configuration['resident_wall_timing_policy'],\n        'Exact future normal0 receipt/checkpoint/config-bound diagnostic admission; no current activation')")
wpath=R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v13.py';assert not wpath.exists();ast.parse(w);wpath.write_text(w,encoding='utf-8');wd=N/'wrapper12_to13_roles_and_normal_entry.diff.txt';wd.write_text(''.join(difflib.unified_diff(wold.read_text().splitlines(True),w.splitlines(True),fromfile=str(wold),tofile=str(wpath))),encoding='utf-8')
manifest={'kind':'ASR18_inactive_existing_operation_wall_diagnostics_source_v1','helper_source':pin(S/'asr18_resident_wall_timings_v1.py'),'RAM_native_source':native_pin,'driver_source':driver_pin,'wrapper_source':pin(wpath),'native_delta':native_diff,'driver_delta':driver_diff,'wrapper_delta':pin(wd),'original_RAM_native':pin(old_native),'original_driver':pin(old_driver),'original_wrapper':pin(wold),'AST_equivalence':results,'timing_sites':sites,'journal_implementation_untouched':True,'no_added_CUDA_synchronize_or_empty_cache_calls':True,'activation_authorized':False,'current_healthy_child_modified':False}
p=N/'inactive_source_handoff_v1.json';assert not p.exists();p.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8');print(json.dumps({'handoff':pin(p),'sources':{k:manifest[k]for k in ('helper_source','RAM_native_source','driver_source','wrapper_source')},'AST_equivalence':results}),flush=True)