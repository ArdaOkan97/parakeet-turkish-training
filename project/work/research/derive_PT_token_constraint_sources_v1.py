"""Minimal new-file derivative; frozen parent source bytes remain unchanged."""
from pathlib import Path
import ast,hashlib
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'outputs/parakeet_turkish'
oldworker=HERE/'train_lora_batched_tsc_uniform_constraint_v2.py';oldcontroller=HERE/'run_TSC_uniform_constraint_v2.py'
def replace_once(s,a,b):
    assert s.count(a)==1,(a,s.count(a));return s.replace(a,b)
s=oldworker.read_text(encoding='utf-8')
s=s.replace('train_lora_batched_tsc_uniform_constraint_v2.py','train_lora_batched_tsc_uniform_pttoken_v1.py').replace('run_TSC_uniform_constraint_v2.py','run_TSC_uniform_pttoken_v1.py').replace('tsc_uniform_retention_constraint_transition_v1','tsc_uniform_pttoken_constraint_transition_v1')
s=s.replace('constraint_declaration','token_constraint_declaration')
s=replace_once(s,'TSC_uniform_feature_retention_constraint_full_state_v1','TSC_uniform_PT_token_retention_constraint_full_state_v1')
s=replace_once(s,'def pt_focused_batch_objective(model,teacher,batch,*,replay_kl,duration_kl,feature_distillation,**kwargs):','def pt_focused_batch_objective(model,teacher,batch,*,replay_kl,duration_kl,feature_distillation,portuguese_token_kl_multiplier,**kwargs):')
s=replace_once(s,'    total=(result.per_clip["tdt"]+replay_kl*scale*result.per_clip["token_kl"]','    token_scale=scale*torch.tensor([portuguese_token_kl_multiplier if row["language"]=="pt" else 1. for row in batch.rows],dtype=torch.float32,device=scale.device)\n    total=(result.per_clip["tdt"]+replay_kl*token_scale*result.per_clip["token_kl"]')
s=replace_once(s,'result = pt_focused_batch_objective(model, teacher, batch, old_size=old_size, replay_kl=args.replay_kl,','result = pt_focused_batch_objective(model, teacher, batch, old_size=old_size, replay_kl=args.replay_kl, portuguese_token_kl_multiplier=args.portuguese_token_kl_multiplier,')
s=replace_once(s,'        corrected_boundary = (selected == source_step and source_step == 8250\n            and current.get("retention_numeric_correction", {}).get("source_step") == source_step\n            and development_gate(context["source_summary"], context["combined_baseline"])["retention_gate_passed"])','        corrected_boundary = (selected == 8250\n            and current.get("retention_numeric_correction", {}).get("source_step") == 8250\n            and context.get("source_best_step") == 8250\n            and development_gate(context["source_best_summary"], context["combined_baseline"])["retention_gate_passed"])')
s=replace_once(s,'        "retention_constraint_transition":transition_identity(args,context,sampler_transition),','        "retention_constraint_transition":context["parent_identity"]["retention_constraint_transition"],\n        "PT_token_retention_constraint":transition_identity(args,context,sampler_transition),')
s=replace_once(s,'        "retention_constraint_transition":config_identity["retention_constraint_transition"],','        "retention_constraint_transition":config_identity["retention_constraint_transition"],\n        "PT_token_retention_constraint":config_identity["PT_token_retention_constraint"],')
s=replace_once(s,'"early_stop_patience","max_retention_failures","target_turkish_wer")','"early_stop_patience","max_retention_failures","target_turkish_wer","portuguese_token_kl_multiplier")')
s=replace_once(s,'        corrected_gate=development_gate(context["source_summary"],baseline)\n        if (source_step!=8250 or start_step!=8250 or retention_failures!=0 or selected_step!=8250\n                or best!=corrected_gate["turkish_wer"] or not corrected_gate["retention_gate_passed"] or not equality["passed"]):\n            raise ValueError("Coherent qualified BEST8250 accounting must carry without reset")','        source_gate=development_gate(context["source_summary"],baseline)\n        inherited_gate=development_gate(context["source_best_summary"],baseline)\n        if (source_step!=8750 or start_step!=8750 or retention_failures!=2 or selected_step!=8250\n                or best!=inherited_gate["turkish_wer"] or not inherited_gate["retention_gate_passed"]\n                or source_gate["retention_gate_passed"] or not all(source_gate["retention_language_passed"][lang] for lang in ("en","es","it","fr","de"))\n                or not equality["passed"]):\n            raise ValueError("Reviewed coherent latest8750 must carry genuine counter2 and separate qualified BEST8250 without reset")')
s=replace_once(s,'"preservation_coefficients_by_language":{"pt":[args.replay_kl,args.duration_kl,args.feature_distillation]','"preservation_coefficients_by_language":{"pt":[args.replay_kl*args.portuguese_token_kl_multiplier,args.duration_kl,args.feature_distillation]')
s=replace_once(s,'                  "PT_replay_clips_per_update":1,"old_five_replay_clips_per_update":1,','                  "Portuguese_token_KL_multiplier":args.portuguese_token_kl_multiplier,"PT_token_constraint_updates":step-source_step,\n                  "PT_replay_clips_per_update":1,"old_five_replay_clips_per_update":1,')
s=replace_once(s,'    p.add_argument("--feature-distillation",type=float,default=20.)','    p.add_argument("--feature-distillation",type=float,default=20.)\n    p.add_argument("--portuguese-token-kl-multiplier",type=float,default=5.)')
s=replace_once(s,'or args.feature_distillation!=20 or args.schedule!="constant":','or args.feature_distillation!=20 or args.portuguese_token_kl_multiplier!=5 or args.schedule!="constant":')
s=s.replace('KD40-duration2-feature20 constant recipe required','baseKD40/PTtoken200-duration2-feature20 constant recipe required')
newworker=HERE/'train_lora_batched_tsc_uniform_pttoken_v1.py'
with newworker.open('x',encoding='utf-8') as f:f.write(s)
tree=ast.parse(s);run=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='run')
calls=[n for n in ast.walk(run) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and ast.unparse(n.func)=='torch.cuda.synchronize']
assert len(calls)==1;sync=calls[0].lineno
c=oldcontroller.read_text(encoding='utf-8').replace('train_lora_batched_tsc_uniform_constraint_v2.py',newworker.name).replace('tsc_uniform_retention_constraint_transition_v1','tsc_uniform_pttoken_constraint_transition_v1')
c=replace_once(c,"WORKER_SHA='f6433bd9cf34a1560f0a0093630659ef365f151e8bad68f6bfc4b9357b7731fe'","WORKER_SHA='"+hashlib.sha256(newworker.read_bytes()).hexdigest()+"'")
c=replace_once(c,"TRANSITION_SHA='ccdef7862d6fec68d28265199514ebdaa4772352be47cc4cfec0230e8370bfba'","TRANSITION_SHA='PENDING'")
c=replace_once(c,'EXPECTED_SYNC_LINE=736',f'EXPECTED_SYNC_LINE={sync}')
c=replace_once(c,"    values['max_wall_seconds']=float(option(argv,'--max-wall-seconds'))","    values['portuguese_token_kl_multiplier']=float(option(argv,'--portuguese-token-kl-multiplier'))\n    values['max_wall_seconds']=float(option(argv,'--max-wall-seconds'))")
c=replace_once(c,"and float(option(argv,'--duration-kl'))==2 and float(option(argv,'--feature-distillation'))==20 and option(argv,'--eval-every-updates')=='250'","and float(option(argv,'--duration-kl'))==2 and float(option(argv,'--feature-distillation'))==20\n        and float(option(argv,'--portuguese-token-kl-multiplier'))==5 and option(argv,'--eval-every-updates')=='250'")
c=replace_once(c,"('--replay-kl',40.),('--duration-kl',2.),('--feature-distillation',20.),","('--replay-kl',40.),('--duration-kl',2.),('--feature-distillation',20.),('--portuguese-token-kl-multiplier',5.),")
c=replace_once(c,"'declared_retention_coefficients':{'replay_kl':40.,'duration_kl':2.,'feature_distillation':20.},","'declared_retention_coefficients':{'base_replay_kl':40.,'Portuguese_token_kl':200.,'old_five_token_kl':40*5/3,'duration_kl':2.,'feature_distillation':20.},")
c=replace_once(c,"'expected_RP_mean_density':{'replay_kl':(40+40*5/3)/16,","'expected_RP_mean_density':{'replay_kl':(200+40*5/3)/16,")
newcontroller=HERE/'run_TSC_uniform_pttoken_v1.py'
with newcontroller.open('x',encoding='utf-8') as f:f.write(c)
compile(s,str(newworker),'exec');compile(c,str(newcontroller),'exec')
print({'worker_sha256':hashlib.sha256(newworker.read_bytes()).hexdigest(),'sync':sync,'controller_pending_helper':str(newcontroller)})
