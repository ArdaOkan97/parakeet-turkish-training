"""NEW inactive siblings only. No source factory/state admission or ML import."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/parakeet_turkish'
def sha(text):return hashlib.sha256(text.encode('utf-8')).hexdigest()
def replace(text,before,after,count=1):
    assert text.count(before)==count,(before,text.count(before),count)
    return text.replace(before,after)
def save(name,text):
    p=OUT/name;ast.parse(text)
    if p.exists():
        assert p.read_text(encoding='utf-8')==text,'Refuse differing existing new source'
        return sha(text)
    p.write_text(text,encoding='utf-8',newline='\n');return sha(text)
def fn(tree,name):return next(n for n in ast.parse(tree).body if isinstance(n,ast.FunctionDef) and n.name==name)
UFILE='TR_tdt_population_weighting_v1.py';USHA=hashlib.sha256((OUT/UFILE).read_bytes()).hexdigest()
helper_old='tsc_uniform_head_lr_transition_v1.py';helper_new='tsc_uniform_TR_loss_transition_v1.py'
creator_old='record_TSC_uniform_head_lr_source_v1.py';creator_new='record_TSC_uniform_TR_loss_source_v1.py'
worker_old='train_lora_batched_tsc_uniform_head_lr_v1.py';worker_new='train_lora_batched_tsc_uniform_TR_loss_v1.py'
core_old='run_TSC_uniform_head_lr_v1.py';core_new='run_TSC_uniform_TR_loss_v1.py'
adapter_old='run_TSC_uniform_head_lr_journal_v1.py';adapter_new='run_TSC_uniform_TR_loss_journal_v1.py'

h=(OUT/helper_old).read_text(encoding='utf-8')
h=h.replace('ROOT_owned_closed_TSC_uniform_head_LR_v1','ROOT_owned_closed_TSC_uniform_TR_loss_v1').replace('ROOT_USER_TSC_uniform_head_LR_control_review_v1','ROOT_USER_TSC_uniform_TR_loss_control_review_v1').replace('actual_CLOSED_selected_encoder_policy_head_LR_control','actual_CLOSED_selected_encoder_head_policy_TR_loss_control').replace('allow_matched_head_LR_control','allow_matched_TR_loss_control')
h=replace(h,"from tsc_uniform_majority_transition_v1 import audit_source_optimizer",f"from tsc_uniform_majority_transition_v1 import audit_source_optimizer\nfrom {UFILE[:-3]} import declaration as loss_declaration, POPULATION_MEAN_U, POPULATION_REVIEW_SHA256, MODES")
h=replace(h,"        and value.get('encoder_lr_schedule') in ENCODER_SCHEDULE_MODES and value.get('head_lr') in HEAD_LR_CHOICES,","        and value.get('encoder_lr_schedule') in ENCODER_SCHEDULE_MODES and value.get('head_lr') in HEAD_LR_CHOICES\n        and value.get('tr_tdt_weighting') in MODES and value.get('tr_tdt_population_mean_U')==POPULATION_MEAN_U,")
head_gate="""
    require(review.get('head_policy_selection_completed') is True and review.get('selected_head_lr')==receipt['head_lr'],
        'Completed actual head policy required; preparation cannot select or admit it')
    head_policy=read(checked_record(review['head_policy_choice_review']))
    require(head_policy.get('kind')=='ROOT_completed_head_LR_policy_selection_v1' and head_policy.get('passed') is True
        and head_policy.get('actual_curve_evidence_used') is True and head_policy.get('full_matched_head_cycle_completed') is True
        and head_policy.get('selected_head_lr')==receipt['head_lr']
        and head_policy.get('selected_encoder_lr_schedule')==receipt['encoder_lr_schedule'],
        'Actual completed matched head curve and explicit policy evidence required')
    population=read(checked_record(receipt['population_review']))
    require(receipt['population_review']['sha256']==POPULATION_REVIEW_SHA256 and population['TR_clips']==253798
        and population['whole_pool_mean_U']==POPULATION_MEAN_U, 'Complete pinned population-U evidence required')
"""
h=replace(h,"    change=review['declared_change']",head_gate+"    change=review['declared_change']")
h=replace(h,"change.get('parameter')=='shared_head_lr' and change.get('choices')==list(HEAD_LR_CHOICES)","change.get('parameter')=='TR_TDT_normalization' and change.get('choices')==list(MODES)\n        and change.get('active_mode')==receipt['tr_tdt_weighting'] and change.get('fixed_population_mean_U')==POPULATION_MEAN_U\n        and change.get('replay_TDT_and_preservation_unchanged') is True and change.get('natural_draws_unchanged') is True")
h=replace(h,"'source_recipe':{**recipe,'steps':43238,'persistence_mode':'double_json','head_lr':receipt['head_lr'],","'source_recipe':{**recipe,'steps':43238,'persistence_mode':'double_json','head_lr':receipt['head_lr'],\n            'tr_tdt_weighting':receipt['tr_tdt_weighting'],'tr_tdt_population_mean_U':POPULATION_MEAN_U,")
h=h.replace("'all_other_math_teacher_inputs_and_replay_unchanged':True","'teacher_inputs_and_replay_unchanged':True,'TR_normalization_separately_declared':True").replace("'only_head_group_rate_changes_relative_to_selected_encoder_policy':True","'head_group_rate_held_to_selected_policy':True")
h+="""

def TR_loss_declaration(context,mode):
    require(mode==context['receipt']['tr_tdt_weighting'],'Exact hash-bound normalization mode required')
    return {**loss_declaration(mode),'source_step':25109,'matched_end_step':43238,
        'ROOT_review':context['receipt']['ROOT_review'],'population_review':context['receipt']['population_review'],
        'selected_head_policy':context['receipt']['head_lr'],
        'head_policy_choice_review':context['head_lr_review']['head_policy_choice_review']}
"""
for name in ('audit_parent_state','restore_parent_sampler'):
    assert ast.dump(fn(h,name),include_attributes=False)==ast.dump(fn((OUT/helper_old).read_text(encoding='utf-8'),name),include_attributes=False)
hsha=save(helper_new,h)

r=(OUT/creator_old).read_text(encoding='utf-8').replace(helper_old[:-3],helper_new[:-3]).replace('allow_matched_head_LR_control','allow_matched_TR_loss_control').replace('actual_CLOSED_selected_encoder_policy_head_LR_control','actual_CLOSED_selected_encoder_head_policy_TR_loss_control')
r=replace(r,"from record_TSC_uniform_majority_source_v1 import option",f"from record_TSC_uniform_majority_source_v1 import option\nfrom {UFILE[:-3]} import MODES, POPULATION_MEAN_U, POPULATION_REVIEW_SHA256")
creator_gate="""
    require(review.get('head_policy_selection_completed') is True and review.get('selected_head_lr')==args.head_lr,
        'Actual head policy must be completed before checkpoint byte reads')
    head_policy=read(checked_record(review['head_policy_choice_review']))
    require(head_policy.get('kind')=='ROOT_completed_head_LR_policy_selection_v1' and head_policy.get('passed') is True
        and head_policy.get('actual_curve_evidence_used') is True and head_policy.get('full_matched_head_cycle_completed') is True
        and head_policy.get('selected_head_lr')==args.head_lr and head_policy.get('selected_encoder_lr_schedule')==args.encoder_lr_schedule,
        'Actual completed matched head policy required; inactive source cannot admit a choice')
    population_record=record(HERE/'original8199_TR_loss_length_exposure_CPU_review_v1.json')
    require(population_record['sha256']==POPULATION_REVIEW_SHA256,'Pinned population evidence changed')
"""
r=replace(r,"    closed=read(checked_record(review['actual_closed_source_review']))",creator_gate+"    closed=read(checked_record(review['actual_closed_source_review']))")
r=replace(r,"'source_counter_preserved':23,'encoder_lr_schedule':args.encoder_lr_schedule,'head_lr':args.head_lr,","'source_counter_preserved':23,'encoder_lr_schedule':args.encoder_lr_schedule,'head_lr':args.head_lr,\n        'tr_tdt_weighting':args.tr_tdt_weighting,'tr_tdt_population_mean_U':POPULATION_MEAN_U,'population_review':population_record,")
r=replace(r,"    p.add_argument('--validate-only',action='store_true');return p","    p.add_argument('--tr-tdt-weighting',choices=MODES,required=True)\n    p.add_argument('--validate-only',action='store_true');return p")
rsha=save(creator_new,r)

original_worker=(OUT/worker_old).read_text(encoding='utf-8')
w=original_worker.replace(helper_old[:-3],helper_new[:-3]).replace(core_old,core_new).replace(worker_old,worker_new)
w=replace(w,"from metrics import normalize_text",f"from metrics import normalize_text\nfrom {UFILE[:-3]} import apply_to_objective as apply_TR_loss, detached_metrics as TR_loss_metrics, POPULATION_MEAN_U, MODES\nfrom {helper_new[:-3]} import TR_loss_declaration")
w=w.replace('HF_TSC_8199_uniform_head_LR_control_v1','HF_TSC_8199_uniform_TR_TDT_weighting_control_v1')
w=replace(w,'"declared": head_lr_declaration(context,args.encoder_lr_schedule,args.head_lr), "sampler_restore": migration,','"declared": head_lr_declaration(context,args.encoder_lr_schedule,args.head_lr), "sampler_restore": migration,\n        "TR_loss_normalization":TR_loss_declaration(context,args.tr_tdt_weighting),')
w=replace(w,'        "head_lr_transition":config_identity["head_lr_transition"],','        "head_lr_transition":config_identity["head_lr_transition"],\n        "TR_loss_normalization":TR_loss_declaration(context,args.tr_tdt_weighting),')
w=replace(w,'totals = {key:0. for key in ("loss","tdt_loss","replay_kl","replay_duration_kl","replay_feature_distillation")}', 'totals = {key:0. for key in ("loss","tdt_loss","replay_kl","replay_duration_kl","replay_feature_distillation","tdt_loss_unweighted","tdt_loss_weighted")}\n        step_TR_weights=[]; TR_tdt_microbatch_components=[]')
w=replace(w,'            scaled = result.loss/args.accumulation','            result,TR_exposure=apply_TR_loss(result,batch.rows,mode=args.tr_tdt_weighting,torch_module=torch,population_mean_U=args.tr_tdt_population_mean_U)\n            step_TR_weights.extend(TR_exposure["actual_TR_weights"])\n            scaled = result.loss/args.accumulation')
w=replace(w,'            metrics = result.detached_metrics()','            metrics = TR_loss_metrics(result,TR_exposure,torch_module=torch)\n            TR_tdt_microbatch_components.append({key:metrics[key] for key in ("TR_tdt_unweighted_mean","TR_tdt_weighted_mean") if key in metrics})')
w=replace(w,'"step_audio_seen_seconds":step_audio,','"step_audio_seen_seconds":step_audio,\n                  "TR_tdt_weighting":args.tr_tdt_weighting,"TR_tdt_population_mean_U":args.tr_tdt_population_mean_U,\n                  "actual_TR_tdt_weights":step_TR_weights,"TR_tdt_microbatch_components":TR_tdt_microbatch_components,')
w=replace(w,'    return p\n\n\ndef validate_arguments(args):','    p.add_argument("--tr-tdt-weighting",choices=MODES,required=True)\n    p.add_argument("--tr-tdt-population-mean-U",type=float,default=POPULATION_MEAN_U)\n    return p\n\n\ndef validate_arguments(args):')
w=replace(w,'    majority_preflight(args)\n    return args','    if args.tr_tdt_population_mean_U!=POPULATION_MEAN_U or args.tr_tdt_weighting not in MODES:raise ValueError("Frozen TR normalization mode/population mean required")\n    majority_preflight(args)\n    return args')
w=replace(w,'"native_tdt_checkpointed_v4.py","atomic_paired_history_v1.py"):',f'"native_tdt_checkpointed_v4.py","atomic_paired_history_v1.py","{UFILE}"):')
for name in ('save_training_state','validate_own_state','encoder_lr_for_update','evaluate','retention_gates'):
    try: before=fn(original_worker,name);after=fn(w,name)
    except StopIteration:continue
    assert ast.dump(before,include_attributes=False)==ast.dump(after,include_attributes=False),name
wsha=save(worker_new,w)
sync=[n.lineno for n in ast.walk(fn(w,'run')) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='synchronize' and ast.unparse(n.func.value)=='torch.cuda'];assert len(sync)==1

c=(OUT/core_old).read_text(encoding='utf-8').replace(helper_old[:-3],helper_new[:-3]).replace(worker_old,worker_new).replace(core_old,core_new).replace('guarded_TSC_8199_uniform_head_LR_controller_v1','guarded_TSC_8199_uniform_TR_loss_controller_v1')
c=replace(c,"WORKER_SHA='46a13e6dcc8e6acf3dc593964fe3531d2adb6816c459a271f448f624e3251f8b'",f"WORKER_SHA='{wsha}'")
c=replace(c,"TRANSITION_SHA='5889125b84de81f43e7c6ea2a2abb3c953f0f802577d781f7844cdfff091585a'",f"TRANSITION_SHA='{hsha}'")
c=replace(c,'EXPECTED_SYNC_LINE=790',f'EXPECTED_SYNC_LINE={sync[0]}')
c=replace(c,"    values['encoder_min_lr']=float(option(argv,'--encoder-min-lr'))","    values['encoder_min_lr']=float(option(argv,'--encoder-min-lr'))\n    values['tr_tdt_weighting']=option(argv,'--tr-tdt-weighting')\n    values['tr_tdt_population_mean_U']=float(option(argv,'--tr-tdt-population-mean-U'))")
c=replace(c,"    pins.update({WORKER:WORKER_SHA,TRANSITION:TRANSITION_SHA,",f"    pins.update({{'{UFILE}':'{USHA}',WORKER:WORKER_SHA,TRANSITION:TRANSITION_SHA,")
c=replace(c,"('--persistence-mode','double_json'),('--encoder-lr-schedule',args.encoder_lr_schedule)","('--tr-tdt-weighting',args.tr_tdt_weighting),('--tr-tdt-population-mean-U',26.642061797177284),\n        ('--persistence-mode','double_json'),('--encoder-lr-schedule',args.encoder_lr_schedule)")
c=replace(c,"    state_path=scoped(option(argv,'--resume-exact' if exact else '--resume-majority'))","    require(option(argv,'--tr-tdt-weighting')==context['receipt']['tr_tdt_weighting'] and float(option(argv,'--tr-tdt-population-mean-U'))==26.642061797177284,'Bound TR normalization mode/mean required')\n    state_path=scoped(option(argv,'--resume-exact' if exact else '--resume-majority'))")
c=replace(c,"    p.add_argument('--head-lr',type=float,choices=(1e-5,3e-5),default=3e-5)","    p.add_argument('--head-lr',type=float,choices=(1e-5,3e-5),default=3e-5)\n    p.add_argument('--tr-tdt-weighting',choices=('clip','token_population'),required=True)")
csha=save(core_new,c)
a=(OUT/adapter_old).read_text(encoding='utf-8').replace(core_old[:-3],core_new[:-3]).replace(worker_old,worker_new).replace('TSC_uniform_head_LR_journal_persistence_runtime_v1','TSC_uniform_TR_loss_journal_persistence_runtime_v1').replace('3edcb4bbc2697a9f9208aac0d10c5497a588f4b6557b930060f36625fa4b97ee',csha).replace('46a13e6dcc8e6acf3dc593964fe3531d2adb6816c459a271f448f624e3251f8b',wsha)
asha=save(adapter_new,a)
report={'kind':'inactive_TR_loss_family_source_binding_v1','passed':True,'files':{helper_new:hsha,creator_new:rsha,worker_new:wsha,core_new:csha,adapter_new:asha,UFILE:USHA},'single_sync_line':sync[0],'serializer_own207_state_RNG_history_source350_sampler_AST_preserved':True,'future_actual_head_policy_required':True,'actual_factory_or_preflight_or_state_admission_performed':False,'actual_GPU_tested':False,'current_sources_edited':False}
path=OUT/'TR_loss_family_inactive_source_binding_v1.json';assert not path.exists();path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report))
