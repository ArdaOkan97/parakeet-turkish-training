"""Source-only NEW controller derivative; never dispatches a training process."""
from pathlib import Path
import ast,hashlib
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'outputs/parakeet_turkish'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def change(text,old,new,count=1):
    if text.count(old)!=count:raise ValueError((old[:120],text.count(old),count))
    return text.replace(old,new)
def function(text,name,new):
    tree=ast.parse(text);node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==name)
    lines=text.splitlines(keepends=True)
    return ''.join(lines[:node.lineno-1])+new.rstrip()+'\n'+''.join(lines[node.end_lineno:])
assert sha(HERE/'run_TSC_uniform_full_v1.py')=='dfbac1e473274728f287169a9b3db3edeb03f39046d694c615c026ae7e2ada4f'
c=(HERE/'run_TSC_uniform_full_v1.py').read_text(encoding='utf-8')
c=c.replace('train_lora_batched_tsc_uniform_full_v1.py','train_lora_batched_tsc_uniform_majority_v1.py')
c=c.replace('tsc_uniform_full_transition_v1.py','tsc_uniform_majority_transition_v1.py')
c=c.replace('tsc_uniform_full_sampler_v1.py','tsc_uniform_majority_sampler_v1.py')
c=c.replace('TSC_uniform_full_','TSC_uniform_majority_')
c=change(c,'guarded_TSC_8199_uniform_full_Portuguese_controller_v1','guarded_TSC_8199_uniform_majority_fullpass_controller_v1')
c=change(c,"WORKER_SHA='b9e550ec145c56a7a75521a997a8a6a33bd98eee7d76c1cb0cabfb116a40631c'","WORKER_SHA='PENDING_FINAL_WORKER'")
c=change(c,"TRANSITION_SHA='08ec46e079cadf2805f72b24078b92fc560e5701228a033228eb3519219079de'","TRANSITION_SHA='PENDING_FINAL_TRANSITION'")
c=change(c,"SAMPLER_SHA='8ebd56d20e47a568db4a7cfe50ed172ad86af253275b3b1172ad78a34b60439a'","SAMPLER_SHA='19adad00ed5d2c00ad3c07e9d31c221c8d171e1202b1ea83348176e276d61976'")
c=change(c,'EXPECTED_SYNC_LINE=667','EXPECTED_SYNC_LINE=0')
c=function(c,'boundary_context','''def boundary_context(argv):
    from tsc_uniform_majority_transition_v1 import preflight,source_context
    source=source_context(option(argv,'--majority-source-receipt'),option(argv,'--run-id'))
    values=dict(source['config']);values.update(source['source_recipe'])
    values.update(run_id=option(argv,'--run-id'),resume_optimization=None,
        optimization_source_receipt=None,resume_weights=None,resume_runtime_migration=None,
        resume_exact=scoped(option(argv,'--resume-exact')) if '--resume-exact' in argv else None)
    for flag in ('--resume-majority','--majority-source-receipt','--resume-uniform-full','--uniform-full-source-receipt',
        '--uniform-full-runtime-proof','--native-backend-provenance','--portuguese-preparation-proof',
        '--portuguese-baseline-proof','--portuguese-replay-manifest','--portuguese-dev-manifest',
        '--indexed-tr-directory','--stop-file'):
        values[flag[2:].replace('-','_')]=scoped(option(argv,flag))
    values['max_wall_seconds']=float(option(argv,'--max-wall-seconds'))
    return preflight(SimpleNamespace(**values))''')
c=change(c,"option(argv,'--steps')=='38225'", "int(option(argv,'--steps'))==context['configured_end_step']\n        and float(option(argv,'--replay-fraction'))==.125 and float(option(argv,'--replay-kl'))==40\n        and float(option(argv,'--duration-kl'))==2 and float(option(argv,'--feature-distillation'))==4")
c=change(c,"'--resume-exact' if exact else '--resume-uniform-full'","'--resume-exact' if exact else '--resume-majority'",3)
c=change(c,"'--uniform-full-source-receipt',\n        '--portuguese-preparation-proof'","'--uniform-full-source-receipt','--majority-source-receipt',\n        '--portuguese-preparation-proof'")
c=change(c,"'declared_optimization_change':'Original207 BEST6500 retained; fulluniformTR+Portuguese replay/retention explicitly added; native4 memory-only cleanup'",
    "'declared_optimization_change':'CLOSED207 frontierH retained; same TR pass14/2 allocation and density40/2/4; PT grace and all6 gates retained'")
c=function(c,'prepare','''def prepare(args):
    from tsc_uniform_majority_transition_v1 import source_context
    require(math.isfinite(args.window_hours) and .1<=args.window_hours<=3. and 5<=args.poll_seconds<=15,'Bounded windows/freshpolls required')
    output=scoped(args.output_dir);receipt_path=scoped(args.source_receipt)
    require(output.parent==HERE/'runs' and not output.exists(),'Fresh first controller directory required')
    context=source_context(str(receipt_path),args.run_id)
    source_status=read(context['receipt']['artifacts']['controller_status']['path'])
    argv=list(source_status['worker_arguments'])
    for flag in ('--resume-optimization','--optimization-source-receipt','--resume-exact','--resume-weights','--resume-runtime-migration'):
        argv=remove_option(argv,flag)
    paths={key:output/(args.run_id+'_'+name) for key,name in (('native','native_backend.json'),('identity','identity.json'),
        ('memory','memory.json'),('cache','cache.json'),('io','io.json'),('stop','stop.txt'),('log','worker.log'))}
    for flag,value in (('--run-id',args.run_id),('--resume-majority',context['source_checkpoint']['path']),
        ('--majority-source-receipt',receipt_path),('--uniform-full-runtime-proof',paths['native']),
        ('--indexed-tr-directory',source_status.get('worker_arguments') and option(argv,'--indexed-tr-directory')),
        ('--sampling-mode','uniform'),('--steps',context['configured_end_step']),('--replay-fraction',.125),
        ('--replay-kl',40.),('--duration-kl',2.),('--feature-distillation',4.),
        ('--native-backend-provenance',paths['native']),('--stop-file',paths['stop']),('--max-wall-seconds',args.window_hours*3600-300)):
        argv=set_option(argv,flag,value)
    if args.indexed_tr_directory:
        require(scoped(args.indexed_tr_directory)==scoped(option(argv,'--indexed-tr-directory')),'Original admitted index directory must remain unchanged')
    context=boundary_context(argv);pins=source_pins(context)
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{2,110}',args.run_id) and not (HERE/'runs'/args.run_id).exists(),
        'Fresh declared majority scientific namespace required')
    for suffix in ('best.pt','latest.pt'):require(not (ROOT/'work'/f'{args.run_id}_{suffix}').exists(),'Checkpoint namespace collision')
    python=ROOT/'work/nemo-native-env-v2/Scripts/python.exe'
    plan={'schema':1,'kind':CONTROLLER_KIND,'run_id':args.run_id,'controller_sha256':sha(__file__),
        'source_sha256':pins,'source_closure':context['receipt_record'],'source_checkpoint':record(context['source_checkpoint']['path']),
        'initial_source_checkpoint':context['source_checkpoint'],'source_selected_step':context['source_step'],'resume_step':context['source_step'],
        'uniform_origin_step':context['uniform_origin_step'],'prior_uniform_draws':context['prior_uniform_draws'],
        'source_history_prefix_sha256':context['source_history_prefix_sha256'],'window_hours':args.window_hours,
        'worker_arguments':argv,'worker_arguments_sha256':domain.guards.argument_sha(argv),
        'command':[str(python),str(Path(__file__).resolve()),'--worker-plan',str(output/'plan.json'),'--gpu-window-authorized'],
        'paths':{key:str(value) for key,value in paths.items()},'configured_steps':context['configured_end_step'],
        'remaining_uniform_updates':context['remaining_update_budget'],
        'competitive_champion_parked':context['parked_original_champion'],'inherited_parent_all6_BEST':context.get('source_best_checkpoint'),
        'declared_retention_coefficients':{'replay_kl':40.,'duration_kl':2.,'feature_distillation':4.},
        'expected_RP_mean_density':{'replay_kl':5.,'duration_kl':.25,'feature_distillation':.5},
        'equal_gradients_or_memory_fit_claimed':False,'PT_grace_reset':False,'objective_complete':False}
    validate_plan(plan);return plan''')
c=change(c,"selection['configured_steps']==38225", "selection['configured_steps']==plan['configured_steps']")
c=change(c,'6500<=len(history)<=38225',"plan['source_selected_step']<=len(history)<=plan['configured_steps']")
c=change(c,"proof=read(run/'uniform_full_initial_source_equivalence.json')", "proof=read(run/'majority_initial_source_equivalence.json')")
start=c.index("    require(proof['passed'] is True",c.index('def verify_completion'))
end=c.index('\n\n\ndef request_stop',start)
c=c[:start]+'''    require(proof['kind']=='TSC_uniform_majority_before_update_source_equivalence_v1'
        and proof['passed'] is True and proof['actual_GPU_decode'] is True
        and proof['source_selected_step']==plan['source_selected_step']
        and proof['compared_fixed_DEV_utterances']==350
        and proof['all_raw_transcripts_and_emitted_paths_exact'] is True
        and proof['RNG_restored_after_added_evaluation'] is True
        and proof['module_training_modes_restored_after_added_evaluation'] is True,
        'Actual same-H seven-language source350 equality required; failed gates remain truthful')
    require(selection['uniform_origin_step']==6500 and selection['majority_updates']==selection['completed_steps']-plan['source_selected_step'],
        'Original uniform clock and new quota-local clock must remain distinct')
    return {'full207_frontier':selection['completed_steps'],'actual_initial_source350':record(run/'majority_initial_source_equivalence.json'),
        'runtime_completion':record(runtime.with_suffix('.completion.json')),'full_unique_uniform_pass_complete':selection['full_new_uniform_pass_complete']}
''' +c[end:]
c=change(c,"'configured_steps':38225,'new_uniform_pass_updates':31725", "'configured_steps':state['configured_steps'],'remaining_uniform_updates_at_H':state['remaining_uniform_updates'],\n        'actual_majority_updates':max(0,state.get('published_step',state['resume_step'])-state['source_selected_step'])")
c=change(c,"original6500 full207, uniform31725 updates", "CLOSEDH full207, remaining uniform{state['remaining_uniform_updates']}14/2 updates")
c=change(c,"6500<selection['completed_steps']<38225", "previous['source_selected_step']<selection['completed_steps']<previous['configured_steps']")
c=change(c,"selection['completed_steps']>=38225", "selection['completed_steps']>=plan['configured_steps']")
c=change(c,"'configured_steps':38225,'new_uniform_updates':31725", "'configured_steps':plan['configured_steps'],'remaining_uniform_updates':plan['remaining_uniform_updates']")
c=change(c,"'explicit_first_material_lazy_allocation_measurement_v1'", "'explicit_first_material_14_2_allocation_measurement_v1'")
c=change(c,"first_measurement=args.measure_new_allocation_policy", "first_measurement=args.measure_new_allocation_policy")
c=c.replace('INACTIVE original8199 full253798 uniform TRAIN with Portuguese retention.','INACTIVE original8199 declared14/2 same-pass full207 continuation.')
c=c.replace('Declared source6500 or SAME-own exact only','Declared CLOSED frontierH or SAME-own exact only')
ast.parse(c)
dest=HERE/'run_TSC_uniform_majority_v1.py'
if dest.exists():raise ValueError('NEW controller already exists')
dest.write_text(c,encoding='utf-8',newline='\n')
print(sha(dest))
