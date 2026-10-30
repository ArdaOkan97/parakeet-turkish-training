"""Create only two new inactive dispatch modules from the closed working family."""
import ast,hashlib,pathlib
R=pathlib.Path(__file__).resolve().parents[2];O=R/'outputs/parakeet_turkish'
old=O/'run_TSC_uniform_TR_loss_v1.py';journal=O/'run_TSC_uniform_TR_loss_journal_v2.py'
assert hashlib.sha256(old.read_bytes()).hexdigest()=='e24a52574dfca48eb20efb25904df57ac14c30efebc767c8dda6eef84545bfb8'
assert hashlib.sha256(journal.read_bytes()).hexdigest()=='abf238408179461a17f7514d572f1a5f258845b85e3ac0b523c1de33b3fdbf00'

def replace_function(text,name,body):
    tree=ast.parse(text);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    lines=text.splitlines(keepends=True);return ''.join(lines[:node.lineno-1])+body.strip()+'\n'+''.join(lines[node.end_lineno:])

core=old.read_text(encoding='utf-8')
for a,b in [
 ('train_lora_batched_tsc_uniform_TR_loss_v1.py','train_lora_batched_tsc_expanded_TRAIN_v1.py'),
 ('tsc_uniform_TR_loss_transition_v1','tsc_expanded_TRAIN_transition_v1'),
 ('guarded_TSC_8199_uniform_TR_loss_controller_v1','guarded_TSC_8199_expanded_TRAIN_controller_v1'),
 ("WORKER_SHA='8df7018a8ac59ee8c95003294983ca913c4d1522ca738d680c256dc938d02484'","WORKER_SHA='PENDING'"),
 ("TRANSITION_SHA='9efa14fdd3c4e4ed75d2a355c39aa6fabdca6153efb30904d77450662f76f594'","TRANSITION_SHA='b41ed732_PENDING'"),
 ("SAMPLER_SHA='8177ae6195ee9ddda042258437c00358f4a6758d18225e3bc7c3735301391ba9'","SAMPLER_SHA='fd73fee28cdaac7bf37dd9ef2ec5089086221c466058dfbcb1ddbfdeb0abb027'"),
 ('EXPECTED_SYNC_LINE=798','EXPECTED_SYNC_LINE=810'),
 ("'tsc_uniform_encoder_schedule_sampler_v1.py':SAMPLER_SHA","'tsc_expanded_TRAIN_sampler_v1.py':SAMPLER_SHA"),
 ("'lazy_training_rows_v1.py':LAZY_ROWS_SHA,","'lazy_training_rows_v1.py':LAZY_ROWS_SHA,'expanded_training_rows_v1.py':EXPANDED_ROWS_SHA,"),
 ("LAZY_ROWS_SHA='faf2b549d2e5e3d2f2fd78d282b612c6943e813cbedbc3f6b16caedab8249fd2'","LAZY_ROWS_SHA='faf2b549d2e5e3d2f2fd78d282b612c6943e813cbedbc3f6b16caedab8249fd2'\nEXPANDED_ROWS_SHA='aaf3b188669c05ab0ff67e2be004b41f2cc21a81c8be368f58e7d501800c2370'"),
 ("source=source_context(option(argv,'--majority-source-receipt'),option(argv,'--run-id'))","source=source_context(option(argv,'--data-expansion-source-receipt'),option(argv,'--run-id'))"),
 ("for flag in ('--resume-majority'","for flag in ('--resume-data-expansion','--data-expansion-source-receipt','--resume-majority'"),
 ("values['persistence_mode']=option(argv,'--persistence-mode')","values['persistence_mode']=option(argv,'--persistence-mode')\n    values['data_arm']=option(argv,'--data-arm')"),
 ("context['receipt']['encoder_lr_schedule']","context['source_recipe']['encoder_lr_schedule']"),
 ("option(argv,'--encoder-schedule-updates')=='18129'","option(argv,'--encoder-schedule-updates')=='25198'"),
 ("context['receipt']['tr_tdt_weighting']","context['source_recipe']['tr_tdt_weighting']"),
 ("'--resume-exact' if exact else '--resume-majority'","'--resume-exact' if exact else '--resume-data-expansion'"),
 ("'lazy_helper_sha256':LAZY_ROWS_SHA,'source_sha256':pins","'lazy_helper_sha256':LAZY_ROWS_SHA,'expanded_rows_helper_sha256':EXPANDED_ROWS_SHA,'source_sha256':pins"),
 ("'declared_optimization_change':'CLOSED207 lowerLR frontier retained; onePT+oneoldfive replay withPT40/2/4 andoldfive5/3 compensation only; no reset'","'declared_optimization_change':'Coherent qualified40750 full207; explicit matched base/expanded TR coverage allocation; unchanged six replay streams and fixed TRpopulation denominator; no optimizer/RNG/counter reset'"),
 ("default=3e-5)","default=1e-5)"),
 ("choices=('constant','cosine'))","choices=('constant','cosine'),default='cosine')"),
 ("choices=('clip','token_population'))","choices=('clip','token_population'),default='token_population')"),
 ("p.add_argument('--indexed-tr-directory',type=Path)","p.add_argument('--indexed-tr-directory',type=Path);p.add_argument('--data-arm',choices=('base','expanded'))"),
]:
    assert a in core,a;core=core.replace(a,b)
# Add the immutable new source/input records to the actual native token.
core=core.replace("'--tr-frame-metadata','--tr-frame-metadata-proof','--data-v3-proof',", "'--tr-frame-metadata','--tr-frame-metadata-proof','--data-v3-proof','--data-expansion-source-receipt','--resume-data-expansion',")
core=core.replace("and int(option(argv,'--steps'))==context['configured_end_step']", "and int(option(argv,'--steps'))==context['configured_end_step']\n        and option(argv,'--data-arm')==context['data_arm'] and int(option(argv,'--native-max-joint-bytes'))==1700000000")
core=replace_function(core,'prepare',r'''
def prepare(args):
    from tsc_expanded_TRAIN_transition_v1 import source_context
    require(math.isfinite(args.window_hours) and .1<=args.window_hours<=3. and 5<=args.poll_seconds<=15,'Bounded windows/fresh polls required')
    output=scoped(args.output_dir);receipt_path=scoped(args.source_receipt)
    require(output.parent==HERE/'runs' and not output.exists(),'Fresh first controller directory required')
    require(args.encoder_lr_schedule=='cosine' and args.persistence_mode=='double_json' and args.head_lr==1e-5
        and args.tr_tdt_weighting=='token_population','Selected matched data recipe required')
    context=source_context(str(receipt_path),args.run_id)
    require(args.data_arm==context['data_arm'],'Actual data arm must match purpose receipt')
    source_status=read(context['receipt']['artifacts']['controller_status']['path'])
    argv=list(source_status['worker_arguments'])
    for flag in ('--resume-optimization','--optimization-source-receipt','--resume-exact','--resume-weights','--resume-runtime-migration','--persistence-benchmark-updates'):
        argv=remove_option(argv,flag)
    paths={key:output/(args.run_id+'_'+name) for key,name in (('native','native_backend.json'),('identity','identity.json'),
        ('memory','memory.json'),('cache','cache.json'),('io','io.json'),('stop','stop.txt'),('log','worker.log'))}
    for flag,value in (('--run-id',args.run_id),('--resume-data-expansion',context['source_checkpoint']['path']),
        ('--data-expansion-source-receipt',receipt_path),('--data-arm',args.data_arm),('--uniform-full-runtime-proof',paths['native']),
        ('--sampling-mode','uniform'),('--steps',context['configured_end_step']),('--head-lr',1e-5),('--lora-lr',.0001),('--replay-fraction',.125),
        ('--replay-kl',40.),('--duration-kl',2.),('--feature-distillation',20.),('--portuguese-token-kl-multiplier',1.),('--retention-stop-policy','monitor_only'),
        ('--tr-tdt-weighting','token_population'),('--tr-tdt-population-mean-U',26.642061797177284),
        ('--persistence-mode','double_json'),('--encoder-lr-schedule','cosine'),('--encoder-schedule-updates',25198),('--encoder-min-lr',3e-5),
        ('--native-backend-provenance',paths['native']),('--stop-file',paths['stop']),('--max-wall-seconds',args.window_hours*3600-300)):
        argv=set_option(argv,flag,value)
    if args.indexed_tr_directory:
        require(scoped(args.indexed_tr_directory)==scoped(option(argv,'--indexed-tr-directory')),'Immutable original lineage index required')
    context=boundary_context(argv);pins=source_pins(context)
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{2,110}',args.run_id) and not (HERE/'runs'/args.run_id).exists(),'Fresh data-arm namespace required')
    for suffix in ('best.pt','latest.pt'):require(not (ROOT/'work'/f'{args.run_id}_{suffix}').exists(),'Checkpoint namespace collision')
    plan={'schema':1,'kind':CONTROLLER_KIND,'run_id':args.run_id,'controller_sha256':sha(__file__),
        'source_sha256':pins,'source_closure':context['receipt_record'],'source_checkpoint':record(context['source_checkpoint']['path']),
        'initial_source_checkpoint':context['source_checkpoint'],'source_selected_step':40750,'resume_step':40750,
        'uniform_origin_step':context['uniform_origin_step'],'quota_boundary_step':context['quota_boundary_step'],
        'PT_focus_boundary_step':context['PT_focus_boundary_step'],
        'data_arm':args.data_arm,'active_TR_pool_clips':context['TR_pool_clips'],'expanded_index_proof':context['expanded_index_proof_record'],
        'persistence_mode':'double_json','encoder_lr_schedule':'cosine','encoder_schedule_updates':25198,'head_lr':1e-5,
        'source_lineage_end_step':43238,'encoder_schedule_end_step':65948,
        'matched_start_state_and_optimizer_replay_budget':True,'matched_future_TR_examples':False,'numerical_bitwise_reproducibility_claimed':False,
        'source_history_prefix_sha256':context['source_history_prefix_sha256'],'window_hours':args.window_hours,
        'same_policy_previous':{'peak_observed_owned_private_bytes':source_status['peak_observed_owned_private_bytes'],
            'finished_at_utc':source_status['finished_at_utc'],'reference_policy_only_new_pool_native_fit_unproven':True},
        'worker_arguments':argv,'worker_arguments_sha256':domain.guards.argument_sha(argv),
        'command':[str(ROOT/'work/nemo-native-env-v2/Scripts/python.exe'),str(Path(__file__).resolve()),'--worker-plan',str(output/'plan.json'),'--gpu-window-authorized'],
        'paths':{key:str(value) for key,value in paths.items()},'configured_steps':65948,'remaining_uniform_updates':25198,
        'competitive_champion_parked':context['parked_original_champion'],'inherited_parent_all6_BEST':context['source_best_checkpoint'],
        'declared_retention_coefficients':{'base_replay_kl':40.,'Portuguese_token_kl':40.,'old_five_token_kl':40*5/3,'duration_kl':2.,'feature_distillation':20.},
        'expected_RP_mean_density':{'replay_kl':(40+40*5/3)/16,'duration_kl':(2+2*5/3)/16,'feature_distillation':(20+20*5/3)/16},
        'equal_gradients_or_memory_fit_claimed':False,'PT_grace_reset':False,'objective_complete':False}
    validate_plan(plan);return plan
''')
start=core.index("    require(selection['run_id']==plan['run_id']",core.index('def verify_completion'))
end=core.index("    proof=read(run/'majority_initial_source_equivalence.json')",start)
core=core[:start]+'''    require(selection['run_id']==plan['run_id'] and selection['configured_steps']==plan['configured_steps']==65948
        and len(history)==selection['completed_steps'] and plan['source_selected_step']<=len(history)<=plan['configured_steps']
        and selection['heldout_tests_evaluated'] is False and config['trainable_parameter_tensors']==207
        and config['turkish_clips_used']==plan['active_TR_pool_clips'] and config['replay_clips_used']==1914,'Full207/activeTR/1914 frontier required')
    local=selection['completed_steps']-40750;epoch=selection['data_expansion_epoch']
    require(config['phase']=='HF_TSC_8199_expanded_TRAIN_control_v1'
        and 0<local<=25198 and selection['encoder_schedule_updates_completed']==local
        and selection['data_expansion_updates']==local and selection['data_arm']==plan['data_arm']==config['data_arm']
        and selection['encoder_lr_schedule']==plan['encoder_lr_schedule']=='cosine'
        and selection['encoder_schedule_horizon']==25198 and selection['persistence_mode']=='double_json'
        and epoch['source_step']==40750 and epoch['local_updates']==local and epoch['data_arm']==plan['data_arm']
        and epoch['draws']==14*local and epoch['pool']==plan['active_TR_pool_clips']
        and 0<epoch['unique']<=min(epoch['draws'],epoch['pool'])
        and epoch['new_epoch_not_inherited_prefix'] is True,'Actual40750+local data-pool clock required')
    if selection['completed_steps']==65948:
        require(epoch['full_pass_complete'] is True and epoch['unique']==epoch['pool'],'Declared full-budget actual coverage incomplete')
        require(epoch['repeat_visits']==(4 if plan['data_arm']=='expanded' else 98974),'Declared arm tail/repeat visits differ')
'''+core[end:]
core=core.replace("'full_unique_uniform_pass_complete':selection['full_new_uniform_pass_complete']}","'full_unique_uniform_pass_complete':selection['full_new_uniform_pass_complete'],'data_expansion_epoch':epoch,'data_arm':plan['data_arm']}")
core=core.replace("'encoder_schedule_updates':18129", "'encoder_schedule_updates':25198")
core=core.replace("state.get('published_step',state['resume_step'])-25109", "state.get('published_step',state['resume_step'])-40750")
core=core.replace("'original_full_pass_already_complete':True,", "'original_full_pass_already_complete':True,'data_arm':state['data_arm'],'active_TR_pool_clips':state['active_TR_pool_clips'],\n        'actual_data_expansion_updates':max(0,state.get('published_step',state['resume_step'])-40750),")
core=core.replace("'same_policy_peak_observed_private_bytes':peak", "'reference_peak_observed_private_bytes':peak,'reference_policy_only_new_pool_native_fit_unproven':bool(previous and previous.get('reference_policy_only_new_pool_native_fit_unproven'))")
core=core.replace('INACTIVE original8199 matched TR-only normalization controller.','INACTIVE coherent40750 matched original/expanded TRAIN controller.')
core=core.replace('Complete selected-head evidence must precede initial source admission.','Completed normalization plus actual source eligibility precedes native admission.')
core=core.replace("TRANSITION_SHA='b41ed732_PENDING'", "TRANSITION_SHA='"+hashlib.sha256((O/'tsc_expanded_TRAIN_transition_v1.py').read_bytes()).hexdigest()+"'")

adapter=journal.read_text(encoding='utf-8')
adapter=adapter.replace('import run_TSC_uniform_TR_loss_v1 as c','import run_TSC_expanded_TRAIN_v1 as c')
adapter=adapter.replace("EXPECTED_CORE='e24a52574dfca48eb20efb25904df57ac14c30efebc767c8dda6eef84545bfb8'","EXPECTED_CORE='PENDING'")
adapter=adapter.replace("EXPECTED_WORKER='8df7018a8ac59ee8c95003294983ca913c4d1522ca738d680c256dc938d02484'","EXPECTED_WORKER='PENDING'")
adapter=adapter.replace('TSC_uniform_TR_loss_journal_persistence_runtime_v1','TSC_expanded_TRAIN_journal_persistence_runtime_v1')
a=adapter.index('# Runtime-only correction');b=adapter.index('def byte_record',a)
adapter=adapter[:a]+'''def load_previous(args):
    previous=read(scoped(args.previous_status))
    require(previous.get('status')!='failed','Failed own window requires a new explicit review; no legacy projection admitted')
    return previous

'''+adapter[b:]
adapter=adapter.replace("require(args.encoder_lr_schedule in ('constant','cosine') and args.head_lr in (1e-5,3e-5)","require(args.encoder_lr_schedule=='cosine' and args.head_lr==1e-5")
adapter=adapter.replace("'completion_phase_literal_correction':{'from':_old_phase,'to':_new_phase,'frozen_core_unchanged':True},", "'completion_phase':'HF_TSC_8199_expanded_TRAIN_control_v1','source40750_new_local_horizon':25198,")
adapter=adapter.replace(";p.add_argument('--reviewed-closure',type=Path)","")
adapter=adapter.replace('INACTIVE matched TR-only loss study with journal publication.','INACTIVE matched base/expanded TRAIN study with journal publication.')

for name,text in [('run_TSC_expanded_TRAIN_v1.py',core),('run_TSC_expanded_TRAIN_journal_v1.py',adapter)]:
    path=O/name;assert not path.exists();ast.parse(text);path.write_text(text,encoding='utf-8',newline='\n');print(name,len(text))
