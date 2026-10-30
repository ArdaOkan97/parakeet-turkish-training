"""CPU guards for versioned continuation orchestration; no CUDA initialization."""
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
OUTPUT=ROOT/'outputs/parakeet_turkish'
sys.path.insert(0,str(OUTPUT))
import run_experiment_loop_continuation_v2 as controller
import resource_watchdog_v1 as memory_guard
from datetime import datetime,timezone,timedelta
import train_lora_batched as trainer

checks=[];rejections=[]
def good(ok,label):
 if not ok:raise AssertionError(label)
 checks.append(label)
def rejects(function,label):
 try:function()
 except (RuntimeError,ValueError,KeyError,FileNotFoundError):rejections.append(label)
 else:raise AssertionError('Expected rejection: '+label)
def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8');return path
def ref(path):return {'path':str(path),'sha256':controller.sha(path)}

cfg=json.loads((OUTPUT/'runs'/controller.SOURCE_RUN/'run_config.json').read_text())
args=controller.parser().parse_args(['--prior-exit-receipt',str(OUTPUT/'continuation_prerequisites_v2/predecessor_exit_receipt.json'),
 '--cache-validation',str(OUTPUT/'cache_launch_gpu_validation_v1.json'),'--train-only'])
argv=controller.training_argv(cfg,args.loop_id+'_resume_B8',resume_exact='immutable907.pt',max_wall_seconds=4500,stop_file='owned.stop')
parsed=vars(trainer.parser().parse_args(argv))
good({key:parsed[key] for key in controller.IDENTITY_KEYS}==cfg['config_identity']['training_arguments'],'Actual frozen parser matches all31 exact identity arguments')
good(parsed['resume_exact']==Path('immutable907.pt') and parsed['max_wall_seconds']==4500,'Exact resume immutable path and finite stage cap')
for challenger in controller.CHALLENGERS:
 changed={k:challenger[k] for k in ('head_lr','lora_lr','new_token_lr_multiplier')};changed.update(epochs=.5,steps=None)
 fresh=vars(trainer.parser().parse_args(controller.training_argv(cfg,'cpu_new_'+challenger['name'],resume_weights='immutable_best.pt',changes=changed,max_wall_seconds=1500,stop_file='fresh.stop')))
 good(fresh['batch_size']==8 and fresh['accumulation']==2 and fresh['warmup_steps']==25 and fresh['epochs']==.5,'Fixed B8x2 fresh challenger '+challenger['name'])
 good(math.isclose(fresh['head_lr']*fresh['new_token_lr_multiplier'],3e-4,rel_tol=1e-15),'Preserved absolute new-row LR '+challenger['name'])
rejects(lambda:controller.training_argv(cfg,'cpu',resume_exact='a',resume_weights='b'),'Reject two resume modes')
rejects(lambda:controller.training_argv(cfg,'cpu',resume_exact='a',changes={'batch_size':1}),'No silent B1 fallback')
for name,value in [('continuation_seconds',float('nan')),('challenger_seconds',0.),('window_hours',float('inf')),('publication_stall_seconds',30.)]:
 bad=copy.copy(args);setattr(bad,name,value)
 rejects(lambda bad=bad:controller.validate_entry(bad),'Bounded malformed argv '+name)

_fixture_stamp=datetime.now(timezone.utc);_fixture_sequence=0
def sysmem(fraction=.5):
 global _fixture_sequence
 _fixture_sequence+=1
 stamp=(_fixture_stamp+timedelta(milliseconds=_fixture_sequence)).isoformat();limit=40_000_000_000
 return {'available':True,'observed_at_utc':stamp,'commit_fraction':fraction,
  'commit_limit_bytes':limit,'commit_used_bytes':limit*fraction,'commit_available_bytes':limit*(1-fraction)}
def observe(watch,history,fraction=.5,t=0.,sample=None):
 sample=sample or sysmem(fraction)
 with patch.object(memory_guard,'_timestamp',return_value=sample['observed_at_utc']):
  return watch.observe(history,sample,{'available':True,'owned_pid':123,'private_bytes':1_000_000_000},observed_monotonic=t)
history=[{'step':i,'current_reserved_vram_gb':31.931,'audio_seen_seconds':i*100.,'train_wall_seconds':i*2.} for i in range(1,1774)]
safe=sysmem();watch=controller.Watchdog(args,1773)
good(observe(watch,history,t=0.)[0] is None,'Inherited31GB reservation ignored before NEW optimizer updates')
good(observe(watch,history,.9083894883746458,t=5.)[0] is None,'Actual recorded single90.84% spike does not stop')
good(observe(watch,history,.7516157484365107,t=10.)[0] is None and watch.resource.moderate_streak==0,'Actual next75.16% observation resets moderate pressure')
watch=controller.Watchdog(args,1773)
for t in (0.,5.): good(observe(watch,history,.91,t=t)[0] is None,'Moderate pressure remains below thirdfreshpoll '+str(t))
good(observe(watch,history,.91,t=10.)[0]=='system_commit_above90percent_three_fresh_observations','Thirdfresh5-second highcommit reading requests cooperative stop')
watch=controller.Watchdog(args,1773);same=sysmem(.91)
good(observe(watch,history,t=0.,sample=same)[0] is None,'First distinctmoderate reading')
good(observe(watch,history,t=5.,sample=same)[0] is None and watch.resource.moderate_streak==1,'DuplicateUTC sample cannot advance streak')
good(observe(watch,history,t=15.,sample=same)[0]=='memory_telemetry_stale','Persistent stale telemetry requests review')
good(observe(controller.Watchdog(args,1773),history,.95,t=0.)[0]=='system_commit_at_or_above95percent','95%critical pressure stops immediately')
low=sysmem(.5);low.update(commit_limit_bytes=3e9,commit_used_bytes=1.5e9,commit_available_bytes=1.5e9)
good(observe(controller.Watchdog(args,1773),history,t=0.,sample=low)[0]=='system_commit_headroom_below2GB','Low2GBheadroom stops evenbelow90%')
good(controller.Watchdog(args,1773).observe(history,{'available':False},observed_monotonic=0.)[0]=='memory_telemetry_unavailable','Unavailableglobal telemetry stops')
new=history+[{'step':1774,'current_reserved_vram_gb':12.1,'audio_seen_seconds':177400.,'train_wall_seconds':3548.}]
watch=controller.Watchdog(args,1773)
for t in (0.,5.):good(observe(watch,new,t=t)[0] is None,'Reservation waits for third observation '+str(t))
good(observe(watch,new,t=10.)[0]=='reserved_above12GB_three_observations','Existing12GBthree-observation guard unchanged')
new[-1]['current_reserved_vram_gb']=5.
good(observe(watch,new,t=15.)[0] is None and watch.reserved_streak==0,'Recovered reservation resetsstreak')
rejects(lambda:observe(watch,[{'step':2}],t=20.),'Reject discontinuous publishedhistory')
watch=controller.Watchdog(args,0)
rows=[{'step':i,'current_reserved_vram_gb':5.,'audio_seen_seconds':i*100.,'train_wall_seconds':float(i)} for i in range(1,102)]
good(observe(watch,rows,t=0.)[0] is None and watch.rate_reference==100.,'100-updateinterval establishes healthyreference')
slow=copy.deepcopy(rows)
for i in range(102,202):slow.append({'step':i,'current_reserved_vram_gb':5.,'audio_seen_seconds':i*100.,'train_wall_seconds':101.+(i-101)*5.})
good(observe(watch,slow,t=5.)[0] is None and watch.rate_streak==1,'Advanced slowinterval starts regression streak')
for t in (10.,15.,20.):observe(watch,slow,t=t)
good(watch.rate_streak==1,'Repeated historytail cannot count as independent rate samples')
for i in (202,203):
 slow.append({'step':i,'current_reserved_vram_gb':5.,'audio_seen_seconds':i*100.,'train_wall_seconds':101.+(i-101)*5.})
 reason,_=observe(watch,slow,t=25.+(i-202)*5.)
good(reason=='sustained_training_rate_regression' and watch.rate_streak==3,'Threeadvancedslow intervals stillstop')
publication=controller.PublicationWatchdog(1773,10.)
good(publication.observe(1773,609.99) is None,'DEV/startup599.99seconds is permitted')
good(publication.observe(1773,610.)=='training_publication_stall_600seconds','600seconds unpublished progress stillstops')
good(publication.observe(1774,610.) is None and publication.latest_progress_at==610.,'Freshupdate resetspublicationwatch')
with patch.object(controller.time,'monotonic',return_value=100.):
 good(controller.remaining_for_child(101.)==1.,'Finalchild needspositive remainingtime')
 rejects(lambda:controller.remaining_for_child(100.),'Expireddeadline cannotlaunchfinalchild')
 rejects(lambda:controller.remaining_for_child(float('nan')),'NaNdeadline cannotlaunchfinalchild')
# Retain small original907GPUproof-shaped fixtures as historicalvalidation only.
history=[{'step':i,'current_reserved_vram_gb':5.,'audio_seen_seconds':i*100.,'train_wall_seconds':i*2.} for i in range(1,908)]
with tempfile.TemporaryDirectory(prefix='continuation_guard_',dir=ROOT/'work/research') as temporary:
 temp=Path(temporary)
 source=temp/'weights.pt';source.write_bytes(b'CPU immutable checkpoint fixture')
 copied,identity=controller.freeze_stage_input(source,'cpu_trial',temp)
 good(identity['immutable_input_sha256']==controller.sha(source)==controller.sha(copied),'Fresh trial input is immutable byte-exact copy')
 source.write_bytes(b'Parent mutable checkpoint legitimately changes later')
 good(controller.sha(copied)==identity['immutable_input_sha256'],'Later parent best updates do not alter trial start input')
 rejects(lambda:controller.freeze_stage_input(source,'cpu_trial',temp),'Reject immutable trial-input overwrite')
 # Build measured causal+real-hook-shaped evidence with actual frozen source hashes.
 smoke='cpu_production_smoke';smokedir=temp/smoke
 smokehistory=copy.deepcopy(history)
 smokehistory.extend([{'step':908,'sampled_source_ids':['tr-a','en-b']},{'step':909,'sampled_source_ids':['tr-c','es-d']}])
 smokeconfig={**cfg,'run_id':smoke}
 history_path=write(smokedir/'training_history.json',smokehistory);config_path=write(smokedir/'run_config.json',smokeconfig)
 arguments=['--run-id',smoke]
 io_path=write(temp/'io.json',{'status':'complete','training_semantics_changed':False,'arguments':arguments})
 cache={'worker_sha256':controller.FROZEN['train_lora_batched.py'],'worker_sha256_after':controller.FROZEN['train_lora_batched.py'],
  'caller_function':'run','caller_line':539,'threshold_reserved_bytes':6_000_000_000,'eligible_update_calls':2,
  'reclamations':1,'failed_reclamations':0,'cleanup_seconds_in_training_timing':.01}
 cache_record={'status':'complete','active_process_attachment':False,'frozen_worker_sha256':controller.FROZEN['train_lora_batched.py'],'cache_reclamation':cache}
 for key,name in [('launcher_sha256','launch_with_cache_reclaim_v1.py'),('cache_helper_sha256','cuda_cache_reclaim_v1.py'),('frozen_io_launcher_sha256','launch_with_io_retry_v1.py'),('frozen_io_helper_sha256','atomic_io_retry_v1.py')]:cache_record[key]=cache_record[key+'_after']=controller.FROZEN[name]
 cache_path=write(temp/'cache.json',cache_record)
 memory={'status':'complete','kind':'memory_guard_training_launcher_v2','allocator_cap_applied':True,'cap_bytes':13_000_000_000,'cleanup_threshold_bytes':6_000_000_000,
  'physical_cuda_device_bytes':16*1024**3,'per_process_allocator_fraction':13e9/(16*1024**3),'training_math_changed':False,
  'backend_flags_changed':False,'RNG_sampler_optimizer_changed':False,'global_driver_settings_changed':False,
  'cache_provenance_path':str(cache_path),'io_provenance_path':str(io_path),'arguments':['--',*arguments]}
 for key,name in [('memory_launcher_sha256','launch_with_memory_guard_v2.py'),('cache_launcher_sha256','launch_with_cache_reclaim_v1.py'),('cache_helper_sha256','cuda_cache_reclaim_v1.py')]:memory[key]=memory[key+'_after']=controller.FROZEN[name]
 memory_path=write(temp/'memory.json',memory)
 capture=temp/'tensor_evidence.pt';capture.write_bytes(b'CPU guard actual tensor capture pointer')
 measured_updates=[{'source_ids':[['tr-a'],['en-b']],'component_losses':{'tdt':1.,'kl':.1},'system':{'available':True,'commit_fraction':.75},'allocator':{'reserved_bytes.all.current':5e9}}, {'source_ids':[['tr-c'],['es-d']],'component_losses':{'tdt':.9,'kl':.2},'system':{'available':True,'commit_fraction':.75},'allocator':{'reserved_bytes.all.current':5e9}}]
 checkpoint=temp/'source907.pt';checkpoint.write_bytes(b'907 full exact checkpoint pointer')
 probe_worker=temp/'probe_worker.py';probe_worker.write_bytes(b'Frozen CPU synthetic probe worker')
 comparison={'sampler_rng_equal':True,'python_cpu_cuda_rng_equal':True,'quality_or_numerical_equivalence_waiver':False,
  'gradients':{'max_absolute_delta':.001,'relative_l2':.01,'cosine':.999},'weights':{'max_absolute_delta':.001,'relative_l2':.01,'cosine':.999}}
 probe={'schema':1,'kind':'cuda_cache_reclaim_gpu_validation_v1','status':'complete','passed':True,'solo_gpu':True,
  'matched_sources_equal':True,'shared_prefix_updates':2,'allocator_hard_cap_gb':13,'maximum_reserved_gb_guard':12,'maximum_commit_fraction':.9,'probe_version':3,'cleanup_threshold_gb':6,
  'identity':{'source_step':907,'source_run_id':controller.CACHE_SOURCE_RUN,'helper_sha256':controller.FROZEN['cuda_cache_reclaim_v1.py'],
   'launcher_sha256':controller.FROZEN['launch_with_cache_reclaim_v1.py'],'probe_sha256':controller.sha(probe_worker),'checkpoint_path':str(checkpoint),
   'checkpoint_sha256':controller.sha(checkpoint),'source_sha256':cfg['code_sha256']},
  'probe_sha256_after':controller.sha(probe_worker),'checkpoint_sha256_after':controller.sha(checkpoint),'original_run_mutated':False,
  'branches':{'control':{'completed_updates':2,'updates':measured_updates},'reclaim':{'completed_updates':100,'updates':copy.deepcopy(measured_updates)*50}},
  'comparisons':{'1':comparison,'2':copy.deepcopy(comparison)},'full_capture_path':str(capture),'full_capture_sha256':controller.sha(capture),
  'max_component_absolute_delta':0.,'intervention_requested_updates_completed':True}
 forks=[{'role':role,'source_step':step,'only_changed_checkpoint_field':'run_id','source_path':str(checkpoint),
   'source_sha256':controller.sha(checkpoint),'fork_path':str(checkpoint),'fork_sha256':controller.sha(checkpoint),'runtime_path':str(checkpoint)}
   for role,step in [('latest',907),('best',801)]]
 smoke_receipt=write(temp/'smoke_fork_receipt.json',{'kind':'cache_guard_frozen_worker_smoke_fork_v2','source_run_id':controller.CACHE_SOURCE_RUN,
  'smoke_run_id':smoke,'forks':forks})
 smokeconfig['warm_start']={'mode':'exact_resume','source_run_id':smoke,'source_step':907,'path':str(checkpoint),'sha256':controller.sha(checkpoint)}
 write(config_path,smokeconfig)
 probe_path=write(temp/'probe.json',probe)
 manifest={'schema':1,'kind':'cuda_cache_reclaim_launch_validation_v1','probe':ref(probe_path),'probe_worker':ref(probe_worker),
  'smoke_fork_receipt':ref(smoke_receipt),'smoke_fork_exact_CPU_checks':185532,'independent_actual_tensor_and_snapshot_recomputation':True,
  'builder_source_sha256':controller.sha(OUTPUT/'build_cache_launch_validation_v1.py'),
  'production_hook':{'cache_provenance':ref(cache_path),'io_provenance':ref(io_path),'memory_provenance':ref(memory_path),
   'training_history':ref(history_path),'run_config':ref(config_path),'initial_step':907,'completed_steps':909},
  'strict_numerical_gate':{'waived':False,'passed':False,'thresholds':{'gradient_relative_l2':.0005,'gradient_min_cosine':.9999995,
   'component_absolute_delta':.0001,'weights_relative_l2':.000001}}}
 manifest_path=write(temp/'manifest.json',manifest)
 good(controller.validate_cache_gpu(manifest_path,cfg)['real_frozen_optimizer_hook_executed'] is True,'Linked causal+production evidence validates actual scoped count and matched TRAIN sampling')
 good(controller.validate_cache_gpu(manifest_path,cfg)['strict_numerical_gate']['passed'] is False,'Measured numerical failure remains visible without automatic waiver')
 for label,mutate in [('wrong source',lambda x:x['branches']['reclaim']['updates'][0]['source_ids'][0].append('unexpected')),
  ('RNG changed',lambda x:x['comparisons']['2'].update(python_cpu_cuda_rng_equal=False)),
  ('missing final tensors',lambda x:x['comparisons'].pop('2')),('nonfinite components',lambda x:x['branches']['control']['updates'][0]['component_losses'].update(tdt=float('nan'))),
  ('original source changed',lambda x:x.update(original_run_mutated=True)),('cap missing',lambda x:x.update(allocator_hard_cap_gb=31)),
  ('intervention incomplete',lambda x:x.update(intervention_requested_updates_completed=False)),
  ('system commit violation',lambda x:x['branches']['reclaim']['updates'][-1]['system'].update(commit_fraction=.91)),
  ('reservation violation',lambda x:x['branches']['reclaim']['updates'][-1]['allocator'].update({'reserved_bytes.all.current':13e9}))]:
  bad=copy.deepcopy(probe);mutate(bad);write(probe_path,bad);manifest['probe']=ref(probe_path);write(manifest_path,manifest)
  rejects(lambda:controller.validate_cache_gpu(manifest_path,cfg),'Reject causal '+label)
 write(probe_path,probe);manifest['probe']=ref(probe_path);write(manifest_path,manifest)
 for label,mutate in [('wrong scope',lambda x:x['cache_reclamation'].update(caller_line=538)),('new update count',lambda x:x['cache_reclamation'].update(eligible_update_calls=909)),('wrong threshold',lambda x:x['cache_reclamation'].update(threshold_reserved_bytes=12e9))]:
  bad=copy.deepcopy(cache_record);mutate(bad);write(cache_path,bad);manifest['production_hook']['cache_provenance']=ref(cache_path);write(manifest_path,manifest)
  rejects(lambda:controller.validate_cache_gpu(manifest_path,cfg),'Reject production '+label)
 write(cache_path,cache_record);manifest['production_hook']['cache_provenance']=ref(cache_path);write(manifest_path,manifest)
 bad=copy.deepcopy(memory);bad['cap_bytes']=31_000_000_000;write(memory_path,bad);manifest['production_hook']['memory_provenance']=ref(memory_path);write(manifest_path,manifest)
 rejects(lambda:controller.validate_cache_gpu(manifest_path,cfg),'Reject production31GB allocator cap')
 write(memory_path,memory);manifest['production_hook']['memory_provenance']=ref(memory_path);write(manifest_path,manifest)
 bad=copy.deepcopy(manifest);bad['strict_numerical_gate']['waived']=True;write(manifest_path,bad)
 rejects(lambda:controller.validate_cache_gpu(manifest_path,cfg),'Reject numerical-threshold waiver')
 bad=copy.deepcopy(manifest);bad['strict_numerical_gate']['passed']=True;write(manifest_path,bad)
 rejects(lambda:controller.validate_cache_gpu(manifest_path,cfg),'Reject false claim of strict numerical pass')
 bad=copy.deepcopy(manifest);bad['strict_numerical_gate']['thresholds']['gradient_relative_l2']=.1;write(manifest_path,bad)
 rejects(lambda:controller.validate_cache_gpu(manifest_path,cfg),'Reject loosened strict numerical threshold')
 bad=copy.deepcopy(manifest);bad['independent_actual_tensor_and_snapshot_recomputation']=False;write(manifest_path,bad)
 rejects(lambda:controller.validate_cache_gpu(manifest_path,cfg),'Reject absent actual tensor/snapshot recomputation')
 # Exercise actual main try/finally on success and failure using a deliberately
 # unavailable TRAIN budget and mocked preflight/awake API. Popen is forbidden.
 import run_experiment_loop_recovery_v2 as prior
 for failed in (False,True):
  fake_output=temp/('awake_failure' if failed else 'awake_success')
  loop=fake_output/'runs'/args.loop_id;forkdir=loop/'fork_snapshot';forkdir.mkdir(parents=True)
  latest=forkdir/'latest.pt';latest.write_bytes(b'Immutable CPU907 fixture')
  fork=write(forkdir/'fork_manifest.json',{'passed':True,'status':'complete','source_run_id':controller.SOURCE_RUN,
    'target_run_id':args.loop_id+'_resume_B8','only_changed_path':'run_id','source_completed_step':1773,'inherited_selected_step':1773,
    'inherited_config_identity':cfg['config_identity'],'fork_checkpoints':{'latest':{'path':str(latest),'sha256':controller.sha(latest)}}})
  archived=write(loop/'fork_initial_states_archive.json',{'states':{'latest':{'archive_path':str(latest)}}})
  write(fake_output/'runs'/controller.PRIOR_LOOP/'frozen_final_selection.json',{'baseline_provenance':{}})
  fake_args=copy.copy(args);fake_args.train_only=True;fake_args.window_hours=.0001
  source_selection={'turkish_validation_wer':.69,'selected_step':1773,'weights':str(latest)}
  preflight={'source_config':cfg,'source_selection':source_selection,'source_sha256':{'guard':'fixed'},
   'old_deadline_at_utc':'2026-10-10T23:41:22.889692+00:00','prior_exit_receipt_path':'CPU fixture receipt',
   'fork_manifest_path':str(fork),'fork_manifest_sha256':'wrong' if failed else controller.sha(fork),
   'initial_fork_archive_path':str(archived),'initial_fork_archive_sha256':controller.sha(archived)}
  awake_calls=[]
  def awake(enable):awake_calls.append(enable);return {'supported':True,'enabled':enable,'flags':0x80000000|(1 if enable else 0)}
  with patch.object(controller,'OUTPUT',fake_output),patch.object(controller,'validate_entry',return_value=preflight),\
   patch.object(controller,'system_memory',return_value=safe),patch.object(controller,'current_sources',return_value={'guard':'fixed'}),\
   patch.object(prior,'validate_selection',return_value=(source_selection,{})),patch.object(prior,'keep_awake',side_effect=awake),\
   patch.object(controller.subprocess,'Popen',side_effect=AssertionError('GPU child forbidden in CPU guard')):
   if failed:rejects(lambda:controller.main(fake_args),'Owned main exception unwinds before GPU child')
   else:good(controller.main(fake_args)==0,'Owned main no-budget success completes without GPU child')
  result=json.loads((loop/'loop_status.json').read_text())
  good(result.get('objective_complete') is False,'Finitewindow never falsely completesqualitygoal')
  good(awake_calls==[True,False] and result['awake_request']['enabled'] and result['awake_clear']['enabled'] is False,
    'Actual main finally clears mocked thread-local awake request '+('after failure' if failed else 'after success'))

# Validate real completed parent receipt/fork/archives/source configuration while
# keeping actual GPU evidence explicitly pending. No loop_status is created.
with patch.object(controller,'validate_cache_gpu',return_value={'GPU_validation_pending':True}):
 actual=controller.validate_entry(args)
 good(actual['passed'] and actual['old_deadline_at_utc']=='2026-10-11T02:36:50.663652+00:00','Actual completed receipt/fork/archive preflight passes without changing old deadline')
 good(actual['fork_manifest_sha256']=='6dae863ad88823382f0e9a2757257fef379e0c15bf013b2622200218c075b446','Actual immutable fork manifest matches completed CPU fork')
 good(actual['initial_fork_archive_sha256']=='e31c992c9904d5aeba5efe2d56a8116110e0b5a8ff1c904dde53418c425bfeae','Actual907/801 initial archive aliases remain unchanged')
good(trainer.torch.cuda.is_initialized() is False,'All guards leave CUDA uninitialized')
report={'schema':1,'kind':'continuation_controller_CPU_guards_v2','passed':True,'gpu_used':False,
 'controller_sha256':controller.sha(OUTPUT/'run_experiment_loop_continuation_v2.py'),'test_source_sha256':controller.sha(__file__),
 'positive_assertions':len(checks),'rejected_corruptions':len(rejections),'checks':checks,'rejections':rejections,
 'scope':'NoGPU: actualfrozen argvparser; real1773 fork/receipt/archive; composedmemoryguard transient/sustained/critical fixtures; preserved historical907cacheproof guards; mainfinallycleanup.'}
target=OUTPUT/'continuation_controller_v2_cpu_verification.json';write(target,report)
print(json.dumps({'passed':True,'positive_assertions':len(checks),'rejected_corruptions':len(rejections),'CUDA_initialized':False,'report':str(target)}))
