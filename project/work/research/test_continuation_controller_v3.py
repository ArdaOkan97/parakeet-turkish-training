"""Small CPU semantic and corruption checks; no model or CUDA execution."""
import copy
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
OUTPUT=ROOT/'outputs/parakeet_turkish'
sys.path.insert(0,str(OUTPUT))
import run_experiment_loop_continuation_v3 as c
import resource_watchdog_v1 as resource

checks=[];rejections=[]
def good(value,label):
    if not value:raise AssertionError(label)
    checks.append(label)
def reject(function,label):
    try:function()
    except (RuntimeError,ValueError,KeyError):rejections.append(label)
    else:raise AssertionError('Expected rejection: '+label)
def put(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False),encoding='utf-8');return path
def ref(path):return {'path':str(path),'sha256':c.sha(path)}

config=c.read(OUTPUT/'runs'/c.SOURCE_RUN/'run_config.json',{})
args=c.parser().parse_args(['--prior-exit-receipt',str(OUTPUT/'continuation_prerequisites_v3/predecessor_exit_receipt.json'),
    '--cache-validation',str(OUTPUT/'cache_launch_gpu_validation_v1.json'),
    '--memory-policy-validation',str(OUTPUT/'memory_policy_launch_validation_v1.json'),'--train-only'])
good(args.loop_id=='adaptive_20261010_continuation_03' and c.SOURCE_STEP==c.SOURCE_BEST_STEP==1821,'Exact new namespace and inherited1821 state')
good(c.FROZEN['launch_with_memory_guard_v3.py']==c.sha(OUTPUT/'launch_with_memory_guard_v3.py'),'Production wrapper source pin')
argv=c.training_argv(config,args.loop_id+'_resume_B8',resume_exact=ROOT/'work/immutable_state.pt',max_wall_seconds=4500,stop_file=ROOT/'work/owned.stop')
good('--resume-exact' in argv and '--resume-weights' not in argv,'Exact state mode preserved')
good(argv[argv.index('--batch-size')+1]=='8' and argv[argv.index('--accumulation')+1]=='2','PhysicalB8 effective16 preserved')
good('--no-encoder-checkpoint' not in argv and argv[argv.index('--chunk-size')+1]=='16','Checkpoint and loss chunk preserved')
good(c.argument_sha(argv)==c.argument_sha(['--',*argv]),'Canonical identity ignores only separator')
different=list(argv);different[different.index('--lora-lr')+1]='0.9'
good(c.argument_sha(argv)!=c.argument_sha(different),'Worker argument identity detects LR change')
reject(lambda:c.training_argv(config,'safe',resume_exact='x',changes={'batch_size':1}), 'No silentB1 fallback')

now=datetime.now(timezone.utc)
identity={'schema':1,'kind':'owned_training_worker_identity_v1','expected_run_id':args.loop_id+'_resume_B8',
    'worker_arguments_sha256':c.argument_sha(argv),'pid':22,'ppid':11,'started_at_utc':now.isoformat(),
    'process_creation_identity':{'pid':22,'available':True,'creation_unix_seconds':now.timestamp()}}
for key,name in [('memory_launcher_sha256','launch_with_memory_guard_v3.py'),('cache_launcher_sha256','launch_with_cache_reclaim_v1.py'),
    ('cache_helper_sha256','cuda_cache_reclaim_v1.py'),('resource_helper_sha256','resource_watchdog_v1.py'),('frozen_worker_sha256','train_lora_batched.py')]:identity[key]=c.FROZEN[name]
good(c.validate_worker_identity(identity,identity['expected_run_id'],argv,11,now.isoformat())['pid']==22,'Venv redirector direct child attribution')
same=copy.deepcopy(identity);same['pid']=11;same['process_creation_identity']['pid']=11
good(c.validate_worker_identity(same,same['expected_run_id'],argv,11,now.isoformat())['pid']==11,'Native interpreter without redirector attribution')
for field,value,label in [('ppid',33,'Unowned PID lineage'),('expected_run_id','other','Wrong worker run'),
    ('worker_arguments_sha256','bad','Changed worker arguments'),('memory_launcher_sha256','bad','Changed worker wrapper')]:
    corrupt=copy.deepcopy(identity);corrupt[field]=value
    reject(lambda:c.validate_worker_identity(corrupt,identity['expected_run_id'],argv,11,now.isoformat()),label)
old=copy.deepcopy(identity);old['process_creation_identity']['creation_unix_seconds']=now.timestamp()-60
reject(lambda:c.validate_worker_identity(old,identity['expected_run_id'],argv,11,now.isoformat()),'Reused PID predates launch')

with tempfile.TemporaryDirectory(dir=ROOT/'work/research',prefix='continuation_v3_') as directory:
    folder=Path(directory);checkpoint=folder/'checkpoint.pt';checkpoint.write_bytes(b'CPU fixture only')
    capture=folder/'capture.pt';capture.write_bytes(b'CPU fixture only')
    source=OUTPUT/'profile_cuda_cache_reclaim_v4.py'
    terms={'loss':1.,'tdt_loss':.4,'replay_kl':.1,'replay_duration_kl':.2,'replay_feature_distillation':.3}
    def sample(index):
        when=now+timedelta(seconds=index*5)
        return {'observed_at_utc':when.isoformat(),'observed_monotonic':float(index*5),'system_memory':
          {'available':True,'observed_at_utc':when.isoformat(),'commit_fraction':.8,'commit_used_bytes':32_000_000_000,
           'commit_limit_bytes':40_000_000_000,'commit_available_bytes':8_000_000_000},'worker_private_memory':{'pid':22,'available':True,'private_bytes':1_000_000_000}}
    updates=[]
    for index in range(100):
        before=sample(index*2);after=sample(index*2+1)
        updates.append({'source_ids':[[f'train{index}_{i}' for i in range(8)],[f'replay{index}_{i}' for i in range(8)]],
          'component_losses':terms,'gradient_norm_before_clip':1.,'audio_seconds':120.,'training_seconds_including_stats_cleanup':2.,
          'system':after['system_memory'],'allocator':{'reserved_bytes.all.current':3_000_000_000},
          'pre_cleanup_allocator':{'reserved_bytes.all.current':5_000_000_000},'before_owned_memory_telemetry':before,
          'owned_memory_telemetry':after,'moderate_commit_streak_after_observation':0})
    comparisons={str(i):{'sampler_rng_equal':True,'python_cpu_cuda_rng_equal':True,'quality_or_numerical_equivalence_waiver':False,
       'gradients':{'relative_l2':.0001,'cosine':.99999999,'max_absolute_delta':.0001},
       'weights':{'relative_l2':1e-8,'cosine':1.,'max_absolute_delta':1e-8}} for i in (1,10)}
    probe={'schema':1,'status':'complete','passed':True,'solo_gpu':True,'probe_version':4,'matched_sources_equal':True,
      'identity':{'source_step':1821,'source_run_id':c.SOURCE_RUN,'source_sha256':config['code_sha256'],
       'checkpoint_path':str(checkpoint),'checkpoint_sha256':c.sha(checkpoint),'probe_sha256':c.sha(source),
       'resource_helper_sha256':c.FROZEN['resource_watchdog_v1.py'],'memory_launcher_v3_sha256':c.FROZEN['launch_with_memory_guard_v3.py'],
       'helper_sha256':c.FROZEN['cuda_cache_reclaim_v1.py'],'launcher_sha256':c.FROZEN['launch_with_cache_reclaim_v1.py']},
      'probe_sha256_after':c.sha(source),'checkpoint_sha256_after':c.sha(checkpoint),'original_run_mutated':False,
      'branch_policies':{'control':{'allocator_hard_cap_gb':13.,'cleanup_threshold_gb':6.},'reclaim':{'allocator_hard_cap_gb':9.,'cleanup_threshold_gb':4.}},
      'resource_policy':dict(resource.POLICY),'maximum_reserved_gb_guard':12.,'allocator_hard_cap_gb':9.,'cleanup_threshold_gb':4.,
      'actual_worker_pid':22,'shared_prefix_updates':10,'intervention_requested_updates_completed':True,
      'branches':{'control':{'updates':copy.deepcopy(updates[:10]),'completed_updates':10},
       'reclaim':{'updates':updates,'completed_updates':100,'stop_reason':None,'capture_memory_events':[]}},
      'comparisons':comparisons,'max_component_absolute_delta':0.,'full_capture_path':str(capture),'full_capture_sha256':c.sha(capture)}
    probe_path=folder/'probe.json';manifest_path=folder/'manifest.json'
    gate={'passed':True,'waived':False,'thresholds':{'gradient_relative_l2':.0005,'gradient_min_cosine':.9999995,
      'component_absolute_delta':.0001,'weights_relative_l2':.000001}}
    def validate(value,gate_value=gate):
        put(probe_path,value)
        put(manifest_path,{'schema':1,'kind':'memory_guard_policy_launch_validation_v1','probe':ref(probe_path),'probe_worker':ref(source),
          'production_wrapper':ref(OUTPUT/'launch_with_memory_guard_v3.py'),'strict_numerical_gate':gate_value})
        return c.validate_memory_policy(manifest_path,config)
    good(validate(probe)['intervention_completed_updates']==100,'Matched100-update9/4 policy acceptance')
    good(validate(probe)['strict_numerical_gate']['historical_full_BF16_original_vs_v2_equivalence_gate_passed'] is False,'Historical failedBF16 gate preserved')
    for mutation,label in [
       (lambda p:p['branches']['reclaim'].update(completed_updates=99),'Incomplete intervention'),
       (lambda p:p['branches']['reclaim'].update(stop_reason='system_commit_at_or_above95percent'),'Critical policy stop'),
       (lambda p:p['branch_policies']['reclaim'].update(allocator_hard_cap_gb=13),'Wrong allocator policy'),
       (lambda p:p['identity'].update(source_step=1773),'Wrong source step'),
       (lambda p:p['branches']['reclaim']['updates'][0]['owned_memory_telemetry']['worker_private_memory'].update(pid=11),'Launcher mistaken for worker'),
       (lambda p:p['comparisons']['10'].update(sampler_rng_equal=False),'Changed sampler'),
       (lambda p:p['comparisons']['10']['gradients'].update(relative_l2=.01),'Falsely passed strict gradient gate'),
       (lambda p:p['branches']['reclaim']['updates'][0]['component_losses'].update(loss=float('nan')),'Nonfinite loss'),
       (lambda p:p['branches']['reclaim']['updates'][0]['allocator'].update({'reserved_bytes.all.current':10_000_000_000}),'Exceeded allocator cap')]:
        corrupt=copy.deepcopy(probe);mutation(corrupt);reject(lambda:validate(corrupt),label)
    failed=copy.deepcopy(probe);failed['comparisons']['10']['gradients']['relative_l2']=.01
    declared=copy.deepcopy(gate);declared['passed']=False
    good(validate(failed,declared)['strict_numerical_gate']['passed'] is False,'Measured numerical failure recorded without waiver or false pass')
    cache=c.read(OUTPUT/'runs'/c.PRIOR_LOOP/(c.SOURCE_RUN+'_cache_provenance.json'),{})
    cache['cache_reclamation']['threshold_reserved_bytes']=4_000_000_000
    io=c.read(OUTPUT/'runs'/c.PRIOR_LOOP/(c.SOURCE_RUN+'_io_provenance.json'),{})
    io['arguments']=argv
    io_path=folder/'io.json';cache_path=folder/'cache.json';memory_path=folder/'memory.json';identity_path=folder/'identity.json'
    put(io_path,io);put(cache_path,cache);put(identity_path,identity)
    memory={'status':'complete','kind':'memory_guard_training_launcher_v3','cleanup_threshold_bytes':4_000_000_000,
      'cap_bytes':9_000_000_000,'allocator_cap_applied':True,'training_math_changed':False,'backend_flags_changed':False,
      'RNG_sampler_optimizer_changed':False,'global_driver_settings_changed':False,'physical_cuda_device_bytes':16_000_000_000,
      'per_process_allocator_fraction':9/16,'cache_provenance_path':str(cache_path),'io_provenance_path':str(io_path),
      'worker_identity_path':str(identity_path),'worker_identity_sha256':c.sha(identity_path),'worker_pid':22,'worker_ppid':11,
      'run_id':identity['expected_run_id'],'worker_arguments_sha256':c.argument_sha(argv),
      'owned_update_memory_telemetry':{'eligible_update_calls':cache['cache_reclamation']['eligible_update_calls'],
        'successful_samples':1,'failed_samples':0,'inside_original_training_timing':True,'minimum_sampling_interval_seconds':5.,
        'sample_capacity':128,'samples_omitted':0,'samples':[sample(0)]}}
    for key,name in [('memory_launcher_sha256','launch_with_memory_guard_v3.py'),('cache_launcher_sha256','launch_with_cache_reclaim_v1.py'),
       ('cache_helper_sha256','cuda_cache_reclaim_v1.py'),('resource_helper_sha256','resource_watchdog_v1.py'),('frozen_worker_sha256','train_lora_batched.py')]:
        memory[key]=memory[key+'_after']=c.FROZEN[name]
    def validate_cache(value):
        put(memory_path,value)
        return c.cache_evidence(cache_path,io_path,identity['expected_run_id'],1821,memory_path)
    good(validate_cache(memory)['actual_worker_pid']==22,'Actualv3 evidence binds9/4 worker identity and exact hook')
    for mutation,label in [(lambda p:p.update(cap_bytes=13_000_000_000),'Old cap rejected for new child'),
       (lambda p:p.update(kind='memory_guard_training_launcher_v2'),'Old wrapper kind rejected'),
       (lambda p:p.update(worker_arguments_sha256='bad'),'Actual worker argument digest mismatch'),
       (lambda p:p['owned_update_memory_telemetry'].update(eligible_update_calls=0),'Telemetry hook count mismatch'),
       (lambda p:p['owned_update_memory_telemetry']['samples'][0]['worker_private_memory'].update(pid=11),'Actual worker vs launcher telemetry mismatch')]:
        corrupt=copy.deepcopy(memory);mutation(corrupt);reject(lambda:validate_cache(corrupt),label)
    with patch.object(c,'validate_memory_policy',return_value={'GPU_validation_pending':True}):
        preflight=c.validate_entry(args)
    good(preflight['passed'] and preflight['memory_policy_validation']['GPU_validation_pending'],'Actual receipt/fork/prior proof CPU preflight excluding pending GPU policy')

report={'schema':1,'passed':True,'source_sha256':c.sha(c.__file__),'CPU_only':True,'GPU_launch_permitted_by_this_report':False,
    'actual_GPU_policy_required_before_launch':True,'checks':checks,'rejected_corruptions':rejections,
    'actual_receipt_sha256':c.sha(args.prior_exit_receipt),'actual_fork_manifest_sha256':c.sha(OUTPUT/'runs'/args.loop_id/'fork_snapshot/fork_manifest.json')}
put(OUTPUT/'continuation_controller_v3_cpu_verification.json',report)
print(json.dumps({'passed':True,'positive':len(checks),'rejections':len(rejections),'GPU_used':False,'source_sha256':report['source_sha256']}))
