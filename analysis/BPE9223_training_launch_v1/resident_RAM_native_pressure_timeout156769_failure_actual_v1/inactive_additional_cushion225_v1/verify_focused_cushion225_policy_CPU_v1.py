"""INACTIVE focused CPU policy-delta proof. Callbacks, samples and clock are synthetic.
No original Monitor thread/transport/audio/model/NumPy/Torch/GPU execution is claimed.
Requires a separate exact ROOT fixture scope before its single actual invocation.
"""
import argparse,ast,copy,hashlib,importlib.util,json,os,shutil,sys,time
from pathlib import Path
import psutil
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from resource_watchdog_v2 import sample_memory
F=Path(__file__).parent;N=F.parents[1]
OLD_GATE=N/'resident_RAM_system_headroom156300_failure_actual_v1/asr18_native_update_commit_pressure_gate_v2.py'
GATE=F/'asr18_native_update_commit_pressure_gate_v3.py'
OLD_CONFIG=N/'resident_RAM_native_pressure_timeout156537_failure_actual_v1/inactive_packet_credit_v1/production_configuration_v1/configuration.json'
began=time.monotonic();peak=0
def require(v,m):
 if not v:raise ValueError(m)
def pin(p):
 p=Path(p);raw=p.read_bytes();return {'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 with Path(p).open('x',encoding='utf-8')as f:json.dump(v,f,indent=2);f.write('\n')
def guard():
 global peak
 s=sample_memory(os.getpid(),'focused_cushion225_policy_CPU_delta');m=s['system_memory'];peak=max(peak,s['worker_private_memory']['private_bytes'])
 require(s['paired_sampling_within_one_second']and peak<512*1024**2
  and m['commit_available_bytes']>=2000000000+268435456 and m['physical_available_bytes']>=1500000000+268435456
  and shutil.disk_usage('E:/').free>=100000000000 and time.monotonic()-began<180,'Original finite512MiB/180s/full256MiB/headroom/E100GB guards')
 return s
class Clock:
 def __init__(self):self.now=0.
 def clock(self):return self.now
 def sleep(self,value):
  require(0<value<=.5,'Original bounded sleep');self.now+=value
def sample(commit,physical):return {'system_memory':{'commit_available_bytes':commit,'physical_available_bytes':physical},'synthetic_sample':True}
def main(a):
 psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
 require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'and all(os.environ.get(k)=='1'for k in
  ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')),'CPU-only thread caps')
 require(Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad')),'E storage applied')
 initial=guard();m=initial['system_memory']
 require(m['commit_available_bytes']>=2000000000+268435456+512*1024**2 and m['physical_available_bytes']>=1500000000+268435456+512*1024**2,'Full512MiB prospective allowance')
 scope=read(a.ROOT_review);configuration=pin(a.configuration)
 require(scope['accepted']is True and scope['reviewer']=='/root'
  and scope['scope']=='ASR18_focused_native_commit_cushion2250000000_CPU_delta_v1'
  and scope['fixture_source']==pin(__file__)and scope['source_configuration']==configuration
  and scope['gate_source']==pin(GATE)and scope['old_gate_source']==pin(OLD_GATE),'Exact separate ROOT CPU fixture scope')
 for item in scope['input_pins']:
  require(pin(item['path'])==item,'Exact CPU scope input pin')
 output=Path(a.output).resolve();require(output.is_relative_to(F)and not output.exists(),'ONE fresh E fixture output');output.mkdir()
 me=psutil.Process();parent=me.parent();identities={'producer':{'pid':me.pid,'create_time':me.create_time(),'ppid':me.ppid()},'launcher':{'pid':parent.pid,'create_time':parent.create_time(),'ppid':parent.ppid()}}
 write(output/'actual_process_entry.json',identities)
 old_bytes=OLD_GATE.read_bytes();new_bytes=GATE.read_bytes()
 require(new_bytes.replace(b'2_250_000_000',b'2_500_000_000')==old_bytes and new_bytes.count(b'2_250_000_000')==1,'Full BYTE inverse of only helper numeric literal')
 spec=importlib.util.spec_from_file_location('ASR18_focused_candidate_cushion225',GATE);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 policy=module.POLICY;required=policy['native_actual_commit_headroom_bytes']+policy['full_additional_prefetch_host_reservation_bytes']+policy['additional_native_commit_pressure_buffer_bytes']
 physical=policy['native_actual_physical_headroom_bytes']+policy['full_additional_prefetch_host_reservation_bytes']
 require(required==4518435456 and physical==1768435456 and policy['additional_native_commit_pressure_buffer_bytes']==2250000000,'Exact approved candidate arithmetic')
 cases=[]
 def run_case(name,commit,available_physical,should_admit):
  clock=Clock();value=sample(commit,available_physical)
  try:
   result=module.wait_for_native_update_commit_pressure(observe=lambda:value,check=lambda:None,policy=policy,clock=clock.clock,sleep=clock.sleep)
  except TimeoutError as error:
   require(not should_admit and clock.now==300 and error.native_pressure_denial['last_denied_sample']is value
    and error.native_pressure_denial['required_commit_available_bytes']==required
    and error.native_pressure_denial['required_physical_available_bytes']==physical
    and error.native_pressure_denial['observed_wait_seconds']==300,'Exact fake-clock timeout')
   result={'admitted':False,'fake_elapsed':clock.now,'observations':error.native_pressure_denial['observations']}
  else:
   require(should_admit and result['wait_seconds']==0 and result['observations']==1,'Exact immediate boundary admission')
   result={'admitted':True,'fake_elapsed':clock.now}
  cases.append({'name':name,'synthetic_commit':commit,'synthetic_physical':available_physical,'result':result})
 for delta in (-1,0,1):run_case('commit_boundary_'+str(delta),required+delta,physical,delta>=0)
 for delta in (-1,0,1):run_case('physical_boundary_'+str(delta),required,physical+delta,delta>=0)
 for name,value in [('historical_source22_last',4676124672),('historical_source24_last',4669493248)]:
  require(value<4768435456,'Historical2.5GB policy would deny this point')
  run_case(name+'_SYNTHETIC_REPLAY',value,11246845952,True)
 def same_error(name,observe,check,sentinel):
  clock=Clock()
  try:module.wait_for_native_update_commit_pressure(observe=observe,check=check,policy=policy,clock=clock.clock,sleep=clock.sleep)
  except BaseException as error:require(error is sentinel,'Original callback exception object identity')
  else:raise ValueError('Expected original callback exception')
  cases.append({'name':name,'exact_original_callback_exception_identity':True,'method_stub_only':True})
 sentinel=ValueError('Sensible real system headroom')
 def denied_observe():raise sentinel
 same_error('observe_floor_error_identity_STUB',denied_observe,lambda:None,sentinel)
 sentinel=RuntimeError('Cooperative stop')
 def stopped_check():raise sentinel
 same_error('first_check_stop_identity_STUB',lambda:sample(required,physical),stopped_check,sentinel)
 sentinel=RuntimeError('Cooperative stop after admitted sample');checks=[0]
 def second_check():
  checks[0]+=1
  if checks[0]==2:raise sentinel
 same_error('second_check_stop_identity_STUB',lambda:sample(required,physical),second_check,sentinel)
 wrong=copy.deepcopy(policy);wrong['additional_native_commit_pressure_buffer_bytes']=2000000000
 try:module.wait_for_native_update_commit_pressure(observe=lambda:sample(required,physical),check=lambda:None,policy=wrong)
 except ValueError:pass
 else:raise ValueError('Wrong candidate policy must be rejected')
 cases.append({'name':'different_policy_rejected','passed':True})
 old=read(OLD_CONFIG);candidate=read(a.configuration);restored=copy.deepcopy(candidate)
 del restored['prior_resident_RAM_native_pressure156769_failure'];restored['native_update_commit_pressure_policy']=old['native_update_commit_pressure_policy']
 require(restored==old,'Every old source24 config field exact after2operational fields')
 require(candidate['native_update_commit_pressure_policy']['actual_CPU_control_review']is None,'Inactive placeholder cannot be a completed production gate')
 require(candidate['resident_RAM_resource_scheduling_policy']['CPU_prospective_commit_reserve_bytes']==2500000000
  and candidate['resident_RAM_packet_credit_policy']['maximum_logical_packets']==1
  and candidate['resident_RAM_resource_scheduling_policy']['maximum_packet_bytes']==268435456
  and candidate['resident_RAM_resource_scheduling_policy']['queue_capacity_bytes']==268435456,'CPU reserve/fullpacket/queue/credit unchanged')
 def monitor_ast(path):
  tree=ast.parse(Path(path).read_text(encoding='utf-8-sig'))
  return ast.dump(next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='Monitor'),include_attributes=False)
 require(monitor_ast(P/'work/research/run_ASR18_BPE9223_resident_RAM_training_v24.py')==monitor_ast(P/'work/research/run_ASR18_BPE9223_resident_RAM_training_v25.py'),'Original Monitor AST unchanged, not executed')
 old_review=N/'ROOT_native_preupdate_commit_pressure_actual_CPU_review_v1.json'
 require(pin(old_review)['sha256']=='2e63113a9d42e0337587384ab11b685c0749f47e20c0502640e03ca83b64af1a','Closed completed genuine Monitor review reused')
 require('numpy'not in sys.modules and 'torch'not in sys.modules,'No NumPy/Torch/model/GPU imports')
 for item in scope['input_pins']:require(pin(item['path'])==item,'All CPU inputs unchanged after actual execution')
 report={'kind':'ASR18_actual_focused_native_commit_cushion2250000000_CPU_delta_v1','passed':True,
  'focused_policy_delta_controls_passed':True,'fake_clock_and_historical_samples_explicit':True,'cases':cases,'case_count':len(cases),
  'actual_native_optimizer_updates':0,'actual_sampler_draws':0,'actual_audio_decode':0,'actual_GPU_activity':False,
  'actual_original_Monitor_thread_executed':False,'original_Monitor_AST_unchanged':True,
  'old_genuine_Monitor_review_reused_without_execution':True,'original_genuine_Monitor_completed_review':pin(old_review),
  'gate_source':pin(GATE),'old_gate_source':pin(OLD_GATE),'policy':policy,'source_configuration':configuration,
  'full_old_source24_config_inverse_exact':True,'only_two_operational_config_fields':True,'all_original_reservations_and_CPU_credit_policies_exact':True,
  'required_commit_available_bytes':required,'required_physical_available_bytes':physical,
  'no_fit_speed_allocation_bound_or_prevention_claim':True,'synthetic_callbacks_are_not_actual_resource_pressure':True,
  'identities':identities,'guard_initial':initial,'guard_final':guard(),'peak_private_bytes':peak,'input_pins_before_after_exact':scope['input_pins']}
 write(output/'report.json',report);print(json.dumps({'report':pin(output/'report.json'),'identities':identities,'case_count':len(cases)}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--ROOT-review',required=True);p.add_argument('--configuration',required=True);p.add_argument('--output',required=True);main(p.parse_args())

