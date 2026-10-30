"""NEW metadata-only controller proof; all clock/process/counter inputs are METHOD_STUB.
Actual owned E scalar files are exercised; no Win32/helper/profile/toy/training operation.
"""
import argparse,hashlib,json,os,sys,time,traceback
from pathlib import Path
import psutil
from fixed_counter_window_and_scalar_files_v1 import FixedCounterWindow,BoundedScalarFiles
F=Path(__file__).parent;OUT=F/'fixed_counter_controller_actual_01'
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def write(p,v):
 with Path(p).open('x',encoding='utf-8',newline='\n')as f:json.dump(v,f,indent=2);f.write('\n')
def require(v,m):
 if not v:raise ValueError(m)
def who(p):return {'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()}
def main(a):
 psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
 review=json.loads(Path(a.ROOT_review).read_text(encoding='utf-8-sig'));require(review['accepted']is True and review['reviewer']=='/root'and review['scope']=='ASR18_fixed60_counter_window_persistence_METHOD_STUB_CPU_v1','Exact separate controller-only ROOT scope')
 require(review['fixture_source']==pin(__file__)and review['output_path']==str(OUT.resolve())and all(pin(p['path'])==p for p in review['input_pins']),'Frozen exact inputs/output')
 require(os.environ['CUDA_VISIBLE_DEVICES']=='-1'and os.environ['PYTHONUTF8']=='1'and all(os.environ[k]=='1'for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')),'CPU-only process-local env')
 require(not OUT.exists(),'One fresh metadata-only proof');OUT.mkdir();cases=[];error=None;files=None;report={};began=time.monotonic()
 def case(name,test):
  require(test,name);cases.append({'case':name,'passed':True,'mode':'METHOD_STUB_clock_process_counter_values'})
 def negative(name,fn):
  try:fn()
  except ValueError as e:cases.append({'case':name,'passed':True,'mode':'METHOD_STUB','actual_exception':str(e)});return
  raise AssertionError(name)
 try:
  c=FixedCounterWindow(10.)
  for k in range(600):
   now=10.+k*.1;case_phase=c.phase(now,None);require(case_phase=='counter_window'and c.due(now),'All600 normal fake clock polls');c.recorded(now)
  case('fixed60_ends_counter_not_nominal_profiler',c.phase(70.,None)=='wait_profiler_only'and c.polls==600 and not c.due(70.))
  case('late_profiler90_does_not_add_counterpolls',c.phase(100.,0)=='done'and c.polls==600)
  c=FixedCounterWindow(100.);case('early_profiler_exit_does_not_shorten_counter_window',c.phase(150.,0)=='counter_window'and c.phase(160.,0)=='done')
  c=FixedCounterWindow(0.);c.phase(10.,None);negative('backward_clock_rejected',lambda:c.phase(9.,None));negative('nonfinite_clock_rejected',lambda:FixedCounterWindow(float('nan')))
  c=FixedCounterWindow(0.);negative('nonzero_profiler_exit_not_success',lambda:c.phase(1.,1))
  c=FixedCounterWindow(0.);c.phase(60.,None);case('hang_stays_guarded_wait_not_more_queries',all(c.phase(x,None)=='wait_profiler_only'for x in (70.,90.,119.)))
  c=FixedCounterWindow(0.);c.recorded(0.);negative('poll_before_due_rejected',lambda:c.recorded(.01))
  folder=OUT/'failure_snapshot';folder.mkdir();files=BoundedScalarFiles(folder)
  vector={'attempted_poll':1,'poll_limit':700,'current_TID_count':3,'unique_TID_count':2,'max_threads':128,'monotonic_ns':100}
  files.append('preconditions',[vector]);sentinel=ValueError('METHOD_STUB original sampler rejection')
  try:raise sentinel
  except ValueError as e:case('fake_failure_preserves_exception_identity',e is sentinel)
  files.append('counters',[{'poll':0,'native_tid':42,'creation_filetime_100ns':134372300000000000,'user_100ns':10,'kernel_100ns':20}]);files.append('brackets',[{'poll':0,'start_monotonic_ns':90,'end_monotonic_ns':99}]);files.close()
  case('actual_E_precondition_flushed_before_fake_assertion',json.loads((folder/'counter_poll_preconditions.jsonl').read_text())==vector)
  case('actual_partial_scalar_files_survive_fake_failure',all((folder/n).stat().st_size>0 for n,cap in files.NAMES.values())and all(f.closed for f in files.handles.values()))
  negative('closed_scalar_writer_rejected',lambda:files.append('preconditions',[vector]));negative('double_close_rejected',files.close)
  second=OUT/'partial_constructor';second.mkdir();(second/'thread_CPU_counters.jsonl').write_bytes(b'existing immutable sentinel')
  baseline=BoundedScalarFiles.opened-BoundedScalarFiles.closed_count
  try:BoundedScalarFiles(second)
  except FileExistsError:
   require(BoundedScalarFiles.opened-BoundedScalarFiles.closed_count==baseline,'Actual partial constructor closes every successful opened file')
   cases.append({'case':'actual_partial_constructor_closes_opened_scalar_file','passed':True,'mode':'real_owned_E_file_constructor_failure'})
  else:raise AssertionError('Fresh files mandatory')
  with(second/'counter_poll_preconditions.jsonl').open('ab')as f:f.write(b'')
  require((second/'thread_CPU_counters.jsonl').read_bytes()==b'existing immutable sentinel','Existing owned sentinel preserved')
  report['passed']=True
 except BaseException as e:error={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}
 finally:
  if files is not None and not files.closed:files.close()
  report.update({'kind':'ASR18_fixed_counter_window_METHOD_STUB_actual_CPU_report_v1','passed':report.get('passed')is True and error is None,'error':error,'cases':cases,'fixture_identity':who(psutil.Process()),'fixture_launcher_identity':who(psutil.Process().parent()),'all_input_pins_unchanged':all(pin(p['path'])==p for p in review['input_pins']),'ROOT_review':pin(a.ROOT_review),'clock_profiler_process_counter_inputs_are_METHOD_STUB':True,'actual_win32_thread_profile_or_training_operation':False,'no_helper_import_extra_GC_GPU_model_audio_or_target_probe':True,'owned_scalar_files_opened':BoundedScalarFiles.opened,'owned_scalar_files_closed':BoundedScalarFiles.closed_count,'all_owned_scalar_files_closed':BoundedScalarFiles.opened==BoundedScalarFiles.closed_count,'elapsed_seconds':time.monotonic()-began})
  write(OUT/'report.json',report)
 print(json.dumps({'report':pin(OUT/'report.json'),'passed':report['passed'],'error':error}),flush=True)
 return 0 if report['passed']else 1
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--ROOT-review',required=True);sys.exit(main(p.parse_args()))
