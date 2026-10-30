"""Local Trackio mirror of genuine controller preparation observations.

Use monitor-env Python. No model, CUDA, training source edits, or fabricated loss/
WER/optimizer steps. Preparation runs have their own observation-step axis and
hand off to the existing training mirror after real optimizer records appear.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

OUTPUT=Path(__file__).resolve().parent;ROOT=OUTPUT.parent.parent
PHASE_CODES={'worker_startup':0,'actual_training_input_verification':1,
 'teacher_acoustic_cache_preparation':2,'development_evaluation':3,
 'development_complete_or_checkpoint':3,'training_ready':4,'training_updates':5,'final_checkpoint_or_exit':6}
TERMINAL={'resource_review_required','failed_review_required','training_window_complete_objective_incomplete'}


def read(path,default):
 try:return json.loads(path.read_text(encoding='utf-8'))
 except (OSError,ValueError):return default
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def finite(value):return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)
def instant(value):
 result=datetime.fromisoformat(value)
 if result.tzinfo is None:raise ValueError('Timezone-aware observation required')
 return result


def metrics_from_state(state,observed_at=None):
 """Pure read-only derivation; counts are stream-local and never training steps."""
 observed_at=observed_at or datetime.now(timezone.utc)
 run_id=state.get('active_run_id')
 trials=[x for x in state.get('trials',[]) if x.get('run_id')==run_id]
 if not run_id or len(trials)!=1:return None
 trial=trials[0];phase=trial.get('phase',{'phase':'worker_startup'})
 stamp=trial.get('latest_observation_at_utc',trial.get('started_at_utc'))
 if not stamp:return None
 age=(observed_at-instant(stamp)).total_seconds()
 if not -1<=age<=30:return None
 elapsed=max(0.,(observed_at-instant(trial['started_at_utc'])).total_seconds())
 published=trial.get('published_steps',0)
 if not isinstance(published,int) or published<0:raise ValueError('Published counter must be a real nonnegative integer')
 values={'prep/elapsed_wall_seconds':elapsed,'prep/observation_age_seconds':max(0,age),
   'prep/phase_code':PHASE_CODES.get(phase.get('phase'),0),'prep/published_optimizer_updates':published,
   'prep/handoff_to_training_mirror':int(published>0)}
 for processed,total,prefix in [('cache_processed','cache_stream_total','teacher_cache_stream'),
   ('inputs_processed','inputs_total','actual_input_verification')]:
  if processed in phase and total in phase:
   count,size=phase[processed],phase[total]
   if not type(count) is int or not type(size) is int or not 0<=count<=size or size<=0:raise ValueError('Invalid real preparation counts')
   values['prep/'+prefix+'_processed']=count;values['prep/'+prefix+'_total']=size
   values['prep/'+prefix+'_fraction']=count/size
 memory=trial.get('system_and_owned_private_memory',{}).get('system_memory',{})
 if memory.get('available'):
  for name in ('commit_used_bytes','commit_limit_bytes','commit_available_bytes'):
   if finite(memory.get(name)):values['prep/system_'+name.replace('_bytes','_GB')]=memory[name]/1e9
 return {'target_run_id':run_id,'name':'preparation__'+run_id,'source_observed_at_utc':stamp,
   'source_phase':phase,'metrics':values,'training_started':published>0}


def main(args):
 loop=args.loop_dir.resolve()
 if not loop.is_relative_to((OUTPUT/'runs').resolve()) or not math.isfinite(args.max_wall_seconds) or args.max_wall_seconds<=0:
  raise ValueError('Scoped loop and finite positive lifetime required')
 if not math.isfinite(args.poll_seconds) or args.poll_seconds<5:raise ValueError('At least5-second polls required')
 os.environ['TRACKIO_DIR']=str(OUTPUT/'trackio');os.environ['HF_HOME']=str(ROOT/'work/monitor-hf-cache')
 os.environ['HF_HUB_DISABLE_TELEMETRY']='1';os.environ['GRADIO_ANALYTICS_ENABLED']='False'
 for key in ('TRACKIO_SPACE_ID','TRACKIO_SERVER_URL','TRACKIO_DATASET_ID','TRACKIO_BUCKET_ID','TRACKIO_WEBHOOK_URL'):
  os.environ.pop(key,None)
 import trackio
 end=time.monotonic()+args.max_wall_seconds;active=None;seen=set();steps={};receipt=[]
 output=ROOT/'work'/('trackio_preparation_v1_'+loop.name+'.json')
 try:
  while time.monotonic()<end:
   status=read(loop/'loop_status.json',{});observation=metrics_from_state(status)
   if observation and observation['name'] not in seen:
    name=observation['name']
    if active!=name:
     if active:trackio.finish()
     trackio.init(project=args.project,name=name,group='Preparation before optimizer records',resume='allow',embed=False,
       auto_log_gpu=False,auto_log_cpu=False,
       config={'loop_id':loop.name,'target_training_run_id':observation['target_run_id'],
         'source_status_path':str(loop/'loop_status.json'),'phase_codes':PHASE_CODES,
         'step_axis':'Preparation observation index, not optimizer updates',
         'cache_count_scope':'Actual printed cache stream-localN/total; may reset between Turkish/replay streams',
         'no_training_loss_WER_or_fabricated_optimizer_steps':True})
     active=name;steps.setdefault(name,0)
    steps[name]+=1;trackio.log(observation['metrics'],step=steps[name])
    receipt.append({'name':name,'observation_step':steps[name],'source_observed_at_utc':observation['source_observed_at_utc'],
      'source_phase':observation['source_phase'],'metrics':observation['metrics']})
    if len(receipt)>128:receipt=receipt[-128:]
    output.write_text(json.dumps({'schema':1,'kind':'local_Trackio_preparation_mirror_v1','loop_id':loop.name,
      'source_code_sha256':sha(Path(__file__)),'project':args.project,'latest_actual_status_sha256':sha(loop/'loop_status.json'),
      'observations_logged':sum(steps.values()),'preparation_run_steps':steps,'latest_bounded_observations':receipt,
      'no_model_or_CUDA_imported':True,'no_training_sources_or_status_changed':True},indent=2),encoding='utf-8')
    if observation['training_started']:
     seen.add(name);trackio.finish();active=None
   if status.get('status') in TERMINAL or args.once:return 0
   time.sleep(args.poll_seconds)
  return 0
 finally:
  if active:trackio.finish()


if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--loop-dir',type=Path,required=True)
 p.add_argument('--project',default='parakeet-turkish')
 p.add_argument('--max-wall-seconds',type=float,default=11100.)
 p.add_argument('--poll-seconds',type=float,default=5.)
 p.add_argument('--once',action='store_true')
 raise SystemExit(main(p.parse_args()))
