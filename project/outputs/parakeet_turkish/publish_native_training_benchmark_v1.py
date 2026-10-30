"""Idempotent local Trackio import of two actual matched TRAIN loss pilots.

Use monitor-env Python. This imports only two new explicitly named training
benchmark runs; existing quality/benchmark runs and their metrics are untouched.
No model, CUDA, inference or automatic GPU/CPU logger is used.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
NAMES={'custom':'training_benchmark__custom_loss_v1',
       'native':'training_benchmark__native_NeMo_loss_v1'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def require(ok,message):
 if not ok:raise ValueError(message)


def existing(database,name):
 """Read-only evidence/idempotence check; all writes go through Trackio API."""
 if not database.exists():return {},set()
 with sqlite3.connect(database.as_uri()+'?mode=ro',uri=True,timeout=10) as db:
  configs=[json.loads(row[0]) for row in db.execute('SELECT config FROM configs WHERE run_name=?',(name,))]
  rows=db.execute('SELECT step,metrics FROM metrics WHERE run_name=? ORDER BY step',(name,)).fetchall()
 require(len(configs)<=1,'Ambiguous existing benchmark name; preserve it and choose a new version')
 result={}
 for step,raw in rows:
  require(step not in result,'Duplicate existing benchmark step requires review')
  result[step]=json.loads(raw)
 return (configs[0] if configs else {}),result


def metric_rows(report):
 wall=tr=replay=0.;result=[]
 for expected,row in enumerate(report['updates'],1):
  require(row['update']==expected and len(row['source_ids'])==16,'Matched effective16 training order required')
  seconds=row['wall_seconds'];require(math.isfinite(seconds) and seconds>0,'Measured positive update time required')
  wall+=seconds;tr+=row['turkish_audio_seconds'];replay+=row['replay_audio_seconds']
  value={'benchmark/seconds_per_update':seconds,
    'benchmark/combined_audio_minutes_per_wall_minute':(tr+replay)/wall,
    'benchmark/turkish_audio_minutes_per_wall_minute':tr/wall,
    'benchmark/measured_train_wall_seconds':wall,
    'benchmark/turkish_audio_seen_seconds':tr,'benchmark/replay_audio_seen_seconds':replay,
    'benchmark/updates_per_second':expected/wall,
    'benchmark/loss':row['weighted_loss'],'benchmark/gradient_norm':row['gradient_norm'],
    'benchmark/tdt_loss':row['components'][0],
    'benchmark/current_allocated_vram_GB':row['allocated_bytes']/1e9,
    'benchmark/current_reserved_vram_GB':row['reserved_bytes']/1e9}
  require(all(math.isfinite(x) for x in value.values()),'Nonfinite benchmark metric')
  if expected==len(report['updates']):
   value.update({'benchmark/cold_first_call_seconds':report['cold_start']['wall_seconds_including_capture_and_JIT'],
     'benchmark/peak_allocated_vram_GB':report['memory']['peak_allocated_bytes']/1e9,
     'benchmark/peak_reserved_vram_GB':report['memory']['peak_reserved_bytes']/1e9,
     'benchmark/max_system_commit_percent':100*report['memory']['system_max_commit_fraction']})
  result.append((expected,value))
 return result


def main(args):
 require(args.project=='parakeet-turkish','Use the existing local experiment project only')
 comparison=read(args.comparison)
 require(comparison['kind']=='matched_native_training_comparison_v1' and comparison['status']=='complete'
   and comparison['common_measured_updates']==20 and comparison['heldout_used'] is False,'Actual complete matched20 TRAIN comparison required')
 reports={}
 for arm,key in (('custom','custom_report'),('native','native_report')):
  item=comparison[key];path=Path(item['path']).resolve()
  require(path.is_relative_to(HERE) and sha(path)==item['sha256'],'Exact scoped measured report required')
  report=read(path)
  require(report['arm']==arm and report['status']=='complete' and len(report['updates'])==report['requested_updates']==20
    and report['frozen_sources_unchanged'] is True and report['heldout_used'] is False,'Complete20 actual TRAIN updates required')
  reports[arm]=(path,report,metric_rows(report))
 require(reports['custom'][1]['identity']==reports['native'][1]['identity']
   and reports['custom'][1]['reset']==reports['native'][1]['reset'],'Actual matched inputs/source/fresh Adam/RNG differs')
 os.environ['TRACKIO_DIR']=str(HERE/'trackio');os.environ['HF_HOME']=str(ROOT/'work/monitor-hf-cache')
 os.environ['HF_HUB_DISABLE_TELEMETRY']='1';os.environ['GRADIO_ANALYTICS_ENABLED']='False'
 for key in ('TRACKIO_SPACE_ID','TRACKIO_SERVER_URL','TRACKIO_DATASET_ID','TRACKIO_BUCKET_ID','TRACKIO_WEBHOOK_URL'):
  os.environ.pop(key,None)
 import trackio
 database=(HERE/'trackio'/(args.project+'.db')).resolve();receipt=[]
 for arm,(path,report,rows) in reports.items():
  name=NAMES[arm];config,prior=existing(database,name)
  require(not config or config.get('source_report_sha256')==sha(path),'Existing benchmark source report differs; no overwrite')
  for step,values in rows:
   require(step not in prior or prior[step]==values,'Existing benchmark metrics differ; no overwrite')
  missing=[(step,values) for step,values in rows if step not in prior]
  if missing:
   trackio.init(project=args.project,name=name,group='Matched TRAIN loss benchmark',resume='allow',embed=False,
    auto_log_gpu=False,auto_log_cpu=False,
    config={'kind':'matched_TRAIN_loss_training_benchmark_v1','arm':arm,'source_report_path':str(path),
     'source_report_sha256':sha(path),'comparison_path':str(args.comparison.resolve()),'comparison_sha256':sha(args.comparison),
     'source_step':1871,'physical_batch':8,'accumulation':2,'effective_batch':16,'measured_updates':20,
     'inference_benchmark':False,'DEV_or_TEST_used':False,'quality_trial':False,'native_full_trainer_equivalence_claimed':False,
     'timing_scope':report['timing_scope'],'cold_JIT_and_warmup_excluded':True,
     'native_loss_only_scope':'Official NeMo loss kernel with existing HF projections, sigma0/omega0/current clip normalization',
     'pilot_speed_ratio':comparison['native_over_custom_training_speed_ratio'],
     'pilot_gain_interpretation':'Short20-update pilot; tiny speed difference is not a reliable adoption advantage'})
   try:
    for step,values in missing:trackio.log(values,step=step)
   finally:trackio.finish()
  _,readback=existing(database,name)
  require(all(readback.get(step)==values for step,values in rows),'Actual Trackio SQLite readback differs')
  receipt.append({'run_name':name,'arm':arm,'expected_steps':20,'actual_readback_steps':len(readback),
    'newly_logged_steps':len(missing),'source_report_sha256':sha(path)})
 args.output.parent.mkdir(parents=True,exist_ok=True)
 args.output.write_text(json.dumps({'kind':'local_matched_TRAIN_benchmark_publication_v1','passed':True,
  'publisher_sha256':sha(__file__),'comparison_sha256':sha(args.comparison),'project':args.project,
  'runs':receipt,'database_read_only_verification':str(database),'database_direct_write_used':False,
  'only_run_names_logged_by_publisher':list(NAMES.values()),'model_or_CUDA_code_invoked':False},indent=2)+'\n',encoding='utf-8')
 print(json.dumps({'passed':True,'runs':receipt,'receipt':str(args.output)}))


if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--comparison',type=Path,default=HERE/'native_training_pilot_comparison_v1.json')
 p.add_argument('--project',default='parakeet-turkish')
 p.add_argument('--output',type=Path,default=HERE/'native_training_benchmark_Trackio_publication_v1.json')
 main(p.parse_args())
