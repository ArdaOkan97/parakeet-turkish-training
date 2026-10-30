"""Fake-Trackio checks; never starts a mirror, dashboard, GPU or database."""
import copy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/parakeet_turkish'
sys.path.insert(0,str(OUT))
events=[]
fake=SimpleNamespace(init=lambda **kwargs:events.append(('init',kwargs)),
                     log=lambda values,step:events.append(('log',step,values)),
                     finish=lambda:events.append(('finish',)),Markdown=lambda x:x)
sys.modules['trackio']=fake
import track_continuation_v1 as module
old=OUT/'runs/adaptive_20261010_recovery_b8_01_b8_quality'
history=json.loads((old/'training_history.json').read_text(encoding='utf-8'))
config=json.loads((old/'run_config.json').read_text(encoding='utf-8'))
raw_sha=hashlib.sha256((OUT/'track_experiment.py').read_bytes()).hexdigest()
checks=0
def require(ok,message):
 global checks
 checks+=1
 if not ok:raise AssertionError(message)

exact=copy.deepcopy(config);exact['warm_start']={'mode':'exact_resume','source_step':907}
context=module.resume_context(exact,history)
require(context['audio_seen_seconds']==95366.82800000001,'actual inherited907 audio')
require(context['train_wall_seconds']==2900.399432999955,'actual inherited907 trainwall')
suffix=dict(history[-1]);suffix.update(step=908,elapsed_seconds=1.,process_wall_seconds=1.,
 audio_seen_seconds=context['audio_seen_seconds']+120.,train_wall_seconds=context['train_wall_seconds']+2.,
 audio_minutes_per_wall_minute=(context['audio_seen_seconds']+120.)/(context['train_wall_seconds']+2.))
values=module.values_for(suffix,context)
require(values['resources/post_resume_audio_minutes_per_wall_minute']==60.,'suffix interval rate')
require(values['resources/audio_minutes_per_wall_minute']==suffix['audio_minutes_per_wall_minute'],'raw cumulative metric preserved')
require(values['data/post_resume_completed_updates']==1,'newupdate counter')
require(values['resources/post_resume_train_wall_seconds']==2.,'newwall counter')
require(values['data/post_resume_audio_seen_seconds']==120.,'newaudio counter')
require(not any('post_resume' in x for x in module.values_for(history[-1],context)),'inherited907 must not acquire new interval metrics')
require(not any('post_resume' in x for x in module.values_for(history[800],context)),'inherited801 must not acquire new interval metrics')
fresh=copy.deepcopy(config);fresh['warm_start']={'mode':'new_optimizer_warm_start','source_step':801}
require(module.resume_context(fresh,history) is None,'fresh optimizer must not subtract source weight step')
require(not any('post_resume' in x for x in module.values_for(suffix,None)),'fresh suffix metric must be absent')
require(module.resume_context({'warm_start':{'mode':'exact_resume','source_step':0}},[])['train_wall_seconds']==0,'exact0 counters')
require(module.resume_context(exact,[])['pending'] is True,'missingbaseline waits without false speed')
for bad_step in [-1,907.0,True,None]:
 try:module.resume_context({'warm_start':{'mode':'exact_resume','source_step':bad_step}},history)
 except ValueError:rejected=True
 else:rejected=False
 require(rejected,'badexact source step accepted')
for field,value in [('train_wall_seconds',context['train_wall_seconds']),('audio_seen_seconds',float('nan'))]:
 bad=dict(suffix);bad[field]=value
 try:module.values_for(bad,context)
 except ValueError:rejected=True
 else:rejected=False
 require(rejected,'invalid suffix interval accepted')

original_read=module.read_json
module.read_json=lambda *args:(_ for _ in ()).throw(AssertionError('Per-row file I/O'))
rows=list(module.cycle_rows(exact,[*history,suffix],{801:{'dev/wer_tr_percent':69.090909}}))
require(len(rows)==908 and rows[-1][1]['resources/post_resume_audio_minutes_per_wall_minute']==60.,'cycle pure metrics')
require(rows[800][1]['dev/wer_tr_percent']==69.090909,'batchcaptured DEV metric applied')
module.read_json=original_read

with tempfile.TemporaryDirectory(prefix='trackio-wrapper-cpu-',dir=ROOT/'work') as folder:
 p=Path(folder);module.OUTPUT=p;module.ROOT=p
 cfg={'run_id':'cpu_fake_only','warm_start':{'mode':'exact_resume','source_step':907},'steps':9611}
 (p/'run_config.json').write_text(json.dumps(cfg),encoding='utf-8')
 (p/'training_history.json').write_text(json.dumps([*history,suffix]),encoding='utf-8')
 state_path=p/'state.json';state_path.write_text('{}',encoding='utf-8')
 read_counts={}
 def counted(path,default=None):
  read_counts[str(path)]=read_counts.get(str(path),0)+1
  return original_read(path,default)
 module.read_json=counted
 module.follow(SimpleNamespace(state_path=state_path,project='cpu_fake_no_db',follow=False))
 require(read_counts[str(p/'run_config.json')]==1,'config read more than once percycle')
 require(read_counts[str(p/'training_history.json')]==1,'history read more than once percycle')
 for step in [0,801,907]:require(read_counts[str(p/f'results/cpu_fake_only_step_{step}/summary.json')]==1,'DEV summary per-row re-read')
 require(sum(e[0]=='init' for e in events)==1,'unexpected multiple dashboard run initializations')
 require(any(e[0]=='log' and e[1]==908 and e[2].get('resources/post_resume_audio_minutes_per_wall_minute')==60. for e in events),'suffix not emitted to fake Trackio')
 require(json.loads(state_path.read_text(encoding='utf-8'))['cpu_fake_only']['908']['resources/post_resume_audio_minutes_per_wall_minute']==60.,'durable mirror bookkeeping lacks suffix')
module.read_json=original_read
require(hashlib.sha256((OUT/'track_experiment.py').read_bytes()).hexdigest()==raw_sha,'original mirror source changed')
require(not events or fake is sys.modules['trackio'],'real Trackio imported')
report={'schema':1,'passed':True,'status':'complete','checks':checks,'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'wrapper_sha256':hashlib.sha256((OUT/'track_continuation_v1.py').read_bytes()).hexdigest(),
        'original_mirror_sha256':raw_sha,'actual907_baseline':context,
        'synthetic_new908_expected_audio_min_per_wall_min':60.,'raw_cumulative_preserved':True,
        'config_history_reads_per_cycle':1,'per_row_file_reads':0,
        'fresh_optimizer_source_step_not_subtracted':True,'real_trackio_or_gpu_or_mirror_started':False,
        'scope':'Actual inherited907 counters with synthetic appended update; fake Trackio singlecycle only. No production monitoring/database changes.'}
(OUT/'trackio_continuation_cpu_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:report[k] for k in ['passed','checks','wrapper_sha256','original_mirror_sha256','real_trackio_or_gpu_or_mirror_started']},indent=2))
