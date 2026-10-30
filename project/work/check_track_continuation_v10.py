"""Tiny isolated V10 fixtures; no Trackio/state/corpus/Torch/GPU imports."""
import argparse,ast,gc,hashlib,io,json,math,os,re,sys,tempfile,time
from pathlib import Path
from datetime import datetime,timezone
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT/'outputs/parakeet_turkish/track_continuation_v10.py';checks=[]
def check(ok,name):
 if not ok:raise AssertionError(name)
 checks.append(name)
def reject(call,name):
 try:call()
 except (ValueError,KeyError):checks.append(name);return
 raise AssertionError(name)
def read_json(path,default=None):
 try:return json.loads(Path(path).read_text(encoding='utf-8'))
 except (OSError,ValueError):return default
def write_json(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False),encoding='utf-8');os.replace(temp,path)
def extracted(path):
 tree=ast.parse(path.read_text(encoding='utf-8'));nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef,ast.Assign))]
 n=dict(globals(),__file__=str(path),original=SimpleNamespace(ROOT=ROOT,OUTPUT=ROOT/'work',trackio=None,metrics_for=lambda r:{'metric':r['step'],'retention/a':1}))
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),n);n['self_memory']=lambda phase:{'phase':phase,'fixture':'hardware unmeasured'};return n,tree

def main(args):
 n,tree=extracted(SOURCE);v9,t9=extracted(SOURCE.with_name('track_continuation_v9.py'))
 old={x.name:ast.dump(x,include_attributes=False) for x in t9.body if isinstance(x,ast.FunctionDef)};new={x.name:ast.dump(x,include_attributes=False) for x in tree.body if isinstance(x,ast.FunctionDef)}
 names=('finite_number','resume_context','values_for','acquisition_context','acquisition_values_for','continued_acquisition_context','continued_acquisition_values_for','scheduled_row_lr','scheduled_acquisition_context','scheduled_acquisition_values_for','adaptation_context','adaptation_values_for','shared_head_context','shared_head_values_for')
 check(all(old[key]==new[key] for key in names),'all14 prior phase context/math functions AST identical')
 history=[{'step':i,'validation':{} if i==2 else None,'text':'ş { [ " \\','audio_seen_seconds':i*100.,'train_wall_seconds':i*10.} for i in range(1,4)]
 raw=json.dumps(history,ensure_ascii=False)
 check(all(list(n['HistoryArray'](io.StringIO(raw),size).records())==history for size in (1,2,7,64)),'streamed array preserves every nested value/Unicode/escape across boundaries')
 reject(lambda:list(n['HistoryArray'](io.StringIO(raw[:-2]),3).records()),'truncated history rejected')
 reject(lambda:list(n['HistoryArray'](io.StringIO('[{"step":2}]')).records()),'history stepgap rejected')
 reject(lambda:list(n['HistoryArray'](io.StringIO('[{"step":1},]')).records()),'history trailingcomma rejected')
 reject(lambda:list(n['HistoryArray'](io.StringIO(raw),7,max_record_chars=10).records()),'oversized individual record rejected')
 config={'run_id':'fixture','warm_start':{'mode':'exact_resume','source_step':1}}
 check(list(n['cycle_rows'](config,history,{}))==list(v9['cycle_rows'](config,history,{})),'refactored record metrics equal V9 exactinterval/inherited metrics')
 with tempfile.TemporaryDirectory(prefix='v10_fixture_',dir=ROOT/'work') as tmp:
  tmp=Path(tmp);path=tmp/'history.json';write_json(path,history)
  cache=n['HistoryScanCache'](tmp);info,changed=cache.scan(path,config)
  check(changed and info['retained_history_record_count']==1 and info['last_step']==3,'firstpass retains one needed baseline rather than full list')
  check(list(cache.rows(path,info))==history,'secondpass streams exact same canonical records')
  cached,changed=cache.scan(path,config);check(not changed and cached is info,'unchangedmtime reuses only bounded contexts')
  write_json(path,history+[{'step':4}]);reject(lambda:list(cache.rows(path,info)),'advance between passes rejected before metrics')
  new_info,changed=cache.scan(path,config);check(changed and new_info['records_count']==4,'new publication rescanned from file')
  stream=cache.rows(path,new_info)
  try:
   next(stream);write_json(path,history)
   reject(lambda:list(stream),'sourcehandle closes after rawcopy; atomicadvance permitted and mtimeguard rejects completecycle')
  finally:stream.close()
  cache.close()
  archive=tmp/'v9_shards';archive.mkdir();state={'0':{'a':1},'unknownflag':{'text':'ş'},'finished':False};body=json.dumps(state,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode();sha=hashlib.sha256(body).hexdigest();p=archive/(sha+'.json');p.write_bytes(body)
  source=tmp/'v9.json';write_json(source,{'schema':1,'kind':'Trackio_run_shard_index_v9','source_import':{'complete':True,'unknown':3},'runs':{'fixture':{'shard':{'path':str(p),'sha256':sha,'bytes':len(body)},'finished_truthy':False}},'unknown_index_flag':['retained']})
  index=tmp/'v10.json';store=n['StreamRunShardStore'](index)
  reject(lambda:store.migrate(source),'handoff requires ROOT sourceV9closed')
  store.migrate(source,source_closed_confirmed=True);check(store.activate('fixture')==state,'V9 archived fullrun values/unknownflags preserved without reading allmaps')
  for i in range(6):store.accept_metrics(i+1,{'a':i});store.flush()
  own=list(store.directory.glob('*.json'));check(len(own)==2,'live superseded fullshards bounded to latesttwo')
  check(p.read_bytes()==body and source.exists(),'source V8/V9 archive shard and index untouched')
  check(store.index['source_import']['source_V9_index_payload']['unknown_index_flag']==['retained'],'unknown sourceindex flags preserved')
  check(not store.index['obsolete_live_shards'] and all(n['file_record'](x['shard']['path'])['sha256']==x['shard']['sha256'] for x in store.index['runs'].values()),'pruning follows committed complete index and keeps all latest refs')
  check(not store.flush(),'unchanged state no new liveversion')
  reopened=n['StreamRunShardStore'](index);check(reopened.activate('fixture')==store.active,'own run reload retains every amended metric')
  # Actual follow dispatcher; mock only Trackio/hardware/time and use tiny files.
  output=tmp/'output';output.mkdir();n['OUTPUT']=output;write_json(output/'run_config.json',config);write_json(output/'training_history.json',history)
  write_json(output/'results/fixture_step_0/summary.json',{'validation_tr':{'asr':{'wer':.4}}})
  events=[];telemetry=[];sleeps=[0];stop=tmp/'stop'
  n['trackio']=SimpleNamespace(init=lambda **k:events.append(('init',k['name'])),finish=lambda:events.append(('finish',)),log=lambda value,step:events.append(('log',step,dict(value))),Markdown=lambda x:x)
  original_publish=n['publish_cycle_memory']
  def publish(*a,**k):
   original_publish(*a,**k);telemetry.append(read_json(a[0].telemetry_path))
  n['publish_cycle_memory']=publish
  def sleep(seconds):
   sleeps[0]+=1
   if sleeps[0]==1:write_json(output/'results/fixture_step_0/summary.json',{'validation_tr':{'asr':{'wer':.39}}})
   else:stop.write_text('stop')
  n['time']=SimpleNamespace(sleep=sleep,time_ns=time.time_ns,monotonic=time.monotonic)
  a=SimpleNamespace(state_path=index,source_state_path=source,source_closed_confirmed=False,source_closure_receipt=None,project='fixture',follow=True,stop_file=stop,telemetry_path=tmp/'memory.json')
  check(n['follow'](a)=='cooperative_stop_file','actual guarded follow uses cooperative stop and flush')
  check(len(telemetry)==2 and telemetry[0]['history_metric_pass_streamed'] and not telemetry[1]['history_metric_pass_streamed'],'unchangedhistory secondcycle skips recordparse but publishes owncycle evidence')
  check(sum(event==('log',0,{'dev/wer_tr_percent':39.}) for event in events)==1,'changed DEV amendment same step logged once')
  check(telemetry[-1]['retained_history_records']==1 and telemetry[-1]['memory']['fixture']=='hardware unmeasured','selfonly telemetry scope explicit no hardware claim')
  replay=n['StreamRunShardStore'](index).activate('fixture');check(replay['0']['dev/wer_tr_percent']==39. and replay['unknownflag']==state['unknownflag'],'stopflush preserves amended metrics and unknown flags')
 report={'schema':1,'kind':'Trackio_V10_tiny_streaming_and_retention_stdlib_verification_v1','passed':True,'checks_count':len(checks),'checks':checks,'source':{'path':str(SOURCE),'sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest()},'fixture':{'path':str(Path(__file__).resolve()),'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},'scope':{'actual_history_or_state_or_shard_reads':False,'Trackio_import_or_DB_writes':False,'ML_Torch_GPU':False,'own_hardware_memory_measured':False,'actual_sourceclosure':False,'fixtures_only_under_work_temp':True}}
 write_json(args.output,report);print(json.dumps({'passed':True,'checks':len(checks),'report':str(args.output),'sha256':hashlib.sha256(args.output.read_bytes()).hexdigest()}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args())
