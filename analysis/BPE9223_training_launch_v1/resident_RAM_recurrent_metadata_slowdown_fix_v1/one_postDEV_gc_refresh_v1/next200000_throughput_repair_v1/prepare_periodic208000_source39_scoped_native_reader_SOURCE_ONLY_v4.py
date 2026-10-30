import ast,copy,hashlib,json,os,pathlib,shutil,sys,time,traceback
import psutil
W=pathlib.Path(__file__).resolve().parent;P=pathlib.Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20')
sys.path.insert(0,str(P/'outputs/parakeet_turkish'))
from resource_watchdog_v2 import sample_memory
def pin(p):
 p=pathlib.Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def require(v,m):
 if not v:raise ValueError(m)
def write(p,v):
 with p.open('x',encoding='utf-8',newline='\n')as f:json.dump(v,f,ensure_ascii=False,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 return pin(p)
me=psutil.Process();me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);began=time.monotonic()
entry=dict(identity=dict(pid=me.pid,create_time=me.create_time(),ppid=me.ppid()),launcher=dict(pid=me.ppid(),create_time=me.parent().create_time(),ppid=me.parent().ppid()))
sample=sample_memory(me.pid,'periodic208000_scoped_native_reader_SOURCE_ONLY')
initial=write(W/'periodic208000_source39_scoped_native_reader_preparation_initial_v4.json',dict(entry=entry,sample=sample))
result=dict(kind='ASR18_periodic208000_source39_scoped_native_reader_SOURCE_ONLY_preparation_v4',entry=entry,initial=initial,outputs={},error=None)
try:
 require(sample['system_memory']['commit_available_bytes']>=5130803840 and sample['system_memory']['physical_available_bytes']>=2380803840,'Original full prospective plus separate72')
 require(me.memory_info().private<536870912 and shutil.disk_usage('E:/gpu_training_scratchpad').free>=100000000000,'Original private/E guard')
 require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'and os.environ.get('TORCH_EXTENSIONS_DIR')=='E:\\gpu_training_scratchpad\\cache\\torch_extensions','Exact E environment')
 beforepath=W/'seal_periodic208000_source39_CPU14_helper3_max2_saved208200_v2.py';afterpath=W/'seal_periodic208000_source39_CPU14_helper3_max2_saved208200_v3.py'
 rawpath=W/'periodic208000_source39_metadata_observer_full_RAW_AST_v3.json';hpath=W/'periodic208000_source39_CPU14_metadata_observer_INACTIVE_handoff_v3.json'
 require(all(not p.exists()for p in (afterpath,rawpath,hpath)),'Fresh immutable successor files')
 require(pin(beforepath)['sha256']=='f5e71bd96d0c4d4dc57ae8f94bdc8f0e3c01d4bb2db21bdb1aba75aae95c956b','Exact frozen source2')
 data=beforepath.read_bytes();nl=b'\r\n'if data.count(b'\r\n')==data.count(b'\n')else b'\n';steps=[]
 def replace(a,b,meaning,count):
  global data
  a=a.encode().replace(b'\r\n',b'\n').replace(b'\n',nl);b=b.encode().replace(b'\r\n',b'\n').replace(b'\n',nl)
  require(data.count(a)==count,'Exact source change occurrence count')
  positions=[];at=0
  for unused in range(count):
   pos=data.index(a,at);data=data[:pos]+b+data[pos+len(a):];positions.append(pos);at=pos+len(b)
  steps.append(dict(old=a.decode(),new=b.decode(),count=count,application_byte_offsets=positions,meaning=meaning))
 needle=" 'Separate exact ROOT execution binding for this immutable source')"
 replace(needle,needle+"\nrequire(execution_review['native']=={'pid':34788,'create_time':1792842518.865024,'ppid':32332},\n 'Exact separate ROOT-bound current source39 native; no closed actual boot reread or rebind')",'Explicit exact frozen current native supplied by future ROOT scope',1)
 replace("read(I/'boot.json')['producer']","execution_review['native']",'Replace both closed actual boot producer rereads by exact separately ROOT-bound native',2)
 text=data.decode();start=text.index("reader=read(reader_root/'current_status.json')");end=text.index("observation=A/'actual_periodic208000_postpositive_progress.json'",start)
 old=text[start:end]
 new="""reader=read(reader_root/'current_status.json')
reader_identity=execution_review['reader_identity'];reader_launcher_identity=execution_review['reader_launcher_identity']
reader_source=execution_review['reader_source']
require(pin(reader_source['path'])==reader_source,'Exact separately reviewed current reader SOURCE pin; no reader import or action')
require(reader['pid']==reader_identity['pid']and abs(reader['create_time']-reader_identity['create_time'])<.001
 and exact_alive(reader_identity)and exact_alive(reader_launcher_identity)
 and psutil.Process(reader_identity['pid']).ppid()==reader_launcher_identity['pid']
 and Path(reader['source']).resolve()==Path(reader_source['path']).resolve()and reader['source_sha256']==reader_source['sha256']
 and reader['error']is None and reader['training_worker_alive']is True
 and reader['native_producer']['pid']==execution_review['native']['pid']and Path(reader['journal_path']).name=='commits_000027.jsonl',
 'Current reader exact identities/SOURCE frozen only by future ROOT scope after actual approved replacement; no auto-rebind or action')
"""
 replace(old,new,'Future ROOT-bound reader pair and source replace hardcoded old40048/40132; UNKNOWN until actual accepted reader transition',1)
 ast_new=ast.parse(data.decode())
 require("read(I/'boot.json')"not in data.decode(),'No closed boot file reread remains')
 require('40048'not in data.decode()and'40132'not in data.decode(),'No fixed historical reader PID remains')
 ledger=json.loads((W/'periodic208000_source39_metadata_observer_full_RAW_AST_v2.json').read_text())
 ledger['ordered_edits'].extend(steps)
 inverse=data
 for edit in reversed(ledger['ordered_edits']):
  a=edit['new'].encode();b=edit['old'].encode()
  for pos in reversed(edit['application_byte_offsets']):
   require(inverse[pos:pos+len(a)]==a,'Exact reverse fullchain byte slice');inverse=inverse[:pos]+b+inverse[pos+len(a):]
 require(inverse==pathlib.Path(ledger['base']['path']).read_bytes(),'Entire accepted196 source whole-byte inverse')
 with afterpath.open('xb')as f:f.write(data);f.flush();os.fsync(f.fileno())
 ledger['successor']=pin(afterpath);ledger['successor_whole_AST_sha256']=hashlib.sha256(ast.dump(ast_new,include_attributes=False).encode()).hexdigest()
 ledger['preserved_unexecuted_v2_source']=pin(beforepath);ledger['preserved_v2_RAW']=pin(W/'periodic208000_source39_metadata_observer_full_RAW_AST_v2.json')
 ledger['new_nonliteral_delta']=[step['meaning']for step in steps]
 rawpin=write(rawpath,ledger)
 h=json.loads((W/'periodic208000_source39_CPU14_metadata_observer_INACTIVE_handoff_v2.json').read_text())
 replacements={h['metadata_source']['path']:pin(afterpath),h['full_RAW']['path']:rawpin}
 h['metadata_source']=pin(afterpath);h['full_RAW']=rawpin;h['input_pins']=[copy.deepcopy(replacements.get(item['path'],item))for item in h['input_pins']]
 h['kind']='ASR18_periodic208000_source39_CPU14_metadata_observer_INACTIVE_handoff_v3';h['proposed_arguments'][0]=str(afterpath)
 h['preparer']=entry['identity'];h['launcher']=entry['launcher'];h['initial']=initial
 h['future_execution_native']=dict(pid=34788,create_time=1792842518.865024,ppid=32332)
 h['future_reader_identity']='UNKNOWN until genuine separately ROOT-reviewed reader transition accepted; future exact execution scope REQUIRED'
 h['future_reader_launcher_identity']='UNKNOWN; no automatic capture/rebind by observer'
 h['future_reader_source']='UNKNOWN exact SOURCE pin; may be original or separately proven new reader only as final ROOT scope binds'
 h['closed_actual_boot_reads_removed']=2
 h['preserved_unexecuted_v2']=dict(source=pin(beforepath),RAW=pin(W/'periodic208000_source39_metadata_observer_full_RAW_AST_v2.json'),only_successor_changes=steps)
 hpin=write(hpath,h)
 require(time.monotonic()-began<180 and me.memory_info().private<536870912,'Original source preparation bounds')
 result['outputs']=dict(source=pin(afterpath),RAW=rawpin,handoff=hpin);result['passed']=True
except BaseException as error:
 result['passed']=False;result['error']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
finally:
 result['wall_seconds']=time.monotonic()-began
 result_pin=write(W/'periodic208000_source39_scoped_native_reader_preparation_result_v4.json',result)
 print(json.dumps(dict(result=result_pin,entry=entry,passed=result.get('passed'),outputs=result['outputs'],error=result['error']),sort_keys=True))
require(result.get('passed'),'Preserve source preparation failure; no actual observer/proof/control')
