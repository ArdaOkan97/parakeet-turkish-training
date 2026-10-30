import ast,copy,hashlib,json,os,pathlib,shutil,sys,time
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
me=psutil.Process();me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);started=time.monotonic()
entry=dict(identity=dict(pid=me.pid,create_time=me.create_time(),ppid=me.ppid()),launcher=dict(pid=me.ppid(),create_time=me.parent().create_time(),ppid=me.parent().ppid()))
sample=sample_memory(me.pid,'periodic208000_final_literal_SOURCE_ONLY')
initial=write(W/'periodic208000_source39_final_literal_preparation_initial_v3.json',dict(entry=entry,sample=sample))
require(sample['system_memory']['commit_available_bytes']>=5130803840 and sample['system_memory']['physical_available_bytes']>=2380803840,'Same full prospective guard before source writes')
require(me.memory_info().private<536870912 and shutil.disk_usage('E:/gpu_training_scratchpad').free>=100000000000,'Original private/E guards')
require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'and os.environ.get('TORCH_EXTENSIONS_DIR')=='E:\\gpu_training_scratchpad\\cache\\torch_extensions','Exact processlocal E CPU environment')
old=W/'seal_periodic208000_source39_CPU14_helper3_max2_saved208200_v1.py';new=W/'seal_periodic208000_source39_CPU14_helper3_max2_saved208200_v2.py'
rawpath=W/'periodic208000_source39_metadata_observer_full_RAW_AST_v2.json';handoffpath=W/'periodic208000_source39_CPU14_metadata_observer_INACTIVE_handoff_v2.json'
require(not new.exists()and not rawpath.exists()and not handoffpath.exists(),'Fresh immutable successors')
before=old.read_bytes();a=b'run_ASR18_BPE9223_resident_RAM_training_v37.py';b=b'run_ASR18_BPE9223_resident_RAM_training_v39.py'
require(before.count(a)==1,'Exactly one stale driver source filename')
offset=before.index(a);after=before[:offset]+b+before[offset+len(a):];ast_new=ast.parse(after.decode('utf-8'))
require(after[:offset]+a+after[offset+len(b):]==before,'Exact one-edit immediate whole-byte inverse')
with new.open('xb')as f:f.write(after);f.flush();os.fsync(f.fileno())
ledger=json.loads((W/'periodic208000_source39_metadata_observer_full_RAW_AST_v1.json').read_text())
ledger['ordered_edits'].append(dict(old=a.decode(),new=b.decode(),count=1,application_byte_offsets=[offset],meaning='Exact current driver39 SOURCE filename literal; v1 never executed'))
inverse=after
for edit in reversed(ledger['ordered_edits']):
 x=edit['new'].encode();y=edit['old'].encode()
 for pos in reversed(edit['application_byte_offsets']):
  require(inverse[pos:pos+len(x)]==x,'Exact ordered full-chain byte slice inverse');inverse=inverse[:pos]+y+inverse[pos+len(x):]
require(hashlib.sha256(inverse).hexdigest()==ledger['base']['sha256']and inverse==pathlib.Path(ledger['base']['path']).read_bytes(),'Entire accepted196 source bytes reconstructed')
ledger['successor']=pin(new);ledger['successor_whole_AST_sha256']=hashlib.sha256(ast.dump(ast_new,include_attributes=False).encode()).hexdigest()
ledger['preserved_unexecuted_v1_source']=pin(old);ledger['preserved_v1_RAW']=pin(W/'periodic208000_source39_metadata_observer_full_RAW_AST_v1.json')
rawpin=write(rawpath,ledger)
h=json.loads((W/'periodic208000_source39_CPU14_metadata_observer_INACTIVE_handoff_v1.json').read_text())
replacements={h['metadata_source']['path']:pin(new),h['full_RAW']['path']:rawpin}
h['metadata_source']=pin(new);h['full_RAW']=rawpin;h['input_pins']=[copy.deepcopy(replacements.get(item['path'],item))for item in h['input_pins']]
h['proposed_arguments'][0]=str(new);h['preparer']=entry['identity'];h['launcher']=entry['launcher'];h['initial']=initial
h['kind']='ASR18_periodic208000_source39_CPU14_metadata_observer_INACTIVE_handoff_v2'
h['preserved_unexecuted_v1']=dict(source=pin(old),RAW=pin(W/'periodic208000_source39_metadata_observer_full_RAW_AST_v1.json'),only_successor_edit='training_v37.py -> training_v39.py')
hpin=write(handoffpath,h)
require(time.monotonic()-started<180 and me.memory_info().private<536870912,'Original metadata wall/private guard')
rpin=write(W/'periodic208000_source39_final_literal_preparation_result_v3.json',dict(passed=True,entry=entry,initial=initial,source=pin(new),RAW=rawpin,handoff=hpin,actual_observer_or_candidate_execution=False))
print(json.dumps(dict(result=rpin,source=pin(new),RAW=rawpin,handoff=hpin,entry=entry),sort_keys=True))
