"""Finite AST-only changed checkpoint-policy CPU controls; no live action."""
import ast,copy,json,hashlib,os,sys,time,argparse,shutil
from pathlib import Path
import psutil
def pin(p):
 p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def write(p,v):
 raw=(json.dumps(v,ensure_ascii=False,sort_keys=True,allow_nan=False,separators=(',',':'))+'\n').encode()
 with Path(p).open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
 return pin(p)
def who(pid=None):
 p=psutil.Process(os.getpid() if pid is None else pid);return dict(pid=p.pid,create_time=p.create_time(),ppid=p.ppid())
def require(v,m):
 if not v:raise ValueError(m)
def main():
 a=argparse.ArgumentParser();a.add_argument('--ROOT-review',required=True);s=a.parse_args()
 scope=json.loads(Path(s.ROOT_review).read_bytes());out=Path(scope['output'])
 require(scope['accepted']is True and scope['reviewer']=='/root'and scope['fixture_source']==pin(__file__),'Exact ROOT CPU proof scope')
 require(not out.exists()and out.drive.lower()=='e:','Fresh E proof output');out.mkdir()
 psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);began=time.monotonic()
 sys.path.insert(0,'C:/Users/ardao/Documents/Codex/2026-10-10/i-x20/outputs/parakeet_turkish')
 from resource_watchdog_v2 import sample_memory
 initial=sample_memory(os.getpid(),'checkpoint204200_policy_CPU_initial')
 init=write(out/'initial_full_numeric_before_predicate.json',initial)
 m=initial['system_memory']
 require(initial['paired_sampling_within_one_second']is True and m['commit_available_bytes']>=5055306368
  and m['physical_available_bytes']>=2305306368 and initial['worker_private_memory']['private_bytes']<536870912
  and shutil.disk_usage('E:/').free>=100000000000,'Same full prospective guards')
 for p in scope['input_pins']:require(pin(p['path'])==p,'All exact inputs')
 require(os.environ['CUDA_VISIBLE_DEVICES']=='-1'and all(os.environ[k]=='1'for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'))and os.environ.get('TORCH_EXTENSIONS_DIR','').lower().startswith('e:'),'CPU hidden/E/caps')
 tree=ast.parse(Path(scope['pause_source']['path']).read_bytes())
 nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in ('check_eval','check_checkpoint','frontier_decision')]
 require(len(nodes)==3,'All three exact source functions')
 ns={'require':require,'TARGET':204200}
 exec(compile(ast.Module(body=nodes,type_ignores=[]),scope['pause_source']['path'],'exec'),ns)
 c=json.loads(Path(scope['actual_DEV_copy']['path']).read_bytes());cases=[]
 ns['check_eval'](c['DEV'],c['entry']);cases.append('actual14_validation_and_calibration_records_pass')
 def negative(name,fn):
  try:fn()
  except ValueError:cases.append(name);return
  raise AssertionError(name)
 for name,mutate in [('missing_validation',lambda v:v['scores'].pop('validation_tr')),('unexpected_record',lambda v:v['scores'].update(extra={})),('wrong_utterances',lambda v:v['scores']['validation_tr']['asr'].update(utterances=49)),('DEV_native',lambda v:v.update(actual_native_updates_in_DEV=1)),('TEST',lambda v:v.update(TEST_used=True))]:
  v=copy.deepcopy(c['DEV']);mutate(v);negative(name,lambda:ns['check_eval'](v,c['entry']))
 src={'source':'fixture_bound_only'};p={'pointer':'fixture_bound_only'};z={'persistence':'fixture_bound_only'}
 l=dict(committed_updates=204200,full_state_verified=True,source=src,pointer=p,persistence=z)
 v=dict(passed=True,optimizer_updates=204200,named_tensors_verified=854,weights_verified=207,Adam_states_verified=215,independent_pre_save_state_sha256='same',saved_state_sha256='same',saved_semantic_state_verified={'passed':True})
 ns['check_checkpoint'](l,v,src,p,z);cases.append('full854_207_215_PRE_SAVED_pass')
 for name,change in [('bad204checkpoint',{'optimizer_updates':204000}),('partialweights',{'weights_verified':206}),('badPRESAVED',{'saved_state_sha256':'different'})]:
  x=copy.deepcopy(v);x.update(change);negative(name,lambda:ns['check_checkpoint'](l,x,src,p,z))
 require([ns['frontier_decision']({'actual_optimizer_updates':u})for u in (204199,204200,204399,204400)]==['WAIT','READY','READY','MISSED'],'Exact separate checkpoint frontier band');cases.append('frontier_WAIT_READY_READY_MISSED')
 final=sample_memory(os.getpid(),'checkpoint204200_policy_CPU_final');m=final['system_memory']
 require(final['paired_sampling_within_one_second']is True and m['commit_available_bytes']>=2268435456 and m['physical_available_bytes']>=1768435456 and final['worker_private_memory']['private_bytes']<536870912 and time.monotonic()-began<120,'Same bounded runtime guards')
 result=write(out/'report.json',dict(kind='ASR18_saved204200_checkpoint_policy_actual_CPU_v1',passed=True,cases=cases,source=scope['pause_source'],fixture_source=pin(__file__),initial=init,actual_original204report_metadata=True,checkpoint_metadata_controls='METHOD_STUB only; future actual204200 persistence bound by live predicate',no_actor_or_production_action=True,identity=who(),launcher=who(os.getppid()),wall_seconds=time.monotonic()-began,final=final))
 print(json.dumps({'result':result,'passed':True,'cases':len(cases),'identity':who()}),flush=True)
if __name__=='__main__':main()

