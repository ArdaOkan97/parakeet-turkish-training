import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'outputs/parakeet_turkish'
sys.path.insert(0, str(OUTPUT))
import owned_worker_memory_telemetry_v1 as t

checks = []
rejected = []
def good(ok, label):
    if not ok: raise AssertionError(label)
    checks.append(label)
def rejects(f, label):
    try: f()
    except (RuntimeError, KeyError): rejected.append(label)
    else: raise AssertionError(label)
def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')

table = {10:{'pid':10,'parent_pid':1,'image_name':'python.exe'},
         11:{'pid':11,'parent_pid':10,'image_name':'python.exe'},
         12:{'pid':12,'parent_pid':11,'image_name':'python.exe'},
         99:{'pid':99,'parent_pid':1,'image_name':'unrelated.exe'}}
good(set(t.owned_subtree(table,10))=={10,11,12},'Only owned launcher and descendants exposed')
rejects(lambda:t.owned_subtree(table,20),'Missing launcher rejects stale ownership')
too_many={10:table[10],**{i:{'pid':i,'parent_pid':10} for i in range(100,140)}}
rejects(lambda:t.owned_subtree(too_many,10),'Excessive descendant count rejects unbounded inventory')
native=t.process_table()
good(os.getpid() in native,'Actual Toolhelp snapshot contains this CPU proof process')
identity=t.process_identity(os.getpid())
good(identity.get('available') and Path(identity['image_path']).name.lower()=='python.exe','Actual own-process image/time API works')
memory=t._worker_memory(os.getpid())
good(memory.get('available') and memory['private_bytes']>0,'Actual own-process private-byte API works')

with tempfile.TemporaryDirectory(dir=ROOT/'work/research') as folder:
    folder=Path(folder); out=folder/'outputs'; loop=out/'runs/loop'; run=out/'runs/run'
    stamp=datetime.now(timezone.utc).isoformat(); log=loop/'owned.log';log.parent.mkdir(parents=True)
    log.write_text('run validation tr: 10/50\n',encoding='utf-8')
    state={'loop_id':'loop','status':'running','active_run_id':'run',
           'preflight':{'source_sha256':{'run_experiment_loop_continuation_v2.py':t.CONTROLLER_SHA}},
           'trials':[{'run_id':'run','pid':10,'started_at_utc':stamp,'log':str(log),'initial_step':1773}]}
    write(loop/'loop_status.json',state);write(run/'training_history.json',[{'step':1774}])
    def image(pid): return {'available':True,'created_at_utc':stamp,'image_path':'C:/owned/python.exe'}
    def private(pid):return {'pid':pid,'available':True,'private_bytes':806912 if pid==10 else 8_000_000_000}
    def sample(pid,phase):return {'system_memory':{'available':True,'commit_fraction':.7},
        'worker_private_memory':None if pid is None else private(pid)}
    with patch.object(t,'OUTPUT',out),patch.object(t,'process_table',return_value=copy.deepcopy(table)),\
         patch.object(t,'process_identity',side_effect=image),patch.object(t,'gpu_pids',return_value={12,99}),\
         patch.object(t,'sample_memory',side_effect=sample),patch.object(t,'_worker_memory',side_effect=private):
        x=t.observe(loop)
        good(x['actual_worker_pid']==12 and x['owned_gpu_context_pids']==[12],'GPU context must intersect validated owned ancestry')
        good(x['actual_worker_private_memory']['private_bytes']==8_000_000_000 and x['launcher_private_memory']['private_bytes']==806912,
             'Tiny launcher and actual GPU worker private memory reported separately')
        good(x['phase']['phase']=='development_evaluation','Bounded diagnostic phase recognizes DEV log')
        good(x['latest_published_step']==1774,'Latest history publication identified without checkpoint claim')
        good(len(x['owned_lineage'])==3,'Unrelated process omitted from saved evidence')
    with patch.object(t,'OUTPUT',out),patch.object(t,'process_table',return_value=copy.deepcopy(table)),\
         patch.object(t,'process_identity',side_effect=image),patch.object(t,'gpu_pids',return_value={11,12,99}),\
         patch.object(t,'sample_memory',side_effect=sample),patch.object(t,'_worker_memory',side_effect=private):
        x=t.observe(loop)
        good(x['actual_worker_pid'] is None and x['status']=='worker_context_unresolved' and x['actual_worker_private_memory'] is None,
             'Ambiguous GPU descendants cannot silently substitute launcher')
    with patch.object(t,'OUTPUT',out),patch.object(t,'process_table',return_value=copy.deepcopy(table)),\
         patch.object(t,'process_identity',return_value={'available':True,'created_at_utc':'2001-01-01T00:00:00+00:00','image_path':'C:/owned/python.exe'}):
        rejects(lambda:t.observe(loop),'Stale reused launcher PID creation time rejected')
    state['active_run_id']=None;write(loop/'loop_status.json',state)
    good(t.observe(loop)['status']=='no_active_owned_worker','Terminal/no-active state stops sampling')

report={'schema':1,'kind':'owned_worker_memory_telemetry_CPU_verification_v1','passed':True,
        'source_sha256':t.sha(t.__file__),'test_sha256':t.sha(__file__),
        'positive_assertions':len(checks),'rejected_corruptions':len(rejected),
        'GPU_initialized_or_jobs_launched':False,'checks':checks,'rejections':rejected,
        'actual_training_process_access_scope':'Sandbox cannot see/open root training PIDs; root-context one-shot evidence required'}
target=OUTPUT/'owned_worker_memory_telemetry_v1_cpu_verification.json';write(target,report)
print(json.dumps({'passed':True,'checks':len(checks),'rejections':len(rejected),'report':str(target)}))
