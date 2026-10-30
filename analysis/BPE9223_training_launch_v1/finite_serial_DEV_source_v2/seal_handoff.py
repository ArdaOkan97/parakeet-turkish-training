import difflib,json,sys
from pathlib import Path
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from bpe_broad_audio_v1 import record
from fastmss_BPE9223_cpu_common_v3 import environment,resource_guard
environment();resource_guard()
import psutil
assert not psutil.pid_exists(29284)
O=Path(__file__).parent;old=json.loads((O.parent/'finite_serial_DEV_source_v1/source_handoff_v1.json').read_text())
closure=record(O/'fixture_guard_01/closure.json');c=json.loads(Path(closure['path']).read_text());assert c['actual_exit_code']==0 and c['monitor_clean'] and c['monitor_joined']
proof={'kind':'ASR18_source_bound_actual_tool_exit_v1','owner':'/root/cpu_data_preparation','actual_exit_code':0,
    'actual_exec_tool_exit_code':0,'actual_tool_chunk_id':'e7a4f4','native_PID':29284,'published_marker':record(O/'fixture_report.json'),
    'guard_closure':closure,'source':record(O/'fixture.py'),'scope':'38 source/JSON metadata controls; genuine saved metadata/PRE only, no tensor payload read'}
f=O/'actual_fixture_CLOSED0.json';assert not f.exists();f.write_text(json.dumps(proof,indent=2))
names=['asr18_finite_BPE9223_serial_DEV_v2.py','finite_BPE9223_DEV_runtime_v2.py','asr18_finite_BPE9223_checkpoint_evidence_v2.py']
sources={n:record(S/n) for n in names}
def rename(v):
    if type(v) is str:
        for name in names:v=v.replace(name.replace('_v2.py','_v1.py'),name)
        return v
    if type(v) is list:return [rename(x) for x in v]
    if type(v) is dict:return {rename(k):rename(x) for k,x in v.items()}
    return v
new=rename(old);new.update(kind='ASR18_finite_BPE9223_serial350DEV_inactive_source_handoff_v2',sources=sources,
    prior_handoff=record(O.parent/'finite_serial_DEV_source_v1/source_handoff_v1.json'),
    fixture_report=record(O/'fixture_report.json'),fixture_actual0=record(f),
    baselines=old['sources'],deltas=['Helper-only canonical normalized state hash equality replaces Python tuple/list equality',
    'Serial2/runtime2 source import/path substitutions only; all evaluator/numerical/resource behavior preserved'],
    real_JSON_PRE_roundtrip_proved=True,actual_PRE_state_sha256='21a7c8ec9a55692a7672cda047ab9cbd5f4e08c9d3395f2863bcfa2201580072')
import asr18_finite_BPE9223_serial_DEV_v2 as finite
new['request_example_IS_NOT_EXECUTABLE']['dependencies']=finite.sources()
for name in names:
    prior=name.replace('_v2.py','_v1.py')
    (O/(name+'.patch')).write_text(''.join(difflib.unified_diff((S/prior).read_text().splitlines(True),(S/name).read_text().splitlines(True),fromfile=prior,tofile=name)))
target=O/'source_handoff_v2.json';assert not target.exists();target.write_text(json.dumps(new,indent=2),encoding='utf-8');print(json.dumps(record(target)))
