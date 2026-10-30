"""Narrow CPU5 source-pin repair plus real imported-pin metadata preflight."""
import ast,copy,hashlib,importlib,json,os,sys
from pathlib import Path
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish';R=P/'work/research'
sys.path[:0]=[str(S),str(R)]
from bpe_broad_audio_v1 import record
from fastmss_BPE9223_cpu_common_v4 import environment,resource_guard
environment();resource_guard();O=Path(__file__).parent;N=O.parent
old=json.loads((N/'CPU_preparation_v4/source_handoff_v4.json').read_text())
mapping={Path(v['path']).stem:Path(v['path']).stem[:-1]+'5' for v in old['sources'].values()}
sources={};checks=[]
for key in ('bridge','engine','core','protocol','adapter','parent','reference','frontend'):
    pin=old['sources'][key];path=Path(pin['path']);assert record(path)==pin;raw=path.read_text()
    for before,after in mapping.items():raw=raw.replace(before,after)
    if key=='engine':
        previous='BRIDGE_SHA = "f53191620dadd320fef178aef95daabf5787871379867a5e7a5078321c17e181"'
        assert raw.count(previous)==1;raw=raw.replace(previous,'BRIDGE_SHA = "'+sources['bridge']['sha256']+'"')
    ast.parse(raw);target=path.with_name(mapping[path.stem]+'.py');assert not target.exists();target.write_text(raw,encoding='utf-8',newline='\n')
    sources[key]=record(target);checks.append(key+': exact version/import substitution'+(' and actual bridge5 hash' if key=='engine' else ''))
helper=(S/'asr18_BPE9223_CPU_window_configuration_v3.py').read_text()
helper=helper.replace('fastmss_BPE9223_feature_protocol_v4','fastmss_BPE9223_feature_protocol_v5').replace(old['sources']['protocol']['sha256'],sources['protocol']['sha256'])
target=S/'asr18_BPE9223_CPU_window_configuration_v4.py';assert not target.exists();ast.parse(helper);target.write_text(helper,encoding='utf-8');helper_pin=record(target)
modules={k:importlib.import_module(Path(v['path']).stem) for k,v in sources.items()}
bridge,engine,core,public,adapter=[modules[k] for k in ('bridge','engine','core','protocol','adapter')]
assert engine.bridge is bridge and core.bridge is bridge and core.sink is engine and public.engine is engine and public.augmentation is adapter
assert engine.BRIDGE_SHA==record(bridge.__file__)['sha256']==sources['bridge']['sha256']
checks.append('LIVE engine5 BRIDGE_SHA equals actual imported bridge5 source; stale CPU3 literal absent')
for role in ('parent','reference','frontend'):
    assert modules[role].core is core and modules[role].public is public
    if role!='reference':assert modules[role].reference is modules['reference']
checks.append('LIVE controller/reference/frontend share exact core5/protocol5/reference5 module cohort')
def paths_in(module,function):
    tree=ast.parse(Path(module.__file__).read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==function)
    nodes=[]
    for n in fn.body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='paths' for t in n.targets):nodes.append(n)
        elif isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and isinstance(n.value.func.value,ast.Name) and n.value.func.value.id=='paths' and n.value.func.attr=='update':nodes.append(n)
    scope=dict(vars(module));exec(compile(ast.Module(body=nodes,type_ignores=[]),'<exact-source-path-table-only>','exec'),scope)
    return scope['paths']
base=json.loads((N.parent/'ASR18_memory_diagnosis_v1/native_throughput_first4_configuration_01/reviewed_CPU_preparation_contract.json').read_text())
core_remap={'common':'core','bridge':'bridge','sink':'engine','reference_worker':'reference','feature_worker':'frontend','controller':'parent'}
corepaths=paths_in(core,'verify_contract')
for key,path in corepaths.items():
    pin=record(path)
    if key in core_remap:assert pin==sources[core_remap[key]],key
    elif key=='cleaner':assert Path(path).name=='asr18_BPE9223_source_context_v2.py'
    else:assert pin==base['sources'][key],key
    checks.append('LIVE numerical source table '+key)
publicpaths=paths_in(public,'verify_source_contract');prior=json.loads((N.parent/'FastMSS_integration_v1/recorded_room_CPU_source_contract_v2.json').read_text())
remap={'protocol':'protocol','engine':'engine','reference':'reference','frontend':'frontend','controller':'parent','core':'core','augmentation_adapter':'adapter'}
for key,path in publicpaths.items():
    pin=record(path)
    assert pin==(sources[remap[key]] if key in remap else prior['sources'][key]),key
    checks.append('LIVE public source table '+key)
for key,(name,expected) in bridge.PINS.items():
    assert record(S/name)['sha256']==expected,key;checks.append('LIVE frozen foundational hash '+key)
h=importlib.import_module(target.stem);assert h.PROTOCOL_SHA256==sources['protocol']['sha256'];checks.append('LIVE pure configuration helper pins protocol5 exact SHA')
actualpin=record(N/'hard_configuration_02/actual_CPU9223_bridge.json');actual=bridge.read_bound(actualpin)
bridge.validate_bridge_scope(actual)
assert actual['mode']==bridge.HARD_MODE and len(actual['updates'])==1 and actual['updates'][0]['rows'][-3]['full_target_ids']
try:engine._load_bridge(actualpin,resource_guard)
except ValueError:checks.append('CPU5 correctly rejects immutable actualCPU4 bridge source identity')
else:raise AssertionError('Old source bridge incorrectly adopted')
fixture=copy.deepcopy(actual);fixture['bridge_source_sha256']=sources['bridge']['sha256']
fixture_path=O/'source_version_fixture_bridge_NOT_EXECUTABLE.json';assert not fixture_path.exists();fixture_path.write_bytes(bridge.canonical(fixture))
assert engine._load_bridge(record(fixture_path),resource_guard)==fixture
checks.append('LIVE engine5 accepts explicit metadata-only source-version fixture from actualhard02 rows; no session/plan/CPU0 authority claimed')
assert all(name not in sys.modules for name in ('torch','numpy','soundfile','pyarrow','av'))
_,identity=adapter.policy_identity(old['policy'],resource_guard)
identity_path=O/'augmentation_identity_v5.json';assert not identity_path.exists();identity_path.write_text(json.dumps(identity,indent=2))
report={'kind':'ASR18_CPU5_exact_import_and_pin_metadata_preflight_v1','passed':True,'PID':os.getpid(),'checks':checks,
    'sources':sources,'configuration_helper':helper_pin,'baseline_CPU4':old['sources'],'actual_hard02_bridge':actualpin,
    'source_version_fixture_NOT_EXECUTABLE':record(fixture_path),'actual_audio_model_GPU_or_payload_reads':0,
    'scope':'Actual module/path/hash closure and metadata loader only; no full hard bridge rebuild or prepared-feature/native authority'}
(O/'preflight_report.json').write_text(json.dumps(report,indent=2))
handoff={**old,'kind':'BPE9223_generic_CPU_frozen_source_handoff_v5','sources':sources,'old_sources_preserved':old['sources'],
    'prior_CPU4_handoff':record(N/'CPU_preparation_v4/source_handoff_v4.json'),'augmentation_identity':record(identity_path),
    'configuration_helper':helper_pin,'exact_delta_map':['Import/path/source-pin closure only; engine BRIDGE_SHA now exact bridge5 hash'],
    'metadata_preflight':record(O/'preflight_report.json'),'metadata_validation':'LIVE complete changed cohort/path tables and literal source pins verified; actualhard02 source-version fixture explicitly not executable',
    'preserved_CPU4_failure':'Actual metadata02 TOOL1/9bbb3f; oldCPU4 engine staleCPU3 bridge hash retained as historical failure'}
target=O/'source_handoff_v5.json';assert not target.exists();target.write_text(json.dumps(handoff,indent=2))
print(json.dumps({'handoff':record(target),'checks':len(checks),'PID':os.getpid(),'sources':sources}),flush=True)
