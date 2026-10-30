"""Independent CPU-only source/API controls. No model, audio or tensor payload reads."""
import contextlib, copy, ctypes, hashlib, importlib, io, json, os, struct, sys, time
from pathlib import Path

ROOT = Path(r'E:\gpu_training_scratchpad')
PROJECT = Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20')
AREA = ROOT/'analysis/BPE9223_training_launch_v1/independent_operational_recovery_review_v1'
sys.path[:0] = [str(PROJECT/'outputs/parakeet_turkish'), str(ROOT/'analysis/Turkish_labels_BPE_circumflex_v1')]
m = importlib.import_module('audit_and_fit_append_only_v5'); m.AREA = AREA
g = m.Guard(); g.max_seconds = 180; g.maximum_bytes = 64*1024**2
# Tighten the already-owned worker job to the explicit finite metadata scope.
from ctypes import wintypes as w
class JobIO(ctypes.Structure):
    _fields_=[(x,ctypes.c_ulonglong) for x in ('r','w','o','rb','wb','ob')]
class Basic(ctypes.Structure):
    _fields_=[('p',ctypes.c_longlong),('j',ctypes.c_longlong),('flags',w.DWORD),('min',ctypes.c_size_t),('max',ctypes.c_size_t),('active',w.DWORD),('affinity',ctypes.c_size_t),('priority',w.DWORD),('scheduling',w.DWORD)]
class Limits(ctypes.Structure):
    _fields_=[('basic',Basic),('io',JobIO),('process_memory',ctypes.c_size_t),('job_memory',ctypes.c_size_t),('peak_process',ctypes.c_size_t),('peak_job',ctypes.c_size_t)]
limits=Limits();limits.basic.flags=0x100|0x200;limits.process_memory=512_000_000;limits.job_memory=512_000_000
api=ctypes.WinDLL('kernel32',use_last_error=True)
api.SetInformationJobObject.argtypes=[w.HANDLE,ctypes.c_int,ctypes.c_void_p,w.DWORD];api.SetInformationJobObject.restype=w.BOOL
if not api.SetInformationJobObject(g.job,9,ctypes.byref(limits),ctypes.sizeof(limits)):raise ctypes.WinError(ctypes.get_last_error())
def guard():
    g.check()
    assert g.process.memory_info().private < 512_000_000
def record(path):
    p=Path(path);raw=p.read_bytes();return {'path':str(p.resolve()),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
checks=[]
def positive(name, fn):
    guard();fn();checks.append({'name':name,'passed':True})
def negative(name, fn):
    guard()
    try:fn()
    except (ValueError,AssertionError,KeyError,TypeError) as exc:
        checks.append({'name':name,'passed':True,'rejected':type(exc).__name__,'message':str(exc)[:240]});return
    raise AssertionError('Negative accepted: '+name)
started=time.monotonic()
names=['asr18_BPE9223_resource_operational_compatibility_v1','asr18_BPE9223_persistence_prefix_v5',
       'asr18_BPE9223_persistence_prefix_v6','asr18_BPE9223_durable_owner_adapter_v4',
       'asr18_BPE9223_indexed_checkpoint_storage_v2','asr18_BPE9223_metadata_resolver_v3']
modules={}
for name in names:
    modules[name]=importlib.import_module(name)
    checks.append({'name':'stdlib_import_'+name,'passed':True})
assert not any(name in sys.modules for name in ('torch','numpy','tokenizers','soundfile'))
comp=modules[names[0]];oldprefix=modules[names[1]];prefix=modules[names[2]]
index=modules[names[4]];resolver=modules[names[5]]
config=json.loads((ROOT/'analysis/BPE9223_training_launch_v1/full_configuration_04/configuration_ready_for_final_execution_review.json').read_text())
identity=config['full_run_identity']
resource_path=ROOT/'analysis/BPE9223_training_launch_v1/full_training_genesis_actual_02/actual_resource_stop_TOOL1_v1.json'
resource_pin=record(resource_path);resource=json.loads(resource_path.read_text())
assert comp.digest(identity)=='3381c4153f10cd5037f0e6f0ba760414db5a95b5ae240287f044e1fc7892e619'
assert identity['full_run_source_contract']['sha256']=='463c220bb3279c3bffdfe7fdf6e019f0d5e6cbf973c2dfed6b8a99012cab6e84'
positive('scientific_identity_and_original432_preserved',lambda:None)
positive('prefix6_typed_int',lambda:prefix.integer(20))
for value in (20.0,True,'20',-1):negative('prefix6_reject_counter_'+repr(value),lambda v=value:prefix.integer(v))
for value in ({},{'owner_validation_only':0},{'owner_validation_only':1}):
    negative('qualification_flag_reject_'+repr(value),lambda v=value:prefix.diagnostic_identity(v))
for value in (True,False):positive('qualification_flag_bool_'+str(value),lambda v=value:prefix.diagnostic_identity({'owner_validation_only':v}))
negative('duplicate_JSON_key',lambda:resolver._json(b'{"a":1,"a":2}'))
negative('nonfinite_JSON',lambda:resolver._json(b'{"x":NaN}'))
negative('foreign_metadata_path',lambda:resolver._path(str(PROJECT/'foreign.json')))
negative('metadata_pin_float_bytes',lambda:resolver._pin({'path':str(AREA/'x.json'),'bytes':1.0,'sha256':'0'*64}))
negative('metadata_chunk_unbounded',lambda:resolver._ReadOnlyChunks(io.BytesIO(b'x'),guard).read(1024**2+1))
# Mock only ROOT-authorization lookup. Every resource/p1 metadata/journal/death
# check still reads its genuine immutable artifact; payload is stat-only.
original_checked=comp.checked_compatibility
comp.checked_compatibility=lambda *a,**k:{'actual_resource_stop':resource_pin}
try:
    positive('real_resource20_metadata_and_closed_journal_under_mock_authority',
        lambda:comp.checked_resource_stop(resource_pin,identity=identity,compatibility={'mock_only':True},guard=guard))
    original_read=comp.read_bound
    for key,value in [('actual_tool_exit_code',0),('actual_committed_optimizer_updates',20.0),
                      ('actual_authoritative_TR_draws',280.0),('actual_native0_or_CURRENT_fresh0_claimed',True),
                      ('normal40_or_normal_TOOL0_claimed',True),('second_CPU_child_started',True),
                      ('all_captured_identities_dead',False)]:
        altered=copy.deepcopy(resource);altered[key]=value
        comp.read_bound=lambda pin,guard=lambda:None,a=altered:copy.deepcopy(a) if pin==resource_pin else original_read(pin,guard)
        negative('resource_reject_'+key,lambda:comp.checked_resource_stop(resource_pin,identity=identity,compatibility={'mock_only':True},guard=guard))
    comp.read_bound=original_read
finally:
    comp.checked_compatibility=original_checked
# Reproduce the old hard historical-resolver gate without inventing an accepted
# review. This fails before authorization lookup, and is preserved as a finding.
negative('prefix5_historical_resolver_gate_reproduced',
    lambda:oldprefix._Metadata(identity,resolver.MetadataResolver.__call__,guard,{'mock_only':True}))
# Scoped read-only fake archive. Exact public class/type and actual resolver/
# framing/hash logic execute, but storage authorization/catalog/XZ are mocked.
# No active namespace is opened and no production authority is fabricated.
store=index.IndexedOwnedFutureCheckpointStore.__new__(index.IndexedOwnedFutureCheckpointStore)
store.folder=AREA/'mock_store';store.folder.mkdir(exist_ok=True)
control=AREA/'control';control.mkdir(exist_ok=True)
owner=store.folder/'ownership.json';owner.write_bytes(b'{"fixture_only":true}\n')
store.marker_pin=record(owner);store.operational_compatibility={'mock_only':True}
store.policy={'maximum_auxiliary_header_bytes':65536,'maximum_auxiliary_raw_bytes':1024**2}
member_bytes=canonical({'kind':'fixture_control_metadata','all4RNG':'not_actual_tensor_state'})
member={'path':str(store.folder/'transient/p000000001/native/PRE.json'),'bytes':len(member_bytes),'sha256':hashlib.sha256(member_bytes).hexdigest()}
manifest={'files':[{'file':member}]}
item={'id':'p000000001','status':'archived','workspace':str(store.folder/'transient/p000000001'),
      'manifest_sha256':comp.digest(manifest),'archive':{'path':str(store.folder/'fixture.xz')}}
header=canonical({'manifest':manifest})
archive=importlib.import_module('asr18_BPE9223_indexed_checkpoint_storage_v1').AUX_MAGIC+struct.pack('<Q',len(header))+header+struct.pack('<Q',len(member_bytes))+member_bytes
loads=[]
store.locked=lambda:contextlib.nullcontext()
def load(*,allow_preparing_auxiliary=False):
    assert allow_preparing_auxiliary is True
    loads.append({'allow_unrelated_preparing':True,'target_status':item['status']})
    return {},None,{}
store._load=load
store._auxiliary=lambda *a,**kw:copy.deepcopy(item)
store._verify_auxiliary_archive=lambda value:copy.deepcopy(manifest)
store._auxiliary_open=lambda path:contextlib.nullcontext(io.BytesIO(archive))
original_resolver_checked=resolver.checked_compatibility
resolver.checked_compatibility=lambda *a,**kw:{'operational_sources':{'metadata_resolver':record(resolver.__file__)}}
try:
    resolved=resolver.MetadataResolver(store,control_root=control,guard=guard,
        metadata_phase=lambda _:contextlib.nullcontext(),identity=identity,operational_compatibility=store.operational_compatibility)
    def read_fake():
        with resolved(member) as stream:
            assert stream.read(member['bytes'])==member_bytes and stream.read(1)==b''
    positive('readonly_fake_completed_archive_with_unrelated_preparing',read_fake)
    assert not Path(member['path']).exists()
    positive('original_member_not_recreated',lambda:None)
    assert not list(control.iterdir())
    positive('owned_temporary_spool_closed_and_removed',lambda:None)
    item['status']='preparing'
    negative('preparing_target_rejected',read_fake)
    item['status']='archived'
    bad=copy.deepcopy(member);bad['sha256']='0'*64
    negative('mismatched_member_rejected',lambda:resolved._manifest('p000000001') and resolved._stream('p000000001',bad,lambda _:None))
    def real_metadata():
        pin=resource['current_checkpoint_metadata']
        with resolved(pin) as stream:
            raw=bytearray()
            while block:=stream.read(1024**2):raw.extend(block)
        assert hashlib.sha256(raw).hexdigest()==pin['sha256']
        parsed=json.loads(raw);assert len(parsed['tensors'])==854
    positive('real_p1_metadata_readonly_handle_SHA854_no_payload',real_metadata)
finally:resolver.checked_compatibility=original_resolver_checked
guard()
report={'kind':'ASR18_independent_operational_recovery_source_metadata_controls_v1','owner':'/root/label_bpe_audit',
    'passed':True,'actual_PID':os.getpid(),'checks':checks,'check_count':len(checks),'source_pins':{n:record(x.__file__) for n,x in modules.items()},
    'original_configuration':record(ROOT/'analysis/BPE9223_training_launch_v1/full_configuration_04/configuration_ready_for_final_execution_review.json'),
    'actual_resource_stop':resource_pin,'elapsed_seconds':time.monotonic()-started,'peak_private_bytes':g.peak,
    'hard_leaf_private_limit_bytes':512_000_000,'maximum_seconds':180,'GPU_audio_model_tensor_payload_used':False,
    'active_storage_cache_sources_modified':False,'real_compatibility_authority_checked':False,
    'mocked_scope':['authorization lookup for resource boundary test','fake storage catalog, archive stream and XZ boundary'],
    'genuine_scope':['immutable scientific identity and 432-source contract','real p1 scalar metadata, whole closed journal and captured process deaths',
                    'read-only genuine p1 metadata handle SHA/854 descriptor count','actual resolver frame/member hash/read-only spool behavior'],
    'production_or_execution_approval':False,'full_fresh_checkpoint_restore_proven':False,
    'prefix5_blocker':'actual resolver callback pinned to historical creator source; fixed in prefix6 source',
    'loads':loads}
(AREA/'metadata_controls_report_v1.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
g.publish(phase='metadata_controls_COMPLETE',check_count=len(checks),tensor_payload_bytes_read=0)
print(json.dumps({'passed':True,'check_count':len(checks),'elapsed_seconds':report['elapsed_seconds'],
                  'peak_private_bytes':g.peak,'report':record(AREA/'metadata_controls_report_v1.json')}))
