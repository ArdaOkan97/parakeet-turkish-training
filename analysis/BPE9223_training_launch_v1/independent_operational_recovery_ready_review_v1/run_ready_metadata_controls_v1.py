"""One finite genuine-authority resource20 request check; no payload/model/audio IO."""
import argparse, ctypes, hashlib, importlib, inspect, json, os, sys, time
from pathlib import Path
ROOT=Path(r'E:\gpu_training_scratchpad')
PROJECT=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20')
AREA=ROOT/'analysis/BPE9223_training_launch_v1/independent_operational_recovery_ready_review_v1'
sys.path[:0]=[str(PROJECT/'outputs/parakeet_turkish'),str(PROJECT/'work/research'),str(ROOT/'analysis/Turkish_labels_BPE_circumflex_v1')]
m=importlib.import_module('audit_and_fit_append_only_v5');m.AREA=AREA
g=m.Guard();g.max_seconds=180;g.maximum_bytes=64*1024**2
from ctypes import wintypes as w
class JobIO(ctypes.Structure):
    _fields_=[(x,ctypes.c_ulonglong) for x in ('r','w','o','rb','wb','ob')]
class Basic(ctypes.Structure):
    _fields_=[('p',ctypes.c_longlong),('j',ctypes.c_longlong),('flags',w.DWORD),('min',ctypes.c_size_t),('max',ctypes.c_size_t),('active',w.DWORD),('affinity',ctypes.c_size_t),('priority',w.DWORD),('scheduling',w.DWORD)]
class Limits(ctypes.Structure):
    _fields_=[('basic',Basic),('io',JobIO),('process_memory',ctypes.c_size_t),('job_memory',ctypes.c_size_t),('peak_process',ctypes.c_size_t),('peak_job',ctypes.c_size_t)]
limits=Limits();limits.basic.flags=0x100|0x200;limits.process_memory=512_000_000;limits.job_memory=512_000_000
api=ctypes.WinDLL('kernel32',use_last_error=True);api.SetInformationJobObject.argtypes=[w.HANDLE,ctypes.c_int,ctypes.c_void_p,w.DWORD];api.SetInformationJobObject.restype=w.BOOL
if not api.SetInformationJobObject(g.job,9,ctypes.byref(limits),ctypes.sizeof(limits)):raise ctypes.WinError(ctypes.get_last_error())
def guard():
    g.check();assert g.process.memory_info().private<512_000_000
def rec(path):
    p=Path(path).resolve();raw=p.read_bytes();return {'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def read(pin):
    assert rec(pin['path'])==pin
    return json.loads(Path(pin['path']).read_text(encoding='utf-8-sig'))
checks=[]
def positive(name,fn):guard();fn();checks.append({'name':name,'passed':True})
def negative(name,fn):
    guard()
    try:fn()
    except (AssertionError,ValueError,TypeError,KeyError) as exc:
        checks.append({'name':name,'passed':True,'rejected':type(exc).__name__,'message':str(exc)[:200]});return
    raise AssertionError('Negative accepted: '+name)
p=argparse.ArgumentParser();p.add_argument('--configuration',required=True);p.add_argument('--execution-review');p.add_argument('--request',required=True);args=p.parse_args()
began=time.monotonic()
handoff_path=ROOT/'analysis/BPE9223_training_launch_v1/operational_recovery_sources_v2/source_handoff_v2.json'
handoff_pin=rec(handoff_path);assert handoff_pin['sha256']=='5d2cc851ea5b374de848db8f4c72d6a7d2d1f09675c54b4f8efe11f92e7ef213'
handoff=read(handoff_pin)
for pin in handoff['sources'].values():assert rec(pin['path'])==pin
checks.append({'name':'all_seven_frozen_source_pins_exact','passed':True})
config_pin=rec(args.configuration);request_pin=rec(args.request)
config=read(config_pin);request=read(request_pin)
review_pin=None if args.execution_review is None else rec(args.execution_review)
if review_pin is None:
    # Explicit API test input only. This is not an accepted ROOT review and is
    # never passed to FullOwner or a process-launch path.
    execution_review={'resource_resume_request':request_pin}
else:
    execution_review=read(review_pin)
    assert execution_review['accepted'] is True and execution_review['reviewer']=='/root'
    assert execution_review['configuration']==config_pin and execution_review['resource_resume_request']==request_pin
comp=importlib.import_module('asr18_BPE9223_resource_operational_compatibility_v1')
controller=importlib.import_module('asr18_BPE9223_full_run_controller_v5')
prefix=importlib.import_module('asr18_BPE9223_persistence_prefix_v6')
parent=importlib.import_module('run_ASR18_BPE9223_full_run_owner_v5')
assert not any(name in sys.modules for name in ('torch','numpy','tokenizers','soundfile'))
compat_pin=config['operational_compatibility']
assert compat_pin=={'path':str(ROOT/'analysis/BPE9223_training_launch_v1/ROOT_BPE9223_resource_stop_operational_compatibility_review_v1.json'),
    'bytes':19713,'sha256':'01072a8a5305113f4ffe975ca1de9974dc994306afa913aea836187d474325f4'}
identity=config['full_run_identity']
assert comp.digest(identity)=='3381c4153f10cd5037f0e6f0ba760414db5a95b5ae240287f044e1fc7892e619'
compat=comp.checked_compatibility(compat_pin,identity,guard)
checks.append({'name':'genuine_ROOT_operational_compatibility_all_callbacks','passed':True})
resource=comp.checked_resource_stop(config['resource_native_closure'],identity=identity,compatibility=compat_pin,guard=guard)
checks.append({'name':'genuine_resource_TOOL1_p1_metadata_journal_quiescence','passed':True})
for name,pin in compat['operational_lifecycle_callback_sources'].items():
    assert rec(inspect.getsourcefile(getattr(controller.FullOwner,name)))==pin
checks.append({'name':'actual_lifecycle_callback_source5_not_legacy_stamp','passed':True})
for name in ('verify_pending_window','verify_durable_resume','verify_writer_authority'):
    assert rec(inspect.getsourcefile(getattr(controller.FullOwner,name)))==compat['operational_proof_callback_sources'][name]
assert rec(inspect.getsourcefile(parent.ParentGuard.close))==compat['legacy_guard_authority_source']
assert compat['legacy_guard_authority_source']==read(compat['original_configuration'])['parent_driver_source']
checks.append({'name':'actual_guard2_and_operational_parent5_sources_separate','passed':True})
snapshot=controller.checked_resource_resume_request(config_pin,execution_review,request_pin,guard=guard)
checks.append({'name':'genuine_complete_resource_resume_request_API_positive','passed':True})
from bpe_streaming_checkpoint_semantics_v2 import select,decode_scalar_tree
metadata=read(resource['current_checkpoint_metadata'])
saved=decode_scalar_tree(select(metadata['tree'],'sampler'))
roundtrip=json.loads(json.dumps(saved))
assert saved!=roundtrip and comp.digest(saved)==comp.digest(roundtrip)
assert controller.normalize_schedule_state(roundtrip)==saved==snapshot['sampler_state']
checks.append({'name':'real_JSON_sampler_tuple_transport_fixed_without_digest_change','passed':True})
assert snapshot['proof_sequence']==2 and snapshot['native_sequence']==1 and snapshot['recovery_sequence']==0
assert snapshot['reserved_preparation']['intent']==resource['reserved_unconsumed_p2']
assert snapshot['current_ack'] is None and snapshot['closed_recovery'] is None
checks.append({'name':'original20_counters_exact_pending1_and_unconsumed_p2_preserved','passed':True})
owner=controller.FullOwner.__new__(controller.FullOwner)
view=prefix._Metadata(identity,owner.resolve_member,guard,compat_pin)
assert view.sources==identity['persistent_prefix_sources']
checks.append({'name':'prefix6_actual_controller5_resolver_and_immutable_legacy_map','passed':True})
old=importlib.import_module('asr18_BPE9223_full_run_controller_v2')
negative('prefix6_reject_actual_legacy_callback',lambda:prefix._Metadata(identity,old.FullOwner.resolve_member,guard,compat_pin))
bad=dict(identity);bad['owner_validation_only']=True
negative('compatibility_reject_scientific_identity_change',lambda:comp.checked_compatibility(compat_pin,bad,guard))
# Examine the exact frozen successor AST for the final durable compatibility
# argument and first recovery-before-loop ordering. No callbacks are run.
import ast
tree=ast.parse(Path(controller.__file__).read_text(encoding='utf-8-sig'))
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='FullOwner')
run=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='run')
assert isinstance(run.body[0],ast.If) and run.body[0].test.attr=='resource_initial_pending'
assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='_recover_resource_stop' for n in ast.walk(run.body[0]))
for call in [n for n in ast.walk(run) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='PackedDurableLifecycleAdapter']:
    assert 'operational_compatibility' in {x.arg for x in call.keywords}
checks.append({'name':'first_fresh_recovery_before_loop_and_final_durable_compatibility_argument','passed':True})
guard()
report={'kind':'ASR18_independent_genuine_resource20_ready_metadata_controls_v1','owner':'/root/label_bpe_audit',
    'passed':True,'producer':g.main_identity,'configuration':config_pin,'execution_review':review_pin,'resume_request':request_pin,
    'genuine_compatibility':compat_pin,'source_handoff':handoff_pin,'check_count':len(checks),'checks':checks,
    'elapsed_seconds':time.monotonic()-began,'peak_private_bytes':g.peak,'hard_private_limit_bytes':512_000_000,'maximum_seconds':180,
    'tensor_payload_bytes_read':0,'GPU_audio_model_used':False,'active_storage_sources_cache_mutated':False,
    'mocked_authority_used':False,'fixture_request_descriptor_only':review_pin is None,
    'execution_authority_checked':review_pin is not None,'production_or_execution_approval':False,
    'actual_fresh_GPU_checkpoint_recovery_proven':False,
    'scientific_identity_sha256':comp.digest(identity),'original_source_manifest':compat['original_source_manifest'],
    'source_review_scope':'Controller5 source delta and genuine readonly compatibility/resource/request metadata only',
    'scope_limits':['No numerical tensor restore performed','No actual owner/controller/GPU callbacks run','No live-store _load/payload rehash performed']}
out=AREA/'genuine_ready_metadata_report_v1.json';assert not out.exists();out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
g.publish(phase='genuine_ready_metadata_COMPLETE',check_count=len(checks),tensor_payload_bytes_read=0)
print(json.dumps({'passed':True,'report':rec(out),'check_count':len(checks),'elapsed_seconds':report['elapsed_seconds'],'peak_private_bytes':g.peak}))
