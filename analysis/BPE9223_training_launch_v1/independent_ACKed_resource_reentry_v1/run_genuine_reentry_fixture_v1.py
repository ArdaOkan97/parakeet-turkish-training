"""Real read-only metadata API reentry, not a production owner or authority.

The __new__ object is only an IO-method consumer. No owner constructor,
protect/reserve/save/archive/ACK/publication, native or CPU-child launch occurs.
Actual resolver-selected member spools may be created and owned-unlinked under
the original control directory; all authoritative metadata must stay identical.
"""
import argparse, ctypes, gc, hashlib, importlib, json, os, sys, threading, time
from pathlib import Path

ROOT = Path(r'E:\gpu_training_scratchpad')
PROJECT = Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20')
AREA = ROOT/'analysis/BPE9223_training_launch_v1/independent_ACKed_resource_reentry_v1'
sys.path[:0] = [str(PROJECT/'outputs/parakeet_turkish'), str(PROJECT/'work/research'),
               str(ROOT/'analysis/Turkish_labels_BPE_circumflex_v1')]
m = importlib.import_module('audit_and_fit_append_only_v5'); m.AREA = AREA
g = m.Guard()
from ctypes import wintypes as w
class IO(ctypes.Structure):
    _fields_ = [(x, ctypes.c_ulonglong) for x in ('r','w','o','rb','wb','ob')]
class Basic(ctypes.Structure):
    _fields_ = [('p',ctypes.c_longlong),('j',ctypes.c_longlong),('flags',w.DWORD),
        ('min',ctypes.c_size_t),('max',ctypes.c_size_t),('active',w.DWORD),
        ('affinity',ctypes.c_size_t),('priority',w.DWORD),('scheduling',w.DWORD)]
class Limits(ctypes.Structure):
    _fields_ = [('basic',Basic),('io',IO),('process_memory',ctypes.c_size_t),
        ('job_memory',ctypes.c_size_t),('peak_process',ctypes.c_size_t),('peak_job',ctypes.c_size_t)]
limits=Limits();limits.basic.flags=0x100|0x200
limits.process_memory=512_000_000;limits.job_memory=512_000_000
api=ctypes.WinDLL('kernel32',use_last_error=True)
api.SetInformationJobObject.argtypes=[w.HANDLE,ctypes.c_int,ctypes.c_void_p,w.DWORD]
api.SetInformationJobObject.restype=w.BOOL
if not api.SetInformationJobObject(g.job,9,ctypes.byref(limits),ctypes.sizeof(limits)):
    raise ctypes.WinError(ctypes.get_last_error())
began=time.monotonic();timer=threading.Timer(180,lambda:os._exit(75));timer.daemon=True;timer.start()
MAX_IO=16*1024**3
def guard():
    g.check()
    counters=g.process.io_counters()
    assert time.monotonic()-began<180
    assert g.process.memory_info().private<512_000_000
    assert counters.read_bytes+counters.write_bytes-g.initial_IO<=MAX_IO
def record(path):
    path=Path(path).resolve();hashed=hashlib.sha256();before=path.stat()
    with path.open('rb') as stream:
        while block:=stream.read(1024**2):guard();hashed.update(block)
    assert (before.st_size,before.st_mtime_ns)==(path.stat().st_size,path.stat().st_mtime_ns)
    return {'path':str(path),'bytes':before.st_size,'sha256':hashed.hexdigest()}
def read(pin):
    assert record(pin['path'])==pin
    return json.loads(Path(pin['path']).read_text(encoding='utf-8-sig'))
def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
        ensure_ascii=False,allow_nan=False).encode()).hexdigest()
checks=[]
def check(name, value):
    guard();assert value,name;checks.append({'name':name,'passed':True})
def actual_module(pin):
    assert record(pin['path'])==pin
    module=importlib.import_module(Path(pin['path']).stem)
    assert record(module.__file__)==pin
    return module
args=argparse.ArgumentParser()
args.add_argument('--config',required=True);args.add_argument('--request',required=True)
args.add_argument('--compatibility',required=True)
a=args.parse_args()
config_pin=record(a.config);request_pin=record(a.request);compat_pin=record(a.compatibility)
c=read(config_pin);request=read(request_pin)
check('genuine_config_request_compatibility_pins',c['operational_compatibility']==compat_pin
    and request['operational_configuration']==config_pin and request['operational_compatibility']==compat_pin)
review=read(compat_pin)
check('ROOT_accepted_compatibility_not_fixture_authority',review['accepted'] is True and review['reviewer']=='/root')
controller=actual_module(review['operational_sources']['controller'])
prefix=actual_module(review['operational_sources']['prefix'])
old_controller=importlib.import_module('asr18_BPE9223_full_run_controller_v6')
original_parent=importlib.import_module('run_ASR18_BPE9223_full_run_owner_v2')
opmetadata=importlib.import_module('asr18_BPE9223_operational_metadata_v2')
check('pure_startup_without_preconfigured_archive_reader',opmetadata._RESOLVER is None and opmetadata._BINDING is None)
snapshot=controller.checked_acknowledged_resource_resume_request(config_pin,
    {'resource_resume_request':request_pin},request_pin,guard=guard)
check('pure_new_checker_before_global_reader_configuration',snapshot['last_window_index']==0
    and read(snapshot['current_capture']['pointer'])['committed_updates']==20)
check('pure_checker_preserves_cumulative_actual_counters',
    [snapshot[x] for x in ('actual_owner_invocation_count','native_sequence','recovery_sequence','recovery_control_sequence')]==[3,1,1,2])
check('prepared_p2_preserves_existing_plan_and_intent',snapshot['prepared_CPU']==request['prepared_CPU']
    and snapshot['prepared_CPU']['workspace']['id']=='p000000002')
check('checker_did_not_initialize_hidden_reader',opmetadata._RESOLVER is None and opmetadata._BINDING is None)
stopped=read(c['prior_acknowledged_resource_stop'])
pins=[stopped[key] for key in ('checkpoint_pointer','storage_state','durable_latest','journal','pool_registry','p2_plan',
    'resource_prefix_acknowledgement','coverage')]
pins += stopped['p2_metadata_files_preserved']
pins += [c['operational_storage_owner']]
state=read(stopped['storage_state'])
# state.catalog is the logical commit root, not a file record3.
pins += [record(Path(c['checkpoint_storage_namespace'])/'catalog.sqlite')]
deduplicated={pin['path']:pin for pin in pins}
before_pins=[record(path) for path in sorted(deduplicated)]
check('all_authoritative_inputs_match_original_stopped_pins',
    all(pin==deduplicated[pin['path']] for pin in before_pins))
control=Path(c['operational_control_root'])
before_spools={str(p.resolve()) for p in control.glob('owned_member_*.tmp')}
check('no_preexisting_selected_member_spools',not before_spools)
guard_folder=AREA/'actual_original_parent_guard';guard_folder.mkdir(exist_ok=False)
parent=None;resolver=None;coverage=None
try:
    parent=original_parent.ParentGuard(guard_folder)
    parent.configure_control_quota(control,c['control_log_policy'],identity=c['full_run_identity'],create=False)
    pool_state_before=parent.control_pool.state_pin
    check('original_readonly_pool_reopen_exact_source_and_pin',pool_state_before==stopped['pool_registry'])
    resolver=opmetadata.configure(storage_owner=c['operational_storage_owner'],control_root=control,
        identity=c['full_run_identity'],operational_compatibility=compat_pin,
        current_native_resource_closure=None,guard=parent.check,metadata_phase=parent.metadata_phase)
    check('actual_resolver3_and_store2',Path(type(resolver).__module__.replace('.','/')).name=='asr18_BPE9223_metadata_resolver_v3'
        and type(resolver.store).__module__=='asr18_BPE9223_indexed_checkpoint_storage_v2')
    consumer=controller.FullOwner.__new__(controller.FullOwner)
    consumer.parent=parent;consumer.resolver=resolver
    check('uninitialized_IO_consumer_not_real_owner',set(consumer.__dict__)=={'parent','resolver'})
    new_callback=consumer.resolve_member
    old_callback=old_controller.FullOwner.resolve_member.__get__(consumer,type(consumer))
    coverage=prefix.verify_acknowledged_resource_reentry(identity=c['full_run_identity'],
        operational_compatibility=compat_pin,resolve_member=new_callback,
        legacy_resolve_member=old_callback,guard=parent.check)
    check('genuine_prefix7_reentry_metadata_positive',coverage['passed'] is True
        and coverage['kind']=='ASR18_verified_prior_ACKed_resource20_reentry_metadata_v1')
    check('actual_prior_ACK_head_and_current_checkpoint_preserved',
        coverage['checkpoint_pointer']==stopped['checkpoint_pointer']
        and coverage['actual_fresh_recovery0']==stopped['actual_normal_fresh_recovery0']
        and coverage['acknowledgement']==stopped['resource_prefix_acknowledgement'])
    check('actual_archived_nextdraw_semantics_after_reader_initialized',
        digest(opmetadata.read_bound(snapshot['current_capture']['next_draw_packet'],parent.check))
        ==snapshot['next_draw_packet_sha256'])
    check('no_fake_new_recovery_or_draw_in_report',coverage['new_GPU_restorations_draws_updates_or_namespace_writes'] is False)
    check('pool_state_unchanged_after_resolver_and_prefix',parent.control_pool.state_pin==pool_state_before
        and record(pool_state_before['path'])==pool_state_before)
    check('no_residual_selected_member_spools',{str(p.resolve()) for p in control.glob('owned_member_*.tmp')}==before_spools)
finally:
    if resolver is not None:resolver.store.close()
    if parent is not None:parent.close()
after_pins=[record(path) for path in sorted(deduplicated)]
check('all_authoritative_state_catalog_pool_ACK_journal_and_p2_exact',after_pins==before_pins)
check('no_numerical_model_audio_GPU_import',not any(x in sys.modules for x in ('torch','numpy','tokenizers','soundfile')))
guard()
report={'kind':'ASR18_independent_genuine_ACKed20_resource_reentry_metadata_API_fixture_v1',
    'passed':True,'owner':'/root/label_bpe_audit','producer':g.main_identity,'source':record(__file__),
    'configuration':config_pin,'resource_resume_request':request_pin,'operational_compatibility':compat_pin,
    'controller_source':record(controller.__file__),'prefix_source':record(prefix.__file__),
    'original_parent_guard_source':record(original_parent.__file__),'checks':checks,'check_count':len(checks),
    'actual_prior_reentry_result':coverage,'authoritative_before_and_after_identical':before_pins,
    'original_parent_guard_closure':record(guard_folder/'owner_guard_completion.json'),
    'transient_spool_lifecycle_exercised':True,'transient_spool_scope':'Existing actual resolver3 owned temporary files may be created/closed/unlinked under RUN/control; no persistent authoritative writes.',
    'no_residual_owned_member_spools':True,'fixture_IO_consumer_not_real_owner':True,
    'actual_owner_protect_reserve_save_archive_ACK_or_publication_operations':False,
    'actual_CPU_GPU_model_audio_or_tensor_restore_operations':False,
    'actual_existing_store_integrity_reads_may_hash_encoded_payload_bytes':True,
    'production_execution_approval_or_absolute64MB_fit_claimed':False,
    'elapsed_seconds':time.monotonic()-began,'peak_private_bytes':g.peak,
    'maximum_seconds':180,'hard_private_bytes':512_000_000,'maximum_actual_process_IO_bytes':MAX_IO,
    'actual_process_IO_bytes':g.process.io_counters().read_bytes+g.process.io_counters().write_bytes-g.initial_IO}
out=AREA/'genuine_reentry_report_v1.json';assert not out.exists()
out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
g.publish(phase='genuine_ACKed_resource_reentry_COMPLETE',check_count=len(checks),GPU_used=False)
timer.cancel()
print(json.dumps({'passed':True,'report':record(out),'checks':len(checks),
    'seconds':report['elapsed_seconds'],'peak_private_bytes':g.peak}))
