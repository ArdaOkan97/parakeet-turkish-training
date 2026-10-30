"""Real isolated pool protocol fixture; no production pool, worker or GPU is opened.

Uses the actual original ParentGuard2/callback wrapper and original pool1 create
path. The wrapper delegates to an explicit fixture-only protection verifier.
Reopening and reserve use genuine ROOT compatibility and the unmodified pool2
implementation. No reserve/_save/compatibility method is mocked or bypassed.
"""
import argparse, copy, ctypes, hashlib, importlib, inspect, json, os, sys, threading, time
from pathlib import Path
ROOT=Path(r'E:\gpu_training_scratchpad')
PROJECT=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20')
AREA=ROOT/'analysis/BPE9223_training_launch_v1/independent_worker_pool_protocol_review_v1'
FIXTURE=ROOT/'runs/BPE9223_worker_control_pool_PROTOCOL_FIXTURE_v1'
sys.path[:0]=[str(PROJECT/'outputs/parakeet_turkish'),str(PROJECT/'work/research'),str(ROOT/'analysis/Turkish_labels_BPE_circumflex_v1')]
m=importlib.import_module('audit_and_fit_append_only_v5');m.AREA=AREA
g=m.Guard();g.max_seconds=180;g.maximum_bytes=256*1024**2
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
began=time.monotonic();timer=threading.Timer(180,lambda:os._exit(75));timer.daemon=True;timer.start()
def guard():
    g.check();assert time.monotonic()-began<180 and g.process.memory_info().private<512_000_000
    counters=g.process.io_counters()
    assert counters.read_bytes+counters.write_bytes-g.initial_IO<=256*1024**2
def record(path):
    path=Path(path).resolve();raw=path.read_bytes()
    return {'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def read(pin):
    assert record(pin['path'])==pin
    return json.loads(Path(pin['path']).read_text(encoding='utf-8-sig'))
def write_new(path,value):
    path=Path(path);assert path.resolve().is_relative_to(FIXTURE) and not path.exists()
    raw=json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False).encode('utf8')
    with path.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    return record(path)
checks=[]
def check(name,value):
    guard();assert value,name;checks.append({'name':name,'passed':True})
def reject_unchanged(name,fn,pool):
    guard();before=record(pool.registry_path)
    try:fn()
    except (ValueError,AssertionError,KeyError,TypeError) as exc:
        assert record(pool.registry_path)==before
        checks.append({'name':name,'passed':True,'registry_unchanged':True,'rejected':type(exc).__name__,'message':str(exc)[:240]});return
    raise AssertionError('Negative accepted: '+name)

p=argparse.ArgumentParser();p.add_argument('--compatibility',required=True);p.add_argument('--configuration');p.add_argument('--request');args=p.parse_args()
assert bool(args.configuration)==bool(args.request)
comp=importlib.import_module('asr18_BPE9223_resource_operational_compatibility_v1')
pool1=importlib.import_module('asr18_worker_control_pool_v1')
pool2=importlib.import_module('asr18_worker_control_pool_v2')
parent2=importlib.import_module('run_ASR18_BPE9223_full_run_owner_v2')
check('pool2_exact_frozen_source',record(pool2.__file__)['sha256']=='51d858cdc4d4f5b291648e2495a98a38b47654ad9888d2e839b13cc4da767e40')
compat_pin=record(args.compatibility);raw_compat=read(compat_pin)
original=read(raw_compat['original_configuration']);identity=original['full_run_identity']
compat=comp.checked_compatibility(compat_pin,identity,guard)
check('genuine_ROOT_compatibility_no_mock',compat['accepted'] is True and compat['reviewer']=='/root')
check('scientific_identity_for_protocol_only',comp.digest(identity)=='3381c4153f10cd5037f0e6f0ba760414db5a95b5ae240287f044e1fc7892e619')
authority=record(parent2.__file__);new_parent=compat['operational_sources']['parent'];worker=compat['operational_sources']['runtime']
check('original_guard2_authority_and_new_parent_distinct',authority==compat['legacy_guard_authority_source'] and authority!=new_parent)
check('actual_pool2_and_callbacks_in_genuine_closure',compat['operational_sources']['worker_control_pool']==record(pool2.__file__))
check('no_heavy_runtime_imported',not any(x in sys.modules for x in ('torch','numpy','tokenizers','soundfile')))
check('fixture_is_new_separate_E_runs_namespace',not FIXTURE.exists() and FIXTURE.parent==ROOT/'runs' and FIXTURE!=Path(original['run_namespace']))
FIXTURE.mkdir();control=FIXTURE/'control';control.mkdir();owner_folder=FIXTURE/'owner_invocation';owner_folder.mkdir()
fixture_note=write_new(control/'PROTOCOL_FIXTURE_ONLY.json',{
    'kind':'ASR18_isolated_real_worker_pool_protocol_fixture_v1','protocol_fixture_only':True,
    'scientific_identity_used_for_compatibility_checks_only':True,'no_production_authority':True,
    'fixture_namespace':str(FIXTURE),'actual_training_namespace_opened_or_mutated':False,
    'real_process_GPU_audio_model_checkpoint_or_tensor_operations':False})
old_config=write_new(control/'old_parent_worker_configuration.json',{'fixture_only':True,'worker_not_launched':True,'slot':'original'})
new_config=write_new(control/'new_parent_worker_configuration.json',{'fixture_only':True,'worker_not_launched':True,'slot':'operational'})
evidence=write_new(control/'complete_protection_fixture_evidence.json',{'fixture_only':True,'complete_protection_of_all_fixture_live_slots':True})

class ExplicitFixtureVerifier:
    def __init__(self):self.pool=None;self.calls=[]
    def verify_worker_control_protection(self,request):
        guard();pool=self.pool;state=pool._load()
        assert request['operation']=='verify_complete_worker_control_protection'
        assert request['identity']==identity and request['owner']==pool.owner_pin
        assert request['revision']==state['revision'] and request['live']==state['live']
        assert request['folders']==sorted(x['folder'] for x in state['live'])
        assert request['files']==[old_config,new_config,fixture_note] and request['evidence']==evidence
        assert pool.control==control and pool.folder.is_relative_to(FIXTURE)
        assert read(evidence)['fixture_only'] is True
        self.calls.append({'operation':request['operation'],'revision':request['revision'],'folders':request['folders']})
        return True
verifier=ExplicitFixtureVerifier();parent2.install_owner(verifier)
pg=None;pool=None;owner_pin=None;snapshots=[];guard_completion=None
try:
    # Actual ParentGuard2 starts its actual paired-memory ResourceGuard/watchdog.
    # Its inherited configure method uses actual original pool1 create=True and
    # source2 control_guard admission; this is not a fabricated creator record.
    pg=parent2.ParentGuard(owner_folder)
    pg.configure_control_quota(control,parent2.CONTROL_POLICY,identity=identity,create=True)
    pool=pg.control_pool;verifier.pool=pool
    check('original_pool1_created_by_actual_guard2',type(pool) is pool1.WorkerControlPool and pool.authority_source==authority)
    owner_pin=record(pool.owner_path);original_owner=read(owner_pin)
    check('original_owner1_component_and_authority_truthful',original_owner['component_source']==record(pool1.__file__) and original_owner['authority_source']==authority)
    check('admission_and_callback_actual_source2',record(inspect.getsourcefile(pg.control_guard))==authority and record(inspect.getsourcefile(parent2.verify_worker_control_protection))==authority)
    pool.protect([],files=[old_config,new_config,fixture_note],evidence=evidence,verify_protection=parent2.verify_worker_control_protection)
    first=pool.reserve(owner_folder/'native_000001',configuration=old_config,worker_source=worker,parent_source=authority)
    pg.worker_slots=list(pool.live_folders());snapshots.append(pool.snapshot_binding())
    check('original_parent2_real_reservation_and_registry_save',first['reservation']==1 and pool._state['reservation_count']==1 and pool.revision==2)
    pool.close();pg.control_pool=None;pool=None

    # The only new pool methods here are genuine pool2 __init__/reserve. The
    # original creator/registry/atomic _save/protection/quotas stay unchanged.
    pool=pool2.WorkerControlPool(control,identity=identity,policy=parent2.CONTROL_POLICY,
        authority_source=authority,guard=pg.pool_guard,metadata_phase=pg.metadata_phase,
        admit_control=pg.control_guard,operational_compatibility=compat_pin,create=False)
    pg.control_pool=pool;verifier.pool=pool;pg.worker_slots=list(pool.live_folders())
    check('operational_reopen_owner_and_cumulative_counters_preserved',record(pool.owner_path)==owner_pin and pool._state['reservation_count']==1 and pool.revision==2)
    target=owner_folder/'recovery_000001'
    reject_unchanged('reserve_without_fresh_protection_rejected',lambda:pool.reserve(target,configuration=new_config,worker_source=worker,parent_source=new_parent),pool)
    reject_unchanged('wrong_original_parent2_rejected',lambda:pool.reserve(target,configuration=new_config,worker_source=worker,parent_source=authority),pool)
    bad_parent=copy.deepcopy(new_parent);bad_parent['sha256']='0'*64
    reject_unchanged('wrong_operational_parent_pin_rejected',lambda:pool.reserve(target,configuration=new_config,worker_source=worker,parent_source=bad_parent),pool)
    pool.protect([x['folder'] for x in pool._state['live']],files=[old_config,new_config,fixture_note],
        evidence=evidence,verify_protection=parent2.verify_worker_control_protection)
    before=pool.snapshot_binding();second=pool.reserve(target,configuration=new_config,worker_source=worker,parent_source=new_parent)
    pg.worker_slots=list(pool.live_folders());after=pool.snapshot_binding();snapshots.extend([before,after])
    check('new_parent6_real_reserve_succeeds_without_directory_or_Popen',second['parent_source']==new_parent and second['reservation']==2 and not target.exists())
    check('real_save_advances_revision_and_reservation_once',after['reservation_count']==before['reservation_count']+1==2 and pool.revision==4 and before['registry']!=after['registry'])
    check('owner1_unchanged_after_operational_save',record(pool.owner_path)==owner_pin)
    bad_policy=copy.deepcopy(parent2.CONTROL_POLICY);bad_policy['fixed_control_allocation_cap_bytes']+=1
    reject_unchanged('quota_policy_change_rejected_without_registry_mutation',lambda:pool2.WorkerControlPool(
        control,identity=identity,policy=bad_policy,authority_source=authority,
        guard=pg.pool_guard,metadata_phase=pg.metadata_phase,admit_control=pg.control_guard,
        operational_compatibility=compat_pin,create=False),pool)
    reject_unchanged('duplicate_registered_folder_rejected',lambda:pool.reserve(target,configuration=new_config,worker_source=worker,parent_source=new_parent),pool)
    bad_config=copy.deepcopy(new_config);bad_config['sha256']='0'*64
    reject_unchanged('wrong_configuration_pin_rejected',lambda:pool.reserve(owner_folder/'recovery_000002',configuration=bad_config,worker_source=worker,parent_source=new_parent),pool)
    pool.close();pg.control_pool=None;pool=None
    pool=pool2.WorkerControlPool(control,identity=identity,policy=parent2.CONTROL_POLICY,
        authority_source=authority,guard=pg.pool_guard,metadata_phase=pg.metadata_phase,
        admit_control=pg.control_guard,operational_compatibility=compat_pin,create=False)
    pg.control_pool=pool;verifier.pool=pool;pg.worker_slots=list(pool.live_folders())
    reopened=pool.snapshot_binding();snapshots.append(reopened)
    check('second_reopen_same_owner_full_cumulative_registry_snapshot',reopened==after and record(pool.owner_path)==owner_pin)
    reject_unchanged('second_supervisor_requires_new_protection',lambda:pool.reserve(owner_folder/'recovery_000002',configuration=new_config,worker_source=worker,parent_source=new_parent),pool)
    pool.protect([x['folder'] for x in pool._state['live']],files=[old_config,new_config,fixture_note],
        evidence=evidence,verify_protection=parent2.verify_worker_control_protection)
    check('original_caps_and_policy_unchanged',pool2.CAPS==pool1.CAPS==original_owner['caps'] and pool.policy==original_owner['policy']==parent2.CONTROL_POLICY)
    final=pool.snapshot_binding();snapshots.append(final)
    check('protection_save_keeps_reservation_and_archive_counters',final['reservation_count']==2 and final['archive_count']==0 and pool.revision==5)
    check('original_registry_save_implementation_is_actual_pool1',inspect.getsourcefile(type(pool)._save)==pool1.__file__)
    check('explicit_fixture_wrapper_verified_each_protection',len(verifier.calls)==3)
finally:
    if pg is not None:
        # Actual original guard2 close joins its watchdog and closes this exact
        # fixture pool. No subprocess or GPU exit is represented by this proof.
        pg.close();guard_completion=record(owner_folder/'owner_guard_completion.json')
    elif pool is not None:pool.close()

preflight=None
if args.configuration:
    config_pin=record(args.configuration);request_pin=record(args.request);config=read(config_pin)
    assert config['operational_compatibility']==compat_pin
    controller=importlib.import_module(Path(compat['operational_sources']['controller']['path']).stem)
    snapshot=controller.checked_resource_resume_request(config_pin,{'resource_resume_request':request_pin},request_pin,guard=guard)
    check('genuine_new_configuration_and_resource_request_API_pass',snapshot['current_ack'] is None and snapshot['closed_recovery'] is None)
    preflight={'configuration':config_pin,'request':request_pin,'controller_source':record(controller.__file__),
        'proof_sequence':snapshot['proof_sequence'],'native_sequence':snapshot['native_sequence'],
        'recovery_sequence':snapshot['recovery_sequence'],'resource_sampler_canonical_sha256':comp.digest(snapshot['sampler_state']),
        'fixture_request_descriptor_only':True,'execution_authority_claimed':False}
guard();check('no_heavy_runtime_at_finish',not any(x in sys.modules for x in ('torch','numpy','tokenizers','soundfile')))
report={'kind':'ASR18_independent_real_isolated_operational_worker_pool_protocol_fixture_v1','owner':'/root/label_bpe_audit',
    'passed':True,'producer':g.main_identity,'source':record(__file__),'compatibility':compat_pin,
    'pool2_source':record(pool2.__file__),'original_pool1_source':record(pool1.__file__),'original_authority2':authority,
    'new_operational_parent':new_parent,'fixture_namespace':str(FIXTURE),'protocol_fixture_only':True,
    'no_full_production_authority_claimed':True,'actual_original_ParentGuard2_joined_completion':guard_completion,
    'original_owner1_unchanged':record(Path(owner_pin['path']))==owner_pin,'original_owner1':owner_pin,
    'check_count':len(checks),'checks':checks,'snapshots':snapshots,'fixture_verifier_calls':verifier.calls,
    'new_full_configuration_preflight':preflight,'elapsed_seconds':time.monotonic()-began,'peak_private_bytes':g.peak,
    'hard_leaf_limit_bytes':512_000_000,'maximum_seconds':180,'maximum_process_IO_bytes':256*1024**2,
    'actual_process_IO_bytes':g.process.io_counters().read_bytes+g.process.io_counters().write_bytes-g.initial_IO,
    'mocked_pool_reserve_save_or_compatibility':False,
    'actual_GPU_audio_model_tensor_or_checkpoint_operations':False,'actual_training_namespace_opened_or_mutated':False,
    'worker_processes_launched':0,'pool_archive_or_retirement_exercised':False,
    'note':'Tiny absent worker directories are genuine registered fixture reservations; no process0 or original artifact adoption is fabricated.'}
out=AREA/'pool_protocol_fixture_report_v1.json';assert not out.exists();out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
g.publish(phase='real_pool_protocol_fixture_COMPLETE',check_count=len(checks),tensor_payload_bytes_read=0)
timer.cancel();print(json.dumps({'passed':True,'check_count':len(checks),'report':record(out),'elapsed_seconds':report['elapsed_seconds'],'peak_private_bytes':g.peak}))
