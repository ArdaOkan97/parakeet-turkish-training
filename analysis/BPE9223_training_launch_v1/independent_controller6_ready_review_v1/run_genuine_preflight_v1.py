"""Bounded genuine config/request API check; no owner, pool, model or GPU job."""
import argparse, ctypes, hashlib, importlib, inspect, json, os, sys, threading, time
from pathlib import Path
ROOT=Path(r'E:\gpu_training_scratchpad');PROJECT=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20')
AREA=ROOT/'analysis/BPE9223_training_launch_v1/independent_controller6_ready_review_v1'
sys.path[:0]=[str(PROJECT/'outputs/parakeet_turkish'),str(PROJECT/'work/research'),str(ROOT/'analysis/Turkish_labels_BPE_circumflex_v1')]
m=importlib.import_module('audit_and_fit_append_only_v5');m.AREA=AREA
g=m.Guard();g.max_seconds=180;g.maximum_bytes=128*1024**2
from ctypes import wintypes as w
class JobIO(ctypes.Structure):_fields_=[(x,ctypes.c_ulonglong) for x in ('r','w','o','rb','wb','ob')]
class Basic(ctypes.Structure):_fields_=[('p',ctypes.c_longlong),('j',ctypes.c_longlong),('flags',w.DWORD),('min',ctypes.c_size_t),('max',ctypes.c_size_t),('active',w.DWORD),('affinity',ctypes.c_size_t),('priority',w.DWORD),('scheduling',w.DWORD)]
class Limits(ctypes.Structure):_fields_=[('basic',Basic),('io',JobIO),('process_memory',ctypes.c_size_t),('job_memory',ctypes.c_size_t),('peak_process',ctypes.c_size_t),('peak_job',ctypes.c_size_t)]
limits=Limits();limits.basic.flags=0x100|0x200;limits.process_memory=512_000_000;limits.job_memory=512_000_000
api=ctypes.WinDLL('kernel32',use_last_error=True);api.SetInformationJobObject.argtypes=[w.HANDLE,ctypes.c_int,ctypes.c_void_p,w.DWORD];api.SetInformationJobObject.restype=w.BOOL
if not api.SetInformationJobObject(g.job,9,ctypes.byref(limits),ctypes.sizeof(limits)):raise ctypes.WinError(ctypes.get_last_error())
began=time.monotonic();timer=threading.Timer(180,lambda:os._exit(75));timer.daemon=True;timer.start()
def guard():
    g.check();assert time.monotonic()-began<180 and g.process.memory_info().private<512_000_000
    io=g.process.io_counters();assert io.read_bytes+io.write_bytes-g.initial_IO<128*1024**2
def rec(path):
    p=Path(path).resolve();raw=p.read_bytes();return {'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def read(pin):
    assert rec(pin['path'])==pin
    return json.loads(Path(pin['path']).read_text(encoding='utf-8-sig'))
checks=[]
def check(name,value):guard();assert value,name;checks.append({'name':name,'passed':True})
p=argparse.ArgumentParser();p.add_argument('--configuration',required=True);p.add_argument('--request',required=True);args=p.parse_args()
config_pin=rec(args.configuration);request_pin=rec(args.request);c=read(config_pin);request=read(request_pin)
comp=importlib.import_module('asr18_BPE9223_resource_operational_compatibility_v1')
compat=comp.checked_compatibility(c['operational_compatibility'],c['full_run_identity'],guard)
check('genuine_ROOT_compatibility2',c['operational_compatibility']['sha256']=='389556622513452ddc7b91ac4159df2960ff7b52179c43be69d3646e75351966')
check('original_scientific_identity_and432_contract_unchanged',comp.digest(c['full_run_identity'])=='3381c4153f10cd5037f0e6f0ba760414db5a95b5ae240287f044e1fc7892e619' and c['full_run_identity']['full_run_source_contract']['sha256']=='463c220bb3279c3bffdfe7fdf6e019f0d5e6cbf973c2dfed6b8a99012cab6e84')
check('exact_controller6',compat['operational_sources']['controller']['sha256']=='1f4edb5550c71e3b797cedf88e098d4a55dfc1b082be996092ebe62ea7ba21b7')
controller=importlib.import_module(Path(compat['operational_sources']['controller']['path']).stem)
check('actual_controller_source_matches_genuine_review',rec(controller.__file__)==compat['operational_sources']['controller'])
check('no_heavy_runtime_import',not any(x in sys.modules for x in ('torch','numpy','tokenizers','soundfile')))
snapshot=controller.checked_resource_resume_request(config_pin,{'resource_resume_request':request_pin},request_pin,guard=guard)
check('genuine_configuration_request_API_pass',snapshot['current_ack'] is None and snapshot['closed_recovery'] is None)
check('real20_p1_and_original_proof2_preserved',snapshot['proof_sequence']==2 and snapshot['native_sequence']==1 and snapshot['last_window_index']==0)
check('failed_control1_is_not_GPU_recovery',snapshot['actual_owner_invocation_count']==2 and snapshot['recovery_control_sequence']==1 and snapshot['recovery_sequence']==0)
check('same_unconsumed_p2',snapshot['reserved_preparation']['intent']['id']=='p000000002' and snapshot['reserved_preparation']['workspace']['id']=='p000000002')
check('failed_operational_supervisor_honestly_bound',request['prior_operational_failure']==c['prior_operational_failure']==compat['prior_operational_failure'] and request['prior_operational_failure']['sha256']=='ab9b56d99be2ef48e3bfa36e2fa79718bdf9404488ebb7834dcf2ea1ffe1e852')
check('pool_helper2_exact',compat['operational_sources']['worker_control_pool']['sha256']=='51d858cdc4d4f5b291648e2495a98a38b47654ad9888d2e839b13cc4da767e40')
for name,pin in compat['operational_lifecycle_callback_sources'].items():assert rec(inspect.getsourcefile(getattr(controller.FullOwner,name)))==pin
check('all_actual_lifecycle_callbacks_controller6',True)
check('same_config_recipe_totals',c['vocabulary']==9223 and c['planned_native_updates']==401072 and c['owner_validation_only'] is False and c['diagnostic_only'] is False)
io=g.process.io_counters();report={'kind':'ASR18_independent_controller6_genuine_resource_request_preflight_v1','owner':'/root/label_bpe_audit','passed':True,
    'producer':g.main_identity,'source':rec(__file__),'configuration':config_pin,'resource_resume_request':request_pin,
    'genuine_ROOT_compatibility':c['operational_compatibility'],'actual_controller_source':rec(controller.__file__),
    'check_count':len(checks),'checks':checks,'elapsed_seconds':time.monotonic()-began,'peak_private_bytes':g.peak,
    'actual_process_IO_bytes':io.read_bytes+io.write_bytes-g.initial_IO,'hard_leaf_limit_bytes':512_000_000,'maximum_seconds':180,
    'fixture_request_descriptor_only':True,'mocked_compatibility_request_or_pool':False,'execution_authority_claimed':False,
    'GPU_model_audio_tensor_payload_or_pool_operation':False,'actual_training_namespace_mutated':False,
    'actual_checkpoint_restoration_proven':False,'authoritative_draws_consumed':0,
    'scope':'Genuine immutable metadata and actual checked_resource_resume_request API only'}
out=AREA/'genuine_controller6_preflight_report_v1.json';assert not out.exists();out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
g.publish(phase='genuine_controller6_preflight_COMPLETE',check_count=len(checks),tensor_payload_bytes_read=0);timer.cancel()
print(json.dumps({'passed':True,'report':rec(out),'check_count':len(checks),'elapsed_seconds':report['elapsed_seconds'],'peak_private_bytes':g.peak}))
