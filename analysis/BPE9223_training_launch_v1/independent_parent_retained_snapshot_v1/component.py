"""Actual startup metadata roots only; no live-parent or absolute64M fit claim."""
import copy,gc,hashlib,json,os,sys,time,threading
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(r'E:\gpu_training_scratchpad')
PROJECT=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20')
AREA=ROOT/'analysis/BPE9223_training_launch_v1/independent_parent_retained_snapshot_v1'
sys.path[:0]=[str(PROJECT/'outputs/parakeet_turkish'),str(PROJECT/'work/research')]
from run_ASR18_frozen_CNN_lifetime_pair_v8 import storage,Watch,pid_identity
from run_ASR18_BPE9223_full_run_owner_v2 import fresh_admit,private_lifetime_limit
from atomic_io_retry_v1 import write_json
from bpe_broad_production_state_v1 import digest
from bpe_final_DATA_schedule_context_v2 import normalize_schedule_state
from bpe_streaming_checkpoint_semantics_v2 import decode_scalar_tree,select
FAILURE=ROOT/'analysis/BPE9223_training_launch_v1/full_training_operational_resource_actual_08/actual_committed60_next_CPU_memory_failure_TOOL1_v1.json'
FIELDS=('current_capture','current_completion','current_response','current_release','current_recovery_report','current_ack',
 'current_coverage','current_packet','current_CPU0','current_CPU_report','current_bridge','current_cleanup0','current_cleanup',
 'current_workspace','current_original_plan','current_intent','closed_native','closed_recovery','current_cadence',
 'native_sequence','recovery_sequence','recovery_control_sequence','proof_sequence','last_window_index')
def record(path):
    path=Path(path).resolve();s=path.stat();h=hashlib.sha256()
    with path.open('rb') as stream:
        while block:=stream.read(1024**2):h.update(block)
    assert (s.st_size,s.st_mtime_ns)==(path.stat().st_size,path.stat().st_mtime_ns)
    return {'path':str(path),'bytes':s.st_size,'sha256':h.hexdigest()}
def read(pin):
    assert pin['bytes']<=16*1024**2 and record(pin['path'])==pin
    return json.loads(Path(pin['path']).read_text(encoding='utf-8-sig'))
def graph_size(root):
    seen=set();stack=[root];total=0
    while stack:
        value=stack.pop();key=id(value)
        if key in seen:continue
        seen.add(key);assert len(seen)<2_000_000;total+=sys.getsizeof(value)
        if type(value) is dict:stack.extend(value.keys());stack.extend(value.values())
        elif type(value) in (list,tuple,set,frozenset):stack.extend(value)
    return {'recursive_unique_python_bytes':total,'unique_objects':len(seen)}
def main():
    import ctypes,psutil
    from ctypes import wintypes
    storage();assert not (AREA/'report.json').exists()
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetCurrentProcess.restype=wintypes.HANDLE
    kernel.SetPriorityClass.argtypes=[wintypes.HANDLE,wintypes.DWORD]
    assert kernel.SetPriorityClass(kernel.GetCurrentProcess(),0x4000)
    for _ in range(3):assert fresh_admit(2_000_000_000)['passed'];time.sleep(.2)
    out=AREA/'actual';out.mkdir(exist_ok=False)
    write_json(out/'boot.json',{'producer':pid_identity(),'source':record(__file__)})
    watch=Watch(out,512_000_000,worker=True);watch.phase='actual_startup_metadata_roots'
    process=psutil.Process();initial=process.io_counters();began=time.monotonic();code=1
    timer=threading.Timer(180,lambda:write_json(out/'stop.json',{'reason':'reviewed180smetadata'}));timer.daemon=True;timer.start()
    def guard():
        watch.check();private_lifetime_limit(512_000_000);io=process.io_counters()
        assert io.read_bytes+io.write_bytes-initial.read_bytes-initial.write_bytes<=16*1024**3
    def measure(name):
        guard();value=private_lifetime_limit(512_000_000);value['phase']=name;return value
    try:
        assert record(FAILURE)['sha256']=='dca44b6c43a754677611d11f801e27082b988d43f39065a1b69d3b6826da2f96'
        failed=json.loads(FAILURE.read_text(encoding='utf-8-sig'))
        assert failed['preserved_production_updates']==60 and failed['next_window_CPU_children_started']==0
        config=read(failed['configuration']);request_pin=failed['resume_request']
        # Independent parses reproduce the two separately checked startup trees,
        # rather than sharing immutable integer/string objects via one deepcopy.
        first=read(request_pin)['owner_state'];second=read(request_pin)['owner_state']
        first['sampler_state']=normalize_schedule_state(first['sampler_state'])
        second['sampler_state']=normalize_schedule_state(second['sampler_state'])
        owner=SimpleNamespace(config=config,resume_request=request_pin,resume_snapshot=first)
        for name in FIELDS:setattr(owner,name,copy.deepcopy(second[name]))
        owner.reserved_preparation=copy.deepcopy(second.get('reserved_preparation'))
        owner.prepared_CPU_resume=copy.deepcopy(second.get('prepared_CPU'))
        pointer=read(failed['checkpoint_pointer']);metadata=read(failed['checkpoint_metadata'])
        owner.live_schedule_state=normalize_schedule_state(decode_scalar_tree(select(metadata['tree'],'sampler')))
        assert digest(owner.live_schedule_state)==pointer['sampler_state_sha256']
        owner.live_capture=copy.deepcopy(failed['current_checkpoint_capture'])
        owner.counters={k:failed[k] for k in ('actual_owner_invocation_count','native_sequence','recovery_sequence','recovery_control_sequence','proof_sequence')}
        owner.reserved_p4=copy.deepcopy(failed['reserved_unconsumed_preparation'])
        compatibility=read(config['operational_compatibility'])
        def live_identity():
            return digest({'sampler':owner.live_schedule_state,'capture':owner.live_capture,'counters':owner.counters,'p4':owner.reserved_p4})
        identity=live_identity()
        pins={k:failed[k] for k in ('checkpoint_pointer','checkpoint_metadata','storage_state','durable_latest',
            'current_registered_journal','pool_registry','pool_index','pool_archive_prefix','storage_operation_witnesses')}
        assert all(record(pin['path'])==pin for pin in pins.values())
        del first,second,metadata,pointer,config;gc.collect()
        # Recursive accounting is separate from OS current commit. It is neither
        # disjoint memory savings nor an estimate of a production parent's peak.
        sizes={'snapshot':graph_size(owner.resume_snapshot),'compatibility':graph_size(compatibility)}
        gc.collect();stages=[measure('held_startup_snapshot_and_compatibility')]
        owner.resume_snapshot={'startup_checks_completed':True,'provenance':request_pin}
        collected1=gc.collect();stages.append(measure('unused_startup_snapshot_released'))
        del compatibility
        collected2=gc.collect();stages.append(measure('unused_CPU_sourcecheck_compatibility_released'))
        assert live_identity()==identity and all(record(pin['path'])==pin for pin in pins.values())
        assert record(request_pin['path'])==request_pin
        assert not any(name in sys.modules for name in ('torch','numpy','scipy','soundfile'))
        guard()
        report={'kind':'ASR18_actual_retained_startup_metadata_root_component_v1','passed':True,
            'source':record(__file__),'producer':pid_identity(),'actual_failure':record(FAILURE),
            'actual_original_resume_request':request_pin,'actual_preserved_optimizer_updates':60,
            'separate_two_parse_startup_root_reconstruction':True,'FullOwner_constructor_or_advance_executed':False,
            'root_graph_sizes_not_disjoint_OS_savings':sizes,'OS_current_measurements':stages,
            'snapshot_collected_objects':collected1,'compatibility_collected_objects':collected2,
            'snapshot_OS_private_delta_bytes':stages[1]['current_private_bytes']-stages[0]['current_private_bytes'],
            'compatibility_OS_private_delta_bytes':stages[2]['current_private_bytes']-stages[1]['current_private_bytes'],
            'current60_sampler_capture_counters_and_p4_unchanged':True,'all_authoritative_pins_unchanged':True,
            'audio_GPU_tensor_restore_optimizer_draws_or_authoritative_mutations':False,
            'production_parent_heap_or_absolute64M_fit_reproduced_or_proved':False,
            'resource_caps_or_source_training_state_changed':False,'wall_seconds':time.monotonic()-began}
        write_json(AREA/'report.json',report);code=0
        print(json.dumps({'passed':True,'report':record(AREA/'report.json'),'OS_current':stages}),flush=True)
    finally:
        timer.cancel();watch.close(code)
if __name__=='__main__':main()
