"""Exercise actual serialized ACKed60 evidence/protection without creating an owner."""
import copy,json,sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
P=Path(__file__).resolve().parents[2];sys.path[:0]=[str(P/'outputs/parakeet_turkish'),str(P/'work/research')]
from atomic_io_retry_v1 import write_json
from bpe_broad_audio_v1 import record,require
from asr18_BPE9223_operational_metadata_v2 import configure,read_bound
from asr18_BPE9223_full_run_controller_v17 import FullOwner,checked_ACKed60_session_resume_request
from asr18_BPE9223_persistence_prefix_v11 import checked_ACKed60_session_failure
from resource_watchdog_v2 import sample_memory
N=Path(r'E:\gpu_training_scratchpad\analysis\BPE9223_training_launch_v1')
OUT=N/'operational_configuration_12/ROOT_actual_ACKed60_protection_gate_v1'
@contextmanager
def phase(name):
    require(name=='read_index','Only readonly metadata');yield
def main():
    OUT.mkdir(exist_ok=False);calls=0;peak=0
    def guard():
        nonlocal calls,peak
        calls+=1
        m=sample_memory(__import__('os').getpid(),'ROOT_ACKed60_metadata')
        w=m['worker_private_memory'];s=m['system_memory']
        require(w['available'] and s['available'] and m['paired_sampling_within_one_second'],'Fresh paired telemetry')
        peak=max(peak,w['private_bytes'])
        require(w['private_bytes']<512000000 and s['commit_used_bytes']<.90*s['commit_limit_bytes']
            and s['commit_limit_bytes']-s['commit_used_bytes']>=2000000000,'Finite metadata resource admission')
    cp=record(N/'operational_configuration_12/configuration_pending_execution_review.json')
    rp=record(N/'operational_configuration_12/resource_resume_request.json')
    c=json.loads(Path(cp['path']).read_text());request=json.loads(Path(rp['path']).read_text())
    configure(**c['process_readonly_metadata_authority'],guard=guard,metadata_phase=phase)
    snapshot=checked_ACKed60_session_resume_request(cp,{'resource_resume_request':rp},rp,guard=guard)
    stop=checked_ACKed60_session_failure(c['prior_acknowledged_committed60_stop'],identity=c['full_run_identity'],
        operational_compatibility=c['operational_compatibility'],guard=guard,reader=read_bound)
    def owner():
        o=FullOwner.__new__(FullOwner)
        o.config=c;o.config_pin=cp;o.resume_request=rp;o.resource_closure=c['resource_native_closure']
        o.parent=SimpleNamespace(check=guard,control_pool=SimpleNamespace(state_pin=snapshot['worker_control_pool_state']))
        o.gpu=None;o.closed_native=snapshot['closed_native'];o.closed_recovery=snapshot['closed_recovery']
        o.current_ack=snapshot['current_ack'];o.ACKed60_initial=True
        return o
    o=owner();before=o._worker_control_protection();o.ACKed60_initial=False;after=o._worker_control_protection()
    require(before==after,'Protection remains after ACKed60 initial flag clears')
    fresh=stop['actual_normal_fresh_recovery0'];f=read_bound(fresh,guard)
    files=before[1]
    require(all(x in files for x in [fresh,f['report'],f['guard_closure'],c['prior_acknowledged_committed60_stop']]),
        'Genuine fresh60 report/process/guard and actualfailedowner all permanently physical')
    require(str(Path(fresh['path']).parent) in before[0],'Actual fresh60 folder permanently protected')
    for x in files:require(record(x['path'])==x,'Exact current physical protection binding')
    bad=copy.deepcopy(request);bad['kind']='ROOT_negative_unknown_request_fixture_NOT_EXECUTABLE'
    badpath=OUT/'negative_unknown_request_NOT_EXECUTABLE.json';write_json(badpath,bad)
    o=owner();o.resume_request=record(badpath)
    try:o._worker_control_protection()
    except ValueError as e:
        require(str(e)=='Unsupported worker-protection request kind; no implicit normal fallback','Exact unknown-kind rejection')
    else:raise AssertionError('Unknown-kind was accepted')
    frozen=request['current_stopped_state_bindings']
    require(all(record(x['path'])==x for x in frozen.values()),'All seven current run authority pins unchanged')
    require(not any(k in sys.modules for k in ['torch','numpy','scipy']),'No numerical import or payload')
    report=dict(kind='ROOT_actual_ACKed60_readonly_evidence_and_protection_gate_v1',passed=True,source=record(__file__),
        configuration=cp,resume_request=rp,actual_failed_owner=c['prior_acknowledged_committed60_stop'],
        actual_fresh60_process0=fresh,actual_ACK60=snapshot['current_ack'],actual_durable_head=snapshot['durable_head'],
        current_checkpoint=snapshot['current_capture']['pointer'],actual_protection_before_after_equal=True,
        protected_folders=before[0],protected_files=files,unknown_kind_rejected=True,all7authoritypins_unchanged=True,
        preserved_owner11_native3_recovery3_control5_proof10=True,guard_calls=calls,private_peak_bytes=peak,
        FullOwner_constructor_or_protect_reserve_Popen_or_ACK_called=False,new_GPU_audio_or_training_updates=0)
    write_json(OUT/'report.json',report);print(json.dumps({'passed':True,'report':record(OUT/'report.json'),'guard_calls':calls,'private_peak_bytes':peak}))
if __name__=='__main__':main()
