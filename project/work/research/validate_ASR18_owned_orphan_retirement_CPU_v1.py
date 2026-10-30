"""Controlled CPU launcher-exited/owned-leaf-live failure-path test."""
import argparse, json, os, subprocess, sys, time
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import require,record
from fastmss_owned_process_closure_v1 import identity
from asr18_owned_family_observation_v2 import owned_family_sample,terminate_known_family,wait_captured_dead,same_process
from resource_watchdog_v2 import sample_memory,ResourceGuard
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')
PYTHON=Path(r'E:\gpu_training_scratchpad\scratch\BPE_broad_CPU_preflight_env_v2\Scripts\python.exe')
GUARD=P/'work/research/ASR18_bounded_CPU_phase_v3.py'


def leaf(folder):
    folder.mkdir(exist_ok=False);write_json(folder/'identity.json',identity())
    deadline=time.monotonic()+30
    while not (folder/'release.json').exists():
        require(time.monotonic()<deadline,'Finite intentionally orphaned CPU marker');time.sleep(.05)


def outer(folder):
    folder.mkdir(exist_ok=False)
    with (folder/'leaf.log').open('xb') as stream:
        child=subprocess.Popen([str(PYTHON),'-B',str(GUARD),'--folder',str(folder/'leaf_guard'),'--seconds','40',
            '--script',str(Path(__file__).resolve()),'--','--mode','leaf','--folder',str(folder/'leaf')],
            stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        deadline=time.monotonic()+15
        while not (folder/'leaf/identity.json').exists():
            require(child.poll() is None and time.monotonic()<deadline,'Actual bounded CPU marker started');time.sleep(.05)
        write_json(folder/'ready.json',{'producer':identity(),'leaf':read_json(folder/'leaf/identity.json'),'leaf_launcher':identity(child.pid)})
        while not (folder/'exit_outer.json').exists():
            require(time.monotonic()<deadline,'Finite controlled outer exit');time.sleep(.05)
        # Intentionally exercise failure cleanup: the CPU marker remains live.
        # No claim of a clean/normal leaf exit or leaf footer is made.


def main(folder):
    require(folder.resolve().is_relative_to(A.resolve()) and not folder.exists(),'Fresh E-only controlled fixture');folder.mkdir()
    checks=[];known={};samples=[];watch=ResourceGuard()
    def guard():
        value=sample_memory(os.getpid(),'owned_orphan_CPU_fixture');require(value['paired_sampling_within_one_second'] and value['worker_private_memory']['available'],'Fresh paired fixture sample')
        stop,_=watch.observe(value['system_memory'],value['worker_private_memory']);require(stop is None,'Unchanged global guard');return value
    def check(name,value):require(value,name);checks.append(name)
    with (folder/'outer.log').open('xb') as stream:
        child=subprocess.Popen([str(PYTHON),'-B',str(GUARD),'--folder',str(folder/'outer_guard'),'--seconds','40',
            '--script',str(Path(__file__).resolve()),'--','--mode','outer','--folder',str(folder/'outer')],
            stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        launcher=identity(child.pid);deadline=time.monotonic()+20
        try:
            while not (folder/'outer/ready.json').exists():
                require(child.poll() is None and time.monotonic()<deadline,'Actual CPU outer alive')
                samples.append(owned_family_sample(child,launcher,known=known,guard=guard));time.sleep(.05)
            ready=read_json(folder/'outer/ready.json')
            samples.append(owned_family_sample(child,launcher,known=known,guard=guard))
            key=(ready['leaf']['pid'],ready['leaf']['create_time'])
            check('Actual guarded CPU leaf identity captured through owned Popen ancestry',key in known)
            check('Outer launcher included in family sample',any(item['outer_launcher_included'] for item in samples))
            require(max(item['private_bytes_sum'] for item in samples)<400_000_000,'Finite400MB whole fixture-family bound')
            write_json(folder/'outer/exit_outer.json',{'controlled_exit':True})
            check('Actual outer Popen0 observed while intentionally orphaning CPU leaf',child.wait(timeout=10)==0)
            check('Captured leaf is still live AFTER outer launcher exit',same_process(ready['leaf']))
            # New helper must act despite child.poll()==0, and wait every
            # captured still-live creation-time identity dead before returning.
            terminate_known_family(child,launcher,known)
            proof=wait_captured_dead(known,timeout=3)
            check('Failure cleanup returns only after all captured identities observed dead',proof['all_captured_owned_identities_observed_dead'])
            check('Controlled orphaned leaf is actually dead',not same_process(ready['leaf']))
            check('Second retirement verification is safe after actual death',wait_captured_dead(known,timeout=1)['all_captured_owned_identities_observed_dead'])
        finally:
            if child.poll() is None or any(same_process(value) for value in known.values()):terminate_known_family(child,launcher,known)
    outer_guard=read_json(folder/'outer_guard/closure.json')
    check('Outer guard actually closed0 clean joined',outer_guard['actual_exit_code']==0 and outer_guard['monitor_clean'] and outer_guard['monitor_joined'])
    check('No numeric/model/audio runtime imports',not any(name in sys.modules for name in ('torch','numpy','scipy','transformers','pyarrow')))
    write_json(folder/'report.json',{'kind':'ASR18_guarded_owned_CPU_orphan_retirement_fixture_v1','passed':True,
        'check_count':len(checks),'checks':checks,'source':record(__file__),'helper':record(S/'asr18_owned_family_observation_v2.py'),
        'known_owned_identities':list(known.values()),'actual_family_private_peak_bytes':max(item['private_bytes_sum'] for item in samples),
        'actual_outer_Popen_exit_code':0,'intentional_leaf_failure_cleanup_exercised':True,
        'normal_leaf_exit0_or_clean_leaf_footer_claimed':False,'captured_identity_retirement':proof,
        'actual_GPU_or_ASR_updates':0,'GPU_lease_changed':False,'real_DATA_or_CACHE_prepared':False})
    print(json.dumps({'passed':True,'checks':len(checks),'all_captured_identities_dead':True,'actual_GPU_updates':0}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('main','outer','leaf'),default='main');parser.add_argument('--folder',required=True)
    args=parser.parse_args();globals()[args.mode](Path(args.folder).resolve())
