"""Intentional watchdog fault: observe OWN child's real75 and no payload."""
import argparse,json,os,subprocess,sys
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT/'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import write_json
from bpe_broad_audio_v1 import record,require

def main():
    import psutil
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);a=p.parse_args();folder=Path(a.folder);folder.mkdir(exist_ok=False)
    payload=folder/'MUST_NOT_START';wrapper=PROJECT/'work/research/ASR18_bounded_CPU_phase_v2.py'
    child=subprocess.Popen([sys.executable,'-X','utf8','-B',str(wrapper),'--folder',str(folder/'fault_guard'),
        '--seconds','10','--fixture-monitor-error','--script',str(PROJECT/'work/research/validate_ASR18_transaction_v1.py'),'--','--folder',str(payload)],cwd=PROJECT)
    code=child.wait(timeout=10);boot=json.loads((folder/'fault_guard/boot.json').read_text());stop=json.loads((folder/'fault_guard/resource_stop.json').read_text())
    require(code==75 and child.poll()==75 and not psutil.pid_exists(boot['pid']),'Actual owned monitor-error child must be closed75')
    require(stop['reason']=='CPU_watchdog_exception' and stop['observed']['monitor_error'] and not payload.exists()
        and not (folder/'fault_guard/closure.json').exists(),'No payload or successful closure after failed watchdog')
    write_json(folder/'report.json',{'passed':True,'wrapper':record(wrapper),'actual_owned_handle_PID':child.pid,'actual_worker_PID':boot['pid'],
        'actual_exit_code':code,'parent_observed_wait':True,'payload_started':False,'successful_closure_forged':False,'fault_injection_only':True,'resource_stop':record(folder/'fault_guard/resource_stop.json')})

if __name__=='__main__':main()
