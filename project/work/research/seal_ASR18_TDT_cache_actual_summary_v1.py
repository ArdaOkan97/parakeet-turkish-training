"""Operator-observed closed component receipts; no numerical work or admission."""
from copy import deepcopy
from datetime import datetime,timezone
import hashlib,json,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];sys.path.insert(0,str(P/'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_production_state_v1 import require
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')
def pin(path):
    path=Path(path);return {'path':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
def main():
    cases={}
    for name,folder,session,chunk in [('seed',A/'official_TDT_cache_seed_actual_02',57958,'c178fa'),('hit',A/'official_TDT_cache_hit_actual_03',91055,'65b33d')]:
        proof=read_json(folder/'actual_process0.json');r=read_json(proof['report']['path']);g=read_json(proof['guard']['path'])
        require(proof['actual_Popen_wait_code']==0 and proof['all_captured_identities_dead'] is True and g['actual_exit_code']==0
            and g['monitor_clean'] and g['monitor_joined'] and r['saved_original_loss_gradient_byte_exact'] is True,'Actual closed exact numerical result')
        stages={v['phase']:v for v in r['phase_stages']};ready=stages['after_original_native_adapter_construction']['worker_private_memory']['private_bytes']
        clean=stages['call2_after_cleanup']['worker_private_memory']['private_bytes']
        cases[name]={'actual_owned_session':session,'actual_operator_observed_tool_exit':0,'actual_tool_chunk_id':chunk,
            'actual_process0':pin(folder/'actual_process0.json'),'report':proof['report'],'guard':proof['guard'],
            'native_PID':proof['native']['pid'],'adapter_ready_private_bytes':ready,'after_cleanup_private_bytes':clean,
            'adapter_ready_through_cleanup_retained_increase_bytes':clean-ready,'lifetime_private_peak_bytes':max(x['private_high_water']['lifetime_peak_private_commit_bytes'] for x in r['phase_stages']),
            'sampled_private_peak_bytes':g['peak_private_bytes'],'global_commit_peak_fraction':g['system_commit_peak_fraction'],
            'kernel_stats':{k:{f:v[f] for f in ('hits','misses','compile_fallback_allowed','every_loaded_kernel_has_current_CC_cubin')} for k,v in r['cache_after'].items()},
            'first_forward_seconds':r['calls'][0]['forward_seconds'],'second_forward_seconds':r['calls'][1]['forward_seconds'],
            'all_full_gradient_loss_input_RNG_gates_passed':True}
    out={'kind':'ASR18_actual_exact_six_kernel_cache_causal_summary_v1','cases':cases,
        'preserved_failed_hit2':{'actual_operator_observed_tool_exit':1,'actual_tool_chunk_id':'496069','log':pin(A/'official_TDT_cache_hit_actual_02/official_TDT/worker.log'),
            'cause':'Validation used hardware CC pair but installed codegen uses canonical triple; rejected before loss/compile. Corrected consumer derives actual exact _ensure_cc(None) key.'},
        'conclusion':'All six exact cache hits remove first-forward compilation latency but do not reduce retained host private memory. No native memory forecast credit.',
        'compiler_preload_omitted':False,'native_training_forecast_changed':False,'production_enabled':False,'source':pin(__file__)}
    path=A/'official_TDT_cache_hit_actual_03/actual_cache_causal_summary_v1.json';require(not path.exists(),'Immutable fresh receipt');write_json(path,out)
    now=datetime.now(timezone.utc).isoformat();stage='GPU_idle_cache_all6_hits_exact0_NO_host_saving_cuDNNv7_component_source_pending_review'
    pointer=read_json(A/'current_launch_readiness.json');old=read_json(pointer['current_readiness']['path']);new=deepcopy(old)
    new.update(kind='ASR18_launch_readiness_v65',updated_at_utc=now,current_stage=stage,actual_GPU_running=False,actual_owned_GPU_PID=None,
        previous_history=pointer['current_readiness'],actual_TDT_cache_summary=pin(path),inactive_cuDNN_v7_handoff=pin(A/'frozen_CNN_pair_configuration_10/handoff.json'))
    newpath=A/'launch_readiness_v65.json';require(not newpath.exists(),'Fresh status successor');write_json(newpath,new)
    pointer.update(updated_at_utc=now,current_readiness=pin(newpath),current_stage=stage,actual_owned_session=None,
        actual_TDT_cache_summary=pin(path),current_run_folder=str(A/'official_TDT_cache_hit_actual_03'))
    write_json(A/'current_launch_readiness.json',pointer)
    leasepath=P/'work/gpu_training_owner.json';lease=read_json(leasepath)
    require(lease['owner']=='/root/memory_diagnosis' and lease['actual_GPU_running'] is False and lease['actual_owned_worker_pid'] is None,'Actual idle exclusive owner')
    lease.update(status=stage,diagnostic_phase=stage,updated_at_utc=now,actual_TDT_cache_summary=pin(path));write_json(leasepath,lease)
    print(json.dumps({'summary':pin(path),'readiness':pin(newpath)}))
if __name__=='__main__':main()
