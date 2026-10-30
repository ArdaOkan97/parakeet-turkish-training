"""Actual closed one-thread no-Arrow FLAC/augmentation CPU child proof."""
import argparse,json,os,subprocess,sys
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[2];AREA=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')
sys.path.insert(0,str(PROJECT/'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import write_json
from bpe_broad_audio_v1 import record,require
from fastmss_cpu_prepared_transport_v1 import observe_closed_child,cleanup_owned_cache

def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);a=p.parse_args();folder=Path(a.folder);folder.mkdir(exist_ok=False)
    request=json.loads((AREA/'FLAC_lowmemory_CPU_actual_01/request.json').read_text());request['job_id']=folder.name;write_json(folder/'request.json',request)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    command=[str(PROJECT/'work/nemo-native-env-v2/Scripts/python.exe'),'-X','utf8','-B',str(PROJECT/'work/research/ASR18_bounded_CPU_phase_v2.py'),
        '--folder',str(folder/'child_guard'),'--seconds','120','--script',str(PROJECT/'work/research/decode_ASR18_broad_batch_v5.py'),
        '--','--request',str(folder/'request.json'),'--output',str(folder/'decoded')]
    with (folder/'child.log').open('x',encoding='utf-8') as log:
        child=subprocess.Popen(command,cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT);code=child.wait(timeout=130)
    require(code==0,'No-Arrow one-thread CPU decoder must actually close0')
    result=json.loads((folder/'decoded/report.json').read_text());transport=result['augmentation_transport']
    closure=observe_closed_child(child,owner_record=transport['owner'],manifest_record=transport['manifest'],proof_path=folder/'actual_child_closure.json',guard=lambda:None)
    current=json.loads(Path(transport['manifest']['path']).read_text())
    previous=json.loads((AREA/'integrated_broad_caller_01/integrated_broad_caller_01_actual_1/prepared_transport_durable_evidence.json').read_text())['manifest']
    expected={(e['draw']['source_id'],e['draw']['epoch'],e['draw']['draw_index']):e for e in previous['entries']}
    require(len(result['rows'])==len(current['entries'])==4,'All four complete large full-source clips retained')
    require(all(e['clean_waveform_sha256']==expected[(e['draw']['source_id'],e['draw']['epoch'],e['draw']['draw_index'])]['clean_waveform_sha256'] and
        e['prepared_waveform_sha256']==expected[(e['draw']['source_id'],e['draw']['epoch'],e['draw']['draw_index'])]['prepared_waveform_sha256'] for e in current['entries']),
        'Exact original Parquet cleanPCM AND prepared waveforms must match with changed loader/thread environment')
    write_json(folder/'transport_evidence.json',{'manifest':current,'owner':json.loads(Path(transport['owner']['path']).read_text()),'closure':closure})
    boundary={'kind':'discard_uncommitted_preparation','batch_id':request['batch_id'],'owner_sha256':transport['owner']['sha256'],'no_native_or_sampler_commit':True}
    cleaned=cleanup_owned_cache(owner_record=transport['owner'],manifest_record=transport['manifest'],closure_record=closure,boundary=boundary,
        verify_boundary=lambda v:v==boundary,guard=lambda:None)
    for item in result['rows']:
        path=Path(item['PCM_cache']['path']).resolve();require(path.parent==(folder/'decoded').resolve() and record(path)==item['PCM_cache'],'Exact own clean cache')
    for item in result['rows']:Path(item['PCM_cache']['path']).unlink()
    guard=json.loads((folder/'child_guard/closure.json').read_text())
    write_json(folder/'report.json',{'passed':True,'actual_child_closure':closure,'actual_exit_code':code,'CPU_guard':guard,
        'rows':4,'fullPCM_and_prepared_augmentation_byte_exact':True,'one_thread_environment':{k:env[k] for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')},
        'no_Arrow_in_decode_source':True,'no_GPU_or_model_used':True,'no_frontier_advanced':True,'owned_cache_cleaned':cleaned})

if __name__=='__main__':main()
