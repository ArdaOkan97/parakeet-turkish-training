"""Fresh bounded CPU child: identical14 full waveforms/augments, mapped source."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[2];ROOT=Path(r'E:\gpu_training_scratchpad');AREA=ROOT/'analysis/ASR18_memory_diagnosis_v1'
sys.path.insert(0,str(PROJECT/'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import write_json
from bpe_broad_audio_v1 import require,record
from fastmss_cpu_prepared_transport_v1 import observe_closed_child,cleanup_owned_cache

def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);a=p.parse_args();folder=Path(a.folder);folder.mkdir(exist_ok=False)
    original=AREA/'integrated_broad_caller_01/integrated_broad_caller_01_actual_1'
    request=json.loads((original/'decode_request.json').read_text());request['job_id']=folder.name
    write_json(folder/'request.json',request)
    runtime=PROJECT/'work/nemo-native-env-v2/Scripts/python.exe';wrapper=PROJECT/'work/research/ASR18_bounded_CPU_phase_v3.py';decoder=PROJECT/'work/research/decode_ASR18_broad_batch_v6.py'
    pinned=[record(p) for p in (wrapper,decoder,PROJECT/'outputs/parakeet_turkish/bpe_broad_audio_FLAC_records_v1.py')]
    command=[str(runtime),'-X','utf8','-B',str(wrapper),'--folder',str(folder/'child_guard'),'--seconds','180','--script',str(decoder),'--','--request',str(folder/'request.json'),'--output',str(folder/'decoded')]
    with (folder/'child.log').open('x',encoding='utf-8') as log:
        child=subprocess.Popen(command,cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT)
        try:code=child.wait(timeout=190)
        except subprocess.TimeoutExpired:
            write_json(folder/'child_guard/stop.json',{'reason':'finite_mapped_loader_timeout'});child.wait(timeout=5);raise
    require(code==0,'Mapped original-source CPU decoder must close0')
    actual=json.loads((folder/'decoded/report.json').read_text());old=json.loads((original/'decoded/report.json').read_text())
    require(len(actual['rows'])==len(old['rows'])==14,'Every pinned row retained')
    require(all(a['row']['broad_original']==b['row']['broad_original'] and a['row']['tr_metadata_labels']==b['row']['tr_metadata_labels']
        and a['reader_evidence']['PCM_f32_sha256']==b['reader_evidence']['PCM_f32_sha256'] and a['reader_evidence']['PCM_i16_sha256']==b['reader_evidence']['PCM_i16_sha256']
        for a,b in zip(actual['rows'],old['rows'])),'Full PCM/rawfields/targetidentity changed')
    transport=actual['augmentation_transport'];closure=observe_closed_child(child,owner_record=transport['owner'],manifest_record=transport['manifest'],proof_path=folder/'actual_child_closure.json',guard=lambda:None)
    progress=json.loads((AREA/'integrated_broad_caller_01/native_progress.json').read_text())
    require(transport['semantic_sha256']==progress['history'][0]['prepared_transport']['semantic_sha256'],'Actual prepared transforms changed despite same draw policy/source')
    manifest=json.loads(Path(transport['manifest']['path']).read_text());owner=json.loads(Path(transport['owner']['path']).read_text())
    write_json(folder/'preserved_transport.json',{'manifest':manifest,'owner':owner,'closure':closure})
    boundary={'kind':'discard_uncommitted_preparation','batch_id':request['batch_id'],'owner_sha256':transport['owner']['sha256'],'no_native_or_sampler_commit':True}
    cleaned=cleanup_owned_cache(owner_record=transport['owner'],manifest_record=transport['manifest'],closure_record=closure,boundary=boundary,verify_boundary=lambda v:v==boundary,guard=lambda:None)
    for item in actual['rows']:
        path=Path(item['PCM_cache']['path']).resolve();require(path.parent==(folder/'decoded').resolve() and record(path)==item['PCM_cache'],'Exact owned clean cache boundary')
    for item in actual['rows']:Path(item['PCM_cache']['path']).unlink()
    guard=json.loads((folder/'child_guard/closure.json').read_text())
    write_json(folder/'report.json',{'passed':True,'actual_owned_child_exit_code':code,'CPU_guard':guard,'actual_child_closure':closure,
        'rows':14,'same_full_PCM_i16_f32_rawlabels_flags':True,'same_prepared_augmentation_semantic_sha256':transport['semantic_sha256'],
        'baseline_CPU_private_peak':780013568,'FLAC_CPU_private_peak':guard['observed']['peak_private_bytes'],'pinned_sources':pinned,
        'source_files_modified':False,'Torch_GPU_model_imported':False,'transient_preparation_caches_cleaned':cleaned,'no_frontier_advanced':True})
    require('torch' not in sys.modules and all(record(v['path'])==v for v in pinned),'No runtime model imports or changed sources')

if __name__=='__main__':main()
