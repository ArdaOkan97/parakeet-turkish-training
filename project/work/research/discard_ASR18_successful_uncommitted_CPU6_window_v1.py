"""Exactly owned successful-but-unconsumed CPU6 proof retirement, review gated."""
import argparse,datetime,os,shutil,stat,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
E=Path(r'E:\gpu_training_scratchpad');A=E/'analysis/ASR18_memory_diagnosis_v1';RUN=A/'CPU6_first_window_only_actual_01'
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
from fastmss_row_serial_cpu_v6 import resource_guard as guard
import fastmss_phase_aware_feature_window_v4 as sink
from fastmss_owned_process_closure_v2 import verify as verify_process
from resource_watchdog_v2 import sample_memory

def main(review_path,folder):
    require(os.environ['CUDA_VISIBLE_DEVICES']=='-1' and read_json(E/'storage_config.json')['storage_root']==str(E),'E CPU-only storage environment')
    guard();sample=sample_memory(os.getpid(),'CPU6_owned_uncommitted_retirement');s=sample['system_memory']
    require(sample['paired_sampling_within_one_second'] and s['available'] and s['commit_used_bytes']+100000000<.9*s['commit_limit_bytes'] and s['commit_available_bytes']-100000000>2000000000 and shutil.disk_usage(E).free>100000000000,'Fresh100MB/system/E admission')
    lease=read_json(P/'work/gpu_training_owner.json');require(lease['owner']=='/root/memory_diagnosis' and lease['actual_GPU_running'] is False and lease['actual_owned_worker_pid'] is None,'Idle sole lease')
    report=read_json(RUN/'report.json');cpu_actual=read_json(report['CPU_actual_process0']['path'])
    require(report['passed'] is True and report['actual_GPU_or_Adam_or_authoritative_draws']==0 and report['prepared_features_retained_uncommitted'] is True,'Exact completed CPU-only proof, never consumed')
    verify_process(cpu_actual,kind='ASR18_owned_CPU_process_completion_v1',owner='/root/memory_diagnosis',guard=guard)
    require(record(cpu_actual['report']['path'])==cpu_actual['report'],'Exact completed controller report')
    import psutil
    for v in (report['producer'],cpu_actual['producer'],cpu_actual['observer'],cpu_actual['launcher']):
        try:live=abs(psutil.Process(v['pid']).create_time()-v['create_time'])<.001
        except psutil.NoSuchProcess:live=False
        require(not live,'Every exact CPU producer/observer/launcher dead')
    owner_pin=report['owner'];manifest_pin=report['manifest']
    require(owner_pin['sha256']=='9fda27f6004ef63e021cae5e95e10ac99351ef2f55c3bea5a96f85fbb5389905' and manifest_pin['sha256']=='82a840fa561c6df929ff5b4803af38d7c3c36c27b03567217d3133daaf2baad5','Exact retained proof owner/manifest only')
    owner,cache,plan=sink.owner_folder(owner_pin,guard)
    require(cache.resolve()==E/'temp/ASR18_CPU_feature_windows_v1/payload' and owner['updates']==20 and owner['prepared_is_committed'] is False and owner['sampler_journal_frontier_advanced'] is False,'Exact original uncommitted shared window')
    inventory_path=RUN/'metadata_inventory_for_owned_retirement_v1.json';inventory=read_json(inventory_path)
    review=read_json(review_path)
    require(review['accepted'] is True and review['reviewer']=='/root' and review['scope']=='discard_successful_uncommitted_CPU6_proof_window_v1' and review['source']==record(__file__)
        and review['owner']==owner_pin and review['manifest']==manifest_pin and review['inventory']==record(inventory_path),'Exact independently reviewed owned discard')
    actual=[]
    for path in sorted(cache.iterdir()):
        require(path.is_file() and path.resolve().parent==cache.resolve() and not getattr(path.lstat(),'st_file_attributes',0)&stat.FILE_ATTRIBUTE_REPARSE_POINT,'Only exact owned regular files/no reparse')
        guard();value=record(path);value['mtime_ns']=path.stat().st_mtime_ns;actual.append(value)
    require(actual==inventory['actual_files'],'Every exact current owned byte/file signature must match reviewed inventory')
    controller=read_json(cpu_actual['report']['path']);closures=controller['closures'];require(len(closures)==20,'All20 genuine frontend closures required')
    boundary={'kind':'discard_uncommitted_feature_window','owner_sha256':owner_pin['sha256'],'session':owner['session'],'phase':owner['phase'],
        'native_work_started':False,'committed_window_updates':0,'reason':'Successful CPU-only finalization/equality proof retained but unconsumed; source-bound new parent requires fresh preparation',
        'actual_CPU_only_report':record(RUN/'report.json'),'actual_CPU_process0':report['CPU_actual_process0'],'ROOT_review':record(review_path)}
    def verify_boundary(value):
        require(value==boundary and read_json(P/'work/gpu_training_owner.json')['actual_GPU_running'] is False,'Exact idle/unconsumed boundary only');return True
    folder=folder.resolve();require(folder.is_relative_to(A) and not folder.exists(),'Fresh E evidence namespace');folder.mkdir()
    result=sink.cleanup_window(owner_pin,closures=closures,manifest_binding=manifest_pin,boundary=boundary,verify_boundary=verify_boundary,evidence_path=folder/'durable_owned_metadata_archive.json',guard=guard)
    require(not cache.exists() and not list(sink.CACHE.iterdir()),'Shared slot empty only after owned archive-first cleanup')
    result.update(source=record(__file__),CPU_proof_preserved=record(RUN/'report.json'),original640_320_20_equality_proof_preserved=report['exact_equivalence'],
        actual_GPU_or_Adam_or_authoritative_draws=0,shared_feature_slot_empty=True,source_audio_checkpoint_journal_untouched=True,inventory=record(inventory_path))
    write_json(folder/'report.json',result);print({'passed':True,'removed_files':len(result['removed_exact_owned_files']),'report':record(folder/'report.json')},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--review',required=True,type=Path);p.add_argument('--folder',required=True,type=Path);a=p.parse_args();main(a.review,a.folder)
