"""Export every cached official test row verbatim to designated E storage."""
import argparse, hashlib, json, os, shutil, sys, time
from pathlib import Path
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20')
sys.path.insert(0,str(P/'outputs/parakeet_turkish'))
from paired_windows_commit_telemetry_v1 import paired_sample
from resource_watchdog_v2 import sample_memory
R=Path(__file__).parent
def save(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
def main(language):
    import psutil
    p=psutil.Process();p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    folder=R/'full_test_data'/language;folder.mkdir(parents=True,exist_ok=False)
    began=time.monotonic();samples=[paired_sample(),paired_sample()];save(folder/'admission.json',samples)
    assert all(s['global_memory_status_ex']['ullAvailPageFile_bytes']>=8066304384 and s['get_performance_info_after']['physical_available_bytes']>=5316304384 for s in samples)
    def guard():
        assert time.monotonic()-began<160
        s=sample_memory();assert s['worker_private_memory']['private_bytes']<=2573741824
        assert s['system_memory']['commit_available_bytes']>=5547868928 and s['system_memory']['physical_available_bytes']>=5047868928
        assert shutil.disk_usage(folder).free>=100000000000
    item=json.loads((R/'input_inventory.json').read_text())['datasets'][language]
    save(folder/'scope.json',{'parquet':item,'all_test_rows':True,'payload_export':'embedded audio bytes verbatim','maximum_private':2573741824,'maximum_wall':160})
    import pyarrow.parquet as pq
    from metrics import normalize_text
    pf=pq.ParquetFile(item['path'],memory_map=True,pre_buffer=False,buffer_size=0)
    rows=[];seen=set();peak=0
    with (folder/'test.jsonl').open('x',encoding='utf-8') as manifest:
        for index,batch in enumerate(pf.iter_batches(batch_size=1,use_threads=False)):
            guard();row=batch.to_pylist()[0];blob=row['audio']['bytes'];assert blob
            source=f"google/fleurs:{language}:test:{row['id']}:{row['path']}";assert source not in seen;seen.add(source)
            reference=row['raw_transcription'];assert normalize_text(reference,language)
            h=hashlib.sha256(blob).hexdigest();path=folder/(str(index).zfill(5)+'_'+h[:16]+'.wav')
            with path.open('xb') as f:f.write(blob)
            record={'source_id':source,'sentence_id':row['id'],'language':language,'split':'test','reference':reference,'audio_filepath':str(path),'audio_sha256':h,'num_samples':row['num_samples'],'row_index':index}
            manifest.write(json.dumps(record,ensure_ascii=False)+'\n');rows.append(record)
            peak=max(peak,p.memory_info().private)
            del row,blob,batch
        manifest.flush();os.fsync(manifest.fileno())
    guard();assert len(rows)==item['rows'] and len(seen)==item['rows']
    result={'language':language,'utterances':len(rows),'seconds':time.monotonic()-began,'private_peak_observed':peak,'manifest_sha256':hashlib.sha256((folder/'test.jsonl').read_bytes()).hexdigest(),'no_filtering_or_audio_transforms':True}
    save(folder/'complete.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('language');main(parser.parse_args().language)
