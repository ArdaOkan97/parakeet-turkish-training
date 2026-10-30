"""Finite remaining four fixture source carriers, one owned CPU child at a time."""
import argparse,json,os,subprocess,sys
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[2];ROOT=Path(r'E:\gpu_training_scratchpad');AREA=ROOT/'analysis/ASR18_memory_diagnosis_v1'
sys.path.insert(0,str(PROJECT/'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import write_json
from bpe_broad_audio_v1 import BroadAudioReader,record,require

def main():
    import psutil
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);a=p.parse_args();folder=Path(a.folder);folder.mkdir(exist_ok=False)
    manifest=ROOT/'analysis/BPE_weak_then_clean_v1/production_reader_contract_v1/manifest.jsonl'
    rows=[json.loads(v) for v in manifest.read_text().splitlines()];sources=sorted({r['shard'] for r in rows if r['shard'].endswith('.parquet')})
    require(len(sources)==5,'Exact five Parquets in fixed14-source fixture')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    reader=BroadAudioReader();results=[]
    for index,path in enumerate(sources):
        source,job=reader.verify_source(path);target=ROOT/'cache/KIRAAT_original_FLAC_records_v1'/job['sha256']
        if (target/'validation.json').exists():results.append({'source':path,'existing_validation':record(target/'validation.json')});continue
        require(not target.exists(),'A partial carrier cannot be silently retried')
        for phase,script in [('prepare','prepare_ASR18_KIRAAT_FLAC_records_v2.py'),('audit','validate_ASR18_FLAC_carrier_v2.py')]:
            guard=folder/(str(index)+'_'+phase+'_guard');sourcefile=PROJECT/'work/research'/script;pin=record(sourcefile)
            command=[str(PROJECT/'work/nemo-native-env-v2/Scripts/python.exe'),'-X','utf8','-B',str(PROJECT/'work/research/ASR18_bounded_CPU_phase_v2.py'),
                '--folder',str(guard),'--seconds','180','--script',str(sourcefile),'--','--source',str(source)]
            with (folder/(str(index)+'_'+phase+'.log')).open('x',encoding='utf-8') as log:
                child=subprocess.Popen(command,cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT);code=child.wait(timeout=190)
            require(code==0 and record(sourcefile)==pin,'Finite carrier source changed or owned child failed')
            boot=json.loads((guard/'boot.json').read_text());closed=json.loads((guard/'closure.json').read_text())
            require(closed['actual_exit_code']==0 and not psutil.pid_exists(boot['pid']) and closed['monitor_clean'],'Actual owned native CPU child must be absent and closed0')
            results.append({'source':path,'phase':phase,'owned_handle_PID':child.pid,'actual_worker_PID':boot['pid'],'actual_exit_code':code,
                'parent_observed_wait':True,'source_file':pin,'guard_closure':record(guard/'closure.json'),'peak_private_bytes':closed['observed']['peak_private_bytes']})
            write_json(folder/'progress.json',{'results':results})
    require(all((ROOT/'cache/KIRAAT_original_FLAC_records_v1'/reader.jobs[str(Path(p).resolve())]['sha256']/'validation.json').exists() for p in sources),'All five full-source carriers require independent allbyte validation')
    write_json(folder/'report.json',{'passed':True,'fixture_manifest':record(manifest),'unique_KIRAAT_sources':5,'actual_owned_results':results,
        'full_corpus_carrier_rollout_claimed':False,'decoded_PCM_pool_created':False,'quality_or_language_filter_added':False})

if __name__=='__main__':main()
