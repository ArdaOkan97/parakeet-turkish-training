"""ZIP directory only: report declared pickle size, without reading its bytes."""
import argparse,json,sys,zipfile
from pathlib import Path
P=Path(__file__).resolve().parents[2];sys.path.insert(0,str(P/'outputs/parakeet_turkish'))
from bpe_broad_audio_v1 import record,require
from atomic_io_retry_v1 import write_json
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')
def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);a=p.parse_args();f=Path(a.folder);require(f.is_relative_to(A),'E-only evidence');f.mkdir(exist_ok=False)
    projection=json.loads((A/'source_projection_v4/report.json').read_text(encoding='utf-8'));source=projection['projections']['bpe']['projected'];path=Path(source['path'])
    before=(path.stat().st_size,path.stat().st_mtime_ns);require(before[0]==source['bytes'],'Declared original compact checkpoint size')
    with zipfile.ZipFile(path) as archive:
        info=[i for i in archive.infolist() if i.filename.endswith('/data.pkl')];require(len(info)==1,'Single metadata pickle expected')
        item=info[0];result={'name':item.filename,'declared_uncompressed_bytes':item.file_size,'compressed_bytes':item.compress_size,'compression':item.compress_type}
    require(before==(path.stat().st_size,path.stat().st_mtime_ns),'Source changed during directory read')
    write_json(f/'report.json',{'passed':True,'source_projection_binding':source,'current_signature':list(before),'pickle_directory':result,
        'source':record(__file__),'current_full_checkpoint_SHA_reverified':False,'pickle_or_tensor_payload_read':False})
    print(json.dumps(result))
if __name__=='__main__':main()
