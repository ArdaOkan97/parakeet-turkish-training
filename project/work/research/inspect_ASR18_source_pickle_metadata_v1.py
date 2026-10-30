"""Read only the pinned Torch ZIP scalar opcode metadata, no tensor runtime."""
import argparse,json,pickletools,sys,zipfile
from pathlib import Path
P=Path(__file__).resolve().parents[2];sys.path.insert(0,str(P/'outputs/parakeet_turkish'))
from bpe_broad_audio_v1 import record,require
from atomic_io_retry_v1 import write_json
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')
def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);a=p.parse_args();f=Path(a.folder);require(f.is_relative_to(A),'E-only evidence');f.mkdir(exist_ok=False)
    projection=json.loads((A/'source_projection_v4/report.json').read_text(encoding='utf-8'));source=projection['projections']['bpe']['projected']
    require(record(source['path'])==source,'Pinned original compact checkpoint required')
    with zipfile.ZipFile(source['path']) as archive:
        names=[n for n in archive.namelist() if n.endswith('/data.pkl')];require(len(names)==1,'Single metadata pickle expected')
        info=archive.getinfo(names[0]);require(info.file_size<=8*1024**2,'Bounded scalar metadata required')
        raw=archive.read(names[0]);ops=list(pickletools.genops(raw))
    globals_=sorted(set(str(arg) for op,arg,pos in ops if op.name=='GLOBAL'))
    require(not any(k in sys.modules for k in ('torch','numpy','scipy','transformers','pyarrow')),'No tensor/numeric runtime')
    write_json(f/'report.json',{'passed':True,'source_checkpoint':source,'metadata_bytes':len(raw),'globals':globals_,
        'opcode_names':sorted(set(op.name for op,arg,pos in ops)),'source':record(__file__),'no_unpickling_executed':True,'no_tensor_payload_read':True})
    print(json.dumps({'passed':True,'metadata_bytes':len(raw),'globals':globals_}))
if __name__=='__main__':main()
