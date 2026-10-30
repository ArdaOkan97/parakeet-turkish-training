"""Stream scalar metadata opcodes of the measured66.54MB original pickle."""
import argparse,json,pickletools,sys,zipfile
from pathlib import Path
P=Path(__file__).resolve().parents[2];sys.path.insert(0,str(P/'outputs/parakeet_turkish'))
from bpe_broad_audio_v1 import record,require
from atomic_io_retry_v1 import write_json
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')
def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);a=p.parse_args();f=Path(a.folder);require(f.is_relative_to(A),'E-only evidence');f.mkdir(exist_ok=False)
    projection=json.loads((A/'source_projection_v4/report.json').read_text(encoding='utf-8'));source=projection['projections']['bpe']['projected']
    require(record(source['path'])==source,'Exact pinned original compact checkpoint required')
    names=set();globals_=set();count=0
    with zipfile.ZipFile(source['path']) as archive:
        entries=[i for i in archive.infolist() if i.filename.endswith('/data.pkl')];require(len(entries)==1 and entries[0].file_size==66_539_150,'Measured original metadata size required')
        with archive.open(entries[0]) as stream:
            for op,arg,pos in pickletools.genops(stream):
                names.add(op.name);count+=1
                if op.name=='GLOBAL':globals_.add(str(arg))
    require(not any(k in sys.modules for k in ('torch','numpy','scipy','transformers','pyarrow')),'No tensor/numeric runtime')
    write_json(f/'report.json',{'passed':True,'source_checkpoint':source,'metadata_bytes':entries[0].file_size,'globals':sorted(globals_),
        'opcode_names':sorted(names),'opcode_count':count,'source':record(__file__),'no_unpickling_executed':True,'no_tensor_storage_payload_read':True,
        'metadata_streamed_without_retained_opcode_list':True})
    print(json.dumps({'passed':True,'opcode_count':count,'globals':sorted(globals_)}))
if __name__=='__main__':main()
