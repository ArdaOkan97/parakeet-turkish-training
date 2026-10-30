"""Finite finalized-source mutation fixture on E only; no audio/model imports."""
import argparse,hashlib,json,os,sys
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT/'outputs/parakeet_turkish'))
from bpe_broad_audio_v1 import record,require
from bpe_broad_source_identity_v1 import FinalSourceIdentity
from atomic_io_retry_v1 import write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);a=p.parse_args();folder=Path(a.folder)
    folder.mkdir(exist_ok=False);plan=folder/'plan.json';write_json(plan,{'fixture':True})
    def bound(path):return {**record(path),'mtime_ns':path.stat().st_mtime_ns}
    paths=[folder/'datasets/kiraat/test.parquet',folder/'datasets/yodas3/audio/0000.tar'];jobs={};items=[]
    for path in paths:
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'original_fixture_bytes')
        receipt=path.with_suffix(path.suffix+'.receipt.json');job={**record(path),'size':path.stat().st_size,'revision':'fixture_pin'}
        write_json(receipt,{'sha256':job['sha256'],'revision':job['revision']});jobs[str(path.resolve())]=job
        index=None
        if path.suffix=='.tar':
            target=folder/'indexes/yodas3_tar_0000.sqlite';target.parent.mkdir();target.write_bytes(b'fixture_index');index=bound(target)
        items.append({**bound(path),'receipt':bound(receipt),'index':index})
    inventory=folder/'inventory.json';write_json(inventory,{'kind':'ASR18_final_source_identity_inventory_v1','complete':True,'plan':record(plan),'sources':items})
    verifier=FinalSourceIdentity(record(inventory),record(plan),jobs)
    for path in paths:verifier.verify(path)
    rejected=[]
    for label,target in [('same_size_source',paths[0]),('receipt',paths[0].with_suffix('.parquet.receipt.json')),('archive_index',folder/'indexes/yodas3_tar_0000.sqlite')]:
        original=target.read_bytes();before=target.stat();target.write_bytes(bytes([original[0]^1])+original[1:]);os.utime(target,ns=(before.st_atime_ns,before.st_mtime_ns+1_000_000_000))
        try:verifier.verify(paths[1] if label=='archive_index' else paths[0])
        except ValueError:rejected.append(label)
        target.write_bytes(original);os.utime(target,ns=(before.st_atime_ns,before.st_mtime_ns))
    require(len(rejected)==3,'Mutated sources/receipts/index must fail despite unchanged source size')
    verifier.verify(paths[0]);verifier.verify(paths[1])
    write_json(folder/'report.json',{'passed':True,'checks':7,'rejected':rejected,'source_identity_inventory':record(inventory),
        'real_source_files_modified':False,'fixture_only':True,'audio_or_model_imported':False})
    require('torch' not in sys.modules and 'numpy' not in sys.modules,'Metadata-only identity checks')

if __name__=='__main__':main()
