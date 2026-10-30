"""Finite crash-tail recovery fixture; no Torch, models, GPU or source data."""
import argparse,hashlib,json,sys
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT/'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import write_json
from bpe_broad_production_state_v1 import CommitJournal,require
from bpe_broad_journal_branch_v1 import branch_checkpoint_prefix

def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);a=p.parse_args()
    folder=Path(a.folder);folder.mkdir(exist_ok=False)
    original=CommitJournal(folder/'original.jsonl');original.append({'update':1,'draws':14});saved=original.binding()
    original.append({'update':2,'draws':28});old=(folder/'original.jsonl').read_bytes()
    branch,receipt=branch_checkpoint_prefix(saved,folder/'resumed.jsonl')
    require(branch.binding()['offset']==saved['offset'] and branch.binding()['sha256']==saved['sha256'],'Prefix authority')
    branch.append({'update':2,'draws':28,'branch':'recomputed_from_checkpoint'})
    require((folder/'original.jsonl').read_bytes()==old and receipt['preserved_tail_bytes']>0,'Old crash tail preservation')
    require(branch.verify(branch.binding()) and original.verify(saved),'Both independent prefix hashes')
    rejects=[]
    for name,binding,dest in [('bad_hash',{**saved,'sha256':'0'*64},folder/'bad.jsonl'),
                              ('truncated',{**saved,'offset':len(old)+1},folder/'truncated.jsonl'),
                              ('overwrite',saved,folder/'resumed.jsonl'),('same_file',saved,folder/'original.jsonl')]:
        try:branch_checkpoint_prefix(binding,dest)
        except ValueError:rejects.append(name)
    require(len(rejects)==4,'Every unsafe recovery must fail closed')
    empty=CommitJournal(folder/'empty.jsonl');empty_branch,_=branch_checkpoint_prefix(empty.binding(),folder/'empty_resume.jsonl')
    require(empty_branch.binding()['offset']==0,'Empty committed prefix')
    require('torch' not in sys.modules,'No runtime model import')
    write_json(folder/'report.json',{'passed':True,'checks':11,'tail_receipt':receipt,'rejected':rejects,'old_journal_sha256':hashlib.sha256(old).hexdigest(),
        'optimizer_or_sampler_advanced':False,'fixture_only':True})

if __name__=='__main__':main()
