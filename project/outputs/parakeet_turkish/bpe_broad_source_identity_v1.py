"""Fail-closed finalized compressed-source signatures, no audio/model imports."""
import json
from pathlib import Path
from bpe_broad_audio_v1 import scoped,record,require

class FinalSourceIdentity:
    def __init__(self,inventory_binding,plan_binding,jobs):
        require(record(inventory_binding['path'])==inventory_binding,'Final source inventory binding changed')
        inventory=json.loads(Path(inventory_binding['path']).read_text(encoding='utf-8-sig'))
        require(inventory['kind']=='ASR18_final_source_identity_inventory_v1' and inventory['complete'] is True,'Actual finalized source identity inventory required')
        require(inventory['plan']==plan_binding,'Final inventory differs from exact pinned preparation plan')
        self.inventory_binding=inventory_binding;self.inventory=inventory;self.jobs=jobs;self.entries={};self.checked={}
        for item in inventory['sources']:
            require(set(item)=={'path','sha256','bytes','mtime_ns','receipt','index'},'Exact finalized source signature schema required')
            key=str(scoped(item['path']));require(key not in self.entries,'Duplicate finalized source')
            self.entries[key]=item
        require(set(self.entries)==set(jobs),'Every pinned new/reused source must occur exactly once')
        for key,item in self.entries.items():
            job=jobs[key]
            require(item['sha256']==job['sha256'] and item['bytes']==job['size'] and type(item['mtime_ns']) is int,'Source identity differs from pinned plan')

    def _bound_metadata(self,item):
        path=scoped(item['path']);stat=path.stat();signature=(stat.st_size,stat.st_mtime_ns)
        require(signature==(item['bytes'],item['mtime_ns']),'Finalized index signature changed')
        key=str(path)
        if key not in self.checked:
            require(record(path)=={k:item[k] for k in ('path','sha256','bytes')},'Finalized metadata hash differs')
            self.checked[key]=signature
        require(self.checked[key]==signature,'Previously verified metadata changed')

    def verify(self,path):
        path=scoped(path);key=str(path);item=self.entries[key];job=self.jobs[key]
        stat=path.stat();require((stat.st_size,stat.st_mtime_ns)==(item['bytes'],item['mtime_ns']),'Finalized compressed source changed; unchanged receipt cannot re-admit it')
        receipt=item['receipt'];self._bound_metadata(receipt)
        metadata=json.loads(Path(receipt['path']).read_text(encoding='utf-8-sig'))
        require(metadata['sha256']==job['sha256'] and metadata['revision']==job['revision'],'Final receipt no longer identifies the planned source')
        expected_receipt=path.with_suffix(path.suffix+'.receipt.json').resolve()
        require(Path(receipt['path']).resolve()==expected_receipt,'Receipt is not the actual source receipt')
        if path.suffix=='.tar':
            expected=path.parents[3]/'indexes'/('yodas3_tar_'+path.stem+'.sqlite')
            require(item['index'] is not None and Path(item['index']['path']).resolve()==expected.resolve(),'Exact finalized archive index binding required')
            self._bound_metadata(item['index'])
        else:require(item['index'] is None,'Parquet source must not bind an unrelated archive index')
        return path,job
