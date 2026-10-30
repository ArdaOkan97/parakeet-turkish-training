"""Separate <=60GB additional pinned compressed archive staging; resumable."""
import json
import pathlib
import sqlite3
import tarfile
import time
import urllib.request

from cpu_preparation_guard_v2 import AREA,ROOT,YODAS_REV,Guard,append_json,now,scoped,sha256,write_json

CAP=60_000_000_000

def download(record, destination, guard):
    path=scoped(destination); path.parent.mkdir(parents=True,exist_ok=True)
    receipt=path.with_suffix(path.suffix+'.receipt.json'); expected=record.get('lfs',{}).get('oid')
    if path.exists():
        bound=json.loads(receipt.read_text(encoding='utf-8'))
        if path.stat().st_size!=record['size'] or bound['revision']!=YODAS_REV or (expected and bound['sha256']!=expected): raise ValueError('Existing pinned archive binding mismatch')
        return bound
    part=path.with_suffix(path.suffix+'.part'); size=record['size']; offset=part.stat().st_size if part.exists() else 0
    if offset>size: raise ValueError('Partial exceeds source')
    url=f'https://huggingface.co/datasets/espnet/yodas3/resolve/{YODAS_REV}/{record["path"]}'
    for attempt in range(6):
        guard.wait(size-offset)
        try:
            if offset<size:
                headers={'User-Agent':'bounded-CPU-preparation-v2'}
                if offset: headers['Range']=f'bytes={offset}-'
                with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=60) as response:
                    if offset and (response.status!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {offset}-')): raise ValueError('Range resume refused; partial preserved')
                    with part.open('ab' if offset else 'wb') as out:
                        last=0
                        while True:
                            guard.wait()
                            block=response.read(2*1024*1024)
                            if not block:break
                            if offset+len(block)>size:raise ValueError('Pinned length exceeded')
                            out.write(block);offset+=len(block)
                            if time.monotonic()-last>20:
                                out.flush();guard.status(phase='downloading_original_compressed',active_source=record['path'],downloaded_bytes=offset,expected_bytes=size);last=time.monotonic()
                if offset!=size:raise OSError('Short response')
            guard.status(phase='verifying_pinned_SHA256',active_source=record['path'],downloaded_bytes=offset)
            observed=sha256(part)
            if expected and observed!=expected:raise ValueError('Pinned LFS SHA256 mismatch; partial preserved')
            part.replace(path)
            value={'path':str(path),'bytes':size,'sha256':observed,'repository':'espnet/yodas3','revision':YODAS_REV,
                'source_path':record['path'],'source_url':url,'completed_at_utc':now(),'no_transcode':True}
            write_json(receipt,value);return value
        except (OSError,TimeoutError) as e:
            guard.status(phase='download_retry_partial_preserved',attempt=attempt+1,error=repr(e),downloaded_bytes=offset)
            if attempt==5:raise
            time.sleep(min(30,5*(attempt+1)))

def index_archive(shard,guard):
    path=ROOT/f'datasets/yodas3/audio/{shard}.tar'; target=ROOT/f'indexes/yodas3_tar_{shard}.sqlite'
    db=sqlite3.connect(target);db.execute('PRAGMA temp_store=MEMORY');db.execute('PRAGMA cache_size=-4096')
    db.execute('CREATE TABLE IF NOT EXISTS recordings(recording_id TEXT PRIMARY KEY,member_name TEXT,offset INTEGER,bytes INTEGER)')
    n=0
    with tarfile.open(path,'r:') as tar:
        for item in tar:
            if n%100==0:guard.wait()
            if not item.isfile() or not item.name.endswith('.webm'):continue
            member=pathlib.PurePosixPath(item.name)
            if member.is_absolute() or '..' in member.parts or item.offset_data<0 or item.offset_data+item.size>path.stat().st_size:raise ValueError('Unsafe compressed extent')
            db.execute('INSERT OR REPLACE INTO recordings VALUES(?,?,?,?)',(member.stem,item.name,item.offset_data,item.size));n+=1
    db.commit();db.close();return n

def main():
    guard=Guard('additional_stage_v2');plan_path=AREA/'additional_stage_plan_v2.json'
    inventory=json.loads((ROOT/'datasets/yodas3/source_audio_inventory.json').read_text(encoding='utf-8'))
    assert inventory['revision']==YODAS_REV
    mapping={pathlib.PurePosixPath(x['path']).stem:x for x in inventory['files'] if x['type']=='file'}
    if plan_path.exists(): plan=json.loads(plan_path.read_text(encoding='utf-8'))
    else:
        review=json.loads((ROOT/'analysis/alania_yodas_preparation_v1/metadata_review.json').read_text(encoding='utf-8'))
        selected=[];total=0
        existing=[s for s in mapping if (ROOT/f'datasets/yodas3/audio/{s}.tar').exists()]
        for row in review['per_shard']:
            s=row['shard'];size=mapping[s]['size']
            if s in existing or total+size>CAP:continue
            selected.append({**row,'compressed_bytes':size});total+=size
        plan={'kind':'separate_additional_compressed_stage_v2','created_at_utc':now(),'revision':YODAS_REV,
            'additional_cap_bytes':CAP,'selected_bytes':total,'minimum_free_E_bytes':100_000_000_000,
            'excluded_already_complete_shards':existing,'selected_shards':selected,'accepted_TRAIN_hours':0,
            'metadata_candidate_hours':sum(x['candidate_hours'] for x in selected),'original_initial_30GB_phase_unchanged':True}
        write_json(plan_path,plan)
    guard.status(plan_path=str(plan_path),planned_shards=[x['shard'] for x in plan['selected_shards']],planned_bytes=plan['selected_bytes'])
    meta=json.load(urllib.request.urlopen(f'https://huggingface.co/api/datasets/espnet/yodas3/tree/{YODAS_REV}/data/tr/metadata?recursive=false&limit=100',timeout=60))
    complete=[]
    for row in plan['selected_shards']:
        shard=row['shard'];record=mapping[shard]
        md=next(x for x in meta if x['path']==f'data/tr/metadata/{shard}.parquet')
        download(md,ROOT/f'datasets/yodas3/metadata/{shard}.parquet',guard)
        receipt=download(record,ROOT/f'datasets/yodas3/audio/{shard}.tar',guard)
        recordings=index_archive(shard,guard)
        value={'shard':shard,'receipt':receipt,'recordings_indexed':recordings,'candidate_hours':row['candidate_hours'],'completed_at_utc':now()}
        append_json(AREA/'additional_stage_ledger_v2.jsonl',value);complete.append(value)
        guard.status(phase='compressed_shard_complete',completed_shards=[x['shard'] for x in complete],completed_bytes=sum(x['receipt']['bytes'] for x in complete),metadata_candidate_hours=sum(x['candidate_hours'] for x in complete))
    guard.status(phase='completed_candidates_only',actual_CLOSED0=True,completed_shards=[x['shard'] for x in complete],completed_bytes=sum(x['receipt']['bytes'] for x in complete),metadata_candidate_hours=sum(x['candidate_hours'] for x in complete))

if __name__=='__main__':main()
