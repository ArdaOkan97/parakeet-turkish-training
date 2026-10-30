"""Pinned header-only ASR18 index successor; explicit ROOT steady-safe pacing."""
import hashlib,json,sqlite3,sys,tarfile,time
from pathlib import Path,PurePosixPath
from ASR18_continuous_data_guard_v3 import DataIOGuard,DataStop,ROOT,AREA
from ASR18_data_io_guard_v2 import PacedInput
from cpu_preparation_guard_v2 import write_json,now

class HeaderGuard(DataIOGuard):
    def pace(self,count):
        super().pace(count)
        deadline=max(getattr(self,'header_next_at',0),time.monotonic())+count/262144
        while time.monotonic()<deadline:
            self.wait();time.sleep(min(.025,max(0,deadline-time.monotonic())))
        self.header_next_at=deadline

def main():
    guard=HeaderGuard('archive_indexer',heavy=True,maximum_bytes=2_000_000_000)
    guard.state['maximum_header_metadata_IO_bytes_s']=262144
    plan=json.loads((AREA/'preparation_plan_v1.json').read_text(encoding='utf-8'))
    jobs=sorted([j for j in plan['download_jobs']+plan['reused_archives'] if j['repository']=='espnet/yodas3'],key=lambda j:j['path'])
    done=[]
    for job in jobs:
        guard.wait();archive=Path(job['destination']);receipt_path=archive.with_suffix(archive.suffix+'.receipt.json')
        while not archive.exists() or not receipt_path.exists():
            guard.wait();guard.publish('waiting_for_pinned_archive',current_shard=Path(job['path']).stem,completed_archives=len(done));time.sleep(5)
        receipt=json.loads(receipt_path.read_text(encoding='utf-8-sig'))
        assert receipt['sha256']==job['sha256'] and receipt['revision']==job['revision'] and archive.stat().st_size==job['size']
        shard=Path(job['path']).stem;target=ROOT/f'indexes/yodas3_tar_{shard}.sqlite'
        db=sqlite3.connect(target);db.execute('PRAGMA cache_size=-4096');db.execute('PRAGMA temp_store=MEMORY')
        db.execute('CREATE TABLE IF NOT EXISTS recordings(recording_id TEXT PRIMARY KEY,member_name TEXT,offset INTEGER,bytes INTEGER)')
        count=db.execute('SELECT COUNT(*) FROM recordings').fetchone()[0]
        # Nonempty indexes are accepted only after full metadata quick_check and pinned extent bounds.
        reused=count>0
        if not reused:
            guard.publish('paced_TAR_header_indexing',current_shard=shard)
            try:
                with PacedInput(archive,guard) as stream,tarfile.open(fileobj=stream,mode='r:') as tar:
                    for item in tar:
                        guard.wait()
                        if not item.isfile() or not item.name.endswith('.webm'):continue
                        member=PurePosixPath(item.name)
                        assert not member.is_absolute() and '..' not in member.parts and 0<=item.offset_data and item.offset_data+item.size<=job['size']
                        db.execute('INSERT OR REPLACE INTO recordings VALUES(?,?,?,?)',(member.stem,item.name,item.offset_data,item.size))
                db.commit()
            except BaseException:db.rollback();db.close();raise
        guard.wait();assert db.execute('PRAGMA quick_check').fetchone()[0]=='ok'
        count,low,high=db.execute('SELECT COUNT(*),MIN(offset),MAX(offset+bytes) FROM recordings').fetchone()
        assert count>0 and low>=0 and high<=job['size'];db.close()
        digest=guard.sha(target)
        done.append({'shard':shard,'archive_sha256':job['sha256'],'index':str(target),'index_sha256':digest,'recordings':count,'reused_index':reused})
        write_json(AREA/'YODAS_archive_index_progress_v3.json',done)
        guard.publish('archive_index_complete',current_shard=shard,completed_archives=len(done),recordings_indexed=sum(x['recordings'] for x in done))
    guard.publish('completed',actual_exit_code=0,completed_archives=len(done))
    write_json(AREA/'YODAS_archive_indexes_actual_CLOSED0_v3.json',{**guard.state,'indexes':done})
if __name__=='__main__':
    try:main()
    except BaseException as error:
        write_json(AREA/'YODAS_archive_index_stop_v3.json',{'at_utc':now(),'exception_class':type(error).__name__,'frontier_preserved':True})
        raise
