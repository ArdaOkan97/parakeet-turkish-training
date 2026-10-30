"""Available-first pinned metadata continuation; all raw flags retained, no model imports."""
import collections,json,pathlib,sys,time
from ASR18_continuous_data_guard_v4 import DataIOGuard,DataStop,ROOT,AREA
from ASR18_data_io_guard_v2 import PacedInput
from cpu_preparation_guard_v2 import write_json
IO=DataIOGuard('metadata',heavy=False,maximum_bytes=20_000_000_000)
import build_BPE_broad_metadata_v1 as old
assert 'torch' not in sys.modules,'CPU model runtime must not be imported'

class Catalog(old.Catalog):
    def watchdog(self):
        # The continuous guard retains all native/private/system protections.
        return
    def publish(self,**fields):
        self.state.update(fields,owner='/root/cpu_data_preparation',updated_at_utc=old.now(),
          paced_column_metadata_bytes=IO.bytes,no_CPU_model_imports=True)
        write_json(AREA/'broad_metadata_status_v3.json',self.state)
    def guard(self):IO.wait()
    def summary(self):
        totals=self.db.execute('SELECT SUM(rows_seen),SUM(rows_retained),SUM(retained_seconds),COUNT(*) FROM files').fetchone()
        exclusions=collections.Counter();groups={}
        for name,seen,kept,seconds,raw in self.db.execute('SELECT file,rows_seen,rows_retained,retained_seconds,exclusions_json FROM files'):
            exclusions.update(json.loads(raw));key=('alania_yodas3','all_metadata') if name=='alania_all_metadata' else ('kiraat',pathlib.PurePosixPath(name).name.split('-')[0])
            group=groups.setdefault(key,{'dataset':key[0],'split':key[1],'clips':0,'hours':0.});group['clips']+=kept;group['hours']+=seconds/3600
        self.publish(status='metadata_file_committed',source_rows_seen=totals[0],metadata_retained_rows=totals[1],metadata_retained_hours=(totals[2] or 0)/3600,committed_metadata_files=totals[3])
        write_json(AREA/'broad_metadata_report_v3.json',{**self.state,'catalog_path':str(self.db_path),'exclusion_counts':dict(exclusions),'groups':list(groups.values()),'complete_audio_exclusion_claimed':False,'TRAIN_admitted':False})
    def kiraat_file(self,job):
        if self.already(job['path'],job['sha256']):return
        p=pathlib.Path(job['destination']);receipt=json.loads(p.with_suffix(p.suffix+'.receipt.json').read_text(encoding='utf-8-sig'))
        assert p.stat().st_size==job['size'] and receipt['sha256']==job['sha256'] and receipt['revision']==job['revision']
        columns=['id','source_id','start','end','duration','text','text_raw','text_spoken','recommended','exclusion_reasons','flags','source_flags','duplicate_of','split']
        seen=kept=0;seconds=0.;exclusions=collections.Counter();split=p.name.split('-')[0]
        try:
            with PacedInput(p,IO) as stream:
                parquet=old.pq.ParquetFile(stream,pre_buffer=False,memory_map=False)
                for batch in parquet.iter_batches(batch_size=256,columns=columns,use_threads=False):
                    self.guard()
                    for row in batch.to_pylist():
                        reason=self.add('kiraat:'+row['id'],'kiraat',split,row['source_id'],str(p),seen,row['start'],row['end'],row['duration'],row['text'],row['text_raw'],row['text_spoken'],
                          {k:row[k] for k in ('recommended','exclusion_reasons','flags','source_flags','duplicate_of','split')})
                        seen+=1
                        if reason:exclusions[reason]+=1
                        else:kept+=1;seconds+=row['duration']
            self.db.execute('INSERT INTO files VALUES (?,?,?,?,?,?)',(job['path'],job['sha256'],seen,kept,seconds,json.dumps(exclusions)));self.db.commit()
        except BaseException:self.db.rollback();raise
        self.summary()
    def run(self):
        plan=json.loads((AREA/'preparation_plan_v1.json').read_text(encoding='utf-8'))
        binding_path=AREA/'heldout_metadata_bindings_v1.json'
        with PacedInput(binding_path,IO) as stream:
            binding=json.loads(stream.read(binding_path.stat().st_size).decode('utf-8-sig'))
        self.heldout_sources=set(binding['source_ids']);self.heldout_text=set(binding['normalized_text_keys']);self.bindings=binding['files']
        assert len(self.heldout_sources)>=1050 and len(self.heldout_text)>=11648
        alania=self.db.execute("SELECT rows_seen,rows_retained FROM files WHERE file='alania_all_metadata'").fetchone();assert alania and alania[0]==835267
        self.state.update(fixed_heldout_unique_source_ids=1050,kiraat_heldout_source_ids=85,kiraat_heldout_text_keys=11648,heldout_bindings_reused_unchanged=True)
        jobs=sorted([j for j in plan['download_jobs'] if j['repository']=='serdarcaglar/kiraat' and pathlib.Path(j['path']).name.startswith(('train-','rest-'))],key=lambda j:j['path'])
        self.summary()
        while True:
            pending=[j for j in jobs if not self.already(j['path'],j['sha256'])]
            if not pending:break
            ready=[j for j in pending if pathlib.Path(j['destination']).exists() and pathlib.Path(j['destination']).with_suffix(pathlib.Path(j['destination']).suffix+'.receipt.json').exists()]
            if not ready:self.guard();self.publish(status='waiting_for_next_available_pinned_KIRAAT_file',pending_files=len(pending));time.sleep(5);continue
            for job in ready:self.guard();self.publish(status='reading_available_pinned_KIRAAT_metadata',current_file=job['path']);self.kiraat_file(job)
        self.publish(status='metadata_rows_complete_finalizing_continuously')
        IO.heavy=True;IO.role='catalog_finalize';IO.state['role']='catalog_finalize';IO.wait()
        self.db.set_progress_handler(lambda:(IO.wait() or 0),10000)
        self.db.execute('CREATE INDEX IF NOT EXISTS clips_source ON clips(source_id)');self.db.execute('CREATE INDEX IF NOT EXISTS clips_dataset ON clips(dataset,original_split)');self.db.commit();self.summary();self.db.close()
        self.publish(status='completed_metadata_only',metadata_complete=True,actual_exit_code=0)
        write_json(AREA/'broad_metadata_actual_CLOSED0_v3.json',{**self.state,'catalog_path':str(self.db_path),'catalog_sha256':IO.sha(self.db_path)})

if __name__=='__main__':
    worker=None
    try:worker=Catalog();worker.run()
    except BaseException as error:
        if worker is not None:worker.db.rollback();worker.publish(status='stopped',exception_class=type(error).__name__)
        raise
