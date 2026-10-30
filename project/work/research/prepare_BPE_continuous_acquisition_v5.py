"""Paced pausable successor of the fixed ASR18 acquisition plan; no model imports."""
import hashlib,json,os,pathlib,sys,time
import requests
from huggingface_hub import get_token
import prepare_BPE_weak_then_clean_data_v1 as original
from ASR18_continuous_data_guard_v5 import DataIOGuard,DataStop,ROOT,AREA
from cpu_preparation_guard_v2 import scoped
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]/"outputs/parakeet_turkish"))
from atomic_io_retry_v1 import write_json
original.write_json=write_json
original.STATUS=AREA/'acquisition_status_v5.json'

class Preparation(original.Preparation):
    def __init__(self):
        self.io=DataIOGuard('acquisition',heavy=True,maximum_bytes=800_000_000_000)
        self.active_output=None;self.active_response=None;self.reconnect=False
        self.io.callback=self.pause_flush_close
        super().__init__()
    def watchdog(self):
        # DataIOGuard owns the unchanged native/private/system watchdog; preserve v1 receipts.
        return
    def publish(self,**fields):
        fields.update(owner='/root/cpu_data_preparation',actual_GPU_used=False,training_started=False,
          paced_IO_bytes=self.io.bytes,source_preserved=True,steady_safe_permit_required=False,continuously_concurrent_IO=True,maximum_IO_bytes_s=33554432,own_RSS_bytes=self.process.memory_info().rss)
        super().publish(**fields)
    def guard(self,force=False,reserve=0):
        self.io.wait(reserve);self.state.update(memory=original.memory(),own_private_bytes=self.process.memory_info().private,
          free_E_bytes=original.shutil.disk_usage(ROOT).free)
    def pause_flush_close(self):
        if self.active_output and not self.active_output.closed:self.active_output.flush();os.fsync(self.active_output.fileno())
        if self.active_response:self.active_response.close();self.reconnect=True
        if hasattr(self,'state'):self.publish(status='paused_resource_or_bounded_named_measurement',partial_flushed=True)
    def download(self,job):
        destination=scoped(job['destination']);destination.parent.mkdir(parents=True,exist_ok=True)
        expected=job['sha256'];assert expected
        receipt_path=destination.with_suffix(destination.suffix+'.receipt.json')
        if destination.exists():
            receipt=json.loads(receipt_path.read_text(encoding='utf-8-sig'))
            assert destination.stat().st_size==job['size'] and receipt['sha256']==expected and receipt['revision']==job['revision']
            return {'path':job['path'],'reused':True,'sha256':expected,'bytes':job['size']}
        partial=destination.with_suffix(destination.suffix+'.part');offset=partial.stat().st_size if partial.exists() else 0
        assert offset<=job['size'];digest=hashlib.sha256()
        if offset:
            self.publish(status='paced_rehash_existing_partial',current_file=job['path'],current_offset_bytes=offset)
            with partial.open('rb',buffering=0) as prefix:
                while True:
                    self.io.pace(1048576);block=prefix.read(1048576)
                    if not block:break
                    digest.update(block)
        with partial.open('ab' if offset else 'wb') as out:
            self.active_output=out
            while offset<job['size']:
                self.guard(force=True)
                headers={'Accept-Encoding':'identity'}
                token=get_token() if job['repository']=='serdarcaglar/kiraat' else None
                if token:headers['Authorization']='Bearer '+token
                if offset:headers['Range']=f'bytes={offset}-'
                url=f"https://huggingface.co/datasets/{job['repository']}/resolve/{job['revision']}/{job['path']}"
                with requests.get(url,headers=headers,stream=True,timeout=(30,60)) as response:
                    assert response.status_code in ([206] if offset else [200,206]),'Pinned source HTTP status failure'
                    if offset:assert response.headers.get('Content-Range','').startswith(f'bytes {offset}-'),'Range resume refused'
                    self.active_response=response;self.reconnect=False;iterator=response.iter_content(chunk_size=1048576)
                    self.publish(status='paced_downloading',current_repository=job['repository'],current_file=job['path'],current_offset_bytes=offset,current_size_bytes=job['size'])
                    last=time.monotonic()
                    while offset<job['size']:
                        self.io.pace(1048576)
                        if self.reconnect:break
                        try:block=next(iterator)
                        except StopIteration:break
                        self.guard()
                        if self.reconnect:break
                        if not block:continue
                        assert offset+len(block)<=job['size'],'Pinned size exceeded'
                        out.write(block);digest.update(block);offset+=len(block);self.received+=len(block)
                        if self.received>self.plan['new_download_bytes']:raise DataStop('fixed_new_download_plan_budget_exceeded')
                        if time.monotonic()-last>10:
                            out.flush();self.publish(status='paced_downloading',current_offset_bytes=offset);last=time.monotonic()
                    self.active_response=None
                if not self.reconnect and offset<job['size']:raise RuntimeError('Short_response_partial_preserved')
            out.flush();os.fsync(out.fileno());self.active_output=None
        assert offset==job['size'] and digest.hexdigest()==expected,'Pinned_size_or_SHA256_mismatch_partial_preserved'
        partial.replace(destination)
        write_json(receipt_path,{**job,'completed_at_utc':original.now(),'sha256':digest.hexdigest(),'bytes':offset,
          'owner':'/root/cpu_data_preparation','task':'ASR-18','quality_filter_applied':False,'source_revision_pinned':True})
        return {'path':job['path'],'sha256':expected,'reused':False,'bytes':offset}
    def run(self):
        jobs=self.plan['download_jobs']
        front=[j for j in jobs if j['repository']=='serdarcaglar/kiraat' and pathlib.PurePosixPath(j['path']).name.startswith(('train-00000-','rest-00000-','dev-','test-'))]
        # One deterministic KIRAAT file then one YODAS file keeps labels progressing; existing partials retained.
        kiraat=[j for j in jobs if j['repository']=='serdarcaglar/kiraat' and j not in front]
        yodas=[j for j in jobs if j['repository']=='espnet/yodas3']
        ordered=list(front)
        while kiraat or yodas:
            if kiraat:ordered.append(kiraat.pop(0))
            if yodas:ordered.append(yodas.pop(0))
        assert len(ordered)==len(jobs)
        for job in ordered:
            self.guard(force=True);item=self.download(job);self.completed.append(item)
            write_json(AREA/'acquisition_completed_files_v5.json',self.completed)
            self.publish(status='file_complete',current_file=job['path'])
        self.publish(status='completed',finished_at_utc=original.now())
        write_json(AREA/'acquisition_actual_CLOSED0_v5.json',{**self.state,'actual_exit_code':0})

if __name__=='__main__':
    worker=None
    try:worker=Preparation();worker.run()
    except BaseException as error:
        safe=str(error) if isinstance(error,(DataStop,ValueError,AssertionError,RuntimeError)) else type(error).__name__
        if worker is not None:worker.pause_flush_close();worker.publish(status='stopped',failure=safe,partials_preserved=True)
        print(json.dumps({'status':'stopped','reason':safe}),flush=True);sys.exit(75 if isinstance(error,DataStop) else 1)
