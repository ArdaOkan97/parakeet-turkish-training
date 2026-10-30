"""NEW bounded SQL live-set and exact packed operation-witness successor."""
from pathlib import Path
import ast,hashlib
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';R=P/'work/research'
old=S/'asr18_full_run_parent_controller_v3.py';new=S/'asr18_full_run_parent_controller_v4.py'
assert not new.exists();text=old.read_text(encoding='utf-8-sig')
text=text.replace('import gc,hashlib,inspect,json,os,secrets,shutil,time','import gc,hashlib,inspect,json,os,secrets,shutil,time,struct,zlib')
a=text.index('    def _retire_witness(');b=text.index('\n    def _DEV_folder(',a)
text=text[:a]+'''    def _retire_witness(self,pin,mutation):
        # An immutable range in ONE bounded source-owned control stream keeps
        # every original witness byte/pin, including retirement operations whose
        # store event itself only contains the original witness binding.
        require(record(pin['path'])==pin,'Same returned-operation witness before exact owned retirement')
        path=Path(pin['path']).resolve()
        require(path.parent==self.control and '.proof.' in path.name,'Only exact new operation witness')
        original=path.read_bytes()
        event={'kind':'ASR18_exact_archived_operation_witness_v1','original':pin,
            'mutation':mutation,'raw_UTF8':original.decode('utf-8')}
        raw=canonical(event);packed=zlib.compress(raw,9)
        require(len(raw)<=65536 and len(packed)<=2048,'Complete lossless witness frame cap; no field dropping')
        log=self.control/'storage_operation_witnesses.bin'
        offset=log.stat().st_size if log.exists() else 0
        frame=struct.pack('<II',len(packed),len(raw))+hashlib.sha256(raw).digest()+packed
        require(offset+len(frame)<=805306368,'Entire declared operation-witness control horizon')
        self.parent.control_guard(log,offset+len(frame))
        with log.open('ab') as stream:stream.write(frame);stream.flush();os.fsync(stream.fileno())
        with log.open('rb') as stream:stream.seek(offset);actual=stream.read(len(frame))
        decoded=zlib.decompress(actual[40:])
        require(actual==frame and decoded==raw and json.loads(decoded)['raw_UTF8'].encode('utf-8')==original
            and hashlib.sha256(original).hexdigest()==pin['sha256'] and len(original)==pin['bytes'],
            'Actual fsynced immutable complete witness range/readback before single owned unlink')
        path.unlink();require(not path.exists(),'Actual exact transient witness retirement')
''' +text[b:]
oldpart="""                state,entries,_=self.store._load()
                protected=sorted(set(state['winners']+([state['latest']] if state['latest'] else [])))
                obsolete=[key for key,entry in entries.items() if key not in protected and not entry.get('retired')]
                db=self.store._db()
                try:aux=[row[0] for row in db.execute("SELECT id FROM auxiliary WHERE status='archived' ORDER BY id LIMIT 6")]
                finally:db.close()"""
newpart="""                state=self.store._load()[0]
                protected=sorted(set(state['winners']+([state['latest']] if state['latest'] else [])))
                db=self.store._db()
                try:
                    live=[row[0] for row in db.execute('SELECT id FROM checkpoints WHERE retired=0 ORDER BY id LIMIT 6')]
                    aux=[row[0] for row in db.execute("SELECT id FROM auxiliary WHERE status='archived' ORDER BY id LIMIT 6")]
                finally:db.close()
                require(len(live)<=5,'Exact bounded B-tree live set; never iterate _Rows')
                for key in live:self.store._entry(key,verify_files=True)
                obsolete=[key for key in live if key not in protected]"""
assert text.count(oldpart)==1;text=text.replace(oldpart,newpart)
ast.parse(text);compile(text,str(new),'exec');new.write_text(text,encoding='utf-8',newline='\n')
data=new.read_bytes();print({'path':str(new),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'AST_only':True})
old=R/'run_ASR18_full_run_owner_v4.py';new=R/'run_ASR18_full_run_owner_v5.py'
assert not new.exists();text=old.read_text(encoding='utf-8-sig')
text=text.replace('from asr18_full_run_parent_controller_v3 import run','from asr18_full_run_parent_controller_v4 import run')
text=text.replace("size<=(268435456 if name=='transient_retirement.jsonl' else 16777216)",
                  "size<=(805306368 if name=='storage_operation_witnesses.bin' else 16777216)")
ast.parse(text);compile(text,str(new),'exec');new.write_text(text,encoding='utf-8',newline='\n')
data=new.read_bytes();print({'path':str(new),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'AST_only':True})
