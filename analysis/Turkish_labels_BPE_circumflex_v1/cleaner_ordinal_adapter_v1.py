"""Exact existing zero-based cleaner label DB versus one-based public rows."""
import array,hashlib,json,sqlite3,struct,sys
from pathlib import Path

def public_to_label_ordinal(public):
    assert type(public) is int and 1<=public<=352768
    return public-1

def validate_cleaner_boundaries(project,labels,tokenizer,guard):
    directory=Path(project)/'work/data/TR_expanded_WorldSpeech_index_v1'
    db=sqlite3.connect(Path(labels).as_uri()+'?mode=ro&immutable=1',uri=True)
    db.execute('PRAGMA query_only=ON');db.execute('PRAGMA cache_size=-1024')
    assert db.execute('SELECT MIN(ordinal),MAX(ordinal) FROM labels').fetchone()==(0,352767)
    checked=[]
    samples=(1,2,72846,72847,74883,74884,253798,253799,352767,352768)
    with (directory/'offsets.u64').open('rb') as offsets,(directory/'rows.jsonl').open('rb') as source:
        for public in samples:
            guard.check();index=public_to_label_ordinal(public);offsets.seek(index*8)
            start,end=struct.unpack('<QQ',offsets.read(16));assert 0<end-start<=65536
            source.seek(start);raw=source.read(end-start);guard.pace(len(raw)+16);row=json.loads(raw)
            ordinal,sid,text_sha,blob=db.execute('SELECT ordinal,source_id,text_sha,label_ids FROM labels WHERE ordinal=?',(index,)).fetchone()
            assert ordinal==index and sid==row['source_id']
            assert hashlib.sha256(row['text'].encode()).hexdigest()==text_sha
            ids=array.array('I');ids.frombytes(blob);assert ids.itemsize==4 and sys.byteorder=='little'
            assert list(ids)==tokenizer.encode(row['text'],add_special_tokens=True).ids
            checked.append({'public_ordinal':public,'label_DB_ordinal':index,'source_id':sid,'text_sha256':text_sha,'target_tokens':len(ids),'IDs_and_source_text_hash_exact':True})
    db.close()
    return {'passed':True,'samples':checked,'scope':'Real first/last cleaner rows plus all four dataset-boundaries; metadata/text/labels only.','all352768_label_alignment_proven':False}
