"""Exact cleaner zero-based ordinal and uint16 BLOB format; no label rewrite."""
import hashlib,json,sqlite3,struct
from pathlib import Path

def public_to_label_ordinal(public):
    assert type(public) is int and 1<=public<=352768
    return public-1

def cleaner_label_ids(blob,token_count,expected_label_sha):
    assert type(token_count) is int and token_count>0 and len(blob)==2*token_count
    ids=list(struct.unpack('<'+'H'*token_count,blob))
    assert all(0<=x<8711 and x!=8192 for x in ids)
    assert hashlib.sha256(json.dumps(ids,sort_keys=True).encode()).hexdigest()==expected_label_sha
    return ids

def validate_cleaner_boundaries(project,labels,tokenizer,guard):
    directory=Path(project)/'work/data/TR_expanded_WorldSpeech_index_v1'
    db=sqlite3.connect(Path(labels).as_uri()+'?mode=ro&immutable=1',uri=True)
    db.execute('PRAGMA query_only=ON');db.execute('PRAGMA cache_size=-1024')
    assert db.execute('SELECT MIN(ordinal),MAX(ordinal) FROM labels').fetchone()==(0,352767)
    checked=[];samples=(1,2,72846,72847,74883,74884,253798,253799,352767,352768)
    with (directory/'offsets.u64').open('rb') as offsets,(directory/'rows.jsonl').open('rb') as source:
        for public in samples:
            guard.check();index=public_to_label_ordinal(public);offsets.seek(index*8)
            start,end=struct.unpack('<QQ',offsets.read(16));assert 0<end-start<=65536
            source.seek(start);raw=source.read(end-start);guard.pace(len(raw)+16);row=json.loads(raw)
            ordinal,sid,text_sha,u,label_sha,blob=db.execute('SELECT ordinal,source_id,text_sha,new_u,new_labels_sha,label_ids FROM labels WHERE ordinal=?',(index,)).fetchone()
            assert ordinal==index and sid==row['source_id']
            assert hashlib.sha256(row['text'].encode()).hexdigest()==text_sha
            ids=cleaner_label_ids(blob,u,label_sha)
            assert ids==tokenizer.encode(row['text'],add_special_tokens=True).ids
            checked.append({'public_ordinal':public,'label_DB_ordinal':index,'source_id':sid,'text_sha256':text_sha,'labels_SHA':label_sha,'target_tokens':len(ids),'packed_dtype':'uint16_le','IDs_and_source_and_label_hash_exact':True})
    db.close()
    return {'passed':True,'samples':checked,'scope':'Real first/last cleaner rows plus four dataset boundaries; exact original uint16 unpacking and zero-based DB ordinal.','all352768_label_alignment_proven':False}
