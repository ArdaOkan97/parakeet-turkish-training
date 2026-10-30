"""Bounded original72 audio duplicate triage against actual frozen heldout WAVs.

Exact PCM/content hashes and conservative spectral retrieval are partial checks;
negative retrieval never certifies complete cross-source audio exclusion.
"""
import collections
import hashlib
import json
import pathlib
import sqlite3
import time
import wave

from cpu_preparation_guard_v2 import AREA,ROOT,Guard,append_json,now,write_json
from validate_alania_independent_cpu_v2 import decode

def features(pcm):
    import numpy as np
    if len(pcm)<1024:return np.empty((0,32),np.float32),np.empty(0,np.int32)
    #32 log-frequency spectral bands,64ms window,20ms hop,5 feature frames/s.
    windows=np.lib.stride_tricks.sliding_window_view(pcm,1024)[::320]
    spectrum=np.abs(np.fft.rfft(windows*np.hanning(1024).astype(np.float32),axis=1))**2
    edges=np.unique(np.rint(np.geomspace(6,490,33)).astype(int))
    # Pad low duplicate geometric edges to a fixed32 bands deterministically.
    edges=np.rint(np.geomspace(6,490,33)).astype(int)
    bands=np.stack([np.log1p(spectrum[:,edges[i]:max(edges[i]+1,edges[i+1])].mean(axis=1)) for i in range(32)],axis=1)
    vectors=[];positions=[]
    for at in range(0,len(bands)-9,10):
        if float(np.mean(pcm[at*320:min(len(pcm),(at+10)*320)]**2))<1e-7:continue
        vector=bands[at:at+10].mean(axis=0);vector-=vector.mean()
        norm=float(np.linalg.norm(vector))
        if norm<1e-6:continue
        vectors.append(vector/norm);positions.append(at//10)
    return np.asarray(vectors,dtype=np.float32).reshape(-1,32),np.asarray(positions,dtype=np.int32)

def read_wav(path):
    import numpy as np
    with wave.open(str(path),'rb') as f:
        if f.getnchannels()!=1 or f.getsampwidth()!=2 or f.getframerate()!=16000:raise ValueError('Unexpected bound heldout format')
        raw=f.readframes(f.getnframes())
    return np.frombuffer(raw,dtype='<i2').astype(np.float32)/32768,raw

def retrieve(q,qpos,allvectors,refids,refpos,sourceids,lengths):
    import numpy as np
    if not len(q):return []
    scores=q@allvectors.T;best=np.argmax(scores,axis=1);similar=scores[np.arange(len(q)),best]
    votes=collections.Counter()
    for i,index in enumerate(best):
        if similar[i]>=.985:
            #200ms frames;400ms offset bins tolerate small frame displacement.
            votes[(int(refids[index]),int(round((int(refpos[index])-int(qpos[i]))/2)))]+=1
    matches=[]
    for (rid,offset),count in votes.most_common(5):
        ratio=count/max(1,min(len(q),lengths[rid]))
        if count>=8 and ratio>=.4:matches.append({'heldout_source_id':sourceids[rid],
            'aligned_best_frame_votes':count,'shorter_feature_sequence_fraction':ratio,'offset_bin_400ms':offset})
    return matches

def main():
    import numpy as np
    guard=Guard('pilot_audio_exclusion_v2');source_db=ROOT/'indexes/alania_source_exclusions_v2.sqlite'
    while True:
        guard.wait()
        if source_db.exists():
            db=sqlite3.connect('file:'+source_db.as_posix()+'?mode=ro',uri=True)
            try:count=db.execute('SELECT COUNT(*) FROM reference_audio').fetchone()[0]
            except sqlite3.OperationalError:count=0
            if count>=900:break
            db.close()
        guard.status(phase='waiting_for_actual900_heldout_audio_hash_bindings');time.sleep(10)
    references=db.execute('SELECT source_id,audio_filepath,PCM_i16_sha256 FROM reference_audio ORDER BY source_id').fetchall();db.close()
    index=sqlite3.connect(ROOT/'indexes/heldout_spectral_fingerprints_v2.sqlite');index.execute('PRAGMA cache_size=-4096')
    index.execute('CREATE TABLE IF NOT EXISTS fingerprints(source_id TEXT PRIMARY KEY,vectors BLOB,positions BLOB,frames INTEGER)')
    allvectors=[];refids=[];refpos=[];sourceids=[];lengths=[];pcm_hashes=collections.defaultdict(list)
    first_pcm=None
    for i,(sid,path,digest) in enumerate(references):
        guard.wait();pcm_hashes[digest].append(sid);sourceids.append(sid)
        row=index.execute('SELECT vectors,positions,frames FROM fingerprints WHERE source_id=?',(sid,)).fetchone()
        if row:
            vec=np.frombuffer(row[0],dtype='<f4').reshape(row[2],32);pos=np.frombuffer(row[1],dtype='<i4')
            if i==0:first_pcm=read_wav(path)[0]
        else:
            pcm,raw=read_wav(path)
            if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('Bound heldout PCM changed')
            if i==0:first_pcm=pcm.copy()
            vec,pos=features(pcm);del pcm
            index.execute('INSERT INTO fingerprints VALUES(?,?,?,?)',(sid,vec.astype('<f4').tobytes(),pos.astype('<i4').tobytes(),len(vec)))
        lengths.append(len(vec));allvectors.append(vec);refids.extend([i]*len(vec));refpos.extend(pos.tolist())
        if (i+1)%50==0:index.commit();guard.status(phase='heldout_spectral_index',indexed_heldout_sources=i+1)
    index.commit();index.close();allvectors=np.concatenate(allvectors);refids=np.asarray(refids,dtype=np.int32);refpos=np.asarray(refpos,dtype=np.int32)
    controls=[]
    for name,pcm in [('identical_bound_heldout',first_pcm),('amplitude0_7',first_pcm*.7),('crop400ms',first_pcm[6400:])]:
        vec,pos=features(pcm);matches=retrieve(vec,pos,allvectors,refids,refpos,sourceids,lengths)
        passed=any(x['heldout_source_id']==sourceids[0] for x in matches)
        controls.append({'control':name,'source_id':sourceids[0],'retrieved_own_source':passed,'matches':matches})
    if not all(c['retrieved_own_source'] for c in controls):
        write_json(AREA/'audio_fingerprint_controls_v2.json',{'passed':False,'controls':controls});raise ValueError('Conservative audio retrieval controls failed')
    write_json(AREA/'audio_fingerprint_controls_v2.json',{'passed':True,'controls':controls,
        'scope':'Three positive synthetic transforms of one bound heldout source; not a complete sensitivity/specificity study'})
    manifest=[json.loads(x) for x in (ROOT/'analysis/alania_yodas_preparation_v1/stratified_QC_sample_v1.jsonl').read_text(encoding='utf-8').splitlines()]
    ledger=AREA/'pilot_audio_exclusion_ledger_v2.jsonl';rows=[json.loads(x) for x in ledger.read_text(encoding='utf-8').splitlines()] if ledger.exists() else []
    done={x['segment_id'] for x in rows};compressed_seen=collections.defaultdict(list)
    for row in rows:
        if row.get('compressed_member_sha256'):compressed_seen[row['compressed_member_sha256']].append(row['recording_id'])
    for row in manifest:
        if row['segment_id'] in done:continue
        guard.wait();out={k:row[k] for k in ['segment_id','recording_id','shard','start_s','duration_s']}
        try:
            pcm,info=decode(row,guard);out.update(info)
            canonical=np.clip(np.rint(pcm*32768),-32768,32767).astype('<i2')
            digest=hashlib.sha256(canonical.tobytes()).hexdigest();vec,pos=features(pcm);del pcm,canonical
            matches=retrieve(vec,pos,allvectors,refids,refpos,sourceids,lengths)
            out.update(audio_check_completed=True,PCM_i16_sha256=digest,exact_heldout_PCM_hash_matches=pcm_hashes.get(digest,[]),
                conservative_heldout_spectral_retrieval_matches=matches,
                identical_compressed_other_recording_ids=sorted(set(compressed_seen[info['compressed_member_sha256']])-{row['recording_id']}))
            compressed_seen[info['compressed_member_sha256']].append(row['recording_id'])
        except Exception as error:out.update(audio_check_completed=False,error=repr(error))
        out.update(completed_at_utc=now(),accepted_TRAIN=False);append_json(ledger,out);rows.append(out);done.add(row['segment_id'])
        report={'kind':'original72_partial_heldout_audio_duplicate_triage_v2','updated_at_utc':now(),'declared_clips':72,
            'durable_observations':len(rows),'audio_checks_completed':sum(bool(x.get('audio_check_completed')) for x in rows),
            'exact_PCM_match_clips':sum(bool(x.get('exact_heldout_PCM_hash_matches')) for x in rows),
            'spectral_match_triage_clips':sum(bool(x.get('conservative_heldout_spectral_retrieval_matches')) for x in rows),
            'identical_compressed_different_recording_clips':sum(bool(x.get('identical_compressed_other_recording_ids')) for x in rows),
            'heldout_audio_sources':len(references),'fingerprint_feature_frames':len(allvectors),'positive_controls_passed':True,
            'accepted_TRAIN_hours':0,'GPU_used':False,'TEST_predictions_read':False,
            'scope':'Exact normalized16kHz PCM hashes; exact compressed recording hashes within tested sample; conservative32band spectral aligned retrieval against900 actually bound heldout audio sources',
            'limitations':['No match is not proof of different source/audio; no complete audio exclusion claim.',
                'Spectral retrieval is a triage heuristic with only positive controls; source rights/speaker/transform invariance remain unresolved.',
                'Declared900 heldout sources cover existing six-language manifests; missing Portuguese coverage is not supplied.',
                'Whole-existing-TRAIN audio fingerprint comparison remains pending.']}
        write_json(AREA/'pilot_audio_exclusion_report_v2.json',report)
        guard.status(phase='candidate_audio_hash_and_spectral_triage',completed_samples=len(rows),audio_checks_completed=report['audio_checks_completed'],spectral_match_triage_clips=report['spectral_match_triage_clips'])
    guard.status(phase='completed_declared_partial_audio_checks',actual_CLOSED0=True)

if __name__=='__main__':main()
