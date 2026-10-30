"""Finite E-only index/reader protocol controls using saved actual metadata.

The waveform protocol stub is not a decoder or new audio compatibility proof.
"""
import argparse,json,os,sqlite3,sys
from fractions import Fraction
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from atomic_io_retry_v1 import write_json
from bpe_broad_audio_v1 import record,require
from bpe_declared_geometry_index_v1 import audit_index,DeclaredGeometryIndex,BeforeDrawDeclaredAudioReader,COLS,signature
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--folder',required=True);a=parser.parse_args();f=Path(a.folder)
    require(f.is_relative_to(A),'E-only fixture');f.mkdir(exist_ok=False);paths=[f/(n+'.sqlite') for n in ('catalog','labels','geometry')]
    source=json.loads((A/'runtime_geometry_CPU_actual_02/report.json').read_text());require(source['passed'],'Prior independent saved-actual metadata control required')
    declarations={v['unique_id']:v['declaration'] for v in source['cases']};cases=[]
    for rp in (A/'full14_FLAC_CPU_actual_01/decoded/report.json',A/'hard_U_rows_decode_01/decoded/report.json'):
        for item in json.loads(rp.read_text())['rows']:
            raw={**item['row']['broad_original'],'ordinal':len(cases)+1};declared={**declarations[raw['unique_id']],'ordinal':raw['ordinal']}
            cases.append((raw,declared,item['reader_evidence']))
    with sqlite3.connect(paths[0]) as c:
        c.execute('CREATE TABLE clips(ordinal INTEGER PRIMARY KEY,dataset TEXT,unique_id TEXT,source_id TEXT,start REAL,end REAL)')
        c.executemany('INSERT INTO clips VALUES(?,?,?,?,?,?)',[(r['ordinal'],r['dataset'],r['unique_id'],r['source_id'],r['start'],r['end']) for r,d,e in cases])
    with sqlite3.connect(paths[1]) as c:
        c.execute('CREATE TABLE labels(ordinal INTEGER PRIMARY KEY,target_tokens INTEGER,target_ids_u32 BLOB)')
        c.executemany('INSERT INTO labels VALUES(?,?,?)',[(r['ordinal'],d['target_tokens'],bytes(4*d['target_tokens'])) for r,d,e in cases])
    with sqlite3.connect(paths[2]) as c:
        c.execute('CREATE TABLE geometry(ordinal INTEGER PRIMARY KEY,source_component INTEGER NOT NULL,native_sr INTEGER,native_samples INTEGER,clean16k_samples INTEGER,mel_frames INTEGER,encoder_frames INTEGER,target_tokens INTEGER NOT NULL,status TEXT NOT NULL)')
        c.executemany('INSERT INTO geometry VALUES('+','.join('?'*len(COLS))+')',[tuple(d[k] for k in COLS) for r,d,e in cases])
    tokens=sum(d['target_tokens'] for r,d,e in cases);checks=[]
    def audit():return audit_index(*paths,rows=len(cases),tokens=tokens)
    proof=audit();require(proof['population']['MAX_U']==574 and proof['population']['MAX_T']==1991 and proof['groups']['alania_yodas3']['retained_rows']==6,'Full hard targets and both source components retained');checks.append('Exact16 saved-actual geometry joins retain U574/T1991 and6Alania')
    def reject(name,call):
        try:call()
        except (ValueError,KeyError):checks.append(name);return
        raise ValueError('Expected rejection: '+name)
    def changed(sql,args,name):
        with sqlite3.connect(paths[2]) as c:c.execute(sql,args)
        reject(name,audit)
    changed("UPDATE geometry SET status=? WHERE ordinal=1",('unresolved_unknown_samples',),'Unresolved declared status rejected')
    with sqlite3.connect(paths[2]) as c:c.execute("UPDATE geometry SET status='declared' WHERE ordinal=1")
    changed('UPDATE geometry SET encoder_frames=encoder_frames+1 WHERE ordinal=1',(),'Incorrect full encoder rounding rejected')
    with sqlite3.connect(paths[2]) as c:c.execute('UPDATE geometry SET encoder_frames=encoder_frames-1 WHERE ordinal=1')
    changed('UPDATE geometry SET ordinal=99 WHERE ordinal=2',(),'Noncontiguous/hole/extra geometry ordinal rejected')
    with sqlite3.connect(paths[2]) as c:c.execute('UPDATE geometry SET ordinal=2 WHERE ordinal=99')
    with sqlite3.connect(paths[1]) as c:c.execute("UPDATE labels SET target_ids_u32=x'00' WHERE ordinal=3")
    reject('Truncated original target BLOB rejected',audit)
    u=cases[2][1]['target_tokens']
    with sqlite3.connect(paths[1]) as c:c.execute('UPDATE labels SET target_ids_u32=? WHERE ordinal=3',(bytes(u*4),))
    reject('Partial final denominator rejected',lambda:audit_index(*paths,rows=15,tokens=tokens))
    binding={**record(paths[2]),'mtime_ns':paths[2].stat().st_mtime_ns}
    reject('Boolean index size rejected',lambda:DeclaredGeometryIndex({**binding,'bytes':True}))
    reject('Wrong index SHA rejected',lambda:DeclaredGeometryIndex({**binding,'sha256':'0'*64}))
    index=DeclaredGeometryIndex(binding);row,d,actual=cases[-2]
    class Wave:
        ndim=1;dtype='float32'
        def __init__(self,n):self.n=n
        def __len__(self):return self.n
    class Reader:
        def __init__(self,e,w):self.last=dict(e);self.wave=w;self.calls=0
        def __call__(self,row,**kwargs):self.calls+=1;return self.wave
    wave=Wave(d['clean16k_samples']);original=Reader(actual,wave)
    gate=BeforeDrawDeclaredAudioReader(index,original,full_target_tokens_for=lambda r:d['target_tokens'])
    require(gate(row) is wave and gate.last['declared_geometry_runtime_proof']['actual_native_and_full16k_and_encoder_frame_geometry_matched'],'Gate returns exact full buffer only after actual agreement');checks.append('Protocol stub preserves exact returned buffer and full target/actual geometry identity')
    index.close();index=DeclaredGeometryIndex(binding);reader=Reader({**actual,'decoded_samples':actual['decoded_samples']-1},wave)
    gate=BeforeDrawDeclaredAudioReader(index,reader,full_target_tokens_for=lambda r:d['target_tokens'])
    reject('Actual decoded length mismatch poisons before returned waveform',lambda:gate(row));reject('Poisoned provisional gate cannot continue',lambda:gate(row));require(reader.calls==1,'Poisoned gate must not decode a second row');index.close()
    index=DeclaredGeometryIndex(binding);reader=Reader(actual,Wave(d['clean16k_samples']-1));gate=BeforeDrawDeclaredAudioReader(index,reader,full_target_tokens_for=lambda r:d['target_tokens'])
    reject('Returned waveform count cannot contradict decoder evidence',lambda:gate(row));index.close()
    index=DeclaredGeometryIndex(binding);reader=Reader(actual,wave);gate=BeforeDrawDeclaredAudioReader(index,reader,full_target_tokens_for=lambda r:d['target_tokens'])
    reject('Training ID rewrite cannot replace original catalog source ID',lambda:gate({**row,'source_id':'rewritten_training_id'}));index.close()
    index=DeclaredGeometryIndex(binding);st=paths[2].stat();os.utime(paths[2],ns=(st.st_atime_ns,st.st_mtime_ns+4_000_000_000))
    reject('Final index changed signature stops before decode',lambda:index.declaration(row,full_target_tokens=d['target_tokens']));index.close()
    require(not any(k in sys.modules for k in ('torch','numpy','scipy','transformers','pyarrow')),'No numerical/audio/model imports');checks.append('No new decoder/audio/frontend/model import or sampler/Adam commit')
    write_json(f/'report.json',{'passed':True,'kind':'ASR18_declared_geometry_index_reader_protocol_control_v1','checks':checks,'checks_count':len(checks),
        'scalar_audit':proof,'source':record(__file__),'helper':record(S/'bpe_declared_geometry_index_v1.py'),'saved_actual_reference':record(A/'runtime_geometry_CPU_actual_02/report.json'),
        'fixture_ordinals_renumbered_for_contiguous_control':True,'target_BLOB_values_synthetic_length_control_only':True,'waveform_stub_not_actual_decoder':True,
        'new_audio_read_or_native_updates':0,'actual_source_decoder_integration_proved':False,'full_final_population_or_production_admitted':False})
    print(json.dumps({'passed':True,'checks':len(checks),'native_updates':0}))
if __name__=='__main__':main()
