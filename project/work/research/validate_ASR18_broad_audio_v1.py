"""Finite real original-source reader fixture. Never a TRAIN admission/filter."""
import argparse
import array
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys

PROJECT=Path(__file__).resolve().parents[2]
ROOT=Path(r'E:\gpu_training_scratchpad')
sys.path.insert(0,str(PROJECT/'outputs/parakeet_turkish'))
from bpe_broad_audio_v1 import BroadAudioReader,record,require
from atomic_io_retry_v1 import write_json
from resource_watchdog_v2 import sample_memory

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--manifest');p.add_argument('--jsonl',action='store_true');a=p.parse_args()
    folder=Path(a.output);folder.mkdir(parents=True,exist_ok=False)
    require(os.environ['CUDA_VISIBLE_DEVICES']=='-1','CPU-only validation')
    db=sqlite3.connect((ROOT/'indexes/BPE_broad_weak_labels_v1.sqlite').as_uri()+'?mode=ro',uri=True)
    db.row_factory=sqlite3.Row
    if a.manifest:
        if a.jsonl:rows=[json.loads(line) for line in Path(a.manifest).read_text(encoding='utf-8').splitlines() if line.strip()]
        else:manifest=json.loads(Path(a.manifest).read_text());rows=manifest['rows']
    else:
        rows=[]
        for query in ["SELECT * FROM clips WHERE dataset='kiraat' AND start>30 LIMIT 1",
                      "SELECT * FROM clips WHERE dataset='alania_yodas3' AND shard='0000' AND start>3 AND duration>5 AND duration<20 LIMIT 1"]:
            item=db.execute(query).fetchone();require(item is not None,'Available representative source row required');rows.append(dict(item))
    labels=sqlite3.connect((ROOT/'indexes/BPE_broad_labels_8711_v1.sqlite').as_uri()+'?mode=ro',uri=True)
    observations=[]
    def guard():
        m=sample_memory(os.getpid(),'real_broad_audio_boundary')
        require(m['worker_private_memory']['private_bytes']<2_000_000_000,'CPU2GB guard')
        require(m['system_memory']['commit_used_bytes']<.95*m['system_memory']['commit_limit_bytes'] and m['system_memory']['commit_available_bytes']>=2_000_000_000,'Unchanged global guards')
    reader=BroadAudioReader(guard=guard)
    linear_proved=False
    for row in rows:
        expected=dict(db.execute('SELECT * FROM clips WHERE ordinal=?',(row['ordinal'],)).fetchone())
        # Metadata-only diagnostics can supply additional selection fields.
        require(all(expected[k]==row[k] for k in expected),'Exact unchanged catalog row required')
        target=labels.execute('SELECT target_ids_u32,target_tokens,text_sha256,conservative_estimated_joint_bytes FROM labels WHERE ordinal=?',(row['ordinal'],)).fetchone()
        require(target is not None and hashlib.sha256(row['text'].encode()).hexdigest()==target[2],'Labels/catalog text identity')
        ids=array.array('I');ids.frombytes(target[0]);require(len(ids)==target[1] and all(0<=v<8711 and v!=8192 for v in ids),'Exact full target IDs')
        audio=reader(row);info=dict(reader.last);info.update(target_tokens=len(ids),full_joint_bytes=info['encoder_frames']*(len(ids)+1)*8716*4,
            catalog_estimated_joint_bytes=target[3],row=row,target_ids=list(ids))
        digest=hashlib.sha256(audio.tobytes()).hexdigest();del audio
        repeated=reader(row);require(hashlib.sha256(repeated.tobytes()).hexdigest()==digest,'Repeated original-source decode must be byte exact');del repeated
        if row['dataset']=='alania_yodas3' and not linear_proved:
            linear=reader(row,linear=True);require(hashlib.sha256(linear.tobytes()).hexdigest()==digest,'PTS seek differs from full linear decode; do not admit reader');del linear
            info['seek_and_linear_decode_byte_exact']=True;linear_proved=True
        info['repeat_decode_byte_exact']=True;observations.append(info)
        write_json(folder/'progress.json',{'completed':len(observations),'latest':info})
        print(json.dumps({'decoded':len(observations),'dataset':row['dataset'],'unique_id':row['unique_id'],'joint_bytes':info['full_joint_bytes']}),flush=True)
    db.close();labels.close()
    require('torch' not in sys.modules and 'transformers' not in sys.modules,'No model/CUDA runtime imports')
    write_json(folder/'report.json',{'passed':True,'actual_decoded_rows':len(rows),'observations':observations,'source':record(__file__),
        'reader':record(PROJECT/'outputs/parakeet_turkish/bpe_broad_audio_v1.py'),'source_plan':reader.plan,
        'heldout_PCM_check_performed':False,'TRAIN_admitted':False,'quality_language_filters':[],
        'sample_selection_is_diagnostic_only':True,'final_population_mean_U':None,'pass_denominator':None})

if __name__=='__main__':main()
