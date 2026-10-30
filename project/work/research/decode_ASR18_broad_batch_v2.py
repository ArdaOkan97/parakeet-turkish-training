"""Finite CPU decoder child. Bounded transient update cache, never a PCM pool."""
import argparse,array,hashlib,json,os
from pathlib import Path
import sqlite3,sys
PROJECT=Path(__file__).resolve().parents[2]
ROOT=Path(r'E:\gpu_training_scratchpad')
sys.path.insert(0,str(PROJECT/'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import write_json
from bpe_broad_audio_v1 import BroadAudioReader,BoundViolation,record,require

def main():
    import numpy as np
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    folder=Path(a.output).resolve();require(folder.is_relative_to(ROOT.resolve()),'E-only bounded update cache')
    folder.mkdir(parents=True,exist_ok=False);request=json.loads(Path(a.request).read_text())
    require(1<=len(request['rows'])<=14,'At most one update of TR audio')
    heldouts=None
    if request.get('heldout_PCM_binding'):
        binding=request['heldout_PCM_binding'];require(record(binding['path'])==binding,'Heldout binding changed')
        heldouts=set(json.loads(Path(binding['path']).read_text())['PCM_i16_sha256'])
    db=sqlite3.connect((ROOT/'indexes/BPE_broad_weak_labels_v1.sqlite').as_uri()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    labels=sqlite3.connect((ROOT/'indexes/BPE_broad_labels_8711_v1.sqlite').as_uri()+'?mode=ro',uri=True)
    if request.get('source_inventory_binding'):
        from bpe_broad_audio_v2 import BroadAudioReader as FinalReader
        reader=FinalReader(source_inventory_binding=request['source_inventory_binding'],heldout_PCM_hashes=heldouts)
    else:
        require(request.get('diagnostic_only') is True,'Production decoder requires finalized source signature inventory')
        reader=BroadAudioReader(heldout_PCM_hashes=heldouts)
    outputs=[];cache_bytes=0
    try:
        for draw in request['rows']:
            raw=dict(db.execute('SELECT * FROM clips WHERE ordinal=?',(draw['ordinal'],)).fetchone())
            target=labels.execute('SELECT target_ids_u32,target_tokens,text_sha256 FROM labels WHERE ordinal=?',(draw['ordinal'],)).fetchone()
            require(hashlib.sha256(raw['text'].encode()).hexdigest()==target[2],'Full raw supplied target differs')
            ids=array.array('I');ids.frombytes(target[0]);require(len(ids)==target[1] and all(0<=v<8711 and v!=8192 for v in ids),'Full exact8711 labels')
            audio=reader(raw);cache_bytes+=audio.nbytes
            if cache_bytes>128*1024**2:raise BoundViolation('transient_update_PCM_bytes',raw['unique_id'],cache_bytes,128*1024**2)
            path=folder/(str(draw['draw_index'])+'.npy');np.save(path,audio,allow_pickle=False)
            row={**raw,'broad_original':raw,'recording_source_id':raw['source_id'],'source_id':raw['unique_id'],
                'language':'tr','augmentation_role':'student_tr','duration':len(audio)/16000,'encoder_frames':reader.last['encoder_frames'],
                'target_tokens':len(ids),'tr_metadata_labels':list(ids),'training_input_type':'TR_frame_target_metadata_v1',
                'audio_filepath':str(path),'durable_draw':draw}
            outputs.append({'row':row,'PCM_cache':record(path),'reader_evidence':reader.last});del audio
            write_json(folder/'progress.json',{'completed_rows':len(outputs),'cache_bytes':cache_bytes})
        from bpe_broad_production_v3 import augmentation_binding
        from fastmss_ordered_draw_binding_v1 import AugmentationDrawContext
        from fastmss_cpu_prepared_transport_v1 import prepare_TR_transport
        contexts=tuple(AugmentationDrawContext(**v) for v in request['contexts'])
        rows=[v['row'] for v in outputs]
        require(len(contexts)==len(rows) and all(c.source_id==r['source_id'] and c.language==r['language'] and d['ordinal']==r['broad_original']['ordinal']
            and c.epoch==d['epoch'] and c.draw_index==d['draw_index'] and c.phase=='weak_label_continued_training' and c.split=='train'
            for c,r,d in zip(contexts,rows,request['rows'])),'Independent durable source/ordinal/epoch/draw/phase identity differs')
        binding=augmentation_binding({'augmentation_policy':request['augmentation_policy']},lambda:None)
        def clean_read(row):
            item=outputs[rows.index(row)]
            require(record(item['PCM_cache']['path'])==item['PCM_cache'],'Independent clean decoder cache changed before transform')
            return np.load(row['audio_filepath'],allow_pickle=False)
        prepared=prepare_TR_transport(binding,rows,contexts,batch_id=request['batch_id'],job_id=request['job_id'],clean_reader=clean_read,guard=lambda:None)
        write_json(folder/'report.json',{'passed':True,'rows':outputs,'cache_bytes':cache_bytes,'reader':record(sys.modules[type(reader).__module__].__file__),
            'canonical_decoder':record(PROJECT/'outputs/parakeet_turkish/bpe_broad_audio_v1.py'),'source_inventory_binding':request.get('source_inventory_binding'),
            'transient_one_update_cache':True,'permanent_full_PCM_pool':False,'heldout_PCM_check_performed':heldouts is not None,
            'augmentation_transport':prepared,'diagnostic_only':request['diagnostic_only'],'sampler_or_journal_advanced':False})
    except BoundViolation as error:
        write_json(folder/'unresolved_bound.json',error.receipt);raise
    finally:db.close();labels.close()
    require('torch' not in sys.modules,'No CPU CUDA/model runtime import')

if __name__=='__main__':main()
