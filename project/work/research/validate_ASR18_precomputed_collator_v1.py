"""Actual full long-TR plus six-language genuine-replay collation comparison."""
import argparse,gc,json,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';T=P/'work/training-env/Lib/site-packages'
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')
sys.path.insert(0,str(T))
import numpy as np
sys.path.remove(str(T))
import torch
assert torch.__version__=='2.7.1+cpu' and torch.version.cuda is None
sys.path.extend([str(T),str(S)])
from atomic_io_retry_v1 import write_json
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_v3 import read_prepared
from asr18_precomputed_collator_v1 import collate_precomputed_row
from batched_lora_tr_metadata_v1 import collate_training_batch

def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);a=p.parse_args();folder=Path(a.folder);require(folder.is_relative_to(A),'E-only');folder.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    from transformers import AutoProcessor
    processor=AutoProcessor.from_pretrained(Path((P/'work/base_model_path.txt').read_text().strip()),local_files_only=True)
    TR=json.loads((A/'hard_U_rows_decode_01/decoded/report.json').read_text())['rows']
    replay=json.loads((A/'window_replay_PCM_actual_01/report.json').read_text())['references'];chosen=[]
    for language in ('en','es','it','fr','de','pt'):chosen.append(next(v for v in replay if v['language']==language))
    rows=[(item['row'],item['row']['tr_metadata_labels']) for item in TR]+[(item['source_row'],item['full_target_ids']) for item in chosen]
    observations=[]
    for row,ids in rows:
        old=collate_training_batch(processor,[row],blank_id=8192,old_size=8193,device='cpu',audio_reader=read_prepared)
        prepared={name:value.detach().cpu().numpy() for name,value in old.inputs.items()}
        new=collate_precomputed_row(row,prepared,ids,device='cpu')
        names=('labels','decoder_input_ids','target_lengths','frame_lengths','replay_base_features')
        def bits(a,b):return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
        equality={name:bits(getattr(old,name),getattr(new,name)) for name in names}
        equality.update({name:bits(old.inputs[name],new.inputs[name]) for name in old.inputs})
        equality['replay_indices']=old.replay_indices==new.replay_indices
        require(all(equality.values()),'Original complete TrainingBatch fields differ')
        damaged=list(ids);damaged[-1]=(damaged[-1]+1)%8192
        try:collate_precomputed_row(row,prepared,damaged,device='cpu')
        except ValueError:negative=True
        else:negative=False
        require(negative,'Full target alteration must fail before native work')
        observations.append({'source_id':row['source_id'],'language':row['language'],'full_U':len(ids),'encoder_frames':row['encoder_frames'],
            'byte_equal':equality,'wrong_full_target_rejected':True,'genuine_foreign_teacher_features_retained':row['language']!='tr'})
        del old,new,prepared;gc.collect();write_json(folder/'progress.json',{'observations':observations})
    write_json(folder/'report.json',{'kind':'ASR18_precomputed_full_TrainingBatch_equivalence_v1','passed':True,'Torch':torch.__version__,
        'observations':observations,'source':record(__file__),'derivation':record(A/'precomputed_collator_derivation_v1.json'),
        'full_TR_U574_U484_plus_all6_original_replay_languages':True,'optimizer_updates':0,'actual_native_or_stability_claimed':False})

if __name__=='__main__':main()
