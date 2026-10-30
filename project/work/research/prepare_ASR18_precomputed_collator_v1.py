"""Preserve original labels/replay/masks; replace only raw frontend input block."""
import ast,json,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')
sys.path.insert(0,str(S))
from bpe_broad_audio_v1 import require,record
from atomic_io_retry_v1 import write_json

def main():
    text=(S/'batched_lora_tr_metadata_v1.py').read_text();node=next(x for x in ast.parse(text).body if isinstance(x,ast.FunctionDef) and x.name=='collate_training_batch')
    old=ast.get_source_segment(text,node);body=old[old.index('    from tr_frame_metadata_v1 import'):]
    body=body.replace('    rows=list(rows)','    rows=[row]')
    old_frontend='    audio=[audio_reader(row) for row in rows]\n    inputs=dict(processor(audio=audio,sampling_rate=16000,return_tensors="pt",padding="longest"))'
    require(body.count(old_frontend)==1,'Exact original frontend block required')
    new_frontend='''    if set(prepared_inputs)!={'input_features','attention_mask'}:
        raise ValueError('Only complete original precomputed feature/mask inputs are admitted')
    inputs={name:torch.from_numpy(value) for name,value in prepared_inputs.items()}
    if (inputs['input_features'].dtype!=torch.float32 or inputs['attention_mask'].dtype!=torch.bool
        or inputs['input_features'].ndim!=3 or inputs['input_features'].shape[0]!=1 or inputs['input_features'].shape[-1]!=128
        or inputs['attention_mask'].shape!=inputs['input_features'].shape[:2]
        or not bool(torch.isfinite(inputs['input_features']).all())):
        raise ValueError('Exact serial original feature/mask dtype/geometry required')'''
    body=body.replace(old_frontend,new_frontend)
    target_check='''    if (type(full_target_ids) is not list or not full_target_ids
        or cache_items[0]['labels'].detach().cpu().long().tolist()!=full_target_ids):
        raise ValueError('Every independently bound full precomputed target ID must match actual metadata/cache')
'''
    point='    if len(cache_items)!=len(rows):raise ValueError("Training item count differs")\n'
    require(body.count(point)==1,'Exact full target check boundary');body=body.replace(point,point+target_check)
    header='''"""Verified CPU-produced inputs; original label/replay/mask collation retained."""
import torch
from torch.nn.utils.rnn import pad_sequence
from asr18_narrow_objective_v1 import TrainingBatch
import math

def collate_precomputed_row(row,prepared_inputs,full_target_ids,cache_items=None,*,blank_id=8192,
                            old_size=8193,device='cuda',non_blocking=False):
'''
    result=header+body+'\n';ast.parse(result)
    path=S/'asr18_precomputed_collator_v1.py'
    with path.open('x',encoding='utf-8',newline='\n') as f:f.write(result)
    inverse=body.replace(new_frontend,old_frontend).replace(target_check,'').replace('    rows=[row]','    rows=list(rows)')
    require(inverse==old[old.index('    from tr_frame_metadata_v1 import'):],'Every other original collation-body byte changed unexpectedly')
    write_json(A/'precomputed_collator_derivation_v1.json',{'kind':'ASR18_precomputed_input_collation_derivation_v1','passed':True,'unadmitted':True,
        'source':record(S/'batched_lora_tr_metadata_v1.py'),'new':record(path),'original_labels_genuine_replay_masks_and_TrainingBatch_return_body_inverse_exact':True,
        'changes':['Serial one-row API','Replace audio/processor block by externally verified complete NumPy feature/mask arrays','Require full target IDs to match actual metadata/genuine replay cache','Require exact float32/bool serial geometry'],
        'actual_native_equivalence_or_stability_proven':False,'production_approval_claimed':False})
    print(json.dumps(record(A/'precomputed_collator_derivation_v1.json')))

if __name__=='__main__':main()
