"""Freeze byte-identical training functions without importing unused CLI stacks."""
import ast,hashlib,json,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')
sys.path.insert(0,str(S))
from atomic_io_retry_v1 import write_json
from bpe_broad_audio_v1 import record,require

def extracted(path,name):
    text=path.read_text();nodes=[n for n in ast.parse(text).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==name]
    require(len(nodes)==1,'Exact unique source function required: '+name)
    return ast.get_source_segment(text,nodes[0])

def create(path,text):
    ast.parse(text)
    with path.open('x',encoding='utf-8',newline='\n') as f:f.write(text)

def main():
    selections=[('train_lora.py','checkpoint_encoder_blocks'),('train_lora_batched_tsc_bpe_full_decoder_v1.py','expand_candidate'),
        ('train_lora_batched_data_v11.py','ids_sha'),('train_lora_batched_data_v11.py','BucketCycler')]
    bodies=[extracted(S/path,name) for path,name in selections]
    header='''"""Exact retained initialization/sampler bodies; no unused CLI imports."""
import hashlib,json,math
import torch
from torch.utils.checkpoint import checkpoint
from extend_vocab import _make_tokenizer
from bpe_full_decoder_optimizer_v1 import TABLES,require

'''
    target=S/'asr18_narrow_training_primitives_v1.py';create(target,header+'\n\n'.join(bodies)+'\n')
    for (old,name),body in zip(selections,bodies):require(extracted(target,name)==body,'Function-body bytes changed: '+name)
    batch=(S/'batched_lora_tr_metadata_v1.py').read_text()
    read_audio=extracted(S/'runtime.py','read_audio')
    replacement='import soundfile as sf\n\n'+read_audio
    require(batch.count('from runtime import read_audio')==1,'Exact original runtime import')
    narrow=batch.replace('from runtime import read_audio',replacement)
    for name in ('preservation_row_scales','pt_focused_batch_objective'):
        narrow+='\n\n'+extracted(S/'train_lora_batched_tsc_expanded_TRAIN_v1.py',name)+'\n'
    objective=S/'asr18_narrow_objective_v1.py';create(objective,narrow)
    original_part=narrow[:narrow.index('\n\ndef preservation_row_scales')]
    require(original_part.replace(replacement,'from runtime import read_audio')==batch,'Every original masked objective/collate byte unchanged')
    for name in ('preservation_row_scales','pt_focused_batch_objective'):
        require(extracted(objective,name)==extracted(S/'train_lora_batched_tsc_expanded_TRAIN_v1.py',name),'Preservation arithmetic changed')
    old_bootstrap=(S/'bpe_broad_streamed_bootstrap_v1.py').read_text()
    bootstrap=old_bootstrap.replace('from train_lora import checkpoint_encoder_blocks','from asr18_narrow_training_primitives_v1 import checkpoint_encoder_blocks')
    bootstrap=bootstrap.replace('from train_lora_batched_tsc_bpe_full_decoder_v1 import expand_candidate','from asr18_narrow_training_primitives_v1 import expand_candidate')
    create(S/'bpe_broad_streamed_bootstrap_v2.py',bootstrap)
    inverse=bootstrap.replace('from asr18_narrow_training_primitives_v1 import checkpoint_encoder_blocks','from train_lora import checkpoint_encoder_blocks').replace('from asr18_narrow_training_primitives_v1 import expand_candidate','from train_lora_batched_tsc_bpe_full_decoder_v1 import expand_candidate')
    require(inverse==old_bootstrap,'All bootstrap/model/history/RNG arithmetic remains byte unchanged')
    audit={'kind':'ASR18_narrow_training_function_derivation_v1','passed':True,'unadmitted':True,'production_approval_claimed':False,
        'source_bindings':[record(S/x) for x in sorted({p for p,n in selections}|{'runtime.py','batched_lora_tr_metadata_v1.py','train_lora_batched_tsc_expanded_TRAIN_v1.py','bpe_broad_streamed_bootstrap_v1.py'})],
        'new_bindings':[record(S/x) for x in ('asr18_narrow_training_primitives_v1.py','asr18_narrow_objective_v1.py','bpe_broad_streamed_bootstrap_v2.py')],
        'source_function_bodies_byte_exact':[name for path,name in selections]+['read_audio','preservation_row_scales','pt_focused_batch_objective'],
        'entire_masked_collate_and_objective_inverse_reconstructed':True,'entire_bootstrap_inverse_reconstructed':True,
        'memory_improvement_measured':False,'actual_native_equivalence_or_stability_proven':False,'installed_or_admitted_sources_modified':False}
    write_json(A/'narrow_training_primitives_derivation_v1.json',audit);print(json.dumps(record(A/'narrow_training_primitives_derivation_v1.json')))

if __name__=='__main__':main()
