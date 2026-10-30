"""Cheap exact original-parent versus CPU-only frontend control; no ASR model."""
import gc,json,os,time
from pathlib import Path
import numpy as np
import torch
from atomic_io_retry_v1 import write_json
from bpe_broad_audio_v1 import record,require
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')

def validate(folder,*,stage,admit,stop_check):
    from resource_watchdog_v2 import sample_memory
    admit('original_parent_frontend_comparison',1_000_000_000)
    def memory(phase):return sample_memory(os.getpid(),phase)['worker_private_memory']
    before=memory('before_original_AutoProcessor');began=time.monotonic()
    from transformers import AutoProcessor
    base=Path((Path(__file__).resolve().parents[2]/'work/base_model_path.txt').read_text().strip())
    processor=AutoProcessor.from_pretrained(base,local_files_only=True);ready=memory('original_AutoProcessor_ready')
    from asr18_token_only_carrier_v1 import load
    token_only=load(base)
    require(token_only.tokenizer._tokenizer.to_str()==processor.tokenizer._tokenizer.to_str() and token_only.blank_token==processor.blank_token and token_only.blank_token_id==processor.blank_token_id,
        'Tokenizer-only original carrier differs from complete original processor')
    proof=A/'CPU_frontend_actual_02/report.json';CPU=json.loads(proof.read_text());require(CPU['passed'],'Actual CPU frontend fixture required')
    results=[]
    for index,row in enumerate(CPU['rows']):
        stop_check();require(record(row['full_PCM']['path'])==row['full_PCM'],'Original full waveform changed')
        audio=np.load(row['full_PCM']['path'],allow_pickle=False);start=memory('parent_before_feature_extraction')
        inputs=dict(processor(audio=[audio],sampling_rate=16000,return_tensors='pt',padding='longest'))
        mask=inputs['attention_mask'].bool();inputs['input_features']=inputs['input_features'].masked_fill(~mask.unsqueeze(-1),0)
        equal={}
        for name,item in row['inputs'].items():
            require(record(item['file']['path'])==item['file'],'Bound CPU frontend output changed')
            expected=torch.from_numpy(np.load(item['file']['path'],allow_pickle=False)).to('cuda')
            actual=inputs[name].to('cuda')
            equal[name]=actual.shape==expected.shape and actual.dtype==expected.dtype and torch.equal(actual.contiguous().view(torch.uint8),expected.contiguous().view(torch.uint8))
            del expected,actual
        require(all(equal.values()),'Every original parent input-feature/mask byte must equal CPU-only output')
        results.append({'source_id':row['source_id'],'inputs_byte_equal':equal,'passed':True,'before':start,'after':memory('parent_after_feature_extraction')})
        del audio,inputs,mask;gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache();stage('parent_original_frontend_comparison_case',case=index+1)
    report={'kind':'ASR18_CPU_and_original_parent_frontend_byte_equivalence_v1','passed':True,'Torch':torch.__version__,'CPU_proof':record(proof),
        'comparisons':results,'before_AutoProcessor':before,'AutoProcessor_ready':ready,'seconds':time.monotonic()-began,
        'original_token_only_carrier_tokenizer_and_blank_exact':True,'ASR_model_loaded':False,'native_optimizer_updates':0,
        'all207_math_or_sustained_stability_claimed':False,'production_ready':False}
    write_json(folder/'frontend_byte_equivalence.json',report);return report
