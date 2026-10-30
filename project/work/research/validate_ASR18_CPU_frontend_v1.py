"""CPU-only Torch frontend fixture using exact existing Parakeet implementation."""
import argparse,gc,hashlib,json,os,sys,time
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish'
T=P/'work/training-env/Lib/site-packages';A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')
sys.path.insert(0,str(T))
import numpy as np
sys.path.remove(str(T))
import torch
assert torch.__version__=='2.7.1+cpu' and torch.version.cuda is None
assert Path(torch.__file__).resolve().is_relative_to(Path(r'E:\gpu_training_scratchpad\scratch\BPE_broad_CPU_preflight_env_v2'))
sys.path.extend([str(T),str(S)])
from atomic_io_retry_v1 import write_json
from bpe_broad_audio_v1 import record,require
from resource_watchdog_v2 import sample_memory

def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);a=p.parse_args();folder=Path(a.folder)
    require(folder.is_relative_to(A),'E-only fixture');folder.mkdir(exist_ok=False)
    torch.set_num_threads(4) # Match the existing parent frontend execution policy.
    def sample(phase):return sample_memory(os.getpid(),phase)['worker_private_memory']
    before_import=sample('CPU_frontend_before_transformers')
    from transformers import AutoFeatureExtractor
    base=Path((P/'work/base_model_path.txt').read_text().strip())
    extractor=AutoFeatureExtractor.from_pretrained(base,local_files_only=True)
    require(type(extractor).__name__=='ParakeetFeatureExtractor','Original exact frontend class')
    import inspect
    module_source=record(inspect.getfile(type(extractor)))
    ready=sample('CPU_frontend_imported');source=A/'hard_U_rows_decode_01/decoded/report.json';decoded=json.loads(source.read_text());rows=[]
    for i,item in enumerate(decoded['rows']):
        require(record(item['PCM_cache']['path'])==item['PCM_cache'],'Pinned full original waveform changed')
        audio=np.load(item['PCM_cache']['path'],allow_pickle=False)
        before=sample('CPU_frontend_before_extract');began=time.monotonic()
        inputs=dict(extractor([audio],sampling_rate=16000,return_tensors='pt',padding='longest',return_attention_mask=True,subsampling_factor=8))
        mask=inputs['attention_mask'].bool();require(torch.equal(mask,torch.arange(mask.shape[1])[None,:]<mask.sum(-1)[:,None]),'Original right-padded mask')
        inputs['input_features']=inputs['input_features'].masked_fill(~mask.unsqueeze(-1),0)
        require(int((mask.sum()+7)//8)==item['row']['encoder_frames'],'Full input original encoder frame agreement')
        results={}
        for name,value in inputs.items():
            value=value.detach().cpu().contiguous();path=folder/f'{i}_{name}.npy';np.save(path,value.numpy(),allow_pickle=False)
            results[name]={'file':record(path),'shape':list(value.shape),'dtype':str(value.dtype),'tensor_bytes_sha256':hashlib.sha256(value.numpy().tobytes()).hexdigest()}
        after=sample('CPU_frontend_after_extract')
        rows.append({'source_id':item['row']['source_id'],'full_PCM':item['PCM_cache'],'input_audio_f32_SHA256':hashlib.sha256(audio.tobytes()).hexdigest(),
            'encoder_frames':item['row']['encoder_frames'],'inputs':results,'seconds':time.monotonic()-began,'before':before,'after':after})
        del audio,inputs,mask,value;gc.collect();write_json(folder/'progress.json',{'rows':rows})
    require(not any(n.startswith(('nemo.collections.asr.models','scipy.signal')) for n in sys.modules),'No ASR model or augmentation runtime')
    report={'passed':True,'kind':'ASR18_original_Parakeet_CPU_frontend_control_v1','Torch':torch.__version__,'Torch_file':torch.__file__,
        'Torch_CUDA_runtime':torch.version.cuda,'CUDA_masked':os.environ['CUDA_VISIBLE_DEVICES']=='-1','torch_threads':torch.get_num_threads(),
        'before_transformers':before_import,'frontend_ready':ready,'source':record(__file__),'frontend_source':module_source,
        'source_full_PCM_proof':record(source),'rows':rows,'weights_or_ASR_model_loaded':False,'cross_runtime_byte_equivalence_proven':False}
    write_json(folder/'report.json',report)

if __name__=='__main__':main()
