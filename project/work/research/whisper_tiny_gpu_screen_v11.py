"""Final bounded screen: fast GPU tiny with canonical FP32 borderline rescoring."""
import pathlib,json,time,hashlib
import whisper_tiny_gpu_production_v10 as runner
from whisper_tiny_gpu_pipeline_v5 import TinyEngine

class ThresholdCheckedEngine(TinyEngine):
    def __init__(self,guard,batch=8):
        super().__init__(guard,batch)
        from transformers import WhisperForConditionalGeneration
        t=self.torch;folder=json.loads((runner.AREA/'HF_download_receipt.json').read_text())['model_directory']
        self.reference=WhisperForConditionalGeneration.from_pretrained(folder,local_files_only=True,dtype=t.float32,attn_implementation='sdpa',low_cpu_mem_usage=True).eval().to('cuda')
        self.rescored=0;self.total_windows=0;self.category_changes=0;self.last_annotations=[];self.rescore_s=0.
        helpers={}
        for name in ['whisper_tiny_gpu_screen_v11.py','whisper_tiny_gpu_production_v10.py','whisper_tiny_gpu_pipeline_v5.py','whisper_tiny_gpu_stream_v7.py','whisper_tiny_gpu_fused_v3.py','whisper_tiny_gpu_batch_v1.py','cpu_preparation_guard_v2.py']:
            p=pathlib.Path(__file__).parent/name;helpers[name]=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
        runner.common.write('final_screening_policy_v11.json',dict(at_utc=runner.common.now(),human_minimum_Turkish_speech_share=.9,window_seconds=4,confidence_for_TR_or_other=.7,uncertainty='Duration of windows below confidence70,or less than250ms speech;not proven non-Turkish',pass_rule='estimated confident Turkish duration / all VAD speech duration >= .90',below_rule='(confident Turkish+uncertain) / speech < .90',remaining='uncertain_review; source decode quarantine; no detected speech',near_threshold_rescore=dict(optimized_top_probability_interval=[.6,.8],reference='Canonical HF full-vocabulary one-SOT-step FP32 + canonical CPU mel,full3000frames',basis='Empirical FP16 audit; retains preliminary vectors;not guarantee of model accuracy'),originals_preserved=True,TRAIN_admission=False,helper_SHA256=helpers,raw_probability_language_order=self.languages))
    def mel(self,pcm):
        # Canonical zero-audio log floor is max(-10,example_log_max-8).
        # Preserve the full 3000-frame encoder while avoiding zero-suffix FFTs.
        t=self.torch;spec=t.stft(pcm,400,160,window=self.hann,return_complex=True);power=(spec[...,:-1].abs()**2).contiguous();log=(self.filters.T@power).clamp_min(1e-10).log10();floor=(log.amax(dim=(-2,-1),keepdim=True)-8.).clamp_min(-10.);log=t.maximum(log,floor);log=(log+4.)/4.
        return t.cat((log,((floor+4.)/4.).expand(len(pcm),80,3000-log.shape[-1])),dim=-1).half()
    def predict(self,waves):
        scores=super().predict(waves);self.total_windows+=len(scores);self.last_annotations=[{'score_path':'optimized_full_tiny_FP16'} for _ in scores];selected=[i for i,p in enumerate(scores) if .6<=max(p)<=.8]
        if not selected:return scores
        t=self.torch;wrapper=t.nn.functional.scaled_dot_product_attention;t.nn.functional.scaled_dot_product_attention=self.original_sdpa;started=time.perf_counter()
        def category(p):
            k=max(range(len(p)),key=p.__getitem__)
            return ('tr' if k==self.tr else 'other') if p[k]>=.7 else 'uncertain'
        try:
            with t.inference_mode():
                for at in range(0,len(selected),4):
                    self.guard.check();indices=selected[at:at+4];features=self.extractor([waves[i] for i in indices],sampling_rate=16000,return_tensors='pt').input_features.to('cuda');sot=t.full((len(indices),1),self.reference.generation_config.decoder_start_token_id,device='cuda',dtype=t.long)
                    logits=self.reference(input_features=features,decoder_input_ids=sot,use_cache=False).logits[:,0].index_select(1,self.ids);reference=logits.softmax(-1).cpu().tolist()
                    for i,p in zip(indices,reference):
                        before=scores[i];self.last_annotations[i]={'score_path':'canonical_HF_FP32_borderline_rescore','optimized_FP16_language_probabilities':before,'confidence70_category_changed':category(before)!=category(p)};self.category_changes+=int(category(before)!=category(p));scores[i]=p
            self.rescored+=len(selected);self.rescore_s+=time.perf_counter()-started;return scores
        finally:t.nn.functional.scaled_dot_product_attention=wrapper
    def close(self):
        runner.common.write('final_FP32_rescore_counts_'+str(__import__('os').getpid())+'.json',dict(total_windows=self.total_windows,canonical_FP32_rescored=self.rescored,confidence70_category_changes=self.category_changes,rescore_wall_s=self.rescore_s))
        del self.reference
        super().close()

original_analyze=runner.analyze
def annotated_analyze(engine,vad,rows,decoded):
    out,m=original_analyze(engine,vad,rows,decoded);annotations=iter(engine.last_annotations)
    for row in out:
        for window in row['windows']:window.update(next(annotations))
    assert next(annotations,None) is None
    return out,m

if __name__=='__main__':
    runner.TinyEngine=ThresholdCheckedEngine;runner.analyze=annotated_analyze;runner.__file__=__file__
    try:runner.main()
    except BaseException as e:
        runner.common.write('screen_v11_failure_'+str(__import__('os').getpid())+'.json',dict(at_utc=runner.common.now(),error=repr(e)))
        raise
