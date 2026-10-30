"""ROOT-only original-v3 and CLOSED6500 Portuguese fixed-DEV evaluation.

--prepare-only uses JSON/source metadata only. Actual GPU decode requires explicit
--gpu-evaluation-authorized. No training/optimizer update, TEST evaluation, old
summary modification, or source checkpoint rewrite is performed.
"""
from __future__ import annotations
import argparse,hashlib,json,math,os,sys
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent
SOURCE_RUN='adaptive_20261011_hf_TSC_headlr02_01';SOURCE_STEP=6500
PHASE='Portuguese_fixed_DEV_retention_extension_v1'

def require(ok,message):
 if not ok:raise ValueError(message)
def record(path):
 path=Path(path).resolve();before=path.stat();h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 require((before.st_size,before.st_mtime_ns)==(path.stat().st_size,path.stat().st_mtime_ns),'Closed file changed while hashing')
 return {'path':str(path),'sha256':h.hexdigest(),'bytes':before.st_size}
def checked(value,*,hash_bytes=True):
 path=Path(value['path']).resolve()
 if hash_bytes:require(record(path)=={k:value[k] for k in ('path','sha256','bytes')},'Pinned closed record differs: '+str(path))
 else:require(path.stat().st_size==value['bytes'],'Closed record size differs')
 return path
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def preflight(args):
 require(not args.output.exists(),'Fresh PT baseline proof required')
 proof=json.loads(args.pt_proof.read_text(encoding='utf-8'))
 require(proof.get('passed') is True and proof.get('kind')=='Portuguese_FLEURS_retention_preparation_v1'
  and proof.get('language')=='pt' and proof.get('dataset_config')=='pt_br'
  and proof.get('revision')=='70bb2e84b976b7e960aa89f1c648e09c59f894dd','Complete pinned PT preparation required')
 files=proof['files'];require(files['replay_train_pt']['utterances']==319 and files['validation_pt']['utterances']==50 and files['test_pt']['utterances']==100,'Fixed319/50/100 PT assets required')
 for value in files.values():checked(value)
 source=json.loads(args.source_receipt.read_text(encoding='utf-8'))
 require(source.get('status')=='actual_CLOSED_qualified_best' and source.get('source_run_id')==SOURCE_RUN
  and source.get('selected_step')==SOURCE_STEP and source.get('retention_gate_passed_at_selected_step') is True,
  'Actual qualifying CLOSED6500 source required')
 exits=source.get('owned_actual_exits',[]);require(exits and all(x.get('exit_code') in (0,1) for x in exits),'Actual observed source closure required')
 if any(x.get('exit_code')==1 for x in exits):
  review=json.loads(checked(source['resource_stop_review']).read_text(encoding='utf-8'))
  require(review.get('passed') is True and review.get('selected_step')==SOURCE_STEP and review.get('child_exit_code')==0
   and review.get('selected_all_five_retention_passed') is True and source.get('source_status_remains_resource_review_required') is True,
   'Truthful reviewed resource closure required, never relabelled normal exit0')
 cfg=json.loads(checked(source['artifacts']['run_config']).read_text(encoding='utf-8'))
 selected=json.loads(checked(source['artifacts']['selection']).read_text(encoding='utf-8'))
 require(cfg['run_id']==SOURCE_RUN and selected['run_id']==SOURCE_RUN and selected['selected_step']==SOURCE_STEP
  and selected.get('accepted') is True and cfg['config_identity']['base_revision']=='541d1f99c6b0c3cd0b11a95167540bb8edefd82b','Source config/selection/base revision differs')
 checkpoint=source['artifacts']['best_checkpoint'];checked(checkpoint,hash_bytes=not args.prepare_only)
 require(Path(files['validation_pt']['path']).name=='validation_pt.jsonl','Expected separate PT validation manifest')
 for tag in (args.base_tag,args.source_tag):
  require(tag and Path(tag).name==tag and not (HERE/'results'/tag).exists(),'Fresh isolated evaluation tags required')
 require(args.base_tag!=args.source_tag,'Distinct source/base PT tags required')
 return {'data':proof,'source':source,'config':cfg,'checkpoint':checkpoint,'manifest':files['validation_pt'],'pt_proof_record':record(args.pt_proof),'source_receipt_record':record(args.source_receipt)}

def main(args):
 context=preflight(args)
 if args.prepare_only:
  print(json.dumps({'passed':True,'kind':'PT_baseline_source_only_preflight_v1','GPU_executed':False,'pt_proof':context['pt_proof_record'],'source_receipt':context['source_receipt_record'],'source_checkpoint_hash_rechecked':False,'tags':[args.base_tag,args.source_tag]},indent=2));return 0
 require(args.gpu_evaluation_authorized,'Actual ROOT GPU evaluation authorization required')
 require('torch' not in sys.modules,'Preload must precede Torch')
 from preload_nemo_linker_v1 import preload
 preloaded=preload();require(preloaded.get('kind')=='coherent_native_cuda12_9_preload_v1' and preloaded.get('cuda_initialized') is False,'Pinned coherent pre-Torch CUDA12.9 required')
 import torch,gc,copy
 from runtime import load_model
 import benchmark
 from extend_vocab import extend_processor_and_model
 from encoder_lora import install_encoder_lora,encoder_lora_metadata
 from train_lora_batched_tsc_headlr_v1 import validate_trainable_checkpoint
 torch.set_num_threads(1);require(torch.cuda.is_available(),'ROOT exclusive actualCUDA slot required')
 total=torch.cuda.get_device_properties(0).total_memory;torch.cuda.set_per_process_memory_fraction(min(1.,8_000_000_000/total))
 prior_DATA=benchmark.DATA;prior_LANGUAGES=benchmark.LANGUAGES
 old_base_path=HERE/'results/base_v3/summary.json';old_base_record=record(old_base_path)
 old_base=json.loads(old_base_path.read_text(encoding='utf-8'));old_DEV={k:v for k,v in old_base.items() if k.startswith('validation_')}
 try:
  benchmark.DATA=Path(context['manifest']['path']).parent;benchmark.LANGUAGES=['pt']
  model,processor=load_model();require(model.config.vocab_size==8193 and model.config.blank_token_id==8192,'Actual originalv3 geometry required')
  baseline=benchmark.evaluate(model,processor,args.base_tag,languages=['pt'],splits=('validation',),limit=50)
  del model,processor;gc.collect();torch.cuda.empty_cache()
  model,processor=load_model();vocabulary=extend_processor_and_model(processor,model,training_texts=None)
  cfg=context['config'];recipe=cfg['config_identity']['training_arguments']
  require(vocabulary['extended_tokenizer_sha256']==cfg['config_identity']['extended_tokenizer_sha256'] and model.config.vocab_size==8199,'Exact source6500 six-character tokenizer required')
  for parameter in model.encoder.parameters():parameter.requires_grad_(False)
  layout=install_encoder_lora(model,rank=recipe['rank'],alpha=recipe['alpha'],last_n_layers=recipe['last_n_layers'])
  model.config.encoder_lora_tracking={'layout':layout,'implementation_sha256':record(HERE/'encoder_lora.py')['sha256'],'kind':'unmerged'}
  require(record(context['checkpoint']['path'])==context['checkpoint'],'Archived6500 checkpoint changed before actual load')
  state=torch.load(context['checkpoint']['path'],map_location='cpu',weights_only=True)
  require(state.get('schema')==2 and state.get('run_id')==SOURCE_RUN and state.get('step')==SOURCE_STEP
   and canonical(state['config_identity'])==canonical(cfg['config_identity']), 'Actual6500 fullstate lineage/config differs')
  validate_trainable_checkpoint(model,state,encoder_lora_metadata(model))
  loaded=model.load_state_dict(state['weights'],strict=False);require(not loaded.unexpected_keys,'Unexpected learned source weights')
  del state;gc.collect();torch.cuda.empty_cache()
  champion=benchmark.evaluate(model,processor,args.source_tag,languages=['pt'],splits=('validation',),limit=50)
  del model,processor;gc.collect();torch.cuda.empty_cache()
 finally:benchmark.DATA=prior_DATA;benchmark.LANGUAGES=prior_LANGUAGES
 require(record(old_base_path)==old_base_record,'Original six-language baseline must remain unchanged')
 for tag,score in ((args.base_tag,baseline),(args.source_tag,champion)):
  require(score['validation_pt']['asr']['utterances']==50,'All50 PT predictions must finish')
  from analyze_TSC_bpe_acquisition_saved_DEV_v1 import Evidence
  from metrics import score_transcriptions
  e=Evidence();rows,_=e.scan(HERE/'results'/tag/'validation_pt.jsonl','pt',8193 if tag==args.base_tag else 8199)
  rescored=score_transcriptions(rows,language='pt');rescored.pop('details');require(rescored==score['validation_pt']['asr'],'Independent PT saved-row rescore differs')
 base=baseline['validation_pt']['asr'];actual=champion['validation_pt']['asr'];limit=max(.005,.1*base['wer']);allowed=math.floor((base['wer']+limit)*base['reference_words']+1e-9)
 overlay={**old_DEV,'validation_pt':baseline['validation_pt']}
 result={'schema':1,'kind':'Portuguese_retention_baselines_v1','passed':True,'actual_GPU_decode':True,'phase':PHASE,
  'pt_preparation':context['pt_proof_record'],'source_receipt':context['source_receipt_record'],'source_checkpoint':context['checkpoint'],
  'base_revision':'541d1f99c6b0c3cd0b11a95167540bb8edefd82b','source_run_id':SOURCE_RUN,'source_step':SOURCE_STEP,
  'fixed_PT_DEV_manifest':context['manifest'],'fixed_PT_DEV_count':50,'baseline_PT':baseline,'champion6500_PT':champion,
  'Portuguese_gate':{'base_wer':base['wer'],'source_wer':actual['wer'],'wer_delta':actual['wer']-base['wer'],'allowed_delta':limit,'allowed_word_errors':allowed,'actual_source_word_errors':actual['word_errors'],'passed':actual['word_errors']<=allowed},
  'baseline_overlay':overlay,'original_base_v3_summary':old_base_record,'original_six_language_baseline_unchanged':True,
  'source_PT_is_new_actual_measure_not_inherited_alias':True,'no_TEST_prediction_or_score_used':True,'no_optimizer_updates':True,
  'preload':preloaded,'created_at_utc':datetime.now(timezone.utc).isoformat(),'script':record(__file__),
  'results':{tag:{'summary':record(HERE/'results'/tag/'summary.json'),'metadata':record(HERE/'results'/tag/'metadata.json'),'predictions':record(HERE/'results'/tag/'validation_pt.jsonl')} for tag in (args.base_tag,args.source_tag)}}
 args.output.parent.mkdir(parents=True,exist_ok=True)
 with args.output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({'proof':record(args.output),'baseline':base,'source6500':actual,'Portuguese_gate':result['Portuguese_gate']},indent=2));return 0

def parser():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare-only',action='store_true');p.add_argument('--gpu-evaluation-authorized',action='store_true')
 p.add_argument('--pt-proof',type=Path,default=HERE/'Portuguese_FLEURS_retention_preparation_v1.json')
 p.add_argument('--source-receipt',type=Path,default=HERE/'TSC_headlr_BPE512_actual_CLOSED_reviewed_main_source_v2.json')
 p.add_argument('--base-tag',default='PT_base_v3_retention_v1');p.add_argument('--source-tag',default='PT_source6500_retention_v1');p.add_argument('--output',type=Path,default=HERE/'Portuguese_retention_baselines_v1.json');return p
if __name__=='__main__':sys.exit(main(parser().parse_args()))
