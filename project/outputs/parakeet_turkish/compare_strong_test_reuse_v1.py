"""CPU-only pairing of a newly frozen winner with immutable measured Whisper TEST.

The original Whisper inference and its original DEV-selection freeze remain
unchanged. Reuse never performs inference and never supplies a training score.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
from pathlib import Path
import json

import benchmark_strong_tr_test_v2 as original
from atomic_io_retry_v1 import read_json,write_json
from metrics import paired_bootstrap_delta,score_transcriptions

OUTPUT=Path(__file__).resolve().parent
ROOT=OUTPUT.parents[1]
PRIOR_LOOP='adaptive_20261010_recovery_b8_01'
PINNED_SCRIPT='91012f94ba80bd1c0bbee7f06855d3120d5c982e461184b30968c9fc0a4c9580'
PINNED_RECEIPT='acd9e58ee75aca9c03e46a9722d9e77a9aca6418353251f5562491e2e98f8e1b'
PINNED_STRONG_SUMMARY='9ad9d345657260253d059d762106da67a7dd17a51717643d493af7e2640ec1c4'
PINNED_STRONG_PREDICTIONS='220fa8a8093698bb0a005485c52a657f62a47a544fa9ec0ec736d356dec96c07'
def require(value,message):
 if not value:raise RuntimeError(message)
def sha(path):return original.sha256(Path(path))

def recompute(frozen_path,source_verification):
 frozen_path=original.inside(frozen_path)
 source_verification=original.inside(source_verification)
 # Read the immutable parent-owned exit receipt rather than parse the old
 #495,229-check audit just to confirm its already verified top-level flags.
 receipt_path=OUTPUT/'continuation_prerequisites_v1/predecessor_exit_receipt.json'
 receipt=read_json(receipt_path,{})
 require(sha(receipt_path)==PINNED_RECEIPT and receipt.get('predecessor_audit_passed') is True and receipt.get('predecessor_status')=='complete'
   and receipt.get('root_observed_owned_session_completed') is True and receipt.get('root_observed_exit_code')==0
   and receipt.get('predecessor_loop_id')==PRIOR_LOOP,'Completed audited predecessor receipt required')
 require(source_verification.resolve()==Path(receipt['predecessor_audit_path']).resolve()
   and sha(source_verification)==receipt['predecessor_audit_sha256'],'Old measured strong TEST independent verification changed')
 require(sha(OUTPUT/'benchmark_strong_tr_test_v2.py')==PINNED_SCRIPT,'Frozen strong TEST worker changed')
 frozen,freeze=original.verified_frozen_selection(frozen_path)
 old_freeze_path=OUTPUT/'runs'/PRIOR_LOOP/'frozen_final_selection.json'
 _,old_freeze=original.verified_frozen_selection(old_freeze_path)
 examples,input_audit=original.verified_test_rows(ROOT/'work/data/test_tr.jsonl')
 assets=original.verified_assets(ROOT/'work/models/whisper-large-v3')
 strong_dir=OUTPUT/'results/strong_whisper_large_v3_greedy_test_v2'
 strong_path=strong_dir/'test_tr.jsonl';summary_path=strong_dir/'summary.json'
 require(sha(summary_path)==PINNED_STRONG_SUMMARY and sha(strong_path)==PINNED_STRONG_PREDICTIONS,
   'Previously verified Whisper TEST summary/predictions must remain byte-identical to immutable fork snapshot')
 summary=read_json(summary_path,{})
 require(summary.get('status')=='complete' and summary.get('repository')==original.REPOSITORY
   and summary.get('revision')==original.REVISION and summary.get('language')=='tr'
   and summary.get('split')=='test' and summary.get('protocol')==original.PROTOCOL,'Immutable measured strong TEST model/protocol differs')
 require(summary.get('script_sha256')==PINNED_SCRIPT and summary.get('metrics_code_sha256')==sha(OUTPUT/'metrics.py')
   and summary.get('predictions_sha256')==sha(strong_path) and summary.get('input_audit')==input_audit
   and summary.get('protected_code')==original.protected_code()
   and all(summary.get(k)==v for k,v in assets.items()),'Strong model/source/audio/decoder provenance changed')
 require(summary.get('frozen_selection_before_inference')==old_freeze
   and datetime.fromisoformat(old_freeze['frozen_at_utc'])<=datetime.fromisoformat(summary['inference_started_at_utc']),
   'Historical Whisper inference must remain bound to its actual old frozen DEV winner')
 strong=original.rows(strong_path)
 selected_dir=Path(freeze['selected_results_path']);selected_path=selected_dir/'test_tr.jsonl'
 adapted=original.rows(selected_path)
 processor=original.AutoProcessor.from_pretrained(str(ROOT/'work/models/whisper-large-v3'),local_files_only=True)
 original.validate_predictions(strong,examples,whisper=True,processor=processor)
 original.validate_predictions(adapted,examples)
 before=score_transcriptions(strong,language='tr');before.pop('details')
 after=score_transcriptions(adapted,language='tr');after.pop('details')
 selected_summary=read_json(selected_dir/'summary.json',{})
 require(before==summary['metrics'] and after==selected_summary['test_tr']['asr'],'Actual recomputed TEST scores differ from saved counters')
 bootstrap=paired_bootstrap_delta([r['reference'] for r in examples],[r['hypothesis'] for r in strong],
   [r['hypothesis'] for r in adapted],language='tr',seed=20261010,resamples=2000,confidence=.95,
   groups=[r['sentence_id'] for r in examples])
 require(bootstrap['wer']['baseline']==before['wer'] and bootstrap['wer']['adapted']==after['wer']
   and bootstrap['cer']['baseline']==before['cer'] and bootstrap['cer']['adapted']==after['cer'],'Paired bootstrap point estimates differ')
 return {'schema':1,'kind':'immutable_strong_TEST_reuse_comparison_v1','status':'complete','passed':True,
  'cpu_comparison':True,'gpu_inference_performed':False,'run_id':frozen['run_id'],'selected_step':frozen['selected_step'],
  'selection_split':'FLEURS validation only','heldout_split':'test','language':'tr','utterances':100,
  'frozen_selection_sha256':freeze['sha256'],'frozen_selection':freeze,'baseline_repository':original.REPOSITORY,
  'baseline_revision':original.REVISION,'baseline_protocol':original.PROTOCOL,'baseline_asr':before,
  'adapted_asr':after,'candidate_asr':after,'wer_delta_percentage_points':bootstrap['wer']['delta_percentage_points'],
  'cer_delta_percentage_points':bootstrap['cer']['delta_percentage_points'],'paired_sentence_bootstrap':bootstrap,
  'delta_direction':'Selected Parakeet minus measured Whisper; positive means worse',
  'baseline_reused_without_new_inference':True,'historical_baseline_frozen_selection':old_freeze,
  'historical_inference_started_at_utc':summary['inference_started_at_utc'],
  'provenance':{'source_verification_path':str(source_verification),'source_verification_sha256':sha(source_verification),
   'predecessor_exit_receipt_path':str(receipt_path),'predecessor_exit_receipt_sha256':sha(receipt_path),
   'strong_summary_path':str(summary_path),'strong_summary_sha256':sha(summary_path),
   'strong_predictions_path':str(strong_path),'strong_predictions_sha256':sha(strong_path),
   'selected_summary_path':str(selected_dir/'summary.json'),'selected_summary_sha256':sha(selected_dir/'summary.json'),
   'selected_predictions_path':str(selected_path),'selected_predictions_sha256':sha(selected_path),
   'input_audit':input_audit,**assets,'protected_code':original.protected_code(),
   'test_script_sha256':PINNED_SCRIPT,'reuse_comparison_script_sha256':sha(__file__),
   'metrics_code_sha256':sha(OUTPUT/'metrics.py')},
  'limitations':['Same100 fixed Turkish TEST recordings, with paired sentence-cluster uncertainty; no full-split claim.',
   'The existing Whisper TEST result predates these continuation trials. Its score is read only AFTER the new DEV-only winner is frozen, for reporting; it never changes the9.9394% DEV reference or its+2pp target.',
   'The full36.84s TEST clip retains actual native timestamp long-form inference;99 other clips retain the original short-form protocol.',
   'Historical TEST and reused DEV exposure mean this is descriptive repeated-pilot evaluation, not an untouched new generalization benchmark.']}

def audit_reused_comparison(path,frozen_path,check):
 """Recompute sources/tokens/counters/bootstrap independently of passed flags."""
 path=original.inside(path);saved=read_json(path,{})
 source=saved.get('provenance',{}).get('source_verification_path')
 require(source is not None,'Comparison missing original measured baseline verification provenance')
 actual=recompute(frozen_path,Path(source))
 for key,value in actual.items():check.equal(saved.get(key),value,'Recomputed reused strong TEST comparison '+key)
 check.require(saved.get('gpu_inference_performed') is False,'Reused baseline never claims new GPU inference')
 return {**actual,'comparison_path':str(path),'comparison_sha256':sha(path)}

def main(args):
 target=original.inside(args.comparison_output,OUTPUT)
 require(not target.exists(),'Preserve comparison evidence; choose a new output path')
 report=recompute(args.frozen_selection,args.source_verification)
 report['created_at_utc']=datetime.now(timezone.utc).isoformat()
 write_json(target,report)
 print(json.dumps({'passed':True,'gpu_used':False,'run_id':report['run_id'],'baseline_WER':report['baseline_asr']['wer'],
   'candidate_WER':report['adapted_asr']['wer'],'comparison':str(target)}))
def parser():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--frozen-selection',type=Path,required=True)
 p.add_argument('--source-verification',type=Path,required=True)
 p.add_argument('--comparison-output',type=Path,required=True)
 return p
if __name__=='__main__':main(parser().parse_args())
