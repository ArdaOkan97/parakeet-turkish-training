"""Publish the completed independent source read; no numerical imports."""
import importlib,json,sys
from pathlib import Path
L=Path(r'E:\gpu_training_scratchpad\analysis\Turkish_labels_BPE_circumflex_v1')
sys.path.insert(0,str(L));m=importlib.import_module('audit_and_fit_append_only_v5')
N=Path(r'E:\gpu_training_scratchpad\analysis\BPE9223_training_launch_v1')
O=N/'independent_DEV9223_source_review_v1';m.AREA=O
P=m.PROJECT/'outputs/parakeet_turkish'

def main():
 g=m.Guard();g.phase='publish_compact_independent_DEV9223_source_findings'
 inventory=json.loads((O/'exact_AST_delta_inventory.json').read_text())
 assert inventory['exact_canonical_joint_math_width_substitution_only'] is True
 checks=json.loads((N/'ROOT_9223_DEV_checkpoint_format_controls_v1.json').read_text());assert checks['passed'] is True and checks['controls']==4
 expected={'tdt_joint_canonical_2d_fp32_9223_v1.py':'cc3f7aa7e48f39adefa9a2f9843283fa2236f61040100d83b6777884d2056fdc',
  'asr18_optimizer_free_DEV_structure_9223_v1.py':'8168c6cea4e6f0efd6eb9fde577a60cdedb4b4922202980652925abf48af0449',
  'asr18_full_run_serial_DEV_9223_v1.py':'d8884947ebfd2b7ffb8759973d8efc7d40a12c96d84eca639dd6444fcf2768e6',
  'asr18_full_run_DEV_runtime_9223_v1.py':'6e21404fa9e40e1f01d17f8cc727cf6f1b3a78890fd413d1abf1e9b8d0b57d95'}
 for name,sha in expected.items():assert m.sha(P/name)==sha
 findings={'kind':'Independent_BPE9223_DEV_quartet_exact_source_delta_findings_v1','at_utc':m.now(),
  'reviewer':'/root/label_bpe_audit','accepted_source_delta_scope':True,'execution_resource_or_production_admission':False,
  'reviewed_new_sources':{name:m.record(P/name) for name in expected},
  'old_frozen_source_and_AST_delta_inventory':m.record(O/'exact_AST_delta_inventory.json'),
  'actual_readonly_metadata_AST_control_exit0':{'tool_chunk':'8e83b7','exit_code':0},
  'ROOT_format_only_controls_reviewed_not_repeated':m.record(N/'ROOT_9223_DEV_checkpoint_format_controls_v1.json'),
  'observations':[
   'Canonical joint source is EXACT complete text width substitution8711→9223/8716→9228 versus frozen2; FP32/reduction/block/order/TF32/locking statements are unchanged. New packed128x9228x4 block4724736B. This is source derivation, not equal old-vocabulary numerics or native fit.',
   'Structure independently pins final tokenizer bytes SHA f2d80f.../9223vocab/14500merges; original8193 tokenizer IDs and blank8192 stay unchanged; embedding9223 and joint9228 are allocated empty then gated.',
   'Original8193:8198 generation duration suppression shifts+1030 to9223:9228. Newly lexical8193:9223 cannot be suppressed. No training-ID permutation or blank-at-end change.',
   'Expected207 layout remains original15 decoder/projector/joint plus192 LoRA; all207 shapes/dtypes checked before transfer and GPU byte hashes matched before _ASR18_DEV_weights_restored=True. assert_ready rejects gradients or a changed readonly294 backing before evaluation.',
   'No Adam/sampler/replay constructor or checkpoint Adam tensor view is introduced in the four-source delta. Selected weight/all4RNG mapper route remains unchanged; source checkpoint optimizer scalars are validated without constructing optimizer state.',
   'Checkpoint format requires new optimizer KIND, integer updates, exact old3 clocks64250/2000/62250+updates, and appended512 clock typed int equal updates; bool/offbyone/oldKIND negative controls are documented. checked_checkpoint invokes this before selective restore.',
   'Fixed7x50 manifest/source membership route, immutable scorer/gate hashes, references, six integer caps, saved-prediction independent rescore, strict winner selection and native/recovery pass-boundary checks are unchanged AST/source outside exact published patches.',
   'Verified completion still requires actual source/parent/guard/PID/ctime-owned process0, all207 afterDEV hashes unchanged, all4RNG exact, zero native updates/Adam/sampler/augmentation movement, separate runtime/model-release report and four joined clean prospective phase records. Report-file existence alone is insufficient.'
  ],
  'immutable_policy':{'fixed_DEV_clips':350,'clips_per_language':50,'languages':['tr','en','es','it','fr','de','pt'],
   'reference_words':{'tr':825,'en':1045,'es':1289,'it':1158,'fr':1072,'de':1004,'pt':1095},
   'retention_caps':{'en':69,'es':52,'it':30,'fr':74,'de':70,'pt':66},
   'scorer_SHA':'433c31d9cbc4975c42d3a0fe659cd0bbd6c1caf2e1eb3d85169baafec7f61047',
   'gate_SHA':'cf0f9c66fe5f1046b3ae151026bb19be3b286233a83391055e89dc450fb3c6af'},
  'blocking_source_gaps_found_in_reviewed_delta':[],
  'remaining_actual_proofs_not_source_findings':[
   'Final exact transitive source/config/tokenizer/base asset binding and ROOT execution/resource reviews.',
   'Fresh whole-worker/phase admission and prospective actual coverage for9223 restore+350DEV; inherited5450000000B proposal/8GB CUDA is not new measured shape fit.',
   'Actual new9223 all207 saved-weight/all4RNG restore, complete350 predictions/scorer/caps/owned0; existing4 controls prove only metadata format.'
  ],'GPU_model_audio_tensor_or_optimizer_job_executed':False,'ROOT_or_active_source_files_modified':False}
 m.write('source_findings_v1.json',findings);g.publish(passed=True)
 print(json.dumps({'findings':m.record(O/'source_findings_v1.json'),'source_scope_accepted':True,'source_blockers':[]}))
if __name__=='__main__':main()
