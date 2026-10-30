"""Exact transitive binding review plus real-metadata-copy selection controls.

No payload/Torch/model/GPU/audio loading. Source metadata is never modified.
"""
import copy,difflib,importlib,json,sys
from pathlib import Path
L=Path(r'E:\gpu_training_scratchpad\analysis\Turkish_labels_BPE_circumflex_v1')
sys.path.insert(0,str(L));m=importlib.import_module('audit_and_fit_append_only_v5')
N=Path(r'E:\gpu_training_scratchpad\analysis\BPE9223_training_launch_v1')
O=N/'independent_DEV9223_mapper_review_v1';O.mkdir(exist_ok=True);m.AREA=O
P=m.PROJECT/'outputs/parakeet_turkish';sys.path.insert(0,str(P))
PATH=m.ROOT/'analysis/ASR18_memory_diagnosis_v1/recorded_room_CPU_native_E2E_actual_01/train/after4.state.json'
KIND='BPE9223_broad_full207_three_AdamW_mixed_fresh_clock_v1'
PAIRS=[('asr18_DEV_weight_only_mapper_v1.py','asr18_DEV_weight_only_mapper_9223_v1.py'),
 ('asr18_full_run_serial_DEV_9223_v1.py','asr18_full_run_serial_DEV_9223_v2.py'),
 ('asr18_full_run_DEV_runtime_9223_v1.py','asr18_full_run_DEV_runtime_9223_v2.py')]

def main():
 g=m.Guard();g.phase='tiny_mapper_import_delta_and_metadata_selection_plan_controls'
 sourceprep=json.loads((N/'ROOT_9223_DEV_mapper_binding_source_preparation_v1.json').read_text())
 deltas=[]
 for (oldname,newname),pin in zip(PAIRS,sourceprep['sources']):
  assert m.sha(P/newname)==pin['sha256']
  actual=''.join(difflib.unified_diff((P/oldname).read_text().splitlines(True),(P/newname).read_text().splitlines(True),fromfile=oldname,tofile=newname))
  assert actual==Path(pin['delta']).read_text()
  deltas.append({'old':m.record(P/oldname),'new':m.record(P/newname),'actual_exact_published_delta':m.record(pin['delta'])})
 assert m.sha(PATH)=='dd1be14dc7d78c04d3406155685d72af23c1a5a0b886c9d857b5cea2555eb5e4'
 original=m.record(PATH);document=json.loads(PATH.read_text())
 from bpe_streaming_checkpoint_semantics_v2 import decode_scalar_tree,select
 from asr18_DEV_weight_only_mapper_9223_v1 import selection_plan
 from asr18_DEV_weight_only_mapper_v1 import selection_plan as previous_plan
 assert 'torch' not in sys.modules and 'numpy' not in sys.modules
 def scalar(doc,*path):return decode_scalar_tree(select(doc['tree'],*path))
 def change(doc,path,value):
  node=select(doc['tree'],*path);assert node[0]=='scalar';node[1]=value
 positive=copy.deepcopy(document);updates=scalar(positive,'optimizer','updates');assert type(updates) is int and updates==4
 change(positive,('optimizer','kind'),KIND)
 opt=select(positive['tree'],'optimizer');assert opt[0]=='dict'
 assert not any(decode_scalar_tree(k)=='appended512_Adam_clock' for k,v in opt[1])
 opt[1].append([['scalar','appended512_Adam_clock'],['scalar',updates]])
 plan=selection_plan(positive)
 assert plan['selected_weight_count']==207 and plan['selected_optimizer_tensor_indices']==[]
 assert plan['actual_tensors_loaded'] is False and plan['full_run_checkpoint_admitted'] is False
 assert set(('python_rng','numpy_rng','torch_rng','cuda_rng')).issubset(plan['value'])
 assert len(plan['value']['cuda_rng'])>=1 and plan['selected_tensor_count']==207+1+len(plan['value']['cuda_rng'])
 assert plan['excluded_optimizer_tensor_count']>0
 assert plan['value']['python_rng']==scalar(positive,'python_rng') and plan['value']['numpy_rng']==scalar(positive,'numpy_rng')
 cases=[]
 def rejected(name,func,value):
  try:func(value)
  except (ValueError,KeyError) as error:cases.append({'name':name,'rejected':True,'reason':str(error)})
  else:raise AssertionError('Negative selection control accepted: '+name)
 rejected('old_mapper_rejects_new9223_kind_reproduces_ROOT_transitive_gap',previous_plan,positive)
 oldkind=copy.deepcopy(positive);change(oldkind,('optimizer','kind'),scalar(document,'optimizer','kind'))
 rejected('new_mapper_rejects_old8711_KIND',selection_plan,oldkind)
 boolean=copy.deepcopy(positive);change(boolean,('optimizer','appended512_Adam_clock'),True)
 rejected('new_mapper_rejects_boolean_clock_on_actual_update4_copy',selection_plan,boolean)
 # Isolate the bool type test from numeric inequality (True == 1): a scalar-only
 # update1/intclock copy passes format, then the equal-valued bool is rejected.
 one=copy.deepcopy(positive);change(one,('optimizer','updates'),1);change(one,('optimizer','appended512_Adam_clock'),1)
 clocks=select(one['tree'],'optimizer','clocks');assert clocks[0]=='list'
 for node,value in zip(clocks[1],[64251,2001,62251]):assert node[0]=='scalar';node[1]=value
 assert selection_plan(one)['selected_weight_count']==207
 change(one,('optimizer','appended512_Adam_clock'),True)
 rejected('equal_numeric_bool_clock1_rejected_by_type_not_only_value',selection_plan,one)
 assert m.record(PATH)==original and 'torch' not in sys.modules and 'numpy' not in sys.modules
 result={'kind':'Actual_DEV9223_mapper_metadata_copy_selection_controls_v1','at_utc':m.now(),'passed':True,
  'actual_source_metadata_unchanged':original,'source_metadata_tensor_entries':len(document['tensors']),
  'positive_copy_changes':'NewKIND plus appended512_Adam_clock int4 only; sourcefile untouched.',
  'selected_named_weights':207,'selected_tensor_descriptors':plan['selected_tensor_count'],
  'all4_RNG_metadata_or_tensor_descriptors_selected':True,'CUDA_RNG_descriptors':len(plan['value']['cuda_rng']),
  'excluded_optimizer_tensor_descriptors':plan['excluded_optimizer_tensor_count'],'selected_Adam_indices':[],
  'negative_cases':cases,'additional_update1_integer_positive_format_control':True,
  'actual_tensor_payload_read_or_Torch_NumPy_GPU_audio_used':False,
  'old845_tensor_geometry_is_not_a_new854_tensor_or9223_model_restore_proof':True}
 m.write('actual_metadata_copy_selection_controls_v1.json',result)
 findings={'kind':'Independent_DEV9223_transitive_mapper_binding_source_findings_successor_v2','at_utc':m.now(),
  'reviewer':'/root/label_bpe_audit','accepted_source_and_metadata_format_scope':True,
  'supersedes_limited_quartet_findings_for_active_DEV_mapper_and_runtime_bindings':m.record(N/'independent_DEV9223_source_review_v1/source_findings_v1.json'),
  'ROOT_transitive_gap_acknowledged':'Old WeightOnlyCheckpoint.selection_plan imported the old8711 format checker and rejected the9223 KIND. The earlier four-source delta review did not include this unchanged transitive mapper implementation.',
  'reviewed_exact_deltas':deltas,'metadata_controls':m.record(O/'actual_metadata_copy_selection_controls_v1.json'),
  'observations':[
   'Only mapper selection_plan checker import changes: newserial9223_v2 checker is now reached by actual selective mapper, no oldKIND alias or weaker format schema.',
   'Serial9223_v2 imports and declares newmapper; runtime9223_v2 imports/requires/binds serialv2/newmapper. Published patches exactly equal full source diffs; no tensor mapping/selection implementation is altered.',
   'Actual frozen after4 metadata copy with new KIND/intclock4 selects207 named weight descriptors plus CPU/CUDA RNG descriptors and Python/NumPy scalar RNG state, excluding all636 optimizer tensor descriptors.',
   'Oldmapper failure is reproduced; oldKIND, booleanclock and equal-numeric True==1 clock cases are rejected by newmapper.',
   'Finaltokenizer,9223embedding/9228head/blank8192/+1030 suppression and immutable350/scorer/caps/owned0/restore-route requirements remain as previous reviewed sources; no Adam/sampler constructor introduced.'
  ],'blocking_source_gaps_found_in_reviewed_binding_delta':[],
  'required_active_successors':['asr18_DEV_weight_only_mapper_9223_v1.py','asr18_full_run_serial_DEV_9223_v2.py','asr18_full_run_DEV_runtime_9223_v2.py'],
  'execution_resource_native_restore_or_DEV_admission':False,
  'remaining_actual_proof':'New9223/854tensor checkpoint selected all207+all4RNG actual restore, fresh resource coverage, full350 saved predictions/scorer/gates and actualowned0 remain native-owner tests.',
  'ROOT_source_or_actual_checkpoint_metadata_files_modified':False}
 m.write('source_findings_successor_v2.json',findings);g.phase='complete_mapper_source_and_metadata_selection_review';g.publish(passed=True)
 print(json.dumps({'findings':m.record(O/'source_findings_successor_v2.json'),'controls':m.record(O/'actual_metadata_copy_selection_controls_v1.json'),'selected':plan['selected_tensor_count'],'excluded_Adam':plan['excluded_optimizer_tensor_count']}),flush=True)
if __name__=='__main__':main()
