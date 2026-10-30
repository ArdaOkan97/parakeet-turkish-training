"""Source/API review of disabled production assembler; pure session boundary."""
import ast,copy,difflib,importlib,json,sys
from pathlib import Path
L=Path(r'E:\gpu_training_scratchpad\analysis\Turkish_labels_BPE_circumflex_v1');sys.path.insert(0,str(L));m=importlib.import_module('audit_and_fit_append_only_v5')
N=Path(r'E:\gpu_training_scratchpad\analysis\BPE9223_training_launch_v1');O=N/'independent_full_configuration_assembler_review_v1';O.mkdir(exist_ok=True);m.AREA=O
S=m.PROJECT/'outputs/parakeet_turkish';R=m.PROJECT/'work/research';sys.path.insert(0,str(S))
def rec(path):return {k:v for k,v in m.record(path).items() if k in ('path','bytes','sha256')}
def main():
 g=m.Guard();g.phase='disabled_full_configuration_source_API_review'
 source=R/'prepare_ASR18_BPE9223_full_configuration_v1.py';old=R/'prepare_ASR18_full_owner_configuration_v11.py'
 assert m.sha(source)=='a2319a543058c910e38c292690258b8f85be15f9cdc25ecbfc9d598705d6e28f'
 tree=ast.parse(source.read_text());assemble=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='assemble')
 assignment=next(n for n in assemble.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='session' for x in n.targets))
 fields={k.value for k in assignment.value.keys};assert 'production_source_eligible' not in fields
 code=compile(ast.Expression(body=copy.deepcopy(assignment.value)),str(source),'eval')
 dummy={'path':str(O/'IN_MEMORY_ONLY.json'),'bytes':1,'sha256':'1'*64}
 template={'augmentation_identity':{'frozen_recorded_room_policy':'fixture'},'planning_arguments':{'snapshot_binding':dummy,'DATA_evidence':{'BPE9223_population':dummy}}}
 namespace={'record':lambda p:copy.deepcopy(dummy),'S':S,'digest':lambda x:__import__('hashlib').sha256(json.dumps(x,sort_keys=True).encode()).hexdigest(),
  'identity_sha':'a'*64,'expected_owner':copy.deepcopy(dummy),'template':template,'plan':{'initial_sampler_sha256':'b'*64},'cp':O/'IN_MEMORY_ONLY.contract','own':rec(source)}
 session=eval(code,namespace)
 import fastmss_BPE9223_plan_bridge_v3 as bridge
 try:bridge.validate_session_scope(session)
 except KeyError as error:assert error.args==('production_source_eligible',);missing=str(error)
 else:raise AssertionError('Current assembler session unexpectedly accepted')
 repaired=copy.deepcopy(session);repaired['production_source_eligible']=True
 assert bridge.validate_session_scope(repaired)==bridge.FULL_MODE
 negatives=[]
 for invalid in (False,0,1,'True',None):
  bad=copy.deepcopy(repaired);bad['production_source_eligible']=invalid
  try:bridge.validate_session_scope(bad)
  except (ValueError,KeyError) as error:negatives.append({'observed':invalid,'type':type(invalid).__name__,'rejected':str(error)})
  else:raise AssertionError('Invalid eligibility flag accepted')
 assert session['owner_validation_only'] is False and session['checkpoint_diagnostic_only'] is False and session['diagnostic_only'] is False
 assert session['vocabulary']==9223 and session['planned_native_updates']==401072 and session['initial_optimizer_updates']==0
 fulltext=source.read_text()
 assert "template['full_run_launch_enabled'] is False" in fulltext and "'ROOT_CPU_execution_review':None" in fulltext
 assert "from asr18_BPE9223_lifecycle_v3 import SourceBoundOwnerCallbacks,CALLBACKS" in fulltext
 assert "from asr18_BPE9223_durable_transitions_v3 import OwnerProofCallbacks" in fulltext
 assert "from asr18_BPE9223_persistence_prefix_v4 import OwnerPrefixVerifier" in fulltext
 assert "'source':record(S/'asr18_BPE9223_durable_owner_adapter_v3.py')" in fulltext
 calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
 assert not any(isinstance(n.func,ast.Attribute) and n.func.attr in ('Popen','run','cuda','backward','step') for n in calls)
 delta=''.join(difflib.unified_diff(old.read_text().splitlines(True),source.read_text().splitlines(True),fromfile=old.name,tofile=source.name));(O/'exact_source_delta_v1.diff').write_text(delta)
 assert not any(k in sys.modules for k in ('torch','numpy','transformers','nemo'));g.check()
 report={'kind':'ASR18_independent_disabled_BPE9223_full_configuration_assembler_source_API_review_v1','reviewer':'/root/label_bpe_audit','controls_passed':True,'source_accepted':False,
  'source':rec(source),'baseline':rec(old),'exact_delta':rec(O/'exact_source_delta_v1.diff'),'session_API_source':rec(bridge.__file__),
  'actual_source_session_dict_evaluated_without_writing_configuration':True,'actual_validate_session_scope_raises':missing,
  'real_blocker':'Emitted session omits production_source_eligible required by bridge3.validate_session_scope, which common3.verify_contract invokes immediately. Metadata assembly cannot reach bridge creation.',
  'supported_successor':'NEW frozen assembler should add production_source_eligible=True alongside typed owner_validation_only=False/checkpoint_diagnostic_only=False/diagnostic_only=False; launch remains disabled.',
  'locally_corrected_metadata_copy_passes_exact_current_API':True,'invalid_flag_negatives':negatives,
  'remaining_read_scope':['9223 original source context/new labels/full means and 401072 source0 two broad plus one cleaner preserved','20 private preview requires original cursor0/280 broad/0cleaner and unchanged sampler digest','Lifecycle3/durability3/prefix4 exact callback constructors and predicted—not-created—durable owner','CPU3/common3/bridge3 generic typed static template; per-window bridge only; transient/p9/CPU/payload output schema','Output launch flag False; ROOT_CPU_execution_review None and exact review still required; no namespace/GPU/model execution'],
  'final_configuration_review_requirements':['ROOT manifest should pin the hard-coded reused passed sustained20 CPU numerical/source-contract inputs and all current static-template dependencies','Verify final template/model/head/blank/duration/scorer/resource/authority joins after actual metadata assembly; this source review alone cannot admit production'],
  'actual_audio_model_payload_GPU_jobs':0,'production_admitted_here':False}
 (O/'source_findings_v1.json').write_text(json.dumps(report,indent=2)+'\n');g.publish(completed=True,findings=rec(O/'source_findings_v1.json'))
 print(json.dumps({'source_accepted':False,'metadata_reproduction_passed':True,'findings':rec(O/'source_findings_v1.json')}))
if __name__=='__main__':main()
