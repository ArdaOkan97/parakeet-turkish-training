"""Metadata-only source/inverse/config handoff. No candidate imports or control."""
import ast,copy,hashlib,json
from pathlib import Path
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish';R=P/'work/research'
F=Path(__file__).parent;N=F.parents[1]
def pin(p):
 p=Path(p);raw=p.read_bytes();return {'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 with Path(p).open('x',encoding='utf-8')as h:json.dump(v,h,indent=2);h.write('\n')
def text(p):return Path(p).read_bytes().decode().replace('\r\n','\n')
old_report=read(F/'source_inverse_and_configuration_report.json');rules=copy.deepcopy(old_report['changes'])
# Only the two real active imports change CPU generation. Historical CPU9 stays9.
rules['driver22_to23'][1]={'before':'from fastmss_resident_feature_stream_v9 import ',
 'after':'from fastmss_resident_feature_stream_v10 import ','count':2}
for item in rules['wrapper22_to23']:
 item['after']=item['after'].replace('training_v23','training_v24')
pairs={
 'CPU9_to10':(S/'fastmss_resident_feature_stream_v9.py',S/'fastmss_resident_feature_stream_v10.py'),
 'driver22_to23':(R/'run_ASR18_BPE9223_resident_RAM_training_v22.py',R/'run_ASR18_BPE9223_resident_RAM_training_v24.py'),
 'wrapper22_to23':(R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v22.py',R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v24.py'),
 'quiescence2_to3':(S/'asr18_resident_periodic_CPU_quiescence_v2.py',S/'asr18_resident_periodic_CPU_quiescence_v3.py')}
inverse={};unchanged={}
for name,(before,after)in pairs.items():
 original=text(before);candidate=text(after);ast.parse(candidate);reconstructed=candidate
 for item in reversed(rules[name]):
  assert reconstructed.count(item['after'])==item['count'],(name,item)
  reconstructed=reconstructed.replace(item['after'],item['before'])
 assert reconstructed==original,name
 # Reconstruct the actual original newline convention before byte comparison.
 raw=before.read_bytes();encoding='CRLF'if b'\r\n'in raw else'LF'
 encoded=(reconstructed.replace('\n','\r\n')if encoding=='CRLF'else reconstructed).encode()
 assert encoded==raw,name
 inverse[name]={'prior':pin(before),'candidate':pin(after),'normalized_text_inverse_exact':True,
   'original_raw_byte_inverse_exact':True,'original_newline_convention':encoding,'declared_changes':rules[name]}
 prior_ast=ast.parse(original);new_ast=ast.parse(candidate)
 old={x.name:x for x in prior_ast.body if isinstance(x,(ast.FunctionDef,ast.ClassDef))}
 new={x.name:x for x in new_ast.body if isinstance(x,(ast.FunctionDef,ast.ClassDef))}
 unchanged[name]=[k for k in old if k in new and ast.dump(old[k])==ast.dump(new[k])]
driver_ast=ast.parse(text(pairs['driver22_to23'][1]));base_ast=ast.parse(text(pairs['driver22_to23'][0]))
def nested(tree):
 main=next(x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name=='main')
 return {x.name:ast.dump(x)for x in main.body if isinstance(x,ast.FunctionDef)}
nb,nn=nested(base_ast),nested(driver_ast)
unchanged['driver_main_nested_functions']=[k for k in nb if nb[k]==nn.get(k)]
def calls(tree,names):
 return [ast.dump(x)for x in ast.walk(tree)if isinstance(x,ast.Call)and isinstance(x.func,ast.Name)and x.func.id in names]
assert calls(base_ast,{'native_update','commit_RAM_transaction'})==calls(driver_ast,{'native_update','commit_RAM_transaction'})
base_pin=old_report['source_base'];base=read(base_pin['path']);config=read(F/'configuration_template_NOT_ADMITTED.json')
reverted=copy.deepcopy(config)
del reverted['resident_RAM_packet_credit_policy'];del reverted['prior_resident_RAM_native_pressure156537_failure']
reverted['resident_RAM_resource_scheduling_policy']['CPU_protocol_source']=base['resident_RAM_resource_scheduling_policy']['CPU_protocol_source']
reverted['periodic_DEV_CPU_quiescence_policy']['source']=base['periodic_DEV_CPU_quiescence_policy']['source']
assert reverted==base
report={'kind':'ASR18_INACTIVE_packet_credit_source24_full_inverse_v1','candidate_bodies_imported_or_executed':False,
 'all_original_raw_byte_inverses_exact':True,'inverse':inverse,'unchanged_top_level_AST_bodies':unchanged,
 'native_update_and_commit_call_AST_exact':True,'configuration_prior_values_exact_after_only4declared_operational_changes':True,
 'base_source_configuration':base_pin,'configuration_template':pin(F/'configuration_template_NOT_ADMITTED.json'),
 'configuration_changed_keys':['resident_RAM_resource_scheduling_policy.CPU_protocol_source',
  'periodic_DEV_CPU_quiescence_policy.source','resident_RAM_packet_credit_policy','prior_resident_RAM_native_pressure156537_failure'],
 'historical_source22_CPU_protocol_remains_CPU9':True,'frozen23_preserved_as_rejected_historical_assertion':True,
 'actual_control_not_yet_executed':True,'production_admitted':False}
write(F/'source24_exact_inverse_and_configuration_report.json',report)
sources={k:pin(p)for k,p in {'helper':F/'asr18_owned_packet_credit_v1.py','CPU10':pairs['CPU9_to10'][1],
 'driver24':pairs['driver22_to23'][1],'wrapper24':pairs['wrapper22_to23'][1],
 'quiescence3':pairs['quiescence2_to3'][1],'fixture':F/'verify_owned_packet_credit_CPU_v1.py'}.items()}
scope=N/'ROOT_INACTIVE_packet_credit_saved156400_implementation_scope_v1.json'
handoff={'kind':'ASR18_INACTIVE_owned_packet_credit_source24_finite_CPU_scope_handoff_v1',
 'ROOT_implementation_scope':pin(scope),'sources':sources,'inverse':pin(F/'source24_exact_inverse_and_configuration_report.json'),
 'configuration_template_NOT_ADMITTED':pin(F/'configuration_template_NOT_ADMITTED.json'),
 'actual_completed_CPU_review_is_null':True,'source_only_preparation':True,'actual_CPU_control_executed':False,'production_admitted':False,
 'proposed_CPU_arguments':[str(Path('E:/gpu_training_scratchpad/scratch/BPE_broad_CPU_preflight_env_v2/Scripts/python.exe')),
  '-Xutf8','-B',sources['fixture']['path'],'--output',str(F/'actual_CPU_01'),'--ROOT-review','SEPARATE_ROOT_SCOPE_REQUIRED'],
 'fixture_SCOPE_required_fields':['accepted','reviewer','fixture_source','helper_source','CPU_source'],
 'proposed_limits':{'wall_seconds':180,'tracked_parent_plus_owned_children_private_bytes':536870912,
  'full_additional_prefetch_reservation_bytes':268435456,'commit_floor_bytes':2000000000,'physical_floor_bytes':1500000000,
  'E_free_reserve_bytes':100000000000,'CUDA_VISIBLE_DEVICES':'-1','all4threadcaps':1,'priority':'BelowNormal'},
 'genuine_scope':'Owned Win32 credit/cancellation/owner-death and real CPU10 parent byte transport; helper3 drain. Synthetic packets/constructor/worker explicitly substituted. Fake-clock600s deadline not actual600s stress wait.',
 'no_audio_model_GPU_or_authoritative_sampler_journal_operation':True,
 'future_saved_state':{'optimizer_updates':156400,'initial_invocation':18,'new_journal':'commits_000018.jsonl',
  'output':'resident_RAM_training_actual_19','full401072_adminNone':True,
  'restore_before_native_and_prefixcopy':True,'only_saved_prefix_bytes':316802360,
  'saved_prefix_sha256':'a5db7e40e8372c32c73b09b15ce87af7d3d9fed12bd8b1fede30f5fc3b461887',
  'historical_unsaved_positive_tail_count':137,'oldwhole17_digest_not_invented':True},
 'limits':['Full2.5GB native buffer, full256MiB queue/packet/prefetch and all original guards remain.',
  'One logical packet may have producer+receiver copies during send; no physical-one-copy claim.',
  'No release quantity, fit, speed, prevention or external-process cause has been measured.',
  'Keep first full gate, NONCONSUMING readiness and second fresh SAME full gate before take/propose/native.',
  'Future typed metadata and production require separately completed CPU and exact ROOT execution binding.']}
write(F/'inactive_source24_and_finite_CPU_handoff_v1.json',handoff)
print(json.dumps({'handoff':pin(F/'inactive_source24_and_finite_CPU_handoff_v1.json'),'inverse':pin(F/'source24_exact_inverse_and_configuration_report.json'),'sources':sources}))
