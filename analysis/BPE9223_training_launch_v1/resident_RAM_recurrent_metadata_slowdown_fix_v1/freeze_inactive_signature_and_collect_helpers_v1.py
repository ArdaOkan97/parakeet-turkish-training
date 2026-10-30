import ast,hashlib,json,os
from pathlib import Path
F=Path(__file__).parent;P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish'
def pin(p):
    p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
assert os.environ['TORCH_EXTENSIONS_DIR'].lower().startswith('e:')
helpers=[F/'asr18_fresh_single_open_metadata_signature_v1.py',F/'asr18_fresh_metadata_signature_scope_v1.py',F/'asr18_observe_original_gc_collect_v1.py']
for p in helpers:ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
native=pin(S/'asr18_BPE9223_grouped_RAM_native_v1.py');assert native['sha256']=='e4be25460f8958e918430837764347273c9288778d06edb016c44e7acb606085'
gc_scope=pin(P/'work/research/run_ASR18_native_throughput_component_v15.py');assert gc_scope['sha256']=='06c2e8d3f46a04b3584528f1206c477997fa5e28f9799672c5248012dcc4123e'
h={'kind':'ASR18_INACTIVE_fresh_signature_and_original_collect_observation_helpers_source_handoff_v1',
 'signature_helper_source':pin(helpers[0]),'integration_helper_source':pin(helpers[1]),'collect_observer_source':pin(helpers[2]),
 'human_fix_authority':pin(F/'ROOT_direct_human_then_fix_authorization_v1.json'),
 'readonly_candidate_design':pin(F/'INACTIVE_single_open_signature_vs_import_prefreeze_candidate_design_v1.json'),
 'original_sources_byte_unchanged':[pin(S/'bpe_declared_geometry_index_v1.py'),pin(S/'asr18_BPE9223_source_context_v2.py'),pin(S/'asr18_BPE9223_private_scalar_prefetch_v1.py'),native,gc_scope],
 'new_standalone_sources_have_no_old_source_inverse':'New helpers only. Existing original source files are byte-unchanged; future driver/wrapper exact literal edit ledger and full raw inverse will be provided before control/transition.',
 'integration_proposed_not_deployed':'After original full context admission/3denywrite locks, and before private worker creation, MainThread explicitly scopes exactly3 signature aliases. Original row/declaration/prefetch methods remain original AST and checks/count/poison messages. Only3canonical E metadata paths fast-supported, all other paths/uncertain API/format cases execute captured original signature. Restore aliases after original private worker join; unjoined abnormal lifetime does not pretend normal closure.',
 'collect_observation_proposed_not_deployed':'Install after original once-only resident freeze and after original builtin captured; scopes LIFO removes observer before original teardown. Same builtin once per call/sameargs/result/exception identity. Count/elapsed/max/raised/clockerror scalar per original20 window only; no frames/payload/exception history. Reentrant/concurrent summed call durations explicitly may overlap and are not CPU/losttime.',
 'proof_to_be_frozen_next':'One metadata-only focused original280 row/oracle + true Win32 signatures/handles/locks/fallback/error/mutation + honest method-stub GC passthrough/reentrant/clockfail + genuine bounded CPU gc cleanup test. No current job admitted.',
 'fixture_source':None,'launcher_source':None,'completed_CPU_review':None,'production_admitted':False,
 'native_math_GC_policy_module_import_order_and_all_science_guards_unchanged':True,
 'future_validation':'start=restored+20/end=max(start+1600,nextscheduledDEV+1020), all original20rows/200saves/realDEV/two full postDEV400blocks; no finite stop or speed guarantee'}
out=F/'inactive_fresh_signature_and_original_collect_helpers_source_handoff_v1.json'
with out.open('x',encoding='utf-8',newline='\n')as f:json.dump(h,f,indent=2);f.write('\n')
print(json.dumps({'handoff':pin(out),'sources':[pin(p)for p in helpers],'AST_only':True,'actual_control_execution':False}))
