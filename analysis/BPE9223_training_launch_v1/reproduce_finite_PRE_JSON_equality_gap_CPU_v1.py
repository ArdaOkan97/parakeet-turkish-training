"""Reproduce only the saved PRE JSON transport rejection in frozen finite v1."""
import ast,importlib.util,json
from pathlib import Path
N=Path(r'E:\gpu_training_scratchpad\analysis\BPE9223_training_launch_v1')
prior=N/'review_finite_checkpoint_evidence_CPU_v1.py'
spec=importlib.util.spec_from_file_location('frozen_independent_review_fixture_v1',prior)
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
O=N/'independent_finite_PRE_JSON_gap_review_v1';O.mkdir(exist_ok=True);v.O=O;v.m.AREA=O
tree=ast.parse(prior.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
first=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assert) and 'normalize_fingerprint' in ast.unparse(n))
fn.body=fn.body[:first]+ast.parse('''
transported=f.normalize_fingerprint(json.loads(json.dumps(fp)))
assert fp['state_sha256']==transported['state_sha256']
assert f.digest(fp)==f.digest(transported)
assert not f.same(fp,transported)
assert type(fp['state']['RNG']['python']) is tuple and type(transported['state']['RNG']['python']) is list
assert type(fp['state']['RNG']['numpy']) is tuple and type(transported['state']['RNG']['numpy']) is list
helper=Path(f.__file__).read_text()
assert "require(same(metadata_fingerprint(metadata),pre), 'All854 saved descriptors differ from independent actual PRE hashes')" in helper
try:f.require(f.same(fp,transported),'All854 saved descriptors differ from independent actual PRE hashes')
except ValueError as error:reason=str(error)
else:raise AssertionError('Frozen finite v1 should reject this legitimate JSON roundtrip')
assert not any(k in sys.modules for k in ('torch','numpy','transformers','nemo'))
g.check()
result={'kind':'ASR18_independent_finite_PRE_JSON_equivalence_gap_reproduction_v1','controls_passed':True,'source_accepted':False,
 'source_handoff':rec(h),'helper':rec(f.__file__),'prior_independent_fixture':rec(prior),
 'prior_actual_fixture_exit1':{'actual_tool_chunk_id':'56f50b','exit_code':1,'failure':'Independent fixture compared metadata tuple reconstruction to JSON-transport PRE lists'},
 'descriptor_count':len(metadata['tensors']),'weights':len(fp['state']['weights']),'Adam_states':sum(len(p['state']) for p in fp['state']['Adam_partitions']),
 'all4_RNG_present':True,'canonical_full_fingerprint_equal':True,'state_sha256_equal':True,'Python_tuple_list_equality':False,
 'actual_frozen_check_rejects':reason,'bug':'Python/NumPy RNG tuple nodes become lists in actual PRE JSON. normalize_fingerprint only restores Adam integer dictionary keys. Frozen helper same() additionally requires Python equality and falsely rejects.',
 'supported_fix':'NEW helper successor should compare normalized full canonical digest/state_sha256 after independently checking both, preserving all existing source/process/854/215/clock/semantic checks. Add actual JSON-roundtrip fixture.',
 'actual_audio_model_payload_GPU_jobs':0,'metadata_fixture_is_not_actual_checkpoint_restore':True}
(O/'source_findings_v1.json').write_text(json.dumps(result,indent=2)+'\\n')
g.publish(completed=True,findings=rec(O/'source_findings_v1.json'))
print(json.dumps({'reproduction_passed':True,'source_accepted':False,'findings':rec(O/'source_findings_v1.json')}))
''').body
fn.name='reproduce';ast.fix_missing_locations(fn)
ns={**v.__dict__,'prior':prior,'O':O};exec(compile(ast.Module(body=[fn],type_ignores=[]),__file__,'exec'),ns)
if __name__=='__main__':ns['reproduce']()
