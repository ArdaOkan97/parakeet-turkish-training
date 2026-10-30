"""Freeze only the next exact ordered decoder observation, never execute it."""
import ast,difflib,hashlib,json
from pathlib import Path
F=Path(__file__).parent;N=F.parent;B=Path(r'E:\gpu_training_scratchpad\runs\BPE9223_resident_RAM_20261018_01')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p):
 p=Path(p);b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
oldpath=N/'resident_RAM_original_interval_post140236_failure_actual_v1/observe_candidate140237_original_intervals_v2.py';old=oldpath.read_text(encoding='utf-8');new=old
candidate=pin(F/'exact_failed_update_candidate_rows.json');remaining=pin(F/'remaining_current_plan_updates.json');configuration=pin(B/'invocations/owner_000015/configuration.json')
for name,value in [('candidate_pin',candidate),('remaining_pin',remaining),('configuration_pin',configuration)]:
 line=next(x for x in new.splitlines()if x.startswith(name+'='));new=new.replace(line,name+'='+repr(value))
changes=[('eight-row candidate','nine-row candidate'),('plan12','plan135'),('index16','index14'),('==5','==6'),('five','six'),
 ('(16,)','(14,)'),('len(candidates)==8','len(candidates)==9'),("fixed['YODAS_candidate_rows']","fixed['ordered_yodas3_original_rows']"),
 ("plan['updates'][16]","plan['updates'][14]"),('post140236','post142894'),('actual_failure_closure_v2.json','actual_failure_closure.json')]
for a,b in changes:
 assert a in new,a;new=new.replace(a,b)
new=new.replace('from fastmss_BPE9223_cpu_common_v12 import numpy_runtime',"guard()\nfrom fastmss_BPE9223_cpu_common_v12 import numpy_runtime")
ast.parse(new);path=F/'observe_candidate142895_original_intervals_v1.py';assert not path.exists();path.write_text(new,encoding='utf-8')
delta=F/'ordered8_to_ordered9_decoder_observation.diff.txt';delta.write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=str(oldpath),tofile=str(path))),encoding='utf-8')
c=read(candidate['path']);plan=read(c['source_plan']['path']);policy=read(configuration['path'])['original_clip_geometry_overlay'];prior=[]
while policy is not None:
 cert=read(policy['certificate']['path']);prior.append({'source':policy['source'],'certificate':policy['certificate'],'ordinal':cert['original_row']['ordinal'],'source_id':cert['original_row']['unique_id']});policy=policy.get('prior_overlay')
assert len(prior)==6
rows=[{'row_index':i,'row':r}for i,r in enumerate(plan['updates'][14]['rows'])if r['dataset']=='alania_yodas3']
assert rows==c['ordered_yodas3_original_rows']and len(rows)==9
assert all(x['row']['ordinal']!=p['ordinal']and x['row']['source_id']!=p['source_id']for x in rows for p in prior)
handoff={'kind':'ASR18_INACTIVE_post142894_exact_ordered9_decoder_observation_scope_handoff_v1','source':pin(path),'prior_reviewed_source':pin(oldpath),'exact_delta':pin(delta),
 'inputs':{'failure_closure':pin(F/'actual_failure_closure.json'),'candidate':candidate,'remaining':remaining,'configuration':configuration,'current_plan':c['source_plan']},
 'actual_candidate_global_updates':142895,'source_plan135_index14_from_CPU2694_initial140200_not_native_plus_one':True,
 'ordered_row_indices':[x['row_index']for x in rows],'all9_rows_exclude_all6_prior_certificates_metadata_checked':True,'six_existing_policy_records':prior,
 'decoder':'unchanged production decoder2 ordinary unmatched yodas branch; original frozen baseline only same first actual failed row',
 'CPU_numpy_initialization':'unchanged production core12.numpy_runtime(frontend=False), no Torch/model import',
 'stop_at_first_actual_failure':True,'one_same_owner_execution_only_after_ROOT_scope':True,'limits':{'private_bytes':536870912,'wall_seconds':180,'full_extra_bytes':268435456,'commit_headroom_bytes':2000000000,'physical_headroom_bytes':1500000000,'E_reserve_bytes':100000000000,'CUDA_VISIBLE_DEVICES':'-1','all_four_thread_caps':1,'priority':'BelowNormal'},
 'fresh_output':str(F/'probe_actual_01'),'no_frontend_model_GPU_sampler_journal_or_source_change':True,'no_fullmember_or_bulk_corpus_pass':True,'no_production_remedy_or_retry_admitted':True}
p=F/'ordered9_decoder_observation_source_handoff_v1.json';assert not p.exists();p.write_text(json.dumps(handoff,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps({'source':pin(path),'handoff':pin(p),'candidate_indices':handoff['ordered_row_indices']}),flush=True)
