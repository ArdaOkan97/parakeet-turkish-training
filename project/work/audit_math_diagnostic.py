"""Independent CPU reconstruction of the completed effective4 MATH diagnostic."""
import hashlib
import json
import math
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'outputs/parakeet_turkish'
sys.path.insert(0,str(HERE))
from verify_batched_profiles import Checks,selected_examples

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
check=Checks()
source=HERE/'batch_math_objective_gradient_diagnostic.json'
report=read(source)
check.require(report['correctness_only'] and not report['timing_measured'] and report['optimizer_updates']==0,
              'correctness-only four-clip scope')
check.require(report['effective_batch']==4 and report['forced_backend']=='SDPBackend.MATH','fixed effective4 MATH protocol')
check.require(report['concurrent_gpu_context']=='solo','declared solo correctness context; no throughput claim')
check.require(report['wrapper_sha256']==sha(HERE/'diagnose_batch_objective_math.py'),'actual wrapper SHA')
for name,expected in report['source_sha256'].items():check.require(sha(HERE/name)==expected,'unchanged source '+name)
check.require(report['encoder_diagnostic_sha256']==sha(report['encoder_diagnostic_path']),'independent encoder diagnostic SHA')
snapshot_path=ROOT/'work/lora_20261010_01_best.pt'
snapshot=torch.load(snapshot_path,map_location='cpu',weights_only=True)
rows=selected_examples('median',check)
expected_identity={'weights_sha256':sha(snapshot_path),'source_ids':[r['source_id'] for r in rows],
    'wav_sha256':[sha(r['audio_filepath']) for r in rows],
    'cache_sha256':[sha(r['cache_path']) for r in rows],
    'original_dependency_sha256':{name:report['source_sha256'][name] for name in
        ('runtime.py','encoder_lora.py','extend_vocab.py','train_lora.py','efficient_tdt.py')}}
cases={}
for case,meta in report['case_artifacts'].items():
    check.require(meta['status']=='complete' and meta['case']==case,'complete case '+case)
    check.require(meta['wrapper_sha256']==report['wrapper_sha256'] and meta['source_sha256']==report['source_sha256'],case+': exact code')
    check.require(sha(meta['profile_report_path'])==meta['profile_report_sha256'],case+': actual profile SHA')
    check.require(sha(meta['gradients_path'])==meta['gradients_sha256'],case+': actual gradient artifact SHA')
    for field in ('backend_flags','backend_flags_after'):
        flags=meta[field]
        check.require(flags['sdpa_math_enabled'] and not any(flags[k] for k in
            ('sdpa_flash_enabled','sdpa_mem_efficient_enabled','sdpa_cudnn_enabled','tf32_matmul_allowed','tf32_cudnn_allowed')),
            case+': exclusive MATH and TF32 off '+field)
    check.require(meta['backend_flags']==meta['backend_flags_after'],case+': stable flags')
    value=torch.load(meta['gradients_path'],map_location='cpu',weights_only=True)
    check.require(value['identity']==dict(expected_identity,precision=meta['precision']),case+': actual reference identity')
    check.require(set(value['gradients'])==set(snapshot['weights']) and len(value['gradients'])==207,case+': full207 layout')
    for name,g in value['gradients'].items():
        check.require(g.dtype==torch.float32 and g.shape==snapshot['weights'][name].shape and bool(torch.isfinite(g).all()),case+': finite FP32 gradient '+name)
    check.require(len(value['encoder_features'])==4,case+': four valid features')
    cases[case]=value

def stats(values):
    aa,bb,dd,ab=values
    return {'reference_norm':math.sqrt(aa),'candidate_norm':math.sqrt(bb),'delta_norm':math.sqrt(dd),
        'relative_l2_difference':math.sqrt(dd/max(aa,1e-30)) if aa or bb or dd else 0.,
        'cosine_similarity':ab/max(math.sqrt(aa*bb),1e-30) if aa or bb or dd else 1.}

def recorded_stats(actual,expected,label):
    for key,value in expected.items():check.near(actual[key],value,label+': '+key,atol=1e-10)

results={}
for label,original in report['comparisons'].items():
    a,b=cases[original['reference_case']],cases[original['candidate_case']]
    groups={'encoder_lora':[0.,0.,0.,0.],'heads':[0.,0.,0.,0.]};total=[0.,0.,0.,0.]
    check.require(len(original['per_parameter'])==207,label+': all207 recorded parameters')
    pmap={row['name']:row for row in original['per_parameter']}
    check.require(set(pmap)==set(a['gradients']),label+': exact parameter names')
    for name,g in a['gradients'].items():
        h=b['gradients'][name]
        left,right=g.double(),h.double()
        values=[float(left.square().sum()),float(right.square().sum()),float((left-right).square().sum()),float((left*right).sum())]
        recorded_stats(pmap[name],stats(values),label+': reconstructed '+name)
        check.require(pmap[name]['bitwise_equal']==torch.equal(g,h),label+': gradient equality '+name)
        check.require(pmap[name]['elements']==g.numel(),label+': gradient size '+name)
        group='encoder_lora' if name.endswith(('lora_A','lora_B')) else 'heads'
        groups[group]=[x+y for x,y in zip(groups[group],values)];total=[x+y for x,y in zip(total,values)]
    gs=stats(total);gstats={name:stats(value) for name,value in groups.items()}
    recorded_stats(original['global_gradient'],gs,label+': independently reconstructed global')
    for name,value in gstats.items():recorded_stats(original['gradient_groups'][name],value,label+': independently reconstructed '+name)
    fdiffs=[];fequal=[]
    for index,(left,right) in enumerate(zip(a['encoder_features'],b['encoder_features'])):
        check.require(left.shape==right.shape and bool(torch.isfinite(left).all()) and bool(torch.isfinite(right).all()),label+': valid feature '+str(index))
        rel=float((left.double()-right.double()).norm()/left.double().norm().clamp_min(1e-30))
        check.near(original['encoder_features'][index]['relative_l2_difference'],rel,label+': feature relative '+str(index))
        check.require(original['encoder_features'][index]['bitwise_equal']==torch.equal(left,right),label+': feature equality '+str(index))
        fdiffs.append(rel);fequal.append(torch.equal(left,right))
    deltas={name:abs(float(x)-float(y)) for name,x,y in zip(('tdt','token_kl','duration_kl','feature_distillation'),a['components'],b['components'])}
    for name,value in deltas.items():check.near(original['component_absolute_deltas'][name],value,label+': component '+name)
    weighted=abs(sum(c*float(x) for c,x in zip((1.,5.,.5,1.),a['components']))-sum(c*float(x) for c,x in zip((1.,5.,.5,1.),b['components'])))
    check.near(original['weighted_objective_absolute_delta'],weighted,label+': weighted objective')
    result={'global_gradient':gs,'gradient_groups':gstats,'feature_max_relative_l2':max(fdiffs),
            'features_bitwise_equal':all(fequal),'component_absolute_deltas':deltas,
            'weighted_objective_absolute_delta':weighted,'gated':original['gated']}
    if original['gated']:
        gate=original['strict_gate'];limits=gate['tolerances'];same=gate['same_shape_encoder_bitwise_equal_required']
        decisions=[]
        for value in [gs,*gstats.values()]:
            decisions.append(value['relative_l2_difference']<limits['gradient_relative_l2'] and value['cosine_similarity']>limits['gradient_cosine_min'])
        decisions.append(all(x<limits['encoder_feature_relative_l2'] for x in fdiffs))
        if same:decisions.extend((all(fequal),max(deltas.values())<1e-4))
        check.require(gate['per_check_passed']==decisions and gate['passed']==all(decisions),label+': exact original strict gate reconstruction')
        result['original_strict_gate_passed']=all(decisions)
        result['additional_component_and_weighted_1e4_passed']=max(deltas.values())<1e-4 and weighted<1e-4
    results[label]=result
overall=all(v['original_strict_gate_passed'] for v in results.values() if v['gated'])
check.require(report['passed']==overall and report['status']==('passed' if overall else 'strict_parity_failed'),'original failure status preserved')
out={'passed':True,'gpu_used':False,'checks_count':len(check.items),'checks':check.items,
    'scope':'Independent CPU audit of effective4 forced-MATH full gradient diagnostic; numerical strict failure is preserved. No speed or batch adoption claim.',
    'source_report_path':str(source),'source_report_sha256':sha(source),'auditor_path':str(Path(__file__).resolve()),
    'auditor_sha256':sha(__file__),'original_numerical_passed':overall,'comparisons':results}
(HERE/'batch_math_objective_independent_audit.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'passed':True,'checks':len(check.items),'original_numerical_passed':overall,'summary':results}))
