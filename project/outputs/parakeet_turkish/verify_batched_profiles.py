"""CPU-only verification of saved matched Parakeet training profiles.

Supply explicit, completed profile JSONs with repeated --input PATH. Checks
immutable source/weights/selected TRAIN WAV+cache identities, recorded gradient
parity arithmetic, and all timing/rate calculations. No model/GPU is loaded.
The comparison concerns four repeated TRAIN clips, not general training speed
or ASR quality. Failed numerical-parity reports never yield accepted speedups.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

import torch

OUTPUT=Path(__file__).resolve().parent
ROOT=OUTPUT.parents[1]


class Checks:
    def __init__(self):self.items=[]
    def require(self,condition,name):
        self.items.append({'name':name,'passed':bool(condition)})
        if not condition:raise AssertionError(name)
    def near(self,a,b,name,rtol=2e-8,atol=1e-10):
        self.require(math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=rtol,abs_tol=atol),name)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def selected_examples(quartet,check):
    assessment=read(OUTPUT/'batching_assessment.json')
    q=assessment['representative_train_quartets'][quartet]
    selected=[q['tr'][0],q['replay'][0],q['tr'][1],q['replay'][1]]
    manifests={}
    for path in {item['manifest'] for item in selected}:
        manifests[path]={row['source_id']:row for row in
            (json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip())}
    rows=[]
    for item in selected:
        sid=item['source_id'];row=manifests[item['manifest']][sid]
        check.require(item['dataset_split']==row['dataset_split']=='train',sid+': TRAIN provenance')
        check.require(item['language']==row['language'],sid+': language identity')
        check.near(item['duration'],row['duration'],sid+': duration identity')
        check.require(sha(row['audio_filepath'])==row['wav_sha256']==item['wav_sha256'],sid+': prepared WAV SHA256')
        check.require(sha(item['cache_path'])==item['cache_sha256'],sid+': teacher cache SHA256')
        cached=torch.load(item['cache_path'],map_location='cpu',weights_only=True)
        check.require(cached['source_id']==sid,sid+': cached recording ID')
        labels=cached['labels'].contiguous()
        label_sha=hashlib.sha256(labels.numpy().tobytes()).hexdigest()
        check.require(label_sha==item['labels_sha256'],sid+': target-label SHA256')
        check.require(tuple(cached['features'].shape)==tuple(item['cached_feature_shape']),sid+': original feature shape')
        check.require(cached['features'].shape[0]==item['encoder_frames'] and labels.numel()==item['target_tokens'],sid+': original T/U')
        check.require(bool(torch.isfinite(cached['features']).all()),sid+': finite teacher features')
        rows.append(dict(row,cache_path=item['cache_path']))
    return rows


def gradient_audit(report,reference,snapshot,check,label):
    parity=report['parity']
    check.require(parity.get('passed') is True,label+': numerical parity accepted')
    check.require(set(reference['gradients'])==set(snapshot['weights']),label+': reference/trainable parameter keys')
    actual_norms={}
    for name,value in reference['gradients'].items():
        check.require(tuple(value.shape)==tuple(snapshot['weights'][name].shape),label+': reference gradient shape '+name)
        check.require(bool(torch.isfinite(value).all()),label+': finite reference gradient '+name)
        actual_norms[name]=float(value.double().square().sum())**.5
    if report['reference_uses_frozen_original_single_clip_objective'] or report.get('reference_batched',False):
        check.require(parity['components']==reference['components'],label+': original reference loss components')
        return {'reference':True,'parameter_count':len(actual_norms)}
    check.require(parity['reference_components']==reference['components'],label+': matching original reference components')
    components=parity['components'];old=parity['reference_components']
    check.require(len(components)==len(old)==4,label+': TDT/tokenKL/durationKL/feature components')
    check.require(all(math.isfinite(x) for x in components+old),label+': finite loss components')
    deltas=[abs(a-b) for a,b in zip(components,old)]
    for index,(actual,expected) in enumerate(zip(parity['component_absolute_deltas'],deltas)):
        check.near(actual,expected,label+f': loss component delta {index}')
    parameters=parity['per_parameter']
    names=[p['name'] for p in parameters]
    check.require(len(names)==len(set(names)) and set(names)==set(actual_norms),label+': complete unique candidate gradient parameters')
    norm2=delta2=candidate2=dot=0.
    grouped={key:{'norm2':0.,'delta2':0.,'candidate2':0.,'dot':0.} for key in ('encoder_lora','heads')}
    full_cosine=True
    for parameter in parameters:
        name=parameter['name'];an=parameter['reference_norm']**2
        relative=parameter['relative_l2_difference']
        check.near(parameter['reference_norm'],actual_norms[name],label+': reconstructed reference norm '+name)
        check.require(math.isfinite(relative) and relative>=0,label+': finite parameter gradient delta '+name)
        dn=max(an,1e-20)*relative**2
        norm2+=an;delta2+=dn
        group=grouped['encoder_lora' if name.endswith(('lora_A','lora_B')) else 'heads']
        group['norm2']+=an;group['delta2']+=dn
        if all(key in parameter for key in ('candidate_norm','delta_norm','dot')):
            bn=parameter['candidate_norm']**2;ab=parameter['dot']
            check.near(parameter['delta_norm']**2,dn,label+': parameter delta norm '+name,atol=1e-12)
            check.near(dn,an+bn-2*ab,label+': parameter norm/dot identity '+name,atol=1e-10)
            candidate2+=bn;dot+=ab;group['candidate2']+=bn;group['dot']+=ab
        else:full_cosine=False
    relative=(delta2/max(norm2,1e-20))**.5
    check.near(parity['gradient_relative_l2_difference'],relative,label+': all-gradient relative L2 reconstruction')
    cosine=parity['gradient_cosine_similarity']
    check.require(math.isfinite(cosine) and -1<=cosine<=1+1e-10,label+': bounded gradient cosine')
    if full_cosine:
        check.near(cosine,dot/max((norm2*candidate2)**.5,1e-20),label+': all-gradient cosine reconstruction')
    tolerance=parity['tolerances']
    features=parity['encoder_feature_relative_l2_differences']
    check.require(len(features)==4 and all(math.isfinite(x) and x>=0 for x in features),label+': valid feature differences')
    accepted=(relative<tolerance['gradient_relative_l2'] and cosine>tolerance['gradient_cosine']
              and deltas[0]<tolerance['tdt_loss_absolute'] and max(deltas[1:])<tolerance['preservation_loss_absolute']
              and max(features)<tolerance['encoder_feature_relative_l2'])
    check.require(accepted,label+': recorded acceptance thresholds actually passed')
    group_summary={}
    for name,value in grouped.items():
        group_summary[name]={'reference_norm':value['norm2']**.5,
                            'relative_l2_difference':(value['delta2']/max(value['norm2'],1e-20))**.5}
        if full_cosine:group_summary[name]['cosine_similarity']=value['dot']/max((value['norm2']*value['candidate2'])**.5,1e-20)
    if 'gradient_groups' in parity:
        check.require(set(parity['gradient_groups'])==set(group_summary),label+': LoRA and head groups present')
        for name,value in parity['gradient_groups'].items():
            for key in ('relative_l2_difference','cosine_similarity'):
                check.near(value[key],group_summary[name][key],label+': reconstructed '+name+' '+key)
    if report['precision']=='fp32':
        check.require(full_cosine,label+': FP32 oracle requires independently reconstructable gradient groups')
        check.require(max(deltas)<1e-4 and max(features)<1e-4,
                      label+': strict FP32 oracle loss and encoder feature limits')
        check.require(relative<.002 and cosine>.99999,
                      label+': strict FP32 oracle global gradient limits')
        for name,value in group_summary.items():
            check.require(value['relative_l2_difference']<.002 and value['cosine_similarity']>.99999,
                          label+': strict FP32 oracle '+name+' gradient limits')
    return {'reference':False,'parameter_count':len(actual_norms),'relative_l2':relative,
            'cosine':cosine,'cosine_independently_reconstructed':full_cosine,'gradient_groups':group_summary}


def audit(path,args,check,snapshot,weights_sha):
    report=read(path);label=Path(path).name
    check.require(report['status']=='passed',label+': complete accepted profile; failed parity cannot yield speed claims')
    check.require(report['weights_sha256']==weights_sha,label+': exact initial trainable snapshot SHA256')
    for name,expected in report['code_sha256'].items():
        source=(OUTPUT/name).resolve()
        check.require(source.is_relative_to(OUTPUT.resolve()),label+': safe dependency path '+name)
        check.require(sha(source)==expected,label+': unchanged source '+name)
    critical={'profile_batched_training.py','batched_lora.py','efficient_tdt.py','runtime.py',
              'encoder_lora.py','extend_vocab.py','train_lora.py'}
    if report['tdt_implementation']=='v2':critical.add('efficient_tdt_v2.py')
    check.require(critical.issubset(report['code_sha256']),label+': core source provenance complete')
    rows=selected_examples(report['quartet'],check)
    check.require(report['source_ids']==[r['source_id'] for r in rows],label+': exact ordered TRAIN quartet')
    check.require(len(set(report['source_ids']))==4,label+': four unique TRAIN clips')
    audio=sum(r['duration'] for r in rows)
    check.near(report['audio_seconds_per_update'],audio,label+': repeated quartet audio per update')
    batch=report['batch_size']
    check.require(batch in (1,2,4) and report['accumulation']==4//batch,label+': effective batch exactly four')
    check.require(report.get('precision') in ('fp32','bf16'),label+': declared compute precision')
    reference_path=Path(report['reference_path']).resolve()
    check.require(reference_path.is_relative_to(ROOT.resolve()),label+': reference path inside workspace')
    reference=torch.load(reference_path,map_location='cpu',weights_only=True)
    identity=reference['identity']
    expected_identity={'weights_sha256':weights_sha,'precision':report['precision'],
                       'source_ids':report['source_ids'],'wav_sha256':[sha(row['audio_filepath']) for row in rows],
                       'cache_sha256':[sha(row['cache_path']) for row in rows],
                       'original_dependency_sha256':{name:report['code_sha256'][name] for name in
                            ('runtime.py','encoder_lora.py','extend_vocab.py','train_lora.py','efficient_tdt.py')}}
    check.require(identity==expected_identity,label+': exact gradient reference weights/precision/quartet/dependency identity')
    if report['reference_uses_frozen_original_single_clip_objective']:
        check.require(batch==1 and report['tdt_implementation']=='original' and report['chunk_size']==16 and report['encoder_checkpointing'],
                      label+': genuine native original B1/chunk16/checkpoint reference')
    gradient=gradient_audit(report,reference,snapshot,check,label)
    if not report['timing']:
        check.require(args.allow_parity_only,label+': timing missing; diagnostic must be explicitly requested')
        return {'path':str(Path(path).resolve()),'sha256':sha(path),'quartet':report['quartet'],
                'batch_size':batch,'tdt_implementation':report['tdt_implementation'],
                'source_ids':report['source_ids'],'weights_sha256':weights_sha,'hardware':report['hardware'],
                'precision':report['precision'],'gradient_audit':gradient,'timing':{},
                'original_reference':report['reference_uses_frozen_original_single_clip_objective'],
                'parity_only':True}
    check.require(set(report['timing'])=={'prefetched_cpu_inputs','end_to_end'},label+': both timing categories complete')
    timings={}
    for category,value in report['timing'].items():
        seconds=value['seconds'];n=value['updates']
        check.require(type(n) is int and n>=args.minimum_updates and len(seconds)==n,label+': adequate complete updates '+category)
        check.require(type(value['warmup_updates']) is int and value['warmup_updates']>=1,label+': optimizer/kernel warmup '+category)
        check.require(all(math.isfinite(x) and x>0 for x in seconds),label+': finite positive timed durations '+category)
        check.near(value['median_seconds_per_update'],statistics.median(seconds),label+': exact median '+category)
        p90=sorted(seconds)[min(n-1,int(.9*n))]
        check.near(value['p90_seconds_per_update'],p90,label+': recorded p90 order statistic '+category)
        check.near(value['audio_min_per_wall_min'],audio*n/sum(seconds),label+': audio/wall ratio '+category)
        check.near(value['clips_per_second'],4*n/sum(seconds),label+': clips/second '+category)
        allocated=value['peak_allocated_vram_gb'];reserved=value['peak_reserved_vram_gb']
        check.require(math.isfinite(allocated) and math.isfinite(reserved) and 0<allocated<=reserved,label+': allocated/reserved VRAM '+category)
        timings[category]=dict(value)
    return {'path':str(Path(path).resolve()),'sha256':sha(path),'quartet':report['quartet'],
            'batch_size':batch,'tdt_implementation':report['tdt_implementation'],
            'source_ids':report['source_ids'],'weights_sha256':weights_sha,'hardware':report['hardware'],
            'precision':report['precision'],'gradient_audit':gradient,'timing':timings,'parity_only':False,
            'original_reference':report['reference_uses_frozen_original_single_clip_objective']}


def audit_selection(path,profiles,check):
    selected=read(path);label=Path(path).name
    check.require(selected['passed'] is True and selected['throughput_improved'] is True,
                  label+': completed favorable selection')
    for name,expected in selected['input_sha256'].items():
        dependency=(OUTPUT/name).resolve()
        check.require(dependency.is_relative_to(OUTPUT.resolve()) and sha(dependency)==expected,
                      label+': input SHA256 '+name)
    original=next(p for p in profiles if p['quartet']=='median' and p['original_reference'] and not p['parity_only'])
    candidate=next(p for p in profiles if p['quartet']=='median' and not p['original_reference'] and not p['parity_only'])
    high=next(p for p in profiles if p['quartet']=='high_lattice' and not p['original_reference'] and not p['parity_only'])
    configuration=selected['selected_configuration']
    check.require(configuration=={'batch_size':1,'accumulation':4,'tdt_implementation':'v2',
                                  'chunk_size':16,'encoder_checkpointing':True},
                  label+': selected verified B1 exact-loss path')
    check.require(candidate['batch_size']==high['batch_size']==1 and candidate['tdt_implementation']==high['tdt_implementation']=='v2',
                  label+': selected measured profiles use B1 v2')
    old=original['timing']['end_to_end'];new=candidate['timing']['end_to_end']
    for prefix,timing in (('before',old),('after',new)):
        for key in ('audio_min_per_wall_min','median_seconds_per_update','peak_allocated_vram_gb'):
            check.near(selected[prefix+'_'+key],timing[key],label+': '+prefix+' '+key)
    ratio=new['audio_min_per_wall_min']/old['audio_min_per_wall_min']
    check.near(selected['throughput_ratio'],ratio,label+': measured throughput ratio')
    check.near(selected['throughput_increase_percent'],(ratio-1)*100,label+': measured throughput percent increase')
    check.require(ratio>1,label+': throughput improvement exists')
    long=selected['long_clip_safety']
    check.require(long['source_ids']==high['source_ids'] and long['parity_passed'] is True,
                  label+': exact high-lattice TRAIN quartet and passed parity')
    check.require(long['timing']==high['timing']['end_to_end'],label+': exact long-clip timing evidence')
    for profile in (candidate,high):
        gradient=profile['gradient_audit']
        check.require(gradient['relative_l2']<.005 and gradient['cosine']>.99999,
                      label+': selected B1 global gradient safety')
        for group,value in gradient['gradient_groups'].items():
            check.require(value['relative_l2_difference']<.005 and value['cosine_similarity']>.99999,
                          label+': selected B1 '+group+' gradient safety')
    historical=read(OUTPUT/'training_audio_throughput.json')
    # The historical whole-run rate is context, not a matched profile denominator.
    check.near(selected['historical_full_run_train_audio_min_per_wall_min'],
               historical['training_intervals_without_dev_pauses']['audio_min_per_wall_min'],
               label+': exact separately labeled historical training context')
    check.require('B2 not selected' in selected['physical_batching_status']
                  and 'not an expanded-run average' in selected['expanded_run_rate'],
                  label+': unresolved B2 and fixed-quartet scope remain explicit')
    return {'path':str(Path(path).resolve()),'sha256':sha(path),'configuration':configuration,
            'end_to_end_mean_throughput_speedup':ratio,'long_peak_allocated_vram_gb':long['timing']['peak_allocated_vram_gb'],
            'historical_context_sha256':sha(OUTPUT/'training_audio_throughput.json')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',action='append',type=Path,required=True)
    parser.add_argument('--weights',type=Path,default=ROOT/'work/lora_20261010_01_best.pt')
    parser.add_argument('--output',type=Path,default=OUTPUT/'batched_profiles_independent_audit.json')
    parser.add_argument('--minimum-updates',type=int,default=10)
    parser.add_argument('--allow-parity-only',action='store_true',help='Verify diagnostics without producing throughput comparisons for them')
    parser.add_argument('--selection',type=Path,help='Verify the saved B1 selection against median and high-lattice input profiles')
    args=parser.parse_args()
    if args.minimum_updates<1:parser.error('--minimum-updates must be positive')
    torch.set_num_threads(2)
    check=Checks()
    result={'passed':False,'gpu_used':False,'auditor_sha256':sha(Path(__file__)),
            'inputs':[str(x.resolve()) for x in args.input],
            'limitations':[
                'Matched repeat profiling uses one quartet of four TRAIN clips; broader throughput needs multiple length buckets.',
                'End-to-end category reads WAVs and recomputes frontend, but teacher cache items are preloaded outside timing.',
                'CPU-prefetched category still includes CPU-to-GPU transfer and optimizer update.',
                'Recorded numerical GPU differences are approximate BF16 parity, not bitwise equality or ASR quality evidence.',
                'This CPU audit recomputes reported gradient/timing arithmetic; no new GPU computation is performed.'
            ]}
    try:
        weights_sha=sha(args.weights)
        snapshot=torch.load(args.weights,map_location='cpu',weights_only=True)
        profiles=[audit(path,args,check,snapshot,weights_sha) for path in args.input]
        comparisons=[]
        for baseline in profiles:
            if not baseline['original_reference'] or baseline['parity_only']:continue
            for candidate in profiles:
                if candidate['original_reference'] or candidate['parity_only'] or candidate['quartet']!=baseline['quartet']:continue
                check.require(candidate['source_ids']==baseline['source_ids'] and candidate['weights_sha256']==baseline['weights_sha256'],
                              'speed comparison exact quartet and initial weights')
                check.require(candidate['hardware']==baseline['hardware'],'speed comparison same GPU')
                check.require(candidate['precision']==baseline['precision'],'speed comparison same compute precision')
                for category in baseline['timing']:
                    old=baseline['timing'][category];new=candidate['timing'][category]
                    comparisons.append({'quartet':baseline['quartet'],'batch_size':candidate['batch_size'],
                        'tdt':candidate['tdt_implementation'],'timing_category':category,
                        'median_update_speedup':old['median_seconds_per_update']/new['median_seconds_per_update'],
                        'mean_throughput_speedup':new['clips_per_second']/old['clips_per_second'],
                        'allocated_memory_change_gb':new['peak_allocated_vram_gb']-old['peak_allocated_vram_gb']})
        selection=audit_selection(args.selection,profiles,check) if args.selection else None
        result.update(passed=True,profiles=profiles,matched_speed_comparisons=comparisons,selection_audit=selection)
    except Exception as error:
        result['error']=f'{type(error).__name__}: {error}'
    result['checks']=check.items;result['check_count']=len(check.items)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'passed':result['passed'],'checks':len(check.items),'error':result.get('error'),'output':str(args.output)},indent=2))
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
