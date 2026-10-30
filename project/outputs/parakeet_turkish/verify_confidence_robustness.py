"""Independently verify a saved Parakeet confidence perturbation experiment.

CPU-only; does not load model weights, run inference, or change existing helpers.
Example:
  python verify_confidence_robustness.py --input runs/RUN/confidence_robustness.json       --reference v3=results/base_v3 --reference selected=results/RUN_adapted

Reference directories contain metadata.json and validation_en.jsonl. Paths are
resolved from the caller's current directory; use absolute paths when elsewhere.
Every model requires an external reference (v2/v3 defaults use saved baselines).
Config identity normalization is accepted only when both hashes are exactly
reconstructed and all other fingerprint components match. Exported canonical
config paths require checked finalization_provenance.json with file hashes.
"""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import sys

import jiwer
import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parents[2]
OUTPUT=ROOT/'outputs/parakeet_turkish'
sys.path.insert(0,str(OUTPUT))
from metrics import normalize_text

parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--input',type=Path,required=True)
parser.add_argument('--output',type=Path)
parser.add_argument('--manifest',type=Path,default=ROOT/'work/data/validation_en.jsonl')
parser.add_argument('--reference',action='append',default=[],metavar='NAME=RESULT_DIRECTORY')
parser.add_argument('--resamples',type=int,default=2000)
args=parser.parse_args()
if args.resamples<1:
    parser.error('--resamples must be positive')
source=args.input
report=json.loads(source.read_text(encoding='utf-8'))
manifest=args.manifest
count=len(report['samples'])
if not count:
    parser.error('The input experiment has no selected recordings')
expected=[json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines()][:count]
references={'v2':OUTPUT/'results/base_v2','v3':OUTPUT/'results/base_v3'}
for specification in args.reference:
    name,delimiter,path=specification.partition('=')
    if not delimiter or not name or not path:
        parser.error('--reference requires NAME=RESULT_DIRECTORY')
    references[name]=Path(path)
for name in report['models']:
    if name not in references:
        parser.error(f'External benchmark reference missing for {name!r}; supply --reference {name}=RESULT_DIRECTORY')
output=args.output or source.with_name(source.stem+'_independent_audit.json')
seed=int(report['seed'])
ids=[row['source_id'] for row in expected]
checks=[]
maximum_difference=0.0

def check(name,condition):
    checks.append({'name':name,'passed':bool(condition)})
    if not condition:
        raise AssertionError(name)

def near(name,left,right,tolerance=1e-10):
    global maximum_difference
    if left is None or right is None:
        check(name,left is right)
        return
    difference=abs(left-right)
    maximum_difference=max(maximum_difference,difference)
    check(name,difference<=tolerance)

def edits(a,b):
    d=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        p=[i]
        for j,y in enumerate(b,1):
            p.append(min(p[-1]+1,d[j]+1,d[j-1]+(x!=y)))
        d=p
    return d[-1]

def asr(rows):
    we=ce=rw=rc=0
    for row in rows:
        reference=normalize_text(row['reference'],'en'); hypothesis=normalize_text(row['hypothesis'],'en')
        we+=edits(reference.split(),hypothesis.split()); rw+=len(reference.split())
        ce+=edits(reference,hypothesis); rc+=len(reference)
    return {'wer':we/rw,'cer':ce/rc,'word_errors':we,'reference_words':rw,
            'character_errors':ce,'reference_characters':rc,'utterances':len(rows)}

def moments(values):
    if not values:
        return {'count':0,'mean':None,'mean_absolute':None,'rms':None,'absolute_p95':None,'absolute_max':None}
    x=np.asarray(values,dtype=float)
    return {'count':len(x),'mean':float(x.mean()),'mean_absolute':float(np.abs(x).mean()),
            'rms':float(np.sqrt(np.mean(x*x))),'absolute_p95':float(np.quantile(np.abs(x),.95)),
            'absolute_max':float(np.abs(x).max())}

check('complete result',report['status']=='complete')
check('fixed development manifest hash',report['manifest_sha256']==hashlib.sha256(manifest.read_bytes()).hexdigest())
check('perturbation script identity',report['script_sha256']==hashlib.sha256((OUTPUT/'confidence_robustness.py').read_bytes()).hexdigest())
check('ordered fixed development recording IDs',[s['source_id'] for s in report['samples']]==ids and len(ids)==count)
check('exact expected variants',report['variants']==['clean','gain_minus_0_2_db','noise_snr_50_db','silence_20_ms_each_side'])
check('valid perturbation seed',isinstance(report['seed'],int) and not isinstance(report['seed'],bool))

# Recreate measurements without calling the wrapper's perturbation function.
for row,sample in zip(expected,report['samples']):
    audio,sr=sf.read(row['audio_filepath'],dtype='float32')
    check('prepared mono16k '+row['source_id'],sr==16000 and audio.ndim==1)
    check('WAV provenance '+row['source_id'],sample['wav_sha256']==row['wav_sha256']==hashlib.sha256(Path(row['audio_filepath']).read_bytes()).hexdigest())
    rms=float(np.sqrt(np.mean(audio.astype(float)**2)))
    noise_seed=int.from_bytes(hashlib.sha256(f"{seed}:{row['source_id']}".encode()).digest()[:8],'little')
    noise=np.random.default_rng(noise_seed).standard_normal(audio.size); noise-=noise.mean()
    noise*=rms*10**(-50/20)/float(np.sqrt(np.mean(noise*noise)))
    variants={'clean':audio,'gain_minus_0_2_db':(audio*10**(-.2/20)).astype(np.float32),
              'noise_snr_50_db':(audio.astype(float)+noise).astype(np.float32),
              'silence_20_ms_each_side':np.pad(audio,(320,320))}
    for variant,wave in variants.items():
        saved=sample['variants'][variant]
        check('float32 waveform SHA '+variant+row['source_id'],saved['waveform_float32_sha256']==hashlib.sha256(np.asarray(wave,dtype='<f4').tobytes()).hexdigest())
        check('variant duration '+variant+row['source_id'],saved['samples']==wave.size and saved['seconds']==wave.size/16000)
        near('variant RMS '+variant+row['source_id'],saved['signal_rms'],float(np.sqrt(np.mean(wave.astype(float)**2))))
        if variant=='noise_snr_50_db':
            achieved=20*math.log10(rms/float(np.sqrt(np.mean((wave.astype(float)-audio.astype(float))**2))))
            near('achieved 50dB SNR '+row['source_id'],saved['measured_snr_db'],achieved)

derived={}
fingerprint_notes=[]
for name,model in report['models'].items():
    baseline=references[name]
    metadata=json.loads((baseline/'metadata.json').read_text(encoding='utf-8'))
    current_identity=model['model_fingerprint']['identity']; original_identity=metadata['model_fingerprint']['identity']
    identity_hash=lambda value:hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,default=str).encode()).hexdigest()
    check('probe fingerprint integrity '+name,identity_hash(current_identity)==model['model_fingerprint']['sha256'])
    check('reference fingerprint integrity '+name,identity_hash(original_identity)==metadata['model_fingerprint']['sha256'])
    mismatched=[key for key in current_identity if current_identity[key]!=original_identity[key]]
    if mismatched:
        check('only config fingerprint differs '+name,mismatched==['model_config_sha256'])
        # _name_or_path is serialized although slash direction does not affect
        # Windows file identity. Reconstruct both actual hashes exactly before
        # accepting the difference; no weight/model inference is needed.
        import torch
        from transformers import AutoConfig, AutoModelForTDT
        model_directory=Path(model['model_path'])
        if not model_directory.exists():
            model_directory=ROOT/model_directory
        if not model_directory.is_dir():
            raise ValueError(f"Cannot locate the recorded local model directory: {model['model_path']}")
        config=AutoConfig.from_pretrained(model_directory,local_files_only=True)
        config._name_or_path=model['model_path']
        # The actual HF model constructor normalizes nested config fields as
        # well as dtype. Use the same constructor on meta, without allocating
        # weights or running inference, rather than approximating its changes.
        with torch.device('meta'):
            template=AutoModelForTDT.from_config(config,dtype=torch.float32)
        config=template.config
        configuration=config.to_dict()
        digest=lambda value:hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,default=str).encode()).hexdigest()
        check('probe config fully reconstructed '+name,digest(configuration)==current_identity['model_config_sha256'])
        candidates=[model['model_path'].replace('/','\\'),model['model_path'].replace('\\','/')]
        provenance_path=model_directory/'finalization_provenance.json'
        provenance_evidence=None
        if provenance_path.exists():
            provenance=json.loads(provenance_path.read_text(encoding='utf-8'))
            canonical=provenance.get('canonical_model_name_or_path')
            hashes=provenance.get('candidate_files_sha256')
            if not canonical or not isinstance(hashes,dict) or not hashes:
                raise ValueError('Canonical provenance requires explicit path identity and exported file SHA256 evidence.')
            for relative,digest_value in hashes.items():
                target=(provenance_path.parent/relative).resolve()
                if not target.is_relative_to(provenance_path.parent.resolve()):
                    raise ValueError('Exported provenance file path is outside its model directory.')
                actual_digest=None
                if target.is_file():
                    with target.open('rb') as stream:
                        actual_digest=hashlib.file_digest(stream,'sha256').hexdigest()
                check('exact published checkpoint file '+name+relative,actual_digest==digest_value)
            candidates.append(canonical)
            provenance_evidence={'path':str(provenance_path.resolve()),
                'sha256':hashlib.sha256(provenance_path.read_bytes()).hexdigest(),
                'canonical_model_name_or_path':canonical,'exported_files_verified':len(hashes)}
        matched_path=None
        for alternate in candidates:
            configuration['_name_or_path']=alternate
            if digest(configuration)==original_identity['model_config_sha256']:
                matched_path=alternate
                break
        check('reference config differs only proven name/path identity '+name,matched_path is not None)
        fingerprint_notes.append({'model':name,'difference':'Only serialized _name_or_path differs.',
            'probe_name_or_path':model['model_path'],'reference_name_or_path':matched_path,
            'all_weights_generation_tokenizer_processor_code_dtypes_match':True,
            'both_config_hashes_exactly_reconstructed':True,'exported_provenance':provenance_evidence})
    else:
        check('original model fingerprint '+name,model['model_fingerprint']==metadata['model_fingerprint'])
    check('complete predictions '+name,len(model['records'])==count*len(report['variants']))
    by_variant={variant:{} for variant in report['variants']}
    wordmaps={variant:{} for variant in report['variants']}
    for record in model['records']:
        sid=record['source_id']; variant=record['variant']; row=next(r for r in expected if r['source_id']==sid)
        check('prediction provenance '+name+variant+sid,record['reference']==row['text'] and sid not in by_variant[variant])
        check('actual decoded confidence path '+name+variant+sid,record['hypothesis'].strip()==record['confidence']['text'].strip())
        reference=normalize_text(record['reference'],'en'); hypothesis=normalize_text(record['hypothesis'],'en')
        alignment=jiwer.process_words(reference,hypothesis)
        labels=[False]*len(hypothesis.split()); indexes=[None]*len(labels)
        for chunk in alignment.alignments[0]:
            if chunk.type=='equal':
                for hi,ri in zip(range(chunk.hyp_start_idx,chunk.hyp_end_idx),range(chunk.ref_start_idx,chunk.ref_end_idx)):
                    labels[hi]=True; indexes[hi]=ri
        words=record['alignment']['words']; tokens=record['confidence']['tokens']
        check('all lexical units captured '+name+variant+sid,[w['word'] for w in words]==hypothesis.split())
        check('independent correctness/reference-index alignment '+name+variant+sid,
              labels==[w['correct'] for w in words] and indexes==[w['reference_word_index'] for w in words])
        correct={}
        for word in words:
            lexical=[tokens[i] for i in word['token_indices']]
            calculated=math.exp(sum(t['logprob_by_temperature']['1'] for t in lexical)/len(lexical))
            conditional=math.exp(sum(t['conditional_nonblank_logprob'] for t in lexical)/len(lexical))
            near('reconstructed raw word probability '+name+variant+sid,word['confidence'],calculated)
            near('reconstructed conditional probability '+name+variant+sid,word['conditional_nonblank_confidence'],conditional)
            if word['correct']:
                correct[word['reference_word_index']]=dict(word,
                    selected_token_mean_logprob=sum(t['logprob'] for t in lexical)/len(lexical),
                    selected_token_mean_conditional_logprob=sum(t['conditional_nonblank_logprob'] for t in lexical)/len(lexical))
        by_variant[variant][sid]=record; wordmaps[variant][sid]=correct
    for variant in report['variants']:
        check('identical recording set '+name+variant,set(by_variant[variant])==set(ids))
    original={r['source_id']:r for r in [json.loads(line) for line in (baseline/'validation_en.jsonl').read_text(encoding='utf-8').splitlines()]}
    check('clean transcripts reproduce complete baseline '+name,all(by_variant['clean'][sid]['hypothesis']==original[sid]['hypothesis'] for sid in ids))
    clean_asr=asr(by_variant['clean'].values())
    for key,value in clean_asr.items():
        near('clean ASR '+name+key,model['summary']['clean_asr'][key],value)
    derived[name]={'clean_asr':clean_asr,'variants':{}}
    for variant in report['variants'][1:]:
        pairs={}; changed=by_variant[variant]; saved=model['summary']['perturbations'][variant]
        measured_asr=asr(changed.values())
        for key,value in measured_asr.items():
            near('perturbed ASR '+name+variant+key,saved['asr'][key],value)
        for sid in ids:
            before=wordmaps['clean'][sid]; after=wordmaps[variant][sid]
            for index in sorted(before.keys()&after.keys()):
                pairs[(sid,index)]={'raw_delta':after[index]['confidence']-before[index]['confidence'],
                    'conditional_delta':after[index]['conditional_nonblank_confidence']-before[index]['conditional_nonblank_confidence'],
                    'word_mean_logprob_delta':after[index]['selected_token_mean_logprob']-before[index]['selected_token_mean_logprob'],
                    'word_mean_conditional_logprob_delta':after[index]['selected_token_mean_conditional_logprob']-before[index]['selected_token_mean_conditional_logprob'],
                    'clean':before[index]['confidence'],'changed':after[index]['confidence'],'word':before[index]['word']}
        recorded={(w['source_id'],w['reference_word_index']):w for w in saved['paired_words']}
        check('exact shared-correct pair keys '+name+variant,set(recorded)==set(pairs) and len(recorded)==len(saved['paired_words']))
        for key in pairs:
            near('pair raw delta '+name+variant,recorded[key]['confidence_delta'],pairs[key]['raw_delta'])
            near('pair conditional delta '+name+variant,recorded[key]['conditional_nonblank_delta'],pairs[key]['conditional_delta'])
        for measure,field in [('raw_delta','shared_correct_word_confidence_delta'),('conditional_delta','shared_correct_word_conditional_nonblank_delta')]:
            for key,value in moments([p[measure] for p in pairs.values()]).items():
                near('sensitivity moment '+name+variant+measure+key,saved[field][key],value)
        raw_changed=sum(changed[s]['hypothesis']!=by_variant['clean'][s]['hypothesis'] for s in ids)
        normalized_changed=sum(normalize_text(changed[s]['hypothesis'],'en')!=normalize_text(by_variant['clean'][s]['hypothesis'],'en') for s in ids)
        check('transcript shifts '+name+variant,saved['raw_transcripts_changed']==raw_changed and saved['normalized_transcripts_changed']==normalized_changed)
        near('WER delta '+name+variant,saved['wer_delta_percentage_points'],100*(measured_asr['wer']-clean_asr['wer']))
        check('low-confidence fractions '+name+variant,saved['shared_correct_words_below_0_5_clean']==sum(p['clean']<.5 for p in pairs.values()) and saved['shared_correct_words_below_0_5_perturbed']==sum(p['changed']<.5 for p in pairs.values()))
        derived[name]['variants'][variant]={'asr':measured_asr,'raw_transcripts_changed':raw_changed,
            'normalized_transcripts_changed':normalized_changed,'raw_confidence_delta':moments([p['raw_delta'] for p in pairs.values()]),
            'conditional_delta':moments([p['conditional_delta'] for p in pairs.values()]),
            'word_mean_logprob_delta':moments([p['word_mean_logprob_delta'] for p in pairs.values()]),
            'word_mean_conditional_logprob_delta':moments([p['word_mean_conditional_logprob_delta'] for p in pairs.values()]),'pairs':pairs}

comparisons={}
for left,right in itertools.combinations(derived,2):
    pair={}
    for variant in report['variants'][1:]:
        a=derived[left]['variants'][variant]['pairs']; b=derived[right]['variants'][variant]['pairs']
        keys=sorted(a.keys()&b.keys()); saved=report['pairwise_model_sensitivity'][f'{left}_vs_{right}'][variant]
        check('same four-decode correct-word subset '+variant,saved['shared_correct_words_all_four_decodes']==len(keys))
        differences={measure:[abs(b[k][measure])-abs(a[k][measure]) for k in keys] for measure in
                     ('raw_delta','conditional_delta','word_mean_logprob_delta','word_mean_conditional_logprob_delta')}
        for measure,field in [('raw_delta','right_minus_left_absolute_raw_change'),('conditional_delta','right_minus_left_absolute_conditional_change')]:
            for key,value in moments(differences[measure]).items():
                near('cross-model sensitivity '+variant+measure+key,saved[field][key],value)
        # Added descriptive recording-bootstrap interval; the same ten source IDs
        # are resampled as units, keeping all their paired words together.
        interval={}
        for measure in differences:
            groups={sid:[abs(b[k][measure])-abs(a[k][measure]) for k in keys if k[0]==sid] for sid in ids}
            rng=random.Random(seed); means=[]
            for _ in range(args.resamples):
                sampled=[x for sid in rng.choices(ids,k=len(ids)) for x in groups[sid]]
                if sampled:
                    means.append(sum(sampled)/len(sampled))
            interval[measure]=np.quantile(means,[.025,.975]).tolist()
        pair[variant]={'shared_correct_words_all_four_decodes':len(keys),
            'left_mean_absolute_raw_change':sum(abs(a[k]['raw_delta']) for k in keys)/len(keys),
            'right_mean_absolute_raw_change':sum(abs(b[k]['raw_delta']) for k in keys)/len(keys),
            'right_minus_left_absolute_raw_change':moments(differences['raw_delta']),
            'right_minus_left_absolute_conditional_change':moments(differences['conditional_delta']),
            'left_mean_absolute_word_mean_logprob_change':sum(abs(a[k]['word_mean_logprob_delta']) for k in keys)/len(keys),
            'right_mean_absolute_word_mean_logprob_change':sum(abs(b[k]['word_mean_logprob_delta']) for k in keys)/len(keys),
            'right_minus_left_absolute_word_mean_logprob_change':moments(differences['word_mean_logprob_delta']),
            'right_minus_left_absolute_word_mean_conditional_logprob_change':moments(differences['word_mean_conditional_logprob_delta']),
            'descriptive_95pct_recording_bootstrap_difference':interval,
            'largest_differential_changes':[dict(source_id=k[0],reference_word_index=k[1],word=a[k]['word'],
                left=a[k],right=b[k]) for k in sorted(keys,key=lambda k:abs(abs(b[k]['raw_delta'])-abs(a[k]['raw_delta'])),reverse=True)[:5]]}
    comparisons[f'{left}_vs_{right}']=pair

for model in derived.values():
    for value in model['variants'].values():
        value.pop('pairs')
audit={'passed':True,'gpu_used':False,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
    'auditor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'check_count':len(checks),'maximum_numerical_recalculation_difference':maximum_difference,
    'all_checks_passed':all(c['passed'] for c in checks),'fingerprint_notes':fingerprint_notes,'models':derived,'paired_model_comparison':comparisons,
    'limitations':[f'{count} fixed English development recordings; descriptive, not an independent robustness test.',
        'Shared-correct confidence deltas exclude words becoming incorrect; transcript/WER changes are retained separately.',
        f'Bootstrap resamples only {count} recordings and does not turn this small probe into strong general evidence.',
        'Conditional nonblank scores diagnose timing competition but are not calibrated probabilities of word correctness.']}
audit['reference_directories']={name:str(references[name].resolve()) for name in report['models']}
audit['bootstrap_resamples']=args.resamples
output.parent.mkdir(parents=True,exist_ok=True)
output.write_text(json.dumps(audit,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps({'passed':audit['passed'],'check_count':audit['check_count'],'maximum_numerical_recalculation_difference':audit['maximum_numerical_recalculation_difference'],'models':list(audit['models']),'model_pairs':list(audit['paired_model_comparison']),'output':str(output.resolve())},ensure_ascii=False,indent=2))
