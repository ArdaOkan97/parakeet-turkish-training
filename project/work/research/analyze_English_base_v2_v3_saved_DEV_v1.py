"""One bounded saved-row DEV audit. No inference, fitting, TEST or ML imports."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'outputs/parakeet_turkish'
sys.path.insert(0,str(HERE))
from confidence import align_word_confidences, confidence_report
from metrics import normalize_text, score_transcriptions

def require(ok,message):
    if not ok:raise ValueError(message)

def record(path):
    path=Path(path)
    return {'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}

def quantile(values,q):
    values=sorted(values)
    if not values:return None
    position=q*(len(values)-1);lo=math.floor(position);hi=math.ceil(position)
    return values[lo]+(values[hi]-values[lo])*(position-lo)

def distribution(values):
    return {'count':len(values),'mean':statistics.fmean(values) if values else None,
        'quantiles':{str(q):quantile(values,q) for q in (0.,.01,.05,.5,.95,1.)},
        'below_0_5_count':sum(v<.5 for v in values),'below_0_5_fraction':sum(v<.5 for v in values)/len(values) if values else None,
        'below_0_8_count':sum(v<.8 for v in values),'below_0_8_fraction':sum(v<.8 for v in values)/len(values) if values else None}

SCORES=('word_geometric_mean_probability','minimum_token_probability','token_probability_product',
    'arithmetic_mean_token_probability','mean_token_logprob','minimum_token_logprob')

def describe(words):
    result={key:distribution([word[key] for word in words]) for key in SCORES[:4]}
    result['token_logprobs']={key:{'mean':statistics.fmean([w[key] for w in words]) if words else None,
        'minimum':min((w[key] for w in words),default=None)} for key in SCORES[4:]}
    tokens=[lp for word in words for lp in word['token_logprobs']]
    result['pooled_emitted_token_logprob']={'count':len(tokens),'mean':statistics.fmean(tokens) if tokens else None,
        'minimum':min(tokens,default=None)}
    result['mean_tokens_per_word']=statistics.fmean(w['token_count'] for w in words) if words else None
    result['word_geomean_at_least_0_8_but_token_min_below_0_5']=sum(w[SCORES[0]]>=.8 and w[SCORES[1]]<.5 for w in words)
    return result

def compact_example(word):
    return {key:word[key] for key in ('source_id','sentence_id','reference_word_index','word','correct',
        'token_count',*SCORES,'token_pieces','token_probabilities','token_blank_probabilities')}

def analyze(version):
    folder=HERE/'results'/version
    path=folder/'validation_en.jsonl'
    require(path.exists(),'Saved English DEV predictions missing: '+version)
    metadata=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
    manifest=metadata['manifests']['validation_en']
    identity=metadata['model_fingerprint']['identity']
    require(identity['evaluation_code_sha256']['confidence.py']==record(HERE/'confidence.py')['sha256']
        and identity['evaluation_code_sha256']['metrics.py']==record(HERE/'metrics.py')['sha256'],
        'Saved aggregation/scorer source differs from imported current functions')
    words=[];alignments=[];rows=[];identities=[];token_total=0;max_temp_difference=0.;max_raw_geo_difference=0.
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():continue
            row=json.loads(line)
            require(row['language']=='en' and row['split']=='validation','DEV English only')
            require(row['confidence']['text']==row['hypothesis'],'Captured decode text differs')
            alignment=align_word_confidences(row['reference'],row['confidence'],language='en')
            require([(w['word'],w['correct'],w['operation'],w['reference_word_index']) for w in alignment['words']]
                ==[(w['word'],w['correct'],w['operation'],w['reference_word_index']) for w in row['alignment']['words']],
                'Independent current alignment differs from saved labels')
            require(normalize_text(row['hypothesis'],'en').split()==[w['word'] for w in alignment['words']],
                'Confidence words do not represent scored hypothesis')
            tokens=row['confidence']['tokens'];token_total+=len(tokens)
            for token in tokens:
                lp=token['logprob'];require(math.isfinite(lp) and lp<=0.,'Invalid raw emitted token logprob')
                max_temp_difference=max(max_temp_difference,abs(lp-token['logprob_by_temperature']['1']))
            for word in alignment['words']:
                chosen=[tokens[index] for index in word['token_indices']]
                require(bool(chosen),'Word has no contributing token')
                logprobs=[token['logprob'] for token in chosen]
                total=math.fsum(logprobs);mean=total/len(logprobs)
                probability=word['confidence_by_temperature']['1']
                require(math.isclose(probability,word['confidence'],abs_tol=1e-14),'Raw stored T1 confidence differs')
                max_raw_geo_difference=max(max_raw_geo_difference,abs(probability-math.exp(mean)))
                words.append({'source_id':row['source_id'],'sentence_id':row['sentence_id'],
                    'reference_word_index':word['reference_word_index'],'word':word['word'],'correct':word['correct'],
                    'operation':word['operation'],'token_count':len(chosen),'word_geometric_mean_probability':probability,
                    'minimum_token_probability':math.exp(min(logprobs)),'token_probability_product':math.exp(total),
                    'arithmetic_mean_token_probability':statistics.fmean(math.exp(lp) for lp in logprobs),
                    'mean_token_logprob':mean,'minimum_token_logprob':min(logprobs),'token_logprobs':logprobs,
                    'token_pieces':[token['piece'] for token in chosen],
                    'token_probabilities':[math.exp(lp) for lp in logprobs],
                    'token_blank_probabilities':[token['blank_probability'] for token in chosen]})
            alignments.append(alignment)
            rows.append({key:row[key] for key in ('source_id','language','reference','hypothesis')})
            identities.append({key:row[key] for key in ('source_id','sentence_id','language','split','reference',
                'dataset_revision','audio_sha256','wav_sha256','duration')})
    require(len(identities)==len({row['source_id'] for row in identities})==50
        and [row['source_id'] for row in identities]==manifest['source_ids'] and manifest['utterances']==50,
        'Complete exact ordered saved50 DEV required')
    scored={key:value for key,value in score_transcriptions(rows,language='en').items() if key!='details'}
    require(scored==json.loads((folder/'summary.json').read_text(encoding='utf-8'))['validation_en']['asr'],
        'Independent saved DEV ASR differs from summary')
    reliability=confidence_report(alignments,temperature=1.,bins=10)
    report={'predictions':record(path),'metadata':record(folder/'metadata.json'),
        'model_fingerprint_sha256':metadata['model_fingerprint']['sha256'],
        'saved_model_identity':{key:identity[key] for key in ('model_state_sha256','model_config_sha256',
            'generation_config_sha256','tokenizer_sha256','processor_sha256','inference_dtype','evaluation_code_sha256')},
        'fixed_DEV_manifest_sha256':manifest['file_sha256'],'ASR_independent_rescore':scored,
        'utterances':50,'emitted_tokens':token_total,'emitted_words':len(words),
        'correct_words':sum(word['correct'] for word in words),'incorrect_words':sum(not word['correct'] for word in words),
        'correct_word_scores':describe([w for w in words if w['correct']]),
        'incorrect_word_scores':describe([w for w in words if not w['correct']]),
        'raw_stored_geomean_word_reliability_descriptive':{key:reliability[key] for key in (
            'word_count','emitted_word_accuracy','mean_confidence','underconfidence_gap','ece','brier','nll',
            'correctness_auroc','unscored_deletions','reference_word_count')},
        'numeric_reconstruction':{'maximum_token_raw_logprob_vs_T1_logprob_difference':max_temp_difference,
            'maximum_word_stored_confidence_vs_raw_logprob_geomean_difference':max_raw_geo_difference},
        'lowest_correct_token_minimum_examples':[compact_example(w) for w in
            sorted((word for word in words if word['correct']),key=lambda w:w['minimum_token_probability'])[:6]],
        'named_correct_examples':[compact_example(w) for w in words if w['correct'] and w['word'] in ('kruger','accords')]}
    return report,words,identities

def main():
    versions={};wordsets={};ids={}
    for version in ('base_v2','base_v3'):
        versions[version],wordsets[version],ids[version]=analyze(version)
    require(ids['base_v2']==ids['base_v3'],'Paired DEV source/reference/PCM identity differs')
    by_version={version:{(w['source_id'],w['reference_word_index'],w['word']):w for w in words if w['correct']}
        for version,words in wordsets.items()}
    common=sorted(set(by_version['base_v2'])&set(by_version['base_v3']))
    paired={version:[by_version[version][key] for key in common] for version in by_version}
    comparative={version:describe(words) for version,words in paired.items()}
    deltas={score:{'mean_v3_minus_v2':statistics.fmean(b[score]-a[score] for a,b in zip(paired['base_v2'],paired['base_v3'])),
        'v3_lower_count':sum(b[score]<a[score] for a,b in zip(paired['base_v2'],paired['base_v3']))} for score in SCORES}
    report={'schema':1,'kind':'saved_paired_English_DEV_v2_v3_word_and_token_confidence_audit_v1','passed':True,
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'scope':'Existing original-model50 English DEV traces only',
        'source':record(__file__),'aggregation_source':record(HERE/'confidence.py'),'scorer_source':record(HERE/'metrics.py'),
        'versions':versions,'paired_DEV_identity':{'utterances':50,'all_ordered_source_reference_revision_PCM_fields_exact':True,
            'identity_sha256':hashlib.sha256(json.dumps(ids['base_v2'],sort_keys=True).encode()).hexdigest()},
        'shared_correct_reference_words':len(common),'shared_correct_score_distributions':comparative,
        'shared_correct_score_deltas':deltas,
        'shared_correct_segmentation':{'different_piece_sequence_count':sum(a['token_pieces']!=b['token_pieces']
            for a,b in zip(paired['base_v2'],paired['base_v3'])),
            'different_token_count_count':sum(a['token_count']!=b['token_count'] for a,b in zip(paired['base_v2'],paired['base_v3']))},
        'historical_report_scope_warning':'english_confidence_comparison shared_correct_words and base_v3_english_confidence_audit refer to100 TEST recordings. This audit neither opens TEST predictions nor rescales their examples.',
        'definitions':{'raw_token_logprob':'natural log full lexical-vocabulary probability including blank; separate duration excluded',
            'stored_word_confidence':'exp(mean of contributing lexical-token logprob_by_temperature[1])',
            'token_minimum':'minimum contributing token probability; word-level minimum not the stored word score',
            'token_product':'exp(sum raw logprobs); penalizes more-token words; not posterior word correctness or TDT sequence marginal',
            'correctness':'existing minimum-edit alignment on current normalized English text; deletions have no emitted-word confidence'},
        'limits':['No user audio or new inference; original fixed read-speech50 DEV sample only',
            'Different vocabularies/segmentation: matched reference words, not matched token IDs or equal-length products',
            'Raw geometric mean, minimum and product are distinct scores; high word geomean can hide a low component',
            'ECE/Brier reported only descriptively for existing raw geometric-word score and emitted correctness; no temperature fit/apply',
            'Blank/duration paths and alternative alignments are not multiplied into the lexical product',
            'No claim about current training cause, final winner confidence or broader calibration'],
        'actual_new_inference':False,'TEST_predictions_read_or_scored':False,'temperature_fitted_or_applied':False,
        'Torch_Transformers_NeMo_imported':any(name in sys.modules for name in ('torch','transformers','nemo'))}
    require(report['Torch_Transformers_NeMo_imported'] is False,'ML module unexpectedly imported')
    destination=HERE/'English_base_v2_v3_saved_DEV_token_word_audit_v1.json'
    with destination.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
    print(json.dumps({'path':str(destination),'sha256':record(destination)['sha256'],'shared_correct_words':len(common),
        'versions':{v:{'correct_words':d['correct_words'],'correct_geo_mean':d['correct_word_scores']['word_geometric_mean_probability']['mean'],
        'correct_token_min_below0_5':d['correct_word_scores']['minimum_token_probability']['below_0_5_count'],
        'correct_geo_below0_5':d['correct_word_scores']['word_geometric_mean_probability']['below_0_5_count'],
        'named_examples':d['named_correct_examples']} for v,d in versions.items()}},ensure_ascii=False))

if __name__=='__main__':main()
