"""Bounded text-only analysis of saved DEV and TRAIN exposure; no model/audio/TEST."""
import collections
import hashlib
import json
import statistics
import sys
from pathlib import Path

from analyze_turkish_dev_errors_v1 import alignment, boundary_oracle, FOLD, top
from metrics import edit_distance, normalize_text, score_transcriptions

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'outputs/parakeet_turkish'
RUN = OUT/'runs/adaptive_20261011_hf_return_01'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    report={'schema':1,'kind':'HF1000_Turkish_DEV_text_error_and_exposure_review_v1',
        'scope':'Saved50 fixed Turkish DEV clips; accepted TRAIN transcripts and recorded sampled IDs only. No new inference, audio, checkpoints or TEST.',
        'models':{},'required_competitive_DEV_wer':.1193939393939394}
    model_rows={}
    identities=[]
    for label,step in [('qualifying_best',0),('latest_unqualified',1000)]:
        folder=OUT/f'results/adaptive_20261011_hf_return_01_step_{step}'
        path=folder/'validation_tr.jsonl'
        rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        model_rows[label]=rows
        identity=[(r['source_id'],r['reference'],r['audio_sha256']) for r in rows]
        if not identities:identities=identity
        assert identity==identities and len(rows)==50
        score=score_transcriptions(rows,language='tr')
        stored=read(folder/'summary.json')['validation_tr']['asr']
        assert all(score[k]==stored[k] for k in ('wer','cer','word_errors','reference_words','character_errors','reference_characters'))
        kinds=collections.Counter();ckinds=collections.Counter();subs=collections.Counter();csubs=collections.Counter()
        letters={ch:collections.Counter() for ch in 'ğış'}
        boundary=folded_words=folded_chars=no_space_chars=0
        examples=[]
        new_tokens=collections.Counter();new_conf=[]
        for row,detail in zip(rows,score['details']):
            r,h=detail['reference_normalized'],detail['hypothesis_normalized']
            for op,a,b in alignment(r.split(),h.split()):
                if op!='equal':kinds[op]+=1
                if op=='sub':subs[a,b]+=1
            for op,a,b in alignment(r,h):
                if op!='equal':ckinds[op]+=1
                if op=='sub':
                    csubs[a,b]+=1
                    if a.translate(FOLD)==b.translate(FOLD):ckinds['diacritic_only_substitutions']+=1
                if op!='equal' and ' ' in (a,b):ckinds['space_operations']+=1
                for ch in letters:
                    if a==ch:letters[ch]['reference_occurrences']+=1;letters[ch]['correctly_aligned' if op=='equal' else op]+=1
                    if b==ch:letters[ch]['emitted_occurrences']+=1
            boundary+=boundary_oracle(r.split(),h.split())
            folded_words+=edit_distance(r.translate(FOLD).split(),h.translate(FOLD).split())
            folded_chars+=edit_distance(r.translate(FOLD),h.translate(FOLD))
            no_space_chars+=edit_distance(r.replace(' ',''),h.replace(' ',''))
            for token in row['confidence']['tokens']:
                if 8193<=token['token_id']<8199:
                    new_tokens[token['piece']]+=1
                    new_conf.append(token['logprob'])
            examples.append({'sentence_id':row['sentence_id'],'reference':row['reference'],
                'hypothesis':row['hypothesis'],'word_errors':detail['word_errors'],
                'reference_words':detail['reference_words']})
        numeric=[r for r in rows if any(ch.isdigit() for ch in r['reference'])]
        nonnumeric=[r for r in rows if not any(ch.isdigit() for ch in r['reference'])]
        nd=score_transcriptions(nonnumeric,language='tr');nd.pop('details')
        numeric_score=score_transcriptions(numeric,language='tr');numeric_score.pop('details')
        score.pop('details')
        report['models'][label]={'metrics':score,'word_edit_counts_one_minimum_traceback':dict(kinds),
            'character_edit_counts_one_minimum_traceback':dict(ckinds),
            'diagnostic_only':{'boundary_oracle_wer':boundary/score['reference_words'],
                'boundary_forgiven_edits':score['word_errors']-boundary,
                'folded_wer':folded_words/score['reference_words'],'folded_cer':folded_chars/score['reference_characters'],
                'no_space_character_errors':no_space_chars},
            'new_letter_alignment':{k:dict(v) for k,v in letters.items()},
            'new6_emitted_token_pieces':dict(new_tokens),'new6_emitted_token_count':sum(new_tokens.values()),
            'new6_emitted_token_mean_logprob':statistics.mean(new_conf) if new_conf else None,
            'empty_hypotheses':sum(not normalize_text(r['hypothesis'],'tr') for r in rows),
            'nonnumeric_reference_subset':nd,'numeric_reference_subset':numeric_score,
            'top_word_substitutions':top(subs,20),'top_character_substitutions':top(csubs,12),
            'representative_examples':examples[:3],'predictions_sha256':sha(path),'summary_sha256':sha(folder/'summary.json')}
    pairs=[]
    for old,new in zip(model_rows['qualifying_best'],model_rows['latest_unqualified']):
        a,b=score_transcriptions([old],language='tr'),score_transcriptions([new],language='tr')
        pairs.append({'sentence_id':old['sentence_id'],'word_error_delta':b['word_errors']-a['word_errors']})
    report['paired_DEV_change']={'improved_clips':sum(r['word_error_delta']<0 for r in pairs),
        'worsened_clips':sum(r['word_error_delta']>0 for r in pairs),'unchanged_clips':sum(r['word_error_delta']==0 for r in pairs),
        'net_word_errors_delta':sum(r['word_error_delta'] for r in pairs),'per_sentence':pairs}
    history=read(RUN/'training_history.json');assert len(history)==1000
    seen={sid for row in history for sid in row['sampled_source_ids'] if not sid.startswith('google/fleurs:') or ':tr_tr:train:' in sid}
    # Exactly TR pool IDs are resolved below; replay IDs do not enter target coverage.
    dev_words=collections.Counter(word for r in model_rows['qualifying_best'] for word in normalize_text(r['reference'],'tr').split())
    presence={name:set() for name in ('all_accepted','FLEURS_accepted','CommonVoice_accepted','seen_TR','seen_FLEURS','seen_CommonVoice')}
    pool=collections.Counter();sampled=collections.Counter();audio=collections.Counter();sampled_audio=collections.Counter()
    path=ROOT/'work/data/tr_public_v3_train.jsonl'
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line);domain='FLEURS' if row['dataset']=='google/fleurs' else 'CommonVoice'
            words=set(normalize_text(row['text'],'tr').split()) & dev_words.keys()
            pool[domain]+=1;audio[domain]+=row['duration']
            presence['all_accepted'].update(words);presence[domain+'_accepted'].update(words)
            if row['source_id'] in seen:
                sampled[domain]+=1;sampled_audio[domain]+=row['duration']
                presence['seen_TR'].update(words);presence['seen_'+domain].update(words)
    assert sum(sampled.values())==8000 and sum(pool.values())==74883
    coverage={name:{'matched_unique_DEV_words':len(words),'DEV_word_occurrences_covered':sum(dev_words[w] for w in words)} for name,words in presence.items()}
    latest=model_rows['latest_unqualified'];novel=collections.Counter();known=collections.Counter()
    for row in latest:
        for op,a,b in alignment(normalize_text(row['reference'],'tr').split(),normalize_text(row['hypothesis'],'tr').split()):
            if not a:continue
            (known if a in presence['seen_TR'] else novel)['correct' if op=='equal' else 'substituted_or_deleted']+=1
    report['TRAIN_exposure']={'pool_clips':dict(pool),'pool_audio_hours':{k:v/3600 for k,v in audio.items()},
        'unique_sampled_TR_clips':dict(sampled),'unique_sampled_TR_audio_hours':{k:v/3600 for k,v in sampled_audio.items()},
        'unique_TR_clips_seen':sum(sampled.values()),'TR_pool_unique_coverage':sum(sampled.values())/sum(pool.values()),
        'recorded_turkish_epoch':history[-1]['turkish_epoch'],'TR_audio_seconds':history[-1]['turkish_audio_seen_seconds'],
        'replay_audio_seconds':history[-1]['replay_audio_seen_seconds'],'TRAIN_manifest_sha256':sha(path),
        'DEV_lexical_presence_diagnostic':coverage,'DEV_unique_reference_words':len(dev_words),
        'latest_ref_word_outcomes_seen_in_sampled_TRAIN':dict(known),
        'latest_ref_word_outcomes_not_seen_in_sampled_TRAIN':dict(novel)}
    report['observed_loss_trend']={k:{'median_first100':statistics.median(r[k] for r in history[:100]),
        'median_last100':statistics.median(r[k] for r in history[-100:])} for k in ('loss','tdt_loss','replay_kl')}
    report['limits']=['50 reused DEV clips/825 reference words; diagnostic folding/boundary oracles do not replace official scores.',
        'Minimum-edit confusion categories depend on tie-breaking and overlap.',
        'Lexical presence in transcripts does not measure acoustical coverage or demonstrate memorization/generalization.',
        'Observed training loss mixes languages and variable examples; its trend is not a controlled Turkish quality measurement.',
        'No complete Turkish pass has occurred; tokenizer acceptance proof was not rerun and no GPU/model was loaded.']
    assert 'torch' not in sys.modules and 'nemo' not in sys.modules
    dest=OUT/'HF1000_Turkish_DEV_error_evidence_v1.json'
    with dest.open('x',encoding='utf-8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
    print(json.dumps({'path':str(dest),'models':{k:{'metrics':v['metrics'],'word_edits':v['word_edit_counts_one_minimum_traceback'],'diagnostics':v['diagnostic_only'],'letters':v['new_letter_alignment'],'new_tokens':v['new6_emitted_token_pieces'],'nonnumeric':v['nonnumeric_reference_subset']} for k,v in report['models'].items()},'TRAIN_exposure':report['TRAIN_exposure'],'loss':report['observed_loss_trend'],'paired':{k:v for k,v in report['paired_DEV_change'].items() if k!='per_sentence'}},ensure_ascii=False))


if __name__=='__main__':main()
