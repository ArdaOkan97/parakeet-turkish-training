"""Offline listening report from saved fixed DEV only. No ML imports or new inference."""
import base64, collections, datetime, hashlib, json, pathlib, re, sys, unicodedata, wave

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]/'work/training-env/Lib/site-packages'))
sys.path.append(r'C:\Users\ardao\Miniconda3\envs\vlm\Lib\site-packages')
from jiwer import process_words

ROOT = pathlib.Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs/parakeet_turkish'
OUT = BASE / 'reports'
RUN = 'adaptive_20261014_TR_WorldSpeech_expanded_01'
STEP = 62250
SOURCE_RUN = 'adaptive_20261014_TR_uniform_token_population_01'
SOURCE_STEP = 40750
CURATED = [10, 29, 35, 15, 20, 47, 3, 43, 0, 31, 2, 33, 22, 5, 24, 14, 17, 36, 37, 9]
NOTES = {
 0: (['Numbers and units', 'Foreign names', 'Word boundaries'], '4892 m’lik is spoken as “dört bin sekiz yüz doksan iki metrelik”; that part is a rendering mismatch. “Vinson” becoming “Vin Son” remains a name/boundary error.'),
 2: (['Foreign names', 'Numbers and acronyms'], 'Arly Velasquez, Adam Hall and Super-G are distorted. The reference also uses ordinal digits while the model spells them out. The second recording of this same sentence is clip 20.'),
 3: (['Suffixes', 'Phonetic substitutions'], '“kaynaklara → kanaklara”, “nüfus → nüfuz” and “özelliklere → özellikleri” show recognition and suffix errors even in an otherwise understandable sentence.'),
 5: (['Severe recognition errors', 'Word boundaries'], '“İspanyollar → İspanya olar”, “asır → asıl” and “süren → üçen”. Six errors on a seven-word reference make this the highest utterance WER; it is not the largest absolute error count.'),
 9: (['Reference quality', 'Word boundaries'], 'The reference writes “bozdura bilirsiniz”; the model joins it as “bozdurabilirsiniz”. That reference-side boundary inflates WER. Other substitutions in this clip remain.'),
 10: (['Exact match'], 'Zero word errors after the frozen normalization. Case and punctuation differences do not count.'),
 12: (['Foreign names', 'Numbers'], '“Casa Branca → kaza bıranka”. The year 1755 is also distorted, beyond a simple digit-versus-word rendering difference.'),
 13: (['Foreign names', 'Numbers'], 'Several French names and the date are garbled. The date errors are not all explained by writing numbers as words.'),
 14: (['Reference format', 'Diacritics'], 'The only scored difference is “Hükumet → Hükûmet”. The metric preserves accents, so this spelling variant alone yields 10% WER.'),
 15: (['Near-correct', 'Phonetic substitutions'], 'Only two substitutions in a 26-word reference, including “Kurye → Kurya” and “dökümanların → dokumenların”.'),
 17: (['Foreign names', 'Numbers'], 'Shyam and Chhappan Bhog are distorted. 108 is not transcribed correctly; the correctly spoken “elli altı” for 56 is a separate rendering issue.'),
 19: (['Foreign names', 'Numbers and acronyms'], 'A second recording of the same sports sentence as clip 3. Names and Super-G remain difficult even though the exact errors differ.'),
 20: (['Near-correct', 'Diacritics'], 'The single scored substitution is “virüs → virus”.'),
 22: (['Severe recognition errors', 'Word boundaries'], '“memeliler → memeriller”, “kaplumbağayı → kaplumbayı” and the corrupted ending are genuine transcript failures, beyond formatting.'),
 24: (['Reference format', 'Word boundaries'], '“Ortadoğu’ya → Orta Doğu’ya” is the only difference. One reference word becomes two output words, accounting for both scored errors.'),
 27: (['Turkish casing'], '“İlgi → Ilgi” changes dotted İ to dotless I. Turkish-aware lowercasing preserves that distinction, so this is scored as a word error.'),
 29: (['Exact match'], 'Zero normalized word errors, including George W. Bush. Foreign names are not uniformly wrong.'),
 31: (['Numbers', 'Phonetic substitutions'], '2010 and 1.400 are correctly verbalized as words, while the 2008/year and final percentage phrase contain recognition errors. Both phenomena occur in the same clip.'),
 33: (['Foreign names', 'Acronyms', 'Word boundaries'], '“CEO’su → Siyosu”, “Siminoff → Siminov” and “kapı zilinin → kapızilinin”. The second recording is clip 47.'),
 35: (['Exact match'], 'Zero normalized word errors. Punctuation changes are ignored by the official scorer.'),
 36: (['Suffixes', 'Phonetic substitutions'], '“donma noktasının → dolman oktasının”, “nem → Nen” and “hissetmenize → hissetmenizi”. Compare clip 38 for the same text in a shorter recording.'),
 37: (['Suffixes', 'Phonetic substitutions'], 'The same reference as clip 37, but a different recording and different errors. Recording duration alone does not establish why one decode is worse.'),
 43: (['Suffixes', 'Phonetic substitutions'], '“çevresine → çevresini” and “popülasyondan → popülasyonlan”. The çevresine/çevresini substitution repeats in clip 50.'),
 44: (['Foreign names', 'Numbers'], 'The same reference as clip 18, with another recording. The foreign terms and 108 remain difficult.'),
 46: (['Foreign names', 'Acronyms'], 'The second Ring/CEO/Siminoff recording. The same proper-name/acronym problems recur.'),
 47: (['Near-correct', 'Phonetic substitutions'], 'The only scored substitution is “uyduyu → uylüyü”.'),
 49: (['Suffixes', 'Phonetic substitutions'], 'The same text as clip 44. “çevresine → çevresini” recurs, alongside further phonetic substitutions.'),
}

def read_jsonl(path):
    with path.open(encoding='utf-8-sig') as f:
        return [json.loads(x) for x in f if x.strip()]

def norm(text):
    text = unicodedata.normalize('NFC', text).replace('I', 'ı').replace('İ', 'i').lower()
    return ' '.join(''.join(c for c in text if not unicodedata.category(c).startswith('P')).split())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def marked_tokens(text, operations):
    pieces, offset = [], 0
    for match in re.finditer(r'\s+|\S+', text):
        raw = match.group()
        value = norm(raw)
        op = 'equal'
        if value:
            if offset >= len(operations):
                raise ValueError('Raw text has excess scored tokens')
            op = operations[offset]
            offset += 1
        pieces.append({'text':raw, 'op':op})
    if offset != len(operations):
        raise ValueError('Raw token mapping mismatch')
    return pieces

def main():
    OUT.mkdir(exist_ok=True)
    selected = BASE/'results'/f'{RUN}_step_{STEP}'
    entry = BASE/'results'/f'{SOURCE_RUN}_step_{SOURCE_STEP}'
    predictions_path = selected/'validation_tr.jsonl'
    predictions = read_jsonl(predictions_path)
    old = {x['source_id']:x for x in read_jsonl(entry/'validation_tr.jsonl')}
    manifest_path = ROOT/'work/data/validation_tr.jsonl'
    manifest = {x['source_id']:x for x in read_jsonl(manifest_path)}
    expected = json.loads((selected/'summary.json').read_text())['validation_tr']['asr']
    assert len(predictions) == 50 == expected['utterances']
    examples, audio_bytes = [], 0
    sums = collections.Counter()
    for index, p in enumerate(predictions):
        m, prior = manifest[p['source_id']], old[p['source_id']]
        assert p['reference'] == m['reference'] == prior['reference']
        assert p['wav_sha256'] == m['wav_sha256'] == prior['wav_sha256']
        ref_words, hyp_words = norm(p['reference']).split(), norm(p['hypothesis']).split()
        alignment = p['alignment']
        assert len(ref_words) == alignment['reference_word_count']
        assert [w['word'] for w in alignment['words']] == hyp_words
        rebuilt = process_words(' '.join(ref_words),' '.join(hyp_words))
        ref_ops, hyp_ops = ['unassigned']*len(ref_words), ['unassigned']*len(hyp_words)
        for chunk in rebuilt.alignments[0]:
            for ri in range(chunk.ref_start_idx,chunk.ref_end_idx): ref_ops[ri] = chunk.type
            for hi in range(chunk.hyp_start_idx,chunk.hyp_end_idx): hyp_ops[hi] = chunk.type
        assert hyp_ops == [w['operation'] for w in alignment['words']]
        assert all(x != 'unassigned' for x in ref_ops+hyp_ops)
        for key, op in [('substitutions','substitute'),('insertions','insert')]:
            assert hyp_ops.count(op) == alignment[key]
        assert ref_ops.count('delete') == alignment['deletions']
        errors = alignment['substitutions']+alignment['deletions']+alignment['insertions']
        assert abs(errors/len(ref_words)-alignment['wer']) < 1e-12
        audio_path = pathlib.Path(m['audio_filepath'])
        raw_audio = audio_path.read_bytes()
        assert hashlib.sha256(raw_audio).hexdigest() == p['wav_sha256']
        with wave.open(str(audio_path),'rb') as audio:
            assert (audio.getnchannels(),audio.getframerate(),audio.getsampwidth()) == (1,16000,2)
            duration = audio.getnframes()/audio.getframerate()
        assert abs(duration-p['duration']) < .001
        audio_bytes += len(raw_audio)
        tags, note = NOTES.get(index, (['Additional DEV example'], 'Inspect the highlighted word differences and listen before assigning an acoustic cause.'))
        old_a = prior['alignment']
        prior_errors = old_a['substitutions']+old_a['deletions']+old_a['insertions']
        example = {
            'index':index, 'clip':index+1, 'sentence_id':p['sentence_id'],
            'source_id':p['source_id'], 'duration':duration,
            'reference':p['reference'], 'prediction':p['hypothesis'],
            'reference_marked':marked_tokens(p['reference'],ref_ops),
            'prediction_marked':marked_tokens(p['hypothesis'],hyp_ops),
            'entry_prediction':prior['hypothesis'], 'entry_errors':prior_errors,
            'words':len(ref_words),'errors':errors,'wer':alignment['wer'],
            'S':alignment['substitutions'],'D':alignment['deletions'],'I':alignment['insertions'],
            'tags':tags,'note':note, 'curated':index in CURATED,
            'curated_rank':CURATED.index(index) if index in CURATED else 100+index,
            'group':'exact' if errors==0 else 'low' if alignment['wer']<=.15 else 'review',
            'audio_sha256':p['audio_sha256'], 'wav_sha256':p['wav_sha256'],
            'audio_source_path':str(audio_path),
            'audio':'data:audio/wav;base64,'+base64.b64encode(raw_audio).decode('ascii'),
        }
        examples.append(example)
        sums.update({'S':example['S'],'D':example['D'],'I':example['I'],'errors':errors,'words':len(ref_words)})
    assert sums['errors'] == expected['word_errors'] == 259
    assert sums['words'] == expected['reference_words'] == 825
    assert set(x['source_id'] for x in predictions) == set(old) == set(manifest)
    source_total = sum(x['entry_errors'] for x in examples)
    assert source_total == 277
    normalizer_path = OUT/'turkish-normalization-evaluation.json'
    normalizer = json.loads(normalizer_path.read_text(encoding='utf-8'))
    normalized_method = normalizer['normalizer']['selected']
    n_current = normalizer['systems']['Parakeet expanded qualified 62250'][normalized_method]
    n_entry = normalizer['systems']['Parakeet entry 40750'][normalized_method]
    for x, n, prior_n in zip(examples,n_current['clips'],n_entry['clips']):
        assert x['source_id'] == n['source_id'] == prior_n['source_id']
        x['normalized'] = n
        x['entry_normalized'] = prior_n
    duplicate_text = collections.defaultdict(list)
    for x in examples: duplicate_text[norm(x['reference'])].append(x['clip'])
    numeric = [x for x in examples if re.search(r'\d',x['reference'])]
    wrong_conf = [w['confidence'] for p in predictions for w in p['alignment']['words'] if not w['correct']]
    stats = {
        **sums,'wer':expected['wer'],'cer':expected['cer'],'clips':50,
        'unique_reference_texts':len(duplicate_text),'duration_seconds':sum(x['duration'] for x in examples),
        'exact':sum(x['errors']==0 for x in examples),
        'low_error':sum(0<x['wer']<=.15 for x in examples),
        'improved_clips':sum(x['errors']<x['entry_errors'] for x in examples),
        'worsened_clips':sum(x['errors']>x['entry_errors'] for x in examples),
        'unchanged_clips':sum(x['errors']==x['entry_errors'] for x in examples),
        'source_word_errors':source_total,'source_wer':source_total/sums['words'],
        'numeric_clips':len(numeric),'numeric_clip_errors':sum(x['errors'] for x in numeric),
        'numeric_clip_reference_words':sum(x['words'] for x in numeric),
        'duplicate_reference_groups':[v for v in duplicate_text.values() if len(v)>1],
        'wrong_emitted_words':len(wrong_conf),'wrong_emitted_words_raw_conf_ge_08':sum(x>=.8 for x in wrong_conf),
    }
    snapshot = json.loads((BASE/'experiment_queue_state.json').read_text(encoding='utf-8-sig'))
    scale = {
        'target_accepted_unique_TR_TRAIN_hours':10000,
        'current_accepted_expanded_pool_hours':675.0964019097222,
        'additional_accepted_hours_needed':10000-675.0964019097222,
        'current_base_pool_hours':266.8543265625,
        'PCM16_16k_mono_target_bytes':10000*115200000,
        'YODAS3_raw_Turkish_recording_hours':9216.530555555555,
        'earlier_YODAS_published_Turkish_hours':4656.378,
        'Alania_selected_hours_subset_of_YODAS3':2398,
        'not_added_to_running_trial':True,
    }
    payload = {
        'title':'Turkish ASR listening review','run_id':RUN,'step':STEP,
        'entry_run_id':SOURCE_RUN,'entry_step':SOURCE_STEP,
        'scope':'Fixed Turkish DEV only; all 50 clips embedded. No TEST or new inference.',
        'created_at_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'normalization':expected['normalization'],'stats':stats,'scale':scale,
        'spoken_form':{'method':normalized_method,'current':n_current['aggregate'],
            'entry':n_entry['aggregate'],
            'whisper':normalizer['systems']['Whisper large-v3 greedy'][normalized_method]['aggregate'],
            'whisper_frozen':normalizer['systems']['Whisper large-v3 greedy']['frozen']['aggregate'],
            'normalizer':normalizer['normalizer'],'checks':normalizer['checks']},
        'status_snapshot':{k:snapshot.get(k) for k in ['updated_at_utc','current_run_id','published_step','actual_data_expansion_updates','data_arm','status']},
        'examples':examples,
    }
    template = (ROOT/'work/research/turkish_dev_listening_template.html').read_text(encoding='utf-8')
    assert template.count('<!-- REPORT_DATA -->') == 1
    embedded = json.dumps(payload,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
    output = OUT/'turkish-dev-listening.html'
    output.write_text(template.replace('<!-- REPORT_DATA -->',embedded),encoding='utf-8')
    evidence = {k:v for k,v in payload.items() if k!='examples'}
    evidence['examples'] = [{k:v for k,v in x.items() if k!='audio'} for x in examples]
    evidence['inputs'] = {str(p.relative_to(ROOT)):sha(p) for p in [predictions_path,entry/'validation_tr.jsonl',manifest_path,selected/'summary.json',normalizer_path]}
    evidence['verification'] = {'all_50_WAV_hashes_match_saved_predictions':True,'PCM_geometry_and_durations_match':True,'official_all50_SDI_WER_matches':True,'entry_same50_audio_and_references':True,'all_50_playable_WAVs_embedded':True,'TEST_read':False,'Torch_imported':'torch' in sys.modules,'model_or_checkpoint_loaded':False,'audio_payload_bytes':audio_bytes,'HTML_bytes':output.stat().st_size,'HTML_sha256':sha(output)}
    (OUT/'turkish-dev-listening-evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'report':str(output),'verification':evidence['verification'],'stats':stats,'scale':scale},ensure_ascii=False),flush=True)

if __name__ == '__main__': main()
