"""After actual experimental worker closure, publish a TRAIN-supported projection.

No refit or new text selection. An unused manually seeded atom is omitted.
One full encoding-equivalence validation pass; no models/audio/GPU/DEV/TEST.
"""
from pathlib import Path
import collections,hashlib,importlib.util,json,os,shutil,sys,time,zipfile
AREA=Path(r'E:\gpu_training_scratchpad\analysis\Turkish_labels_BPE_circumflex_v1')
EXPERIMENT=AREA/'all_TRAIN_extension_v5'
sys.path.insert(0,str(AREA))
s=importlib.util.spec_from_file_location('bounded_audit',AREA/'audit_and_fit_append_only_v5.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def main():
    proof=json.loads((EXPERIMENT/'completed_handoff.json').read_text(encoding='utf8'))
    closure=json.loads((EXPERIMENT/'actual_owned_CLOSED0.json').read_text(encoding='utf8'))
    assert proof['passed'] and closure['actual_exit_code']==0 and closure['all_captured_identities_dead']
    audit=json.loads((EXPERIMENT/'source_label_audit.json').read_text(encoding='utf8'))
    validation=json.loads((EXPERIMENT/'all_TRAIN_candidate_validation.json').read_text(encoding='utf8'))
    normalized_U=sum(group['normalized_char_counts'].get('Û',0) for group in audit['groups'].values())
    literal_U=sum(group['source_char_counts'].get('Û',0) for group in audit['groups'].values())
    remove=normalized_U==0
    base=json.loads(m.INCUMBENT.read_text(encoding='utf8'));experimental=json.loads((EXPERIMENT/'tokenizer.json').read_text(encoding='utf8'))
    assert m.sha(m.INCUMBENT)==m.INCUMBENT_SHA
    assert len(experimental['model']['vocab'])==9224 and experimental['model']['vocab']['Û']==8711
    learned=json.loads((EXPERIMENT/'added_pieces.json').read_text(encoding='utf8'))[1:];assert len(learned)==512
    assert all(x['weighted_pair_frequency']>0 for x in learned)
    if remove:assert all('Û' not in x['piece'] and 'Û' not in x['left'] and 'Û' not in x['right'] for x in learned)
    def remap(i):
        if remove:
            assert i!=8711,'Unobserved seededÛ appeared in real encoded labels'
            return i-1 if i>8711 else i
        return i
    candidate=json.loads(json.dumps(experimental))
    candidate['model']['vocab']={piece:remap(identifier) for piece,identifier in experimental['model']['vocab'].items() if not(remove and identifier==8711)}
    size=len(candidate['model']['vocab']);assert size==(9223 if remove else 9224)
    for piece,identifier in base['model']['vocab'].items():assert candidate['model']['vocab'][piece]==identifier
    assert candidate['model']['merges']==experimental['model']['merges']
    assert candidate['model']['merges'][:13988]==base['model']['merges']
    for k in base:
        if k!='model':assert candidate[k]==base[k]
    folder=AREA/('corpus_driven_BPE%d_v1'%size);assert not folder.exists();folder.mkdir();m.AREA=folder
    g=m.Guard();g.max_seconds=1800;g.maximum_bytes=8000000000;g.phase='full_TRAIN_projection_encoding_equivalence'
    from tokenizers import Tokenizer
    import tokenizers
    old=Tokenizer.from_str(json.dumps(base,ensure_ascii=False));reference=Tokenizer.from_str(json.dumps(experimental,ensure_ascii=False))
    updated=Tokenizer.from_str(json.dumps(candidate,ensure_ascii=False))
    m.write('tokenizer.json',candidate)
    additions=[]
    for record in learned:
        x=dict(record)
        for k in ('id','left_id','right_id'):x[k]=remap(x[k])
        additions.append(x)
    m.write('added_pieces.json',additions)
    m.write('vocab.json',candidate['model']['vocab'])
    (folder/'vocab.tsv').write_text('\n'.join(str(i)+'\t'+p for p,i in sorted(candidate['model']['vocab'].items(),key=lambda x:x[1]))+'\n',encoding='utf8')
    (folder/'merges.txt').write_text('#version: 0.2\n'+'\n'.join(' '.join(p) for p in candidate['model']['merges'])+'\n',encoding='utf8')
    (folder/'tokenizer_config.json').write_bytes((m.PROJECT/'work/models/lora_20261010_01_merged/tokenizer_config.json').read_bytes())
    m.write('projection_recipe_frozen_before_validation.json',{'at_utc':m.now(),'incumbent':m.record(m.INCUMBENT),'experimental_source':m.record(EXPERIMENT/'tokenizer.json'),
        'experimental_actual_closure':m.record(EXPERIMENT/'actual_owned_CLOSED0.json'),'source_label_audit':m.record(EXPERIMENT/'source_label_audit.json'),
        'candidate':m.record(folder/'tokenizer.json'),'new_vocab_size':size,'existing8711_IDs_fixed':True,'new_merges':512,
        'raw_TRAIN_Ucircumflex_count':literal_U,'normalized_TRAIN_Ucircumflex_count':normalized_U,
        'manually_seeded_unobserved_Ucircumflex_removed':remove,'only_appended_IDs_remapped':remove,'all_merge_strings_ranks_frequencies_unchanged':True,
        'new_chars_require_actual_normalized_TRAIN_support':True,'no_case_augmentation_or_transcript_correction':True,'no_refit_or_new_count_audit':True,
        'incumbent_byte_fallback_flag':base['model']['byte_fallback'],'incumbent_actual_byte_fallback_vocab_tokens':sum(p.startswith('<0x') for p in base['model']['vocab']),
        'scope':'Tokenizer-only experimental candidate, not an active decoder, training launch or measured ASR gain.'})
    reference_digest=hashlib.sha256();projection_digest=hashlib.sha256();groups={};uses=collections.Counter()
    def check_rows(pool,rows):
        batch=[]
        def consume(values):
            enc_a=reference.encode_batch([x[3] for x in values],add_special_tokens=True)
            enc_b=updated.encode_batch([x[3] for x in values],add_special_tokens=True)
            for (ordinal,sid,dataset,text),a,b in zip(values,enc_a,enc_b):
                expected=[remap(i) for i in a.ids];assert b.ids==expected,(pool,ordinal,'ID projection differs')
                assert reference.decode(a.ids,skip_special_tokens=False)==updated.decode(b.ids,skip_special_tokens=False)
                assert a.ids.count(0)==b.ids.count(0)
                reference_digest.update((m.canonical([pool,ordinal,sid,a.ids])+'\n').encode());projection_digest.update((m.canonical([pool,ordinal,sid,b.ids])+'\n').encode())
                key=pool+':'+dataset;v=groups.setdefault(key,{'rows':0,'tokens':0,'unknown_rows':0,'new_piece_occurrences':0})
                v['rows']+=1;v['tokens']+=len(b.ids);v['unknown_rows']+=0 in b.ids;v['new_piece_occurrences']+=sum(i>=8711 for i in b.ids)
                uses.update(i for i in b.ids if i>=8711)
                g.pace(len(text.encode())+len(a.ids)*8+128)
        for row in rows:
            batch.append(row)
            if len(batch)==256:consume(batch);batch.clear()
        if batch:consume(batch)
    source=m.ro(m.BROAD)
    check_rows('broad',source.execute('SELECT ordinal,unique_id,dataset,text FROM clips ORDER BY ordinal'));source.close()
    def cleaner():
        with m.CLEAN.open('rb') as f:
            for ordinal,line in enumerate(f,1):
                g.pace(len(line));row=json.loads(line);yield ordinal,row['source_id'],row['dataset'],row['text']
    check_rows('cleaner',cleaner())
    assert reference_digest.hexdigest()==validation['ordered_candidate_labels_sha256']
    assert sum(x['rows'] for x in groups.values())==2983883
    for key,value in groups.items():
        prior=validation['groups'][key];assert value['rows']==prior['rows'] and value['tokens']==prior['candidate_tokens']
        value['incumbent_tokens']=prior['incumbent_tokens'];value['token_reduction_fraction']=1-value['tokens']/value['incumbent_tokens']
    pool=m.PROJECT/'outputs/parakeet_turkish/runs/adaptive_20261014_TR_WorldSpeech_expanded_01/training_pool.json'
    assert m.sha(pool)=='00b839c717762aba74326e9867969d85b7ffaaabd7ead7ddbe732fe6be8c6cf4'
    replay=json.loads(pool.read_text(encoding='utf8'))['replay'];assert len(replay)==1914
    retained={};examples=[]
    for row in replay:
        assert row['dataset_split']=='train';a=old.encode(row['text']);b=updated.encode(row['text']);lang=row['language'];assert lang in ('en','es','it','fr','de','pt')
        v=retained.setdefault(lang,{'rows':0,'incumbent_tokens':0,'candidate_tokens':0,'encoding_changes':0,'decode_changes':0,'incumbent_unknown_rows':0,'candidate_unknown_rows':0})
        v['rows']+=1;v['incumbent_tokens']+=len(a.ids);v['candidate_tokens']+=len(b.ids);v['encoding_changes']+=a.ids!=b.ids
        v['decode_changes']+=old.decode(a.ids,skip_special_tokens=False)!=updated.decode(b.ids,skip_special_tokens=False)
        v['incumbent_unknown_rows']+=0 in a.ids;v['candidate_unknown_rows']+=0 in b.ids
        if a.ids!=b.ids and len(examples)<12:examples.append({'language':lang,'text':row['text'],'incumbent_tokens':a.tokens,'candidate_tokens':b.tokens})
    result={'kind':'Corpus_driven_TRAIN_BPE_projection_actual_v1','at_utc':m.now(),'passed':True,'PID':os.getpid(),'vocabulary':size,
        'all_TRAIN_rows_validated':2983883,'reference_full_TRAIN_encoding_digest_exact':True,'exact_reference_encoding_SHA':reference_digest.hexdigest(),
        'exact_projected_encoding_SHA':projection_digest.hexdigest(),'all_TRAIN_token_sequences_exact_up_to_only_new_ID_projection':True,
        'groups':groups,'appended_piece_types_directly_emitted':len(uses),'appended_piece_token_occurrences':sum(uses.values()),
        'six_language_original1914_TRAIN_replay':retained,'replay_encoding_change_examples':examples,
        'original_all8711_vocab_IDs_and_merges_preserved':True,'new_pieces_learned_from_actual_TRAIN_pair_frequencies':512,
        'forced_unobserved_character_added':False,'normalized_Ucircumflex_count':normalized_U,
        'no_new_models_or_GPU_or_DEV_TEST_or_source_mutations':True,'candidate_activated':False,'measured_ASR_quality_gain':False,
        'elapsed_seconds':time.monotonic()-g.started,'peak_private_bytes':g.peak,'tokenizers_version':tokenizers.__version__}
    m.write('full_TRAIN_projection_validation.json',result)
    m.write('unicode_roundtrip_tests.json',m.fixture(updated))
    m.write('row_migration_plan.json',{'tokenizer_candidate_only':True,'old_vocabulary':8711,'new_vocabulary':size,'blank_id_unchanged':8192,
        'all_existing_8711_rows_keep_IDs':True,'appended_rows_half_open':[8711,size],'TDT_duration_rows_old_half_open':[8711,8716],
        'TDT_duration_rows_new_half_open':[size,size+5],'new_joint_width':size+5,'active_model_or_optimizer_migrated':False,
        'explicit_new_row_initialization_and_optimizer_extension_required':True,'preserve_mature_moments_clocks_RNG_source_history':True,
        'new_TRAIN_label_catalogs_full_token_means_geometry_and_retention_validation_required':True,
        'protected_replay_labels_teacher_DEV_TEST_must_not_be_silently_retokenized':True,'exact_resume_of_8711_decoder':False})
    text=('Corpus-driven Turkish BPE candidate, %d tokens.\n\n'%size+
        'Use: from tokenizers import Tokenizer; t=Tokenizer.from_file("tokenizer.json")\n'
        'Learned512 append-only Latin BPE merges from every retained broad+clean TRAIN text occurrence (2,983,883 clips).\n'
        'No DEV/TEST text or predictions, no label corrections, no synthetic uppercase examples, no frequency reweighting.\n'
        'Existing8711 IDs, old13988 merge ranks, normalizer, case policy and special tokens remain fixed.\n'+
        ('Existing â/î/û/Â/Î/ğ/Ğ stay inherited. No forcedÛ: raw or canonically normalized TRAINÛ count is zero.\n' if remove else
         'Existing â/î/û/Â/Î/ğ/Ğ stay inherited. NewÛ is supported by actual canonically normalized TRAIN occurrences.\n')+
        'The separate8712 coverage-only package can encodeÛ, but its new row has no positive TRAIN examples and was not activated.\n'
        'Every TRAIN encoding exactly matches the completed experimental candidate after only appended-ID renumbering.\n'
        'Full report includes all1914 six-language TRAIN replay resegmentations. Preserving IDs does not preserve token sequences or ASR quality.\n'
        'Byte_fallback=true is inherited but there are zero actual<0x..> fallback tokens: existing punctuation/foreign-character unknowns remain.\n'
        'This is a tokenizer artifact, not a trained model or measured quality gain. New head/embedding rows, optimizer extension,\n'
        'TDT duration-row relocation, new TRAIN labels/denominators/geometry and retention tests are required before adoption.\n'
        'Original HF ParakeetTokenizer config is included as metadata; the tested loader is the Rust Tokenizer JSON API.\n'
        'HF ParakeetTokenizer uses CTC grouping by default; TDT decode needs group_tokens=False. HF offline/no-Torch load is unverified.\n')
    (folder/'README.txt').write_text(text,encoding='utf8')
    names=['tokenizer.json','vocab.json','vocab.tsv','merges.txt','tokenizer_config.json','added_pieces.json','unicode_roundtrip_tests.json',
        'full_TRAIN_projection_validation.json','projection_recipe_frozen_before_validation.json','row_migration_plan.json','README.txt']
    manifest={'kind':'Final_corpus_driven_Turkish_BPE_tokenizer_candidate_v1','at_utc':m.now(),'vocabulary':size,'candidate_only':True,
        'artifacts':{n:m.record(folder/n) for n in names},'full_source_label_audit':m.record(EXPERIMENT/'source_label_audit.json')}
    m.write('manifest.json',manifest)
    package=AREA/('Turkish_all_TRAIN_BPE%d_candidate.zip'%size)
    with zipfile.ZipFile(package,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in [*names,'manifest.json']:z.write(folder/name,folder.name+'/'+name)
    handoff={'kind':'Final_corpus_driven_Turkish_BPE_delivery_v1','at_utc':m.now(),'vocabulary':size,'passed':True,'PID':os.getpid(),
        'candidate':m.record(folder/'tokenizer.json'),'manifest':m.record(folder/'manifest.json'),'package':m.record(package),
        'full_TRAIN_validation':m.record(folder/'full_TRAIN_projection_validation.json'),'active_tokenizer_model_labels_untouched':True,
        'remaining':'ROOT review plus explicit model/optimizer/labels/geometry/retention integration before adoption; no production launch.'}
    (AREA/'latest_finalized_corpus_BPE_handoff.json').write_text(json.dumps(handoff,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    g.phase='completed_corpus_projection';g.publish(passed=True)
    print(json.dumps(handoff,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
