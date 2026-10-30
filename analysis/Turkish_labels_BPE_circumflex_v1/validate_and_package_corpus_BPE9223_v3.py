"""Actual Rust all-TRAIN validation and tokenizer-only packaging after fit exit0.

No new frequency counting, label-catalog writes, model loading or GPU use.
"""
import collections,gc,hashlib,importlib,json,os,sys,time,unicodedata,zipfile
from pathlib import Path
D=Path(r'E:\gpu_training_scratchpad\analysis\Turkish_labels_BPE_circumflex_v1')
sys.path.insert(0,str(D));m=importlib.import_module('audit_and_fit_append_only_v5')
FIT=D/'compact_fit_v6';SOURCE=D/'all_TRAIN_extension_v5'
OUT=D/'corpus_driven_BPE9223_v2'

def main():
 proof=json.loads((FIT/'completed_fit_handoff.json').read_text())
 closure=json.loads((FIT/'actual_owned_CLOSED0.json').read_text())
 assert proof['passed'] and closure['actual_exit_code']==0 and closure['all_captured_identities_dead']
 assert m.sha(FIT/'tokenizer.json')==proof['tokenizer']['sha256']
 audit=json.loads((SOURCE/'source_label_audit.json').read_text())
 assert m.sha(SOURCE/'source_label_audit.json')==proof['source_audit']['sha256']
 assert sum(x['normalized_char_counts'].get('Û',0) for x in audit['groups'].values())==0
 recipe=json.loads((SOURCE/'candidate_recipe_pinned_before_fit.json').read_text())
 for key,path in [('broad_text',m.BROAD),('broad_labels',m.BROAD_LABELS),('clean_text',m.CLEAN),('clean_labels',m.CLEAN_LABELS)]:
  assert m.stat(path)==recipe['sources'][key],('Source stat changed',key)
 assert not OUT.exists();OUT.mkdir();m.AREA=OUT
 g=m.Guard();g.phase='freeze_corpus_driven_candidate_and_Rust_validation'
 baseline=g.process.memory_info().private;parents=sum(p.memory_info().private for p in g.parents)
 forecast=512000000;mem=m.memory();assert baseline<=384000000 and parents+forecast<2000000000
 commit_limit=mem['commit_headroom_bytes']/(1-mem['commit_ratio'])
 assert mem['commit_ratio']+(forecast-baseline)/commit_limit<.90 and mem['commit_headroom_bytes']-(forecast-baseline)>=2000000000
 m.write('actual_validation_memory_admission.json',{'at_utc':m.now(),'baseline_private':baseline,'actual_parent_private':parents,'forecast_leaf_private':forecast,'memory':mem,'projected_commit_ratio':mem['commit_ratio']+(forecast-baseline)/commit_limit,'passed':True})
 from tokenizers import Tokenizer
 import tokenizers
 assert not any(k in sys.modules for k in ('torch','transformers','nemo'))
 base=json.loads(m.INCUMBENT.read_text());candidate=json.loads((FIT/'tokenizer.json').read_text())
 assert m.sha(m.INCUMBENT)==m.INCUMBENT_SHA
 assert len(candidate['model']['vocab'])==9223 and len(candidate['model']['merges'])==14500
 for piece,identifier in base['model']['vocab'].items():assert candidate['model']['vocab'][piece]==identifier
 assert candidate['model']['merges'][:13988]==base['model']['merges']
 for key in base:
  if key!='model':assert candidate[key]==base[key]
 assert 'Û' not in candidate['model']['vocab']
 additions=json.loads((FIT/'added_pieces.json').read_text());assert len(additions)==512
 assert all(x['weighted_pair_frequency']>0 and 'Û' not in x['piece'] for x in additions)
 old=Tokenizer.from_str(json.dumps(base,ensure_ascii=False));new=Tokenizer.from_str(json.dumps(candidate,ensure_ascii=False))
 # A shadow reinstates ONLY the unused manual atom. Prove on actual full text
 # that its omission changes neither merges nor encodings except appended IDs.
 shadow=json.loads(json.dumps(candidate));shadow['model']['vocab']={p:(i+1 if i>=8711 else i) for p,i in candidate['model']['vocab'].items()}
 shadow['model']['vocab']['Û']=8711
 shadow_tok=Tokenizer.from_str(json.dumps(shadow,ensure_ascii=False));del shadow;gc.collect()
 m.write('tokenizer.json',candidate);m.write('added_pieces.json',additions);m.write('vocab.json',candidate['model']['vocab'])
 (OUT/'vocab.tsv').write_text('\n'.join(str(i)+'\t'+p for p,i in sorted(candidate['model']['vocab'].items(),key=lambda x:x[1]))+'\n',encoding='utf8')
 (OUT/'merges.txt').write_text('#version: 0.2\n'+'\n'.join(' '.join(p) for p in candidate['model']['merges'])+'\n',encoding='utf8')
 (OUT/'tokenizer_config.json').write_bytes((m.PROJECT/'work/models/lora_20261010_01_merged/tokenizer_config.json').read_bytes())
 m.write('candidate_recipe_frozen_before_validation.json',{'at_utc':m.now(),'candidate':m.record(OUT/'tokenizer.json'),
  'incumbent':m.record(m.INCUMBENT),'fit_actual_closure':m.record(FIT/'actual_owned_CLOSED0.json'),
  'fit_handoff':m.record(FIT/'completed_fit_handoff.json'),'source_label_audit':m.record(SOURCE/'source_label_audit.json'),
  'all_Turkish_TRAIN_clip_occurrences':2983883,'new_merges':512,'vocabulary':9223,
  'existing8711_IDs_and_13988_merge_ranks_and_normalizer_and_special_tokens_fixed':True,
  'manual_unobserved_character_not_forced':True,'raw_and_normalized_TRAIN_Ucircumflex_count':0,
  'word_frequency_policy':audit['policy'],'Latin_pair_scope':'Only observed Latin-letter merges; all original inherited tokens retained.',
  'new_full_corpus_frequencies_not_recomputed':True,'DEV_TEST_not_used':True,'model_or_labels_not_adopted':True})
 groups={};uses=collections.Counter();digest=hashlib.sha256();shadow_digest=hashlib.sha256();examples=[]
 source_order_digest=hashlib.sha256();rows=0
 def consume(pool,values):
  nonlocal rows
  texts=[x[3] for x in values]
  aa=old.encode_batch(texts);bb=new.encode_batch(texts);cc=shadow_tok.encode_batch(texts)
  for (ordinal,sid,dataset,text,split),a,b,c in zip(values,aa,bb,cc):
   assert b.ids==[(i-1 if i>8711 else i) for i in c.ids] and 8711 not in c.ids
   assert old.decode(a.ids,skip_special_tokens=False)==new.decode(b.ids,skip_special_tokens=False)
   assert a.ids.count(0)==b.ids.count(0)
   assert all(0<=i<9223 and i!=8192 for i in b.ids)
   key=pool+':'+dataset
   v=groups.setdefault(key,{'rows':0,'incumbent_tokens':0,'candidate_tokens':0,'changed_rows':0,
    'unknown_rows':0,'unknown_tokens':0,'maximum_candidate_U':0,'new_piece_occurrences':0,
    'rows_with_any_of_eight_audit_characters':0,'eight_character_decode_matches_incumbent_rows':0})
   v['rows']+=1;v['incumbent_tokens']+=len(a.ids);v['candidate_tokens']+=len(b.ids);v['changed_rows']+=a.ids!=b.ids
   v['unknown_rows']+=0 in b.ids;v['unknown_tokens']+=b.ids.count(0);v['maximum_candidate_U']=max(v['maximum_candidate_U'],len(b.ids))
   v['new_piece_occurrences']+=sum(i>=8711 for i in b.ids)
   if any(ch in text for ch in m.CHARS):v['rows_with_any_of_eight_audit_characters']+=1;v['eight_character_decode_matches_incumbent_rows']+=1
   uses.update(i for i in b.ids if i>=8711)
   digest.update((m.canonical([pool,ordinal,sid,b.ids])+'\n').encode())
   shadow_digest.update((m.canonical([pool,ordinal,sid,c.ids])+'\n').encode())
   source_order_digest.update((m.canonical([pool,ordinal,sid,dataset,split,text])+'\n').encode())
   if len(examples)<24 and a.ids!=b.ids and sum(x['group']==key for x in examples)<4:
    examples.append({'group':key,'source_id':sid,'text':text,'incumbent_tokens':a.tokens,'candidate_tokens':b.tokens})
   g.pace(len(text.encode())+len(b.ids)*8+128);rows+=1
  if rows%8192==0:g.publish(validated_rows=rows)
 def check(pool,iterator):
  batch=[]
  for row in iterator:
   batch.append(row)
   if len(batch)==256:consume(pool,batch);batch.clear()
  if batch:consume(pool,batch)
 g.phase='actual_Rust_all2983883_TRAIN_candidate_projection_and_roundtrip_validation'
 db=m.ro(m.BROAD)
 check('broad',db.execute('SELECT ordinal,unique_id,dataset,text,original_split FROM clips ORDER BY ordinal'));db.close()
 def cleaner():
  with m.CLEAN.open('rb') as f:
   for ordinal,line in enumerate(f,1):
    g.pace(len(line));row=json.loads(line)
    yield ordinal,row['source_id'],row['dataset'],row['text'],row['dataset_split']
 check('cleaner',cleaner())
 assert rows==2983883
 assert sum(v['incumbent_tokens'] for v in groups.values())==113443643
 # Source membership/order already audited. Reconfirm exact digest with the
 # actual declared split convention rather than a guessed replacement label.
 assert source_order_digest.hexdigest()==audit['source_ordered_text_sha256'],('Source text/order digest differs',source_order_digest.hexdigest(),audit['source_ordered_text_sha256'])
 for key,v in groups.items():
  prior=audit['groups'][key];assert v['rows']==prior['rows'] and v['incumbent_tokens']==prior['existing_label_tokens']
  v['token_reduction_fraction']=1-v['candidate_tokens']/v['incumbent_tokens']
 pool=m.PROJECT/'outputs/parakeet_turkish/runs/adaptive_20261014_TR_WorldSpeech_expanded_01/training_pool.json'
 assert m.sha(pool)=='00b839c717762aba74326e9867969d85b7ffaaabd7ead7ddbe732fe6be8c6cf4'
 replay=json.loads(pool.read_text())['replay'];assert len(replay)==1914
 retained={};replay_examples=[]
 for row in replay:
  assert row['dataset_split']=='train'
  a=old.encode(row['text']);b=new.encode(row['text']);lang=row['language']
  assert lang in ('en','es','it','fr','de','pt')
  v=retained.setdefault(lang,{'rows':0,'incumbent_tokens':0,'candidate_tokens':0,'encoding_changes':0,'decode_changes':0,'incumbent_unknown_rows':0,'candidate_unknown_rows':0})
  v['rows']+=1;v['incumbent_tokens']+=len(a.ids);v['candidate_tokens']+=len(b.ids);v['encoding_changes']+=a.ids!=b.ids
  v['decode_changes']+=old.decode(a.ids,skip_special_tokens=False)!=new.decode(b.ids,skip_special_tokens=False)
  v['incumbent_unknown_rows']+=0 in a.ids;v['candidate_unknown_rows']+=0 in b.ids
  if a.ids!=b.ids and len(replay_examples)<18:replay_examples.append({'language':lang,'text':row['text'],'incumbent_tokens':a.tokens,'candidate_tokens':b.tokens})
 unicode=[]
 for text in ['hâlâ millî sükûn','HÂLÂ MİLLÎ','ağaç Ğ','hükûmet sükût','Â Î â î û ğ Ğ']:
  a=new.encode(text);b=new.encode(unicodedata.normalize('NFD',text))
  assert a.ids==b.ids and new.decode(a.ids,skip_special_tokens=False)==text
  unicode.append({'text':text,'IDs':a.ids,'NFD_encodes_identically':True,'roundtrip':text})
 assert new.encode('Û').ids==old.encode('Û').ids and 0 in new.encode('Û').ids
 for token in base['added_tokens']:assert new.encode(token['content']).ids==old.encode(token['content']).ids
 m.write('unicode_roundtrip_tests.json',{'positive_observed_forms':unicode,'unsupported_unobserved_uppercase_Ucircumflex':'Inherited unknown handling unchanged; no new untrained slot forced.',
  'all_added_special_token_encodings_unchanged':True,'normalizer_pretokenizer_decoder_unchanged':True})
 result={'kind':'Actual_all_TRAIN_Rust_BPE9223_validation_v2','at_utc':m.now(),'passed':True,'PID':os.getpid(),
  'all_TRAIN_rows':rows,'all_original_source_label_audit_reused':True,'source_text_order_digest_exact':True,
  'source_ordered_text_sha256':source_order_digest.hexdigest(),'ordered_candidate_labels_sha256':digest.hexdigest(),
  'manual_atom_shadow_ordered_labels_sha256':shadow_digest.hexdigest(),'all_encodings_exact_up_to_only_new_ID_projection':True,
  'decoded_text_and_unknown_count_match_incumbent_all_rows':True,'groups':groups,
  'appended_piece_types_directly_emitted':len(uses),'appended_piece_token_occurrences':sum(uses.values()),
  'examples':examples,'six_language_original1914_TRAIN_replay':retained,'replay_encoding_change_examples':replay_examples,
  'no_new_models_GPU_DEV_TEST_or_active_source_mutations':True,'candidate_activated':False,'measured_ASR_quality_gain':False,
  'elapsed_seconds':time.monotonic()-g.started,'peak_private_bytes':g.peak,'tokenizers_version':tokenizers.__version__}
 m.write('full_TRAIN_validation.json',result)
 m.write('row_migration_plan.json',{'tokenizer_candidate_only':True,'old_vocabulary':8711,'new_vocabulary':9223,'blank_id':8192,
  'all_existing8711_rows_keep_IDs':True,'appended_rows_half_open':[8711,9223],
  'TDT_duration_rows_old_half_open':[8711,8716],'TDT_duration_rows_new_half_open':[9223,9228],
  'new_joint_width':9228,'active_model_or_optimizer_migrated':False,
  'explicit_new_row_initialization_optimizer_extension_and_new_labels_required':True,
  'mature_moments_clocks_RNG_history_must_be_preserved':True,'protected_replay_teacher_DEV_TEST_not_silently_retokenized':True})
 readme=('Turkish all-TRAIN BPE candidate: 9223 tokens\n\n'
  'Tested use: from tokenizers import Tokenizer; t=Tokenizer.from_file("tokenizer.json")\n'
  '512 append-only Latin BPE merges learned from all 2,983,883 available Turkish TRAIN clip occurrences: '
  '2,631,115 broad weak-label and 352,768 cleaner rows. All duplicate text/source overlaps retain their occurrence frequency.\n'
  'No DEV/TEST text, transcript corrections, case augmentation, quality filtering or invented diacritics. '
  'The original 8711 IDs/13988 merge ranks, normalizer, case policy and special tokens are fixed.\n'
  'Existing â î û Â Î ğ Ğ remain inherited. No forced Û: zero raw or normalized TRAIN occurrences. '
  'The separate optional8712 coverage patch has an untrainedÛ slot; this corpus candidate does not.\n'
  'Each actual TRAIN encoding was checked with the Rust tokenizer: decoded text/unknown count unchanged, '
  'and exact shadow-ID equivalence proves removal of the unused manual atom does not alter segmentation.\n'
  'Existing byte_fallback=true has zero actual<0x..> vocabulary tokens, so pre-existing punctuation/foreign-character unknowns remain. '
  'This is not complete Unicode coverage or a measured ASR gain.\n'
  'The full report measures changed encodings on all1914 six-language TRAIN replay examples. '
  'Stable old IDs do not promise identical multilingual segmentation or retention.\n'
  'Before adoption: initialize new embedding/output rows, extend optimizer without resetting mature state, '
  'relocate the5 TDT duration rows, regenerate TRAIN labels/means/geometry, and measure six-language retention.\n'
  'HF tokenizer_config is original metadata; only RustJSON loader was tested. '
  'ParakeetTokenizer defaults to CTC grouping; TDT decode needs group_tokens=False. HF/TDT load is unverified.\n')
 (OUT/'README.txt').write_text(readme,encoding='utf8')
 for key,path in [('broad_text',m.BROAD),('broad_labels',m.BROAD_LABELS),('clean_text',m.CLEAN),('clean_labels',m.CLEAN_LABELS)]:assert m.stat(path)==recipe['sources'][key]
 assert m.sha(m.INCUMBENT)==m.INCUMBENT_SHA
 names=['tokenizer.json','vocab.json','vocab.tsv','merges.txt','tokenizer_config.json','added_pieces.json','unicode_roundtrip_tests.json',
  'full_TRAIN_validation.json','candidate_recipe_frozen_before_validation.json','row_migration_plan.json','README.txt']
 m.write('manifest.json',{'kind':'Turkish_all_TRAIN_BPE9223_final_tokenizer_candidate_v2','at_utc':m.now(),'vocabulary':9223,
  'candidate_only':True,'artifacts':{n:m.record(OUT/n) for n in names},'source_label_audit':m.record(SOURCE/'source_label_audit.json'),
  'exact512_fit':m.record(FIT/'completed_fit_handoff.json'),'actual_fit_closure':m.record(FIT/'actual_owned_CLOSED0.json')})
 package=D/'Turkish_all_TRAIN_BPE9223_candidate_v2.zip'
 with zipfile.ZipFile(package,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for name in [*names,'manifest.json']:z.write(OUT/name,OUT.name+'/'+name)
 handoff={'kind':'Finalized_corpus_driven_Turkish_BPE_delivery_v2','at_utc':m.now(),'passed':True,'PID':os.getpid(),
  'create_time':g.main_identity['create_time'],'vocabulary':9223,'new_corpus_merges':512,'TRAIN_rows':rows,
  'candidate':m.record(OUT/'tokenizer.json'),'manifest':m.record(OUT/'manifest.json'),'package':m.record(package),
  'validation':m.record(OUT/'full_TRAIN_validation.json'),'source_audit':m.record(SOURCE/'source_label_audit.json'),
  'actual_fit_closure':m.record(FIT/'actual_owned_CLOSED0.json'),'elapsed_seconds':time.monotonic()-g.started,
  'peak_private_bytes':g.peak,'remaining':'ROOT artifact review and explicit model/optimizer/label/geometry/retention integration before adoption.',
  'active_tokenizer_model_labels_untouched':True,'measured_ASR_quality_gain':False}
 m.write('completed_handoff.json',handoff)
 (D/'latest_finalized_corpus_BPE_handoff.json').write_text(json.dumps(handoff,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
 g.phase='finalized_corpus_BPE_candidate_complete';g.publish(passed=True)
 print(json.dumps(handoff,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
