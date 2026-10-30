import sys,json,collections,unicodedata
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'outputs/parakeet_turkish'))
from metrics import normalize_text,score_transcriptions
root=Path.cwd()/'outputs/parakeet_turkish/results'
new='ğĞıİşŞ'
def read(tag):
 return [json.loads(line) for line in (root/tag/'validation_tr.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
def script_counts(text):
 result=collections.Counter()
 for char in text:
  if unicodedata.category(char).startswith('L'):
   name=unicodedata.name(char,'UNKNOWN')
   script=next((item for item in ['LATIN','CYRILLIC','GREEK','ARABIC'] if item in name),'OTHER')
   result[script]+=1
 return result
def alignment(ref,hyp):
 mat=[[0]*(len(hyp)+1) for _ in range(len(ref)+1)]
 for i in range(len(ref)+1): mat[i][0]=i
 for j in range(len(hyp)+1): mat[0][j]=j
 for i,r in enumerate(ref,1):
  for j,h in enumerate(hyp,1): mat[i][j]=min(mat[i-1][j]+1,mat[i][j-1]+1,mat[i-1][j-1]+(r!=h))
 i,j=len(ref),len(hyp); result=[]
 while i or j:
  if i and j and mat[i][j]==mat[i-1][j-1]+(ref[i-1]!=hyp[j-1]):
   result.append((ref[i-1],hyp[j-1]));i-=1;j-=1
  elif i and mat[i][j]==mat[i-1][j]+1:
   result.append((ref[i-1],None));i-=1
  else:
   result.append((None,hyp[j-1]));j-=1
 return result[::-1]

import hashlib,datetime
from metrics import edit_distance
out=Path.cwd()/'outputs/parakeet_turkish'
run_id='lora_20261010_01'
run=out/'runs'/run_id
provenance={}
def capture(path):
 data=path.read_bytes()
 provenance[str(path.relative_to(Path.cwd())).replace('\\','/')]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
 return data.decode('utf-8-sig')
def jsonfile(path): return json.loads(capture(path))
def rowsfile(path): return [json.loads(x) for x in capture(path).splitlines() if x.strip()]
config=jsonfile(run/'run_config.json')
history=jsonfile(run/'training_history.json')
worker=jsonfile(run/'completion_status.json')
data_audit=jsonfile(out/'lora_training_data_verification.json')
prior_audit=jsonfile(out/'lora_new_token_verification_step200.json')
capture(out/'metrics.py');capture(out/'train_lora.py')
completed=[r for r in history if 'validation' in r]
langs=['en','es','it','fr','de','tr']
base={}
dev=[]
def score(rows,lang):
 result=score_transcriptions(rows,language=lang);result.pop('details')
 result['wer_percent']=100*result['wer'];result['cer_percent']=100*result['cer']
 return result
def diagnostic(rows,lang):
 emission=collections.Counter();scripts=collections.Counter();matched=collections.Counter();refcounts=collections.Counter()
 spaces=other=exact=hypwords=nsp_err=nsp_ref=0
 for row in rows:
  ref=normalize_text(row['reference'],lang);hyp=normalize_text(row['hypothesis'],lang)
  emission.update(c for c in row['hypothesis'] if c in new);scripts.update(script_counts(hyp))
  exact+=int(ref==hyp);hypwords+=len(hyp.split())
  rr=ref.replace(' ','');hh=hyp.replace(' ','')
  nsp_err+=edit_distance(rr,hh);nsp_ref+=len(rr)
  for a,b in alignment(ref,hyp):
   if a is not None and a in new:
    refcounts[a]+=1
    if a==b:matched[a]+=1
   if a!=b:
    if a==' ' or b==' ':spaces+=1
    else:other+=1
 return {'raw_added_character_emissions':{c:emission[c] for c in new},'normalized_hypothesis_script_letters':dict(scripts),
  'exact_normalized_utterances':exact,'hypothesis_words':hypwords,
  'character_alignment_reference_counts':dict(refcounts),'character_alignment_matches':dict(matched),
  'space_involving_character_edits':spaces,'other_character_edits':other,
  'space_involving_fraction_of_character_edits':spaces/(spaces+other) if spaces+other else 0,
  'cer_after_removing_all_spaces_diagnostic_only':nsp_err/nsp_ref,
  'alignment_note':'One optimal Levenshtein character alignment, ties prefer diagonal then deletion then insertion; decomposition is descriptive and can vary with equally optimal ties. No-space CER is not the benchmark score.'}
for lang in langs:
 rows=rowsfile(root/'base_v3'/f'validation_{lang}.jsonl')
 assert len(rows)==50 and all(r['split']=='validation' for r in rows)
 base[lang]={'metrics':score(rows,lang),'rows':rows}
for record in completed:
 step=record['step'];tag=f'{run_id}_step_{step}'
 entry={'step':step,'turkish_draws':2*step,'turkish_unique_clips_seen':record['turkish_unique_clips_seen'],'training_pool_equivalent_draw_cycles':2*step/config['turkish_clips_used'],'languages':{},'old_language_gates_passed':True}
 jsonfile(root/tag/'metadata.json')
 for lang in langs:
  rows=rowsfile(root/tag/f'validation_{lang}.jsonl')
  assert len(rows)==50 and all(r['split']=='validation' for r in rows)
  assert [(r['source_id'],r['reference'],r['audio_sha256']) for r in rows]==[(r['source_id'],r['reference'],r['audio_sha256']) for r in base[lang]['rows']]
  metrics=score(rows,lang)
  item={'metrics':metrics,'diagnostic':diagnostic(rows,lang)}
  if lang!='tr':
   allowance=max(.005,.1*base[lang]['metrics']['wer']);delta=metrics['wer']-base[lang]['metrics']['wer']
   item['retention_gate']={'baseline_wer':base[lang]['metrics']['wer'],'delta_wer':delta,'delta_percentage_points':100*delta,'allowed_delta_wer':allowance,'allowed_delta_percentage_points':100*allowance,'passed':delta<=allowance+1e-12}
   entry['old_language_gates_passed'] &= item['retention_gate']['passed']
  entry['languages'][lang]=item
 entry['turkish_improved']=entry['languages']['tr']['metrics']['wer']<base['tr']['metrics']['wer']
 dev.append(entry)
best=min((d for d in dev if d['old_language_gates_passed'] and d['turkish_improved']),key=lambda d:d['languages']['tr']['metrics']['wer'])
streams={}
for stream,manifest in [('turkish_commonvoice','tr_train.jsonl'),('turkish_fleurs_train','tr_fleurs_train.jsonl'),('retention_fleurs_train','replay_train.jsonl')]:
 source={r['source_id']:r for r in rowsfile(Path.cwd()/'work/data'/manifest)}
 records=data_audit['streams'][stream]['accepted_source_ids_and_labels']
 words=sum(len(normalize_text(source[r['source_id']]['reference'],r['language']).split()) for r in records)
 tokens=sum(r['target_tokens'] for r in records)
 streams[stream]={'clips':len(records),'hours':sum(r['duration'] for r in records)/3600,'target_tokens':tokens,'normalized_reference_words':words,'tokens_per_normalized_word':tokens/words}
trtokens=streams['turkish_commonvoice']['target_tokens']+streams['turkish_fleurs_train']['target_tokens']
trwords=streams['turkish_commonvoice']['normalized_reference_words']+streams['turkish_fleurs_train']['normalized_reference_words']
report={
 'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'run_id':run_id,
 'scope':'CPU-only diagnosis of completed FLEURS validation results and training data/configuration. No held-out test predictions were accessed for this diagnosis. No GPU, model weights, active trainer or helper edits.',
 'quality_statement':'Turkish remains highly inaccurate and is not ready for useful ASR. Relative improvement plus old-language retention gates are experiment criteria, not a claim that Turkish quality is achieved.',
 'snapshot':{'completed_training_step':history[-1]['step'],'configured_steps':config['steps'],'completed_dev_steps':[d['step'] for d in dev],
  'worker_stage_at_snapshot':worker['stage'],'best_qualifying_dev_step_at_snapshot':best['step'],
  'best_qualifying_turkish_dev_metrics':best['languages']['tr']['metrics'],
  'best_qualifying_old_languages':{l:best['languages'][l]['retention_gate'] for l in langs[:-1]},
  'peak_allocated_vram_gb_in_history':max(r['peak_allocated_vram_gb'] for r in history)},
 'base_dev_metrics':{l:base[l]['metrics'] for l in langs},
 'completed_dev_progress':dev,
 'training_exposure':{
  'streams':streams,'turkish_pool_clips':config['turkish_clips_used'],'turkish_pool_hours':data_audit['totals']['turkish_hours'],
  'combined_turkish_tokens_per_normalized_word':trtokens/trwords,
  'planned_turkish_draws':2*config['steps'],'planned_turkish_pool_equivalent_draw_cycles':2*config['steps']/config['turkish_clips_used'],
  'microbatch':1,'gradient_accumulation':config['accumulation'],'turkish_clips_per_optimizer_step':2,'replay_clips_per_optimizer_step':2,
  'note':'Current step counting represents accumulated single-clip forwards. The configured 1200 steps expose only about 1.11 Turkish pool-equivalent cycles, and longer FLEURS recordings remain harder than the short Common Voice prefix.'},
 'vocabulary_and_mapping_evidence':{
  'prior_exact_cache_label_and_generation_audit':'outputs/parakeet_turkish/lora_new_token_verification_step200.json',
  'appended_token_ids':{'ğ':8193,'Ğ':8194,'ı':8195,'İ':8196,'ş':8197,'Ş':8198},
  'blank_token_id':8192,'duration_output_ids':list(range(8199,8204)),
  'conclusion':'The prior CPU cache audit verified valid labels and relocated generation suppression; current dev emissions confirm the lowercase new tokens can be decoded. Missing or wrong Turkish characters are now a learning problem rather than a demonstrated label/suppression mapping error.',
  'startup_note':'New output rows were initialized from donors with bias -2 and learn with effective row update rate 3e-4. No new character was emitted by step200; lowercase characters emerged later. Uppercase Ğ had zero training target occurrences in this pool.'},
 'inferences':[
  'Word-boundary errors matter, but the character alignment at the best qualifying checkpoint leaves most edits outside spaces. Adding subwords alone cannot explain or fix the remaining vowel/consonant spelling errors.',
  'The apparent flattening between steps600 and800 was not a demonstrated plateau: step1000 improved again before even one full Turkish cycle had been completed.',
  'Original-language retention is a mixed constraint: step400 failed the English gate while later checkpoints passed. Added dotless ı still leaks into old-language words even when aggregate WER passes; monitor this explicitly.',
  'Turkish uses about3.8 target tokens per normalized word versus about2.23 for replay, motivating a later controlled Turkish-subword ablation, but unique audio exposure and efficient true batching have higher immediate priority.'],
 'next_run_priorities':[
  {'priority':1,'action':'Finish the current finite1200-step evaluation/export pipeline and preserve the best qualifying validation checkpoint.','status':'Active existing experiment; no new run launched by this diagnosis.','decision_rule':'Select by Turkish dev improvement and each old-language gate; report held-out results separately without using them for further tuning.'},
  {'priority':2,'action':'Profile length-bucketed true batch2, then4, with variable-length TDT/masking correctness checks, while keeping approximately50% Turkish and50% language-uniform replay.','status':'Proposed future implementation/profile.','rationale':'The recorded peak allocated memory is only about3.29GB on a16GBGPU, and single-clip accumulation serializes computation. Fit and speed must be measured before claiming a larger batch works.'},
  {'priority':3,'action':'Increase Turkish exposure: compare a controlled2–3-pass continuation with an expanded, independently audited Turkish training pool from the pinned Common Voice and FLEURS train sources.','status':'Proposed future training; not performed.','rationale':'The current2.28hours and1.11planned pool cycles are too limited to establish the achievable Turkish quality. More epochs can clarify learning progress; more unique speakers/utterances provide the higher ceiling.','guardrails':'Preserve held-out source/sentence/audio isolation, report actual usable hours after filtering, retain replay/KD and dev gates, and count exposure in clips/hours/epochs instead of arbitrary steps.'},
  {'priority':4,'action':'If spelling still plateaus after adequate exposure, run a decoder-specific learning-rate ablation, for example5e-5 versus current2e-5, while leaving shared joint/projector and new-row rates conservative.','status':'Conditional proposed ablation.','rationale':'Original decoder/head parameters receive a much smaller rate than LoRA and appended rows; word formation may adapt slowly. Broadly increasing new-row or shared-head learning rates risks the observed old-language ı leakage.'},
  {'priority':5,'action':'Only then compare a modest Turkish BPE extension restricted initially to pieces containing the new Turkish characters.','status':'Conditional proposed tokenizer experiment.','rationale':'Lower fragmentation may help sequence length and word formation. New pieces add untrained rows and can change shared-language tokenization, so they are not the first correction for this pilot.','guardrails':'Preserve original IDs/merge order/blank rows and exact old-language tokenization; relocate duration/suppression IDs again; audit save/reload and paired dev retention. Use training text only for piece selection.'}],
 'preservation_plan':'Keep language-uniform replay, original-teacher token/duration/feature distillation and per-language dev gates. Add a dev diagnostic for off-language Turkish-character leakage; use new replay TRAIN examples rather than dev/test examples if constructing hard negatives.',
 'limitations':['Only50FLEURS validation utterances per language; small development gates are noisy and descriptive.','No causal ablation has separated data quantity, tokenizer fragmentation, decoder rate and acoustic adaptation.','Training continues; best qualifying dev step and final outcomes may change after this captured snapshot.','No useful Turkish accuracy claim follows from a relative-improvement gate.'],
 'provenance':{'base_revision':config['base_revision'],'transformers_commit':config['transformers_commit'],'training_code_sha256_as_configured':config['code_sha256'],'inputs_captured_sha256':provenance}
}
target=out/'next_run_dev_diagnosis.json'
target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'saved':str(target),'snapshot':report['snapshot'],'training_exposure':report['training_exposure'],'best_TR_diagnostic':best['languages']['tr']['diagnostic']},ensure_ascii=False))

