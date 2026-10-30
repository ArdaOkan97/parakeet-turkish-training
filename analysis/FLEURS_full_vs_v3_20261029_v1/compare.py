"""Independently rescore complete paired predictions; sentence-cluster CIs."""
import csv, hashlib, json, os, sys, time
from pathlib import Path
R=Path(__file__).parent
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20')
sys.path.insert(0,str(P/'outputs/parakeet_turkish'))
from paired_windows_commit_telemetry_v1 import paired_sample
from resource_watchdog_v2 import sample_memory
from metrics import normalize_text,edit_distance
import psutil
psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
began=time.monotonic()
samples=[paired_sample(),paired_sample()]
with (R/'comparison_admission.json').open('x',encoding='utf-8') as f:json.dump(samples,f);f.flush();os.fsync(f.fileno())
assert all(s['global_memory_status_ex']['ullAvailPageFile_bytes']>=6028384896 and s['get_performance_info_after']['physical_available_bytes']>=3278384896 for s in samples)
def guard():
    s=sample_memory()
    assert time.monotonic()-began<180 and s['worker_private_memory']['private_bytes']<=536870912
    assert s['system_memory']['commit_available_bytes']>=3509949440 and s['system_memory']['physical_available_bytes']>=3009949440
guard()
with (R/'comparison_scope.json').open('x',encoding='utf-8') as f:json.dump({'inputs':'New complete full-test predictions and scores only, both models and seven languages','original_training_inputs':False,'scoring':'Independent corpus word edit counts; 10000 paired sentence cluster bootstrap resamples','maximum_wall_seconds':180,'maximum_private_bytes':536870912},f);f.flush();os.fsync(f.fileno())
def word_scores(rows,language):
    details=[]
    for r in rows:
        reference=normalize_text(r['reference'],language).split();hypothesis=normalize_text(r['hypothesis'],language).split()
        details.append({'sentence_id':r['sentence_id'],'reference_words':len(reference),'word_errors':edit_distance(reference,hypothesis)})
    words=sum(d['reference_words'] for d in details);errors=sum(d['word_errors'] for d in details)
    return {'details':details,'reference_words':words,'word_errors':errors,'utterances':len(details),'wer':errors/words}
import numpy as np
names={'tr':'Turkish','en':'English','es':'Spanish','it':'Italian','fr':'French','de':'German','pt':'Portuguese'}
inventory=json.loads((R/'input_inventory.json').read_text())
for tag in ('v3_v3','adapted_v3'):
    complete=json.loads((R/tag/'complete.json').read_text())
    assert complete['all_full_TEST_rows']==5620
results=[]
for language,expected in inventory['datasets'].items():
    guard()
    predictions={tag:[json.loads(line) for line in (R/tag/(language+'_test_predictions.jsonl')).read_text(encoding='utf-8').splitlines()] for tag in ('v3_v3','adapted_v3')}
    baseline,adapted=predictions.values()
    assert len(baseline)==len(adapted)==expected['rows']
    assert all(all(a[k]==b[k] for k in ('source_id','sentence_id','reference','audio_sha256','duration_seconds','row_index')) for a,b in zip(baseline,adapted))
    scores=[word_scores(rows,language=language) for rows in (baseline,adapted)]
    for tag,score in zip(predictions,scores):
        saved=json.loads((R/tag/(language+'_test_score.json')).read_text())
        assert all(saved[k]==score[k] for k in ('word_errors','reference_words','utterances'))
    groups={}
    for a,b in zip(scores[0]['details'],scores[1]['details']):
        group=groups.setdefault(a['sentence_id'],[0,0]);group[0]+=a['reference_words'];group[1]+=b['word_errors']-a['word_errors']
    values=np.array(list(groups.values()),dtype=np.int64);rng=np.random.default_rng(20261029)
    deltas=[]
    for _ in range(10):
        indices=rng.integers(0,len(values),size=(1000,len(values)))
        samples=values[indices].sum(axis=1)
        deltas.extend((100*samples[:,1]/samples[:,0]).tolist())
    before,after=scores
    result={'language':language,'name':names[language],'utterances':expected['rows'],'audio_hours':sum(r['duration_seconds'] for r in baseline)/3600,'reference_words':before['reference_words'],'v3_word_errors':before['word_errors'],'adapted_word_errors':after['word_errors'],'v3_wer_percent':100*before['wer'],'adapted_wer_percent':100*after['wer'],'delta_percentage_points':100*(after['wer']-before['wer']),'relative_change_percent':100*(after['wer']/before['wer']-1),'delta_95pct_sentence_cluster_CI':np.quantile(deltas,[.025,.975]).tolist(),'sentence_clusters':len(values),'bootstrap_resamples':10000}
    results.append(result)
retention=[r for r in results if r['language']!='tr']
aggregate={}
for label,subset in [('six_retention_languages',retention),('all_seven',results)]:
    denominator=sum(r['reference_words'] for r in subset)
    before=100*sum(r['v3_word_errors'] for r in subset)/denominator;after=100*sum(r['adapted_word_errors'] for r in subset)/denominator
    aggregate[label]={'reference_words':denominator,'v3_wer_percent':before,'adapted_wer_percent':after,'delta_percentage_points':after-before,'macro_average_delta_percentage_points':float(np.mean([r['delta_percentage_points'] for r in subset]))}
guard()
report={'dataset':'google/fleurs','revision':inventory['revision'],'split':'complete official test','full_test_utterances':5620,'checkpoint_update':400000,'checkpoint_selection':'Best Turkish DEV checkpoint, selected before this test run','baseline_revision':'541d1f99c6b0c3cd0b11a95167540bb8edefd82b','baseline':'nvidia/parakeet-tdt-0.6b-v3','normalization':saved['normalization'],'decode':'Serial greedy, original complete audio frontend and model generation settings. BF16 encoder/decoder autocast, unmerged adapted LoRA and production FP32 joint. Frozen matrices materialized on GPU without changing their values. Same conv tiling on both models.','loader_DEV_check':{'word_errors':174,'reference_words':825,'WER_percent':100*174/825},'results':results,'aggregate':aggregate,'no_test_selection_or_training':True}
with (R/'report.json').open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
with (R/'comparison.csv').open('x',encoding='utf-8',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
lines=['Full FLEURS test comparison','', 'Best Turkish DEV checkpoint: 400,000. Baseline: original NVIDIA Parakeet TDT 0.6B v3.','', '| Language | Utterances | v3 WER | Adapted WER | Change (pp) | 95% CI for change |','|---|---:|---:|---:|---:|---:|']
for r in results:
    low,high=r['delta_95pct_sentence_cluster_CI'];lines.append(f"| {r['name']} | {r['utterances']} | {r['v3_wer_percent']:.2f}% | {r['adapted_wer_percent']:.2f}% | {r['delta_percentage_points']:+.2f} | [{low:+.2f}, {high:+.2f}] |")
lines+=['','Positive change means regression. Corpus WER uses summed word errors divided by summed reference words. Confidence intervals use 10,000 paired resamples of FLEURS sentence clusters.','',f"All {sum(r['utterances'] for r in results):,} official test utterances were evaluated by both models with the same references and complete audio. No sampling, duration filtering, cropping, skipped failures or test-based checkpoint selection.",'','Scoring matches the existing project: NFC, Turkish-aware lowercase, punctuation deletion and collapsed whitespace, with accents preserved. This is not a claim of exact equivalence to the original FLEURS FST normalizer.','', 'See report.json for error counts, reference word counts, audio hours, source revisions and decoding details.']
(R/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(report,ensure_ascii=False))
