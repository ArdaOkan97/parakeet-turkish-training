"""New illustrative derivation from retained full TEST predictions; no inference."""
import json,os,sys,time
from pathlib import Path
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');F=Path(__file__).parent
sys.path.insert(0,str(P/'outputs/parakeet_turkish'))
from paired_windows_commit_telemetry_v1 import paired_sample
from resource_watchdog_v2 import sample_memory
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
start=time.monotonic();out=F/'illustrative_examples_20261029_v1';out.mkdir(exist_ok=False)
samples=[paired_sample(),paired_sample()];write(out/'admission.json',samples)
assert all(s['global_memory_status_ex']['ullAvailPageFile_bytes']>=6028384896 and s['get_performance_info_after']['physical_available_bytes']>=3278384896 for s in samples)
paths=[F/'adapted_v3/tr_test_predictions.jsonl',F/'adapted_v3/tr_test_score.json',F/'full_test_data/tr/test.jsonl']
write(out/'scope.json',{'inputs':[str(p) for p in paths],'purpose':'Illustrative good/bad output selection from retained TEST evaluation; no model or scoring rerun','max_input_each':2097152,'self_limit':536870912,'wall':180})
data=[]
for index,path in enumerate(paths):
    m=sample_memory();assert m['system_memory']['commit_available_bytes']>=3509949440 and m['system_memory']['physical_available_bytes']>=3009949440
    assert m['worker_private_memory']['private_bytes']<536870912 and time.monotonic()-start<180
    with path.open('rb') as f:raw=f.read(2097153)
    assert len(raw)<=2097152
    with (out/f'retained_input_{index}_RAW').open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    data.append(raw.decode('utf-8'))
predictions=[json.loads(x) for x in data[0].splitlines()];scores=json.loads(data[1]);audio={x['source_id']:x for x in map(json.loads,data[2].splitlines())}
details={x['source_id']:x for x in scores['details']};rows=[]
for x in predictions:
    d=details[x['source_id']];a=audio[x['source_id']]
    assert a['reference']==x['reference'] and a['audio_sha256']==x['audio_sha256']
    rows.append({'row':x['row_index'],'sentence_id':x['sentence_id'],'reference':x['reference'],'hypothesis':x['hypothesis'],'word_errors':d['word_errors'],'reference_words':d['reference_words'],'wer_percent':100*d['word_errors']/d['reference_words'],'duration_seconds':x['duration_seconds'],'audio_filepath':a['audio_filepath']})
assert len(rows)==743
summary={'total':len(rows),'zero_word_error_utterances':sum(r['word_errors']==0 for r in rows),'at_most_10_percent_WER_utterances':sum(r['wer_percent']<=10 for r in rows),'at_least_50_percent_WER_utterances':sum(r['wer_percent']>=50 for r in rows),'corpus_WER_percent':100*scores['wer']}
good=[r for r in rows if r['word_errors']==0 and 12<=r['reference_words']<=22][:15]
bad=sorted([r for r in rows if 12<=r['reference_words']<=25 and r['word_errors']>0],key=lambda r:r['wer_percent'],reverse=True)[:16]
names=[r for r in rows if r['row'] in (0,1,2)]
write(out/'all_retained_rows.json',{'summary':summary,'rows':rows})
write(out/'candidate_examples.json',{'summary':summary,'good':good,'bad':bad,'names':names})
print(json.dumps({'summary':summary,'good':good,'bad':bad,'names':names},ensure_ascii=False))
