"""Seeded random passed/uncertain/rejected listening review, CPU-only."""
import os,sys,pathlib,json,random,hashlib,sqlite3,datetime,time,importlib.util,base64,html
ROOT=pathlib.Path(r'E:\gpu_training_scratchpad');PROJECT=pathlib.Path(__file__).resolve().parents[2]
SOURCE=ROOT/'analysis/whisper_tiny_GPU_batch_v1';OUT=ROOT/'analysis/language_screen_listening_v1'
SEED=202610162310;N=10;GROUPS=['pass','uncertain_review','below_cutoff']
sys.path.insert(0,str(PROJECT/'work/research'))

def write(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def select():
    OUT.mkdir(exist_ok=False,parents=True);reservoirs={g:[] for g in GROUPS};counts={g:0 for g in GROUPS};rngs={g:random.Random(SEED+i) for i,g in enumerate(GROUPS)}
    decisions=SOURCE/'staged_full_v11_duration_ordered_decisions.jsonl'
    with decisions.open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line);g=r['decision']
            if g not in reservoirs:continue
            counts[g]+=1;pool=reservoirs[g]
            if len(pool)<N:pool.append(r['segment_id'])
            else:
                j=rngs[g].randrange(counts[g])
                if j<N:pool[j]=r['segment_id']
    for g in GROUPS:assert len(reservoirs[g])==N;rngs[g].shuffle(reservoirs[g])
    selection=dict(at_utc=now(),seed=SEED,per_group=N,method='Uniform reservoir sample without replacement within each language-screen group; no content-based replacement',group_population_counts=counts,selected_IDs=reservoirs,decision_source=str(decisions),decision_source_SHA256=sha(decisions),rejected_means='below_cutoff language group; excludes source-coverage and oversized-member quarantines')
    write('pinned_selection.json',selection)
    wanted={sid.encode() for ids in reservoirs.values() for sid in ids};scores={};digest=hashlib.sha256();ledger=SOURCE/'staged_full_v11_duration_ordered_ledger.jsonl';prefix=b'{"segment_id":"'
    with ledger.open('rb') as f:
        for raw in f:
            digest.update(raw);assert raw.startswith(prefix);sid=raw[len(prefix):raw.find(b'"',len(prefix))]
            if sid in wanted:scores[sid.decode()]=json.loads(raw)
    sealed=json.loads((SOURCE/'screen90_v11_summary.json').read_text());assert digest.hexdigest()==sealed['ledger_SHA256'] and len(scores)==N*3
    db=sqlite3.connect((ROOT/'indexes/alania_metadata_candidates_v1.sqlite').as_uri()+'?mode=ro',uri=True);rows=[]
    for g in GROUPS:
        for i,sid in enumerate(reservoirs[g],1):
            r=scores[sid];assert r['language_screening_decision']==g
            text=db.execute('SELECT transcript,start_s,duration_s,recording_id,shard FROM candidates WHERE segment_id=?',(sid,)).fetchone();assert text and text[3]==r['recording_id'] and text[4]==r['shard'] and text[1]==r['start_s'];r.update(transcript=text[0],sample_group=g,sample_number=i);rows.append(r)
    db.close();write('selected_examples.json',rows);write('source_verification.json',dict(ledger_SHA256=digest.hexdigest(),matches_sealed_ledger=True,selected_rows=len(rows),transcripts='Original Alania supplied labels, unchanged',selection_pinned_before_transcript_or_audio_review=True))
    return rows

def prepare_audio(rows):
    spec=importlib.util.spec_from_file_location('original_packet_excerpt',ROOT/'analysis/flagged_listening_v1/build_listening.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.AREA=OUT;module.base.AREA=OUT;module.paced.AREA=OUT
    import psutil
    proc=psutil.Process();proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    class Guard:
        def __init__(self):self.started=time.monotonic();self.file_bytes_read=0;self.paused_seconds=0.;self.last=0.;self.sample_at=0.;self.high=0;self.clip_started=0.
        def wait(self):
            at=time.monotonic()
            if at-self.last>.5:
                self.last=at;m=module.base.memory();owner=json.loads((PROJECT/'work/gpu_training_owner.json').read_text());assert not owner.get('actual_GPU_running'),'GPU work started; stop audio preparation at boundary'
                if at-self.sample_at>=5:self.sample_at=at;self.high=self.high+1 if m['commit_ratio']>=.9 else 0
                if m['commit_ratio']>=.95 or self.high>=3 or m['commit_headroom_bytes']<2_000_000_000 or proc.memory_info().private>2_000_000_000:raise RuntimeError('Unchanged CPU resource protection')
                if self.clip_started and at-self.clip_started>90:raise TimeoutError('90s clip preparation bound')
        def pace(self,n):
            self.wait();self.file_bytes_read+=n
            if self.file_bytes_read>512_000_000:raise RuntimeError('Declared512MB read budget exceeded')
            delay=self.file_bytes_read/(1024*1024)-(time.monotonic()-self.started)
            if delay>0:time.sleep(delay)
    guard=Guard();guard.wait();cache=[]
    for i,r in enumerate(rows,1):
        guard.clip_started=time.monotonic();audio,proof=module.excerpt(r,guard);cache.append(dict(segment_id=r['segment_id'],audio_b64=base64.b64encode(audio).decode(),proof=proof));print(json.dumps(dict(audio_ready=i,total=len(rows),group=r['sample_group'],segment_id=r['segment_id'])),flush=True)
        with (OUT/'original_opus_cache.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(cache[-1],separators=(',',':'))+'\n')
    write('audio_verification.json',dict(at_utc=now(),clips=len(rows),all_original_compressed_payloads_identical=True,all_clips_decoded=True,source_archives_unchanged=True,no_reencoding=True,no_permanent_WAV=True,GPU_used=False,threads=1,read_bytes=guard.file_bytes_read,read_cap_bytes_per_s=1048576,cache_SHA256=sha(OUT/'original_opus_cache.jsonl'),proofs=[dict(segment_id=x['segment_id'],**x['proof']) for x in cache]))
    return {r['segment_id']:r for r in cache}

CSS='''
:root{font-family:Inter,Segoe UI,system-ui,sans-serif;color:#18313a;background:#f4f6f5;--green:#218364;--amber:#c28a27;--red:#c55251}*{box-sizing:border-box}body{margin:0}main{max-width:1100px;margin:auto;padding:42px 24px}header{margin-bottom:28px}h1{font-size:clamp(28px,4vw,43px);letter-spacing:-.035em;line-height:1.15;margin:8px 0 18px}p{line-height:1.65}.eyebrow{letter-spacing:.11em;font-size:11px;font-weight:750;text-transform:uppercase;color:#4b7167}.intro{color:#536762;max-width:930px}.rule{background:#e8eeeb;padding:16px 20px;border-radius:10px;font-size:14px}.legend{display:flex;gap:18px;flex-wrap:wrap;margin:14px 0;color:#536762;font-size:13px}.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px}.tr{background:var(--green)}.uncertain{background:var(--amber)}.other{background:var(--red)}.toolbar{position:sticky;top:0;z-index:2;background:#f4f6f5f5;padding:14px 0;display:flex;gap:8px;align-items:center;flex-wrap:wrap;border-bottom:1px solid #dce3df;margin-bottom:22px}button,select,input{font:inherit;border:1px solid #c9d4ce;border-radius:7px;background:white;padding:9px 12px}button{cursor:pointer}button.active{color:white;background:#244f43;border-color:#244f43}input{flex:1;min-width:180px}label,#visible{font-size:12px;color:#536762}.clip{background:white;border:1px solid #dce3df;border-radius:14px;padding:24px;margin:0 0 20px;box-shadow:0 3px 12px #17332904}.top{display:flex;justify-content:space-between;gap:20px;align-items:start}.group{font-size:12px;font-weight:750;text-transform:uppercase;letter-spacing:.06em}.pass .group{color:var(--green)}.uncertain_review .group{color:#92600c}.below_cutoff .group{color:var(--red)}h2{font-size:19px;margin:7px 0 4px}.duration{font-size:12px;color:#6a7b75}.score{text-align:right;white-space:nowrap}.score strong{font-size:28px;letter-spacing:-.04em}.score span{display:block;font-size:11px;color:#60746a}.share{display:flex;height:10px;background:#edf0ee;border-radius:7px;overflow:hidden;margin:19px 0 9px}.shares{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:#556c61;margin-bottom:19px}audio{width:100%;height:42px}h3{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:#60746a;margin:21px 0 7px}.transcript{font-size:19px;line-height:1.75;margin:0 0 15px;overflow-wrap:anywhere}.chips{display:flex;gap:6px;flex-wrap:wrap;margin:10px 0 16px}.chips button{font-size:11px;border:0;background:#f1f4f2;padding:6px 9px;border-bottom:3px solid #ddd}.chips button[data-cat=tr]{border-color:var(--green)}.chips button[data-cat=uncertain]{border-color:var(--amber)}.chips button[data-cat=other]{border-color:var(--red)}details{font-size:12px;color:#60746a;margin-top:16px}summary{cursor:pointer}.table{overflow:auto}table{width:100%;border-collapse:collapse;text-align:left;font-size:12px;margin:10px 0}th,td{padding:8px;border-bottom:1px solid #e2e9e4;white-space:nowrap}.source{overflow-wrap:anywhere;font-size:11px}footer{font-size:12px;color:#60746a;padding:12px 0 25px}[hidden]{display:none!important}a{color:#385f51}@media(max-width:600px){main{padding:25px 13px}.clip{padding:18px}.score strong{font-size:23px}.transcript{font-size:17px}.toolbar{gap:6px}.toolbar button{padding:8px}.top{gap:10px}h2{font-size:17px}}
'''

def render(rows,audio):
    esc=lambda x:html.escape(str(x),quote=True);cards=[];names={'pass':'Passed','uncertain_review':'Uncertain','below_cutoff':'Rejected'};langnames={'tr':'Turkish','ar':'Arabic','en':'English','de':'German','es':'Spanish','fr':'French','it':'Italian','pt':'Portuguese','ko':'Korean','ru':'Russian','sv':'Swedish','fa':'Persian','el':'Greek','he':'Hebrew','az':'Azerbaijani','uk':'Ukrainian','nl':'Dutch','pl':'Polish'}
    for j,r in enumerate(rows,1):
        a=audio[r['segment_id']];g=r['sample_group'];lo=r['estimated_Turkish_share_lower'];hi=r['estimated_Turkish_share_upper'];unknown=r['uncertain_speech_fraction'];other=r['confident_other_speech_fraction'];score=f'{lo:.1%}' if abs(lo-hi)<1e-8 else f'{lo:.1%}–{hi:.1%}'
        table=[];chips=[]
        for w in r['windows']:
            language=langnames.get(w['top_language'],w['top_language'].upper());label={'tr':'Turkish','other':'Other language','uncertain':'Uncertain'}[w['category']]
            table.append(f"<tr><td>{w['start_s']:.1f}–{w['end_s']:.1f}s</td><td>{w['speech_s']:.2f}s</td><td>{esc(language)}</td><td>{w['top_probability']:.1%}</td><td>{w['Turkish_probability']:.1%}</td><td>{label}</td></tr>")
            seek=max(0,w['start_s']+r['start_s']-a['proof']['copied_packet_start_s']);chips.append(f'<button data-seek="{seek:.4f}" data-cat="{w["category"]}" aria-label="Play window starting at {w["start_s"]:.1f} seconds">{w["start_s"]:.0f}–{w["end_s"]:.1f}s · {esc(language)}</button>')
        cards.append(f'''<article class="clip {g}" data-group="{g}" id="clip-{j}"><div class="top"><div><span class="group">{names[g]} · random sample {r['sample_number']:02}</span><h2>Listen to the original audio</h2><span class="duration">{r['duration_s']:.2f}s clip · {r['speech_seconds']:.2f}s detected speech</span></div><div class="score"><strong>{score}</strong><span>estimated Turkish speech share</span></div></div><div class="share" aria-label="{lo:.1%} confidently Turkish, {unknown:.1%} uncertain, {other:.1%} other"><span class="tr" style="width:{lo*100:.5f}%"></span><span class="uncertain" style="width:{unknown*100:.5f}%"></span><span class="other" style="width:{other*100:.5f}%"></span></div><div class="shares"><span>Turkish <b>{lo:.1%}</b></span><span>Uncertain <b>{unknown:.1%}</b></span><span>Other <b>{other:.1%}</b></span></div><audio controls preload="none" aria-label="{names[g]} sample {r['sample_number']}" src="data:audio/webm;base64,{a['audio_b64']}"></audio><h3>Supplied transcript · original Alania label</h3><p class="transcript" dir="auto" lang="tr">{esc(r['transcript'])}</p><div class="chips">{''.join(chips)}</div><details><summary>Window scores and source details</summary><div class="table"><table><thead><tr><th>Within clip</th><th>Speech</th><th>Top language</th><th>Top confidence</th><th>Tiny P(Turkish)</th><th>Duration bucket</th></tr></thead><tbody>{''.join(table)}</tbody></table></div><p>Windows classify speech as Turkish or another language when their top score is at least 70%. The displayed share is weighted by detected speech duration, not the average confidence.</p><p class="source">Segment {esc(r['segment_id'])}<br>Recording {esc(r['recording_id'])} · shard {r['shard']} · original interval {r['start_s']:.3f}–{r['start_s']+r['duration_s']:.3f}s.<br>Original Opus packets copied without re-encoding; maximum clip-boundary difference {a['proof']['maximum_boundary_difference_s']*1000:.0f}ms.</p></details></article>''')
    page='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Random Turkish language-screen samples</title><style>'''+CSS+'''</style></head><body><main><header><span class="eyebrow">Turkish dataset · listening review</span><h1>Passed, uncertain, and rejected.</h1><p class="intro">30 randomly selected clips: 10 from each completed language-screen group. Listen to the original audio alongside its supplied transcript. These are uniform samples within each group, not a proportional sample of the dataset.</p><div class="rule"><strong>Your filter: at least 90% Turkish speech.</strong> A range such as 70–100% means 70% of detected speech was confidently classified Turkish and 30% is unresolved. It is not proof that the unresolved part is another language. Passed: lower estimate ≥90%; uncertain: the range crosses 90%; rejected: even the upper estimate is below 90%.</div><div class="legend"><span><i class="dot tr"></i>Confidently Turkish</span><span><i class="dot uncertain"></i>Unresolved language</span><span><i class="dot other"></i>Confidently another language</span></div><p class="intro">These are Whisper-tiny estimates from four-second windows, not human labels. Language changes inside a window can be missed. Transcripts below are the original supplied labels; no new transcription was generated.</p></header><nav class="toolbar" aria-label="Review controls"><button class="active" data-filter="all">All 30</button><button data-filter="pass">Passed 10</button><button data-filter="uncertain_review">Uncertain 10</button><button data-filter="below_cutoff">Rejected 10</button><input id="search" type="search" aria-label="Search transcript or ID" placeholder="Search transcript or ID"><label>Speed <select id="speed"><option value=".75">0.75×</option><option value="1" selected>1×</option><option value="1.25">1.25×</option><option value="1.5">1.5×</option><option value="2">2×</option></select></label><span id="visible" aria-live="polite"></span></nav>'''+''.join(cards)+f'''<footer><p>Group populations: 74,957 passed · 44,703 uncertain · 5,039 rejected. Rejected here means the language-screen group; unreadable and oversized sources are not sampled. No files were deleted and this page does not admit data to training.</p><p>Random seed {SEED}; IDs pinned before viewing transcripts or audio. Scores reused from the completed screen; multilingual Whisper-tiny language detection only. All audio is embedded, so the HTML also works offline.</p><p>Audio: YODAS3 @81d802f45a1c1e1f5331260484dacf1fbea81981. Supplied labels: Alania @8b0fdd976ed9668b64f5ee0696317a8f241ce39c.</p></footer></main><script>
const cards=[...document.querySelectorAll('.clip')];let group='all';
function filter(){{const q=document.querySelector('#search').value.toLocaleLowerCase('tr');let n=0;for(const c of cards){{c.hidden=!((group==='all'||c.dataset.group===group)&&c.textContent.toLocaleLowerCase('tr').includes(q));if(c.hidden)c.querySelector('audio').pause();else n++}}document.querySelector('#visible').textContent=n+' shown'}}
document.querySelector('#search').addEventListener('input',filter);for(const b of document.querySelectorAll('[data-filter]'))b.addEventListener('click',()=>{{group=b.dataset.filter;for(const x of document.querySelectorAll('[data-filter]'))x.classList.toggle('active',x===b);filter()}});
document.querySelector('#speed').addEventListener('change',e=>{{for(const a of document.querySelectorAll('audio'))a.playbackRate=Number(e.target.value)}});
for(const a of document.querySelectorAll('audio'))a.addEventListener('play',()=>{{for(const other of document.querySelectorAll('audio'))if(a!==other)other.pause()}});
for(const b of document.querySelectorAll('[data-seek]'))b.addEventListener('click',()=>{{const a=b.closest('article').querySelector('audio');a.currentTime=Number(b.dataset.seek);a.play()}});filter();
</script></body></html>'''
    target=OUT/'random-language-screen-samples.html';target.write_text(page,encoding='utf-8');write('page_receipt.json',dict(at_utc=now(),HTML=str(target),SHA256=sha(target),bytes=target.stat().st_size,cards=len(cards),groups={g:N for g in GROUPS},self_contained_audio=True,no_new_ASR=True));print(json.dumps(dict(complete=True,HTML=str(target),samples=len(cards))),flush=True)

if __name__=='__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
    rows=select();audios=prepare_audio(rows);render(rows,audios)
