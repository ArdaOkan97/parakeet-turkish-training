"""CPU-only, disk-backed TRAIN label audit and isolated append-only BPE candidate.

Never writes source catalogs, incumbent tokenizers, model/checkpoint/configuration.
"""
from __future__ import annotations
import argparse, array, collections, ctypes, datetime, hashlib, json, os
from pathlib import Path
import shutil, sqlite3, sys, threading, time, unicodedata

ROOT = Path(r'E:\gpu_training_scratchpad')
PROJECT = Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20')
AREA = ROOT / 'analysis/Turkish_labels_BPE_circumflex_v1'
sys.path.insert(0, str(PROJECT / 'work/research'))
from cpu_preparation_guard_v2 import memory
from ASR18_data_io_guard_v2 import hard_job
import psutil

INCUMBENT = PROJECT / 'work/hf_turkish_bpe_candidate_512_v1/tokenizer.json'
INCUMBENT_SHA = '6c656b9dc5f85e3dd2c2abe16983908e5d2293da1b2ae096e92e1dc5551df327'
BROAD = ROOT / 'indexes/BPE_broad_weak_labels_v1.sqlite'
BROAD_LABELS = ROOT / 'indexes/BPE_broad_labels_8711_v1.sqlite'
CLEAN = PROJECT / 'work/data/TR_expanded_WorldSpeech_index_v1/rows.jsonl'
CLEAN_LABELS = ROOT / 'indexes/BPE_full_decoder_expanded_labels_v1/labels.sqlite3'
CHARS = 'âîûÂÎÛ'
COUNT = 512

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def canonical(value): return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
def write(name, value):
    path = AREA / name
    tmp = path.with_suffix(path.suffix + '.writing')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    tmp.replace(path)
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()
def record(path):
    path = Path(path); st = path.stat()
    return {'path': str(path), 'bytes': st.st_size, 'mtime_ns': st.st_mtime_ns, 'sha256': sha(path)}
def stat(path):
    path = Path(path); st = path.stat()
    return {'path': str(path), 'bytes': st.st_size, 'mtime_ns': st.st_mtime_ns}
def ro(path):
    db = sqlite3.connect(Path(path).as_uri() + '?mode=ro&immutable=1', uri=True)
    db.execute('PRAGMA query_only=ON'); db.execute('PRAGMA cache_size=-4096')
    db.execute('PRAGMA mmap_size=0'); db.execute('PRAGMA temp_store=MEMORY')
    return db

class Guard:
    def __init__(self):
        config = json.loads((ROOT / 'storage_config.json').read_text(encoding='utf-8-sig'))
        assert Path(config['storage_root']).resolve() == ROOT.resolve() and not config['allow_fallback_outside_root']
        for k in ('TEMP', 'TMP', 'HF_HOME', 'TORCH_EXTENSIONS_DIR', 'PYTHONPYCACHEPREFIX'):
            assert Path(os.environ[k]).resolve().is_relative_to(ROOT.resolve()), k
        assert os.environ['CUDA_VISIBLE_DEVICES'] == '-1'
        assert all(os.environ[k] == '1' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'))
        assert os.environ['TOKENIZERS_PARALLELISM'] == 'false'
        self.process = psutil.Process(); self.process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        self.job = hard_job()
        # Tighten the existing Windows job to 1.8GB for the sole worker. The
        # measured PowerShell parent gets the remaining200MB whole-family budget.
        from ctypes import wintypes as w
        class JobIO(ctypes.Structure):
            _fields_=[(x,ctypes.c_ulonglong) for x in ('r','w','o','rb','wb','ob')]
        class Basic(ctypes.Structure):
            _fields_=[('p',ctypes.c_longlong),('j',ctypes.c_longlong),('flags',w.DWORD),('min',ctypes.c_size_t),('max',ctypes.c_size_t),('active',w.DWORD),('affinity',ctypes.c_size_t),('priority',w.DWORD),('scheduling',w.DWORD)]
        class Limits(ctypes.Structure):
            _fields_=[('basic',Basic),('io',JobIO),('process_memory',ctypes.c_size_t),('job_memory',ctypes.c_size_t),('peak_process',ctypes.c_size_t),('peak_job',ctypes.c_size_t)]
        limits=Limits();limits.basic.flags=0x100|0x200;limits.process_memory=1_800_000_000;limits.job_memory=1_800_000_000
        api=ctypes.WinDLL('kernel32',use_last_error=True)
        api.SetInformationJobObject.argtypes=[w.HANDLE,ctypes.c_int,ctypes.c_void_p,w.DWORD];api.SetInformationJobObject.restype=w.BOOL
        if not api.SetInformationJobObject(self.job,9,ctypes.byref(limits),ctypes.sizeof(limits)):raise ctypes.WinError(ctypes.get_last_error())
        immediate=self.process.parent();assert immediate
        self.parents=[immediate]
        if immediate.name().lower()=='python.exe':self.parents.append(immediate.parent())
        assert self.parents[-1] and self.parents[-1].name().lower() in ('powershell.exe','pwsh.exe')
        self.parent_identity=[{'PID':p.pid,'create_time':p.create_time(),'name':p.name()} for p in self.parents]
        self.started = time.monotonic(); self.max_seconds = 3600
        self.bytes = 0; self.maximum_bytes = 16_000_000_000; self.deadline = time.monotonic();self.pending_bytes=0
        counters=self.process.io_counters();self.initial_IO=counters.read_bytes+counters.write_bytes
        self.peak = 0; self.last_publish = 0; self.phase = 'starting'
        self.main_identity = {'PID': os.getpid(), 'create_time': self.process.create_time()}
        threading.Thread(target=self.watchdog, daemon=True).start()
    def publish(self, **extra):
        m = memory(); priv = self.process.memory_info().private; self.peak = max(self.peak, priv)
        write('current_status.json', {'owner': '/root/label_bpe_audit', 'phase': self.phase, 'at_utc': now(),
            **self.main_identity, 'parent_identity':self.parent_identity,'parent_private_bytes':sum(p.memory_info().private for p in self.parents),
            'private_bytes': priv, 'peak_private_bytes': self.peak, 'memory': m,
            'elapsed_seconds': time.monotonic() - self.started, 'paced_logical_bytes': self.bytes,
            'actual_process_IO_bytes_since_start':self.process.io_counters().read_bytes+self.process.io_counters().write_bytes-self.initial_IO,
            'CPU_only': True, 'thread_caps': 1, 'GPU_used': False, 'active_sources_modified': False, **extra})
        self.last_publish = time.monotonic()
    def check(self):
        m = memory(); priv = self.process.memory_info().private; self.peak = max(self.peak, priv)
        parent_private=sum(p.memory_info().private for p in self.parents)
        if (m['commit_ratio'] >= .95 or m['commit_headroom_bytes'] < 2_000_000_000 or priv >= 1_800_000_000
            or parent_private>200_000_000 or parent_private+priv>=2_000_000_000):
            raise RuntimeError('Unchanged system/private hard resource guard')
        if shutil.disk_usage(ROOT).free < 100_000_000_000: raise RuntimeError('E:100GB reserve')
        counters=self.process.io_counters()
        if counters.read_bytes+counters.write_bytes-self.initial_IO>16_000_000_000:raise RuntimeError('Finite16GB actual process IO quota')
        if time.monotonic() - self.started > self.max_seconds: raise RuntimeError('Finite one-hour work budget')
        while m['commit_ratio'] >= .90:
            self.publish(resource_yield=True); time.sleep(2); m = memory()
            if m['commit_ratio'] >= .95 or m['commit_headroom_bytes'] < 2_000_000_000: raise RuntimeError('System resource guard while yielding')
        if time.monotonic() - self.last_publish > 15: self.publish()
    def pace(self, amount):
        self.bytes += amount;self.pending_bytes+=amount
        if self.bytes > self.maximum_bytes: raise RuntimeError('Finite16GB logical I/O quota')
        if self.pending_bytes<262144:return
        self.check()
        self.deadline = max(time.monotonic(), self.deadline) + self.pending_bytes / (32 * 1024 * 1024)
        self.pending_bytes=0
        delay = self.deadline - time.monotonic()
        if delay > 0: time.sleep(delay)
    def watchdog(self):
        while True:
            time.sleep(2)
            try:
                m = memory(); priv = self.process.memory_info().private; self.peak = max(self.peak, priv)
                parent_private=sum(p.memory_info().private for p in self.parents)
                if (m['commit_ratio'] >= .95 or m['commit_headroom_bytes'] < 2_000_000_000 or priv >= 1_800_000_000
                    or parent_private>200_000_000 or parent_private+priv>=2_000_000_000):
                    self.publish(resource_stop=True); os._exit(75)
            except BaseException: os._exit(76)

def words(tokens):
    current = []
    for piece in tokens:
        if '▁' in piece and current:
            yield tuple(current); current = []
        current.append(piece)
    if current: yield tuple(current)

def permitted(pair, vocab):
    piece = ''.join(pair); letters = piece[1:] if piece.startswith('▁') else piece
    return (1 < len(letters) <= 24 and '▁' not in letters and piece not in vocab
        and letters.isalpha() and all('LATIN' in unicodedata.name(c, '') for c in letters))

def pair_counts(sequence, vocab):
    return collections.Counter(canonical(p) for p in zip(sequence, sequence[1:]) if permitted(p, vocab))

def corpus(guard):
    source = ro(BROAD)
    source.execute('ATTACH DATABASE ? AS targets', (BROAD_LABELS.as_uri() + '?mode=ro&immutable=1',))
    count = 0
    for row in source.execute('SELECT c.ordinal,c.unique_id,c.dataset,c.original_split,c.text,c.text_raw,c.text_spoken,l.target_ids_u32,l.text_sha256 '
                             'FROM clips c JOIN targets.labels l ON l.ordinal=c.ordinal ORDER BY c.ordinal'):
        ordinal, sid, dataset, split, text, raw, spoken, blob, text_sha = row
        assert ordinal == count + 1; count += 1
        yield 'broad', ordinal, sid, dataset, split, text, {'text_raw': raw, 'text_spoken': spoken}, blob, text_sha
    source.close(); assert count == 2631115, count
    labels = ro(CLEAN_LABELS); cursor = iter(labels.execute('SELECT ordinal,source_id,text_sha,label_ids FROM labels ORDER BY ordinal'))
    count = 0
    with CLEAN.open('rb') as stream:
        for rawline in stream:
            guard.pace(len(rawline)); row = json.loads(rawline); count += 1
            ordinal, sid, text_sha, blob = next(cursor)
            assert ordinal == count and sid == row['source_id']
            assert row['language'] == 'tr' and row['dataset_split'] in ('train','validated')
            yield 'cleaner', ordinal, sid, row['dataset'], row['dataset_split'], row['text'], {
                'reference': row.get('reference'), 'reference_normalized': row.get('reference_normalized')}, blob, text_sha
    assert count == 352768; assert next(cursor, None) is None; labels.close()

def fixture(tokenizer):
    cases = ['â î û Â Î Û', 'hâlâ millî sükûn HÂLÂ MİLLÎ SÜKÛN',
             'a\u0302 i\u0302 u\u0302 A\u0302 I\u0302 U\u0302', 'a i u A I U',
             '<unk> <blank> <|tr|>', 'kâr kar hâl hal millî milli ruhî ruhi sükûn sukun']
    return [{'text': t, 'normalized': tokenizer.normalizer.normalize_str(t),
             'tokens': tokenizer.encode(t).tokens, 'ids': tokenizer.encode(t).ids,
             'decoded_with_specials': tokenizer.decode(tokenizer.encode(t).ids, skip_special_tokens=False)} for t in cases]

def publish_coverage(base, coverage):
    directory=AREA/'coverage_only_8712'
    if directory.exists():
        manifest=json.loads((directory/'manifest.json').read_text(encoding='utf8'))
        for item in manifest['artifacts'].values():assert record(item['path'])==item
        return manifest
    directory.mkdir()
    updated=json.loads(coverage.to_str())
    # Backend serialization is allowed to canonicalize formatting, never token IDs or semantics.
    for piece,identifier in base['model']['vocab'].items():assert updated['model']['vocab'][piece]==identifier
    assert updated['model']['vocab']['Û']==8711 and len(updated['model']['vocab'])==8712
    assert updated['model']['merges']==base['model']['merges']
    for k in base:
        if k!='model':assert updated[k]==base[k],k
    for c in CHARS:
        encoded=coverage.encode(c);assert 0 not in encoded.ids and coverage.decode(encoded.ids,skip_special_tokens=False)==c
        decomposed=unicodedata.normalize('NFD',c)
        assert coverage.encode(decomposed).ids==encoded.ids
    for text in ('hâlâ millî sükûn','HÂLÂ MİLLÎ SÜKÛN'):
        assert coverage.decode(coverage.encode(text).ids,skip_special_tokens=False)==text
    incumbent=coverage.__class__.from_str(json.dumps(base,ensure_ascii=False))
    for token in base['added_tokens']:
        assert coverage.encode(token['content']).ids==incumbent.encode(token['content']).ids
    (directory/'tokenizer.json').write_text(json.dumps(updated,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    (directory/'vocab.json').write_text(json.dumps(updated['model']['vocab'],ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    (directory/'vocab.tsv').write_text('\n'.join(str(i)+'\t'+p for p,i in sorted(updated['model']['vocab'].items(),key=lambda x:x[1]))+'\n',encoding='utf8')
    (directory/'merges.txt').write_text('#version: 0.2\n'+'\n'.join(' '.join(pair) for pair in updated['model']['merges'])+'\n',encoding='utf8')
    config=PROJECT/'work/models/lora_20261010_01_merged/tokenizer_config.json'
    (directory/'tokenizer_config.json').write_bytes(config.read_bytes())
    (directory/'unicode_roundtrip_tests.json').write_text(json.dumps(fixture(coverage),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    (directory/'README.txt').write_text('Turkish circumflex coverage-only tokenizer,8712 IDs.\n\n'
        'Usable directly: from tokenizers import Tokenizer; t=Tokenizer.from_file("tokenizer.json")\n'
        'All existing8711 token IDs and13988 merge ranks are fixed. Added literalÛ has ID8711.\n'
        'â8024 î8001 û8113 Â8180 Î8087 Û8711; blank8192 unchanged.\n'
        'NFC and decomposed Unicode variants encode identically; case/diacritics are preserved.\n'
        'No transcripts were corrected or rewritten. This is an isolated tokenizer candidate, not an active model.\n'
        'The accompanying original ParakeetTokenizer config is supplied for metadata compatibility;\n'
        'Transformers AutoTokenizer loading and model migration were not executed.\n'
        'A model needs an explicitly initialized extra embedding/output row and decoder vocab_size8712.\n'
        'TDT duration rows must move from8711..8715 to8712..8716; joint width becomes8717.\n'
        'Keep mature model/optimizer state, rederive TRAIN labels/token denominators and revalidate geometry/retention.\n'
        'Unchanged multilingual IDs do not guarantee unchanged multilingual token sequences or ASR quality.\n'
        'This8712 artifact does not claim all-corpus byte coverage or any measured model gain.\n',encoding='utf8')
    manifest={'kind':'Turkish_circumflex_coverage_only_BPE8712_v1','at_utc':now(),'incumbent':record(INCUMBENT),
        'vocabulary':8712,'append_only_character':{'Û':8711},'all_existing_8711_IDs_and_13988_merges_fixed':True,
        'normalizer_pretokenizer_decoder_added_tokens_special_IDs_fixed':True,
        'NFC_NFD_all_six_caseforms_roundtrip_passed':True,'production_activated':False,'full_corpus_validation_complete':False,
        'artifacts':{n:record(directory/n) for n in ['tokenizer.json','vocab.json','vocab.tsv','merges.txt','tokenizer_config.json','unicode_roundtrip_tests.json','README.txt']}}
    (directory/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return manifest

def source_audit(db, guard, incumbent, coverage):
    guard.phase = 'all_TRAIN_source_label_audit_and_disk_word_counts'
    metrics = {}; examples = []; all_digest = hashlib.sha256(); counts = collections.Counter(); batch = collections.Counter()
    db.execute('CREATE TABLE words(id INTEGER PRIMARY KEY,seq TEXT UNIQUE NOT NULL,freq INTEGER NOT NULL)')
    for pool, ordinal, sid, dataset, split, text, fields, blob, text_sha in corpus(guard):
        assert isinstance(text, str) and text
        guard.pace(sum(len(v.encode()) for v in [sid,dataset,split,text,*fields.values()] if isinstance(v,str)) + len(blob) + 128)
        ids = array.array('I'); ids.frombytes(blob)
        assert ids.itemsize == 4 and sys.byteorder == 'little'
        old = incumbent.encode(text, add_special_tokens=True)
        assert old.ids == list(ids), ('Existing TRAIN label mismatch', pool, ordinal)
        assert hashlib.sha256(text.encode()).hexdigest() == text_sha
        cov = coverage.encode(text, add_special_tokens=True)
        normalized = incumbent.normalizer.normalize_str(text)
        key = pool + ':' + dataset
        m = metrics.setdefault(key, {'rows': 0, 'source_char_counts': collections.Counter(), 'normalized_char_counts': collections.Counter(),
            'rows_with_circumflex': 0, 'existing_unknown_rows': 0, 'existing_unknown_tokens': 0, 'coverage_unknown_rows': 0,
            'existing_label_tokens': 0, 'coverage_tokens': 0, 'coverage_changes': 0, 'label_hash_and_ids_exact_rows': 0,
            'normalizer_circumflex_removed_rows': 0, 'source_field_char_counts': {}, 'source_field_circumflex_removed_rows': {}})
        m['rows'] += 1; m['label_hash_and_ids_exact_rows'] += 1; m['existing_label_tokens'] += len(old.ids); m['coverage_tokens'] += len(cov.ids)
        m['source_char_counts'].update(c for c in text if c in CHARS); m['normalized_char_counts'].update(c for c in normalized if c in CHARS)
        m['rows_with_circumflex'] += any(c in text for c in CHARS)
        m['existing_unknown_rows'] += 0 in old.ids; m['existing_unknown_tokens'] += old.ids.count(0); m['coverage_unknown_rows'] += 0 in cov.ids
        m['coverage_changes'] += old.ids != cov.ids
        m['normalizer_circumflex_removed_rows'] += any(text.count(c) > normalized.count(c) for c in CHARS)
        for field, value in fields.items():
            if value is None: continue
            assert isinstance(value, str)
            m['source_field_char_counts'].setdefault(field, collections.Counter()).update(c for c in value if c in CHARS)
            # Compare after canonical composition and case folding to distinguish case policy from loss.
            a = unicodedata.normalize('NFC', value).casefold(); b = unicodedata.normalize('NFC', text).casefold()
            m['source_field_circumflex_removed_rows'][field] = m['source_field_circumflex_removed_rows'].get(field, 0) + int(any(a.count(c) > b.count(c) for c in 'âîû'))
        if len(examples) < 36 and any(c in text for c in CHARS) and (len([x for x in examples if x['group']==key]) < 4 or 'Û' in text):
            examples.append({'group': key, 'ordinal': ordinal, 'source_id': sid, 'text': text, 'normalized': normalized,
                'existing_decode': incumbent.decode(old.ids, skip_special_tokens=False), 'coverage_decode': coverage.decode(cov.ids, skip_special_tokens=False)})
        for seq in words(cov.tokens): batch[canonical(seq)] += 1
        all_digest.update((canonical([pool, ordinal, sid, dataset, split, text])+'\n').encode())
        counts[pool] += 1
        if sum(counts.values()) % 512 == 0:
            db.executemany('INSERT INTO words(seq,freq) VALUES (?,?) ON CONFLICT(seq) DO UPDATE SET freq=freq+excluded.freq', batch.items())
            db.commit(); batch.clear()
        if sum(counts.values()) % 10000 == 0:
            guard.publish(rows=sum(counts.values()), counts=dict(counts))
    if batch:
        db.executemany('INSERT INTO words(seq,freq) VALUES (?,?) ON CONFLICT(seq) DO UPDATE SET freq=freq+excluded.freq', batch.items()); db.commit()
    result = {'counts': dict(counts), 'groups': metrics, 'examples': examples, 'source_ordered_text_sha256': all_digest.hexdigest(),
        'weighted_segmented_word_types': db.execute('SELECT COUNT(*) FROM words').fetchone()[0],
        'policy': 'One frequency vote per TRAIN clip occurrence; all broad and cleaner texts, exact source text, no added diacritics; duplicate texts and source overlaps remain weighted occurrences, not unique hours.'}
    write('source_label_audit.json', result); return result

def fit(db, guard, candidate):
    guard.phase = 'disk_backed_append_only_512_merges'
    db.execute('CREATE TABLE working(id INTEGER PRIMARY KEY,seq TEXT NOT NULL,freq INTEGER NOT NULL)')
    db.execute('INSERT INTO working SELECT id,seq,freq FROM words'); db.commit()
    db.execute('CREATE TABLE pairs(pair TEXT PRIMARY KEY,freq INTEGER NOT NULL) WITHOUT ROWID')
    db.execute('CREATE TABLE members(pair TEXT NOT NULL,word_id INTEGER NOT NULL,n INTEGER NOT NULL,PRIMARY KEY(pair,word_id)) WITHOUT ROWID')
    vocab = candidate['model']['vocab']; count = 0; totals = collections.Counter(); inserts = []
    for wid, seq_raw, frequency in db.execute('SELECT id,seq,freq FROM working ORDER BY id'):
        seq = json.loads(seq_raw); guard.pace(len(seq_raw.encode())+24)
        for key, n in pair_counts(seq, vocab).items():
            totals[key] += n * frequency; inserts.append((key, wid, n))
        count += 1
        if count % 1024 == 0:
            db.executemany('INSERT INTO members VALUES (?,?,?)', inserts)
            db.executemany('INSERT INTO pairs VALUES (?,?) ON CONFLICT(pair) DO UPDATE SET freq=freq+excluded.freq', totals.items())
            db.commit(); inserts.clear(); totals.clear()
    if inserts:
        db.executemany('INSERT INTO members VALUES (?,?,?)', inserts)
        db.executemany('INSERT INTO pairs VALUES (?,?) ON CONFLICT(pair) DO UPDATE SET freq=freq+excluded.freq', totals.items()); db.commit()
    db.execute('CREATE INDEX pair_priority ON pairs(freq DESC,pair ASC)'); db.commit()
    additions = []
    for step in range(COUNT):
        guard.check()
        while True:
            row = db.execute('SELECT pair,freq FROM pairs WHERE freq>0 ORDER BY freq DESC,pair ASC LIMIT 1').fetchone()
            if row is None: raise RuntimeError('Insufficient observed eligible TRAIN merges')
            pair_raw, frequency = row; pair = tuple(json.loads(pair_raw)); piece = ''.join(pair)
            if permitted(pair, vocab): break
            db.execute('UPDATE pairs SET freq=0 WHERE pair=?', (pair_raw,)); db.commit()
        # Snapshot impacted IDs in a disk table, never fetch an unbounded word population.
        db.execute('CREATE TEMP TABLE affected(id INTEGER PRIMARY KEY)')
        db.execute('INSERT INTO affected SELECT word_id FROM members WHERE pair=?', (pair_raw,)); db.commit()
        affected = 0
        for (wid,) in db.execute('SELECT id FROM affected ORDER BY id'):
            seq_raw, weight = db.execute('SELECT seq,freq FROM working WHERE id=?', (wid,)).fetchone(); seq = json.loads(seq_raw)
            old_pairs = pair_counts(seq, vocab); merged = []; at = 0
            while at < len(seq):
                if at+1 < len(seq) and (seq[at],seq[at+1]) == pair: merged.append(piece); at += 2
                else: merged.append(seq[at]); at += 1
            assert merged != seq
            new_pairs = pair_counts(merged, vocab)
            db.execute('UPDATE working SET seq=? WHERE id=?', (canonical(merged), wid))
            for key, n in old_pairs.items():
                db.execute('UPDATE pairs SET freq=freq-? WHERE pair=?', (n*weight, key))
                db.execute('DELETE FROM members WHERE pair=? AND word_id=?', (key,wid))
            for key, n in new_pairs.items():
                db.execute('INSERT INTO pairs VALUES (?,?) ON CONFLICT(pair) DO UPDATE SET freq=freq+excluded.freq', (key,n*weight))
                db.execute('INSERT INTO members VALUES (?,?,?)', (key,wid,n))
            affected += 1; guard.pace(len(seq_raw.encode())+len(canonical(merged).encode())+len(old_pairs)*64+len(new_pairs)*64)
            if affected % 1024 == 0: db.commit()
        db.execute('DROP TABLE affected'); db.commit()
        identifier = len(vocab); left, right = pair
        additions.append({'piece':piece,'id':identifier,'left':left,'right':right,'left_id':vocab[left],'right_id':vocab[right],
            'merge_rank':len(candidate['model']['merges']),'weighted_pair_frequency':frequency,'affected_word_types':affected})
        vocab[piece] = identifier; candidate['model']['merges'].append(list(pair))
        # Existing pieces created by the merge must never be selected again via another pair.
        if (step+1) % 16 == 0:
            write('learned_merges_partial.json', additions); guard.publish(learned_merges=step+1,last_piece=piece)
    return additions

def validate(guard, incumbent, coverage, candidate):
    guard.phase='all_TRAIN_candidate_validation'
    groups={}; digest=hashlib.sha256(); changed_examples=[]
    for pool, ordinal, sid, dataset, split, text, fields, blob, text_sha in corpus(guard):
        guard.pace(len(text.encode())+len(blob)+64)
        old=incumbent.encode(text); cov=coverage.encode(text); new=candidate.encode(text)
        a=coverage.decode(cov.ids,skip_special_tokens=False); b=candidate.decode(new.ids,skip_special_tokens=False)
        assert a == b, ('Changed decoded text', pool,ordinal,a,b)
        assert all(0<=i<candidate.get_vocab_size() and i!=8192 for i in new.ids)
        assert new.ids.count(0)==cov.ids.count(0), ('Unknown regression',pool,ordinal)
        key=pool+':'+dataset
        m=groups.setdefault(key,{'rows':0,'incumbent_tokens':0,'coverage_tokens':0,'candidate_tokens':0,'changed_rows':0,
            'unknown_rows':0,'circumflex_source_rows':0,'circumflex_decode_matches_coverage_rows':0,'max_candidate_U':0,'new_piece_occurrences':0})
        m['rows']+=1;m['incumbent_tokens']+=len(old.ids);m['coverage_tokens']+=len(cov.ids);m['candidate_tokens']+=len(new.ids)
        m['changed_rows']+=old.ids!=new.ids;m['unknown_rows']+=0 in new.ids;m['max_candidate_U']=max(m['max_candidate_U'],len(new.ids))
        m['new_piece_occurrences']+=sum(i>=8711 for i in new.ids)
        if any(c in text for c in CHARS):m['circumflex_source_rows']+=1;m['circumflex_decode_matches_coverage_rows']+=1
        digest.update((canonical([pool,ordinal,sid,new.ids])+'\n').encode())
        if len(changed_examples)<24 and old.ids!=new.ids and len([x for x in changed_examples if x['group']==key])<4:
            changed_examples.append({'group':key,'source_id':sid,'text':text,'incumbent_tokens':old.tokens,'candidate_tokens':new.tokens})
        if sum(x['rows'] for x in groups.values())%10000==0:guard.publish(validated_rows=sum(x['rows'] for x in groups.values()))
    for value in groups.values():value['token_reduction_fraction']=1-value['candidate_tokens']/value['incumbent_tokens']
    return {'groups':groups,'all_rows':sum(x['rows'] for x in groups.values()),'ordered_candidate_labels_sha256':digest.hexdigest(),
        'decoded_text_matches_coverage_baseline_all_rows':True,'all_full_targets_retained':True,'changed_examples':changed_examples}

def main():
    AREA.mkdir(parents=True,exist_ok=True); args=argparse.ArgumentParser(); args.add_argument('--fixture-only',action='store_true'); a=args.parse_args()
    guard=Guard()
    from tokenizers import Tokenizer
    import tokenizers
    assert not any(x in sys.modules for x in ('torch','transformers','nemo'))
    assert sha(INCUMBENT)==INCUMBENT_SHA
    base=json.loads(INCUMBENT.read_text(encoding='utf8')); assert len(base['model']['vocab'])==8711 and len(base['model']['merges'])==13988
    incumbent=Tokenizer.from_str(json.dumps(base,ensure_ascii=False))
    candidate=json.loads(json.dumps(base)); assert 'Û' not in candidate['model']['vocab']
    candidate['model']['vocab']['Û']=8711
    coverage=Tokenizer.from_str(json.dumps(candidate,ensure_ascii=False))
    coverage_manifest=publish_coverage(base,coverage)
    write('coverage_only_current_handoff.json',{'coverage_manifest':record(AREA/'coverage_only_8712/manifest.json'),
        'tokenizer':coverage_manifest['artifacts']['tokenizer.json'],'production_activated':False,'all_TRAIN_extension_pending':True})
    write('incumbent_coverage_and_unicode_fixtures.json',{'incumbent':record(INCUMBENT),'tokenizers_version':tokenizers.__version__,
        'incumbent_fixtures':fixture(incumbent),'coverage_fixtures':fixture(coverage),'correction':'Append literalÛ at8711; no original tokenID/merge/normalizer/special change.'})
    if a.fixture_only:
        guard.publish(completed_fixture_only=True);return
    assert not (AREA/'training_counts.sqlite').exists(), 'Fresh isolated candidate namespace required'
    sources={k:stat(p) for k,p in [('broad_text',BROAD),('broad_labels',BROAD_LABELS),('clean_text',CLEAN),('clean_labels',CLEAN_LABELS)]}
    write('candidate_recipe_pinned_before_fit.json',{'kind':'Turkish_all_TRAIN_append_only_BPE9224_recipe_v1','at_utc':now(),
        'incumbent':record(INCUMBENT),'sources':sources,'source':record(__file__),'added_character':'Û','appended_merges':COUNT,
        'candidate_vocabulary':9224,'all_existing_8711_tokenIDs_and_13988_merge_priority_prefix_fixed':True,
        'TRAIN_only':True,'DEV_TEST_text_or_predictions_used':False,'GPU_or_models_used':False,'maximum_seconds':3600,
        'worker_soft_hard_guard_private_bytes':1800000000,'Windows_worker_job_commit_limit_bytes':1800000000,
        'measured_PowerShell_parent_maximum_private_bytes':200000000,'whole_worker_and_parent_private_limit_bytes':2000000000,
        'CPU_thread_caps':1,'bounded_batch_texts':512,'maximum_paced_logical_bytes':16000000000,
        'maximum_metadata_IO_bytes_s':33554432,'weighting':'Every broad+clean TRAIN clip occurrence once; identical texts/source overlaps remain counted, no unique-hour claim.',
        'model_integration_admitted':False,'active_assets_changed':False})
    db=sqlite3.connect(AREA/'training_counts.sqlite');db.execute('PRAGMA cache_size=-8192');db.execute('PRAGMA mmap_size=0')
    db.execute('PRAGMA temp_store=FILE');db.execute('PRAGMA journal_mode=DELETE');db.execute('PRAGMA synchronous=FULL')
    audit=source_audit(db,guard,incumbent,coverage)
    additions=fit(db,guard,candidate);db.close()
    assert len(candidate['model']['vocab'])==9224 and len(candidate['model']['merges'])==14500
    for k,v in base['model']['vocab'].items():assert candidate['model']['vocab'][k]==v
    assert candidate['model']['merges'][:13988]==base['model']['merges']
    for k in base:
        if k!='model':assert candidate[k]==base[k]
    write('tokenizer.json',candidate);write('added_pieces.json',[{'piece':'Û','id':8711,'kind':'missing_character_coverage','donor_init':'requires explicit new-row initialization; not an existing-token merge'},*additions])
    write('vocab.json',candidate['model']['vocab'])
    (AREA/'vocab.tsv').write_text('\n'.join(str(i)+'\t'+p for p,i in sorted(candidate['model']['vocab'].items(),key=lambda x:x[1]))+'\n',encoding='utf8')
    (AREA/'merges.txt').write_text('#version: 0.2\n'+'\n'.join(' '.join(p) for p in candidate['model']['merges'])+'\n',encoding='utf8')
    # Freeze before re-encoding the complete TRAIN corpus. No model/DEV/TEST data are consulted.
    write('candidate_frozen_before_measurement.json',{'at_utc':now(),'tokenizer':record(AREA/'tokenizer.json'),
        'added_pieces':record(AREA/'added_pieces.json'),'source_ordered_text_sha256':audit['source_ordered_text_sha256'],
        'TRAIN_sources_only':True,'no_ASR_quality_gain_or_production_admission_claim':True})
    updated=Tokenizer.from_file(str(AREA/'tokenizer.json'))
    result=validate(guard,incumbent,coverage,updated);write('all_TRAIN_candidate_validation.json',result)
    assert result['all_rows']==2983883
    for k,p in [('broad_text',BROAD),('broad_labels',BROAD_LABELS),('clean_text',CLEAN),('clean_labels',CLEAN_LABELS)]:assert stat(p)==sources[k],('Source changed',k)
    assert sha(INCUMBENT)==INCUMBENT_SHA
    write('row_migration_plan.json',{'candidate_only':True,'active_model_modified':False,'old_vocabulary':8711,'candidate_vocabulary':9224,
        'old_blank_id':8192,'new_blank_id':8192,'all_existing_8711_rows_keep_IDs':True,'new_rows_half_open':[8711,9224],
        'duration_rows_old_half_open':[8711,8716],'duration_rows_new_half_open':[9224,9229],
        'existing_three_Adam_partitions_mature_clocks_and_207_state_must_not_be_reset':True,
        'needs_explicit_new_row_initialization_optimizer_extension_new_labels_population_and_geometry':True,
        'not_exact_resume_of_current_8711_decoder':True,'old_multilingual_token_ID_preservation_does_not_prove_same_encodings_or_retention':True,
        'additional_trainable_parameters_hidden640':513*(640+640+1),
        'no_model_migration_or_training_executed':True})
    summary={'kind':'Turkish_all_TRAIN_circumflex_label_and_BPE9224_actual_v1','at_utc':now(),'passed':True,
        **guard.main_identity,'actual_worker_exit_intended':0,'GPU_models_DEV_TEST_used':False,'incumbent_8711_unchanged':True,
        'candidate_activated':False,'sources_stat_unchanged':True,'source_rows':audit['counts'],'tokenizer_vocabulary':9224,
        'incumbent_missing_character':'Û','existing_lowercase_circumflex_preserved':True,'all_existing_label_hash_and_IDs_checked':True,
        'candidate_validation':result,'elapsed_seconds':time.monotonic()-guard.started,'peak_private_bytes':guard.peak,
        'paced_logical_bytes':guard.bytes,'artifacts':{n:record(AREA/n) for n in ['tokenizer.json','vocab.json','vocab.tsv','merges.txt','added_pieces.json','source_label_audit.json','row_migration_plan.json','candidate_frozen_before_measurement.json']}}
    write('completed_handoff.json',summary);guard.phase='completed';guard.publish(passed=True)
    print(json.dumps({'passed':True,'handoff':record(AREA/'completed_handoff.json'),'elapsed_seconds':summary['elapsed_seconds']},ensure_ascii=False),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as error:
        write('actual_worker_failure_PID'+str(os.getpid())+'.json',{'at_utc':now(),'PID':os.getpid(),'error':repr(error),'sources_modified':False});raise
