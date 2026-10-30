"""Bounded final scalar-index audit and before-draw decoded-length gate.

This source does not publish a final population or infer decoded validation
from STREAMINFO. The final envelope/actual producer/source proof chain must be
verified by the caller before opening its pinned index. No sampler is owned.
"""
import hashlib
from itertools import zip_longest
from pathlib import Path
import sqlite3
from bpe_broad_production_state_v1 import require
from bpe_runtime_declared_geometry_v1 import FIELDS,validate_declared_row,verify_actual_row

ROOT=Path(r'E:\gpu_training_scratchpad')
COLS=('ordinal','source_component','native_sr','native_samples','clean16k_samples','mel_frames','encoder_frames','target_tokens','status')
GROUP_KEYS=('retained_rows','total_target_tokens','MAX_T','MAX_U','MAX_T_times_U_plus_1','MAX_native_samples','MAX_clean16k_samples','unresolved_retained_rows')

def signature(path):
    p=Path(path).resolve();s=p.stat()
    return {'path':str(p),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}

def open_final(path):
    p=Path(path).resolve();require(p.is_relative_to(ROOT.resolve()),'Final geometry/catalog/labels remain E-only')
    for suffix in ('-wal','-journal'):
        q=Path(str(p)+suffix);require(not q.exists() or q.stat().st_size==0,'Immutable final DB has an outstanding write journal')
    c=sqlite3.connect(p.as_uri()+'?mode=ro&immutable=1',uri=True)
    c.row_factory=sqlite3.Row;c.execute('PRAGMA query_only=ON');c.execute('PRAGMA temp_store=MEMORY');c.execute('PRAGMA cache_size=-4096')
    return c

def _group():return {k:0 for k in GROUP_KEYS}

def _observe(group,d):
    t=d['encoder_frames'];u=d['target_tokens']
    group['retained_rows']+=1;group['total_target_tokens']+=u
    for key,value in (('MAX_T',t),('MAX_U',u),('MAX_T_times_U_plus_1',t*(u+1)),('MAX_native_samples',d['native_samples']),('MAX_clean16k_samples',d['clean16k_samples'])):
        group[key]=max(group[key],value)

def audit_index(catalog_path,labels_path,geometry_path,*,rows,tokens,guard=lambda:None):
    """Stream exact retained ordinal/target/rounding joins, never load audio."""
    require(type(rows) is int and rows>0 and type(tokens) is int and tokens>=rows,'Exact final denominator required')
    paths=[Path(v).resolve() for v in (catalog_path,labels_path,geometry_path)];before=[signature(v) for v in paths]
    connections=[];groups={v:_group() for v in ('all','alania_yodas3','kiraat')};count=0;above458=0;above1024=0
    try:
        for p in paths:
            guard();c=open_final(p);connections.append(c);c.set_progress_handler(lambda:(guard() or 0),10000)
        catalog,labels,geometry=connections
        schema=list(geometry.execute('PRAGMA table_info(geometry)'))
        require(tuple(v['name'] for v in schema)==COLS and schema[0]['type'].upper()=='INTEGER' and schema[0]['pk']==1,
                'Exact scalar geometry schema and INTEGER PRIMARY KEY required')
        for table,c in (('clips',catalog),('labels',labels)):
            columns={v['name']:v for v in c.execute('PRAGMA table_info('+table+')')}
            require(columns['ordinal']['type'].upper()=='INTEGER' and columns['ordinal']['pk']==1,'Final source ordinals require unique INTEGER PRIMARY KEY')
        streams=(catalog.execute('SELECT ordinal,dataset,unique_id,source_id,start,end FROM clips ORDER BY ordinal'),
            labels.execute('SELECT ordinal,target_tokens,typeof(target_tokens) AS token_type,typeof(target_ids_u32) AS target_type,length(target_ids_u32) AS target_bytes FROM labels ORDER BY ordinal'),
            geometry.execute('SELECT '+','.join(COLS)+' FROM geometry ORDER BY ordinal'))
        for raw,label,declared in zip_longest(*streams):
            guard();require(raw is not None and label is not None and declared is not None,'Final catalog/labels/geometry cardinalities differ')
            count+=1;require(raw['ordinal']==label['ordinal']==declared['ordinal']==count<=rows,'Retained final joins must be exactly contiguous1..N')
            u=label['target_tokens'];require(label['token_type']=='integer' and type(u) is int and u>0 and label['target_type']=='blob' and label['target_bytes']==4*u,'Every full original target BLOB/length required')
            d=validate_declared_row(dict(declared),dict(raw),full_target_tokens=u)
            _observe(groups['all'],d);_observe(groups[raw['dataset']],d);above458+=int(u>458);above1024+=int(u+1>1024)
        require(count==rows and groups['all']['total_target_tokens']==tokens,'Final retained denominator/token sum differs')
    finally:
        for c in connections:c.close()
    guard();require(before==[signature(v) for v in paths],'Final source/index signature changed during audit')
    all_group=groups['all']
    population={'retained_rows':count,'total_target_tokens':tokens,**{k:all_group[k] for k in ('MAX_T','MAX_U','MAX_T_times_U_plus_1','MAX_native_samples','MAX_clean16k_samples')},
        **{k:1 for k in ('catalog_min_ordinal','labels_min_ordinal','geometry_min_ordinal')},
        **{k:count for k in ('catalog_max_ordinal','labels_max_ordinal','geometry_max_ordinal')},'count_U_gt_458':above458,'count_U_plus_1_gt_1024':above1024}
    return {'kind':'ASR18_retained_declared_geometry_scalar_audit_v1','passed':True,'population':population,'groups':groups,
        'source_signatures':before,'actual_audio_decoded':False,'actual_full_corpus_decoded_validation_claimed':False,'actual_optimizer_updates':0}

class DeclaredGeometryIndex:
    def __init__(self,binding,*,guard=lambda:None):
        require(type(binding) is dict and set(binding)=={'path','bytes','sha256','mtime_ns'} and all(type(binding[k]) is int and binding[k]>0 for k in ('bytes','mtime_ns')),
                'Exact four-field source-bound final geometry index required')
        self.guard=guard;self.path=Path(binding['path']).resolve();require(self.path.is_relative_to(ROOT.resolve()) and binding['bytes']<=2_000_000_000,'E-only scalar index bound required')
        self.signature=signature(self.path);require(self.signature=={k:binding[k] for k in ('path','bytes','mtime_ns')},'Final index signature differs')
        digest=hashlib.sha256()
        with self.path.open('rb') as f:
            while chunk:=f.read(1024*1024):guard();digest.update(chunk)
        require(digest.hexdigest()==binding['sha256'] and signature(self.path)==self.signature,'Final index hash/signature differs')
        self.connection=open_final(self.path);self.poisoned=False

    def declaration(self,row,*,full_target_tokens):
        require(not self.poisoned,'Failed geometry gate cannot be reused')
        try:
            self.guard();require(signature(self.path)==self.signature,'Final geometry index changed before draw')
            require(type(row['ordinal']) is int,'Original catalog ordinal required')
            value=self.connection.execute('SELECT '+','.join(COLS)+' FROM geometry WHERE ordinal=?',(row['ordinal'],)).fetchone()
            require(value is not None,'Missing retained declared row remains unresolved')
            return validate_declared_row(dict(value),row,full_target_tokens=full_target_tokens)
        except BaseException:self.poisoned=True;raise

    def close(self):self.connection.close()

class BeforeDrawDeclaredAudioReader:
    """Call only with original catalog rows, before training-ID rewriting.

    The original reader must retain its normal exact heldout-PCM protection.
    Any decode/heldout/geometry failure poisons this provisional reader. This
    wrapper returns no waveform until the actual full length agrees.
    """
    def __init__(self,index,original_reader,*,full_target_tokens_for,guard=lambda:None):
        self.index=index;self.original_reader=original_reader;self.target_count=full_target_tokens_for;self.guard=guard;self.poisoned=False;self.last=None

    def __call__(self,row,**kwargs):
        require(not self.poisoned,'Failed provisional audio reader cannot continue')
        try:
            self.guard();u=self.target_count(row);d=self.index.declaration(row,full_target_tokens=u)
            audio=self.original_reader(row,**kwargs);actual=self.original_reader.last
            proof=verify_actual_row(d,row,actual,full_target_tokens=u)
            require(getattr(audio,'ndim',None)==1 and len(audio)==d['clean16k_samples'] and str(audio.dtype)=='float32','Returned full waveform differs from actual decoded observation')
            self.guard();self.last={**actual,'declared_geometry_runtime_proof':proof};return audio
        except BaseException:self.poisoned=True;self.index.poisoned=True;raise
