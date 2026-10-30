"""Read compact immutable TRAIN columns and1536 sampled text rows; no ML."""
import collections
import datetime
import hashlib
import json
import math
import mmap
from pathlib import Path
import random
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/parakeet_turkish'
sys.path.insert(0, str(OUT))
from metrics import normalize_text

COL = struct.Struct('<IIdB')
OFF = struct.Struct('<Q')
GROUPS = ('fleurs','cv','tsc')
CONFIG = OUT / 'runs/adaptive_20261013_TR_uniform_headlr3e5_cosine_01/run_config.json'
def record(path):
    path = Path(path).resolve()
    return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def quantile(hist, q):
    rank = max(1, math.ceil(sum(hist.values()) * q))
    total = 0
    for value, count in sorted(hist.items()):
        total += count
        if total >= rank: return value
def summarize(hist):
    n=sum(hist.values())
    return {'n':n,'mean':sum(x*c for x,c in hist.items())/n,'min':min(hist),'p10':quantile(hist,.1),'p25':quantile(hist,.25),'median':quantile(hist,.5),'p75':quantile(hist,.75),'p90':quantile(hist,.9),'p95':quantile(hist,.95),'p99':quantile(hist,.99),'max':max(hist)}
config = read(CONFIG)
directory = Path(config['indexed_tr_directory'])
receipt_path = directory / 'index_provenance.json'
receipt = read(receipt_path)
assert receipt['rows'] == 253798 and receipt['complete'] and receipt['provenance']['training_admission_complete']
assert receipt['provenance']['lazy_helper']['sha256'] == 'faf2b549d2e5e3d2f2fd78d282b612c6943e813cbedbc3f6b16caedab8249fd2'
for name in ('columns.bin','offsets.u64'):
    actual=record(directory/name)
    assert actual['sha256']==receipt['files'][name]['sha256'] and actual['bytes']==receipt['files'][name]['bytes']
assert (directory/'rows.jsonl').stat().st_size==receipt['files']['rows.jsonl']['bytes']
stats={g:{'count':0,'U':collections.Counter(),'T':collections.Counter(),'duration_quarterseconds':collections.Counter(),'seconds':0.,'tokens':0,'inverse_U_sum':0.,'lattice_cells':0} for g in GROUPS}
sample={g:[] for g in GROUPS}
rng=random.Random(37001)
with (directory/'columns.bin').open('rb') as stream:
    while payload:=stream.read(COL.size):
        t,u,d,gidx=COL.unpack(payload)
        assert t>0 and 1<=u<=180 and math.isfinite(d) and d>0
        g=GROUPS[gidx];s=stats[g]
        s['count']+=1;s['U'][u]+=1;s['T'][t]+=1;s['duration_quarterseconds'][math.ceil(d*4)]+=1
        s['seconds']+=d;s['tokens']+=u;s['inverse_U_sum']+=1/u;s['lattice_cells']+=t*(u+1)
        index=sum(x['count'] for x in stats.values())-1
        if len(sample[g])<512:sample[g].append(index)
        else:
            chosen=rng.randrange(s['count'])
            if chosen<512:sample[g][chosen]=index
N=sum(s['count'] for s in stats.values())
assert N==253798
mu=sum(s['tokens'] for s in stats.values())/N
word_samples={g:[] for g in GROUPS}
sample_digest=hashlib.sha256()
with (directory/'offsets.u64').open('rb') as offsets_file, (directory/'rows.jsonl').open('rb') as carrier, (directory/'columns.bin').open('rb') as columns_file:
    with mmap.mmap(offsets_file.fileno(),0,access=mmap.ACCESS_READ) as offsets, mmap.mmap(columns_file.fileno(),0,access=mmap.ACCESS_READ) as columns:
        for g in GROUPS:
            for index in sorted(sample[g]):
                start=OFF.unpack_from(offsets,index*8)[0];end=OFF.unpack_from(offsets,(index+1)*8)[0]
                assert 0<end-start<4*1024*1024
                carrier.seek(start);payload=carrier.read(end-start);row=json.loads(payload)
                t,u,d,gidx=COL.unpack_from(columns,index*COL.size)
                assert GROUPS[gidx]==g and row['encoder_frames']==t and row['target_tokens']==u
                text=normalize_text(row['text'],'tr')
                words=len(text.split())
                assert words>0
                word_samples[g].append((u,words,d))
                sample_digest.update(json.dumps([index,row['source_id'],u,words],ensure_ascii=False,separators=(',',':')).encode()+b'\n')
def correlation(values):
    n=len(values);sx=sum(x for x,y,d in values);sy=sum(y for x,y,d in values)
    numerator=n*sum(x*y for x,y,d in values)-sx*sy
    denominator=math.sqrt((n*sum(x*x for x,y,d in values)-sx*sx)*(n*sum(y*y for x,y,d in values)-sy*sy))
    return numerator/denominator if denominator else None
group_report={}
total_tokens=N*mu
estimated_words=0.
for g,s in stats.items():
    ws=word_samples[g];mean_words=sum(w for u,w,d in ws)/len(ws);estimated_words+=mean_words*s['count']
    group_report[g]={
        'clips':s['count'],'clip_fraction':s['count']/N,'audio_hours':s['seconds']/3600,
        'target_tokens_total':s['tokens'],'target_token_mass_fraction':s['tokens']/total_tokens,
        'target_tokens':summarize(s['U']),'encoder_frames':summarize(s['T']),
        'duration_seconds_mean':s['seconds']/s['count'],
        'duration_seconds_quartersecond_ceil_quantiles':{k:v/4 for k,v in summarize(s['duration_quarterseconds']).items() if k not in ('n','mean')},
        'mean_inverse_U':s['inverse_U_sum']/s['count'],'native_T_times_Uplus1_cells':s['lattice_cells'],
        'word_sample_n':len(ws),'sample_normalized_words':summarize(collections.Counter(w for u,w,d in ws)),
        'sample_target_tokens_per_word_corpus':sum(u for u,w,d in ws)/sum(w for u,w,d in ws),
        'sample_U_word_Pearson_correlation':correlation(ws),
        'estimated_full_pool_reference_words':mean_words*s['count'],
        'proposed_fixed_meanU_TR_weight_group_mean':s['tokens']/s['count']/mu,
    }
for g in GROUPS:
    group_report[g]['estimated_word_mass_fraction']=group_report[g]['estimated_full_pool_reference_words']/estimated_words
all_u=collections.Counter()
for s in stats.values():all_u.update(s['U'])
current={'per_accumulated16_objective':'(sum_i raw_TDT_NLL_i/U_i + sum_RP weighted_preservation_i)/16',
    'physical_microbatches':[[8,0],[6,2]],'accumulation':2,'native_reduction':'none; one raw sequence TDT NLL per clip',
    'per_clip_total_weight':1/16,'sequence_NLL_gradient_multiplier':'1/(16*U_i)',
    'replay_policy':'one PT and one uniform-old-five per update',
    'PT_preservation':[40.,2.,20.],'old_five_preservation':[40*5/3,2*5/3,20*5/3],
    'native_alignment_and_duration_loss_unchanged':True}
report={'kind':'original8199_TR_length_normalization_CPU_evidence_v1','created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'scope':'Complete4.3MB compact T/U/duration/group columns plus512 sampled text rows per group; no full row array, audio, model, tensor, GPU, checkpoint, DEV/TEST predictions or source edits.',
    'current_trial_unchanged':True,'config':record(CONFIG),'index_receipt':record(receipt_path),
    'compact_columns_verified':record(directory/'columns.bin'),'compact_offsets_verified':record(directory/'offsets.u64'),
    'row_carrier_stat_only':{'path':str(directory/'rows.jsonl'),'bytes':(directory/'rows.jsonl').stat().st_size,'expected_original_sha256':receipt['files']['rows.jsonl']['sha256'],'full_rescan_performed':False},
    'sample_selection':'Independent CPU Random37001 reservoir512 per group; oversampled smallFLEURS group, whole-pool word estimates reweighted by exact group clipcounts.',
    'sample_digest':sample_digest.hexdigest(),'sample_rows':sum(len(s) for s in word_samples.values()),
    'TR_clips':N,'whole_pool_target_tokens':summarize(all_u),'whole_pool_mean_U':mu,
    'groups':group_report,'current_objective':current,
    'conditional_control':{
        'status':'Evidence supports one controlled hypothesis, not a quality fix or chosen head policy.',
        'proposed_TR_only_TDT_factor':'U_i / frozen_population_mean_U',
        'equivalent_TR_term':'raw_TDT_NLL_i / frozen_population_mean_U',
        'frozen_population_mean_U':mu,'whole_uniform_pool_mean_factor_exactly_one':True,
        'replay_TDT_and_all_KD_terms_unchanged':True,'no_microbatch_sumU_renormalization':True,
        'draws_sourceIDs_and_full_coverage_unchanged':True,
        'effective_source_group_weight_changes_explicit':True,
        'purpose':'Test whether equal clip weighting over-emphasizes shorttargets relative to token/word exposure while Turkish DEV scoring aggregates reference words.',
        'gradient_evidence_available':False,'raw_per_clip_NLL_length_covariance_available':False,
        'not_word_WER_equivalence':'Subword U is not wordcount; TDT NLL includes blank/time/duration alignment contributions. Word exposure is estimated from1536 TRAIN texts.',
        'needed_before_GPU_admission':'Finish active head cycle, choose head policy from actual complete comparison, then one matched normalization control from one coherent frozen state, with identical future TR/RP examples and all other recipe/source350/gates/guards fixed.',
        'adopted_or_GPU_tested':False,
    },
    'source_records':{name:record(OUT/name) for name in ('batched_lora_tr_metadata_v1.py','train_lora_batched_tsc_uniform_head_lr_v1.py','metrics.py')},
    'limitations':['TRAIN token counts are complete; normalized word counts are bounded stratified-sample estimates.','No measured gradient or loss-length covariance proves this weighting will improve WER.','Do not choose active head policy or start a new GPU job before the current full head cycle closes.'],
}
path=OUT/'original8199_TR_loss_length_exposure_CPU_review_v1.json'
assert not path.exists()
with path.open('x',encoding='utf-8',newline='\n') as stream:json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
assert 'torch' not in sys.modules
print(json.dumps({'passed':True,'report':record(path),'TR_clips':N,'mean_U':mu,'sample_rows':report['sample_rows'],'groups':{g:{k:group_report[g][k] for k in ('clip_fraction','target_token_mass_fraction','duration_seconds_mean','sample_target_tokens_per_word_corpus','sample_U_word_Pearson_correlation','proposed_fixed_meanU_TR_weight_group_mean')} for g in GROUPS},'Torch_imported':False}))
