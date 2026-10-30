"""Inert stdlib sampler:30% FLEURS /70% CV Turkish clips, unchanged50% replay.

Future integration must supply the frozen trainer's BucketCycler explicitly;
this module never imports Torch, reads audio, or changes a running trainer.
Fractions are clip quotas, not audio quotas. A global TR draw/pool-size ratio
does not mean full-pool coverage under domain oversampling.
"""
from __future__ import annotations
from fractions import Fraction
import hashlib
import json
import math
import random

LANGUAGES=('en','es','it','fr','de')
FLEURS_REVISION='70bb2e84b976b7e960aa89f1c648e09c59f894dd'
CV_REVISION='9d8a025f1ae7c5a2f5c43298c7c4910b03995a4b'


def turkish_domain(row):
    if row.get('language')!='tr':raise ValueError('Only accepted Turkish TRAIN rows belong to domain streams')
    if (row.get('dataset')=='google/fleurs' and row.get('dataset_config')=='tr_tr'
        and row.get('dataset_split')=='train' and row.get('dataset_revision')==FLEURS_REVISION):return 'fleurs'
    if (row.get('dataset')=='ysdede/commonvoice_17_tr_fixed' and row.get('dataset_revision')==CV_REVISION
        and (row.get('dataset_split')=='train' or
            row.get('dataset_split')=='validated' and row.get('training_role')=='train')):return 'cv'
    raise ValueError('Only vetted pinned FLEURS TRAIN /CV TRAIN-role recordings are eligible')


def fingerprint(rows):
    digest=hashlib.sha256()
    for row in sorted(rows,key=lambda r:r['source_id']):
        value={key:row.get(key) for key in ('source_id','dataset','dataset_revision','dataset_split','training_role',
            'language','audio_sha256','text','encoder_frames','target_tokens','label_ids_sha256','duration')}
        digest.update(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()+b'\n')
    return digest.hexdigest()


class DomainBalancedBucketSampler:
    def __init__(self,tr,replay,*,seed,batch_size,accumulation,replay_fraction=.5,fleurs_fraction=.3,
                 bucket_size=128,cycler_factory,cycler_source_sha256):
        self.rng=random.Random(seed+1)
        self.batch_size,self.accumulation=batch_size,accumulation
        self.effective_batch=batch_size*accumulation
        self.replay_per_update=round(self.effective_batch*replay_fraction)
        if (batch_size<1 or accumulation<1 or not math.isclose(self.replay_per_update,
            self.effective_batch*replay_fraction,abs_tol=1e-9) or not 0<=self.replay_per_update<self.effective_batch):
            raise ValueError('Integer replay quota and positive Turkish effective batch required')
        ratio=Fraction(str(fleurs_fraction))
        if not 0<ratio<1:raise ValueError('Both Turkish domains need positive fractions')
        if len(cycler_source_sha256)!=64:raise ValueError('Bind the exact supplied BucketCycler source SHA')
        self.fleurs_ratio=ratio;self.tr_per_update=self.effective_batch-self.replay_per_update
        by_domain={name:[] for name in ('fleurs','cv')}
        if len({r['source_id'] for r in tr})!=len(tr):raise ValueError('Turkish source IDs must be globally unique')
        for row in tr:by_domain[turkish_domain(row)].append(row)
        if any(not pool for pool in by_domain.values()):raise ValueError('Both domain pools must be nonempty')
        self.domains={name:cycler_factory(rows,self.rng,bucket_size) for name,rows in by_domain.items()}
        if any(row.get('language') not in LANGUAGES for row in replay):raise ValueError('Only old-language replay is eligible')
        self.replay={lang:cycler_factory([r for r in replay if r['language']==lang],self.rng,bucket_size) for lang in LANGUAGES}
        self.replay_language_counts={lang:0 for lang in LANGUAGES};self.updates=0
        self.domain_audio_seconds={name:0. for name in self.domains}
        self.identity={'batch_size':batch_size,'accumulation':accumulation,'replay_per_update':self.replay_per_update,
            'fleurs_fraction_numerator':ratio.numerator,'fleurs_fraction_denominator':ratio.denominator,
            'bucket_size':bucket_size,'cycler_source_sha256':cycler_source_sha256,
            'turkish_pool_fingerprint':fingerprint(tr),'replay_pool_fingerprint':fingerprint(replay)}

    def fleurs_quota_at(self,tr_draws):
        return tr_draws*self.fleurs_ratio.numerator//self.fleurs_ratio.denominator

    def next_batches(self):
        before=self.updates*self.tr_per_update;after=before+self.tr_per_update
        nf=self.fleurs_quota_at(after)-self.fleurs_quota_at(before)
        tr=[]
        for domain,count in (('fleurs',nf),('cv',self.tr_per_update-nf)):
            for _ in range(count):
                row=self.domains[domain].next();tr.append(row)
                self.domain_audio_seconds[domain]+=float(row['duration'])
        # Existing reservoir T/U length buckets remain inside each domain.
        # Sort only the selected Turkish rows, never discard/pad a source row.
        tr.sort(key=lambda r:(int(math.log2(int(r['encoder_frames']))),int(math.log2(int(r['target_tokens']))),
                              int(r['encoder_frames']),int(r['target_tokens']),r['source_id']))
        replay=[]
        for _ in range(self.replay_per_update):
            lang=self.rng.choice(list(self.replay));replay.append(self.replay[lang].next())
            self.replay_language_counts[lang]+=1
        batches,remainder=[],[]
        for stream in (tr,replay):
            full=len(stream)//self.batch_size*self.batch_size
            batches.extend(stream[start:start+self.batch_size] for start in range(0,full,self.batch_size))
            remainder.extend(stream[full:])
        if remainder:
            if len(remainder)!=self.batch_size:raise RuntimeError('One complete mixed remainder batch required')
            batches.append(remainder)
        if len(batches)!=self.accumulation or any(len(batch)!=self.batch_size for batch in batches):
            raise RuntimeError('Effective optimizer batch changed')
        self.updates+=1
        return batches

    def coverage(self):
        domain={name:{'draws':stream.draws,'unique_clips_seen':len(stream.seen),'pool_clips':len(stream.rows),
            'repeat_visits':stream.draws-len(stream.seen),'completed_cycles':stream.draws//len(stream.rows),
            'cursor':stream.position,'cycles_started':stream.cycles_started,
            'fraction_unique_seen':len(stream.seen)/len(stream.rows),'audio_seen_seconds':self.domain_audio_seconds[name]}
            for name,stream in self.domains.items()}
        total_draws=sum(s.draws for s in self.domains.values());total_pool=sum(len(s.rows) for s in self.domains.values())
        return {'domain':domain,'turkish_draws':total_draws,'turkish_unique_clips_seen':sum(len(s.seen) for s in self.domains.values()),
            'turkish_draw_equivalent_epochs':total_draws/total_pool,
            'minimum_domain_exposure_epochs':min(s.draws/len(s.rows) for s in self.domains.values()),
            'all_Turkish_pool_sources_seen':all(len(s.seen)==len(s.rows) for s in self.domains.values()),
            'full_pool_pass_claim_requires_all_sources_seen':True,'domain_ratio_is_clip_weighted':True}

    def updates_for_minimum_domain_exposure(self,epochs):
        if not math.isfinite(epochs) or epochs<=0:raise ValueError('Positive domain exposure required')
        nf=math.ceil(epochs*len(self.domains['fleurs'].rows));nc=math.ceil(epochs*len(self.domains['cv'].rows))
        low,high=0,math.ceil(max(nf/float(self.fleurs_ratio),nc/(1-float(self.fleurs_ratio)))/self.tr_per_update)+2
        while low<high:
            middle=(low+high)//2;draws=middle*self.tr_per_update;actual_f=self.fleurs_quota_at(draws)
            if actual_f>=nf and draws-actual_f>=nc:high=middle
            else:low=middle+1
        return low

    def state_dict(self):
        return {'schema':1,'kind':'Turkish_domain_two_stream_bucket_sampler_v1','identity':self.identity.copy(),
            'rng':self.rng.getstate(),'updates':self.updates,'domains':{n:s.state_dict() for n,s in self.domains.items()},
            'replay':{n:s.state_dict() for n,s in self.replay.items()},'replay_language_counts':self.replay_language_counts.copy(),
            'domain_audio_seconds':self.domain_audio_seconds.copy()}

    def load_state_dict(self,state):
        if state.get('schema')!=1 or state.get('kind')!='Turkish_domain_two_stream_bucket_sampler_v1' or state.get('identity')!=self.identity:
            raise ValueError('Exact domain sampler layout/ratio/pool metadata identity changed')
        updates=state['updates']
        if type(updates) is not int or updates<0:raise ValueError('Nonnegative completed optimizer updates required')
        if list(state['domains'])!=list(self.domains) or list(state['replay'])!=list(self.replay):raise ValueError('Stream order changed')
        expected_f=self.fleurs_quota_at(updates*self.tr_per_update)
        for name,expected in (('fleurs',expected_f),('cv',updates*self.tr_per_update-expected_f)):
            if state['domains'][name]['draws']!=expected:raise ValueError('Domain quota/cursor counters disagree')
        if sum(s['draws'] for s in state['replay'].values())!=updates*self.replay_per_update:raise ValueError('Replay quota changed')
        if any(state['replay_language_counts'][lang]!=s['draws'] for lang,s in state['replay'].items()):raise ValueError('Replay language counters disagree')
        if set(state['domain_audio_seconds'])!=set(self.domains) or any(not math.isfinite(v) or v<0 for v in state['domain_audio_seconds'].values()):
            raise ValueError('Invalid domain audio exposure counters')
        for name,stream in self.domains.items():stream.load_state_dict(state['domains'][name])
        for name,stream in self.replay.items():stream.load_state_dict(state['replay'][name])
        self.rng.setstate(state['rng']);self.updates=updates
        self.replay_language_counts=state['replay_language_counts'].copy();self.domain_audio_seconds=state['domain_audio_seconds'].copy()
