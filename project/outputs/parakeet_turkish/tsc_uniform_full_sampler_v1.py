"""Declared full-pool uniform TR fork; historical exposure and replay state carry.

The two source domain queues cannot become a uniform cursor. A new full-pool
TR BucketCycler starts at localdraw0 using the source TR RNG. Its first cycle
visits every clip once. Historical counters/seen IDs stay separate and are
unioned in cumulative coverage. Portuguese has a separate fresh shuffle RNG;
the copied replay selector now chooses six languages, an explicit change.
No ML imports or corpus/state reads occur here.
"""
from __future__ import annotations
from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from array import array
import hashlib
import json
import math
import random

from tsc_domain_sampler_v3 import TSCDomainBalancedBucketSampler
from turkish_domain_sampler_v1 import fingerprint

KIND='TSC_uniform_full_TR_six_replay_sampler_v1'
OLD_KIND='TSC_FLEURS30_CV_TSC70_independent_replay_sampler_v3'
OLD_LANGUAGES=('en','es','it','fr','de')


def require(value,message):
    if not value:raise ValueError(message)


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def ids_sha(ids):
    digest=hashlib.sha256()
    for index,source in enumerate(sorted(ids)):
        if index:digest.update(b'\n')
        digest.update(source.encode())
    return digest.hexdigest()


class CatalogGroups(Mapping):
    """Scalar catalog view; never retains/materializes a Turkish row."""
    def __init__(self,rows,domain=False):self.catalog=rows;self.domain=domain
    def __len__(self):return len(self.catalog)
    def __iter__(self):return iter(self.catalog.source_ids)
    def __getitem__(self,source):
        group=self.catalog.groups[self.catalog.index_by_source_id[source]]
        return ('fleurs' if group=='fleurs' else 'cv_tsc') if self.domain else group


class IndexedIDSet:
    """One bit per source, exposing the native source-ID seen interface."""
    def __init__(self,catalog,ids=()):
        self.catalog=catalog;self.bits=bytearray(len(catalog));self.count=0
        for source in ids:self.add(source)
    def __len__(self):return self.count
    def __iter__(self):
        for index,seen in enumerate(self.bits):
            if seen:yield self.catalog.source_ids[index]
    def add_index(self,index):
        if not self.bits[index]:self.bits[index]=1;self.count+=1
    def add(self,source):self.add_index(self.catalog.index_by_source_id[source])
    def indices(self):return (index for index,seen in enumerate(self.bits) if seen)
    def __contains__(self,source):
        try:return bool(self.bits[self.catalog.index_by_source_id[source]])
        except KeyError:return False


class IndexedBucketCycler:
    """Frozen shuffle/reservoir semantics over compact integer indices."""
    def __init__(self,catalog,rng,bucket_size=128):
        require(len(catalog)>0 and type(bucket_size) is int and bucket_size>0,'Positive indexed pool/bucket required')
        self.catalog=catalog;self.rows=array('I',range(len(catalog)));self.rng=rng;self.bucket_size=bucket_size
        self.position=len(catalog);self.draws=0;self.cycles_started=0
        self.seen=IndexedIDSet(catalog);self.pool_sha256=catalog.ids_sha
    def next(self):
        if self.position==len(self.rows):
            self.rng.shuffle(self.rows)
            for start in range(0,len(self.rows),self.bucket_size):
                self.rows[start:start+self.bucket_size]=array('I',sorted(self.rows[start:start+self.bucket_size],key=self.catalog.length_key))
            self.position=0;self.cycles_started+=1
        index=self.rows[self.position];self.position+=1;self.draws+=1;self.seen.add_index(index)
        return self.catalog[index]
    def state_dict(self):
        return {'pool_sha256':self.pool_sha256,'order':[self.catalog.source_ids[index] for index in self.rows],
            'position':self.position,'draws':self.draws,'cycles_started':self.cycles_started,
            'seen':sorted(self.seen),'bucket_size':self.bucket_size}
    def load_state_dict(self,state):
        n=len(self.rows);require(state['pool_sha256']==self.pool_sha256 and state['bucket_size']==self.bucket_size
            and len(state['order'])==n,'Exact indexed pool/order/bucket identity required')
        order=array('I');visited=bytearray(n)
        for source in state['order']:
            index=self.catalog.index_by_source_id[source]
            require(not visited[index],'Duplicate indexed source order');visited[index]=1;order.append(index)
        position,cycles=state['position'],state['cycles_started']
        require(type(position) is int and 0<=position<=n and type(cycles) is int and cycles>=0,'Invalid indexed native cursor')
        seen=IndexedIDSet(self.catalog,state['seen'])
        require(len(seen)==len(state['seen']),'Duplicate indexed seen IDs')
        expected=bytearray(n)
        if cycles>1:expected[:]=b'\1'*n
        elif cycles==1:
            for index in order[:position]:expected[index]=1
        require(state['draws']==(0 if cycles==0 else (cycles-1)*n+position) and seen.bits==expected
            and (cycles!=0 or position==n),'Exact indexed native cycle/seen invariants differ')
        self.rows=order;self.position=position;self.cycles_started=cycles;self.draws=state['draws'];self.seen=seen


class TSCUniformFullSampler(TSCDomainBalancedBucketSampler):
    def __init__(self,tr,replay,*,portuguese_replay,portuguese_seed,**kwargs):
        require(kwargs.get('sampling_mode')=='uniform','Explicit uniform mode required')
        require(type(portuguese_seed) is int,'Explicit derived Portuguese shuffle seed required')
        # Do not invoke the legacy constructor: it indexes decoded TR rowdicts.
        for name in ('source_ids','index_by_source_id','encoder_frames','target_tokens','durations','groups','length_key','fingerprint','ids_sha','receipt'):
            require(hasattr(tr,name),'Verified indexed Turkish catalog required: '+name)
        admission=tr.receipt.get('provenance',{})
        require(tr.receipt.get('complete') is True and tr.receipt.get('rows')==len(tr)
            and admission.get('training_admission_complete') is True and admission.get('all_rows_verified') is True
            and admission.get('verified_clips')==len(tr),'Only completed whole-pool indexed training admission may be sampled')
        require(type(kwargs['seed']) is int and type(kwargs['replay_seed']) is int
            and len(kwargs['cycler_source_sha256'])==64,'Explicit copied sampler seeds/frozen replay source required')
        require(tr.fingerprint==tr.receipt.get('fingerprint') and tr.ids_sha==tr.receipt.get('source_ids_sha256')
            and len(tr.fingerprint)==64 and len(tr.ids_sha)==64,'Catalog scalar identity and complete receipt must match')
        require(len(tr)>0 and len(tr.source_ids)==len(tr) and len(tr.index_by_source_id)==len(tr),'Complete unique indexed Turkish catalog required')
        self.catalog=tr;self.tr_by_id=tr.index_by_source_id
        self.source_groups=CatalogGroups(tr);self.source_domains=CatalogGroups(tr,domain=True)
        self.group_pool_sizes=dict(Counter(tr.groups));self.domain_pool_sizes={'fleurs':self.group_pool_sizes.get('fleurs',0),
            'cv_tsc':self.group_pool_sizes.get('cv',0)+self.group_pool_sizes.get('tsc',0)}
        require(set(self.group_pool_sizes)=={'fleurs','cv','tsc'} and min(self.domain_pool_sizes.values())>0,'All three admitted Turkish sources required')
        require(self.group_pool_sizes==tr.receipt.get('source_group_counts'),'Compact source groups disagree with admitted full pool')
        self.tr_rng=random.Random(kwargs['seed']+1);self.replay_rng=random.Random(kwargs['replay_seed'])
        self.sampling_mode='uniform';self.batch_size=kwargs['batch_size'];self.accumulation=kwargs['accumulation']
        self.effective_batch=self.batch_size*self.accumulation
        fraction=kwargs.get('replay_fraction',.5);self.replay_per_update=round(self.effective_batch*fraction)
        require(self.batch_size>0 and self.accumulation>0 and math.isclose(self.replay_per_update,self.effective_batch*fraction,abs_tol=1e-9)
            and 0<=self.replay_per_update<self.effective_batch and kwargs.get('fleurs_fraction',.3)==.3,'Declared original batch/quota required')
        self.tr_per_update=self.effective_batch-self.replay_per_update
        bucket=kwargs.get('bucket_size',128);self.turkish_streams={'uniform':IndexedBucketCycler(tr,self.tr_rng,bucket)}
        replay_ids={row['source_id'] for row in replay}
        require(len(replay_ids)==len(replay) and not any(source in self.tr_by_id for source in replay_ids)
            and all(row.get('language') in OLD_LANGUAGES for row in replay),'Unique disjoint original replay required')
        self.replay={name:kwargs['cycler_factory']([row for row in replay if row['language']==name],self.replay_rng,bucket) for name in OLD_LANGUAGES}
        self.updates=0;self.replay_language_counts={name:0 for name in self.replay}
        self.domain_draws={name:0 for name in self.domain_pool_sizes};self.group_draws={name:0 for name in self.group_pool_sizes}
        self.domain_seen={name:IndexedIDSet(tr) for name in self.domain_pool_sizes};self.group_seen={name:IndexedIDSet(tr) for name in self.group_pool_sizes}
        self.domain_audio_seconds={name:0. for name in self.domain_pool_sizes};self.group_audio_seconds={name:0. for name in self.group_pool_sizes}
        self.identity={'sampling_mode':'uniform','turkish_seed':kwargs['seed'],'replay_seed':kwargs['replay_seed'],
            'replay_rng_independent_of_turkish':True,'batch_size':self.batch_size,'accumulation':self.accumulation,
            'replay_per_update':self.replay_per_update,'fleurs_fraction_numerator':3,'fleurs_fraction_denominator':10,
            'bucket_size':bucket,'cycler_source_sha256':kwargs['cycler_source_sha256'],'turkish_pool_fingerprint':tr.fingerprint,
            'replay_pool_fingerprint':fingerprint(replay),'sampler_family':'TSC_domain30_v3',
            'second_domain':'uniform_CV_plus_TSC','source_group_pool_counts':self.group_pool_sizes.copy()}
        require(tuple(self.replay)==OLD_LANGUAGES,'Original five replay stream order changed')
        pt_ids={row['source_id'] for row in portuguese_replay}
        require(portuguese_replay and len(pt_ids)==len(portuguese_replay)
            and all(row.get('language')=='pt' for row in portuguese_replay)
            and not pt_ids.intersection(self.tr_by_id)
            and not pt_ids.intersection(row['source_id'] for row in replay),'Separate disjoint Portuguese TRAIN stream required')
        self.pt_rng=random.Random(portuguese_seed)
        self.replay['pt']=kwargs['cycler_factory'](portuguese_replay,self.pt_rng,kwargs.get('bucket_size',128))
        self.replay_language_counts['pt']=0
        self.original_replay_fingerprint=self.identity['replay_pool_fingerprint']
        self.identity.update(sampler_family=KIND,source_sampling_mode='domain30',
            turkish_uniform_pool_size=len(tr),portuguese_seed=portuguese_seed,
            original_replay_pool_fingerprint=self.original_replay_fingerprint,
            replay_pool_fingerprint=fingerprint([*replay,*portuguese_replay]),
            replay_languages=list(self.replay),portuguese_pool_fingerprint=fingerprint(portuguese_replay),
            replay_language_choices_changed_to_six=True,old_five_future_replay_sequence_claimed=False,
            source_fleurs_fraction_legacy_only=True,uniform_epoch_clock='new_uniform_TR_draws_over_full_pool')
        self.identity.update(TR_storage='indexed_disk_rows_bounded_cache',TR_queue='compact_uint32_indices',
            TR_seen='compact_byte_maps',TR_rowdict_pool_retained=False,
            indexed_TR_fingerprint=tr.fingerprint,indexed_TR_ids_sha256=tr.ids_sha)
        self.source_prefix=None

    @property
    def turkish_epoch(self):
        return self.turkish_streams['uniform'].draws/len(self.tr_by_id)

    @property
    def uniform_pass_updates(self):
        return math.ceil(len(self.tr_by_id)/self.tr_per_update)

    def _old_tr_counters(self,state):
        draws={name:0 for name in self.domain_pool_sizes};seen={name:set() for name in draws}
        groups={name:0 for name in self.group_pool_sizes};group_seen={name:set() for name in groups}
        require(list(state['turkish_streams'])==['fleurs','cv_tsc'],'Original domain stream ordering changed')
        for domain,stream in state['turkish_streams'].items():
            members={sid for sid,value in self.source_domains.items() if value==domain}
            order=stream['order'];position=stream['position'];cycles=stream['cycles_started']
            require(len(order)==len(members) and len(set(order))==len(order) and set(order)==members
                and stream['pool_sha256']==ids_sha(members) and stream['bucket_size']==self.identity['bucket_size']
                and type(position) is int and 0<=position<=len(order) and type(cycles) is int and cycles>=0,
                'Original domain queue/order/pool/cursor changed')
            expected_draws=0 if cycles==0 else (cycles-1)*len(order)+position
            expected_seen=set() if cycles==0 else set(order[:position]) if cycles==1 else members
            require(stream['draws']==expected_draws and set(stream['seen'])==expected_seen
                and (cycles!=0 or position==len(order)),'Original native cycle/seen invariants changed')
            draws[domain]=expected_draws;seen[domain]=expected_seen
            counts=Counter(self.source_groups[sid] for sid in order)
            for name,count in counts.items():groups[name]+=max(0,cycles-1)*count
            for sid in (order[:position] if cycles else []):groups[self.source_groups[sid]]+=1
            for sid in expected_seen:group_seen[self.source_groups[sid]].add(sid)
        require(state['domain_draws']==draws and state['domain_seen']=={name:sorted(ids) for name,ids in seen.items()}
            and state['source_group_draws']==groups and state['source_group_seen']=={name:sorted(ids) for name,ids in group_seen.items()},
            'Historical source/domain draws and actual unique IDs changed')
        require(sum(draws.values())==state['updates']*self.tr_per_update
            and draws['fleurs']==sum(draws.values())*3//10,'Original 30% prefix quota changed')
        for key,pools in (('domain_audio_seconds',self.domain_pool_sizes),('source_group_audio_seconds',self.group_pool_sizes)):
            require(set(state[key])==set(pools) and all(type(value) in (int,float) and math.isfinite(value) and value>=0
                for value in state[key].values()),'Finite unchanged source audio counters required')
        require(math.isclose(math.fsum(state['domain_audio_seconds'].values()),math.fsum(state['source_group_audio_seconds'].values()),
            rel_tol=1e-10,abs_tol=1e-7),'Source domain/source audio counters disagree')

    def migrate_source_state(self,state,source_identity):
        require(state.get('schema')==3 and state.get('kind')==OLD_KIND and state.get('identity')==source_identity
            and source_identity.get('sampling_mode')=='domain30' and source_identity.get('sampler_family')=='TSC_domain30_v3'
            and type(state.get('updates')) is int and state['updates']>0,'Whole actual domain30 source sampler required')
        for key in ('turkish_seed','replay_seed','batch_size','accumulation','replay_per_update','bucket_size','cycler_source_sha256',
                    'turkish_pool_fingerprint','source_group_pool_counts'):
            require(source_identity.get(key)==self.identity[key],'Source/input/layout changed: '+key)
        require(source_identity.get('replay_pool_fingerprint')==self.original_replay_fingerprint
            and source_identity.get('replay_rng_independent_of_turkish') is True
            and tuple(state['replay'])==OLD_LANGUAGES,'Original replay identity/order changed')
        self._old_tr_counters(state)
        for name in OLD_LANGUAGES:self.replay[name].load_state_dict(state['replay'][name])
        require(state['replay_language_counts']=={name:self.replay[name].draws for name in OLD_LANGUAGES}
            and sum(state['replay_language_counts'].values())==state['updates']*self.replay_per_update,
            'Original replay native cursors/draws/counts disagree')
        self.tr_rng.setstate(state['turkish_rng']);self.replay_rng.setstate(state['replay_rng'])
        self.updates=state['updates'];self.replay_language_counts={**state['replay_language_counts'],'pt':0}
        self.domain_draws=state['domain_draws'].copy();self.domain_seen={name:IndexedIDSet(self.catalog,ids) for name,ids in state['domain_seen'].items()}
        self.domain_audio_seconds=state['domain_audio_seconds'].copy()
        self.group_draws=state['source_group_draws'].copy();self.group_seen={name:IndexedIDSet(self.catalog,ids) for name,ids in state['source_group_seen'].items()}
        self.group_audio_seconds=state['source_group_audio_seconds'].copy()
        self.source_prefix={key:deepcopy(state[key]) for key in ('updates','domain_draws','domain_seen','domain_audio_seconds',
            'source_group_draws','source_group_seen','source_group_audio_seconds','replay_language_counts')}
        self.source_prefix.update(source_identity=deepcopy(source_identity),source_sampler_sha256=canonical(state),
            old_minimum_domain_exposure=min(state['domain_draws'][name]/count for name,count in self.domain_pool_sizes.items()))
        self.identity.update(source_boundary_step=self.updates,source_domain_identity_sha256=canonical(source_identity),
            source_sampler_sha256=self.source_prefix['source_sampler_sha256'])
        require(self.turkish_streams['uniform'].draws==0 and not self.turkish_streams['uniform'].seen
            and self.replay['pt'].draws==0,'Fresh uniform/PT queues must not pre-draw')
        return {'kind':'declared_TSC_domain30_to_uniform_full_TR_with_Portuguese_v1','passed':True,
            'source_boundary_step':self.updates,'historical_TR_and_source_seen_counters_preserved':True,
            'old_five_replay_state_and_selector_RNG_preserved_at_boundary':True,
            'future_replay_language_selection_changes_to_six':True,'old_five_future_replay_sequence_claimed':False,
            'source_TR_RNG_copied_before_fresh_uniform_shuffle':True,'source_two_TR_orders_not_relabelled_as_uniform':True,
            'new_uniform_TR_queue_draws':0,'new_Portuguese_queue_draws':0,'new_Portuguese_shuffle_seed':self.identity['portuguese_seed'],
            'new_uniform_pass_updates':self.uniform_pass_updates,'new_uniform_pass_end_step':self.updates+self.uniform_pass_updates,
            'last_full_batch_extra_next_cycle_repeats':self.uniform_pass_updates*self.tr_per_update-len(self.tr_by_id),
            'epoch_clock_changed_explicitly':True,'source_epoch_clock':self.source_prefix['old_minimum_domain_exposure'],
            'new_uniform_epoch_clock_at_boundary':0.,'source_sampler_sha256':self.source_prefix['source_sampler_sha256']}

    def next_batches(self):
        require(self.source_prefix is not None,'Actual source migration or own exact state load required before draws')
        return super().next_batches()

    def coverage(self):
        result=super().coverage();stream=self.turkish_streams['uniform'];start=self.source_prefix['updates'] if self.source_prefix else None
        result.update(epoch_clock='new_uniform_pass_draw_equivalent',uniform_pass={
            'source_step':start,'local_updates':self.updates-start if start is not None else 0,
            'draws':stream.draws,'unique_clips_seen':len(stream.seen),'pool_clips':len(self.tr_by_id),
            'repeat_visits':stream.draws-len(stream.seen),'fraction_unique_seen':len(stream.seen)/len(self.tr_by_id),
            'full_pass_complete':len(stream.seen)==len(self.tr_by_id),'epoch_clock':self.turkish_epoch},
            inherited_30percent_history_not_claimed_uniform=True,Portuguese_replay_draws=self.replay['pt'].draws)
        return result

    def state_dict(self):
        require(self.source_prefix is not None,'No unsourced uniform sampler serialization')
        value=super().state_dict();value.update(schema=1,kind=KIND,source_prefix=deepcopy(self.source_prefix),Portuguese_rng=self.pt_rng.getstate())
        return value

    def load_state_dict(self,state):
        require(self.source_prefix is not None and state.get('schema')==1 and state.get('kind')==KIND
            and state.get('identity')==self.identity and canonical(state.get('source_prefix'))==canonical(self.source_prefix),
            'Own exact identity/source-prefix required; never load domain state as uniform')
        require(list(state['turkish_streams'])==['uniform'] and tuple(state['replay'])==(*OLD_LANGUAGES,'pt'),'Own uniform/replay stream order changed')
        stream=self.turkish_streams['uniform'];stream.load_state_dict(state['turkish_streams']['uniform'])
        for name,value in self.replay.items():value.load_state_dict(state['replay'][name])
        require(type(state['updates']) is int and state['updates']>=self.source_prefix['updates']
            and stream.draws==(state['updates']-self.source_prefix['updates'])*self.tr_per_update
            and sum(value.draws for value in self.replay.values())==state['updates']*self.replay_per_update
            and state['replay_language_counts']=={name:value.draws for name,value in self.replay.items()},'Actual uniform global/local draw frontiers differ')
        draws=self.source_prefix['domain_draws'].copy();groups=self.source_prefix['source_group_draws'].copy()
        seen={name:IndexedIDSet(self.catalog,ids) for name,ids in self.source_prefix['domain_seen'].items()}
        group_seen={name:IndexedIDSet(self.catalog,ids) for name,ids in self.source_prefix['source_group_seen'].items()}
        counts=self.domain_pool_sizes
        group_counts=self.group_pool_sizes
        for name,count in counts.items():draws[name]+=max(0,stream.cycles_started-1)*count
        for name,count in group_counts.items():groups[name]+=max(0,stream.cycles_started-1)*count
        for index in (stream.rows[:stream.position] if stream.cycles_started else []):
            group=self.catalog.groups[index];domain='fleurs' if group=='fleurs' else 'cv_tsc'
            draws[domain]+=1;groups[group]+=1
        for index in stream.seen.indices():
            group=self.catalog.groups[index];domain='fleurs' if group=='fleurs' else 'cv_tsc'
            seen[domain].add_index(index);group_seen[group].add_index(index)
        require(state['domain_draws']==draws and state['source_group_draws']==groups
            and state['domain_seen']=={name:sorted(ids) for name,ids in seen.items()}
            and state['source_group_seen']=={name:sorted(ids) for name,ids in group_seen.items()},'Own cumulative carried+uniform coverage changed')
        for key,pools in (('domain_audio_seconds',self.domain_pool_sizes),('source_group_audio_seconds',self.group_pool_sizes)):
            require(set(state[key])==set(pools) and all(type(v) in (int,float) and math.isfinite(v) and v>=self.source_prefix[key][name]
                for name,v in state[key].items()),'Own cumulative audio counters invalid')
        require(math.isclose(math.fsum(state['domain_audio_seconds'].values()),math.fsum(state['source_group_audio_seconds'].values()),
            rel_tol=1e-10,abs_tol=1e-7),'Own domain/source audio disagrees')
        self.tr_rng.setstate(state['turkish_rng']);self.replay_rng.setstate(state['replay_rng']);self.pt_rng.setstate(state['Portuguese_rng'])
        self.updates=state['updates'];self.replay_language_counts=state['replay_language_counts'].copy()
        self.domain_draws=draws;self.domain_seen=seen;self.domain_audio_seconds=state['domain_audio_seconds'].copy()
        self.group_draws=groups;self.group_seen=group_seen;self.group_audio_seconds=state['source_group_audio_seconds'].copy()
