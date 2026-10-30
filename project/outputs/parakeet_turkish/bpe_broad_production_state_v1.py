"""Bounded exact-pass state and journal primitives, without model/runtime imports."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random

KIND='BPE_broad_two_complete_passes_v1'
LANGUAGES=('en','es','it','fr','de','pt')

def require(ok,message):
    if not ok:raise ValueError(message)

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()

def digest(value):return hashlib.sha256(canonical(value)).hexdigest()

def permutation(index,count,key):
    """Eight-round balanced Feistel + cycle walking; bijective over [0,count)."""
    require(type(index) is int and 0<=index<count and count>0,'Permutation input outside exact population')
    half=max(1,((count-1).bit_length()+1)//2);mask=(1<<half)-1
    value=index
    while True:
        left,right=value>>half,value&mask
        for round_ in range(8):
            raw=hashlib.blake2b(canonical([key,round_,right]),digest_size=8).digest()
            left,right=right,left^(int.from_bytes(raw,'little')&mask)
        value=(left<<half)|right
        if value<count:return value

class CompletePassCursor:
    """No whole-pool order/seen copy; a committed prefix of a proven bijection."""
    def __init__(self,*,population,identity,seed,passes=2):
        require(type(population) is int and population>0 and passes in (1,2),'Final positive population and declared passes required')
        self.population=population;self.identity=deepcopy(identity);self.seed=seed;self.passes=passes;self.draws=0
    @property
    def remaining(self):return self.population*self.passes-self.draws
    def preview(self,count):
        require(type(count) is int and 1<=count<=min(14,self.remaining),'No tail discard, wrap or extra pass draw')
        return [{'ordinal':permutation((self.draws+j)%self.population,self.population,[self.seed,(self.draws+j)//self.population])+1,
                 'epoch':(self.draws+j)//self.population,'draw_index':self.draws+j} for j in range(count)]
    def commit(self,prepared):
        require(prepared==self.preview(len(prepared)),'Only exact prepared prefix can advance after optimizer commit')
        self.draws+=len(prepared)
    def state_dict(self):return {'kind':KIND,'population':self.population,'identity':deepcopy(self.identity),'seed':self.seed,'passes':self.passes,'draws':self.draws}
    def load_state_dict(self,value):
        require({k:v for k,v in value.items() if k!='draws'}=={k:v for k,v in self.state_dict().items() if k!='draws'},'Exact corpus/seed/pass identity differs')
        require(type(value['draws']) is int and 0<=value['draws']<=self.population*self.passes,'Invalid exact-pass frontier')
        self.draws=value['draws']
    def coverage(self):
        return {'population':self.population,'committed_draws':self.draws,'completed_full_passes':self.draws//self.population,
                'current_pass_unique_prefix':self.draws%self.population,'required_passes':self.passes,
                'complete':self.remaining==0,'no_excluded_or_unresolved_row_can_advance':True}

class CarriedReplay:
    """Original six queues plus independent selector/PT RNG, without reseeding."""
    def __init__(self,rows,snapshot,cycler_factory):
        require(snapshot['source_step']==64250 and set(snapshot['replay'])==set(LANGUAGES),'Original BPE64250 replay queues required')
        self.source_identity=snapshot['source_identity'];self.source_counts=deepcopy(snapshot['replay_language_counts'])
        self.rng=random.Random();self.rng.setstate(snapshot['replay_rng'])
        self.pt_rng=random.Random();self.pt_rng.setstate(snapshot['Portuguese_rng'])
        self.streams={}
        for language in LANGUAGES:
            saved=snapshot['replay'][language]
            stream=cycler_factory([r for r in rows if r['language']==language],self.pt_rng if language=='pt' else self.rng,bucket_size=saved['bucket_size'])
            stream.load_state_dict(saved);self.streams[language]=stream
            require(stream.draws==self.source_counts[language],'Original carried replay count differs')
        self.updates=0
    def next(self):
        old=self.rng.choice(LANGUAGES[:-1]);self.updates+=1
        return [self.streams[old].next(),self.streams['pt'].next()]
    def state_dict(self):
        return {'source_identity':self.source_identity,'source_counts':deepcopy(self.source_counts),'updates':self.updates,
            'replay':{l:s.state_dict() for l,s in self.streams.items()},'replay_rng':self.rng.getstate(),'Portuguese_rng':self.pt_rng.getstate()}
    def load_state_dict(self,value):
        require(value['source_identity']==self.source_identity and value['source_counts']==self.source_counts,'Replay lineage changed')
        n=value['updates'];require(type(n) is int and n>=0,'Positive replay update frontier required')
        for l,s in self.streams.items():s.load_state_dict(value['replay'][l])
        require(self.streams['pt'].draws==self.source_counts['pt']+n and
                sum(self.streams[l].draws-self.source_counts[l] for l in LANGUAGES[:-1])==n,'Carried +1PT/+1oldfive clocks differ')
        self.rng.setstate(value['replay_rng']);self.pt_rng.setstate(value['Portuguese_rng']);self.updates=n

class CommitJournal:
    """Append only small committed-update events. Checkpoint owns prefix authority."""
    def __init__(self,path):
        self.path=Path(path).resolve();require(str(self.path).startswith('E:\\gpu_training_scratchpad\\'),'Journal E-only')
        self.path.parent.mkdir(parents=True,exist_ok=True)
        if not self.path.exists():self.path.touch()
        self._hash=hashlib.sha256();self._offset=0
        with self.path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024**2),b''):self._hash.update(block);self._offset+=len(block)
    def append(self,event):
        import os
        raw=canonical(event)+b'\n'
        require(self.path.stat().st_size==self._offset,'Concurrent journal writer is prohibited')
        with self.path.open('ab') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
        self._hash.update(raw);self._offset+=len(raw)
        return self.binding()
    def binding(self):
        require(self.path.stat().st_size==self._offset,'Concurrent journal writer is prohibited')
        return {'path':str(self.path),'offset':self._offset,'sha256':self._hash.hexdigest()}
    def verify(self,value):
        require(Path(value['path']).resolve()==self.path and type(value['offset']) is int and 0<=value['offset']<=self.path.stat().st_size,'Exact committed prefix boundary required')
        sha=hashlib.sha256();remaining=value['offset']
        with self.path.open('rb') as stream:
            while remaining:
                block=stream.read(min(1024**2,remaining));require(block,'Truncated committed journal');sha.update(block);remaining-=len(block)
        return sha.hexdigest()==value['sha256']

def final_population(receipt,*,verify_record):
    """No incremental status/mean can masquerade as a final population."""
    require(receipt.get('kind')=='ASR18_final_production_population_v1' and receipt.get('complete') is True
            and receipt.get('all_pinned_sources_complete') is True and receipt.get('metadata_native_exit_code')==0
            and receipt.get('labels_native_exit_code')==0,'Complete pinned native-closed corpus required')
    n=receipt['retained_rows'];tokens=receipt['total_target_tokens']
    require(type(n) is int and n>0 and type(tokens) is int and tokens>=n and receipt['label_rows']==n
            and receipt['population_mean_U']==tokens/n,'Exact full-population sum/count/mean required')
    require(receipt['quality_filters']==[] and receipt['language_filters']==[] and receipt['pass_count']==2,'Unfiltered complete two-pass contract required')
    require(receipt['runtime_heldout_PCM_binding_complete'] is True and receipt['unresolved_geometry_rows']==0,
            'Heldout PCM binding and supported handling for every geometry required; unresolved rows stay pending')
    for item in receipt['bindings'].values():require(verify_record(item),'Final source/catalog/labels/heldout/geometry binding changed')
    return n,tokens/n

def weight_objective(objective,rows,population_mean_U,*,torch_module):
    """New final-corpus mean; no inherited180-token or old-pool restrictions."""
    from dataclasses import replace
    require(math.isfinite(population_mean_U) and population_mean_U>0,'Final fixed population denominator required')
    require(len(rows)==len(objective.target_tokens),'Exact target lengths required')
    weights=[]
    for row,u in zip(rows,objective.target_tokens):
        require(type(u) is int and u>0 and row['target_tokens']==u,'Full original targets required')
        weights.append(u/population_mean_U if row['language']=='tr' else 1.)
    original=objective.per_clip['tdt'];w=torch_module.tensor(weights,dtype=original.dtype,device=original.device)
    total=objective.per_clip['total']+original*(w-1.)
    return replace(objective,total=total.mean(),tdt=(original*w).mean(),per_clip={**objective.per_clip,'total':total,'tdt':original*w})
