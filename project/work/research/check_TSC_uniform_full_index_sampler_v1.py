"""Actual disk index + sampler on ten synthetic TRAIN rows; no real pool/ML."""
from pathlib import Path
import ast
import copy
import hashlib
import json
import math
import random
import sys
import tempfile
from array import array

ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'outputs/parakeet_turkish';sys.path.insert(0,str(HERE))
from tsc_domain_sampler_v3 import TSCDomainBalancedBucketSampler,TSC_REPO,TSC_REVISION
from tsc_uniform_full_sampler_v1 import TSCUniformFullSampler,IndexedBucketCycler,canonical
from lazy_training_rows_v1 import prepare_indexed_rows
from turkish_domain_sampler_v1 import FLEURS_REVISION,CV_REVISION,fingerprint
tree=ast.parse((HERE/'train_lora_batched_data_v11.py').read_text(encoding='utf-8'))
nodes=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in ('ids_sha','BucketCycler')]
ns={'hashlib':hashlib,'math':math};exec(compile(ast.Module(body=nodes,type_ignores=[]),'frozen BucketCycler AST','exec'),ns)
cycler=ns['BucketCycler'];checks=[];rejected=[]
def ok(value,message):
    if not value:raise AssertionError(message)
    checks.append(message)
def bad(call,message):
    try:call()
    except (ValueError,KeyError):checks.append(message);rejected.append(message);return
    raise AssertionError('unexpected acceptance '+message)
def row(source,language='tr',domain='cv'):
    dataset='google/fleurs' if domain=='fleurs' else TSC_REPO if domain=='tsc' else 'ysdede/commonvoice_17_tr_fixed'
    revision=FLEURS_REVISION if domain=='fleurs' else TSC_REVISION if domain=='tsc' else CV_REVISION
    return {'source_id':source,'language':language,'dataset':dataset,'dataset_revision':revision,
        'dataset_config':'tr_tr' if language=='tr' else 'pt_br' if language=='pt' else language,
        'dataset_split':'train','training_role':'train','text':'tiny training text','duration':1.,
        'encoder_frames':10+len(source),'target_tokens':3,'audio_sha256':'0'*64,'label_ids_sha256':'1'*64}
tr=[row('F'+str(i),domain='fleurs') for i in range(3)]+[row('C'+str(i)) for i in range(2)]+[row('T'+str(i),domain='tsc') for i in range(5)]
rp=[row(lang+str(i),language=lang,domain='fleurs') for lang in ('en','es','it','fr','de') for i in range(2)]
pt=[row('pt'+str(i),language='pt',domain='fleurs') for i in range(2)]
kwargs={'seed':11,'replay_seed':22,'batch_size':8,'accumulation':2,'bucket_size':4,'replay_fraction':.5,'fleurs_fraction':.3,
    'cycler_factory':cycler,'cycler_source_sha256':'2'*64}
old=TSCDomainBalancedBucketSampler(tr,rp,sampling_mode='domain30',**kwargs)
for _ in range(5):old.next_batches()
source=old.state_dict();before=canonical(source)
class CountRows:
    def __init__(self,parent):self.parent=parent;self.reads=0
    def __getattr__(self,name):return getattr(self.parent,name)
    def __len__(self):return len(self.parent)
    def __getitem__(self,index):self.reads+=1;return self.parent[index]
    def __iter__(self):raise AssertionError('Full decoded TRAIN iteration forbidden in sampler')
with tempfile.TemporaryDirectory(prefix='uniform_index_tiny_',dir=ROOT/'work/research') as directory:
    indexed=prepare_indexed_rows(iter(tr),Path(directory)/'rows',expected_count=len(tr),provenance={
        'synthetic_fixture_only':True,'training_admission_complete':True,'all_rows_verified':True,'verified_clips':len(tr)})
    catalog=CountRows(indexed)
    try:
        ok(indexed.fingerprint==fingerprint(tr) and indexed.ids_sha==ns['ids_sha'](tr),'disk scalar fingerprints match frozen exact functions')
        def fresh():return TSCUniformFullSampler(catalog,rp,portuguese_replay=pt,portuguese_seed=33,sampling_mode='uniform',**kwargs)
        new=fresh();evidence=new.migrate_source_state(source,old.identity)
        ok(catalog.reads==0,'constructor and source projection decode no Turkish dictionaries')
        ok(isinstance(new.turkish_streams['uniform'].rows,array) and new.turkish_streams['uniform'].rows.typecode=='I',
            'permanent Turkish permutation is compact uint32 index array')
        ok(isinstance(new.turkish_streams['uniform'].seen.bits,bytearray) and all(type(v) is int for v in new.tr_by_id.values()),
            'Turkish lookup values are indices and seen maps are compact bytes')
        ok(canonical(source)==before and new.updates==5 and new.tr_draws==40 and new.tr_unique==old.tr_unique,'original sampler and historical coverage preserved')
        ok(new.turkish_epoch==0 and new.tr_rng.getstate()==source['turkish_rng'] and new.replay_rng.getstate()==source['replay_rng'],'new uniform clock zero and old RNGs copied')
        for lang in ('en','es','it','fr','de'):ok(new.replay[lang].state_dict()==source['replay'][lang],'original '+lang+' replay native state exact')
        ok(new.replay['pt'].draws==0 and not evidence['old_five_future_replay_sequence_claimed'],'Portuguese fresh and future six-choice nuisance declared')
        reference_rng=random.Random();reference_rng.setstate(source['turkish_rng'])
        reference=cycler(tr,reference_rng,4);drawn=[]
        for update in range(2):
            count=catalog.reads;batches=new.next_batches();actual=[r['source_id'] for r in batches[0]]
            expected=[reference.next()['source_id'] for _ in range(8)]
            ok(actual==expected,'index shuffle/reservoir order equals frozen row cycler '+str(update))
            ok(catalog.reads-count==8,'only eight active TR rows materialized '+str(update));drawn+=actual
        ok(len(set(drawn[:10]))==10 and new.coverage()['uniform_pass']['full_pass_complete'],'one local cycle visits all sources without cropping')
        saved=new.state_dict();reads=catalog.reads
        restored=fresh();restored.migrate_source_state(source,old.identity);restored.load_state_dict(saved)
        ok(restored.state_dict()==saved and catalog.reads==reads,'own exact restore and snapshots decode no Turkish rows')
        for index in range(4):
            ok([[r['source_id'] for r in b] for b in new.next_batches()]==[[r['source_id'] for r in b] for b in restored.next_batches()],
                'future same-run TR/PT/replay sequence exact '+str(index))
        for key in ('updates','replay_language_counts','domain_draws','source_group_draws'):
            changed=copy.deepcopy(saved)
            if key=='updates':changed[key]+=1
            else:changed[key][next(iter(changed[key]))]+=1
            target=fresh();target.migrate_source_state(source,old.identity)
            bad(lambda s=changed,t=target:t.load_state_dict(s),'corrupt own '+key+' rejected')
        changed=copy.deepcopy(saved);changed['turkish_streams']['uniform']['order'][0]=changed['turkish_streams']['uniform']['order'][1]
        target=fresh();target.migrate_source_state(source,old.identity);bad(lambda:target.load_state_dict(changed),'duplicate indexed order rejected')
        changed=copy.deepcopy(source);changed['source_group_seen']['tsc']=[]
        bad(lambda:fresh().migrate_source_state(changed,old.identity),'historical source coverage cannot be erased')
        provenance=indexed.receipt['provenance'];indexed.receipt['provenance']={}
        bad(fresh,'profile/index without completed training admission rejected');indexed.receipt['provenance']=provenance
        ok('torch' not in sys.modules and 'trackio' not in sys.modules,'no ML or mirror imported')
    finally:indexed.close()
report={'schema':1,'kind':'inactive_TSC_uniform_full_index_sampler_tiny_CPU_verification_v1','passed':True,
    'checks':checks,'assertions':len(checks),'rejections':len(rejected),
    'sources':{name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ('tsc_uniform_full_sampler_v1.py','lazy_training_rows_v1.py')},
    'scope':'Actual disk indexed reader and sampler with frozen row cycler AST, ten synthetic TRAIN rows; no real corpus/source/PT/ML/GPU admission.',
    'actual_full_pool_loaded':False,'actual_source_admitted':False}
output=HERE/'TSC_uniform_full_index_sampler_tiny_CPU_verification_v2.json'
with output.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2);stream.write('\n')
print(json.dumps({'passed':True,'checks':len(checks),'rejections':len(rejected),'sources':report['sources'],
    'report_sha256':hashlib.sha256(output.read_bytes()).hexdigest()}))
