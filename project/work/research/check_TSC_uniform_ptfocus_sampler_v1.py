"""Tiny native indexed-selector restore fixture; no real state/corpus/ML."""
from pathlib import Path
from copy import deepcopy
import argparse
import hashlib
import json
import sys
import tempfile

from check_TSC_uniform_majority_sampler_v1 import (ROOT,HERE,row,CountRows,cycler,
    prepare_indexed_rows,TSCDomainBalancedBucketSampler,TSCUniformFullSampler,
    TSCUniformMajoritySampler,canonical)
from tsc_uniform_majority_ptfocus_sampler_v1 import TSCUniformMajorityPTFocusSampler

def main(args):
    assert not args.output.exists(),'Fresh proof path required'
    checks=[];rejects=[]
    def ok(value,label):
        if not value:raise AssertionError(label)
        checks.append(label)
    def bad(call,label):
        try:call()
        except (ValueError,KeyError):checks.append(label);rejects.append(label);return
        raise AssertionError('Unexpected acceptance '+label)
    tr=[row('F'+str(i),group='fleurs') for i in range(36)]+[row('C'+str(i)) for i in range(24)]+[row('T'+str(i),group='tsc') for i in range(60)]
    replay=[row(lang+str(i),lang,'fleurs') for lang in ('en','es','it','fr','de') for i in range(4)];pt=[row('pt'+str(i),'pt','fleurs') for i in range(4)]
    kwargs=dict(seed=11,replay_seed=22,batch_size=8,accumulation=2,bucket_size=8,replay_fraction=.5,fleurs_fraction=.3,cycler_factory=cycler,cycler_source_sha256='2'*64)
    domain=TSCDomainBalancedBucketSampler(tr,replay,sampling_mode='domain30',**kwargs)
    for _ in range(5):domain.next_batches()
    with tempfile.TemporaryDirectory(prefix='ptfocus_sampler_tiny_',dir=ROOT/'work/research') as directory:
        indexed=prepare_indexed_rows(iter(tr),Path(directory)/'rows',expected_count=120,
            provenance={'synthetic_fixture_only':True,'training_admission_complete':True,'all_rows_verified':True,'verified_clips':120})
        catalog=CountRows(indexed)
        try:
            full=TSCUniformFullSampler(catalog,replay,portuguese_replay=pt,portuguese_seed=33,sampling_mode='uniform',**kwargs)
            full.migrate_source_state(domain.state_dict(),domain.identity)
            for _ in range(3):full.next_batches()
            majority=TSCUniformMajoritySampler(catalog,replay,portuguese_replay=pt,portuguese_seed=33,sampling_mode='uniform',**{**kwargs,'replay_fraction':.125})
            majority.migrate_closed_state(full.state_dict(),full.identity)
            for _ in range(2):majority.next_batches()
            closed=majority.state_dict();before=canonical(closed)
            def fresh():return TSCUniformMajorityPTFocusSampler(catalog,replay,portuguese_replay=pt,portuguese_seed=33,sampling_mode='uniform',**{**kwargs,'replay_fraction':.125})
            changed=deepcopy(closed);changed['source_prefix']['domain_audio_seconds']['fleurs']+=1
            bad(lambda:fresh().migrate_closed_state(changed,closed['identity']),'Historical source prefix must match immutable quota digest')
            new=fresh();reads=catalog.reads;evidence=new.migrate_closed_state(closed,majority.identity);saved0=new.state_dict()
            ok(catalog.reads==reads,'Migration/snapshot decode zero TR dictionaries')
            ok(canonical(closed)==before and all(saved0[k]==closed[k] for k in closed if k not in ('kind','identity')),'Every source native queue/RNG/seen/counter/quota preserved')
            ok(new.focus_source_step==10 and new.transition_source_step==8 and new.source_prefix['updates']==5 and new.configured_end_step==15,'Selector/quota/uniform boundaries and finalhorizon distinct')
            bad(lambda:fresh().load_state_dict(saved0),'Own exact requires immutable selector origin first')
            batch=new.next_batches();flat=[r for b in batch for r in b]
            ok([len(b) for b in batch]==[8,8] and [r['language'] for r in batch[0]]==['tr']*8 and [r['language'] for r in batch[1]][:6]==['tr']*6,'Physical grouping8TR then6TR2RP unchanged')
            rp=[r for r in flat if r['language']!='tr'];ok(len(rp)==2 and rp[1]['language']=='pt' and rp[0]['language'] in ('en','es','it','fr','de'),'Exactlyone PT andone uniformoldfive replay selected')
            expected=[majority.turkish_streams['uniform'].next()['source_id'] for _ in range(14)]
            ok([r['source_id'] for r in flat if r['language']=='tr']==expected,'Original TRqueue suffix preserved without reshuffle')
            old=majority.replay_rng.choice(('en','es','it','fr','de'))
            expected_rp=[majority.replay[old].next()['source_id'],majority.replay['pt'].next()['source_id']]
            ok([r['source_id'] for r in rp]==expected_rp,'Carried oldselector RNG andindependent PTqueue give exact nextvisits')
            saved=new.state_dict();copy=fresh();copy.migrate_closed_state(closed,closed['identity']);copy.load_state_dict(saved)
            ok(copy.state_dict()==saved,'Ownnewkind restore preserves full typedstate')
            for _ in range(4):
                ok([[r['source_id'] for r in b] for b in new.next_batches()]==[[r['source_id'] for r in b] for b in copy.next_batches()],'Future ownexact sequences identical')
            final=new.state_dict();n=new.updates-new.focus_source_step
            ok(final['replay']['pt']['draws']-closed['replay']['pt']['draws']==n and sum(final['replay'][l]['draws']-closed['replay'][l]['draws'] for l in ('en','es','it','fr','de'))==n,'Native percategory drawclocks exactlyN/N')
            ok(new.coverage()['uniform_pass']['full_pass_complete'] and new.coverage()['uniform_pass']['repeat_visits']==2,'Samefullpass finishes with samewhole14 tailrepeats')
            restored=fresh();restored.migrate_closed_state(closed,closed['identity']);restored.load_state_dict(final);ok(restored.state_dict()==final,'Secondcycle tail exactrestorable')
            bad(new.next_batches,'No implicit extension after existing horizon')
            for key in ('updates','PT_focus_transition','quota_segments'):
                changed=deepcopy(saved)
                if key=='updates':changed[key]-=1
                elif key=='PT_focus_transition':changed[key]['source_step']-=1
                else:changed[key][1]['replay_per_update']=8
                target=fresh();target.migrate_closed_state(closed,closed['identity']);bad(lambda t=target,s=changed:t.load_state_dict(s),'Reject owncorrupt '+key)
            changed=deepcopy(saved);changed['replay']['pt']['draws']+=1
            target=fresh();target.migrate_closed_state(closed,closed['identity']);bad(lambda:target.load_state_dict(changed),'Cannotreset or alter PTnative drawclock')
            ok('torch' not in sys.modules and 'trackio' not in sys.modules,'No ML/GPU/mirror imports')
        finally:indexed.close()
    source=HERE/'tsc_uniform_majority_ptfocus_sampler_v1.py'
    report={'schema':1,'kind':'inactive_PT_focus_index_sampler_tiny_stdlib_verification_v1','passed':True,'assertions':len(checks),'rejections':len(rejects),'checks':checks,
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'scope':'Actual indexed disk/native cyclers with120 synthetic TR+24replay rows. No actual CLOSED8000/207Adam/corpus/ML/GPU admission.',
        'actual_source_or_GPU_restore_claimed':False}
    with args.output.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({'passed':True,'checks':len(checks),'rejects':len(rejects),'source_sha256':report['source_sha256'],'proof':str(args.output),'proof_sha256':hashlib.sha256(args.output.read_bytes()).hexdigest()}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=HERE/'TSC_uniform_majority_PT_focus_sampler_tiny_stdlib_verification_v2.json');main(parser.parse_args())
