"""Synthetic scalar/official-text controls only; never admit DATA/native work."""
import argparse
from copy import deepcopy
import json, os, sys
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT/'outputs/parakeet_turkish'))
import bpe_full_run_lifecycle_v2 as module
from bpe_broad_production_state_v1 import digest, permutation
from fastmss_row_serial_cpu_v2 import environment, resource_guard, admit
from fastmss_owned_process_closure_v1 import identity
from metrics import score_transcriptions


def metadata_state(n=15, b=0, c=0):
    def cursor(population, draws, passes, seed):
        return {'kind': 'BPE_broad_two_complete_passes_v1', 'population': population,
                'draws': draws, 'passes': passes, 'seed': seed, 'identity': {'synthetic_only': True}}
    phase = module.WEAK if b < 2*n else module.CLEANER if c < 352768 else module.COMPLETE
    u = sum((draws//pop)*((pop+13)//14)+(draws % pop)//14 for draws, pop in ((b,n), (c,352768)))
    return {'kind': module.SCHEDULE_KIND, 'phase': phase, 'broad_cursor': cursor(n,b,2,'fixture-broad'),
            'cleaner_cursor': cursor(352768,c,1,'fixture-cleaner'), 'carried_replay': {'updates': u},
            'population_mean_U': {module.WEAK: 20., module.CLEANER:11972886/352768}}


def metadata_plan(before, updates):
    state=deepcopy(before); result=[]; phase=state['phase']; cursor_key='broad_cursor' if phase==module.WEAK else 'cleaner_cursor'
    for index in range(updates):
        previous=digest(state); p=module.progress(state); cursor=state[cursor_key]; n,start=cursor['population'],cursor['draws']; epoch=start//n
        count=min(14,n-start % n)
        draws=[{'ordinal':permutation((start+j)%n,n,[cursor['seed'],epoch])+1,'epoch':epoch,'draw_index':start+j} for j in range(count)]
        cursor['draws']+=count; state['carried_replay']['updates']+=1
        state['phase']=module.WEAK if state['broad_cursor']['draws']<2*state['broad_cursor']['population'] else module.CLEANER if state['cleaner_cursor']['draws']<352768 else module.COMPLETE
        result.append({'phase':phase,'global_update':p['optimizer_updates']+1,'before_sampler_sha256':previous,
            'after_sampler_sha256':digest(state),'TR_draws':draws,'rows':[{'synthetic_only':True}]*(count+2),
            'replay_rows':[{'language':'en'},{'language':'pt'}]})
    return {'kind':'ASR18_phase_aware_provisional_production_window_plan_v1','production_enabled':False,
            'phase':phase,'initial_sampler_sha256':digest(before),'initial_committed_updates':module.progress(before)['optimizer_updates'],
            'updates':result,'final_private_planning_sampler':deepcopy(state)}


def pointer(after, diagnostic=False):
    u=module.progress(after)['optimizer_updates']; prefix={'path':'E:\\synthetic-prefix-only','offset':u,'sha256':'f'*64}
    return {'kind':'ASR18_production_window_completed_checkpoint_pointer_v2','sampler_state_sha256':digest(after),
            'committed_updates':u,'all_window_updates_committed':True,'writer_returned_and_payload_verified':True,
            'saved_semantic_state_verified':{'passed':True,'optimizer_updates':u,'sampler_state_sha256':digest(after),'journal_prefix':prefix},
            'journal_prefix':prefix,'diagnostic_only':diagnostic}


def main(output):
    environment(); resource_guard(); forecast=admit(80_000_000)
    output=Path(output).resolve(); module.require(output.is_relative_to(module.ROOT) and not output.exists(),'Fresh E-only synthetic metadata fixture')
    output.mkdir(); checks=[]
    def check(name, value):
        resource_guard(); module.require(value,name); checks.append(name)
    def rejects(name, function):
        try:function()
        except (ValueError,KeyError,TypeError):checks.append(name);return
        raise ValueError('Expected rejection: '+name)
    source_before=module.record(module.__file__)
    check('No Torch/frontend/numerical import',not any(k=='torch' or k.startswith(('torch.','numpy','scipy','transformers','pyarrow')) for k in sys.modules))
    s=metadata_state(); request=module.next_boundary_request(s,0)
    check('First weak window capped at broad1 tail',request['maximum_updates']==2)
    plan=metadata_plan(s,2); module.validate_window_plan(s,request,plan); checks.append('Exact14+1 broad pass tail accepted')
    cross=metadata_plan(s,3)
    rejects('Weak pass1-to-pass2 crossing rejected before preparation',lambda:module.validate_window_plan(s,request,cross))
    bad=deepcopy(plan);bad['updates'][0]['TR_draws'][0]['ordinal']=999
    rejects('Original seeded ordinal mismatch rejected',lambda:module.validate_window_plan(s,request,bad))
    bad=deepcopy(plan);bad['updates'][1]['TR_draws'][0]['epoch']=1
    rejects('Wrong pass epoch rejected',lambda:module.validate_window_plan(s,request,bad))
    bad=deepcopy(plan);bad['updates'][1]['replay_rows'].pop()
    rejects('Missing carried replay pair rejected',lambda:module.validate_window_plan(s,request,bad))
    bad=deepcopy(plan);bad['updates'][1]['before_sampler_sha256']='0'*64
    rejects('Broken checkpoint sampler chain rejected',lambda:module.validate_window_plan(s,request,bad))
    bad=deepcopy(plan);bad['final_private_planning_sampler']['carried_replay']['updates']=99
    rejects('Prepared virtual state cannot fabricate replay frontier',lambda:module.validate_window_plan(s,request,bad))
    for field, replacement in (('seed','changed'),('identity',{'changed':True}),('population',16)):
        bad=deepcopy(plan);bad['final_private_planning_sampler']['broad_cursor'][field]=replacement
        bad['updates'][-1]['after_sampler_sha256']=digest(bad['final_private_planning_sampler'])
        rejects('Final planned corpus '+field+' reset rejected',lambda:module.validate_window_plan(s,request,bad))
    bad=deepcopy(plan);bad['final_private_planning_sampler']['population_mean_U'][module.WEAK]=21.
    bad['updates'][-1]['after_sampler_sha256']=digest(bad['final_private_planning_sampler'])
    rejects('Final full-population denominator cannot change',lambda:module.validate_window_plan(s,request,bad))
    # Exercise the real original TrainingSchedule and make_plan implementations
    # with a distinctly synthetic metadata-only replay/resolver protocol.
    from bpe_training_schedule_v1 import TrainingSchedule
    class SyntheticReplay:
        def __init__(self): self.updates=0
        def next(self): self.updates+=1;return [{'source_id':f'fixture-en-{self.updates}','language':'en'},{'source_id':f'fixture-pt-{self.updates}','language':'pt'}]
        def state_dict(self):return {'updates':self.updates,'source_identity':'fixture-source','source_counts':{'fixture':0}}
        def load_state_dict(self,v):self.updates=v['updates']
    def new_schedule():
        return TrainingSchedule(broad_population=15,broad_identity={'synthetic_only':True},broad_mean_U=20.,
            cleaner_population=352768,cleaner_identity={'synthetic_only':True},cleaner_mean_U=11972886/352768,
            source_turkish_rng={'synthetic_only':True},replay=SyntheticReplay())
    def resolve(phase,ordinal):return {'ordinal':ordinal,'source_id':f'fixture-{phase}-{ordinal}','language':'tr'}
    def rows(phase,draws,replay):return [resolve(phase,d['ordinal']) for d in draws]+list(replay)
    fixture_pin_path=output/'synthetic_population_protocol_only.json';fixture_pin_path.write_text('{"synthetic_only":true,"not_final_DATA":true}')
    fixture_pin=module.record(fixture_pin_path)
    def planning():return {'clone_schedule':new_schedule,'resolve_identity':resolve,'resolve_rows':rows,
        'feature_bytes_upper_bound':lambda phase,rows:len(rows)*4096,'finalized_population':fixture_pin,
        'verify_finalized_population':lambda pin:pin==fixture_pin}
    real=new_schedule();real_before=deepcopy(real.state_dict())
    real_plan=module.make_pass_bounded_plan(real,completed_boundaries=0,**planning())
    check('Real schedule/planner wrapper stops weak pass1 tail',len(real_plan['updates'])==2 and real_plan['final_private_planning_sampler']['broad_cursor']['draws']==15)
    check('Real planner leaves original cursor/replay state exact',real.state_dict()==real_before)
    real.load_state_dict(real_plan['final_private_planning_sampler'])
    rejects('Real planner blocks next pass until DEV acknowledgment',lambda:module.make_pass_bounded_plan(real,completed_boundaries=0,**planning()))
    real_plan2=module.make_pass_bounded_plan(real,completed_boundaries=1,**planning())
    check('Real planner supports complete second broad tail',len(real_plan2['updates'])==2 and real_plan2['final_private_planning_sampler']['broad_cursor']['draws']==30)
    real.load_state_dict(real_plan2['final_private_planning_sampler'])
    real_clean=module.make_pass_bounded_plan(real,completed_boundaries=2,**planning())
    check('Real planner cleaner route bounded20 without reset',real_clean['phase']==module.CLEANER and len(real_clean['updates'])==20 and real_clean['final_private_planning_sampler']['cleaner_cursor']['draws']==280)
    bad=deepcopy(real_plan);bad['final_private_planning_sampler']['carried_replay']['source_identity']='changed'
    bad['updates'][-1]['after_sampler_sha256']=digest(bad['final_private_planning_sampler'])
    rejects('Carried replay lineage cannot reset',lambda:module.validate_window_plan(real_before,module.next_boundary_request(real_before,0),bad))
    initial_winner={'artifact':{'synthetic_winner_only':True},'word_errors':{'tr':259,**module.LIMITS},'reference_words':module.REFERENCES,'strict_qualified':True}
    protocol=module._LifecycleProtocol(s,{'fixture':True},initial_winner); after=plan['final_private_planning_sampler']; pin={'synthetic_pointer':1}
    rejects('Diagnostic20/21 cannot be promoted',lambda:protocol.accept_window(s,plan,after,pin,pointer(after,True)))
    action=protocol.accept_window(s,plan,after,pin,pointer(after));check('Actual tail blocks next training',action['kind']=='checkpoint_then_serial350DEV')
    rejects('Unacknowledged pass1 cannot be skipped',lambda:module.next_boundary_request(metadata_state(b=29),0))
    rejects('Boolean counter rejected',lambda:module.progress({**s,'broad_cursor':{**s['broad_cursor'],'draws':True}}))
    rejects('Cleaner cannot start before two broad passes',lambda:module.progress(metadata_state(b=15,c=14)))
    # Complete seven-language text fixtures, explicitly synthetic and never
    # passed through any public native owner/admission callback.
    policy={'official_scorer':module.record(module.HERE/'metrics.py'),'integer_gate_contract':module.record(module.HERE/'tsc_bpe_full_decoder_transition_v1.py'),
            'reference_words':deepcopy(module.REFERENCES),'retention_limits':deepcopy(module.LIMITS),'DEV_manifests':{}}
    def write(name,value,jsonl=False):
        path=output/name; text='\n'.join(json.dumps(v,ensure_ascii=False) for v in value)+'\n' if jsonl else json.dumps(value,ensure_ascii=False)
        path.write_text(text,encoding='utf-8');return module.record(path)
    manifests={}
    for language,words in module.REFERENCES.items():
        q,r=divmod(words,50); rows=[{'source_id':f'synthetic-{language}-{i}','language':language,'text':' '.join(['merhaba']*(q+(i<r)))} for i in range(50)]
        manifests[language]=rows;policy['DEV_manifests'][language]=write('manifest_'+language+'.jsonl',rows,True)
    def make_DEV(name,errors):
        report={'kind':'ASR18_serial_complete350_boundary_DEV_v1','passed':True,'serial_optimizer_free':True,'full350_DEV_complete':True,
            'split':'validation','TEST_used':False,'checkpoint_pointer':pin,'scorer_policy':deepcopy(policy),'predictions':{},'scores':{}}
        for language,manifest in manifests.items():
            remaining=errors.get(language,0); rows=[]
            for source in manifest:
                tokens=source['text'].split(); count=min(remaining,len(tokens));remaining-=count
                rows.append({'source_id':source['source_id'],'language':language,'split':'validation','reference':source['text'],
                    'hypothesis':' '.join(['yanlis']*count+tokens[count:])})
            module.require(remaining==0,'Fixture errors fit complete references')
            report['predictions'][language]=write(name+'_'+language+'.jsonl',rows,True)
            score=score_transcriptions(rows,language=language);score.pop('details');report['scores']['validation_'+language]={'asr':score}
        return report
    report=make_DEV('pass',{'tr':258,**module.LIMITS}); score=module.official_DEV(report,pin,policy)
    check('Immutable official350 rescore/all6 integer equality passes',score['strict_all_six_passed'] and score['word_errors']['tr']==258)
    candidate={'synthetic_winner_only':2}; result=protocol.accept_DEV(after,{'synthetic_DEV':1},score,candidate)
    check('Strict-qualified lower TR replaces winner',result['winner_changed'] and protocol.state['winner']['artifact']==candidate)
    check('Pass1 DEV unlocks broad2 without run shortening',result['next']['phase']==module.WEAK and result['next']['boundary']['name']=='broad_pass_2')
    saved=deepcopy(protocol.state); restored=module._LifecycleProtocol(after,{'fixture':True},None,restored=saved)
    check('Exact lifecycle/scientific resume retains qualified winner',restored.state==saved)
    bad=deepcopy(saved);bad['last_sampler_sha256']='0'*64
    rejects('Restore sampler mismatch rejected',lambda:module._LifecycleProtocol(after,{'fixture':True},None,restored=bad))
    bad=deepcopy(saved);bad['completed_boundaries']*=2
    rejects('Noncontiguous duplicate pass acknowledgments rejected',lambda:module._LifecycleProtocol(after,{'fixture':True},None,restored=bad))
    bad=deepcopy(report);bad['TEST_used']=True
    rejects('TEST predictions never gate this lifecycle',lambda:module.official_DEV(bad,pin,policy))
    bad=deepcopy(report);bad['scores']['validation_tr']['asr']['word_errors']=257
    rejects('Claimed summary disagrees with official saved text',lambda:module.official_DEV(bad,pin,policy))
    bad=deepcopy(report);bad['scorer_policy']['retention_limits']['pt']=67
    rejects('Relaxed integer PT threshold rejected',lambda:module.official_DEV(bad,pin,policy))
    bad=deepcopy(report);bad['predictions'].pop('pt')
    rejects('Incomplete seven-language DEV rejected',lambda:module.official_DEV(bad,pin,policy))
    bad=deepcopy(report);bad['checkpoint_pointer']={'wrong':True}
    rejects('Different checkpoint DEV rejected',lambda:module.official_DEV(bad,pin,policy))
    # Broad2 and failure gating use actual scalar tail metadata, not CPU Adam.
    plan2=metadata_plan(after,2); after2=plan2['final_private_planning_sampler']; protocol.accept_window(after,plan2,after2,pin,pointer(after2))
    fail_report=make_DEV('failed',{'tr':1,**module.LIMITS,'pt':67});failed=module.official_DEV(fail_report,pin,policy)
    result=protocol.accept_DEV(after2,{'synthetic_DEV':2},failed,candidate)
    check('Failed retention preserves qualified winner',not result['winner_changed'] and protocol.state['winner']['word_errors']['tr']==258)
    check('Failed retention continues complete cleaner phase',result['next']['phase']==module.CLEANER and not result['retention_early_stop'])
    # Resume close to real cleaner tail using synthetic-only scalar checkpoint
    # state; no claim that the omitted ~25k actual updates were executed.
    tail=metadata_state(b=30,c=352758); snapshot=deepcopy(protocol.state);snapshot['last_sampler_sha256']=digest(tail)
    last=module._LifecycleProtocol(tail,{'fixture':True},None,restored=snapshot)
    tail_request=last.request(tail);check('Cleaner exact10 tail preserves complete pool',tail_request['maximum_updates']==1)
    tail_plan=metadata_plan(tail,1);done=tail_plan['final_private_planning_sampler'];last.accept_window(tail,tail_plan,done,pin,pointer(done))
    check('All draws alone still await complete finalDEV',last.request(done)['kind']=='checkpoint_then_serial350DEV' and not last.state['training_complete'])
    # Equal qualified score does not replace or reset the source winner.
    result=last.accept_DEV(done,{'synthetic_DEV':3},score,candidate)
    check('Equal TR score preserves winner',not result['winner_changed'])
    check('Exactly2 broad +1 cleaner +3DEV boundaries complete',last.state['training_complete'] and len(last.state['completed_boundaries'])==3)
    check('TEST and export stay unfinished',last.state['TEST_complete'] is False and last.state['export_complete'] is False)
    check('No fourth pass window available',last.request(done)['kind']=='training_and_declared_DEV_complete')
    storage={'kind':'ASR18_reviewed_owned_checkpoint_storage_boundary_v1','passed':True,'policy':{'synthetic':True},
        'owner':'/root/memory_diagnosis','bounded_publication_and_retirement_implementation_reviewed':True,
        'preserve_qualified_winner_coherent_latest_source_history_DEV_and_journal':True,
        'existing_or_original_artifact_deletion_authorized':False,'unbounded_per_window_checkpoint_retention':False,
        'maximum_additional_owned_bytes':244398096}
    check('Bounded prospective checkpoint accounting accepted',module.checkpoint_storage_gate(storage,{'synthetic':True},actual_free_bytes=101_000_000_000)['deletion_performed_or_authorized'] is False)
    rejects('E100GB reserve blocks checkpoint growth',lambda:module.checkpoint_storage_gate(storage,{'synthetic':True},actual_free_bytes=100_100_000_000))
    bad=deepcopy(storage);bad['unbounded_per_window_checkpoint_retention']=True
    rejects('Blind full-run checkpoint accumulation prohibited',lambda:module.checkpoint_storage_gate(bad,{'synthetic':True},actual_free_bytes=1_000_000_000_000))
    bad=deepcopy(storage);bad['existing_or_original_artifact_deletion_authorized']=True
    rejects('No original checkpoint deletion authority',lambda:module.checkpoint_storage_gate(bad,{'synthetic':True},actual_free_bytes=1_000_000_000_000))
    bad=deepcopy(storage);bad['bounded_publication_and_retirement_implementation_reviewed']=False
    rejects('Unimplemented publication retirement remains production gate',lambda:module.checkpoint_storage_gate(bad,{'synthetic':True},actual_free_bytes=1_000_000_000_000))
    check('No model/runtime imports after controls',not any(k=='torch' or k.startswith(('torch.','numpy','scipy','transformers','pyarrow')) for k in sys.modules))
    check('Source unchanged through actual controls',source_before==module.record(module.__file__))
    result={'kind':'ASR18_full_run_lifecycle_synthetic_metadata_controls_v1','passed':True,'synthetic_only':True,
        'producer':identity(),'worker_source':module.record(__file__),'parent_source':module.record(__file__),
        'lifecycle_source':source_before,'checks':checks,'check_count':len(checks),'CPU_admission':forecast,
        'actual_native_updates':0,'actual_original_DATA_or_audio_or_frontend_or_GPU_work':False,
        'actual_fullpasses_or_quality_or_native_lifecycle_proved':False,'shared_CACHE_changed':False,'production_enabled':False}
    write('report.json',result); print(json.dumps({'passed':True,'checks':len(checks),'native_updates':0}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args();main(args.output)
