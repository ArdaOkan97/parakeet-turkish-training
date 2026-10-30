"""ONE supported checkpoint pause after genuine204000 eval. CPU metadata only."""
import argparse,importlib.util,json,os,shutil,sys,time
from pathlib import Path
import psutil
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20')
B=Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01')
I=B/'invocations/owner_000026'
W=Path(__file__).parent
TARGET=204800
def require(v,m):
    if not v:raise ValueError(m)
def resource_admission(value):
    m=value['system_memory']
    require(value['paired_sampling_within_one_second'] is True
        and value['worker_private_memory']['private_bytes']<536870912
        and m['commit_available_bytes']>=2268435456
        and m['physical_available_bytes']>=1768435456,
        'Unchanged failfast runtime/paired/self guard, never deferred')
    return m['commit_available_bytes']>=5055306368 and m['physical_available_bytes']>=2305306368
def load_original(pin):
    import hashlib
    raw=Path(pin['path']).read_bytes()
    require(len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['sha256'],'Exact tested original actor source')
    spec=importlib.util.spec_from_file_location('tested_original204_actor',pin['path'])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
def check_eval(report,entry):
    langs={'tr','en','es','it','fr','de','pt'}
    require(report['kind']=='ASR18_actual_resident_optimizer_idle_serial350_DEV_v1'
        and report['passed'] is True and report['optimizer_updates']==204000
        and report['checkpoint_pointer']==entry['checkpoint_pointer']
        and report['actual_native_updates_in_DEV']==0
        and report['all207_threeAdam_all4RNG_unchanged'] is True
        and report['TEST_used'] is False and report['retention_early_stop'] is False
        and set(report['predictions'])==langs
        and set(report['scores'])=={'validation_'+x for x in langs}|{'calibration_'+x for x in langs}
        and all(report['scores']['validation_'+x]['asr']['utterances']==50 for x in langs)
        and report['official_saved_text_score']['full350_DEV_complete'] is True
        and entry['score']==report['official_saved_text_score']
        and entry['shared_with_full_pass_DEV'] is False
        and entry['periodic_record_advances_full_pass_or_winner'] is False,
        'Genuine original seven ASR50 plus original seven confidence calibration records')
def check_checkpoint(latest,persistence,expected_source,pointer,persist_pin):
    require(latest['committed_updates']==TARGET and latest['full_state_verified'] is True
        and latest['source']==expected_source and latest['pointer']==pointer and latest['persistence']==persist_pin,
        'Exact authoritative full204800 checkpoint')
    require(persistence['passed'] is True and persistence['optimizer_updates']==TARGET
        and persistence['named_tensors_verified']==854 and persistence['weights_verified']==207
        and persistence['Adam_states_verified']==215
        and persistence['independent_pre_save_state_sha256']==persistence['saved_state_sha256']
        and persistence['saved_semantic_state_verified']['passed'] is True,
        'Original full854/207/215 semantic independent PRE-SAVED persistence')
def frontier_decision(progress):
    u=progress['actual_optimizer_updates']
    require(type(u)is int,'Integer actual frontier')
    return 'WAIT' if u<TARGET else('READY' if u<TARGET+200 else 'MISSED')
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--ROOT-review',required=True);args=parser.parse_args()
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    scope=json.loads(Path(args.ROOT_review).read_bytes())
    actor=load_original(scope['original_actor_source'])
    require(scope['kind']=='ROOT_source37_postDEV204000_saved204800_checkpoint_pause_exact_ONCE_v1'
        and scope['accepted'] is True and scope['reviewer']=='/root'
        and scope['source']==actor.pin(__file__),'Exact NEW checkpoint policy; never re-run failed pre-ready watcher')
    out=Path(scope['output']);require(out.drive.lower()=='e:' and not out.exists(),'Fresh E namespace');out.mkdir()
    start=time.monotonic()
    report={'kind':'ASR18_actual_source37_postDEV204000_saved204800_checkpoint_pause_v1',
        'observer':actor.identity(),'launcher':actor.identity(os.getppid()),'source':actor.pin(__file__),
        'stop_published':False,'passed':False,'no_normal0_or_no_tail_claim':True}
    sys.path.insert(0,str(P/'outputs/parakeet_turkish'))
    from resource_watchdog_v2 import sample_memory
    def sample(label,prospective=False,defer=False):
        value=sample_memory(os.getpid(),label)
        if prospective:actor.write_once(out/(label+'.json'),value)
        full=resource_admission(value)
        require(shutil.disk_usage('E:/').free>=100000000000 and time.monotonic()-start<1200,
            'Unchanged E/wall guard, never deferred')
        if prospective and not defer:require(full,'Fresh initial full admission is fatal if unavailable')
        return (value,full)if defer else value
    def current():
        require(not(I/'stop.json').exists() and not(I/'actual_failure.json').exists(),'No foreign marker/current failure')
        for who in scope['protected_live_identities']:require(actor.exact_alive(who),'Exact current production/reader identity')
        p=actor.read(B/'progress/latest.json');e=scope['expected_current']
        require(p['producer']==e['native'] and p['source']==e['source']
            and p['planned_native_updates']==401072 and p['administrative_update_stop'] is None
            and p['optimizer_poisoned'] is False and p['schedule_poisoned'] is False,'Unchanged healthy full401072 source37')
        d=frontier_decision(p)
        if d!='READY':return {'decision':d,'actual':p['actual_optimizer_updates']}
        latest=actor.read(B/'latest_checkpoint.json')
        if latest['committed_updates']<TARGET:return {'decision':'WAIT','actual':p['actual_optimizer_updates']}
        if latest['committed_updates']>TARGET:return {'decision':'MISSED','actual':p['actual_optimizer_updates']}
        ck=B/'checkpoints/update_000204800'
        pointer=actor.pin(ck/'pointer.json');persist=actor.pin(ck/'persistence_verified.json')
        check_checkpoint(latest,actor.read(persist['path']),e['source'],pointer,persist)
        reader=actor.read(Path(scope['reader_status_path']))
        require(reader['native_producer']==e['native'] and reader['training_worker_alive'] is True and reader['error'] is None,'Same healthy reader')
        capture=actor.capture_native_descendants(e['native'])
        chain=actor.match_current_starting_CPU_chain(capture,e)
        require(chain is not None and chain['native_body']==scope['current_CPU']
            and chain['launcher']==scope['current_CPU_launcher'],'Exact currently ready204k CPU chain, no automatic rebind')
        return {'decision':'READY','actual':p['actual_optimizer_updates'],'saved':TARGET,'latest':latest,
            'current_native_descendants':capture,'actual_CPU_chain':chain,'DEV_entry':scope['DEV_entry'],
            'ready_absent':False,'resumed_normal_training_before_pause':True}
    try:
        report['initial_sample']=sample('initial_full_prospective',True)
        require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
            and all(os.environ.get(k)=='1'for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'))
            and os.environ.get('TORCH_EXTENSIONS_DIR','').lower().startswith('e:'),'Exact CPU/E environment')
        for p in scope['input_pins']:require(actor.pin(p['path'])==p,'All immutable scoped inputs')
        check_eval(actor.read(scope['DEV_entry']['DEV']['path']),scope['DEV_entry'])
        require(not(B/'stop.json').exists(),'No unrelated root marker')
        while True:
            report['last_sample']=sample('runtime')
            stage=current();report['last_stage']=stage
            if stage['decision']=='MISSED':
                report['passed']=True;report['no_write_reason']='Outside separately authorized204800..204999 checkpoint band';break
            if stage['decision']=='READY':
                attempt=report.get('full_admission_observations',0)
                require(attempt<256,'Bounded final admission waiting exhausted; no publication')
                candidate,admitted=sample('final_full_prospective_'+str(attempt).zfill(3),True,True)
                report['full_admission_observations']=attempt+1
                report['last_full_admission_candidate']=candidate
                if not admitted:
                    report['full_admission_deferred_count']=report.get('full_admission_deferred_count',0)+1
                    time.sleep(.5);continue
                report['final_sample']=candidate
                evidence=actor.write_once(out/'prestop_complete_stage_evidence.json',stage)
                report['prestop_evidence']=evidence
                def recheck():
                    for p in scope['input_pins']:require(actor.pin(p['path'])==p,'All inputs exact immediately before publication')
                    v=current();require(v['decision']=='READY','Exact checkpoint band still eligible');return v
                payload={'kind':'ASR18_human_authorized_postDEV204000_saved204800_cooperative_checkpoint_stop_v1',
                    'reason':'Direct human throughput repair at next coherent checkpoint following original204000 eval',
                    'ROOT_review':actor.pin(args.ROOT_review),'observer':report['observer'],'evidence':evidence,
                    'saved_optimizer_updates':TARGET,'optimizer_updates_observed':stage['actual'],
                    'normal_exit_or_no_tail_race_claimed':False}
                report['publication']=actor.publish_marker(I/'stop.json',out/'owned_stop_request.tmp',payload,before_publish=recheck)
                report['stop_published']=True;report['passed']=True;break
            time.sleep(.25)
    except BaseException as exc:
        report['error']={'type':type(exc).__name__,'message':str(exc)};raise
    finally:
        report['wall_seconds']=time.monotonic()-start
        result=actor.write_once(out/'result.json',report)
        print(json.dumps({'result':result,'passed':report['passed'],'stop_published':report['stop_published']}),flush=True)
if __name__=='__main__':main()

