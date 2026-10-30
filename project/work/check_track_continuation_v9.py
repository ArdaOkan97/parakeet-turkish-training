"""Tiny stdlib-only V9 shard/mirror fixtures; no Trackio import or actual state."""
import argparse,ast,gc,hashlib,io,json,math,os,re,sys,tempfile,time
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'outputs/parakeet_turkish/track_continuation_v9.py'
checks=[]
def check(condition,label):
    if not condition:raise AssertionError(label)
    checks.append(label)
def reject(call,label):
    try:call()
    except (ValueError,KeyError):checks.append(label);return
    raise AssertionError(label)
def read_json(path,default=None):
    try:return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError,ValueError):return default
def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False),encoding='utf-8');os.replace(temp,path)
def extracted():
    tree=ast.parse(SOURCE.read_text(encoding='utf-8'))
    nodes=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) or isinstance(node,ast.Assign)]
    namespace=dict(globals(),__file__=str(SOURCE),OUTPUT=ROOT/'work',original=SimpleNamespace(metrics_for=lambda r:{},OUTPUT=ROOT/'work',ROOT=ROOT,trackio=None),trackio=None)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE),'exec'),namespace)
    namespace['self_memory']=lambda phase:{'phase':phase,'fixture':'no hardware sample'}
    return namespace

def main(args):
    n=extracted()
    old=ast.parse((SOURCE.parent/'track_continuation_v8.py').read_text(encoding='utf-8'))
    new=ast.parse(SOURCE.read_text(encoding='utf-8'))
    oldfunc={x.name:ast.dump(x,include_attributes=False) for x in old.body if isinstance(x,ast.FunctionDef)}
    newfunc={x.name:ast.dump(x,include_attributes=False) for x in new.body if isinstance(x,ast.FunctionDef)}
    unchanged=['finite_number','resume_context','values_for','acquisition_context','acquisition_values_for','continued_acquisition_context','continued_acquisition_values_for','scheduled_row_lr','scheduled_acquisition_context','scheduled_acquisition_values_for','adaptation_context','adaptation_values_for']
    check(all(oldfunc[key]==newfunc[key] for key in unchanged),'all12 old phase math/context functions AST identical')
    cache_source=ast.get_source_segment(SOURCE.read_text(encoding='utf-8'),next(node for node in new.body if isinstance(node,ast.ClassDef) and node.name=='StatJsonCache'))
    follow_source=ast.get_source_segment(SOURCE.read_text(encoding='utf-8'),next(node for node in new.body if isinstance(node,ast.FunctionDef) and node.name=='follow'))
    check(cache_source.index('del entry;gc.collect()')<cache_source.index('value=read_json(path,None)'), 'oldcached history references released before changed fullparse')
    check(follow_source.index('history=None;config=None')<follow_source.index('history=cache.read(history_path,[])'), 'follow oldhistory list released before cachedreader')
    parser=n['TopLevelRuns']
    fixture={'../odd/\"run':{'flag':{'unicode':'ş','quoted':'{["\\\\"','end':'\\'},'finished':False,'0':{'x':1.,'minus':-0.},'unknown':[None,True,3]},'closed':{'finished':'heldout-evaluation-complete','88':{'wer':7}}}
    raw=json.dumps(fixture,ensure_ascii=False)
    check(all(dict(parser(io.StringIO(raw),chunk_size=size).runs())==fixture for size in (1,2,3,7,64)),'stream JSON boundaries/escapes/Unicode/type-preserving')
    reject(lambda:dict(parser(io.StringIO(raw[:-2]),3).runs()),'truncated source rejected')
    reject(lambda:dict(parser(io.StringIO('{"x":{},"x":{}}'),3).runs()),'duplicate run rejected')
    reject(lambda:dict(parser(io.StringIO('{"x":[]}'),3).runs()),'non-object run rejected')
    reject(lambda:dict(parser(io.StringIO(raw+' trailing'),3).runs()),'trailing source rejected')
    with tempfile.TemporaryDirectory(prefix='v9_fixture_',dir=ROOT/'work') as tmp:
        tmp=Path(tmp);source=tmp/'old.json';write_json(source,fixture)
        index=tmp/'index.json';store=n['RunShardStore'](index)
        reject(lambda:store.migrate(source),'migration refused until sourceclosed flag')
        receipt=store.migrate(source,source_closed_confirmed=True)
        check(receipt['runs_count']==2 and receipt['source_state_unchanged'] and index.exists(),'complete migration checksum bound')
        check(read_json(source)==fixture,'source remains untouched')
        reconstructed={key:read_json(value['shard']['path']) for key,value in store.index['runs'].items()}
        check(reconstructed==fixture,'every unknown flag/value preserved in immutable shards')
        check(all(Path(value['shard']['path']).name==value['shard']['sha256']+'.json' for value in store.index['runs'].values()),'untrusted run ID never used in shard filename')
        active=store.activate('../odd/\"run')
        check(store.update_metrics(0,{'x':1.})=={},'sameID existing equal point has no duplicate log')
        changed=store.update_metrics(0,{'dev/wer_tr_percent':40.})
        store.accept_metrics(0,changed);check(store.flush(),'changedDEV amendment flushed')
        prior=index.read_bytes();check(not store.flush() and index.read_bytes()==prior,'unchanged flush makes no index write')
        active=store.activate('closed');check(store.is_finished('closed'),'finished flag survives restart/switch')
        check(store.active_id=='closed' and set(vars(store))=={'path','directory','active_id','active','dirty','index'},'onlyone run metric dictionary retained by store')
        store.activate('../odd/\"run');check(store.active['0']['dev/wer_tr_percent']==40.,'resumed run reloads amendment')
        store.accept_metrics(1,{'rate':2});store.flush()
        reopened=n['RunShardStore'](index);check(reopened.activate('../odd/\"run')['1']=={'rate':2},'dirty stop-boundary flush reloads every key')
        corrupt=tmp/'corrupt.json';write_json(corrupt,{'kind':'other'})
        reject(lambda:n['RunShardStore'](corrupt),'invalid index rejected rather than source reimport')
        bad=tmp/'truncated.json';bad.write_text(raw[:-3],encoding='utf-8');badindex=tmp/'badindex.json'
        reject(lambda:n['RunShardStore'](badindex).migrate(bad,source_closed_confirmed=True),'partial import refused')
        check(not badindex.exists(),'failed source cutoff publishes no index')
        mutable=tmp/'mutable.json';write_json(mutable,fixture);mutindex=tmp/'mutindex.json'
        mutstore=n['RunShardStore'](mutindex);actualwrite=mutstore.write_shard;mutated=[False]
        def mutate(value):
            result=actualwrite(value)
            if not mutated[0]:mutable.write_text(raw+' ',encoding='utf-8');mutated[0]=True
            return result
        mutstore.write_shard=mutate
        reject(lambda:mutstore.migrate(mutable,source_closed_confirmed=True),'changing source rejected at finalchecksum')
        check(not mutindex.exists(),'source change cannot publish partial index')
        cache=n['StatJsonCache']();history=tmp/'history.json';write_json(history,[{'step':1}]);one=cache.read(history,[]);two=cache.read(history,[])
        check(one is two,'mtime/size cache retains unchanged history without secondparse')
        write_json(history,[{'step':2}]);check(cache.read(history,[])[0]['step']==2,'changed history loads new publication')
        # Actual follow loop uses mocked Trackio; files are only disposable fixture paths.
        output=tmp/'output';output.mkdir();n['OUTPUT']=output
        write_json(output/'run_config.json',{'run_id':'fresh'})
        write_json(output/'training_history.json',[{'step':1}])
        write_json(output/'results/fresh_step_0/summary.json',{'validation_tr':{'asr':{'wer':.4}}})
        events=[]
        n['trackio']=SimpleNamespace(init=lambda **kwargs:events.append(('init',kwargs['name'])),finish=lambda:events.append(('finish',None)),log=lambda value,step:events.append(('log',step,dict(value))),Markdown=lambda text:text)
        n['cycle_rows']=lambda config,history,development:iter([(1,{'a':2,**development.get(1,{})})])
        a=SimpleNamespace(state_path=index,source_state_path=source,source_closed_confirmed=False,source_closure_receipt=None,project='fixture',follow=False,stop_file=None)
        check(n['follow'](a)=='one_cycle_complete','real follow onecycle dispatcher fixture')
        events.clear();n['follow'](a)
        check(not any(event[0]=='log' for event in events),'repeat sameID no unchanged metric logs')
        write_json(output/'results/fresh_step_0/summary.json',{'validation_tr':{'asr':{'wer':.39}}})
        events.clear();n['follow'](a)
        check([event for event in events if event[0]=='log']==[('log',0,{'dev/wer_tr_percent':39.})],'DEV amendment emits onlychangedkey atsame step')
        stop=tmp/'stop';stop.write_text('stop');a.stop_file=stop
        events.clear();check(n['follow'](a)=='cooperative_stop_file' and not events,'stop beforecycle prevents Trackio/initlog')
    # Actual shared producer identity shape, entirely synthetic numbers.
    n=extracted();phase=n['SHARED_HEAD_PHASE'];utility=n['SHARED_HEAD_UTILITY_SHA']
    p={'kind':phase,'source_phase1_step':7000,'original_source_selected_step':6500,'selected_acquisition_updates':500,'active_parameter_tensors':15,'active_shared_head_tensors':12,'frozen_encoder_LoRA_tensors':192,'row_lr':3e-4,'shared_lr':1e-5,'original12_shared_head_moments_active':True,'original3_resized_table_moments_active':False,'original192_encoder_LoRA_moments_active':False,'original207_Adam_artifact_retained_dormant':True,'further_unfreeze_allowed':False}
    o={'kind':n['SHARED_HEAD_IDENTITY'],'utility_sha256':utility,'phase':phase,'source_selected_step':6500,'selected_acquisition_updates':500,'active_parameter_tensors':15,'active_shared_head_tensors':12,'frozen_encoder_LoRA_tensors':192,'active_row_lr':3e-4,'source_row_lr':1e-4,'active_shared_lr':1e-5,'source_shared_lr':1e-5,'row_group':{'lr':3e-4,'weight_decay':0},'shared_group':{'lr':1e-5,'weight_decay':.01},'source_moments_or_clocks_reset':False,'original12_shared_moments_active':True,'original3_resized_table_moments_active':False,'original192_encoder_LoRA_moments_active':False,'effective_new_rows':[8199,8711],'protected_blank':8192,'protected_six_rows':[8193,8199],'protected_duration_rows':[8711,8716],'mask_before_single_global_clip':True,'clip_max_norm':1.,'encoder_projector_must_be_outside_encoder_no_grad':True,'further_unfreeze_allowed':False}
    config={'phase':phase,'source_phase1_step':7000,'original_source_selected_step':6500,'selected_acquisition_updates':500,'inherited_history_steps':7000,'config_identity':{'phase':phase,'adaptation_phase':p,'adaptation_optimizers':o,'training_arguments':{'lora_lr':3e-4,'head_lr':1e-5}}}
    baseline={'step':7000,'acquisition_updates_completed':500,'audio_seen_seconds':1000.,'train_wall_seconds':100.,'turkish_audio_seen_seconds':600.,'replay_audio_seen_seconds':400.}
    c=n['shared_head_context'](config,[baseline]);r={'step':7250,'adaptation_updates_completed':250,'source_phase1_step':7000,'original_source_selected_step':6500,'selected_acquisition_updates':500,'row_Adam_clock':750,'shared_Adam_clock':6750,'row_lr':3e-4,'shared_lr':1e-5,'head_lr':1e-5,'lora_lr':0.,'new_token_update_multiplier':1,'old_six_character_update_multiplier_applied':False,'audio_seen_seconds':1900.,'train_wall_seconds':110.,'turkish_audio_seen_seconds':950.,'replay_audio_seen_seconds':950.}
    values=n['shared_head_values_for'](r,c)
    check(values['resources/post_shared_head_audio_minutes_per_wall_minute']==90. and values['phase/shared_Adam_clock']==6750,'sharedhead phase throughput/15active/two clocks')
    reject(lambda:n['shared_head_values_for']({**r,'shared_Adam_clock':6751},c),'wrongshared Adamclock rejected')
    reject(lambda:n['shared_head_values_for']({**r,'audio_seen_seconds':float('nan')},c),'nonfinite throughput rejected')
    reject(lambda:n['shared_head_context']({**config,'config_identity':{**config['config_identity'],'adaptation_optimizers':{**o,'utility_sha256':'bad'}}},[baseline]),'wrong sharedutility identity rejected')
    report={'schema':1,'kind':'Trackio_V9_tiny_stdlib_shard_and_shared_phase_verification_v1','passed':True,'checks':checks,'checks_count':len(checks),'source':{'path':str(SOURCE),'sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest()},'fixture':{'path':str(Path(__file__).resolve()),'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},'scope':{'actual_V8_state_read':False,'actual_migration':False,'Trackio_import_or_DB_writes':False,'GPU_or_ML_import':False,'actual_hardware_memory_measured':False,'external_source_closure_observed':False,'all_run_state_and_trackio_calls_are_isolated_fixtures':True}}
    write_json(args.output,report);print(json.dumps({'passed':True,'checks':len(checks),'report':str(args.output),'sha256':hashlib.sha256(args.output.read_bytes()).hexdigest()}))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);main(parser.parse_args())
