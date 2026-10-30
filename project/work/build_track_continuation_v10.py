from pathlib import Path
import ast
source=Path('outputs/parakeet_turkish/track_continuation_v9.py').read_text(encoding='utf-8')
def replace_function(text,name,new):
    node=next(n for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
    lines=text.splitlines(keepends=True)
    return ''.join(lines[:node.lineno-1])+new+'\n'+''.join(lines[node.end_lineno:])
# Refactor exactly the existing metric body; no phase math/context changes.
tree=ast.parse(source);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='cycle_rows')
loop=next(n for n in node.body if isinstance(n,ast.For))
body=loop.body[:-1]+[ast.Return(value=loop.body[-1].value.value.elts[1])]
parameters=['record','contexts','development']
loads=[ast.Assign(targets=[ast.Name(id=name,ctx=ast.Store())],value=ast.Subscript(value=ast.Name(id='contexts',ctx=ast.Load()),slice=ast.Constant(name),ctx=ast.Load())) for name in ['context','phase_context','adaptation','continued','scheduled','shared']]
newnode=ast.FunctionDef(name='value_for_record',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=k) for k in parameters],kwonlyargs=[],kw_defaults=[],defaults=[]),body=loads+body,decorator_list=[])
ast.fix_missing_locations(newnode)
contexts="""def contexts_for(config,baseline_rows):
    return {'context':resume_context(config,baseline_rows),
        'phase_context':acquisition_context(config,baseline_rows),
        'adaptation':adaptation_context(config,baseline_rows),
        'continued':continued_acquisition_context(config,baseline_rows),
        'scheduled':scheduled_acquisition_context(config,baseline_rows),
        'shared':shared_head_context(config,baseline_rows)}


def cycle_rows(config,history,development):
    contexts=contexts_for(config,history)
    for record in history:
        yield record['step'],value_for_record(record,contexts,development)


"""+ast.unparse(newnode)
source=replace_function(source,'cycle_rows',contexts)
addition=r"""
SOURCE_V9_SHA='076ab5f58fe6cbcd2b061af6346d98fa58aaa500700b853a8d35b356fe44a674'


def stat_token(path):
    value=Path(path).stat()
    return (value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns)


class HistoryArray(TopLevelRuns):
    # One complete object record at a time; inherited nested/escape parser.
    def records(self):
        self.whitespace()
        if self.take()!='[':raise ValueError('Worker history must be a top-level array')
        self.whitespace();count=0
        if self.peek()==']':self.take()
        else:
            while True:
                record=self.value();count+=1
                if type(record.get('step')) is not int or record['step']!=count:
                    raise ValueError('Canonical worker history must have contiguous step1..N')
                yield record
                self.whitespace();separator=self.take()
                if separator==']':break
                if separator!=',':raise ValueError('Truncated/invalid history separator')
                self.whitespace()
                if self.peek()==']':raise ValueError('Trailing history comma')
        self.whitespace()
        if self.peek():raise ValueError('Trailing history content')


def baseline_steps(config):
    result=set()
    warm=config.get('warm_start',{})
    if warm.get('mode')=='exact_resume' and type(warm.get('source_step')) is int:
        result.add(warm['source_step'])
    if config.get('phase') in (ADAPTATION_PHASE,SHARED_HEAD_PHASE):result.add(config['source_phase1_step'])
    elif config.get('phase') in (ACQUISITION_PHASE,CONTINUED_ACQUISITION_PHASE,SCHEDULED_ACQUISITION_PHASE):
        result.add(config['source_selected_step'])
    return result


def initial_development_step(config):
    if config.get('phase') in (ADAPTATION_PHASE,SHARED_HEAD_PHASE):return config['source_phase1_step']
    if config.get('phase') in (ACQUISITION_PHASE,CONTINUED_ACQUISITION_PHASE,SCHEDULED_ACQUISITION_PHASE):return config['source_selected_step']
    return 0


def actual_development_row(config,record):
    if 'validation' not in record:return False
    phase=config.get('phase')
    if phase in (ADAPTATION_PHASE,SHARED_HEAD_PHASE):
        return 'adaptation_updates_completed' in record and record['step']>config['source_phase1_step']
    if phase in (CONTINUED_ACQUISITION_PHASE,SCHEDULED_ACQUISITION_PHASE):
        return 'acquisition_updates_completed' in record and record['step']>config['source_selected_step']
    if phase==ACQUISITION_PHASE:return 'acquisition_updates_completed' in record
    return True


def development_from_steps(run_id,steps):
    result={}
    for step in steps:
        summary=read_json(OUTPUT/f'results/{run_id}_step_{step}/summary.json',{})
        if summary:
            result[step]={f'dev/wer_{key.removeprefix("validation_")}_percent':100*value['asr']['wer']
                for key,value in summary.items() if key.startswith('validation_')}
    return result


class HistoryScanCache:
    # Bounded retained data: phase baselines, contexts, DEV step integers only.
    def __init__(self):self.key=None;self.info=None
    def scan(self,path,config):
        path=Path(path)
        if not path.exists():self.key=None;self.info=None;return None,False
        token=stat_token(path)
        config_sha=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
        key=(str(path.resolve()),token,config_sha)
        if self.key==key:return self.info,False
        wanted=baseline_steps(config);baselines=[];steps={initial_development_step(config)};last_step=0
        with path.open('r',encoding='utf-8') as stream:
            fd=os.fstat(stream.fileno())
            if (fd.st_dev,fd.st_ino,fd.st_size,fd.st_mtime_ns)!=token:raise PublishedJsonChanged('History replaced before scan')
            for record in HistoryArray(stream).records():
                last_step=record['step']
                if last_step in wanted:baselines.append(record)
                if actual_development_row(config,record):steps.add(last_step)
            after=os.fstat(stream.fileno())
            if (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)!=token or stat_token(path)!=token:
                raise PublishedJsonChanged('History publication changed during baseline pass')
        info={'token':token,'contexts':contexts_for(config,baselines),'baseline_rows':baselines,
            'development_steps':steps,'last_step':last_step,'records_count':last_step,
            'retained_history_record_count':len(baselines),'source_bytes':token[2]}
        self.key=key;self.info=info
        return info,True
    def rows(self,path,info):
        path=Path(path);token=info['token']
        if stat_token(path)!=token:raise PublishedJsonChanged('History replaced before metric pass')
        with path.open('r',encoding='utf-8') as stream:
            before=os.fstat(stream.fileno())
            if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)!=token:
                raise PublishedJsonChanged('History open differs from baseline snapshot')
            for record in HistoryArray(stream).records():yield record
            after=os.fstat(stream.fileno())
            if (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)!=token or stat_token(path)!=token:
                # Prior logged rows are consistent old-snapshot values, never a
                # mixed file; flush dedup state and retry the new atomic publication.
                raise PublishedJsonChanged('History advanced during metric pass; old snapshot rows remain valid')


class StreamRunShardStore(RunShardStore):
    # V9 source archive stays immutable; V10 deletes only own superseded shards.
    def __init__(self,index_path):
        self.path=Path(index_path).resolve();self.directory=self.path.parent/(self.path.stem+'_shards')
        self.active_id=None;self.active=None;self.dirty=False
        self.index=read_json(self.path,None) if self.path.exists() else None
        if self.path.exists() and (type(self.index) is not dict or self.index.get('schema')!=1
            or self.index.get('kind')!='Trackio_run_shard_index_v10'
            or self.index.get('source_import',{}).get('complete') is not True
            or type(self.index.get('runs')) is not dict or type(self.index.get('live_versions')) is not dict
            or type(self.index.get('obsolete_live_shards')) is not list):
            raise ValueError('Invalid/incomplete V10 index; no silent reimport')
        if self.index is not None:
            for entry in self.index['runs'].values():self.shard_path(entry['shard'])
            self.prune_obsolete()
    def shard_path(self,record):
        sha=record.get('sha256')
        if type(sha) is not str or re.fullmatch('[0-9a-f]{64}',sha) is None or type(record.get('bytes')) is not int or record['bytes']<2:
            raise ValueError('Invalid shard SHA/size')
        path=Path(record.get('path','')).resolve()
        archive=Path(self.index['source_import']['source_archive_directory']).resolve() if self.index is not None else None
        allowed=[self.directory.resolve()]+([archive] if archive else [])
        if path.parent not in allowed or path.name!=sha+'.json':raise ValueError('Shard is outside its immutable content directories')
        return path
    def migrate(self,source_path,*,source_closed_confirmed=False,source_closure_receipt=None):
        if self.index is not None:return None
        if not source_closed_confirmed:raise ValueError('ROOT must observe V9 CLOSED before V10 handoff')
        source_path=Path(source_path).resolve()
        if source_path==self.path:raise ValueError('V9 source/V10 index must be distinct')
        producer=Path(__file__).with_name('track_continuation_v9.py')
        if file_record(producer)['sha256']!=SOURCE_V9_SHA:raise ValueError('Frozen V9 producer changed')
        before=file_record(source_path);old=read_json(source_path,None)
        if (type(old) is not dict or old.get('kind')!='Trackio_run_shard_index_v9'
            or old.get('schema')!=1 or old.get('source_import',{}).get('complete') is not True
            or type(old.get('runs')) is not dict):raise ValueError('Complete pinned V9 index required')
        if file_record(source_path)!=before:raise ValueError('V9 index changed during handoff')
        archive=source_path.parent/(source_path.stem+'_shards')
        imported={'complete':True,'source_index':before,'source_v9_script':file_record(producer),
            'source_archive_directory':str(archive.resolve()),'source_V9_index_payload':old,
            'source_closed_confirmed_by_ROOT':True,'source_closure_receipt':file_record(source_closure_receipt) if source_closure_receipt else None,
            'source_all_shards_retained_without_deletion':True,'source_shards_rehashed_at_handoff':False,
            'active_shard_hash_verified_on_load':True,'completed_at_utc':datetime.now(timezone.utc).isoformat()}
        index={'schema':1,'kind':'Trackio_run_shard_index_v10','source_import':imported,
            'runs':dict(old['runs']),'live_versions':{},'obsolete_live_shards':[],
            'retention_policy':{'latest_live_versions_per_run':2,'source_V8_V9_archive_pruned':False,
                'delete_obsolete_own_shards_only_after_atomic_index_commit':True}}
        self.index=index
        try:
            for entry in index['runs'].values():self.shard_path(entry['shard'])
            write_json(self.path,index)
        except BaseException:self.index=None;raise
        return imported
    def prune_obsolete(self):
        if self.index is None or not self.index['obsolete_live_shards']:return False
        protected={entry['shard']['sha256'] for entry in self.index['runs'].values()}
        protected.update(record['sha256'] for versions in self.index['live_versions'].values() for record in versions)
        pending=list(self.index['obsolete_live_shards'])
        for record in pending:
            path=self.shard_path(record)
            if path.parent!=self.directory.resolve():raise ValueError('Never prune source archive shard')
            if record['sha256'] in protected:continue
            if path.exists():
                actual=file_record(path)
                if actual['sha256']!=record['sha256'] or actual['bytes']!=record['bytes']:
                    raise ValueError('Obsolete immutable shard checksum differs; refuse deletion')
                path.unlink()
        self.index['obsolete_live_shards']=[];write_json(self.path,self.index)
        return True
    def flush(self):
        if not self.dirty:self.prune_obsolete();return False
        record=self.write_shard(self.active)
        previous=self.index['live_versions'].get(self.active_id,[])
        versions=[record]+[prior for prior in previous if prior['sha256']!=record['sha256']]
        retained=versions[:2];obsolete=versions[2:]
        self.index['runs'][self.active_id]={'shard':record,'finished_truthy':bool(self.active.get('finished'))}
        self.index['live_versions'][self.active_id]=retained
        self.index['obsolete_live_shards'].extend(obsolete)
        write_json(self.path,self.index);self.dirty=False
        self.prune_obsolete()
        return True


def publish_cycle_memory(args,control,run_id,scan_info,history_streamed,completed=True):
    value={'schema':1,'kind':'owned_Trackio_v10_cycle_memory_v1','run_id':run_id,'pid':os.getpid(),'ppid':os.getppid(),
        'observed_at_utc':datetime.now(timezone.utc).isoformat(),'completed_metric_cycle':completed,
        'history_metric_pass_streamed':history_streamed,'history_bytes':scan_info['source_bytes'] if scan_info else None,
        'history_records':scan_info['records_count'] if scan_info else None,
        'retained_history_records':scan_info['retained_history_record_count'] if scan_info else 0,
        'memory':self_memory('v10_after_metric_cycle'),
        'scope':'Own mirror PID only; no other-process attribution or GPU measurements'}
    if args.telemetry_path is not None:write_json(args.telemetry_path,value)
    if control is not None and control['path'] is not None:
        control['identity']['cycle_telemetry']=value
        control['identity']['updated_at_utc']=value['observed_at_utc']
        write_json(control['path'],control['identity'])
"""
source=source.replace('\ndef stop_requested(args):',addition+'\n\ndef stop_requested(args):')
follow=r"""def follow(args,control=None):
    store=StreamRunShardStore(args.state_path)
    store.migrate(args.source_state_path,source_closed_confirmed=args.source_closed_confirmed,
        source_closure_receipt=args.source_closure_receipt)
    active=None;config_cache=StatJsonCache();history_cache=HistoryScanCache();processed_key=None
    try:
        while True:
            if stop_requested(args):return 'cooperative_stop_file'
            config_path=OUTPUT/'run_config.json';history_path=OUTPUT/'training_history.json'
            try:
                config=config_cache.read(config_path,{})
                info,history_changed=history_cache.scan(history_path,config)
            except PublishedJsonChanged:
                if not args.follow:return 'publication_changed_no_metrics_logged'
                time.sleep(5);continue
            run_id=config.get('run_id')
            coherent=bool(run_id and info and info['records_count'] and config_path.exists() and history_path.exists()
                and history_path.stat().st_mtime>=config_path.stat().st_mtime)
            streamed=False
            if coherent and not store.is_finished(run_id):
                development=development_from_steps(run_id,info['development_steps'])
                switching=active!=run_id
                if switching:
                    if active is not None:trackio.finish()
                    state=store.activate(run_id)
                    trackio.init(project=args.project,name=run_id,group='Turkish adaptation',config=config,resume='allow',gpu_log_interval=10.)
                    active=run_id;publish_control_run(control,run_id)
                    if 'reference_base_v3' not in state:
                        baseline=read_json(OUTPUT/'results/base_v3/summary.json',{})
                        trackio.log({f'reference/base_v3_wer_{key.removeprefix("validation_")}_percent':100*value['asr']['wer']
                            for key,value in baseline.items() if key.startswith('validation_')},step=0)
                        store.set_flag('reference_base_v3',True)
                    print(f'Mirroring {run_id}; streamed history and active-run immutable shards',flush=True)
                initial_step=initial_development_step(config)
                initial=store.update_metrics(initial_step,development.get(initial_step,{}))
                if initial:trackio.log(initial,step=initial_step);store.accept_metrics(initial_step,initial)
                cycle_key=(run_id,history_cache.key)
                try:
                    if processed_key!=cycle_key or switching:
                        streamed=True
                        for record in history_cache.rows(history_path,info):
                            step=record['step'];values=value_for_record(record,info['contexts'],development)
                            changed=store.update_metrics(step,values)
                            if changed:trackio.log(changed,step=step);store.accept_metrics(step,changed)
                        processed_key=cycle_key
                    else:
                        # No history reparse: DEV amendments still log each changed
                        # key; a source-baseline row restores inherited retention
                        # fields exactly when its real initial summary appears.
                        for record in info['baseline_rows']:
                            step=record['step'];values=value_for_record(record,info['contexts'],development)
                            changed=store.update_metrics(step,values)
                            if changed:trackio.log(changed,step=step);store.accept_metrics(step,changed)
                        for step,values in development.items():
                            changed=store.update_metrics(step,values)
                            if changed:trackio.log(changed,step=step);store.accept_metrics(step,changed)
                except PublishedJsonChanged:
                    store.flush();publish_cycle_memory(args,control,run_id,info,streamed,completed=False)
                    if not args.follow:return 'publication_advanced_valid_old_snapshot_rows_logged'
                    time.sleep(5);continue
                store.flush()
                selected=read_json(OUTPUT/'selection.json',{});pointer=read_json(OUTPUT/'adapted_results_path.json',{})
                if pointer.get('tag')==run_id+'_adapted':
                    final=read_json(OUTPUT/f'results/{run_id}_adapted/summary.json',{})
                    if all('test_'+lang in final for lang in ['en','es','it','fr','de','tr']):
                        values={f'test/wer_{key.removeprefix("test_")}_percent':100*value['asr']['wer'] for key,value in final.items() if key.startswith('test_')}
                        values['selection/best_step']=selected.get('selected_step',0)
                        trackio.log(values,step=info['last_step']);trackio.finish()
                        store.set_flag('finished','heldout-evaluation-complete');store.flush();active=None
                elif (selected.get('run_id')==run_id and selected.get('accepted') is False
                    and isinstance(selected.get('completed_steps'),int) and info['last_step']>=selected['completed_steps']):
                    trackio.log({'selection/accepted':0,'selection/report':trackio.Markdown(selected.get('reason','Rejected on development gates.'))},step=info['last_step'])
                    trackio.finish();store.set_flag('finished','rejected-on-development');store.flush();active=None
            publish_cycle_memory(args,control,run_id,info,streamed)
            if not args.follow:
                if active is not None:trackio.finish()
                return 'one_cycle_complete'
            if stop_requested(args):return 'cooperative_stop_file'
            time.sleep(5)
    finally:store.flush()
"""
source=replace_function(source,'follow',follow)
source=source.replace('INACTIVE Trackio mirror v9: shared-head metrics and bounded resident run shards.', 'INACTIVE Trackio mirror v10: streamed history and bounded live shard retention.')
source=source.replace('V8 is preserved. ROOT must confirm V8 CLOSED before streaming first migration.','V8/V9 archives are preserved. ROOT must confirm V9 CLOSED before small-index handoff.')
source=source.replace('Growing\nworker history still requires one full parse when its stat changes.','Worker history is read as one record at a time in two passes when changed;\nunchanged history reuses bounded contexts and checks small DEV amendments.')
source=source.replace("'owned_trackio_continuation_mirror_v9'","'owned_trackio_continuation_mirror_v10'")
source=source.replace("default=ROOT/'work/trackio_continuation_mirror_state_v9.json'", "default=ROOT/'work/trackio_continuation_mirror_state_v10.json'")
source=source.replace("default=ROOT/'work/trackio_continuation_mirror_state_v8.json'", "default=ROOT/'work/trackio_continuation_mirror_state_v9.json'")
source=source.replace('ROOT observed source V8 process CLOSED; required for first streaming import','ROOT observed source V9 process CLOSED; required for first small-index handoff')
source=source.replace("    p.add_argument('--stop-file'", "    p.add_argument('--telemetry-path',type=Path,help='Fresh workspace path for actual own-PID memory after each complete metric cycle')\n    p.add_argument('--stop-file'")
source=source.replace('New v9 index must be separate','New v10 index must be separate')
source=source.replace("    paths.extend(path for path in (stop,identity) if path is not None)", "    telemetry=scoped_control_path(getattr(args,'telemetry_path',None));args.telemetry_path=telemetry\n    paths.extend(path for path in (stop,identity,telemetry) if path is not None)\n    if telemetry is not None and telemetry.exists():raise ValueError('Own memory telemetry must use a fresh path')")
Path('outputs/parakeet_turkish/track_continuation_v10.py').write_text(source,encoding='utf-8',newline='\n')
ast.parse(source)
print('V10 AST parsed; no state/history import or runtime execution')
