"""INACTIVE matched TR-only loss study with journal publication.

Scientific state and the declared loss are owned by its new pinned worker.
Only transport/observation change here; full snapshots remain at DEV/closure.
One fresh-parent exact budget window; source350 is not repeated on exact.
No automatic resource retry, pagefile changes or guard relaxation.
"""
from pathlib import Path
import argparse,functools,hashlib,inspect,json,os,subprocess,sys,time
import run_TSC_uniform_TR_loss_v1 as c
EXPECTED_CORE='e24a52574dfca48eb20efb25904df57ac14c30efebc767c8dda6eef84545bfb8'
EXPECTED_WORKER='8df7018a8ac59ee8c95003294983ca913c4d1522ca738d680c256dc938d02484'
JOURNAL_SHA='e3822b4b6f37c9e389b5f89200475bad4c928d4bef9c3b0336c2980ec9594980'
KIND='TSC_uniform_TR_loss_journal_persistence_runtime_v1'
HERE=c.HERE;ROOT=c.ROOT
for _name in ('scoped','read','write','now','record','require','ResourceGuard','domain','sample_memory','request_stop','verify_completion'):
    globals()[_name]=getattr(c,_name)

def byte_record(path):
    p=scoped(path);return {**record(p),'bytes':p.stat().st_size}

def validate_runtime(plan):
    c.validate_plan(plan)
    require(c.sha(c.__file__)==EXPECTED_CORE and c.sha(HERE/c.WORKER)==EXPECTED_WORKER,'Frozen scientific body changed')
    m=plan['history_runtime_migration'];require(m['kind']==KIND and m['adapter']==byte_record(__file__)
        and m['journal_helper']==byte_record(HERE/'append_only_history_v1.py')
        and m['journal_helper']['sha256']==JOURNAL_SHA and JOURNAL_SHA!='PENDING','Frozen journal/executor binding differs')
    require(scoped(m['journal_directory'])==HERE/'runs'/plan['run_id']/'history_journal_v1'
        and m['activation_source_step']==plan['resume_step']
        and m['scientific_identity_changed'] is False and m['pagefile_or_guard_changes'] is False,'Scoped operational migration only')
    expected=[str(ROOT/'work/nemo-native-env-v2/Scripts/python.exe'),str(Path(__file__).resolve()),'--worker-plan',str(Path(plan['paths']['log']).parent/'plan.json'),'--gpu-window-authorized']
    require(plan['command']==expected,'Actual journal child command differs')
    return m

def publish_queue(state):
    c.publish_queue(state)
    value=read(HERE/'experiment_queue_state.json');value['history_runtime_migration']=state['history_runtime_migration']
    value['effective_history_transport']='append_only_journal_with_DEV_and_closure_snapshots'
    write(HERE/'experiment_queue_state.json',value)

def validate_identity(value,plan,child_pid,started):
    result=c.validate_identity(value,plan,child_pid,started)
    require(result.get('history_runtime_migration')==plan['history_runtime_migration'],'Actual child journal provenance missing')
    return result

class WindowObserver:
    """Parent reads only its committed window suffix; prefix remains on disk."""
    def __init__(self,directory,source_step):
        self.directory=scoped(directory);self.source_step=source_step;self.journal=None
    def rows(self):
        if not (self.directory/'HEAD.json').exists():return [{'step':self.source_step}]
        from append_only_history_v1 import HistoryJournal
        if self.journal is None:self.journal=HistoryJournal(self.directory)
        self.journal.refresh();frontier=self.journal.max_step
        require(frontier>=self.source_step,'Journal cannot rewind a live observer')
        anchor=self.journal.get_step(self.source_step)
        rows=[anchor,*self.journal.iter_latest(self.source_step+1,refresh=False)]
        require(len(rows)==frontier-self.source_step+1 and rows[-1]['step']==frontier,'Committed journal suffix is incomplete')
        return rows

class HistoryRouter:
    def __init__(self,plan,original):
        self.plan=plan;self.migration=validate_runtime(plan);self.original=original;self.journal=None
        self.run=HERE/'runs'/plan['run_id'];self.targets={self.run/'training_history.json',HERE/'training_history.json'}
        self.source_checkpoint=byte_record(plan['source_checkpoint']['path']);self.initialized=False
        runtime=Path(plan['paths']['fresh_runtime'] if plan.get('continuation_mode')=='same_new_run_exact_budget_continuation' else plan['paths']['native']).resolve()
        self.provenance_targets={runtime,runtime.with_suffix('.completion.json'),*[Path(plan['paths'][k]).resolve() for k in ('identity','memory','cache','io')]}
    def label(self):
        frame=inspect.currentframe().f_back
        try:
            while frame:
                if frame.f_code.co_name=='publish_history' and Path(frame.f_code.co_filename).resolve()==(HERE/c.WORKER).resolve():
                    label=frame.f_locals.get('label');require(label in ('initial','periodic','DEV','final'),'Unknown frozen publication boundary');return label
                frame=frame.f_back
        finally:del frame
        raise ValueError('History routing requires the actual frozen publisher call')
    def __call__(self,path,value,**kwargs):
        p=Path(path).resolve()
        if p in self.targets:
            require(isinstance(value,list) and value and value[0]['step']==1,'Complete authoritative in-memory history required')
            label=self.label()
            if self.journal is None:
                from append_only_history_v1 import HistoryJournal
                self.journal=HistoryJournal(self.migration['journal_directory'])
            if not self.initialized:
                require(label=='initial' and len(value)==self.plan['resume_step'],'Initial restored checkpoint frontier required')
                if self.journal.max_step:
                    require(self.journal.max_step==len(value),'Normal continuation cannot hide a journal/model rewind')
                    self.journal.reconcile_checkpoint(value,self.source_checkpoint,preserve_orphan_suffix=False)
                else:self.journal.seed(value,self.source_checkpoint)
                self.initialized=True
            else:self.journal.publish(value,publication='end' if label=='final' else 'checkpoint' if label=='initial' else label)
            if label in ('initial','DEV','final'):self.original(path,value,**kwargs)
            self.original(self.run/'history_publication_state.json',{'kind':KIND,'run_id':self.plan['run_id'],
                'effective_history_transport':'append_only_journal_with_DEV_and_closure_snapshots','journal_directory':self.migration['journal_directory'],
                'committed_frontier':self.journal.max_step,'event_id':self.journal.event_id,'snapshot_requested':label!='periodic',
                'publication':label,'journal_report':self.journal.last_report,'immutable_scientific_persistence_mode':'double_json',
                'no_model_RNG_or_training_state_from_journal':True},**kwargs)
            return self.journal.last_report
        if p.name in ('run_config.json','persistence_timing.json') and p.parent in (self.run,HERE) and isinstance(value,dict):
            value={**value,'history_runtime_migration':self.migration,'effective_history_transport':'append_only_journal_with_DEV_and_closure_snapshots'}
            if p.name=='persistence_timing.json':value.update(legacy_worker_original_format_and_frequency=value.get('original_format_and_frequency'),original_format_and_frequency=False)
        if p in self.provenance_targets:
            if isinstance(value,dict):
                value={**value,'history_runtime_migration':self.migration,'actual_additional_runtime_adapter':byte_record(__file__)}
                if p==Path(self.plan['paths']['io']).resolve():value.update(original_serializer_preserved_scope='All non-history JSON, DEV/closure history snapshots, and full207 Torch checkpoints; frozen helpers remain unchanged.',periodic_history_transport_replaced=True)
        return self.original(path,value,**kwargs)

def run_child(plan):
    migration=validate_runtime(plan);original=c.write;router=HistoryRouter(plan,original);c.write=router
    try:c.run_worker(plan)
    finally:
        c.write=original
        require(byte_record(__file__)==migration['adapter'] and byte_record(HERE/'append_only_history_v1.py')==migration['journal_helper'],'Journal/executor source changed during native window')

def run_window(state,deadline,args):
    paths={key:Path(value) for key,value in state['paths'].items()};loop=scoped(args.output_dir)
    child=None;stop_at=None;stop_reason=None;started=time.monotonic();started_utc=now();last_phase=None;last_dev=state['resume_step']
    resource=ResourceGuard();progress=domain.guards.ProgressGuard(started)
    observer=WindowObserver(state['history_runtime_migration']['journal_directory'],state['resume_step'])
    try:
        with paths['log'].open('x',encoding='utf-8') as log:
            child=subprocess.Popen(state['command'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            state.update(status='running',owned_launcher_pid=child.pid,started_at_utc=started_utc)
            print(f"ACTUAL owned full-pass {state['run_id']} PID{child.pid}; CLOSEDH full207, remaining uniform{state['remaining_uniform_updates']}14/2 updates",flush=True)
            while child.poll() is None:
                history=observer.rows()
                identity=validate_identity(read(paths['identity']),state,child.pid,started_utc) if paths['identity'].exists() else None
                suffix=[{**row,'step':i} for i,row in enumerate(history[1:],1)]
                phase=domain.guards.log_phase(paths['log'],suffix)
                sample=sample_memory(identity['pid'] if identity else child.pid,phase['phase'])
                reason,observation=resource.observe(sample['system_memory'],sample['worker_private_memory'],observed_monotonic=sample['observed_monotonic'])
                self_reason,self_observation=domain.guards.self_memory(Path(str(paths['identity'])+'.live_memory.json'),identity,paths['identity']) if identity else (None,{'available':False})
                progress_reason,progress_info=progress.observe(suffix,time.monotonic());reason=reason or self_reason or progress_reason
                if not identity and time.monotonic()-started>120:reason=reason or 'owned_worker_identity_unavailable'
                if identity and not self_observation['available'] and time.monotonic()-started>120:reason=reason or 'actual_worker_self_memory_unavailable_or_stale'
                if not suffix and time.monotonic()-started>1800:reason=reason or 'bounded_preparation_timeout'
                if paths['stop'].exists() and stop_at is None:reason=reason or 'ROOT_manual_cooperative_stop'
                if time.monotonic()>=deadline-300:reason=reason or 'window_budget_checkpoint_boundary'
                state.update(phase=phase,published_step=history[-1]['step'] if history else state['resume_step'],
                    system_and_owned_private_memory=observation,actual_worker_self_memory=self_observation,
                    progress=progress_info,updated_at_utc=now())
                private=observation.get('worker_private_memory') or {}
                if private.get('available') is True and type(private.get('private_bytes')) is int:
                    state['peak_observed_owned_private_bytes']=max(state.get('peak_observed_owned_private_bytes',0),private['private_bytes'])
                if phase['phase']!=last_phase:print(state['run_id']+' phase: '+phase['phase'],flush=True);last_phase=phase['phase']
                for row in history:
                    if row['step']>last_dev and 'validation' in row:
                        gate=row['validation'];print(f"{state['run_id']} DEV{row['step']}: TR{100*gate['turkish_wer']:.4f}%, all6={gate['retention_gate_passed']}",flush=True);last_dev=row['step']
                with (loop/'memory_observations.jsonl').open('a',encoding='utf-8') as out:
                    out.write(json.dumps({'at_utc':now(),'phase':phase,'memory':observation,'worker_self':self_observation,'progress':progress_info})+'\n')
                if reason and stop_at is None:
                    stop_at=time.monotonic();stop_reason=reason
                    request_stop(paths['stop'],reason)
                    state.update(cooperative_stop_reason=reason,cooperative_stop_requested_at_utc=now())
                write(loop/'loop_status.json',state)
                publish_queue(state)
                if stop_at and time.monotonic()-stop_at>300:raise RuntimeError('Owned child exceeded cooperative grace; preserve and review')
                time.sleep(args.poll_seconds)
        state['child_exit_code']=child.wait()
        require(child.returncode==0,'Actual child failed; no automatic GPU retry')
        selection_path=HERE/'runs'/state['run_id']/'selection.json';selection=read(selection_path,{})
        state['actual_runtime_completion']=verify_completion(state,selection)
        resource_stop=bool(stop_reason and stop_reason not in ('window_budget_checkpoint_boundary','ROOT_manual_cooperative_stop'))
        state.update(status='resource_review_required' if resource_stop else 'train_window_ended',
            selected_step=selection['selected_step'],completed_steps=selection['completed_steps'],
            selection=record(selection_path),stop_early_reason=selection['stop_early_reason'],
            resource_review_required=resource_stop,finished_at_utc=now(),objective_complete=False,
            stop_category=('resource' if resource_stop else 'manual' if stop_reason=='ROOT_manual_cooperative_stop' else 'retention' if selection['stop_early_reason']=='consecutive_retention_gate_failures'
                else 'budget' if stop_reason=='window_budget_checkpoint_boundary' or selection['stop_early_reason']=='soft_max_wall_seconds'
                else selection['stop_early_reason']),
            next_action='Only actual budget CLOSED0 may continue the same fullstate; resource/manual/scientific stops require review')
        return 1 if resource_stop else 0
    finally:
        if child is not None and child.poll() is None:
            if not paths['stop'].exists():write_stop=paths['stop'].open('x',encoding='utf-8');write_stop.write('Controller exit: cooperative save/DEV\n');write_stop.close()
            try:child.wait(timeout=300)
            except subprocess.TimeoutExpired:
                child.terminate();child.wait(timeout=30)

def prepare_plan(args):
    if args.previous_status:
        previous=read(scoped(args.previous_status));run=HERE/'runs'/previous['run_id'];selection=read(run/'selection.json')
        require(previous['kind']==c.CONTROLLER_KIND and previous['controller_sha256']==EXPECTED_CORE
            and previous['status']=='train_window_ended' and previous['child_exit_code']==0
            and previous['resource_review_required'] is False and previous['stop_category']=='budget'
            and record(run/'selection.json')==previous['selection'] and selection['completed_steps']==previous['completed_steps']
            and selection['completed_steps']<previous['configured_steps'],'Actual coherent normal-budget CLOSED0 source required')
        done=read(previous['actual_runtime_completion']['runtime_completion']['path'])
        require(done['status']=='complete' and done['sources_unchanged'] is True,'Actual native closure required')
        archived=scoped(args.closed_latest);require(c.sha(archived)==c.sha(scoped(selection['latest_weights'])),'Coherent closed latest archive required')
        output=scoped(args.output_dir);require(not output.exists() and output.parent==HERE/'runs','Fresh executor namespace required')
        paths={k:output/(previous['run_id']+'_'+n) for k,n in (('native','unused_original.json'),('fresh_runtime','fresh_native.json'),('identity','identity.json'),('memory','memory.json'),('cache','cache.json'),('io','io.json'),('stop','stop.txt'),('log','worker.log'))}
        paths['native']=Path(previous['paths']['native']);argv=list(previous['worker_arguments'])
        for flag,value in (('--resume-exact',archived),('--uniform-full-runtime-proof',paths['fresh_runtime']),('--stop-file',paths['stop']),('--max-wall-seconds',args.window_hours*3600-300)):argv=c.set_option(argv,flag,value)
        plan=dict(previous);plan.update(continuation_mode='same_new_run_exact_budget_continuation',original_ready=record(paths['native']),source_checkpoint=record(archived),resume_step=selection['completed_steps'],worker_arguments=argv,worker_arguments_sha256=c.domain.guards.argument_sha(argv),paths={k:str(v) for k,v in paths.items()})
        for key in ('child_exit_code','resource_review_required','stop_category','cooperative_stop_reason','cooperative_stop_requested_at_utc','actual_runtime_completion','finished_at_utc','error','owned_launcher_pid','started_at_utc'):plan.pop(key,None)
        args.run_id=plan['run_id'];source=record(args.previous_status)
    else:
        require(args.encoder_lr_schedule in ('constant','cosine') and args.head_lr in (1e-5,3e-5),'Actual selected encoder policy and declared head-LR only; frozen core requires its choice receipt')
        plan=c.prepare(args);source=plan['source_closure']
    output=scoped(args.output_dir);plan['command']=[str(ROOT/'work/nemo-native-env-v2/Scripts/python.exe'),str(Path(__file__).resolve()),'--worker-plan',str(output/'plan.json'),'--gpu-window-authorized']
    plan['history_runtime_migration']={'kind':KIND,'adapter':byte_record(__file__),'journal_helper':byte_record(HERE/'append_only_history_v1.py'),
        'journal_directory':str(HERE/'runs'/plan['run_id']/'history_journal_v1'),'activation_source_step':plan['resume_step'],
        'authoritative_closed_source':source,'authoritative_checkpoint':byte_record(plan['source_checkpoint']['path']),
        'scientific_identity_changed':False,'pagefile_or_guard_changes':False,'source350_repeated_on_exact':False,
        'prefix_snapshot_and_all_metrics_preserved':True,'early_publication_timelines_differ_disclosed':True,
        'numerical_bitwise_or_causal_throughput_equivalence_claimed':False}
    validate_runtime(plan);return plan

def main(args):
    if args.worker_plan:
        require(args.gpu_window_authorized,'Exclusive native ownership required');run_child(read(scoped(args.worker_plan)));return 0
    require(.1<=args.window_hours<=3 and 5<=args.poll_seconds<=15,'Original bounded windows and fresh observations required')
    plan=prepare_plan(args)
    if args.validate_only:
        if args.preflight_output:require(not scoped(args.preflight_output).exists(),'Fresh preflight output');write(args.preflight_output,{**plan,'passed':True,'GPU_launched':False,'Torch_imported':'torch' in sys.modules})
        print(json.dumps({'passed':True,'GPU_launched':False,'resume_step':plan['resume_step']}));return 0
    require(args.gpu_window_authorized,'Sole authorized GPU ownership required')
    previous=read(args.previous_status) if args.previous_status else plan['same_policy_previous']
    admission=c.current_memory_admission(previous=previous);output=scoped(args.output_dir);output.mkdir(parents=True)
    state={**plan,'status':'starting','fresh_memory_admission':admission,'updated_at_utc':now()};write(output/'plan.json',plan);write(output/'loop_status.json',state);publish_queue(state)
    state['awake_request']=domain.guards.prior.keep_awake(True)
    try:return run_window(state,time.monotonic()+args.window_hours*3600,args)
    except BaseException as error:state.update(status='failed',error=repr(error),finished_at_utc=now());raise
    finally:state['awake_clear']=domain.guards.prior.keep_awake(False);write(output/'loop_status.json',state);publish_queue(state)

def parser():
    p=c.parser();p.description=__doc__;p.add_argument('--previous-status',type=Path);p.add_argument('--closed-latest',type=Path);return p

if __name__=='__main__':sys.exit(main(parser().parse_args()))
