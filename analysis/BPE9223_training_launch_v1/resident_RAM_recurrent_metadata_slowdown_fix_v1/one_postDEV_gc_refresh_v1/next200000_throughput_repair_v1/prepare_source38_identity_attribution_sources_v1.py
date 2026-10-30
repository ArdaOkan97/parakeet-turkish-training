"""Metadata-only source38 generation; AST parse/RAW inverse, NEVER import it."""
import ast,copy,hashlib,json,os,shutil,sys,time
from pathlib import Path
import psutil

P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');R=P/'work/research';S=P/'outputs/parakeet_turkish'
W=Path(__file__).parent;sys.path.insert(0,str(S))
from resource_watchdog_v2 import sample_memory


def pin(path):
    path=Path(path);b=path.read_bytes();return dict(path=str(path),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def write(path,value):
    raw=(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    return pin(path)


def who(proc):return dict(pid=proc.pid,create_time=proc.create_time(),ppid=proc.ppid())


me=psutil.Process();me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
initial=sample_memory(os.getpid(),'source38_inactive_metadata_preparation')
ip=write(W/'source38_inactive_preparation_initial_numeric_v1.json',dict(sample=initial,preparer=who(me),launcher=who(me.parent()),before_predicate=True))
assert initial['system_memory']['commit_available_bytes']>=5130803840 and initial['system_memory']['physical_available_bytes']>=2380803840
assert initial['worker_private_memory']['private_bytes']<536870912 and shutil.disk_usage('E:/').free>=100000000000
assert os.environ['CUDA_VISIBLE_DEVICES']=='-1'
assert all(os.environ[k]=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'))


def transform(base,target,make):
    raw=base.read_bytes();text=raw.decode();steps=[]
    def edit(old,new,count=1):
        nonlocal text
        assert text.count(old)==count,(old[:100],text.count(old),count)
        text=text.replace(old,new);steps.append(dict(old=old,new=new,count=count))
    make(edit,text)
    out=text.encode();ast.parse(out)
    inverse=text
    for e in reversed(steps):
        assert inverse.count(e['new'])==e['count'];inverse=inverse.replace(e['new'],e['old'])
    assert inverse.encode()==raw
    with target.open('xb') as f:f.write(out);f.flush();os.fsync(f.fileno())
    return dict(base=pin(base),successor=pin(target),ordered_edits=steps,whole_byte_inverse=True,
        AST=hashlib.sha256(ast.dump(ast.parse(out),include_attributes=False).encode()).hexdigest())


def driver_edits(edit,text):
    edit("sys.path.insert(0,str(NINTH_TRANSITION_FOLDER))", "sys.path.insert(0,str(NINTH_TRANSITION_FOLDER))\nTHROUGHPUT_FOLDER=Path("+repr(str(W))+ ")\nsys.path.insert(0,str(THROUGHPUT_FOLDER))")
    edit("    base=read_json(a.configuration);pointer_pin=record(a.pointer);pointer=read_json(a.pointer)",'''    candidate_base=read_json(a.configuration)
    from asr18_current_plan_throughput_checkpoint_transition_v1 import (checked_current_plan_throughput_policy,
        checked_current_plan_throughput_checkpoint_resume,legacy_source37_configuration,EXTRA_HOST_BYTES)
    throughput_active='authoritative_current_plan_identity_and_main_path_attribution_policy'in candidate_base
    throughput_reserve=EXTRA_HOST_BYTES if throughput_active else 0
    if throughput_active:
        OriginalSourceAuthority,current_plan_identity_view,MainPathScalarAttribution=checked_current_plan_throughput_policy(candidate_base)
    base=legacy_source37_configuration(candidate_base)if throughput_active else candidate_base
    pointer_pin=record(a.pointer);pointer=read_json(a.pointer)
    throughput_transition=None if a.throughput_resume is None else checked_current_plan_throughput_checkpoint_resume(
        record(a.throughput_resume),base=candidate_base,folder=folder,pointer_pin=pointer_pin)
    if throughput_transition is not None:
        require(throughput_active and a.invocation==27 and all(x is None for x in (a.prior_process,a.ninth_original_resume,
            a.gc_refresh_resume,a.eighth_original_resume,a.fresh_signature_resume,a.canonical_archive_resume,a.lookahead_resume,
            a.cushion225_resume,a.postrestore_selector_resume,a.packet_credit_resume,a.native_headroom_resume,a.abnormal_resume,
            a.source_transition_resume,a.original_interval_resume,a.DEV_frontend_resume,a.DEV_admission_resume,a.periodic_quiescence_resume)),
            'Only exact source37 post204000DEV cooperative stop, never normal0 or an old failure lane')''')
    edit("    require(ninth_interval_failure is not None or gc_transition is not None or a.prior_process is not None,", "    require(throughput_transition is not None or ninth_interval_failure is not None or gc_transition is not None or a.prior_process is not None,")
    old="    transition=gc_transition if gc_transition is not None else(signature_transition if signature_transition is not None else(canonical_transition if canonical_transition is not None else(lookahead_transition if lookahead_transition is not None else(None if a.source_transition_resume is None else checked_source_transition_resume(\n        record(a.source_transition_resume),base=base,folder=folder,pointer_pin=pointer_pin)))))"
    edit(old,old.replace('transition=','legacy_transition=',1)+"\n    transition=throughput_transition if throughput_transition is not None else legacy_transition")
    old=next(line for line in text.splitlines() if line.startswith('    preserved_tail='))
    edit(old,'    preserved_tail=throughput_transition if throughput_transition is not None else('+old.split('=',1)[1]+')')
    edit('    c=copy.deepcopy(base);c.update(', '    c=copy.deepcopy(candidate_base);c.update(')
    edit('    def __init__(self,folder):\n        self.folder=folder;self.stop=', '    def __init__(self,folder,*,additional_reserved_host_bytes=0):\n        require(type(additional_reserved_host_bytes)is int and additional_reserved_host_bytes>=0,\n            \'Explicit separately charged extra host bookkeeping, no original guard reduction\')\n        self.additional_reserved_host_bytes=additional_reserved_host_bytes\n        self.folder=folder;self.stop=')
    edit("        if m['commit_available_bytes']<2000000000 or m['physical_available_bytes']<1500000000:","        if m['commit_available_bytes']<2000000000+self.additional_reserved_host_bytes or m['physical_available_bytes']<1500000000+self.additional_reserved_host_bytes:")
    edit("        require(m['commit_available_bytes']>=2000000000 and m['physical_available_bytes']>=1500000000,", "        require(m['commit_available_bytes']>=2000000000+self.additional_reserved_host_bytes and m['physical_available_bytes']>=1500000000+self.additional_reserved_host_bytes,")
    edit('    monitor=Monitor(invocation);stream=None;', '    monitor=Monitor(invocation,additional_reserved_host_bytes=throughput_reserve);stream=None;')
    edit("            extra=0 if name=='private_scalar_prefetch_additional_host_state' else reserve", "            extra=(0 if name=='private_scalar_prefetch_additional_host_state' else reserve)+throughput_reserve")
    edit("s['system_memory']['physical_available_bytes']-reserve>=1500000000", "s['system_memory']['physical_available_bytes']-reserve-throughput_reserve>=1500000000")
    edit("        def journal_factory(saved):\n            if ninth_interval_failure is not None:", "        def journal_factory(saved):\n            if throughput_transition is not None:\n                require(saved==source_journal,'Exact actual204000 full saved prefix; all later tail historical and genuinely redone')\n                return SavedJournalPrefixView(saved,guard),{'kind':'ASR18_readonly_saved204000_prefix_after_source37_throughput_transition_v1',\n                    'actual_transition':record(a.throughput_resume),'original_tail_modified':False,'no_tail_replay':True}\n            if ninth_interval_failure is not None:")
    edit("'actual_transition':record(a.gc_refresh_resume if a.gc_refresh_resume is not None else", "'actual_transition':record(a.throughput_resume if a.throughput_resume is not None else a.gc_refresh_resume if a.gc_refresh_resume is not None else")
    edit("(a.ninth_original_resume is not None or a.gc_refresh_resume is not None", "(a.throughput_resume is not None or a.ninth_original_resume is not None or a.gc_refresh_resume is not None")
    edit("'predecessor':record(a.ninth_original_resume if a.ninth_original_resume is not None else", "'predecessor':record(a.throughput_resume if a.throughput_resume is not None else a.ninth_original_resume if a.ninth_original_resume is not None else")
    edit("        while optimizer.updates<c['planned_native_updates']:", "        identity_authority=OriginalSourceAuthority(engine.context)if throughput_active else None\n        phase_observer=MainPathScalarAttribution()if throughput_active else None\n        while optimizer.updates<c['planned_native_updates']:")
    start=text.index('            for index,entry in enumerate(plan[\'updates\']):')
    end=text.index('                rows=entry[\'rows\'];arrays=packet[\'arrays\']',start)
    old=text[start:end]
    new='''            for index,entry in enumerate(plan['updates']):
                phase_stamp=phase_observer.start('host_gate')if phase_observer is not None else None
                try:
                    pressure_admission=wait_for_native_commit_pressure(observe=monitor._observe,check=guard,policy=native_pressure_policy['parameters'])
                    metadata_timing['native_commit_pressure_wait_seconds']=metadata_timing.get('native_commit_pressure_wait_seconds',0.0)+pressure_admission['wait_seconds']
                    metadata_timing['native_commit_pressure_waited_updates']=metadata_timing.get('native_commit_pressure_waited_updates',0)+int(pressure_admission['observations']>1)
                finally:
                    if phase_stamp is not None:phase_observer.finish(phase_stamp,raised=sys.exc_info()[0]is not None)
                ready_call=stream.wait_update_ready
                try:
                    phase_stamp=phase_observer.start('digest_readiness')if phase_observer is not None else None
                    try:readiness_plan_digest=digest(plan)
                    finally:
                        if phase_stamp is not None:phase_observer.finish(phase_stamp,raised=sys.exc_info()[0]is not None)
                    phase_stamp=phase_observer.start('readiness_wait')if phase_observer is not None else None
                    try:ready_call(readiness_plan_digest,index)
                    finally:
                        if phase_stamp is not None:phase_observer.finish(phase_stamp,raised=sys.exc_info()[0]is not None)
                finally:del ready_call
                phase_stamp=phase_observer.start('host_gate')if phase_observer is not None else None
                try:
                    pressure_after_readiness=wait_for_native_commit_pressure(observe=monitor._observe,check=guard,policy=native_pressure_policy['parameters'])
                    metadata_timing['native_commit_pressure_wait_seconds']+=pressure_after_readiness['wait_seconds']
                    metadata_timing['native_commit_pressure_waited_updates']+=int(pressure_after_readiness['observations']>1 and pressure_admission['observations']<=1)
                finally:
                    if phase_stamp is not None:phase_observer.finish(phase_stamp,raised=sys.exc_info()[0]is not None)
                guard();next_call=stream.next_update
                try:
                    phase_stamp=phase_observer.start('digest_take')if phase_observer is not None else None
                    try:take_plan_digest=digest(plan)
                    finally:
                        if phase_stamp is not None:phase_observer.finish(phase_stamp,raised=sys.exc_info()[0]is not None)
                    phase_stamp=phase_observer.start('packet_take')if phase_observer is not None else None
                    try:packet=next_call(take_plan_digest,index)
                    finally:
                        if phase_stamp is not None:phase_observer.finish(phase_stamp,raised=sys.exc_info()[0]is not None)
                finally:del next_call
                metadata=packet.get('metadata',packet['update'])
                phase_stamp=phase_observer.start('digest_binding')if phase_observer is not None else None
                try:
                    require(packet['plan_sha256']==(current_plan_digest:=digest(plan)) and metadata['update_index']==index
                        and all(metadata[k]==entry[k]for k in ('native_transaction_sha256','before_sampler_sha256','after_sampler_sha256','batch_id')),
                        'Current actual transaction matches privately prepared RAM packet')
                finally:
                    if phase_stamp is not None:phase_observer.finish(phase_stamp,raised=sys.exc_info()[0]is not None)
                if phase_observer is not None:
                    try:phase_observer.copy_preparation_scalar(packet['timings']['preparation_seconds'])
                    except (KeyError,TypeError,ValueError):metadata_timing['packet_preparation_scalar_observation_errors']=metadata_timing.get('packet_preparation_scalar_observation_errors',0)+1
                phase_stamp=phase_observer.start('authoritative_propose')if phase_observer is not None else None
                try:
                    if throughput_active:
                        with current_plan_identity_view(engine.context,engine.schedule,plan,index,entry,authority=identity_authority,
                            fresh_plan_sha256=current_plan_digest,packet_plan_sha256=packet['plan_sha256'],optimizer_updates=optimizer.updates)as identity_used:
                            tx=engine.schedule.propose(optimizer_updates=optimizer.updates,resolve_TR=engine.context.resolve_identity)
                        metadata_timing['identity_view_eligible_updates']=metadata_timing.get('identity_view_eligible_updates',0)+int(identity_used)
                        metadata_timing['identity_view_fallback_updates']=metadata_timing.get('identity_view_fallback_updates',0)+int(not identity_used)
                    else:tx=engine.schedule.propose(optimizer_updates=optimizer.updates,resolve_TR=engine.context.resolve_identity)
                finally:
                    if phase_stamp is not None:phase_observer.finish(phase_stamp,raised=sys.exc_info()[0]is not None)
                require(tx.identity_digest==entry['native_transaction_sha256'] and digest(tx.before)==entry['before_sampler_sha256'],
                    'Original source schedule/replay selection before positive Adam')
'''
    edit(old,new)
    # Wrap exact original metadata/native/archive/progress blocks by indentation.
    def wrap(old,label,indent='                '):
        body=''.join('    '+line if line.strip() else line for line in old.splitlines(keepends=True))
        return indent+"phase_stamp=phase_observer.start('"+label+"')if phase_observer is not None else None\n"+indent+'try:\n'+body+indent+'finally:\n'+indent+'    if phase_stamp is not None:phase_observer.finish(phase_stamp,raised=sys.exc_info()[0]is not None)\n'
    a=text.index("                rows=entry['rows'];arrays=packet['arrays']",start)
    b=text.index('                start=time.monotonic()',a)
    old=text[a:b];edit(old,wrap(old,'metadata_binding'))
    a=text.index('                start=time.monotonic()',a);b=text.index('                prepared_archive.append(',a)
    old=text[a:b];edit(old,wrap(old,'native_and_commit'))
    old=next(line+'\n' for line in text.splitlines() if line.startswith('                prepared_archive.append('));edit(old,wrap(old,'archive_drop_release'))
    a=text.index('                if optimizer.updates==initial_updates+1 or optimizer.updates%20==0:');b=text.index('            if future_ticket is not None:',a)
    old=text[a:b];edit(old,wrap(old,'progress_emit'))
    edit("            metadata_timing['original_collect_observation']=original_collect_observer.take_window()", "            if phase_observer is not None:\n                phase_summary=phase_observer.snapshot_and_reset()\n                require(len(canonical(phase_summary))<=16*1024,'Fixed bounded per20 phase scalars, no payload/history')\n                metadata_timing['main_path_scalar_attribution']=phase_summary\n            metadata_timing['original_collect_observation']=original_collect_observer.take_window()")
    edit("p.add_argument('--ninth-original-resume');", "p.add_argument('--throughput-resume');p.add_argument('--ninth-original-resume');")


def wrapper_edits(edit,text):
    edit("sys.path.insert(0,str(NINTH_TRANSITION_FOLDER))", "sys.path.insert(0,str(NINTH_TRANSITION_FOLDER))\nTHROUGHPUT_FOLDER=Path("+repr(str(W))+ ")\nsys.path.insert(0,str(THROUGHPUT_FOLDER))")
    edit('run_ASR18_BPE9223_resident_RAM_training_v37','run_ASR18_BPE9223_resident_RAM_training_v38',10)
    edit("    configuration=read_json(a.configuration);policy=configuration['private_scalar_prefetch_policy']",'''    candidate_configuration=read_json(a.configuration)
    from asr18_current_plan_throughput_checkpoint_transition_v1 import (checked_current_plan_throughput_policy,
        checked_current_plan_throughput_checkpoint_resume,legacy_source37_configuration)
    throughput_active='authoritative_current_plan_identity_and_main_path_attribution_policy'in candidate_configuration
    if throughput_active:checked_current_plan_throughput_policy(candidate_configuration)
    configuration=legacy_source37_configuration(candidate_configuration)if throughput_active else candidate_configuration
    policy=configuration['private_scalar_prefetch_policy']''')
    edit("    if a.prior_process is not None:\n        require(a.ninth_original_resume", "    if a.prior_process is not None:\n        require(a.throughput_resume is None and a.ninth_original_resume")
    marker="    elif a.ninth_original_resume is not None:"
    edit(marker,'''    elif a.throughput_resume is not None:
        require(throughput_active and all(x is None for x in (a.ninth_original_resume,a.gc_refresh_resume,a.eighth_original_resume,
            a.fresh_signature_resume,a.canonical_archive_resume,a.lookahead_resume,a.cushion225_resume,a.postrestore_selector_resume,
            a.packet_credit_resume,a.native_headroom_resume,a.abnormal_resume,a.original_interval_resume,a.source_transition_resume,
            a.DEV_frontend_resume,a.DEV_admission_resume,a.periodic_quiescence_resume))
            and review['throughput_resume']==record(a.throughput_resume)
            and review['throughput_transition_validator']==record(THROUGHPUT_FOLDER/'asr18_current_plan_throughput_checkpoint_transition_v1.py')
            and review['authoritative_current_plan_identity_and_main_path_attribution_policy']==candidate_configuration['authoritative_current_plan_identity_and_main_path_attribution_policy']
            and a.start_invocation==review['initial_owner_invocation']==27,
            'Only reviewed current source37 post204000DEV cooperative checkpoint transition, no normal0 alias')
    elif a.ninth_original_resume is not None:''')
    marker="    if a.ninth_original_resume is not None:\n        checked_ninth_original_resume"
    edit(marker,"    if a.throughput_resume is not None:\n        checked_current_plan_throughput_checkpoint_resume(record(a.throughput_resume),base=candidate_configuration,folder=branch,pointer_pin=record(a.pointer))\n    if a.ninth_original_resume is not None:\n        checked_ninth_original_resume")
    edit("            if a.ninth_original_resume is not None:command+=['--ninth-original-resume',a.ninth_original_resume]", "            if a.throughput_resume is not None:command+=['--throughput-resume',a.throughput_resume]\n            if a.ninth_original_resume is not None:command+=['--ninth-original-resume',a.ninth_original_resume]")
    edit("p.add_argument('--ninth-original-resume');", "p.add_argument('--throughput-resume');p.add_argument('--ninth-original-resume');")


driver=transform(R/'run_ASR18_BPE9223_resident_RAM_training_v37.py',R/'run_ASR18_BPE9223_resident_RAM_training_v38.py',driver_edits)
wrapper=transform(R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v37.py',R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v38.py',wrapper_edits)
raw=write(W/'source37_to38_driver_wrapper_full_RAW_AST_v1.json',dict(driver=driver,wrapper=wrapper))
runtime=json.loads((Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01/invocations/owner_000026/configuration.json')).read_text())
proper_pin=runtime['source_configuration'];proper=json.loads(Path(proper_pin['path']).read_text());new=copy.deepcopy(proper)
new['authoritative_current_plan_identity_and_main_path_attribution_policy']=dict(kind='ASR18_authoritative_current_plan_identity_and_main_path_attribution_v1',identity_helper=pin(W/'current_plan_identity_view_v3.py'),phase_helper=pin(W/'main_path_scalar_attribution_v1.py'),actual_CPU_control_review=pin(W/'ROOT_current_plan_identity_view_actual_CPU_independent_acceptance_v1.json'),additional_host_reservation_bytes=75497472,identity_reservation_bytes=8388608,conservative_phase_bookkeeping_reservation_bytes=67108864,original_three_fresh_canonical_plan_digests_unchanged=True,same_original_row_guard_two_fresh_signatures_poison_and_deepcopy=True,original225_MAX2_FULLprefetch_guards_and_consumed180k_unchanged=True)
new['prior_resident_RAM_source37_postDEV204000_throughput_transition']=None
cfg=write(W/'source38_candidate_configuration_v1.json',new)
proof=copy.deepcopy(new);del proof['authoritative_current_plan_identity_and_main_path_attribution_policy'];del proof['prior_resident_RAM_source37_postDEV204000_throughput_transition'];assert proof==proper
handoff=write(W/'source38_driver_wrapper_configuration_INACTIVE_handoff_v1.json',dict(driver=driver['successor'],wrapper=wrapper['successor'],transition_helper=pin(W/'asr18_current_plan_throughput_checkpoint_transition_v1.py'),configuration=cfg,proper_source37=proper_pin,current_runtime37=pin(Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01/invocations/owner_000026/configuration.json')),full_RAW=raw,only_two_operational_configuration_fields=True,actual_CPU_gate_pending=True,actual_stop_pending=True,typed_metadata_proof_pending=True,validation=dict(restored=204000,start=204020,end=209020,all_original20rows=True,all_saves=True,real208000DEV_plus1020=True,sustained_inclusive_rate_strictly_above=70,maximum_save_excluded_first_final_decline_percent=5),initial=ip,preparer=who(me),launcher=who(me.parent())))
print(json.dumps(dict(driver=driver['successor'],wrapper=wrapper['successor'],RAW=raw,configuration=cfg,handoff=handoff,preparer=who(me),launcher=who(me.parent()))))
