"""Resident CPU/RAM/native continuation of the exact saved training state.

One GPU owner, private CPU planning, real positive-Adam journal commits, full
streamed checkpoints every 200 updates and at declared pass/DEV boundaries.
Feature arrays never become files. Old RUN07 files are read, never rewritten.
"""
import argparse,copy,gc,gzip,hashlib,inspect,json,os,secrets,shutil,sys,threading,time,traceback
from contextlib import contextmanager,ExitStack
from pathlib import Path
P=Path(r"C:/Users/ardao/Documents/Codex/2026-10-10/i-x20");S=P/'outputs/parakeet_turkish'
sys.path[:0]=[str(S),str(P/'work/research')]
NINTH_TRANSITION_FOLDER=Path('E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1/resident_RAM_source36_original_interval186563_failure_actual_v1')
sys.path.insert(0,str(NINTH_TRANSITION_FOLDER))
THROUGHPUT_FOLDER=Path('E:\\gpu_training_scratchpad\\analysis\\BPE9223_training_launch_v1\\resident_RAM_recurrent_metadata_slowdown_fix_v1\\one_postDEV_gc_refresh_v1\\next200000_throughput_repair_v1')
sys.path.insert(0,str(THROUGHPUT_FOLDER))
RECOVERY_FOLDER=Path(__file__).resolve().parent
sys.path.insert(0,str(RECOVERY_FOLDER))
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import digest,canonical
ROOT=Path(r'E:\gpu_training_scratchpad');OWNER='/root/memory_diagnosis'


def commit_RAM_transaction(tx,optimizer,journal,native,evidence):
    """Same scientific commit as ScheduledTransaction.commit, explicit RAM evidence."""
    schedule=tx.schedule
    require(not tx.closed and not tx.poisoned and not schedule.poisoned and schedule.state_dict()==tx.before,
        'Only current untouched sampler transaction commits')
    require(not optimizer.poisoned and optimizer.updates==tx.optimizer_updates+1,'Exactly one actual positive Adam')
    try:
        expected=tx.TR_identity+list(tx.replay_rows)
        require(native['update']==optimizer.updates and native['phase']==tx.phase
            and native['source_ids']==[r['source_id']for r in expected]
            and native['languages']==[r['language']for r in expected], 'Exact ordered source/phase native receipt')
        require(evidence['kind']=='ASR18_resident_RAM_preparation_commit_evidence_v1'
            and len(evidence['metadata_sha256'])==64 and evidence['actual_positive_native_update']==optimizer.updates,
            'Truthful consumed RAM packet, no fictitious closed file/process')
        tx.cursor.commit(tx.TR_draws);schedule.replay.load_state_dict(tx.after_replay);state=schedule.state_dict()
        compact={k:native[k]for k in ('update','phase','source_ids','languages','mean_objective','global_grad_norm')}
        compact.update(full_native_receipt_sha256=digest(native),augmentation_sha256=digest(native['augmentation']))
        journal.append({'kind':'ASR18_committed_training_update_v1','phase':tx.phase,'optimizer_updates':optimizer.updates,
            'transaction_sha256':tx.identity_digest,'sampler_state_sha256':digest(state),
            'broad_draws':schedule.broad.draws,'cleaner_draws':schedule.cleaner.draws,'replay_updates':schedule.replay.updates,
            'prepared_evidence':evidence,'native':compact})
        tx.closed=True;return state
    except BaseException:
        tx.poisoned=schedule.poisoned=optimizer.poisoned=True;raise


class SavedJournalPrefixView:
    """Readonly saved frontier while the real failed writer's later tail is retained."""
    def __init__(self,saved,guard):self.saved=copy.deepcopy(saved);self.guard=guard;self.verify(saved)
    def binding(self):return copy.deepcopy(self.saved)
    def verify(self,binding):
        from bpe_bounded_production_journal_v4 import prefix
        require(binding==self.saved and prefix(Path(binding['path']),binding['offset'],self.guard).hexdigest()==binding['sha256'],
            'Actual unchanged saved journal prefix; later tail is never replayed')
        return True
    def append(self,event):raise RuntimeError('Readonly checkpoint restore view cannot append')


def checked_headroom_resume(pin,*,base,folder,pointer_pin):
    import psutil
    require(pin==base['prior_resident_RAM_headroom_failure'] and record(pin['path'])==pin,'Exact ROOT-bound actual133137 resource failure')
    value=read_json(pin['path']);e=value['source_evidence']
    require(value['kind']=='ASR18_actual_resident_RAM_headroom133137_saved133000_abnormal_closure_v1'
        and value['actual_TOOL_exit_code']==1 and value['actual_TOOL_chunk_id']=='d0a49f'
        and value['actual_failed_optimizer_updates']==133137 and value['saved_optimizer_updates']==133000
        and value['normal_completion_claimed'] is False and value['original_failure_was_resource_guard_not_normal0'] is True
        and value['native_OS_exit_code'] is None and value['native_OS_exit_code_not_observed'] is True,
        'Truthful abnormal stop, unknown numeric native exit and exact saved133000')
    require(e['saved_pointer']==pointer_pin and Path(pointer_pin['path']).resolve().is_relative_to(folder/'checkpoints')
        and e['driver_source']==record(P/'work/research/run_ASR18_BPE9223_resident_RAM_training_v12.py')
        and e['wrapper_source']==record(P/'work/research/run_ASR18_BPE9223_resident_RAM_autocontinue_v12.py'),
        'Exact original source12 failure and preserved branch checkpoint')
    for item in e.values():require(record(item['path'])==item,'Exact original failure/source/checkpoint evidence')
    require(record(value['owned_process_deaths']['path'])==value['owned_process_deaths'],'Exact actual owned-death observation')
    observed=read_json(value['owned_process_deaths']['path'])['identities']
    require(len(observed)==5 and {x['label']for x in observed}=={'native','native interpreter launcher','resident CPU','CPU interpreter launcher','outer wrapper'},'Only the five actually recorded identities')
    for item in observed:
        who=item['original_identity']
        try:current=psutil.Process(who['pid']).create_time()
        except psutil.NoSuchProcess:continue
        require(who['create_time'] is not None and abs(current-who['create_time'])>=.001,'Original owned identity is dead; never invent missing create time')
    failure=read_json(e['actual_failure']['path']);monitor=read_json(e['monitor_completion']['path'])
    sample=read_json(e['failing_headroom_sample']['path'])
    require(failure['actual_optimizer_updates']==133137 and failure['last_checkpoint']==133000
        and 'Sensible real system headroom' in failure['error'] and monitor['joined'] is True and monitor['clean'] is False
        and sample['system_memory']['commit_available_bytes']==1639993344
        and sample['system_memory']['physical_available_bytes']==12542930944,
        'Genuine captured system-commit guard failure, not normal0 or an interval repair')
    saved={k:value['saved_journal_prefix'][k]for k in ('path','offset','sha256')}
    whole=value['whole_old_journal'];tail=value['preserved_unsaved_positive_tail']
    require(saved=={'path':str(folder/'journals/commits_000010.jsonl'),'offset':270832675,
        'sha256':'a2811751b95fd8c6df021d3495b4af77e0ccd315134e667a09e893da97dfdc0e'}
        and read_json(pointer_pin['path'])['journal_prefix']==saved
        and whole=={'path':saved['path'],'bytes':271101236,'sha256':'99014a7440de7360f3c1065b09029604e8c0e8a6025e84ba6872a7398876abd1'}
        and Path(whole['path']).stat().st_size==whole['bytes'],'Exact saved prefix and preserved whole old lineage')
    tail_pin={k:tail[k]for k in ('path','bytes','sha256')}
    require(tail_pin['bytes']==268561 and tail_pin['sha256']=='227764ba62b5fee6fa87aaeabf9d4f0197ea2b414f41556d2a60310afb24dc8c'
        and record(tail_pin['path'])==tail_pin,'Exact137 historical positive tail, never replayed')
    events=[__import__('json').loads(line)for line in Path(tail_pin['path']).read_bytes().splitlines()]
    require(len(events)==137 and all(x['optimizer_updates']==u and x['native']['update']==u
        and x['prepared_evidence']['actual_positive_native_update']==u for u,x in zip(range(133001,133138),events)),
        'Actual ordered137 positive events remain historical')
    return {**value,'saved_journal_prefix':saved,'whole_journal_including_preserved_tail':whole,
        'preserved_unsaved_tail':tail_pin,'preserved_tail_positive_updates':list(range(133001,133138))}


def checked_native_commit_pressure_policy(base):
    """Exact completed CPU composition gate, still no workload/prevention claim."""
    import importlib.util
    policy=base['native_update_commit_pressure_policy']
    require(set(policy)=={'gate_source','parameters','actual_CPU_control_review'},
        'Exact three-field native commit-pressure scheduling policy')
    path=ROOT/'analysis/BPE9223_training_launch_v1/resident_RAM_native_pressure_timeout156769_failure_actual_v1/inactive_additional_cushion225_v1/asr18_native_update_commit_pressure_gate_v3.py'
    require(policy['gate_source']==record(path) and policy['gate_source']['sha256']=='f2aa9d3206ada6eb2496dc176e29751dc04b6535750694903664eca6245b65ff',
        'Exact separately reviewed native pressure helper3, no source alias')
    spec=importlib.util.spec_from_file_location('ASR18_exact_native_commit_pressure_gate_v3',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    require(policy['parameters']==module.POLICY,'Exact stricter scheduling buffer and unchanged original floors/full reservation')
    review_pin=policy['actual_CPU_control_review']
    require(isinstance(review_pin,dict) and record(review_pin['path'])==review_pin,'Genuine completed CPU composition acceptance')
    review=read_json(review_pin['path'])
    require(review['accepted'] is True and review['reviewer']=='/root'
        and review['scope']=='ASR18_native_preupdate_commit_cushion2250000000_policy_v1'
        and review['human_approved_additional_buffer_bytes']==2250000000
        and review['gate_source']==policy['gate_source'] and review['policy']==policy['parameters'],
        'Exact ROOT focused candidate proof and explicit human225 approval, no allocation guarantee')
    proof_pin=review['actual_CPU_control_report'];require(record(proof_pin['path'])==proof_pin,'Exact actual CPU control report')
    proof=read_json(proof_pin['path'])
    require(proof['passed'] is True and proof['actual_native_optimizer_updates']==proof['actual_sampler_draws']==proof['actual_audio_decode']==0
        and proof['actual_GPU_activity'] is False and proof['focused_policy_delta_controls_passed'] is True
        and proof['fake_clock_and_historical_samples_explicit'] is True and proof['original_Monitor_AST_unchanged'] is True
        and proof['old_genuine_Monitor_review_reused_without_execution'] is True
        and proof['original_genuine_Monitor_completed_review']==record(ROOT/'analysis/BPE9223_training_launch_v1/ROOT_native_preupdate_commit_pressure_actual_CPU_review_v1.json')
        and proof['gate_source']==policy['gate_source'] and proof['policy']==policy['parameters'],
        'Honest focused delta proof; closed genuine Monitor evidence reused, not re-executed')
    return policy,module.wait_for_native_update_commit_pressure


def checked_cushion225_resume(pin,*,base,folder,pointer_pin):
    """Actual source24 first-gate timeout; saved156600+169 historical positives."""
    import psutil
    require(pin==base['prior_resident_RAM_native_pressure156769_failure'] and record(pin['path'])==pin
        and pin['sha256']=='84d010159e56666de37cedfc890f098cae1b1678da90215e328bd6bbaa0ec298',
        'Exact source24 added-buffer TOOL1, never normal0 or original-floor failure')
    value=read_json(pin['path']);e=value['inputs']
    require(value['kind']=='ASR18_actual_source24_saved156600_native_preupdate_pressure_timeout156769_closure_v1'
        and value['owned_session']==96057 and value['actual_outer_TOOL_exit_code']==1
        and value['actual_outer_TOOL_chunk_id']=='102679' and value['actual_native_optimizer_updates']==156769
        and value['last_fully_saved_optimizer_updates']==156600 and value['unsaved_positive_tail_count']==169
        and value['native_numeric_OS_exit_code'] is None and value['CPU_numeric_OS_exit_code'] is None
        and value['wrapper_numeric_OS_exit_code_separately_recorded'] is None
        and value['no_unknown_numeric_OS_exit_invented'] is True,
        'Real outer1/unknown separate OS codes/169 historical attempts')
    for item in e.values():require(record(item['path'])==item,'Exact compact source24 evidence before ownership writes')
    require(e['pointer']==pointer_pin and record(pointer_pin['path'])==pointer_pin
        and e['effective_configuration']==record(folder/'invocations/owner_000018/configuration.json'),
        'Exact source24 runtime and full saved156600 checkpoint')
    boot=read_json(e['boot']['path']);ready=read_json(e['CPU_ready']['path'])
    require(boot['source']==record(P/'work/research/run_ASR18_BPE9223_resident_RAM_training_v24.py')
        and ready['source']['protocol']==record(S/'fastmss_resident_feature_stream_v10.py')
        and read_json(e['current_child']['path'])['invocation']==18,
        'Historical source24/CPU10 pins remain historical, not successor aliases')
    failure=read_json(e['actual_failure']['path']);monitor=read_json(e['original_monitor_completion']['path'])
    d=failure['native_pressure_denial'];m=d['last_denied_sample']['system_memory']
    require(failure==value['actual_failure'] and failure['actual_optimizer_updates']==156769
        and failure['last_checkpoint']==156600 and 'line 910, in main' in failure['traceback']
        and failure['error']=="TimeoutError('Bounded native pre-update commit-pressure wait exhausted; preserve state, no unchanged retry')"
        and d['observed_wait_seconds']==d['maximum_wait_seconds']==300 and d['observations']==600
        and d['parameters']['additional_native_commit_pressure_buffer_bytes']==2500000000
        and d['required_commit_available_bytes']==4768435456 and d['required_physical_available_bytes']==1768435456
        and m['commit_available_bytes']==4669493248 and m['physical_available_bytes']==11246845952
        and d['no_packet_propose_native_or_authoritative_operation_admitted'] is True
        and monitor==value['original_monitor_completion'] and monitor['joined'] is True
        and monitor['clean'] is True and monitor['error'] is None,
        'Exact FIRST-gate additional-buffer denial and clean original Monitor')
    expected=[('native',42304,1792716760.1698885),('native_interpreter_launcher',38476,1792716760.1632268),
        ('wrapper',15020,1792716759.746992),('CPU',3868,1792716948.434736),('CPU_interpreter_launcher',40792,1792716948.4150295)]
    require([(x['role'],x['pid'],x['create_time'])for x in value['all_five_recorded_runtime_identities_dead']]==expected,
        'Exactly five recorded source24 runtime identities')
    for role,pid,ctime in expected:
        try:now=psutil.Process(pid).create_time();dead=abs(now-ctime)>=.001
        except psutil.NoSuchProcess:dead=True
        require(dead,'Exact source24 predecessor actually dead '+role)
    saved=value['saved_journal_prefix_as_verified_by_published_persistence'];stat=value['closed_journal_stat'];tail=value['preserved_unsaved_positive_tail']
    require(saved=={'path':str(folder/'journals/commits_000018.jsonl'),'offset':317194933,
        'sha256':'eb6b816bff1f40706a4595a045c2c3b9cafc5eaedccff6dcc010fab0d0e2a199'}
        and read_json(pointer_pin['path'])['journal_prefix']==saved and stat['path']==saved['path']
        and stat['bytes']==317525825 and tail['bytes']==330892
        and tail['sha256']=='b37117924c094f321b0898ea3e20218488fa68defae598229996a1f276282186'
        and record(tail['path'])==tail and saved['offset']+tail['bytes']==stat['bytes'],
        'Exact saved156600 prefix and retained169 tail, no invented whole18 hash')
    current=Path(stat['path']).stat()
    require(current.st_size==stat['bytes'] and current.st_mtime_ns==stat['mtime_ns'],'Original closed18 STAT unchanged')
    events=[json.loads(line)for line in Path(tail['path']).read_bytes().splitlines()]
    require(len(events)==169 and all(x['optimizer_updates']==x['native']['update']==x['prepared_evidence']['actual_positive_native_update']==u
        for u,x in zip(range(156601,156770),events)),'169 ordered historical positives require genuine redo')
    persistence=read_json(e['persistence']['path'])
    require(persistence['passed'] is True and persistence['optimizer_updates']==156600
        and persistence['named_tensors_verified']==854 and persistence['weights_verified']==207 and persistence['Adam_states_verified']==215
        and persistence['independent_pre_save_state_sha256']==persistence['saved_state_sha256']
        and persistence['journal_prefix']==saved and persistence['saved_semantic_state_verified']['passed'] is True,
        'Full saved156600 PRE-SAVED semantic publication; fresh restore remains required')
    return {**value,'saved_journal_prefix':saved,'whole_journal_including_preserved_tail':stat,
        'preserved_unsaved_tail':tail,'preserved_tail_positive_updates':list(range(156601,156770))}



def checked_postrestore_selector_resume(pin,*,base,folder,pointer_pin):
    """Exact source26 metadata failure AFTER restore/prefix copy, BEFORE CPU/native."""
    import psutil
    require(pin==base['prior_resident_RAM_source26_postrestore_selector_failure'] and record(pin['path'])==pin
        and pin['sha256']=='0e36fb3035ff05bcf7e917a07d511b5a93145120c15756d175acb84f73d17942',
        'Exact actual source26 TOOL1 metadata closure, never normal0 or a225 admission failure')
    value=read_json(pin['path']);e=value['evidence']
    require(value['kind']=='ASR18_actual_source26_postrestore_predecessor_selector_startup_failure_v1'
        and value['owned_session']==24476 and value['failure_class']=='startup_metadata_wiring_not_resource_admission'
        and value['actual_optimizer_updates']==value['last_checkpoint']==156600
        and value['new_native_updates']==value['new_draws']==0 and value['CPU_or_CPU_launcher_started'] is False
        and value['native_update_or_pressure_gate_reached'] is False and value['training_stopped'] is True,
        'Genuine zero-update startup failure, not a resource or training retry')
    for item in e.values():require(record(item['path'])==item,'Exact compact source26 failure evidence')
    tool_pin=value['actual_TOOL'];require(record(tool_pin['path'])==tool_pin,'Exact consumed owned session outcome')
    tool=read_json(tool_pin['path'])
    require(tool['owned_session']==24476 and tool['actual_TOOL_exit_code']==1 and tool['actual_chunk_id']=='35b265'
        and tool['no_normal0_claim'] is True and tool['separate_native_OS_exit_code']=='UNKNOWN_NOT_PERSISTED',
        'Honest outer TOOL1 and unknown separate numeric native exit')
    old_source=ROOT/'analysis/BPE9223_training_launch_v1/resident_RAM_native_pressure_timeout156769_failure_actual_v1/inactive_additional_cushion225_v1/production_configuration_v1/configuration.json'
    restored=copy.deepcopy(base);del restored['prior_resident_RAM_source26_postrestore_selector_failure']
    require(restored==read_json(old_source),'ONLY new startup-failure provenance field; approved225 and all old values exact')
    require(e['pointer']==pointer_pin and e['driver']==record(P/'work/research/run_ASR18_BPE9223_resident_RAM_training_v26.py')
        and e['wrapper']==record(P/'work/research/run_ASR18_BPE9223_resident_RAM_autocontinue_v26.py')
        and e['failure']==record(folder/'invocations/owner_000019/actual_failure.json')
        and e['restore']==record(folder/'invocations/owner_000019/actual_parent_restore.json'),
        'Exact historical26 invocation19 and saved156600, not successor aliases')
    failure=read_json(e['failure']['path']);restore=read_json(e['restore']['path']);monitor=read_json(e['monitor_completion']['path'])
    require(failure==value['failure'] and failure['actual_optimizer_updates']==failure['last_checkpoint']==156600
        and failure['error']=="TypeError('expected str, bytes or os.PathLike object, not NoneType')"
        and 'line 820, in main' in failure['traceback'] and "'predecessor':record(" in failure['traceback']
        and restore==value['genuine_all_state_restore'] and restore['passed'] is True
        and restore['all207_threeAdam_all4RNG_match_actual_pre_save'] is True and restore['next_draw_packet_match'] is True
        and restore['restore_occurred_only_at_startup_before_native'] is True
        and monitor==value['original_monitor'] and monitor['samples']==409 and monitor['joined'] is True
        and monitor['clean'] is True and monitor['error'] is None,
        'Exact selector TypeError after genuine restore, clean original Monitor')
    expected=[('native',34604,1792725704.227828),('native_interpreter_launcher',40624,1792725704.2208068),
        ('wrapper',37676,1792725703.8532965)]
    require([(x['role'],x['pid'],x['create_time'])for x in value['known_runtime_exact_deaths']]==expected,
        'Only the three genuinely started runtime identities, CPU never created')
    for role,pid,ctime in expected:
        try:now=psutil.Process(pid).create_time();dead=abs(now-ctime)>=.001
        except psutil.NoSuchProcess:dead=True
        require(dead,'Exact source26 identity dead '+role)
    orphan=value['orphan19_saved_prefix_STAT'];old=value['original18_STAT_unchanged']
    require(orphan=={'path':str(folder/'journals/commits_000019.jsonl'),'bytes':317194933,
        'mtime_ns':1792725912000000000,'content_read':False}
        and old=={'path':str(folder/'journals/commits_000018.jsonl'),'bytes':317525825,
        'mtime_ns':1792718256000000000,'content_read':False},'Exact inactive orphan19 and historical whole18 STAT bindings')
    for item in (orphan,old):
        current=Path(item['path']).stat()
        require(current.st_size==item['bytes'] and current.st_mtime_ns==item['mtime_ns'],'Preserve orphan19 and original18 without promotion or reread')
    require(not(folder/'invocations/owner_000019/actual_journal_branch.json').exists()
        and not(folder/'invocations/owner_000019/CPU').exists()
        and not(folder/'invocations/owner_000019/normal_completion.json').exists()
        and Path(read_json(folder/'active_journal.json')['journal_prefix']['path'])==folder/'journals/commits_000018.jsonl',
        'Absent branch receipt/CPU/normal0 and unchanged authoritative active18')
    require(read_json(pointer_pin['path'])['committed_updates']==156600
        and read_json(folder/'latest_checkpoint.json')['pointer']==pointer_pin,
        'Still genuine fully saved156600; old169 attempts must be redone')
    return value


def checked_packet_credit_resume(pin,*,base,folder,pointer_pin):
    """Actual added-buffer timeout, clean original monitor, saved156400+137tail."""
    import psutil
    require(pin==base['prior_resident_RAM_native_pressure156537_failure'] and record(pin['path'])==pin
        and pin['sha256']=='7e17d0ebf376135c554f2a86b98779376cdba1cc7756d4b585a4372806a05fb9',
        'Exact source22 TOOL1 pressure timeout; never original floor failure or normal0')
    value=read_json(pin['path']);e=value['inputs']
    require(value['kind']=='ASR18_actual_saved156400_source22_native_preupdate_pressure_timeout_closure_v1'
        and value['owned_session']==35083 and value['actual_outer_TOOL_exit_code']==1
        and value['actual_outer_TOOL_chunk_id']=='dd5bb7' and value['actual_native_optimizer_updates']==156537
        and value['last_fully_saved_optimizer_updates']==156400 and value['unsaved_positive_tail_count']==137
        and value['native_numeric_OS_exit_code'] is None and value['CPU_numeric_OS_exit_code'] is None
        and value['wrapper_numeric_OS_exit_code_separately_recorded'] is None
        and value['no_unknown_numeric_OS_exit_invented'] is True,
        'Actual unknown native/CPU/wrapper OS codes; real outer1 and137historical positives')
    for item in e.values():require(record(item['path'])==item,'Exact compact source22 timeout evidence before owner writes')
    require(e['pointer']==pointer_pin and record(pointer_pin['path'])==pointer_pin
        and e['driver']==record(P/'work/research/run_ASR18_BPE9223_resident_RAM_training_v22.py')
        and e['CPU_protocol']==record(S/'fastmss_resident_feature_stream_v9.py')
        and e['effective_configuration']==record(folder/'invocations/owner_000017/configuration.json'),
        'Exact failed source22/CPU9 and genuine156400 checkpoint provenance')
    failure=read_json(e['actual_failure']['path']);monitor=read_json(e['monitor_completion']['path'])
    d=failure['native_pressure_denial'];m=d['last_denied_sample']['system_memory']
    require(failure==value['actual_failure'] and failure['actual_optimizer_updates']==156537
        and failure['last_checkpoint']==156400 and failure['error']=="TimeoutError('Bounded native pre-update commit-pressure wait exhausted; preserve state, no unchanged retry')"
        and d['observed_wait_seconds']==300.0 and d['observations']==600
        and d['required_commit_available_bytes']==4768435456 and d['required_physical_available_bytes']==1768435456
        and m['commit_available_bytes']==4676124672 and m['physical_available_bytes']==11188850688
        and d['no_packet_propose_native_or_authoritative_operation_admitted'] is True
        and monitor==value['monitor_completion'] and monitor['joined'] is True and monitor['clean'] is True and monitor['error'] is None,
        'Added-buffer scheduling timeout is distinct from original monitor/OOM/data failure')
    expected=[('native',12328,1792711453.1490774),('native_interpreter_launcher',38036,1792711453.1400745),
        ('wrapper',5628,1792711452.6819165),('CPU',28556,1792711652.8914447),('CPU_interpreter_launcher',37316,1792711652.8715985)]
    require([(x['role'],x['pid'],x['recorded_create_time'])for x in value['all_recorded_identities_dead']]==expected,'Exact five recorded predecessor identities')
    for role,pid,ctime in expected:
        try:now=psutil.Process(pid).create_time();dead=abs(now-ctime)>=.001
        except psutil.NoSuchProcess:dead=True
        require(dead,'Exact predecessor actually dead '+role)
    saved=value['saved_journal_prefix_as_verified_by_published_persistence'];stat=value['closed_journal_stat'];tail=value['preserved_unsaved_positive_tail']
    require(saved=={'path':str(folder/'journals/commits_000017.jsonl'),'offset':316802360,
        'sha256':'a5db7e40e8372c32c73b09b15ce87af7d3d9fed12bd8b1fede30f5fc3b461887'}
        and read_json(pointer_pin['path'])['journal_prefix']==saved and stat['path']==saved['path']
        and stat['bytes']==317071928 and tail['bytes']==269568
        and tail['sha256']=='e5bf113be5ea2947b3fbbcc29d9467804dadb9742b941ca8b2c50d1b754b1ec3'
        and record(tail['path'])==tail and saved['offset']+tail['bytes']==stat['bytes'],
        'Exact published savedprefix and retained137tail; no invented whole17hash')
    current=Path(stat['path']).stat()
    require(current.st_size==stat['bytes'] and current.st_mtime_ns==stat['mtime_ns'],'Closed original17 STAT unchanged')
    events=[json.loads(line)for line in Path(tail['path']).read_bytes().splitlines()]
    require(len(events)==137 and all(e['optimizer_updates']==e['native']['update']==e['prepared_evidence']['actual_positive_native_update']==u
        for u,e in zip(range(156401,156538),events)),'137 ordered original positive attempts remain historical')
    persistence=read_json(e['persistence']['path'])
    require(persistence['passed'] is True and persistence['optimizer_updates']==156400
        and persistence['named_tensors_verified']==854 and persistence['weights_verified']==207 and persistence['Adam_states_verified']==215
        and persistence['independent_pre_save_state_sha256']==persistence['saved_state_sha256']
        and persistence['journal_prefix']==saved and persistence['saved_semantic_state_verified']['passed'] is True,
        'Full saved156400854/207/215 PRE-SAVED and semantics; fresh restore still required')
    return {**value,'saved_journal_prefix':saved,'whole_journal_including_preserved_tail':stat,
        'preserved_unsaved_tail':tail,'preserved_tail_positive_updates':list(range(156401,156538))}


def check_original_closed_journal_stat(expected):
    observed=Path(expected['path']).stat()
    require(observed.st_size==expected['bytes'] and observed.st_mtime_ns==expected['mtime_ns'],
        'Original closed journal17 unchanged; no wholehash claimed or tail replay')
    return True


def checked_current_native_headroom_resume(pin,*,base,folder,pointer_pin):
    """Exact source20 abnormal resource stop; saved156200 and100 historical events."""
    import psutil
    require(pin==base['prior_resident_RAM_native_headroom156300_failure'] and record(pin['path'])==pin
        and pin['sha256']=='3f673aa583b6d4286ceef8a09deb85534a2b4ef45dba06e76237353eb31927c8',
        'Exact actual source20 system-commit failure, never normal0')
    value=read_json(pin['path']);e=value['inputs']
    require(value['kind']=='ASR18_actual_saved156200_source20_system_commit_failure_closure_v1'
        and value['owned_session']==47103 and value['actual_outer_TOOL_exit_code']==1
        and value['actual_outer_TOOL_chunk_id']=='3f03f5'
        and value['actual_native_optimizer_updates']==156300 and value['last_fully_saved_optimizer_updates']==156200
        and value['unsaved_positive_tail_count']==100 and value['native_numeric_OS_exit_code'] is None
        and value['CPU_numeric_OS_exit_code'] is None and value['wrapper_numeric_OS_exit_code_separately_recorded'] is None
        and value['no_unknown_numeric_OS_exit_invented'] is True,
        'Truthful TOOL1/current source20/unknown numeric exits and100 historical positives')
    for evidence in e.values():require(record(evidence['path'])==evidence,'Exact compact actual156300 failure evidence before new ownership writes')
    require(e['pointer']==pointer_pin and record(pointer_pin['path'])==pointer_pin
        and e['effective_configuration']==record(folder/'invocations/owner_000016/configuration.json'),
        'Exact saved156200 and source20 runtime provenance')
    boot=read_json(e['boot']['path'])
    require(boot['source']==record(P/'work/research/run_ASR18_BPE9223_resident_RAM_training_v20.py')
        and read_json(e['current_child']['path'])['invocation']==16,'Exact actual failed source20 invocation')
    failure=read_json(e['actual_failure']['path']);monitor=read_json(e['monitor_completion']['path'])
    sample=read_json(e['failing_headroom_sample']['path'])
    require(failure==value['actual_failure'] and failure['actual_optimizer_updates']==156300
        and failure['last_checkpoint']==156200 and 'Sensible real system headroom' in failure['error']
        and 'grouped_native_graph_boundary' in failure['traceback']
        and monitor==value['monitor_completion'] and monitor['joined'] is True and monitor['clean'] is False
        and 'Sensible real system headroom' in monitor['error']
        and sample==value['actual_failing_headroom_sample'] and sample['paired_sampling_within_one_second'] is True
        and sample['system_memory']['commit_available_bytes']==1943629824
        and sample['system_memory']['physical_available_bytes']==10373808128
        and sample['worker_private_memory']['private_bytes']==7415681024,
        'Actual captured commit-floor crossing and joined failed monitor, not EOF/normal0')
    expected=[('native',24824,1792682123.6010237),('native_interpreter_launcher',24240,1792682123.5941548),
        ('wrapper',41344,None),('CPU',32980,1792708037.735279),('CPU_interpreter_launcher',24936,None)]
    require([(x['role'],x['pid'],x['recorded_create_time'])for x in value['all_recorded_identities_dead']]==expected,
        'Exactly the five actually recorded identities with historical timestamp limitations')
    for role,pid,ctime in expected:
        try:now=psutil.Process(pid).create_time();dead=ctime is not None and abs(now-ctime)>=.001
        except psutil.NoSuchProcess:dead=True
        require(dead,'Recorded source20 predecessor identity actually dead '+role)
    saved=value['saved_prefix_verified_once'];whole=value['whole_journal_verified_once'];tail=value['preserved_unsaved_positive_tail']
    require(saved=={'path':str(folder/'journals/commits_000016.jsonl'),'offset':316408290,
        'sha256':'93fcf92d536874cd3d3f4f02adfd3ff318eec3cb85b455b32591aa4a3971f422'}
        and read_json(pointer_pin['path'])['journal_prefix']==saved
        and whole=={'path':saved['path'],'bytes':316605174,
            'sha256':'43d572ffa23c2c50e3c1c5818c0421921a2aa4400c8f74a12b4b1adcee6b9617'}
        and Path(whole['path']).stat().st_size==whole['bytes'],'Exact saved prefix and preserved whole old16 lineage')
    require(record(tail['path'])==tail and tail['bytes']==196884
        and tail['sha256']=='cf14b11497b1a93b0c5463547e8ff2ebdde7e3a8fca5411a346e1ae98378f7d7'
        and whole['bytes']-saved['offset']==tail['bytes'],'Exact100 historical tail bytes, no truncation/replay')
    events=[json.loads(line)for line in Path(tail['path']).read_bytes().splitlines()]
    require(len(events)==100 and all(x['optimizer_updates']==x['native']['update']==x['prepared_evidence']['actual_positive_native_update']==u
        for u,x in zip(range(156201,156301),events)),'All100 ordered positive attempts remain historical and require redo')
    persistence=read_json(e['persistence']['path'])
    require(persistence['passed'] is True and persistence['optimizer_updates']==156200
        and persistence['named_tensors_verified']==854 and persistence['weights_verified']==207
        and persistence['Adam_states_verified']==215 and persistence['journal_prefix']==saved
        and persistence['independent_pre_save_state_sha256']==persistence['saved_state_sha256'],
        'Actual all854/207/215 saved156200 PRE-SAVED equality, no new restore claim')
    return {**value,'saved_optimizer_updates':156200,'saved_journal_prefix':saved,
        'whole_journal_including_preserved_tail':whole,'preserved_unsaved_tail':tail,
        'preserved_tail_positive_updates':list(range(156201,156301))}


def checked_DEV_frontend_resume(pin,*,base,folder,pointer_pin):
    from fastmss_owned_process_closure_v2 import dead
    require(pin==base['prior_resident_RAM_DEV_frontend_failure'] and record(pin['path'])==pin,
        'Exact reviewed failed periodic frontend owner; no normal0/resource-stop relabel')
    value=read_json(pin['path'])
    require(value['kind']=='ASR18_actual_resident_RAM_periodic_DEV_frontend_failure_saved4000_v1'
        and value['actual_exec_tool_exit_code']==1 and value['normal_native0_claimed'] is False
        and value['intentional_stop_claimed'] is False and value['branch']==str(folder)
        and value['checkpoint_pointer']==pointer_pin
        and value['saved_optimizer_updates']==value['actual_optimizer_updates_before_failure']==4000
        and value['unsaved_tail_updates']==[] and value['periodic4000_still_required_before4001'] is True
        and value['actual_WER_report_produced'] is False and all(dead(w)for w in value['captured_identities']),
        'Actual failed first-row audio frontend/dead identities and coherent4000')
    for key in ('original_failure','original_monitor_completion','original_CPU_stderr','whole_journal',
                'checkpoint_pointer','checkpoint_persistence','original_native_log','source_configuration'):
        require(record(value[key]['path'])==value[key],'Preserved source3 failure/saved state/wholejournal')
    for pin in value['failed_results'].values():require(record(pin['path'])==pin,'Old partial DEV predictions/results untouched')
    require(read_json(pointer_pin['path'])['journal_prefix']==value['saved_journal_prefix']
        and value['whole_journal']=={'path':value['saved_journal_prefix']['path'],
            'bytes':value['saved_journal_prefix']['offset'],'sha256':value['saved_journal_prefix']['sha256']},
        'No unsaved tail and no new journal lineage/copy at saved4000')
    require(record(S/'asr18_resident_serial350_DEV_v2.py')==base['resident_DEV_helper_source'],
        'Exact new full-processor helper source')
    return value


def checked_original_interval_resume(pin,*,base,folder,pointer_pin):
    """Exact source19 EOF142894, saved142800, all94 positive attempts retained."""
    import psutil
    from datetime import datetime
    require(pin==base['prior_resident_RAM_original_interval_failure']and record(pin['path'])==pin
        and pin['sha256']=='d10f4b81cb390115bd502041ffee408acfec1c8ec8b28d748a2ac1b6fc6a3418',
        'Exact ROOT-bound actual source19 EOF failure; no normal0 relabel')
    value=read_json(pin['path']);e=value['inputs']
    require(value['kind']=='ASR18_actual_original_interval_post142894_failure_closed_v1'
        and value['owned_session']==1903 and value['actual_outer_TOOL_exit_code']==1 and value['actual_TOOL_chunk']=='1f7856'
        and value['actual_optimizer_updates']==142894 and value['last_fully_saved_updates']==142800
        and value['unsaved_positive_tail_updates']==94 and value['CPU_failure_completed_prepared_updates']==2694
        and value['native_numeric_OS_exit_code']is None and value['CPU_numeric_OS_exit_code']is None
        and value['native_Popen_wait_known_nonzero_from_wrapper_assertion']is True
        and value['all94_ordered_positive_events_verified']is True,
        'Truthful actual TOOL1/unknown OS codes; saved142800 and94 historical positives')
    require(e['pointer']==pointer_pin and Path(pointer_pin['path']).resolve().is_relative_to(folder/'checkpoints')
        and e['source19']==record(P/'work/research/run_ASR18_BPE9223_resident_RAM_training_v19.py')
        and e['wrapper19']==record(P/'work/research/run_ASR18_BPE9223_resident_RAM_autocontinue_v19.py')
        and e['CPU8_source']==record(S/'fastmss_resident_feature_stream_v8.py'),
        'Exact failed source19/wrapper19/CPU8 and original checkpoint branch')
    for item in e.values():require(record(item['path'])==item,'Exact compact frozen failed-owner/source/checkpoint evidence')
    death_pin=value['owned_process_deaths'];require(record(death_pin['path'])==death_pin,'Exact recorded seven-identity death observation')
    observed=read_json(death_pin['path'])
    expected=[{'role':'native','pid':37316,'create_time':1792674053.7525675,'ppid':41416},
        {'role':'native_interpreter_launcher','pid':41416,'create_time':1792674053.7455387,'ppid':41820},
        {'role':'CPU_producer','pid':38236,'create_time':1792674086.2008965,'ppid':23980},
        {'role':'CPU_interpreter_launcher','pid':23980,'historical_exact_create_time':None,'ppid':None},
        {'role':'wrapper','pid':41820,'creation_UTC':'2026-10-22T13:00:53.441969Z','ppid':41796},
        {'role':'wrapper_interpreter_launcher','pid':41796,'creation_UTC':'2026-10-22T13:00:53.432518Z','ppid':35016},
        {'role':'outer_tool_shell','pid':35016,'historical_exact_create_time':None,'ppid':None}]
    require([x['original_identity']for x in observed]==expected and all(x['actual_PID_absent']is True
        and x['current_if_any']is None and x['no_historical_timestamp_or_numeric_exit_invented']is True for x in observed),
        'Only exact seven recorded identities; three numeric/two UTC/two genuinely unknown historical timestamps')
    for who in expected:
        try:now=psutil.Process(who['pid']).create_time()
        except psutil.NoSuchProcess:continue
        ctime=who.get('create_time')
        if 'creation_UTC'in who:ctime=datetime.fromisoformat(who['creation_UTC'].replace('Z','+00:00')).timestamp()
        require(ctime is not None and abs(now-ctime)>=.001,'Original recorded identity actually dead '+who['role'])
    failure=read_json(e['actual_failure']['path']);monitor=read_json(e['monitor_completion']['path']);CPU=read_json(e['CPU_failure']['path'])
    require('Original interval beyond source unresolved'in failure['error']
        and 'Original interval beyond source unresolved'in Path(e['CPU_stderr']['path']).read_text(encoding='utf-8')
        and CPU['completed_updates']==2694 and failure['actual_optimizer_updates']==142894 and failure['last_checkpoint']==142800
        and monitor==value['monitor_clean_and_joined']and monitor['joined']is True and monitor['clean']is True and monitor['error']is None,
        'Actual ordered CPU source coverage failure; clean monitor, not a claimed absence of CPU allocation waits')
    require(value['captured_CPU_resource_telemetry_distinct_from_coverage_error']['prospective_commit_reserve_bytes']==2500000000,
        'Preserve source19 stronger prospective reservation without claiming freed memory or zero waits')
    remaining_pin=value['remaining_current_plan'];require(record(remaining_pin['path'])==remaining_pin,'Exact ordered remaining-plan metadata')
    remaining=read_json(remaining_pin['path']);plan_pin=value['current_plan'];future_pin=value['future_private_plan']
    require(record(plan_pin['path'])==plan_pin and record(future_pin['path'])==future_pin,'Exact current and future private plan bytes')
    plan=read_json(plan_pin['path']);future=read_json(future_pin['path'])
    require(plan['initial_committed_updates']==142880 and len(plan['updates'])==20
        and future['initial_committed_updates']==142900 and len(future['updates'])==20
        and remaining=={'source_plan':plan_pin,'initial_updates':142880,'first_unconsumed_update_index':14,
            'candidate_global_updates':142895,'not_assumed_native_plus_one':True,'CPU_completed_updates':2694,
            'CPU_initial_frontier':140200,'future_plan':future_pin,'updates':plan['updates'][14:]},
        'Exact nine-key remaining schema and actual CPU localization, never native+1 inference')
    saved=value['saved_prefix_verified_once'];whole=value['whole_journal_verified_once']
    require(saved=={'path':str(folder/'journals/commits_000015.jsonl'),'offset':290088487,
        'sha256':'e45e1adb098b5c1fdf4388116c4db6a61feb348320a1f396f0bdb73f3cbb4598'}
        and read_json(pointer_pin['path'])['journal_prefix']==saved
        and whole=={'path':saved['path'],'bytes':290273015,'sha256':'1c69569994392387686b823c2bbec97ed069aa07f2ffc2677c5c410b5cb90f0d'}
        and Path(whole['path']).stat().st_size==whole['bytes'],'Exact saved142800 prefix and preserved whole failed lineage')
    tail=value['preserved_unsaved_positive_tail']
    require(record(tail['path'])==tail and whole['bytes']-saved['offset']==tail['bytes']==184528,
        'Exact retained94-event suffix; never truncate, discard or replay historical tail')
    events=[json.loads(line)for line in Path(tail['path']).read_bytes().splitlines()]
    require(len(events)==94 and all(x['optimizer_updates']==x['native']['update']==x['prepared_evidence']['actual_positive_native_update']==u
        for u,x in zip(range(142801,142895),events)),'All94 ordered positive attempts remain historical')
    persistence=read_json(e['persistence']['path'])
    require(persistence['passed']and persistence['optimizer_updates']==142800 and persistence['weights_verified']==207
        and persistence['Adam_states_verified']==215 and persistence['named_tensors_verified']==854
        and persistence['journal_prefix']==saved and persistence['independent_pre_save_state_sha256']==persistence['saved_state_sha256'],
        'Actual fully verified854/207/215 saved142800, not a new GPU restore claim')
    return {**value,'saved_optimizer_updates':142800,'saved_journal_prefix':saved,
        'whole_journal_including_preserved_tail':whole,'preserved_unsaved_tail':tail,
        'preserved_tail_positive_updates':list(range(142801,142895))}



def checked_DEV_admission_resume(pin,*,base,folder,pointer_pin):
    import psutil
    require(pin==base['prior_resident_RAM_DEV_admission_failure'] and record(pin['path'])==pin
        and pin['sha256']=='89788d67c857c84c74216f90eef72c891a18eb7a216514a68eecf0e4872bee21',
        'Exact actual136000 DEV prospective admission failure, not normal0')
    value=read_json(pin['path']);inputs=value['inputs'];failure=value['actual_failure']
    require(value['kind']=='ASR18_actual_saved136000_resident_DEV_prospective_admission_failure_closure_v1'
        and value['owned_session']==64681 and value['actual_outer_TOOL_exit_code']==1
        and value['actual_committed_and_fully_saved_updates']==136000 and value['unsaved_positive_tail_updates']==0
        and value['pending136000_DEV_not_completed'] is True and value['pending136000_DEV_must_precede136001'] is True
        and value['native_numeric_OS_exit_code'] is None and value['CPU_numeric_OS_exit_code'] is None,
        'Truthful abnormal wait1 with unknown numeric native exit and no136k DEV result')
    for evidence in inputs.values():require(record(evidence['path'])==evidence,'Exact compact actual136000 evidence')
    require(failure==read_json(inputs['actual_failure']['path'])
        and failure['error']=="ValueError('Actual allocation increment plus full extra prefetch reservation and unchanged commit headroom')"
        and failure['actual_optimizer_updates']==failure['last_checkpoint']==136000
        and "admit('resident_optimizer_idle_serial350_DEV',2100000000)" in failure['traceback'],
        'Exact unchanged prospective DEV guard, not an audio or normal stop relabel')
    expected=[('native',27232,1792655703.93878),('native_interpreter_launcher',25192,1792655703.9323223),
        ('wrapper',35268,None),('CPU',548,1792655894.7857172),('CPU_interpreter_launcher',35232,None)]
    require([(v['role'],v['pid'],v['recorded_create_time'])for v in value['all_recorded_identities_dead']]==expected,
        'Exact old identities; never invent historical create-times')
    for role,pid,ctime in expected:
        try:current=psutil.Process(pid).create_time();dead=ctime is not None and abs(current-ctime)>.001
        except psutil.NoSuchProcess:dead=True
        require(dead,'Actual recorded136000 predecessor identity dead '+role)
    require(pointer_pin==inputs['pointer'] and record(pointer_pin['path'])==pointer_pin,'Exact fully verified saved136000 pointer')
    pointer=read_json(pointer_pin['path']);saved=pointer['journal_prefix'];whole=value['whole_journal_verified_once']
    require(pointer['committed_updates']==136000 and saved=={'path':str(folder/'journals/commits_000012.jsonl'),
        'offset':276724922,'sha256':'40bce7c886d3e053543e6bd0c4ec6c111c5b6f29aab13dcc1beab50443010b6e'}
        and whole=={'path':saved['path'],'bytes':saved['offset'],'sha256':saved['sha256']}
        and Path(saved['path']).stat().st_size==saved['offset'],'Exact whole saved lineage; no tail, truncation or branch copy')
    persistence=read_json(inputs['persistence']['path']);monitor=read_json(inputs['monitor_completion']['path'])
    require(persistence['passed'] and persistence['named_tensors_verified']==854 and persistence['weights_verified']==207
        and persistence['Adam_states_verified']==215 and persistence['journal_prefix']==saved
        and persistence['independent_pre_save_state_sha256']==persistence['saved_state_sha256']
        and monitor['joined'] is True and monitor['clean'] is True and monitor['error'] is None,
        'Actual saved all-state metadata and joined clean monitor')
    return {'saved_journal_prefix':saved,'whole_journal':whole,'actual_failure':pin,
        'pending_periodic_DEV_updates':136000,'unsaved_tail_bytes':0}


def checked_periodic_quiescence_resume(pin,*,base,folder,pointer_pin):
    import psutil
    require(pin==base['prior_resident_RAM_periodic_quiescence_failure'] and record(pin['path'])==pin
        and pin['sha256']=='3c5d30512e62a77c846ec5ad570f88addea46cfc58498b2caa38e05274596ac9',
        'Exact actual140000 private-plan equality failure, never normal0')
    value=read_json(pin['path']);inputs=value['inputs'];failure=value['actual_failure']
    require(value['kind']=='ASR18_actual_saved140000_periodic_CPU_quiescence_plan_identity_failure_closure_v1'
        and value['owned_session']==86052 and value['actual_outer_TOOL_exit_code']==1
        and value['all_committed_and_fully_saved_updates']==140000 and value['unsaved_positive_tail_updates']==0
        and value['pending140000_DEV_not_completed'] is True and value['pending140000_DEV_must_precede140001'] is True
        and value['native_numeric_OS_exit_code'] is None and value['CPU_numeric_OS_exit_code'] is None
        and value['real_periodic_CPU_quiescence_wait0_not_claimed'] is True,
        'Preserve actual TOOL1 and uncompleted140k DEV/CPUclose, no numeric exit invention')
    for evidence in inputs.values():require(record(evidence['path'])==evidence,'Exact compact140000 failure evidence')
    require(failure==read_json(inputs['actual_failure']['path'])
        and failure['error']=="ValueError('Keep the exact previously submitted private plan')"
        and failure['actual_optimizer_updates']==failure['last_checkpoint']==140000
        and 'close_private_preparation_for_DEV' in failure['traceback'],
        'Actual original helper1 failed before any packet drain, not a memory or EOF relabel')
    expected=[('native',41200,1792660196.3122573),('native_interpreter_launcher',38908,1792660196.3051417),
        ('wrapper',31740,None),('CPU',26908,1792660472.4757633),('CPU_interpreter_launcher',33564,None)]
    require([(x['role'],x['pid'],x['recorded_create_time'])for x in value['all_recorded_identities_dead']]==expected,
        'Exact140000 predecessor identities and recorded timestamp limitations')
    for role,pid,ctime in expected:
        try:current=psutil.Process(pid).create_time();dead=ctime is not None and abs(current-ctime)>.001
        except psutil.NoSuchProcess:dead=True
        require(dead,'Recorded140000 predecessor actually dead '+role)
    require(pointer_pin==inputs['pointer'] and record(pointer_pin['path'])==pointer_pin,'Exact fully saved140000 pointer')
    pointer=read_json(pointer_pin['path']);saved=pointer['journal_prefix'];whole=value['whole_journal_verified_once']
    require(pointer['committed_updates']==140000 and saved=={'path':str(folder/'journals/commits_000012.jsonl'),
        'offset':284584907,'sha256':'caeb1f6fe40c867028bff40b0aa7f838a7a9d185872855213adc4d2a75d697af'}
        and whole=={'path':saved['path'],'bytes':saved['offset'],'sha256':saved['sha256']}
        and Path(saved['path']).stat().st_size==saved['offset'],'Same whole saved140000 journal, no unsaved tail/copy/truncation')
    persistence=read_json(inputs['persistence']['path']);monitor=read_json(inputs['monitor_completion']['path'])
    require(persistence['passed'] and persistence['named_tensors_verified']==854 and persistence['weights_verified']==207
        and persistence['Adam_states_verified']==215 and persistence['journal_prefix']==saved
        and persistence['independent_pre_save_state_sha256']==persistence['saved_state_sha256']
        and monitor['joined'] and monitor['clean'] and monitor['error'] is None,'Actual all-state saved metadata and clean monitor')
    current=read_json(inputs['last_consumed_plan']['path']);future=read_json(inputs['next_unconsumed_plan']['path'])
    require(current['initial_committed_updates']==139980 and future['initial_committed_updates']==140000
        and len(current['updates'])==len(future['updates'])==20,'Exact current200 and unconsumed successor201 provenance')
    return {'saved_journal_prefix':saved,'whole_journal':whole,'actual_failure':pin,
        'pending_periodic_DEV_updates':140000,'unsaved_tail_bytes':0}


def private_successor_request(state,*,completed_boundaries,validation_stop,budget_stop):
    """Original MainThread pass bounds, plus the unchanged periodic stop."""
    if validation_stop or budget_stop:return None
    from asr18_BPE9223_lifecycle_v3 import next_boundary_request
    request=next_boundary_request(state,completed_boundaries)
    if request['kind']!='bounded_training_window':return None
    remaining=4000-request['initial_optimizer_updates']%4000
    return {'maximum_updates':min(request['maximum_updates'],remaining),
        'before_sampler_sha256':digest(state),'phase':request['phase']}


def adopt_private_successor(prefetch,ticket,schedule,submit_plan,*,wait,timings=None):
    """Only ordered full-row ingress changes; original planning stays MainThread."""
    if not wait and not prefetch.ready():
        if timings is not None:timings['not_ready_polls']+=1
        return None
    began=time.monotonic();prepared=prefetch.take(ticket,wait=wait);taken=time.monotonic()
    if timings is not None:
        timings.update(worker_preparation_seconds=prepared['preparation_seconds'],worker_metadata_bytes=prepared['metadata_bytes'],
            main_take_wait_seconds=taken-began,end_window_wait_count=int(wait))
    try:
        with prefetch.planning_rows(prepared,schedule):
            queued=submit_plan(schedule)
        require(queued[0]['initial_sampler_sha256']==digest(schedule.state_dict()),
            'Prepared successor begins at the exact private future state')
        if timings is not None:timings['main_adoption_and_submission_seconds']=time.monotonic()-taken
        return queued
    finally:del prepared


class Monitor:
    def __init__(self,folder,*,additional_reserved_host_bytes=0):
        require(type(additional_reserved_host_bytes)is int and additional_reserved_host_bytes>=0,
            'Explicit separately charged extra host bookkeeping, no original guard reduction')
        self.additional_reserved_host_bytes=additional_reserved_host_bytes
        self.folder=folder;self.stop=threading.Event();self.error=None;self.samples=0;self.peak_private=0
        self.thread=threading.Thread(target=self._run,daemon=True);self.thread.start()
    def _observe(self):
        from resource_watchdog_v2 import sample_memory
        s=sample_memory(os.getpid(),'resident_RAM_system_headroom');self.samples+=1
        require(s['paired_sampling_within_one_second'],'Fresh actual process/system memory')
        self.peak_private=max(self.peak_private,s['worker_private_memory']['private_bytes'])
        m=s['system_memory']
        if m['commit_available_bytes']<2000000000+self.additional_reserved_host_bytes or m['physical_available_bytes']<1500000000+self.additional_reserved_host_bytes:
            write_json(self.folder/'actual_failing_headroom_sample.json',s)
        require(m['commit_available_bytes']>=2000000000+self.additional_reserved_host_bytes and m['physical_available_bytes']>=1500000000+self.additional_reserved_host_bytes,
            'Sensible real system headroom; no obsolete2GB CPU-family or90percent prospective ceiling')
        return s
    def _run(self):
        try:
            while not self.stop.wait(.5):self._observe()
        except BaseException as e:self.error=e
    def check(self):
        if self.error is not None:raise RuntimeError('Actual resident memory monitor: '+repr(self.error))
        require(not (self.folder/'stop.json').exists(),'Actual owner cooperative stop')
    def close(self):
        self.stop.set();self.thread.join(timeout=3);require(not self.thread.is_alive(),'Actual RAM monitor joined')
        write_json(self.folder/'monitor_completion.json',{'samples':self.samples,'maximum_sampled_private_bytes':self.peak_private,
            'joined':True,'clean':self.error is None,'error':None if self.error is None else repr(self.error)})


def main(a):
    import psutil
    from asr18_memory_diagnosis_v2 import storage
    storage();require(sys.flags.utf8_mode==1 and os.environ.get('PYTHONUTF8')=='1','UTF8 before source reads')
    def E_reserve(prospective=0):
        require(shutil.disk_usage(ROOT).free>=100000000000+prospective,'E100GB reserve plus actual prospective output')
    E_reserve(64*1024**2)
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0','One explicitly visible training GPU')
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    folder=Path(a.folder).resolve();require(folder.is_relative_to(ROOT/'runs'),'Exact E continuation branch')
    lease_path=P/'work/gpu_training_owner.json';lease=read_json(lease_path)
    require(lease['owner']==OWNER and lease['actual_GPU_running'] is False and lease['actual_owned_worker_pid'] is None,
        'Exclusive idle training lease before resident GPU owner')
    producer={'pid':os.getpid(),'create_time':psutil.Process().create_time(),'ppid':os.getppid()}
    candidate_base=read_json(a.configuration)
    from asr18_reader9_recovery276000_transition_v8 import (checked_current_plan_throughput_policy,
        checked_current_plan_throughput_checkpoint_resume,legacy_source37_configuration,EXTRA_HOST_BYTES)
    throughput_active='authoritative_current_plan_identity_and_main_path_attribution_policy'in candidate_base
    throughput_reserve=EXTRA_HOST_BYTES if throughput_active else 0
    if throughput_active:
        OriginalSourceAuthority,current_plan_identity_view,MainPathScalarAttribution,throughput_pressure_wait=checked_current_plan_throughput_policy(candidate_base)
    base=legacy_source37_configuration(candidate_base)if throughput_active else candidate_base
    pointer_pin=record(a.pointer);pointer=read_json(a.pointer)
    throughput_transition=None if a.throughput_resume is None else checked_current_plan_throughput_checkpoint_resume(
        record(a.throughput_resume),base=candidate_base,folder=folder,pointer_pin=pointer_pin)
    if throughput_transition is not None:
        require(throughput_active and a.invocation==32 and all(x is None for x in (a.prior_process,a.ninth_original_resume,
            a.gc_refresh_resume,a.eighth_original_resume,a.fresh_signature_resume,a.canonical_archive_resume,a.lookahead_resume,
            a.cushion225_resume,a.postrestore_selector_resume,a.packet_credit_resume,a.native_headroom_resume,a.abnormal_resume,
            a.source_transition_resume,a.original_interval_resume,a.DEV_frontend_resume,a.DEV_admission_resume,a.periodic_quiescence_resume)),
            'Only exact accepted abnormal source39 saved212000 startup failure, never normal0 or an old failure lane')
    initial_updates=pointer['committed_updates']
    prefetch_policy=base['private_scalar_prefetch_policy']
    require(prefetch_policy=={'kind':'ASR18_private_scalar_row_prefetch_policy_v1',
        'source':record(S/'asr18_BPE9223_private_scalar_prefetch_v1.py'),
        'actual_CPU_control_review':record(ROOT/'analysis/BPE9223_training_launch_v1/ROOT_private_scalar_prefetch_actual_CPU_control_review_v1.json'),
        'additional_host_reservation_bytes':268435456,'maximum_TR_rows':280,'maximum_canonical_row_bytes':8388608,
        'SQLite_cache_bytes_each':4194304,'supported_phase':'weak_label_continued_training',
        'original_MainThread_planning_and_commits_unchanged':True,'idle_during_DEV':True},
        'Exact inactive-reviewed scalar prefetch source/resource policy; production still needs execution review')
    periodic_policy=base['periodic_DEV_policy']
    require(periodic_policy=={'kind':'ASR18_periodic_fixed350_DEV_monitor_policy_v1','interval_updates':4000,'first_update':4000,
        'scorer_policy':base['full_run_identity']['scorer_policy'],'periodic_result_path':str(folder/'periodic_DEV.json'),
        'full_pass_boundaries_and_winner_policy_unchanged':True},'Exact human-authorized periodic4000 fixed350 monitor policy')
    require(a.queue_bytes==268435456,'Unchanged256MiB queue retains every full admissible256MiB packet')
    from fastmss_resident_feature_stream_v14 import checked_resource_profile
    checked_resource_profile(base)
    native_pressure_policy,wait_for_native_commit_pressure=checked_native_commit_pressure_policy(base)
    from asr18_ninth_original_interval_checkpoint_resume_v1 import (checked_ninth_original_policy,checked_ninth_original_resume,
        legacy_source36_policy_configuration,inherit_consumed_postDEV_opportunity)
    ninth_policy_active='prior_resident_RAM_ninth_original_interval_failure'in base
    if ninth_policy_active:checked_ninth_original_policy(base)
    original_gc_policy_base=legacy_source36_policy_configuration(base)if ninth_policy_active else base
    ninth_interval_failure=None if a.ninth_original_resume is None else checked_ninth_original_resume(
        record(a.ninth_original_resume),base=base,folder=folder,pointer_pin=pointer_pin)
    if ninth_interval_failure is not None:
        require(a.invocation==26 and all(x is None for x in (a.prior_process,a.gc_refresh_resume,a.eighth_original_resume,
            a.fresh_signature_resume,a.canonical_archive_resume,a.lookahead_resume,a.cushion225_resume,
            a.postrestore_selector_resume,a.packet_credit_resume,a.native_headroom_resume,a.abnormal_resume,
            a.source_transition_resume,a.original_interval_resume,a.DEV_frontend_resume,a.DEV_admission_resume,
            a.periodic_quiescence_resume)),
            'Only exact source36 original interval186563 saved186400 ninth failure, no historical or normal0 alias')
    from asr18_one_postDEV_gc_refresh_checkpoint_transition_v2 import (checked_one_postDEV_refresh_policy,
        checked_one_postDEV_refresh_checkpoint_resume,legacy_source33_configuration)
    OnePostDEVRefresh,capture_idle_state,compare_idle_state=checked_one_postDEV_refresh_policy(original_gc_policy_base)
    legacy_base=legacy_source33_configuration(original_gc_policy_base)
    gc_transition=None if a.gc_refresh_resume is None else checked_one_postDEV_refresh_checkpoint_resume(
        record(a.gc_refresh_resume),base=base,folder=folder,pointer_pin=pointer_pin)
    require(throughput_transition is not None or ninth_interval_failure is not None or gc_transition is not None or a.prior_process is not None,
        'New GC policy enters only exact actual source33 intentional stop or genuine later normal0')
    if gc_transition is not None:
        require(a.invocation==25 and all(x is None for x in (a.prior_process,a.eighth_original_resume,a.fresh_signature_resume,
            a.canonical_archive_resume,a.lookahead_resume,a.cushion225_resume,a.postrestore_selector_resume,a.packet_credit_resume,
            a.native_headroom_resume,a.abnormal_resume,a.original_interval_resume,a.source_transition_resume,
            a.DEV_frontend_resume,a.DEV_admission_resume,a.periodic_quiescence_resume)),
            'Only actual source33 intentional checkpoint stop, no historical or normal0 alias')
    if str(Path('E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1/resident_RAM_original_interval170469_failure_actual_v1'))not in sys.path:sys.path.insert(0,str(Path('E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1/resident_RAM_original_interval170469_failure_actual_v1')))
    from asr18_eighth_original_interval_checkpoint_resume_v1 import checked_eighth_original_policy,checked_eighth_original_resume
    checked_eighth_original_policy(legacy_base)
    eighth_interval_failure=None if a.eighth_original_resume is None else checked_eighth_original_resume(
        record(a.eighth_original_resume),base=base,folder=folder,pointer_pin=pointer_pin)
    if eighth_interval_failure is not None:
        require(a.invocation==24 and all(x is None for x in (a.prior_process,a.fresh_signature_resume,a.canonical_archive_resume,
            a.lookahead_resume,a.cushion225_resume,a.postrestore_selector_resume,a.packet_credit_resume,a.native_headroom_resume,
            a.abnormal_resume,a.source_transition_resume,a.original_interval_resume,a.DEV_frontend_resume,a.DEV_admission_resume,
            a.periodic_quiescence_resume)), 'Only exact source31 interval TOOL1 saved170400, never normal0 or historical lane')
    from asr18_fresh_signature_collect_checkpoint_transition_v1 import checked_fresh_signature_collect_policy,checked_fresh_signature_collect_checkpoint_resume
    FreshMetadataSignatureScope,ObserveOriginalCollectScope=checked_fresh_signature_collect_policy(base)
    signature_transition=None if a.fresh_signature_resume is None else checked_fresh_signature_collect_checkpoint_resume(
        record(a.fresh_signature_resume),base=base,folder=folder,pointer_pin=pointer_pin)
    if signature_transition is not None:
        require(a.invocation==23 and all(x is None for x in (a.prior_process,a.canonical_archive_resume,a.lookahead_resume,a.cushion225_resume,
            a.postrestore_selector_resume,a.packet_credit_resume,a.native_headroom_resume,a.abnormal_resume,
            a.source_transition_resume,a.original_interval_resume,a.DEV_frontend_resume,a.DEV_admission_resume,a.periodic_quiescence_resume)),
            'Only exact actual source30 fresh-signature checkpoint stop, never historical or normal0')
    from asr18_canonical_archive_checkpoint_transition_v1 import checked_archive_representation_policy,checked_canonical_archive_checkpoint_resume
    archive_codec=checked_archive_representation_policy(base)
    canonical_transition=None if a.canonical_archive_resume is None else checked_canonical_archive_checkpoint_resume(
        record(a.canonical_archive_resume),base=base,folder=folder,pointer_pin=pointer_pin)
    if canonical_transition is not None:
        require(a.invocation==22 and all(x is None for x in (a.prior_process,a.lookahead_resume,a.cushion225_resume,
            a.postrestore_selector_resume,a.packet_credit_resume,a.native_headroom_resume,a.abnormal_resume,
            a.source_transition_resume,a.original_interval_resume,a.DEV_frontend_resume,a.DEV_admission_resume,
            a.periodic_quiescence_resume)),
            'Only exact reviewed actual source29 archive transition, never historical or normal0')
    from asr18_resident_periodic_CPU_quiescence_v3 import close_private_preparation_for_DEV
    require(base['periodic_DEV_CPU_quiescence_policy']=={
        'kind':'ASR18_periodic_DEV_private_CPU_quiescence_policy_v1',
        'source':record(S/'asr18_resident_periodic_CPU_quiescence_v3.py'),
        'whole_unconsumed_private_successor_only':True,'same_plan_resubmitted_after_DEV':True,
        'CPU_close_requires_actual_wait0_and_joined_handles':True,
        'DEV_incremental_allowance_bytes':2100000000,'additional_prefetch_reservation_bytes':268435456,
        'commit_headroom_bytes':2000000000,'physical_headroom_bytes':1500000000},
        'Only changed phase scheduling; preserve every allocation and actual headroom guard')
    require(a.abnormal_resume is None or a.native_headroom_resume is None,'One truthful abnormal resource lane')
    require(sum(x is not None for x in (a.cushion225_resume,a.packet_credit_resume,a.native_headroom_resume,a.abnormal_resume))<=1,'Only one typed abnormal resource lane')
    cushion225_failure=None if a.cushion225_resume is None else checked_cushion225_resume(
        record(a.cushion225_resume),base=base,folder=folder,pointer_pin=pointer_pin)
    require(a.postrestore_selector_resume is None or (a.cushion225_resume is not None and a.prior_process is None),
        'Startup metadata repair retains original approved225 abnormal lane; never normal0')
    if a.postrestore_selector_resume is not None:
        checked_postrestore_selector_resume(record(a.postrestore_selector_resume),base=base,folder=folder,pointer_pin=pointer_pin)
    packet_credit_failure=None if a.packet_credit_resume is None else checked_packet_credit_resume(
        record(a.packet_credit_resume),base=base,folder=folder,pointer_pin=pointer_pin)
    abnormal=(checked_current_native_headroom_resume(record(a.native_headroom_resume),base=base,folder=folder,pointer_pin=pointer_pin)
        if a.native_headroom_resume is not None else(None if a.abnormal_resume is None else checked_headroom_resume(record(a.abnormal_resume),base=base,folder=folder,pointer_pin=pointer_pin)))
    if packet_credit_failure is not None:abnormal=packet_credit_failure
    if cushion225_failure is not None:abnormal=cushion225_failure
    frontend_failure=None if a.DEV_frontend_resume is None else checked_DEV_frontend_resume(record(a.DEV_frontend_resume),base=base,folder=folder,pointer_pin=pointer_pin)
    from asr18_approved_lookahead_checkpoint_transition_v3 import checked_lookahead_checkpoint_resume
    lookahead_transition=None if a.lookahead_resume is None else checked_lookahead_checkpoint_resume(
        record(a.lookahead_resume),base=base,folder=folder,pointer_pin=pointer_pin)
    if lookahead_transition is not None:
        require(a.invocation==21 and all(x is None for x in (a.prior_process,a.cushion225_resume,
            a.postrestore_selector_resume,a.packet_credit_resume,a.native_headroom_resume,a.abnormal_resume,
            a.source_transition_resume,a.original_interval_resume,a.DEV_frontend_resume,a.DEV_admission_resume,
            a.periodic_quiescence_resume)), 'Only the exact approved actual source27 stop, never historical or normal0')
    from asr18_resident_source_transition_v2 import checked_source_transition_resume
    legacy_transition=gc_transition if gc_transition is not None else(signature_transition if signature_transition is not None else(canonical_transition if canonical_transition is not None else(lookahead_transition if lookahead_transition is not None else(None if a.source_transition_resume is None else checked_source_transition_resume(
        record(a.source_transition_resume),base=base,folder=folder,pointer_pin=pointer_pin)))))
    transition=throughput_transition if throughput_transition is not None else legacy_transition
    interval_failure=ninth_interval_failure if ninth_interval_failure is not None else(eighth_interval_failure if eighth_interval_failure is not None else(None if a.original_interval_resume is None else checked_original_interval_resume(
        record(a.original_interval_resume),base=base,folder=folder,pointer_pin=pointer_pin)))
    DEV_admission_failure=None if a.DEV_admission_resume is None else checked_DEV_admission_resume(
        record(a.DEV_admission_resume),base=base,folder=folder,pointer_pin=pointer_pin)
    periodic_quiescence_failure=None if a.periodic_quiescence_resume is None else checked_periodic_quiescence_resume(
        record(a.periodic_quiescence_resume),base=base,folder=folder,pointer_pin=pointer_pin)
    require(sum(x is not None for x in (abnormal,frontend_failure,transition,interval_failure,DEV_admission_failure,periodic_quiescence_failure))<=1,'One truthful predecessor lane')
    preserved_tail=throughput_transition if throughput_transition is not None else(gc_transition if gc_transition is not None else(signature_transition if signature_transition is not None else(canonical_transition if canonical_transition is not None else(lookahead_transition if lookahead_transition is not None else(interval_failure if interval_failure is not None else(abnormal if abnormal is not None else(transition if transition is not None and transition['unsaved_tail_bytes'] else None)))))))
    require(initial_updates>=260 and pointer['all_window_updates_committed'] is True
        and pointer['diagnostic_only'] is False
        and pointer['identity']['caller_identity']['full_run_lifecycle_identity']==base['full_run_identity'],
        'Exact coherent full-state source, no reset or diagnostic adoption')
    failed=read_json(a.latest_failure)
    require(failed['kind']=='ASR18_actual_ACKed260_singlepass_control_namespace_failure_TOOL1_v1'
        and failed['actual_tool_exit_code']==1 and failed['new_training_GPU_children']==failed['new_CPU_children']==0
        and failed['durable_ACK_committed'] is True,
        'Real last failed owner before any new CPU/GPU update; preserve genuine ACK260')
    from fastmss_owned_process_closure_v2 import dead
    require(dead(failed['producer']),'Actual previous parent dead')
    if not folder.exists():
        require(initial_updates==260 and failed['checkpoint_pointer']==pointer_pin,'Fresh branch only from genuine260')
        folder.mkdir();(folder/'checkpoints').mkdir();(folder/'progress').mkdir();(folder/'invocations').mkdir()
        write_json(folder/'owner.json',{'kind':'ASR18_exact_saved260_resident_RAM_branch_v1','source_checkpoint':pointer_pin,
            'identity':base['full_run_identity'],'original_RUN07_failure':record(a.latest_failure),'source':record(__file__)})
    else:
        owner=read_json(folder/'owner.json')
        require(owner['kind']=='ASR18_exact_saved260_resident_RAM_branch_v1' and owner['identity']==base['full_run_identity']
            and Path(pointer_pin['path']).resolve().is_relative_to(folder/'checkpoints')
            and (a.prior_process is not None or abnormal is not None or frontend_failure is not None or transition is not None or interval_failure is not None or DEV_admission_failure is not None or periodic_quiescence_failure is not None),
            'Existing branch resumes its exact coherent RAM checkpoint and original scientific identity')
        if abnormal is None and frontend_failure is None and transition is None and interval_failure is None and DEV_admission_failure is None and periodic_quiescence_failure is None:
            prior=read_json(a.prior_process);report=read_json(prior['worker_report']['path'])
            require(record(prior['worker_report']['path'])==prior['worker_report'] and prior['actual_Popen_wait_code']==0
                and dead(prior['producer']) and report['latest_checkpoint']['pointer']==pointer_pin
                and report['all_live_updates_saved'] is True and report['actual_optimizer_updates']==initial_updates,
                'Genuine previous native normal wait0/death and exact latest all-state checkpoint')
        elif abnormal is not None:require(a.prior_process is None and a.invocation==((20 if a.postrestore_selector_resume is not None else 19) if a.cushion225_resume is not None else(18 if a.packet_credit_resume is not None else(17 if a.native_headroom_resume is not None else 11))),'Actual resource failure is not a normal0 predecessor')
        elif frontend_failure is not None:require(a.prior_process is None and a.invocation==5,'Frontend failure is not a normal0 predecessor')
        elif interval_failure is not None:require(a.prior_process is None and a.invocation==(26 if a.ninth_original_resume is not None else(24 if a.eighth_original_resume is not None else 16)),'Actual original-interval TOOL1 is not normal0/intentional stop')
        elif DEV_admission_failure is not None:require(a.prior_process is None and a.invocation==13,'Actual DEV prospective admission TOOL1 is never normal0')
        elif periodic_quiescence_failure is not None:require(a.prior_process is None and a.invocation==14,'Actual helper1 equality failure is never normal0')
        else:require(a.prior_process is None and a.invocation>5,'Intentional TOOL1 source transition is never normal0')
    invocation=folder/'invocations'/('owner_'+str(a.invocation).zfill(6))
    require(not invocation.exists(),'Unique actual native invocation; preserve prior controls');invocation.mkdir();(invocation/'plans').mkdir()
    source_journal=pointer['journal_prefix']
    if abnormal is None and transition is None and interval_failure is None:
        require(record(source_journal['path'])=={'path':source_journal['path'],'bytes':source_journal['offset'],
            'sha256':source_journal['sha256']},'Exact whole coherent saved journal')
    else:
        require(source_journal==(interval_failure if interval_failure is not None else(abnormal if abnormal is not None else transition))['saved_journal_prefix'],'Exact reviewed saved prefix')
        SavedJournalPrefixView(source_journal,lambda:None)
    c=copy.deepcopy(candidate_base);c.update(worker_driver_source=record(__file__),parent_driver_source=record(__file__),
        resident_RAM_stream_configuration=record(a.stream_configuration),resident_branch_namespace=str(folder),
        checkpoint_interval_updates=200,source_checkpoint=pointer_pin,source_configuration=record(a.configuration))
    config_path=invocation/'configuration.json';write_json(config_path,c);c['_path']=str(config_path)
    write_json(invocation/'boot.json',{'producer':producer,'source':record(__file__),'configuration':record(config_path),
        'source_checkpoint':pointer_pin,'latest_failure':record(a.latest_failure),'old_RUN07_preserved':True})
    lease.update(running=True,actual_GPU_running=True,actual_training_optimizer_running=False,actual_owned_worker_pid=os.getpid(),
        actual_parent=producer,status='production_resident_RAM_restoring260',current_run_folder=str(folder),mode='training',
        current_configuration=record(config_path),committed_updates=initial_updates)
    write_json(lease_path,lease)
    monitor=Monitor(invocation,additional_reserved_host_bytes=throughput_reserve);stream=None;engine=None;torch=None;began=time.monotonic();stage_stats={};scopes=ExitStack()
    prefetch=None;prefetch_generation=0;prefetch_close_attempted_generation=None
    def close_private_metadata():
        nonlocal prefetch_close_attempted_generation
        if prefetch is None or prefetch_close_attempted_generation==prefetch_generation:return
        prefetch_close_attempted_generation=prefetch_generation
        try:
            proof=prefetch.close(discard_unconsumed=True)
            write_json(invocation/'private_metadata_worker_completion.json',dict(proof,source=prefetch_policy['source']))
        except BaseException as error:
            write_json(invocation/'private_metadata_worker_close_failure.json',{'error':repr(error),
                'worker_thread_alive':prefetch.thread.is_alive(),'worker_status_closed':prefetch.worker_status['closed'],
                'normal_worker_closure_claimed':False})
            raise
    def release_current_lease(committed):
        current=read_json(lease_path)
        if current.get('actual_owned_worker_pid')==os.getpid():
            current.update(running=False,actual_GPU_running=False,actual_training_optimizer_running=False,actual_owned_worker_pid=None,
                status='production_resident_RAM_process_closed',committed_updates=committed);write_json(lease_path,current)
    try:
        require('torch' not in sys.modules,'Coherent compiler/linker preload beforeTorch')
        from preload_nemo_linker_v1 import preload
        linker=preload();require(linker['actual_binding_linker']['version']==[12,9] and not linker['cuda_initialized'],'Same CUDA compiler/runtime')
        import torch
        torch.set_num_threads(4)
        physical=torch.cuda.get_device_properties(0).total_memory
        require(physical>=17000000000,'Existing physical GPU capacity')
        torch.cuda.set_per_process_memory_fraction(12000000000/physical,device=0)
        def guard():monitor.check()
        def stage(name,**extra):
            guard();free,total=torch.cuda.mem_get_info()
            require(free-int(extra.get('prospective_joint_GPU_bytes',0))>=2000000000,'Physical desktop/GPU reserve')
            require(torch.cuda.memory_allocated()<=12000000000,'Same12GB CUDA allocator policy')
            stage_stats[name]=stage_stats.get(name,0)+1
            return {'cuda':{'allocated_bytes':torch.cuda.memory_allocated(),'reserved_bytes':torch.cuda.memory_reserved()}}
        def admit(name,increment):
            guard();s=monitor._observe()
            reserve=prefetch_policy['additional_host_reservation_bytes']
            extra=0 if name=='private_scalar_prefetch_additional_host_state' else reserve
            if s['system_memory']['commit_available_bytes']-increment-extra-throughput_reserve<2000000000:
                write_json(invocation/'actual_prospective_admission_denial.json',{'name':name,'sample':s,
                    'increment_bytes':increment,'additional_prefetch_reservation_bytes':extra,'additional_throughput_bookkeeping_reservation_bytes':throughput_reserve,
                    'required_remaining_commit_bytes':2000000000,'no_allocation_admitted':True})
            require(s['system_memory']['commit_available_bytes']-increment-extra-throughput_reserve>=2000000000,
                'Actual allocation increment plus full extra prefetch reservation and unchanged commit headroom')
            require(s['system_memory']['physical_available_bytes']-reserve-throughput_reserve>=1500000000,
                'Full extra256MiB host reservation plus unchanged real physical headroom; no freed-memory credit')
            return {'passed':True,'increment':increment,'additional_prefetch_reservation_bytes':extra,'additional_throughput_bookkeeping_reservation_bytes':throughput_reserve,'sample':s}
        from asr18_BPE9223_operational_metadata_v2 import configure
        @contextmanager
        def metadata_phase(name):guard();yield;guard()
        configure(**base['process_readonly_metadata_authority'],guard=guard,metadata_phase=metadata_phase)
        from native_tdt_minimal_import_v1 import prepare
        minimal=prepare()
        from asr18_token_only_carrier_v1 import load as token_carrier
        from stream_frozen_encoder_matrices_v1 import load
        from streamed_generation_device_v1 import apply
        base_path=Path((P/'work/base_model_path.txt').read_text().strip())
        model,loader=load(base_path);logical=apply(model);processor=token_carrier(base_path)
        teacher={name:copy.deepcopy(getattr(model,name)).eval().requires_grad_(False)for name in ('encoder_projector','decoder','joint')}
        from asr18_BPE9223_bootstrap_v2 import bootstrap
        optimizer,provenance,snapshot=bootstrap(model,processor,torch=torch,stage=stage,admit=admit,guard=guard,migration_plan=c['row_migration_plan'])
        from tdt_joint_canonical_2d_fp32_9223_v1 import apply as canonical_joint
        provenance.update(frozen_loader=record(S/'stream_frozen_encoder_matrices_v1.py'),bootstrap=record(S/'asr18_BPE9223_bootstrap_v2.py'),
            minimal_native=minimal,logical_device=logical,NEW_joint_numerical_policy=canonical_joint(model.joint),frozen_loader_policy=loader)
        from bpe_bounded_production_journal_v4 import JournalNamespace
        source_namespace=JournalNamespace(Path(source_journal['path']).parent,identity=c['full_run_identity'],guard=guard)
        def journal_factory(saved):
            if throughput_transition is not None:
                require(saved==source_journal,'Exact saved276000 prefix; original180 positives preserved')
                return SavedJournalPrefixView(saved,guard),{'kind':'ASR18_readonly_saved276000_prefix_after_I31_unknown_exit_v1',
                    'actual_transition':record(a.throughput_resume),'original_tail_modified':False,'no_tail_replay':True}
            if ninth_interval_failure is not None:
                require(saved==source_journal,'Exact actual186400 saved prefix; historical163 COPY never replayed')
                return SavedJournalPrefixView(saved,guard),{'kind':'ASR18_readonly_saved186400_prefix_after_ninth_original_interval_failure_v1',
                    'failure':record(a.ninth_original_resume),'original_tail_modified':False,'no_tail_replay':True}
            if eighth_interval_failure is not None:
                require(saved==source_journal,'Exact actual170400 saved prefix; preserved69 COPY never replayed')
                return SavedJournalPrefixView(saved,guard),{'kind':'ASR18_readonly_saved170400_prefix_after_eighth_original_interval_failure_v1',
                    'failure':record(a.eighth_original_resume),'original_tail_modified':False,'no_tail_replay':True}
            if interval_failure is not None:
                require(saved==source_journal,'Exact actual142800 saved prefix, preserved94-tail never replayed')
                return SavedJournalPrefixView(saved,guard),{'kind':'ASR18_readonly_saved142800_prefix_after_original_interval_failure_v1',
                    'failure':record(a.original_interval_resume),'original_tail_modified':False,'no_tail_replay':True}
            if transition is not None:
                require(saved==source_journal,'Exact intentionally closed owner saved prefix')
                return SavedJournalPrefixView(saved,guard),{'kind':'ASR18_readonly_source_transition_saved_prefix_restore_v1',
                    'actual_transition':record(a.throughput_resume if a.throughput_resume is not None else a.gc_refresh_resume if a.gc_refresh_resume is not None else a.fresh_signature_resume if a.fresh_signature_resume is not None else a.canonical_archive_resume if a.canonical_archive_resume is not None else a.lookahead_resume if a.lookahead_resume is not None else a.source_transition_resume),'no_tail_replay':True,'original_modified':False}
            if periodic_quiescence_failure is not None:
                require(saved==source_journal and record(saved['path'])==periodic_quiescence_failure['whole_journal'],'Exact whole saved140000 journal')
                return SavedJournalPrefixView(saved,guard),{'kind':'ASR18_readonly_saved140000_prefix_restore_after_periodic_quiescence_failure_v1',
                    'failure':record(a.periodic_quiescence_resume),'original_tail_modified':False,'no_tail_replay':True}
            if DEV_admission_failure is not None:
                require(saved==source_journal and record(saved['path'])==DEV_admission_failure['whole_journal'],'Exact whole saved136000 journal')
                return SavedJournalPrefixView(saved,guard),{'kind':'ASR18_readonly_saved136000_prefix_restore_after_DEV_admission_failure_v1',
                    'failure':record(a.DEV_admission_resume),'original_tail_modified':False,'no_tail_replay':True}
            if frontend_failure is not None:
                require(saved==source_journal and record(saved['path'])==frontend_failure['whole_journal'],'Exact whole saved4000 journal')
                return SavedJournalPrefixView(saved,guard),{'kind':'ASR18_readonly_saved4000_prefix_restore_after_DEV_frontend_failure_v1',
                    'failure':record(a.DEV_frontend_resume),'original_modified':False,'new_journal_branch_created':False}
            if abnormal is not None:
                require(saved==source_journal and (check_original_closed_journal_stat(abnormal['whole_journal_including_preserved_tail'])if (a.fresh_signature_resume is not None or a.canonical_archive_resume is not None or a.lookahead_resume is not None or a.cushion225_resume is not None or a.packet_credit_resume is not None) else record(saved['path'])==abnormal['whole_journal_including_preserved_tail']),'Exact preserved failed journal')
                return SavedJournalPrefixView(saved,guard),{'kind':('ASR18_readonly_saved156600_prefix_restore_with_retained169_positive_tail_v1' if a.cushion225_resume is not None else 'ASR18_readonly_saved156400_prefix_restore_with_retained137_positive_tail_v1' if a.packet_credit_resume is not None else('ASR18_readonly_saved156200_prefix_restore_with_retained100_positive_tail_v1' if a.native_headroom_resume is not None else 'ASR18_readonly_saved133000_prefix_restore_with_retained137_positive_tail_v1')),
                    'failure':record(a.postrestore_selector_resume if a.postrestore_selector_resume is not None else a.cushion225_resume if a.cushion225_resume is not None else a.packet_credit_resume if a.packet_credit_resume is not None else(a.native_headroom_resume if a.native_headroom_resume is not None else a.abnormal_resume)),'original_tail_modified':False}
            require(saved==source_journal and record(saved['path'])=={'path':saved['path'],'bytes':saved['offset'],'sha256':saved['sha256']},
                'Saved whole coherent journal remains untouched during exact restore')
            return source_namespace.writer(Path(saved['path'])),{'kind':'ASR18_exact_coherent_readonly_journal_restore_v1','original_modified':False}
        from asr18_BPE9223_full_run_native_v12 import FullRunNativeOwner
        engine=FullRunNativeOwner(model,processor,teacher,optimizer,provenance,snapshot,invocation,c,torch=torch,stage=stage,
            admit=admit,guard=guard,journal=(SavedJournalPrefixView(source_journal,guard)if abnormal is not None or frontend_failure is not None or transition is not None or interval_failure is not None or DEV_admission_failure is not None or periodic_quiescence_failure is not None
                else source_namespace.writer(Path(source_journal['path']))),
            journal_factory=journal_factory,journal_factory_source=record(__file__))
        engine.restore_parent(pointer_pin,augmentation_identity=c['augmentation_identity'])
        require(optimizer.updates==initial_updates and not optimizer.poisoned and not engine.schedule.poisoned,'Actual854 restore before work')
        if preserved_tail is not None:
            destination=source_namespace.register('commits_'+str(a.invocation).zfill(6)+'.jsonl',prospective_bytes=source_journal['offset'])
            remaining=source_journal['offset']
            with Path(source_journal['path']).open('rb')as src,destination.open('ab')as dst:
                while remaining:
                    guard();block=src.read(min(1024**2,remaining));require(block,'Complete saved prefix remains available')
                    dst.write(block);remaining-=len(block)
                dst.flush();os.fsync(dst.fileno())
            require(check_original_closed_journal_stat(preserved_tail['whole_journal_including_preserved_tail'])if (a.throughput_resume is not None or a.ninth_original_resume is not None or a.gc_refresh_resume is not None or a.eighth_original_resume is not None or a.fresh_signature_resume is not None or a.canonical_archive_resume is not None or a.lookahead_resume is not None or a.cushion225_resume is not None or a.packet_credit_resume is not None) else record(source_journal['path'])==preserved_tail['whole_journal_including_preserved_tail'],'Old tail byteexact after prefix copy')
            journal=source_namespace.writer(destination)
            write_json(invocation/'actual_journal_branch.json',{'kind':'ASR18_actual_saved_prefix_recovery_journal_branch_v1',
                'predecessor':record(a.throughput_resume if a.throughput_resume is not None else a.ninth_original_resume if a.ninth_original_resume is not None else a.gc_refresh_resume if a.gc_refresh_resume is not None else a.eighth_original_resume if a.eighth_original_resume is not None else a.fresh_signature_resume if a.fresh_signature_resume is not None else a.canonical_archive_resume if a.canonical_archive_resume is not None else a.lookahead_resume if a.lookahead_resume is not None else a.postrestore_selector_resume if a.postrestore_selector_resume is not None else a.cushion225_resume if a.cushion225_resume is not None else a.original_interval_resume if interval_failure is not None else((a.packet_credit_resume if a.packet_credit_resume is not None else(a.native_headroom_resume if a.native_headroom_resume is not None else a.abnormal_resume)) if abnormal is not None else a.source_transition_resume)),
                'saved_prefix':source_journal,'preserved_original':preserved_tail['whole_journal_including_preserved_tail'],
                'new_active_journal':journal.binding(),'historical_unsaved_tail_updates':preserved_tail['preserved_tail_positive_updates'],
                'native_updates_or_draws_in_copy':0})
        elif initial_updates==260:
            namespace=JournalNamespace(folder/'journals',identity=c['full_run_identity'],guard=guard,create=True)
            destination=namespace.register('commits_000000.jsonl',prospective_bytes=source_journal['offset'])
            with Path(source_journal['path']).open('rb')as src,destination.open('ab')as dst:
                while True:
                    block=src.read(1024**2)
                    if not block:break
                    guard();dst.write(block)
                dst.flush();os.fsync(dst.fileno())
            journal=namespace.writer(destination)
        else:journal=source_namespace.writer(Path(source_journal['path']))
        require(journal.binding()['offset']==source_journal['offset']
            and journal.binding()['sha256']==source_journal['sha256'],'One exact original journal prefix copy, no tail replay')
        engine.journal=journal
        # Same already-proven runtime27 bank/backward/resident-GC policy.
        # The RAM transport changes neither this policy nor numerical updates.
        from asr18_frozen_encoder_cuda_cache_v3 import install,assert_idle
        from asr18_live_state_fingerprint_v1 import live_state_fingerprint
        from run_ASR18_native_throughput_component_v15 import cache_backward_calls,resident_gc_scope
        def owner_state_digest():return digest(live_state_fingerprint(optimizer,torch=torch,guard=guard))
        def admit_cache(request):
            require(request['logical_new_persistent_cuda_bytes']==1_217_066_496 and request['host_private_savings_credited_bytes']==0,'Entire frozenbank/no credit')
            admit('persistent294_cache_upload',1_217_066_496+4*1024**2)
            require(torch.cuda.memory_allocated()+1_217_066_496+16*1024**2<3_000_000_000,'Bank included in3GB endcache')
            return True
        policy=install(model,torch=torch,bank_binding=loader['bank'],guard=guard,admit_cache=admit_cache,owner_state_digest=owner_state_digest)
        write_json(invocation/'actual_cache_install.json',{'policy':policy,'idle':assert_idle(model,torch=torch,guard=guard)})
        scopes.enter_context(cache_backward_calls(model,torch=torch,enabled=True))
        scopes.enter_context(resident_gc_scope(invocation,enabled=True,optimizer=optimizer,torch=torch,guard=guard))
        import CPU14_startup_launch_adapter_v15 as startup_adapter
        startup_adapter.configure_startup_receipts(candidate_base['original_CPU14_startup_reader7_recovery_policy']['CPU_startup_policy'],record(a.stream_configuration))
        ResidentFeatureProducer=startup_adapter.ResidentFeatureProducer
        resident_source=record(S/'fastmss_resident_feature_stream_v14.py')
        write_json(invocation/'actual_CPU_launch_adapter_binding.json',{'original_worker_source':resident_source,
            'launch_adapter_source':record(inspect.getsourcefile(startup_adapter)),
            'startup_runner_source':candidate_base['original_CPU14_startup_reader7_recovery_policy']['CPU_startup_policy']['runner_source'],
            'startup_only_no_per_update_observer':True,'worker_stream_configuration':record(a.stream_configuration),'separate_parent_runtime_configuration':record(config_path),'additional_host_bytes_above_original72':8*1024**2})
        from asr18_BPE9223_grouped_RAM_native_v1 import native_update
        from asr18_BPE9223_lifecycle_v3 import next_boundary_request,validate_window_plan
        from bpe_production_window_coordinator_v4 import WindowCoordinator
        from bpe_final_DATA_schedule_context_v2 import normalize_schedule_state
        lifecycle=read_json(folder/'lifecycle.json')if(folder/'lifecycle.json').exists()else {'completed_boundaries':[],'winner':base['qualified_source_winner']}
        periodic_path=folder/'periodic_DEV.json'
        periodic=read_json(periodic_path)if periodic_path.exists()else{'kind':'ASR18_periodic_fixed350_DEV_state_v1',
            'interval_updates':4000,'first_update':4000,'completed_evaluations':[]}
        require(periodic['kind']=='ASR18_periodic_fixed350_DEV_state_v1' and periodic['interval_updates']==4000
            and periodic['first_update']==4000,'Separate actual periodic monitor state')
        evaluated={v['optimizer_updates']for v in periodic['completed_evaluations']}
        require(len(evaluated)==len(periodic['completed_evaluations']) and all(type(v)is int and v>=4000 and v%4000==0 for v in evaluated),
            'Unique exact periodic evaluation frontiers')
        completed=len(lifecycle['completed_boundaries']);plan_sequence=0;queued=None;last_checkpoint=initial_updates
        latest=read_json(folder/'latest_checkpoint.json')
        require(latest['pointer']==pointer_pin and latest['full_state_verified'] is True,'Current coherent checkpoint before pending DEV')
        checkpoint_paths=sorted((folder/'checkpoints').glob('update_*'))[-5:];prepared_archive=[];winner_folders=set()
        if lifecycle['winner'].get('resident_checkpoint_folder'):winner_folders.add(lifecycle['winner']['resident_checkpoint_folder'])
        audio_seconds=0.;native_seconds=0.;active_began=time.monotonic();normal_reason=None
        lease.update(status='production_resident_RAM_training',actual_training_optimizer_running=True);write_json(lease_path,lease)
        restored_status={'kind':'ASR18_actual_resident_RAM_training_progress_v1','producer':producer,'source':record(__file__),
            'actual_optimizer_updates':initial_updates,'native_updates_this_process':0,'clocks':[x+initial_updates for x in (64250,2000,62250)],
            'committed_sampler_sha256':digest(engine.schedule.state_dict()),'journal_prefix':journal.binding(),'last_saved_updates':initial_updates,
            'native_seconds':0.,'active_pipeline_seconds':0.,'source_audio_seconds':0.,'audio_minutes_per_wall_minute':0.,
            'queue_bytes':a.queue_bytes,'optimizer_poisoned':False,'schedule_poisoned':False,'checkpoint_interval_updates':200,
            'planned_native_updates':401072,'administrative_update_stop':None,'phase':'coherent_restore_before_first_native',
            'periodic_DEV_interval_updates':4000}
        write_json(folder/'progress/latest.json',restored_status)
        write_json(folder/'active_journal.json',{'kind':'ASR18_actual_current_resident_training_journal_v1','journal_prefix':journal.binding(),
            'source_checkpoint':pointer_pin,'actual_optimizer_updates':initial_updates,'producer':producer,'source':record(__file__)})
        def submit_plan(schedule):
            nonlocal plan_sequence
            E_reserve(8*1024**2)
            remaining=4000-schedule.committed_updates()%4000
            plan=(engine.context.plan(schedule,completed_boundaries=completed)if remaining>=20 else
                engine.context.validation_plan(schedule,completed_boundaries=completed,maximum_updates=remaining));plan_sequence+=1
            path=invocation/'plans'/('plan_'+str(plan_sequence).zfill(6)+'.json');write_json(path,plan)
            pin=record(path);stream.submit(pin);return plan,pin
        def evaluate_coherent_DEV(destination):
            from asr18_resident_serial350_DEV_v2 import evaluate
            from asr18_BPE9223_source_context_v2 import full_run_next_draw_packet
            from asr18_phase_private_monitor_v1 import PhasePrivateMonitor
            E_reserve(64*1024**2)
            before_schedule=copy.deepcopy(engine.schedule.state_dict());before_journal=journal.binding()
            before_next=full_run_next_draw_packet(engine.schedule,engine.context,optimizer_updates=optimizer.updates,guard=guard)
            before_lifecycle=copy.deepcopy(lifecycle);pending=None if queued is None else copy.deepcopy(queued[1])
            admission=admit('resident_optimizer_idle_serial350_DEV',2100000000)
            before_CUDA=stage('resident_serial350_DEV_before');started=time.monotonic();phase=PhasePrivateMonitor();phase.start()
            def DEV_guard():stage('resident_serial350_DEV_live')
            try:dev_pin,score=evaluate(model,processor,optimizer,torch=torch,config=c,pointer=latest['pointer'],folder=destination,guard=DEV_guard)
            finally:observed=phase.finish()
            require(observed['private_peak_bytes']<=admission['sample']['worker_private_memory']['private_bytes']+2100000000,
                'Measured resident DEV covered by unchanged2.1GB incremental allowance')
            require(engine.schedule.state_dict()==before_schedule and journal.binding()==before_journal and lifecycle==before_lifecycle,
                'DEV leaves both cursors/replay/journal/fullpass/winner state unchanged')
            require(full_run_next_draw_packet(engine.schedule,engine.context,optimizer_updates=optimizer.updates,guard=guard)==before_next,
                'DEV leaves actual next draw unchanged')
            if pending is not None:require(queued[1]==pending and record(pending['path'])==pending,'Pending private plan retained byteexact')
            resource={'kind':'ASR18_resident_DEV_actual_resource_and_frontier_preservation_v1','optimizer_updates':optimizer.updates,
                'DEV':dev_pin,'actual_native_updates_in_DEV':0,'actual_sampler_or_replay_advance':False,'all207_threeAdam_all4RNG_unchanged':True,
                'both_cursors_journal_and_nextdraw_unchanged':True,'fullpass_and_winner_state_unchanged_here':True,
                'pending_private_plan_retained':pending,'actual_phase_private':observed,'incremental_private_allowance_bytes':2100000000,
                'admission':admission,'CUDA_before':before_CUDA,'CUDA_after':stage('resident_serial350_DEV_after'),
                'CUDA_lifetime_peak_allocated_bytes_after':torch.cuda.max_memory_allocated(),'wall_seconds':time.monotonic()-started,
                'separate_concurrent_GPU_job':False,'resident_optimizer_objects_idle':True}
            path=Path(destination)/'training_frontier_preservation.json';write_json(path,resource)
            return dev_pin,score,record(path)
        def record_periodic(dev_pin,score,preservation,shared_fullpass):
            periodic['completed_evaluations'].append({'optimizer_updates':optimizer.updates,'DEV':dev_pin,'score':copy.deepcopy(score),
                'checkpoint_pointer':latest['pointer'],'frontier_preservation':preservation,'shared_with_full_pass_DEV':shared_fullpass,
                'periodic_record_advances_full_pass_or_winner':False})
            evaluated.add(optimizer.updates);write_json(periodic_path,periodic)
        if frontend_failure is not None:
            require(optimizer.updates==4000 and 4000 not in evaluated and queued is None,'Pending failed4000 evaluation before any4001/planning')
            lease.update(status='production_resident_RAM_serial350_DEV',actual_training_optimizer_running=False);write_json(lease_path,lease)
            restored_status.update(phase='pending_periodic_DEV_4000_before4001');write_json(folder/'progress/latest.json',restored_status)
            periodic_root=folder/'periodic_DEV';periodic_root.mkdir(exist_ok=True)
            dev_pin,score,preservation=evaluate_coherent_DEV(periodic_root/('update_000004000_attempt_'+str(a.invocation).zfill(6)))
            record_periodic(dev_pin,score,preservation,False)
            require(optimizer.updates==4000 and journal.binding()==source_journal,'Evaluation completed before positive Adam/new journal bytes')
            lease.update(status='production_resident_RAM_training',actual_training_optimizer_running=True);write_json(lease_path,lease)
        if DEV_admission_failure is not None:
            require(optimizer.updates==136000 and 136000 not in evaluated and queued is None,
                'Pending failed136000 DEV before any producer submission or136001')
            lease.update(status='production_resident_RAM_serial350_DEV',actual_training_optimizer_running=False);write_json(lease_path,lease)
            restored_status.update(phase='pending_periodic_DEV_136000_before136001');write_json(folder/'progress/latest.json',restored_status)
            periodic_root=folder/'periodic_DEV';periodic_root.mkdir(exist_ok=True)
            dev_pin,score,preservation=evaluate_coherent_DEV(periodic_root/('update_000136000_attempt_'+str(a.invocation).zfill(6)))
            record_periodic(dev_pin,score,preservation,False)
            require(optimizer.updates==136000 and journal.binding()==source_journal,
                'Pending136000 DEV completed before positive Adam, draws or new journal bytes')
            lease.update(status='production_resident_RAM_training',actual_training_optimizer_running=True);write_json(lease_path,lease)
        if periodic_quiescence_failure is not None:
            require(optimizer.updates==140000 and 140000 not in evaluated and queued is None,
                'Pending failed140000 DEV before any producer submission or140001')
            lease.update(status='production_resident_RAM_serial350_DEV',actual_training_optimizer_running=False);write_json(lease_path,lease)
            restored_status.update(phase='pending_periodic_DEV_140000_before140001');write_json(folder/'progress/latest.json',restored_status)
            periodic_root=folder/'periodic_DEV';periodic_root.mkdir(exist_ok=True)
            dev_pin,score,preservation=evaluate_coherent_DEV(periodic_root/('update_000140000_attempt_'+str(a.invocation).zfill(6)))
            record_periodic(dev_pin,score,preservation,False)
            require(optimizer.updates==140000 and journal.binding()==source_journal,
                'Pending140000 DEV completed before positive Adam, draws or new journal bytes')
            lease.update(status='production_resident_RAM_training',actual_training_optimizer_running=True);write_json(lease_path,lease)
        if throughput_active:admit('throughput_bookkeeping_above_full_CPU_startup',2500000000)
        stream=ResidentFeatureProducer(record(a.stream_configuration),invocation/'CPU',queue_bytes=a.queue_bytes,guard=guard)
        plan,plan_pin=submit_plan(engine.schedule)
        from asr18_BPE9223_private_scalar_prefetch_v1 import PrivateScalarPrefetch
        signature_scope=scopes.enter_context(FreshMetadataSignatureScope(engine.context,private_worker_closed=lambda:
            prefetch is None or(not prefetch.thread.is_alive()and prefetch.worker_status['closed'])))
        original_collect_observer=scopes.enter_context(ObserveOriginalCollectScope())
        postDEV_refresh=OnePostDEVRefresh(initial_updates)
        if ninth_policy_active:
            inherit_consumed_postDEV_opportunity(postDEV_refresh,base=base,restored_updates=initial_updates,guard=guard,
                emit=lambda value:write_json(invocation/'actual_inherited_consumed180000_refresh_opportunity.json',value))
        write_json(invocation/'actual_fresh_signature_collect_observation_admission.json',
            {'policy':base['fresh_metadata_signature_and_collect_observation_policy'],
             'signature_scope':signature_scope.status(),'original_collect_installed_after_startup_freeze':True})
        if engine.schedule.phase=='weak_label_continued_training':
            prefetch=PrivateScalarPrefetch(engine.context,guard=guard,admit=admit)
            prefetch_generation+=1
            write_json(invocation/'private_metadata_worker_admission.json',prefetch.admission)
        identity_authority=OriginalSourceAuthority(engine.context)if throughput_active else None
        phase_observer=MainPathScalarAttribution()if throughput_active else None
        while optimizer.updates<c['planned_native_updates']:
            action=next_boundary_request(engine.schedule.state_dict(),completed)
            if action['kind']!='bounded_training_window':
                raise RuntimeError('Declared complete350 DEV barrier reached; no pass/winner advance until real evaluator callback')
            validate_window_plan(engine.schedule.state_dict(),action,plan)
            window_started=time.monotonic();window_native_before=native_seconds;window_audio_before=audio_seconds
            metadata_timing={'not_ready_polls':0,'end_window_wait_count':0,'serial_fallback_count':0}
            private=engine.context.new_schedule();private.load_state_dict(normalize_schedule_state(plan['final_private_planning_sampler']))
            final_updates=plan['initial_committed_updates']+len(plan['updates'])
            validation_stop=a.validation_stop is not None and final_updates==a.validation_stop
            budget_stop=time.monotonic()-began>=a.maximum_wall_seconds-180 and final_updates-last_checkpoint>=200
            successor=private_successor_request(private.state_dict(),completed_boundaries=completed,
                validation_stop=validation_stop,budget_stop=budget_stop)
            queued=None;future_ticket=None
            if successor is not None:
                if prefetch is not None and successor['phase']=='weak_label_continued_training':
                    # Recheck fresh combined headroom before each new bounded private request.
                    admit('private_scalar_prefetch_additional_host_state',prefetch_policy['additional_host_reservation_bytes'])
                    future_ticket=prefetch.submit(private.state_dict(),maximum_updates=successor['maximum_updates'])
                else:
                    planning_started=time.monotonic();queued=submit_plan(private)
                    metadata_timing.update(serial_fallback_count=1,serial_planning_and_submission_seconds=time.monotonic()-planning_started)
            for index,entry in enumerate(plan['updates']):
                phase_stamp=phase_observer.start('host_gate')if phase_observer is not None else None
                try:
                    pressure_admission=(throughput_pressure_wait(observe=monitor._observe,check=guard,policy=native_pressure_policy['parameters'],additional_bookkeeping_reservation_bytes=throughput_reserve)
                        if throughput_active else wait_for_native_commit_pressure(observe=monitor._observe,check=guard,policy=native_pressure_policy['parameters']))
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
                    pressure_after_readiness=(throughput_pressure_wait(observe=monitor._observe,check=guard,policy=native_pressure_policy['parameters'],additional_bookkeeping_reservation_bytes=throughput_reserve)
                        if throughput_active else wait_for_native_commit_pressure(observe=monitor._observe,check=guard,policy=native_pressure_policy['parameters']))
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
                phase_stamp=phase_observer.start('metadata_binding')if phase_observer is not None else None
                try:
                    rows=entry['rows'];arrays=packet['arrays'];require(len(arrays)==len(rows),'Every full row present')
                    source=packet['source'];require(source['protocol']==resident_source,'Exact admitted resident CPU source, checked once at startup')
                    binding={'kind':'ASR18_verified_resident_RAM_preparation_v1','metadata_sha256':digest(metadata),
                        'producer_source_sha256':source['protocol']['sha256'],'token':packet['token']}
                finally:
                    if phase_stamp is not None:phase_observer.finish(phase_stamp,raised=sys.exc_info()[0]is not None)
                phase_stamp=phase_observer.start('native_and_commit')if phase_observer is not None else None
                try:
                    start=time.monotonic()
                    try:
                        native=native_update(model,teacher,optimizer,rows,entry['draws'],feature_for_row=lambda i:arrays[i],
                            update_binding=binding,prepared_update=metadata,torch=torch,population_mean_U=engine.schedule.means[tx.phase],
                            update=optimizer.updates+1,stage=stage,stop_check=guard,phase=tx.phase,feature_update_index=index,
                            reference_original_head=False)
                        require(native['augmentation']==metadata['prepared_payload'],'Exact consumed frozen augmentation payload')
                        evidence=dict(binding,kind='ASR18_resident_RAM_preparation_commit_evidence_v1',actual_positive_native_update=optimizer.updates)
                        state=commit_RAM_transaction(tx,optimizer,journal,native,evidence)
                        require(digest(state)==entry['after_sampler_sha256'],'Exact both cursor/replay post-update state')
                    except BaseException:
                        optimizer.poisoned=engine.schedule.poisoned=True;raise
                    native_seconds+=time.monotonic()-start;audio_seconds+=sum(float(r['duration'])for r in rows)
                finally:
                    if phase_stamp is not None:phase_observer.finish(phase_stamp,raised=sys.exc_info()[0]is not None)
                phase_stamp=phase_observer.start('archive_drop_release')if phase_observer is not None else None
                try:
                    prepared_archive.append(archive_codec.encode_record({'binding':binding,'metadata':metadata,'native':native}));token=packet['token'];del packet,arrays;stream.release(token)
                finally:
                    if phase_stamp is not None:phase_observer.finish(phase_stamp,raised=sys.exc_info()[0]is not None)
                if future_ticket is not None:
                    queued=adopt_private_successor(prefetch,future_ticket,private,submit_plan,wait=False,timings=metadata_timing)
                    if queued is not None:future_ticket=None
                phase_stamp=phase_observer.start('progress_emit')if phase_observer is not None else None
                try:
                    if optimizer.updates==initial_updates+1 or optimizer.updates%20==0:
                        elapsed=time.monotonic()-active_began
                        status={'kind':'ASR18_actual_resident_RAM_training_progress_v1','producer':producer,'source':record(__file__),
                            'actual_optimizer_updates':optimizer.updates,'native_updates_this_process':optimizer.updates-initial_updates,
                            'clocks':[x+optimizer.updates for x in (64250,2000,62250)],'committed_sampler_sha256':digest(state),
                            'journal_prefix':journal.binding(),'last_saved_updates':last_checkpoint,'native_seconds':native_seconds,
                            'active_pipeline_seconds':elapsed,'source_audio_seconds':audio_seconds,
                            'audio_minutes_per_wall_minute':audio_seconds/elapsed,'queue_bytes':a.queue_bytes,
                            'optimizer_poisoned':optimizer.poisoned,'schedule_poisoned':engine.schedule.poisoned,
                            'checkpoint_interval_updates':200,'planned_native_updates':401072,'administrative_update_stop':None}
                        write_json(folder/'progress/latest.json',status);print(json.dumps(status),flush=True)
                finally:
                    if phase_stamp is not None:phase_observer.finish(phase_stamp,raised=sys.exc_info()[0]is not None)
            if future_ticket is not None:
                queued=adopt_private_successor(prefetch,future_ticket,private,submit_plan,wait=True,timings=metadata_timing)
                future_ticket=None
            require(prefetch is None or(prefetch.outstanding is None and prefetch.held_result is None),
                'All ordered metadata adopted and submitted; no active/pending metadata during checkpoint or DEV')
            del private
            periodic_due=optimizer.updates>=4000 and optimizer.updates%4000==0 and optimizer.updates not in evaluated
            need_checkpoint=optimizer.updates-last_checkpoint>=200 or queued is None or periodic_due
            checkpoint_started=time.monotonic()
            if need_checkpoint:
                E_reserve(pointer['payload']['bytes']+64*1024**2)
                from asr18_live_state_fingerprint_v1 import live_state_fingerprint,require_exact_live_restore
                from asr18_streamed_persistence_fingerprint_v1 import verify_persistence
                from asr18_BPE9223_source_context_v2 import full_run_next_draw_packet
                from asr18_BPE9223_training_state_v1 import checkpoint
                ck=folder/'checkpoints'/('update_'+str(optimizer.updates).zfill(9));ck.mkdir()
                before=live_state_fingerprint(optimizer,torch=torch,guard=guard);write_json(ck/'pre_save.json',before)
                nxt=full_run_next_draw_packet(engine.schedule,engine.context,optimizer_updates=optimizer.updates,guard=guard);write_json(ck/'next_draw.json',nxt)
                archive_codec.write_gzip_archive(ck/'prepared_metadata.json.gz',prepared_archive)
                caller={'kind':'ASR18_resident_RAM_training_checkpoint_caller_v1','full_run_lifecycle_identity':c['full_run_identity'],
                    'run_identity_sha256':digest(c['full_run_identity']),'config':record(config_path),'engine':record(__file__),
                    'session':c['session'],'pre_save_fingerprint':record(ck/'pre_save.json'),'next_draw_packet':record(ck/'next_draw.json'),
                    'native_forecast_high_water':engine.forecast.state_dict(),'preparation_archive':record(ck/'prepared_metadata.json.gz'),
                    'checkpoint_span_updates':optimizer.updates-last_checkpoint,'checkpoint_interval_updates':200,
                    'geometry_forecast_used_for_hotpath_admission':False,
                    'source_checkpoint':pointer_pin,'native_source':record(S/'asr18_BPE9223_grouped_RAM_native_v1.py')}
                coordinator=WindowCoordinator.__new__(WindowCoordinator)
                coordinator.schedule=engine.schedule;coordinator.optimizer=optimizer;coordinator.journal=journal
                coordinator.plan=plan;coordinator.index=len(plan['updates']);coordinator.poisoned=False
                def writer(saved,identity):
                    return checkpoint(ck/'checkpoint.json',torch=torch,optimizer=optimizer,state=saved,provenance=provenance,
                        identity=identity,augmentation=engine.binding,journal=journal,diagnostic_only=False,guard=guard)
                ptr=coordinator.publish_checkpoint(write_coherent_checkpoint=writer,pointer_path=ck/'pointer.json',checkpoint_identity=caller,diagnostic_only=False)
                proof=verify_persistence(ptr['checkpoint'],expected_pre_save=before,identity=ptr['identity'],sampler_state=engine.schedule.state_dict(),
                    journal_prefix=journal.binding(),diagnostic_only=False,guard=guard)
                require(proof['named_tensors_verified']==854 and proof['Adam_states_verified']==215,'Full854/215 checkpoint persistence')
                require_exact_live_restore(before,live_state_fingerprint(optimizer,torch=torch,guard=guard))
                write_json(ck/'persistence_verified.json',proof);latest={'pointer':record(ck/'pointer.json'),'persistence':record(ck/'persistence_verified.json'),
                    'committed_updates':optimizer.updates,'full_state_verified':True,'source':record(__file__)}
                write_json(folder/'latest_checkpoint.json',latest);last_checkpoint=optimizer.updates;checkpoint_paths.append(ck);prepared_archive=[]
                # Exact branch-owned payload retirement only; preserve all metadata and original RUN07.
                for old in checkpoint_paths[:-5]:
                    if str(old)in winner_folders:continue
                    meta=read_json(old/'checkpoint.json');payload=Path(meta['payload']['path']).resolve()
                    require(payload.parent==old and payload.is_relative_to(folder/'checkpoints'),'Exact owned old branch payload only')
                    if payload.exists():require(record(payload)==meta['payload'],'Unchanged owned retired payload');payload.unlink()
                checkpoint_paths=checkpoint_paths[-5:]
                print(json.dumps({'actual_saved_updates':last_checkpoint,'checkpoint':latest,'old_RUN07_untouched':True}),flush=True)
            if phase_observer is not None:
                phase_summary=phase_observer.snapshot_and_reset()
                require(len(canonical(phase_summary))<=16*1024,'Fixed bounded per20 phase scalars, no payload/history')
                metadata_timing['main_path_scalar_attribution']=phase_summary
            metadata_timing['original_collect_observation']=original_collect_observer.take_window()
            metadata_timing.update(kind='ASR18_actual_private_metadata_window_timing_v1',
                initial_updates=plan['initial_committed_updates'],committed_updates=optimizer.updates,
                native_seconds=native_seconds-window_native_before,source_audio_seconds=audio_seconds-window_audio_before,
                checkpoint_included=need_checkpoint,checkpoint_seconds=time.monotonic()-checkpoint_started if need_checkpoint else 0.,
                window_pipeline_seconds=time.monotonic()-window_started)
            with(invocation/'private_metadata_window_timings.jsonl').open('a',encoding='utf-8')as timing_log:
                timing_log.write(json.dumps(metadata_timing,separators=(',',':'))+'\n')
            action=next_boundary_request(engine.schedule.state_dict(),completed)
            if action['kind']=='checkpoint_then_serial350DEV':
                CPU_closed=stream.close();stream=None;write_json(invocation/('CPU_pass_'+str(completed)+'_normal0.json'),CPU_closed)
                dev_pin,score,preservation=evaluate_coherent_DEV(folder/('DEV_'+str(completed+1)))
                if periodic_due:record_periodic(dev_pin,score,preservation,True)
                old=lifecycle['winner']
                if score['strict_all_six_passed']and score['word_errors']['tr']<old['word_errors']['tr']:
                    lifecycle['winner']={'artifact':ptr['checkpoint'],'checkpoint_pointer':latest['pointer'],'word_errors':score['word_errors'],
                        'reference_words':score['reference_words'],'DEV':dev_pin,'strict_qualified':True,'resident_checkpoint_folder':str(ck)}
                    winner_folders.add(str(ck))
                lifecycle['completed_boundaries'].append({'boundary':action['boundary'],'DEV':dev_pin,'score':score,'checkpoint_pointer':latest['pointer']})
                completed+=1;write_json(folder/'lifecycle.json',lifecycle)
                if completed==3:normal_reason='all401072_and_three_declared_DEVs_complete';break
                if validation_stop or budget_stop:
                    normal_reason='first200_checkpoint_fresh_restore_validation'if validation_stop else 'normal_owner_wall_budget'
                    break
                if throughput_active:admit('throughput_bookkeeping_above_full_CPU_startup',2500000000)
                stream=ResidentFeatureProducer(record(a.stream_configuration),invocation/('CPU_after_DEV_'+str(completed)),queue_bytes=a.queue_bytes,guard=guard)
                plan,plan_pin=submit_plan(engine.schedule);continue
            if periodic_due:
                CPU_quiescence=close_private_preparation_for_DEV(stream,queued,committed_updates=optimizer.updates,guard=guard)
                stream=None
                require_exact_live_restore(before,live_state_fingerprint(optimizer,torch=torch,guard=guard))
                require(journal.binding()==ptr['journal_prefix'] and digest(engine.schedule.state_dict())==ptr['sampler_state_sha256'],
                    'Private CPU closure leaves saved weights/threeAdam/fourRNG/both cursors/journal unchanged')
                write_json(invocation/('CPU_periodic_'+str(optimizer.updates).zfill(9)+'_quiescence.json'),CPU_quiescence)
                periodic_root=folder/'periodic_DEV';periodic_root.mkdir(exist_ok=True)
                dev_pin,score,preservation=evaluate_coherent_DEV(periodic_root/('update_'+str(optimizer.updates).zfill(9)))
                record_periodic(dev_pin,score,preservation,False)
                if not postDEV_refresh.opportunity_consumed:
                    require(optimizer.updates==postDEV_refresh.expected_update,
                        'Only first genuine scheduled periodic DEV in this native process')
                    if validation_stop or budget_stop:
                        postDEV_refresh.consume_ineligible(optimizer.updates,reason='original_coincident_normal_stop',
                            phase=engine.schedule.phase,scalar_was_constructed=bool(prefetch_generation),
                            prefetch_absent=prefetch is None,validation_stop=validation_stop,budget_stop=budget_stop,guard=guard,
                            emit=lambda value:write_json(invocation/'actual_one_postDEV_gc_refresh_ineligible.json',value))
                    elif prefetch is None and engine.schedule.phase=='cleaner_data_fine_tuning' and prefetch_generation==0:
                        postDEV_refresh.consume_ineligible(optimizer.updates,reason='original_cleaner_phase_scalar_never_constructed',
                            phase=engine.schedule.phase,scalar_was_constructed=False,prefetch_absent=True,
                            validation_stop=False,budget_stop=False,guard=guard,
                            emit=lambda value:write_json(invocation/'actual_one_postDEV_gc_refresh_ineligible.json',value))
                    else:
                        require(stream is None and queued is not None
                            and optimizer.updates==last_checkpoint==latest['committed_updates']
                            and latest['full_state_verified']is True and not optimizer.poisoned and not engine.schedule.poisoned,
                            'Successful complete persistence/DEV before new preparation, no normal-stop substitution')
                        cpu_closed=CPU_quiescence['CPU_actual0']
                        phase_preserved=read_json(preservation['path'])
                        require(cpu_closed['actual_Popen_wait_code']==0 and cpu_closed['transport_threads_joined']is True
                            and cpu_closed['pipe_handles_closed']is True
                            and cpu_closed['packet_credit']['all_owned_kernel_handles_closed']is True
                            and cpu_closed['packet_credit_owner']['all_owned_kernel_handles_closed']is True
                            and phase_preserved['actual_phase_private']['joined']is True
                            and phase_preserved['actual_phase_private']['error']is None
                            and phase_preserved['all207_threeAdam_all4RNG_unchanged']is True
                            and phase_preserved['both_cursors_journal_and_nextdraw_unchanged']is True,
                            'Genuine feature closure and successful real DEV preservation/monitor join')
                        require(prefetch is not None and not prefetch.closed and prefetch.thread.is_alive()
                            and prefetch.worker_status=={'closed':False,'error':None}
                            and future_ticket is None and prefetch.outstanding is None
                            and prefetch.held_result is None and engine.context.resolver is prefetch.original,
                            'Fully adopted idle original scalar worker; no unconsumed work to discard')
                        import weakref
                        original_resolver_identity_for_refresh=id(prefetch.original)
                        old_scalar_ref=weakref.ref(prefetch)
                        prefetch_close_attempted_generation=prefetch_generation
                        scalar_closed=prefetch.close(discard_unconsumed=False)
                        require(scalar_closed['joined']is True and scalar_closed['worker_connections_closed_on_worker']is True
                            and scalar_closed['discarded_unconsumed_request']is False
                            and not prefetch.thread.is_alive()and prefetch.worker_status=={'closed':True,'error':None},
                            'Actual empty worker join/DB+locks closure before dropping old object/frame')
                        write_json(invocation/'first_periodic_refresh_scalar_worker_closure.json',
                            dict(scalar_closed,source=prefetch_policy['source'],generation=prefetch_generation))
                        prefetch=None
                        require(old_scalar_ref()is None,'Old joined scalar worker object/frame is genuinely dropped')
                        del old_scalar_ref
                        # The exact queued tuple/plan/pin and scientific owners stay
                        # live. These completed aliases have no remaining use on the
                        # non-fullpass periodic branch; never release a packet twice.
                        del plan,plan_pin,coordinator,writer,before,nxt,caller,ptr,proof,ck
                        del tx,entry,rows,metadata,native,source,binding,evidence,state,CPU_quiescence
                        del dev_pin,score,preservation,cpu_closed,phase_preserved,scalar_closed
                        refresh_barrier={'kind':'ASR18_actual_first_periodic_DEV_refresh_barrier_v1','updates':optimizer.updates,
                            'genuine_periodic_DEV_returned_and_recorded':optimizer.updates in evaluated,
                            'checkpoint_full_persistence_same_frontier':latest['full_state_verified']and latest['committed_updates']==optimizer.updates,
                            'feature_CPU_actual_wait0_joined_pipes_and_both_credit_sets_closed':True,
                            'DEV_preservation_and_phase_monitor_join_succeeded':True,
                            'optimizer_idle_unpoisoned':not optimizer.poisoned and not engine.schedule.poisoned,
                            'scalar_empty_joined_closed_errorNone_worker_DB_and_locks_closed':True,
                            'old_scalar_object_and_frame_dropped':prefetch is None,
                            'parent_original_context_and_locks_continuous':not engine.context.closed and len(engine.context.resolver.metadata_locks.handles)==3,
                            'no_live_packet_or_arrays':True,'checkpoint_writer_and_completed_aliases_dropped':True,
                            'queued_original_same_object_and_pin':record(queued[1]['path'])==queued[1],
                            'main_Monitor_active':monitor.thread.is_alive()and monitor.error is None,
                            'stream_is_None':stream is None,'future_ticket_is_None':future_ticket is None,
                            'resolver_is_original':id(engine.context.resolver)==original_resolver_identity_for_refresh,
                            'capture_contract':'original_bounded_live_fingerprint_and_identity_manifest_v1'}
                        def refresh_admission(increment):
                            require(increment==16777216,'Exact additional event reserve, no freed CPU/DEV credit')
                            E_reserve(increment);original=admit('one_postDEV_refresh_additional_bookkeeping',increment)
                            sample=monitor._observe();system=sample['system_memory']
                            require(system['commit_available_bytes']>=4535212672 and system['physical_available_bytes']>=1785212672,
                                'Full original225+FULLprefetch thresholds plus separate16MiB event reserve')
                            return {'passed':True,'original_admission':original,'sample':sample,
                                'additional_bookkeeping_reserve_bytes':increment,'required_commit_bytes':4535212672,
                                'required_physical_bytes':1785212672,'host_credit_bytes':0}
                        postDEV_refresh.execute_real(refresh_barrier,
                            capture=lambda:capture_idle_state(optimizer=optimizer,torch=torch,guard=guard,schedule=engine.schedule,
                                context=engine.context,journal=journal,lifecycle=lifecycle,periodic=periodic,queued=queued,
                                latest=latest,completed=completed),compare=compare_idle_state,guard=guard,admit=refresh_admission,
                            original_collect_observer=original_collect_observer,
                            emit=lambda value:write_json(invocation/'actual_one_postDEV_gc_refresh.json',value))
                        del refresh_barrier,original_resolver_identity_for_refresh
                        prefetch=PrivateScalarPrefetch(engine.context,guard=guard,admit=admit)
                        prefetch_generation+=1
                        write_json(invocation/'private_metadata_worker_after_first_periodic_refresh_admission.json',
                            dict(prefetch.admission,generation=prefetch_generation,only_after_completed_refresh=postDEV_refresh.completed))
            if validation_stop or budget_stop:
                normal_reason='first200_checkpoint_fresh_restore_validation'if validation_stop else 'normal_owner_wall_budget'
                break
            require(queued is not None,'Exact private successor after periodic or normal bounded window')
            if periodic_due:
                if throughput_active:admit('throughput_bookkeeping_above_full_CPU_startup',2500000000)
                stream=ResidentFeatureProducer(record(a.stream_configuration),invocation/('CPU_after_periodic_DEV_'+str(optimizer.updates).zfill(9)),queue_bytes=a.queue_bytes,guard=guard)
                stream.submit(queued[1]) # Same immutable private plan; no sampler/plan regeneration.
            plan,plan_pin=queued
        require(last_checkpoint==optimizer.updates,'Normal close saves every live update')
        close_private_metadata() # Genuine joined closure before normal0/engine context closure.
        if stream is not None:
            CPU_closed=stream.close();stream=None;write_json(invocation/'CPU_normal0.json',CPU_closed)
        latest=read_json(folder/'latest_checkpoint.json')
        write_json(invocation/'normal_completion.json',{'kind':'ASR18_actual_resident_RAM_coherent_native_completion_v1','passed':True,
            'producer':producer,'source':record(__file__),'actual_optimizer_updates':optimizer.updates,'all_live_updates_saved':True,
            'latest_checkpoint':latest,'reason':normal_reason,'completed_boundaries':completed,'actual_Popen_wait_still_unobserved_inside_process':True,
            'elapsed_seconds':time.monotonic()-began,'native_seconds':native_seconds,'source_audio_seconds':audio_seconds})
        if completed==3:write_json(folder/'training_complete.json',{'actual_optimizer_updates':optimizer.updates,'last_checkpoint':last_checkpoint,'completed_boundaries':completed})
    except BaseException as error:
        if stream is not None:
            try:stream.abort()
            except BaseException as abort_error:write_json(invocation/'CPU_abort_error.json',{'error':repr(abort_error),'normal_CPU0_claimed':False})
            stream=None
        write_json(invocation/'actual_failure.json',{'producer':producer,'error':repr(error),'traceback':traceback.format_exc(),
            'actual_optimizer_updates':None if engine is None else engine.optimizer.updates,'last_checkpoint':locals().get('last_checkpoint',initial_updates),
            'all_unsaved_live_updates_require_coherent_reload':True,'elapsed_seconds':time.monotonic()-began,
            **({'native_pressure_denial':error.native_pressure_denial}if hasattr(error,'native_pressure_denial')else{}),
            **({'packet_credit_timeout':error.packet_credit_timeout}if hasattr(error,'packet_credit_timeout')else{})})
        raise
    finally:
        try:close_private_metadata()
        finally:
            context_safe=prefetch is None or(not prefetch.thread.is_alive() and prefetch.worker_status['closed'])
            with ExitStack() as cleanup:
                cleanup.callback(release_current_lease,locals().get('last_checkpoint',initial_updates))
                cleanup.callback(monitor.close)
                if engine is not None and context_safe:cleanup.callback(engine.close)
                elif engine is not None:
                    write_json(invocation/'context_close_deferred_to_abnormal_process_exit.json',
                        {'reason':'Private worker did not join; never close its continuously locked source authority early',
                         'normal_context_or_worker_closure_claimed':False})
                cleanup.callback(scopes.close)
                if stream is not None:cleanup.callback(stream.close)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--configuration',required=True);p.add_argument('--stream-configuration',required=True)
    p.add_argument('--pointer',required=True);p.add_argument('--latest-failure',required=True);p.add_argument('--folder',required=True)
    p.add_argument('--queue-bytes',type=int,default=268435456);p.add_argument('--invocation',type=int,default=1)
    p.add_argument('--throughput-resume');p.add_argument('--ninth-original-resume');p.add_argument('--gc-refresh-resume');p.add_argument('--eighth-original-resume');p.add_argument('--fresh-signature-resume');p.add_argument('--canonical-archive-resume');p.add_argument('--lookahead-resume');p.add_argument('--postrestore-selector-resume');p.add_argument('--cushion225-resume');p.add_argument('--packet-credit-resume');p.add_argument('--native-headroom-resume');p.add_argument('--periodic-quiescence-resume');p.add_argument('--DEV-admission-resume');p.add_argument('--original-interval-resume');p.add_argument('--source-transition-resume');p.add_argument('--prior-process');p.add_argument('--abnormal-resume');p.add_argument('--DEV-frontend-resume');p.add_argument('--validation-stop',type=int)
    p.add_argument('--maximum-wall-seconds',type=int,default=86400);main(p.parse_args())
