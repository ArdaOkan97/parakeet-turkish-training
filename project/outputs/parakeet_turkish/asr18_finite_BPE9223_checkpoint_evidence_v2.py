"""Honest finite20 checkpoint evidence; no durable/full-pass authority.

Only scalar/source/process metadata is interpreted here. The DEV adapter keeps
the original complete payload hash and selected207/all4RNG restoration checks.
No model, optimizer, numerical package, launcher or writer lives in this module.
"""
from copy import deepcopy
import hashlib
import math
from pathlib import Path

from bpe_broad_audio_v1 import record, require
from bpe_broad_production_state_v1 import digest
from asr18_BPE9223_lifecycle_v2 import read_bound, bound
from bpe_streaming_checkpoint_semantics_v2 import decode_scalar_tree, select, verify_saved_metadata
from asr18_streamed_persistence_fingerprint_v1 import normalize_fingerprint

ROOT = Path(r'E:\gpu_training_scratchpad')
PROJECT = Path(__file__).resolve().parents[2]
OWNER = '/root/memory_diagnosis'
KIND = 'ASR18_finite_BPE9223_recovered_checkpoint_binding_v1'
TOOL_KIND = 'ASR18_finite_BPE9223_owner_tool_completion_v1'
PROCESS_KIND = 'ASR18_throughput_component_actual_owned_process0_v1'
RUNNER = PROJECT/'work/research/run_ASR18_BPE9223_sustained_v2.py'
RUNNER_SHA = '231ba3c8cfa15f4237c4117d5ab1fb046d7e568b80d8d2e25c0cadc43d2f41ff'


def same(a, b):
    return a == b and digest(a) == digest(b)


def integer(value, low=0):
    require(type(value) is int and value >= low, 'Exact finite nonboolean integer required')
    return value


def identity_dead(value):
    require(type(value) is dict and set(value) == {'pid','ppid','create_time'}
        and type(value['create_time']) in (int,float) and math.isfinite(value['create_time'])
        and value['create_time'] > 0, 'Captured real PID/create-time identity required')
    integer(value['pid'], 1); integer(value['ppid'], 1)
    import psutil
    try:
        process = psutil.Process(value['pid'])
        require(process.create_time() != value['create_time'] or not process.is_running(),
            'Exact finite native/monitor/launcher identity remains alive')
    except psutil.NoSuchProcess:
        pass
    return True


def checked_guard(value, pid):
    require(value['pid'] == pid and type(value['pid']) is int
        and type(value['actual_exit_code']) is int and value['actual_exit_code'] == 0
        and value['monitor_clean'] is True and value['monitor_joined'] is True
        and value['error'] is None and integer(value['samples'], 3) >= 3,
        'Genuine finite clean joined watchdog0 required')


def checked_process(value, *, source, configuration, report, read_artifact=read_bound,
                    check_dead=identity_dead):
    require(value['kind'] == PROCESS_KIND and type(value['actual_Popen_wait_code']) is int
        and value['actual_Popen_wait_code'] == 0 and value['all_captured_identities_dead'] is True
        and same(value['source'], source) and same(value['configuration'], configuration)
        and same(value['report'], report), 'Exact finite actual parent wait0/source/config/report required')
    identities = value['captured_identities']
    require(type(identities) is list and 1 <= len(identities) <= 32
        and len({(v['pid'],v['create_time']) for v in identities}) == len(identities)
        and value['native'] in identities, 'Every captured finite identity, including real native')
    for item in identities:
        require(check_dead(item) is True, 'Every exact captured identity must actually be dead')
    checked_guard(read_artifact(value['guard']), value['native']['pid'])
    actual = read_artifact(report)
    require(actual['passed'] is True and same(actual['producer'], value['native'])
        and same(actual['source'], source), 'Finite process/report producer/source join')
    return value


def checked_owner(value, *, source, configuration, report, read_artifact=read_bound,
                  check_dead=identity_dead):
    require(value['kind'] == TOOL_KIND and value['owner'] == OWNER
        and type(value['actual_native_exit_code']) is int and value['actual_native_exit_code'] == 0
        and type(value['actual_exec_tool_exit_code']) is int and value['actual_exec_tool_exit_code'] == 0
        and type(value['actual_tool_chunk_id']) is str and bool(value['actual_tool_chunk_id'])
        and same(value['source'], source) and same(value['configuration'], configuration)
        and same(value['report'], report) and integer(value['native_PID'], 1) == value['producer']['pid']
        and check_dead(value['producer']) is True,
        'Actual source-bound finite owner TOOL0, never marker-only success')
    checked_guard(read_artifact(value['guard_closure']), value['native_PID'])
    return value


def metadata_fingerprint(metadata):
    """Same descriptor equations as saved_fingerprint, without payload reads."""
    widths = {'float32':4,'bfloat16':2,'float16':2,'float64':8,'int64':8,
        'int32':4,'int16':2,'int8':1,'uint8':1,'bool':1}
    entries = metadata['tensors']; descriptors = []
    require(type(entries) is list and len(entries) == 854, 'Exact full9223 854 named tensors')
    for index, entry in enumerate(entries):
        require(type(entry['index']) is int and entry['index'] == index
            and entry['dtype'] in widths and type(entry['shape']) is list
            and all(type(v) is int and v >= 0 for v in entry['shape'])
            and type(entry['bytes']) is int
            and entry['bytes'] == math.prod(entry['shape'])*widths[entry['dtype']],
            'Finite complete descriptor shape/dtype/bytes')
        descriptors.append({'shape':entry['shape'],'dtype':'torch.'+entry['dtype'],
            'bytes':entry['bytes'],'sha256':entry['sha256']})
    used = set(); budget = [0]
    def decode(node, depth=0):
        budget[0] += 1
        require(depth <= 64 and budget[0] <= 2_000_000 and type(node) is list and len(node) == 2,
            'Bounded finite named checkpoint tree')
        tag, value = node
        if tag == 'tensor':
            require(type(value) is int and 0 <= value < len(descriptors), 'Exact named tensor reference')
            used.add(value); return deepcopy(descriptors[value])
        if tag == 'scalar':
            require(value is None or type(value) in (str,int,float,bool), 'Finite scalar type')
            require(type(value) is not float or math.isfinite(value), 'Finite scalar')
            return value
        require(tag in ('dict','list','tuple') and type(value) is list, 'Exact finite tagged tree')
        if tag != 'dict':
            items = [decode(v,depth+1) for v in value]
            return tuple(items) if tag == 'tuple' else items
        result = {}
        for pair in value:
            require(type(pair) is list and len(pair) == 2, 'Finite dictionary pair')
            key = decode(pair[0],depth+1)
            require(type(key) in (str,int) and key not in result, 'Unique finite scalar dictionary key')
            result[key] = decode(pair[1],depth+1)
        return result
    saved = decode(metadata['tree'])
    require(len(used) == 854, 'All854 named tensors must be referenced')
    opt, weights = saved['optimizer'], saved['weights']; partitions = []; count = 0
    require(len(weights) == 207 and len(opt['optimizers']) == 3, 'All207/threeAdam saved descriptor coverage')
    for part in opt['optimizers']:
        indices = [i for group in part['param_groups'] for i in group['params']]
        require(all(type(i) is int and i >= 0 for i in indices) and len(indices) == len(set(indices))
            and set(indices) == set(part['state'])
            and all(set(part['state'][i]) == {'step','exp_avg','exp_avg_sq'} for i in indices),
            'Complete finite mature/appended Adam indexed state')
        partitions.append({'param_groups':part['param_groups'],'state':part['state']}); count += len(indices)
    require(count == 215, 'Exact9223 215 mature/appended Adam states')
    state = {'kind':'ASR18_actual_live207_threeAdam_all4RNG_fingerprint_v1',
        'updates':opt['updates'],'identity':opt['identity'],'clocks':opt['clocks'],
        'weights':{name:weights[name] for name in sorted(weights)},'Adam_partitions':partitions,
        'RNG':{'python':saved['python_rng'],'numpy':saved['numpy_rng'],
            'torch':saved['torch_rng'],'CUDA':saved['cuda_rng']},'maximum_transfer_slice_bytes':4*1024**2}
    return {'state_sha256':digest(state),'state':state}


def _checked_binding(value, *, read_artifact, check_process, check_owner,
                     check_format, check_payload, check_prefix):
    require(value['kind'] == KIND and value['owner'] == OWNER and value['diagnostic_only'] is True
        and value['qualified'] is False and value['production_enabled'] is False
        and integer(value['optimizer_updates'],1) == 20,
        'Only the genuine finite20 unqualified checkpoint, no full-pass authority')
    source, config_pin = value['native_source'], value['configuration']
    require(source['path'] == str(RUNNER) and source['sha256'] == RUNNER_SHA,
        'Exact source-reviewed finite9223 runner2')
    folder = Path(value['overall_result']['path']).resolve().parent
    require(folder.is_relative_to(ROOT.resolve())
        and Path(value['overall_result']['path']).name == 'overall_result.json'
        and Path(value['native_process0']['path']).resolve() == folder/'train/actual_process0.json'
        and Path(value['recovery_process0']['path']).resolve() == folder/'recover/actual_process0.json'
        and Path(value['native_report']['path']).resolve() == folder/'train/training_result.json'
        and Path(value['recovery_report']['path']).resolve() == folder/'recover/recovery_result.json',
        'The actual same finite parent train/recover report and process namespaces')
    config = read_artifact(config_pin)
    require(config['kind'] == 'ASR18_BPE9223_sustained_original_source_configuration_v1'
        and config['production_enabled'] is False and config['owner_validation_only'] is True
        and config['checkpoint_diagnostic_only'] is True and type(config['vocabulary']) is int
        and config['vocabulary'] == 9223 and type(config['native_updates']) is int
        and config['native_updates'] == 20 and config['window_count'] == 1
        and same(config['harness_source'],source), 'Exact qualification20 configuration')
    overall = read_artifact(value['overall_result'])
    require(overall['passed'] is True and overall['production_enabled'] is False
        and overall['arm'] == 'candidate' and integer(overall['actual_native_updates'],1) == 20
        and same(overall['source'],source) and same(overall['configuration'],config_pin)
        and same(overall['actual_native_process0'],value['native_process0'])
        and same(overall['actual_fresh_recovery_process0'],value['recovery_process0']),
        'Genuine source-bound finite overall result')
    check_owner(read_artifact(value['owner_tool_actual0']), source=source,
        configuration=config_pin, report=value['overall_result'])
    native0 = check_process(read_artifact(value['native_process0']), source=source,
        configuration=config_pin, report=value['native_report'])
    recovery0 = check_process(read_artifact(value['recovery_process0']), source=source,
        configuration=config_pin, report=value['recovery_report'])
    require(native0['native'] != recovery0['native'], 'A separate actual fresh recovery process is mandatory')
    native, recovery = read_artifact(value['native_report']), read_artifact(value['recovery_report'])
    require(native['passed'] is True and native['production_enabled'] is False
        and native['production_source_eligible'] is False and native['arm'] == overall['arm']
        and same(native['configuration'],config_pin) and same(native['source'],source)
        and integer(native['actual_optimizer_updates'],1) == 20
        and type(native['history']) is list and len(native['history']) == 20
        and all(type(row['update']) is int and row['update'] == i+1 for i,row in enumerate(native['history']))
        and recovery['passed'] is True and recovery['production_enabled'] is False
        and recovery['arm'] == native['arm'] and same(recovery['source'],source)
        and recovery['all207_threeAdam_all4RNG_sampler_nextdraw_exact'] is True
        and integer(recovery['actual_native_updates']) == 0,
        'All20 native updates and genuine zero-update all207/threeAdam/all4RNG fresh recovery')
    for key in ('checkpoint','PRE_save_fingerprint','persistence','next_draw','journal_prefix'):
        require(same(value[key],native[key]), 'Finite binding differs from actual native report: '+key)
    require(value['sampler_state_sha256'] == native['sampler_sha256'], 'Finite full sampler digest differs')
    next_draw = read_artifact(value['next_draw'])
    require(digest(next_draw) == native['next_draw_sha256']
        and next_draw['kind'] == 'ASR18_actual_checkpoint_provisional_next_draw_packet_v1'
        and integer(next_draw['optimizer_updates'],1) == 20
        and next_draw['before_sampler_sha256'] == value['sampler_state_sha256']
        and integer(next_draw['draws_consumed']) == 0 and next_draw['replay_frontier_advanced'] is False
        and next_draw['audio_or_augmentation_applied'] is False, 'Actual provisional next-draw packet differs')
    plan = read_artifact(config['plan'])
    require(type(plan['updates']) is list and len(plan['updates']) == 20
        and plan['updates'][-1]['after_sampler_sha256'] == value['sampler_state_sha256'],
        'The actual complete20 plan ends at the same full sampler')
    branch = recovery['branch']
    require(branch['kind'] == 'ASR18_preserved_tail_checkpoint_branch_v1'
        and same(branch['original_checkpoint_prefix'],value['journal_prefix'])
        and branch['old_journal_modified'] is False and branch['tail_replayed_or_sampler_advanced'] is False
        and Path(branch['branch_checkpoint_prefix']['path']).resolve() == folder/'recover/restored_journal_prefix.jsonl'
        and branch['branch_checkpoint_prefix']['offset'] == value['journal_prefix']['offset']
        and branch['branch_checkpoint_prefix']['sha256'] == value['journal_prefix']['sha256'],
        'Fresh recovery is tied to this actual checkpoint prefix without replaying a tail')
    shared = read_artifact(overall['shared_features'])
    identity = value['identity']
    expected_identity = {'kind':'ASR18_original_first4_throughput_unqualified_checkpoint_identity_v1',
        'config':config_pin,'arm':'candidate','shared_feature_owner':shared['owner'],
        'BPE9223_population':config['planning_arguments']['DATA_evidence']['BPE9223_population'],
        'row_migration_plan':config['row_migration_plan'],'production_enabled':False,
        'production_source_eligible':False}
    require(same(identity,expected_identity), 'Actual finite checkpoint identity, not a synthetic full-run identity')
    metadata = read_artifact(value['checkpoint']); check_format(metadata)
    require(same(metadata['payload'],value['payload']), 'Finite immutable payload binding')
    check_payload(value['checkpoint'], value['payload'])
    scalar = lambda *path: decode_scalar_tree(select(metadata['tree'],*path))
    sampler = scalar('sampler')
    semantic = verify_saved_metadata(metadata,identity=identity,optimizer_updates=20,
        sampler_state=sampler,journal_prefix=value['journal_prefix'],diagnostic_only=True)
    require(digest(sampler) == value['sampler_state_sha256']
        and scalar('optimizer','kind') == 'BPE9223_broad_full207_three_AdamW_mixed_fresh_clock_v1'
        and type(scalar('optimizer','appended512_Adam_clock')) is int
        and scalar('optimizer','appended512_Adam_clock') == 20
        and same(scalar('optimizer','clocks'),[64270,2020,62270]), 'Saved full sampler/new Adam clock/source clocks')
    pre = normalize_fingerprint(read_artifact(value['PRE_save_fingerprint']))
    actual_pre = metadata_fingerprint(metadata)
    require(actual_pre['state_sha256'] == digest(actual_pre['state'])
        == pre['state_sha256'] == digest(pre['state']),
        'All854 saved descriptors differ from independent actual PRE canonical hashes')
    persistence = read_artifact(value['persistence'])
    require(persistence['kind'] == 'ASR18_actual_streamed_saved_state_matches_independent_pre_save_v1'
        and persistence['passed'] is True and same(persistence['checkpoint'],value['checkpoint'])
        and same(persistence['payload'],value['payload'])
        and persistence['independent_pre_save_state_sha256'] == pre['state_sha256'] == persistence['saved_state_sha256']
        and integer(persistence['optimizer_updates'],1) == 20
        and integer(persistence['weights_verified'],1) == 207
        and integer(persistence['Adam_states_verified'],1) == 215
        and integer(persistence['named_tensors_verified'],1) == 854
        and persistence['sampler_state_sha256'] == value['sampler_state_sha256']
        and same(persistence['journal_prefix'],value['journal_prefix'])
        and same(persistence['saved_semantic_state_verified'],semantic)
        and same(value['saved_semantic_state_verified'],semantic)
        and persistence['actual_GPU_restore_performed'] is False
        and persistence['live_model_optimizer_or_RNG_mutated_here'] is False
        and persistence['independent_post_check_live_fingerprint_comparison_required'] is True
        and persistence['full_fresh_GPU_restore_required_before_overall_diagnostic_pass'] is True
        and same(persistence['source'],record(Path(__file__).with_name('asr18_streamed_persistence_fingerprint_v1.py'))),
        'Actual independent saved-byte/PRE persistence proof plus separate actual fresh restoration')
    check_prefix(value['journal_prefix'])
    return value, metadata, sampler


def checked_binding(pin):
    from asr18_full_run_serial_DEV_9223_v2 import checkpoint_format_view
    def payload(checkpoint, value):
        path = Path(value['path']).resolve()
        require(path.is_relative_to(ROOT.resolve()) and path == Path(checkpoint['path']).resolve().with_suffix('.tensors.bin')
            and same(record(path),value), 'Complete immutable finite payload SHA/size')
    def prefix(value):
        require(type(value) is dict and set(value) == {'path','offset','sha256'}, 'Exact finite journal prefix')
        path = Path(value['path']).resolve(); count = integer(value['offset'])
        require(path.is_relative_to(ROOT.resolve()) and path.stat().st_size >= count, 'Saved finite journal prefix exists')
        hashed = hashlib.sha256()
        with path.open('rb') as stream:
            while count:
                block = stream.read(min(count,4*1024**2)); require(block, 'Complete finite journal prefix')
                hashed.update(block); count -= len(block)
        require(hashed.hexdigest() == value['sha256'], 'Actual finite journal prefix SHA differs')
    value = read_bound(pin); bound(value['native_source'])
    return _checked_binding(value,read_artifact=read_bound,check_process=checked_process,
        check_owner=checked_owner,check_format=checkpoint_format_view,check_payload=payload,check_prefix=prefix)


def build_binding(*, configuration, overall_result, owner_tool_actual0):
    """Return a candidate packet; caller writes and checks its own E binding."""
    overall = read_bound(overall_result); native0 = read_bound(overall['actual_native_process0'])
    recovery0 = read_bound(overall['actual_fresh_recovery_process0']); native = read_bound(native0['report'])
    metadata = read_bound(native['checkpoint']); persistence = read_bound(native['persistence'])
    value = {'kind':KIND,'owner':OWNER,'diagnostic_only':True,'qualified':False,'production_enabled':False,
        'native_source':overall['source'],'configuration':configuration,'overall_result':overall_result,
        'owner_tool_actual0':owner_tool_actual0,'native_report':native0['report'],'native_process0':overall['actual_native_process0'],
        'recovery_report':recovery0['report'],'recovery_process0':overall['actual_fresh_recovery_process0'],
        'optimizer_updates':native['actual_optimizer_updates'],'payload':metadata['payload'],
        'sampler_state_sha256':native['sampler_sha256'],'identity':decode_scalar_tree(select(metadata['tree'],'identity')),
        'saved_semantic_state_verified':persistence['saved_semantic_state_verified']}
    for key in ('checkpoint','PRE_save_fingerprint','persistence','next_draw','journal_prefix'): value[key] = native[key]
    # Full verification occurs via checked_binding after the caller publishes the
    # exact packet. This constructor never fabricates process or tool evidence.
    return value
