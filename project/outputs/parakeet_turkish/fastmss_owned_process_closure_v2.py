"""Owned process observations, distinct from later outer tool completion.

No model/audio runtime imports. The only constructor requires a real Popen
handle owned by this parent; consumers still verify their scientific boundary.
These proofs never satisfy the final DATA actual-tool-completion contract.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(r'E:\gpu_training_scratchpad')
PROJECT = Path(__file__).resolve().parents[2]
KINDS = {'ASR18_owned_CPU_process_completion_v1', 'ASR18_owned_GPU_process_completion_v1'}

def require(value, message):
    if not value: raise ValueError(message)

def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()

def same(a, b): return a == b and canonical(a) == canonical(b)

def scoped(path):
    value = Path(path).resolve()
    require(value.is_relative_to(ROOT.resolve()) or value.is_relative_to(PROJECT), 'Existing project/E input only')
    return value

def record(path, guard=lambda: None):
    path = scoped(path); before = path.stat(); sha = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(1024**2): guard(); sha.update(block)
    require((before.st_size, before.st_mtime_ns) == (path.stat().st_size, path.stat().st_mtime_ns), 'File changed while binding')
    return {'path': str(path), 'bytes': before.st_size, 'sha256': sha.hexdigest()}

def read(pin, guard=lambda: None):
    require(type(pin) is dict and set(pin) in ({'path','bytes','sha256'}, {'path','bytes','sha256','mtime_ns'}), 'Exact binding')
    require(type(pin['bytes']) is int and 0 < pin['bytes'] <= 16*1024**2, 'Bounded proof/source only')
    require(same(record(pin['path'], guard), {k: pin[k] for k in ('path','bytes','sha256')}), 'Proof/source hash differs')
    if 'mtime_ns' in pin: require(type(pin['mtime_ns']) is int and scoped(pin['path']).stat().st_mtime_ns == pin['mtime_ns'], 'Proof timestamp differs')
    return json.loads(scoped(pin['path']).read_text(encoding='utf-8-sig'))

def identity(pid=None):
    import psutil
    p = psutil.Process(os.getpid() if pid is None else pid)
    return {'pid': p.pid, 'ppid': p.ppid(), 'create_time': p.create_time()}

def _identity(value):
    require(type(value) is dict and set(value) == {'pid','ppid','create_time'} and type(value['pid']) is int and value['pid'] > 0 and type(value['ppid']) is int and value['ppid'] > 0 and type(value['create_time']) in (int,float) and value['create_time'] > 0, 'Exact non-boolean process identity')

def dead(value):
    import psutil
    _identity(value)
    try: return abs(psutil.Process(value['pid']).create_time() - value['create_time']) > .001
    except psutil.NoSuchProcess: return True

def clean_guard(pin, producer, guard):
    v = read(pin, guard)
    require(v['pid'] == producer['pid'] and type(v['actual_exit_code']) is int and v['actual_exit_code'] == 0 and v['monitor_joined'] is True and v['monitor_clean'] is True and v['observed']['monitor_error'] is None and type(v['observed']['samples']) is int and v['observed']['samples'] >= 3, 'Actual clean joined child guard with >=3 samples')
    return v

def parent_sample(value, observer):
    require(type(value) is dict and value['paired_sampling_within_one_second'] is True, 'Fresh paired parent resource observation')
    w,s = value['worker_private_memory'], value['system_memory']
    require(w['pid']==observer['pid'] and w['available'] is True and s['available'] is True and w['private_bytes'] < 8_000_000_000 and s['commit_used_bytes'] < .90*s['commit_limit_bytes'] and s['commit_available_bytes'] >= 2_000_000_000, 'Exact parent PID and unchanged parent/global resource protection')
    # Native watchdog samples include PID under process_id or pid in some
    # versions. The observer additionally records identity at handle wait.
    for key in ('pid','process_id'):
        if key in value: require(value[key] == observer['pid'], 'Parent sample belongs to another process')

def _persistence_contract(value, pointer, guard=lambda: None):
    """Metadata part only; live PID/nonce/witness checks are always separate.

    A persisted PRE-save equality proof is not a GPU restore. The final native
    caller must keep the explicit restore obligation until a separately owned
    restore-only worker actually completes its exact-state and next-draw proof.
    """
    require(value['kind']=='ASR18_resident_native_persistence_verified_idle_boundary_v2'
            and value['checkpoint_persistence_verified'] is True
            and value['checkpoint_exact_restore_verified'] is False
            and value['deferred_restore_required_before_overall_diagnostic_pass'] is True,
            'Persistence idle must explicitly deny GPU restore and retain deferred restore obligation')
    proof=read(value['checkpoint_persistence_proof'],guard)
    from asr18_streamed_persistence_fingerprint_v1 import normalize_fingerprint
    source=record(Path(__file__).with_name('asr18_streamed_persistence_fingerprint_v1.py'),guard)
    require(proof['kind']=='ASR18_actual_streamed_saved_state_matches_independent_pre_save_v1'
            and proof['passed'] is True and same(proof['source'],source)
            and same(proof['worker_source'],value['worker_source'])
            and same(proof['checkpoint'],pointer['checkpoint']) and same(proof['payload'],pointer['payload'])
            and type(proof['optimizer_updates']) is int and proof['optimizer_updates']==pointer['committed_updates']
            and type(proof['weights_verified']) is int and proof['weights_verified']==207
            and type(proof['Adam_states_verified']) is int and proof['Adam_states_verified']>=207
            and type(proof['named_tensors_verified']) is int and 0<proof['named_tensors_verified']<=4096
            and proof['sampler_state_sha256']==pointer['sampler_state_sha256']
            and same(proof['journal_prefix'],pointer['journal_prefix'])
            and same(proof['saved_semantic_state_verified'],pointer['saved_semantic_state_verified']),
            'Actual source-bound persisted207/Adam/RNG/pointer/sampler/prefix proof differs')
    require(type(proof['maximum_payload_read_bytes']) is int and proof['maximum_payload_read_bytes']==4*1024**2
            and proof['actual_GPU_restore_performed'] is False
            and proof['live_model_optimizer_or_RNG_mutated_here'] is False
            and proof['independent_post_check_live_fingerprint_comparison_required'] is True
            and proof['full_fresh_GPU_restore_required_before_overall_diagnostic_pass'] is True
            and proof['live_unchanged_after_persistence'] is True,
            'Bounded read-only persistence proof cannot discharge restore or mutate live state')
    caller=pointer['identity']['caller_identity']
    require(same(proof['pre_save_fingerprint'],caller['pre_save_fingerprint'])
            and same(proof['next_draw_packet'],caller['next_draw_packet']),
            'Immutable checkpoint identity must bind exact independent PRE-save and next-draw artifacts')
    expected=normalize_fingerprint(read(proof['pre_save_fingerprint'],guard))
    require(type(expected['state']['updates']) is int and expected['state']['updates']==pointer['committed_updates']
            and len(expected['state']['weights'])==207 and len(expected['state']['Adam_partitions'])==3
            and expected['state']['clocks']==[clock+pointer['committed_updates'] for clock in (64250,2000,62250)]
            and pointer['saved_semantic_state_verified']['passed'] is True,
            'Independent PRE-save named207/threeAdam frontier and clocks must match the coherent pointer')
    require(type(proof['independent_pre_save_state_sha256']) is str
            and len(proof['independent_pre_save_state_sha256'])==64
            and proof['independent_pre_save_state_sha256']==proof['saved_state_sha256']==proof['live_after_state_sha256']==expected['state_sha256'],
            'Independent PRE-save, actual saved and actual post-check live state must agree')
    read(proof['next_draw_packet'],guard)
    return proof


def verify_resident_idle(value, *, pointer, proof_binding, manifest=None, phase=None, guard=lambda: None):
    """Live coherent idle is not exit. A native-owner callback remains required.

    This structural check does not attest a command nonce or freeze CUDA work;
    the parent must validate a fresh nonce acknowledgment and current liveness
    at its actual cleanup/next-window boundary.
    """
    import psutil
    require(value['owner']=='/root/memory_diagnosis' and 'actual_native_exit_code' not in value and 'actual_tool_chunk_id' not in value,'Distinct live idle boundary, never fake CLOSED0')
    if value['kind']=='ASR18_resident_native_coherent_idle_boundary_v1':
        require(value['checkpoint_exact_restore_verified'] is True,'Historical exact-restored idle must retain its actual restore requirement')
    else:
        _persistence_contract(value,pointer,guard)
    for key in ('producer','observer'): _identity(value[key])
    producer=value['producer'];p=psutil.Process(producer['pid'])
    require(abs(p.create_time()-producer['create_time'])<=.001 and p.ppid()==producer['ppid'] and value['live_producer'] is True,'Exact currently live native identity')
    require(value['producer']['ppid']==value['observer']['pid'] or value['launcher_PID']==value['producer']['ppid'],'Owned live native/shim relation')
    for key in ('worker_source','parent_source'):
        require(same(record(value[key]['path'],guard),value[key]),'Resident source binding changed')
    require(type(value['request_nonce']) is str and 16<=len(value['request_nonce'])<=128 and value['request_nonce_acknowledged'] is True,'Distinct explicitly acknowledged idle command')
    for key in ('all_window_updates_committed','cuda_synchronized','no_inflight_native_or_forward_hooks'):
        require(value[key] is True,'Actual coherent quiescent native state required')
    bound=read(value['checkpoint_pointer'],guard)
    require(same(bound,pointer) and same(value['checkpoint'],pointer['checkpoint']) and same(value['payload'],pointer['payload']) and value['optimizer_updates']==pointer['committed_updates'] and type(value['optimizer_updates']) is int and value['committed_window_updates']==pointer['committed_window_updates'] and type(value['committed_window_updates']) is int and value['sampler_state_sha256']==pointer['sampler_state_sha256'] and same(value['journal_prefix'],pointer['journal_prefix']),'Idle checkpoint/full semantic prefix differs')
    if manifest is not None:require(same(value['window_manifest'],manifest),'Idle window manifest differs')
    if phase is not None:require(value['phase']==phase,'Idle phase differs')
    parent_sample(value['parent_guard_observation'],value['observer'])
    require(value['actual_GPU_process_closed'] is False,'Live idle must not claim actual exit')
    require(same(read(proof_binding,guard),value),'Exact immutable resident idle proof binding')
    witness_path=Path(value['live_witness_path']).resolve()
    require(witness_path.is_relative_to(ROOT.resolve()) and witness_path.stat().st_size<=64*1024,'Bounded mutable live witness on E only')
    # Exact parent atomically replaces this witness. It deliberately has no
    # immutable content pin: freshness and current nonce/state are mandatory.
    witness=json.loads(witness_path.read_text(encoding='utf-8-sig'))
    require(witness['kind']=='ASR18_resident_native_live_guard_witness_v1' and witness['owner']==value['owner'] and same(witness['idle_proof'],proof_binding) and same(witness['parent_source'],value['parent_source']) and same(witness['producer'],producer) and same(witness['observer'],value['observer']) and witness['request_nonce']==value['request_nonce'],'Current witness identity/immutable nonce acknowledgment differs')
    require(witness['command_state']=='coherent_idle_CPU_preparation' and witness['resume_command_issued'] is False and witness['parent_monitor_live'] is True,'Resident worker no longer quiescent under live parent')
    now=time.monotonic()
    for observation in (witness,witness['parent_guard_observation'],witness['native_guard_observation']):
        require(type(observation['observed_monotonic']) in (int,float) and 0<=now-observation['observed_monotonic']<=2,'Fresh <=2s live resident witness and paired samples')
    parent_sample(witness['parent_guard_observation'],value['observer'])
    parent_sample(witness['native_guard_observation'],producer)
    observer=psutil.Process(value['observer']['pid'])
    require(abs(observer.create_time()-value['observer']['create_time'])<=.001,'Witness parent identity must remain live')
    return deepcopy(value)

def verify(value, *, kind, owner, report=None, guard=lambda: None):
    require(kind in KINDS and value['kind'] == kind and value['owner'] == owner, 'Narrow owned process kind/owner required')
    require(not any('tool' in key.lower() for key in value), 'Process proof cannot claim an outer tool result')
    require(type(value['actual_Popen_wait_code']) is int and value['actual_Popen_wait_code'] == 0 and type(value['actual_native_exit_code']) is int and value['actual_native_exit_code'] == 0 and value['parent_observed_wait'] is True and value['producer_dead_after_wait'] is True, 'Actual handle/native exit0 required')
    producer, observer, launcher = value['producer'], value['observer'], value['launcher']
    for v in (producer,observer,launcher): _identity(v)
    require(launcher['ppid'] == observer['pid'] and (same(producer,launcher) or producer['ppid'] == launcher['pid']) and producer['create_time'] >= observer['create_time'] and launcher['create_time'] >= observer['create_time'], 'Owned Popen/native shim identity relation')
    require(dead(producer) and dead(launcher), 'Actual owned producer/launcher must remain closed')
    for key in ('worker_source','parent_source','guard_source','observer_source'):
        pin=value[key]; require(type(pin['bytes']) is int and 0 < pin['bytes'] <= 256*1024 and same(record(pin['path'],guard),pin), 'Exact bound implementation source')
    require(same(value['observer_source'],record(__file__,guard)), 'Process observer source differs')
    result=read(value['report'],guard)
    if report is not None: require(same(value['report'],report), 'Completion binds another report')
    require(same(result['producer'],producer) and same(result['worker_source'],value['worker_source']) and same(result['parent_source'],value['parent_source']), 'Report/actual source and producer identity differ')
    require(result['passed'] is True, 'Successful source-bound report required')
    clean_guard(value['guard_closure'],producer,guard); parent_sample(value['parent_guard_observation'],observer)
    return deepcopy(value)

def observe(child, *, kind, owner, report, worker_source, parent_source, guard_source,
            guard_closure, launcher_identity, parent_guard_observation, output,
            guard=lambda: None, timeout=10):
    require(isinstance(child,subprocess.Popen) and kind in KINDS, 'Actual owned Popen handle, never status inference')
    observer=identity(); require(launcher_identity['pid']==child.pid and launcher_identity['ppid']==os.getpid(), 'Capture own launcher identity while live')
    require(type(parent_guard_observation.get('observed_monotonic')) in (int,float) and 0<=time.monotonic()-parent_guard_observation['observed_monotonic']<=1, 'Fresh parent sample at actual wait observation')
    code=child.wait(timeout=timeout); require(type(code) is int and code==0 and child.poll()==0, 'Actual owned handle exit0')
    result=read(report,guard);producer=result['producer']
    value={'kind':kind,'owner':owner,'report':report,'producer':producer,'observer':observer,
        'launcher':deepcopy(launcher_identity),'worker_source':worker_source,'parent_source':parent_source,
        'guard_source':guard_source,'observer_source':record(__file__,guard),'guard_closure':guard_closure,
        'actual_Popen_wait_code':code,'actual_native_exit_code':code,'parent_observed_wait':True,
        'producer_dead_after_wait':dead(producer),'parent_guard_observation':parent_guard_observation}
    verify(value,kind=kind,owner=owner,report=report,guard=guard)
    output=Path(output).resolve();require(output.is_relative_to(ROOT.resolve()),'New process proof E only')
    with output.open('xb') as stream:stream.write(canonical(value));stream.flush();os.fsync(stream.fileno())
    return record(output,guard)
