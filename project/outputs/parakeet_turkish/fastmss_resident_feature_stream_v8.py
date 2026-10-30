"""Resident original-audio preparation and bounded RAM transport.

The numerical reader, recorded-noise/room transform and Parakeet frontend are
the previously measured single-pass implementations. Only their resource
admission, raw-buffer retirement and feature transport change here. Preparing
future plans never commits a sampler, optimizer, RNG or journal frontier.
"""
from collections import deque
from copy import deepcopy
import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import queue
import struct
import subprocess
import sys
import threading
import time
import traceback
import types

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
ROOT = Path(r'E:\gpu_training_scratchpad')
sys.path[:0] = [str(HERE), str(PROJECT / 'work/research')]
HEADER_LIMIT = 8 * 1024**2
PACKET_LIMIT = 256 * 1024**2
DEFAULT_QUEUE = 1024**3


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def pin(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT) or path.is_relative_to(PROJECT), 'Allowed E/project file only')
    before = path.stat()
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'File changed while binding')
    return {'path': str(path), 'bytes': before.st_size, 'sha256': h.hexdigest()}


def bound(value):
    if isinstance(value, (str, Path)):
        value = pin(value)
    actual = pin(value['path'])
    require(all(actual[k] == value[k] for k in actual), 'Bound file identity changed')
    return json.loads(Path(value['path']).read_text(encoding='utf-8-sig'))


def _read_exact(stream, count):
    data = bytearray(count)
    view = memoryview(data)
    offset = 0
    while offset < count:
        received = stream.readinto(view[offset:])
        if not received:
            raise EOFError('Resident feature pipe closed before a complete frame')
        offset += received
    return data


def _header(stream):
    size = struct.unpack('<Q', _read_exact(stream, 8))[0]
    require(0 < size <= HEADER_LIMIT, 'Bounded resident feature header')
    return json.loads(_read_exact(stream, size))


def _send(stream, header, arrays=()):
    encoded = canonical(header)
    require(0 < len(encoded) <= HEADER_LIMIT, 'Bounded resident feature header')
    stream.write(struct.pack('<Q', len(encoded)))
    stream.write(encoded)
    for array in arrays:
        view = memoryview(array).cast('B')
        for start in range(0, len(view), 1024**2):
            stream.write(view[start:start + 1024**2])
    stream.flush()


def checked_resource_profile(config):
    expected={'kind':'ASR18_resident_RAM_single_live_packet_resource_scheduling_v1',
        'CPU_protocol_source':pin(__file__),'maximum_live_receiver_packets':1,
        'queue_capacity_bytes':268435456,'maximum_packet_bytes':268435456,
        'CPU_prospective_commit_reserve_bytes':2500000000,'CPU_allocation_wait_maximum_seconds':300,
        'CPU_actual_commit_headroom_bytes':1500000000,'native_actual_commit_headroom_bytes':2000000000,
        'native_actual_physical_headroom_bytes':1500000000,'CPU_private_limit_bytes':8000000000,
        'additional_prefetch_host_reservation_bytes':268435456,'freed_reservation_credit_claimed':False}
    require(config.get('resident_RAM_resource_scheduling_policy')==expected,
            'Exact one-live-packet workload and stronger prospective CPU reserve; original guards retained')
    return expected


class AvailableMemoryPolicy:
    """Human-authorized larger RAM budget, with real machine headroom.

    No summation of unused leaf maxima is charged against the current system.
    Actual private memory plus each imminent allocation is admitted instead.
    """
    def __init__(self, private_limit_bytes=8_000_000_000, headroom_bytes=1_500_000_000):
        require(type(private_limit_bytes) is int and 2_000_000_000 <= private_limit_bytes <= 16_000_000_000,
                'Explicit larger resident CPU budget')
        self.private_limit = private_limit_bytes
        self.headroom = headroom_bytes
        self.last = None
        self.peak_private = 0
        self.peak_commit_fraction = 0.
        self.prospective_reserve_bytes = 2_500_000_000
        self.allocation_wait_seconds = 300.
        self.allocation_wait_count = 0
        self.allocation_wait_seconds_total = 0.
        self.last_observed = None

    def __call__(self):
        from resource_watchdog_v2 import sample_memory
        now = time.monotonic()
        if self.last is not None and now - self.last['observed_monotonic'] < .1:
            return self.last
        value = sample_memory(os.getpid(), 'resident_CPU_RAM_preparation')
        worker, system = value['worker_private_memory'], value['system_memory']
        require(value['paired_sampling_within_one_second'] and worker['available'] and system['available'],
                'Fresh actual private/system memory telemetry')
        self.last_observed = value
        self.peak_private = max(self.peak_private, worker['private_bytes'])
        fraction = system['commit_used_bytes'] / system['commit_limit_bytes']
        self.peak_commit_fraction = max(self.peak_commit_fraction, fraction)
        require(worker['private_bytes'] < self.private_limit, 'Resident CPU private budget exhausted')
        require(fraction < .97 and system['commit_available_bytes'] >= self.headroom,
                'Actual machine commit/headroom exhausted')
        self.last = value
        return value

    def admit(self, budget, guard=None):
        require(type(budget) is int and 0 < budget <= self.private_limit, 'Explicit imminent allocation budget')
        began=time.monotonic();waited=False
        while True:
            value=(guard or self)();system=value['system_memory']
            require(value['worker_private_memory']['private_bytes']+budget<=self.private_limit,
                    'Current resident private plus imminent allocation exceeds budget')
            if system['commit_available_bytes']-budget>=self.prospective_reserve_bytes:break
            if not waited:self.allocation_wait_count+=1;waited=True
            elapsed=time.monotonic()-began
            if elapsed>=self.allocation_wait_seconds:
                self.allocation_wait_seconds_total+=elapsed
                raise TimeoutError('Bounded prospective CPU allocation headroom wait exhausted')
            time.sleep(.1)
        elapsed=time.monotonic()-began
        if waited:self.allocation_wait_seconds_total+=elapsed
        return {'sample':value,'forecast_incremental_bytes':budget,'passed':True,
                'policy':'single_live_packet_stronger_prospective_CPU_reserve_v1',
                'prospective_commit_reserve_bytes':self.prospective_reserve_bytes,
                'allocation_wait_seconds':elapsed if waited else 0.}


def _raw_finish_without_full_heap_collection(self, key):
    """NumPy buffers are acyclic: observe their weakrefs before releasing credit.

    A full collection is retained only when an actual buffer remains live; it
    is no longer run unconditionally for every audio row.
    """
    self.guard()
    job = self.jobs[key]
    require(not job['stages'], 'No active raw operation at retirement')
    if any(ref() is not None for _, ref, _ in job['arrays']):
        gc.collect()
    require(all(ref() is None for _, ref, _ in job['arrays']), 'A raw audio buffer is still live')
    del self.jobs[key]
    self.events.append({'event': 'raw_buffers_observed_released', 'job': key})


class _MemoryStore:
    def __init__(self, owner):
        self.owner = owner
        self.arrays = []

    def save(self, name, array):
        import numpy as np
        require(array.flags.c_contiguous and array.dtype in (np.dtype('float32'), np.dtype('bool')),
                'Original contiguous float32/bool feature transport')
        self.arrays.append(array)
        return {'shape': list(array.shape), 'dtype': str(array.dtype),
                'array_sha256': hashlib.sha256(memoryview(array).cast('B')).hexdigest(),
                'transport': {'kind': 'resident_RAM_array_v1', 'slot': len(self.arrays) - 1,
                              'name': name, 'bytes': array.nbytes}}

    def write_metadata(self, name, value):
        return value


def _startup(full_configuration, policy):
    import fastmss_BPE9223_cpu_common_v12 as core
    import fastmss_singlepass_CPU_v1 as single
    import fastmss_BPE9223_recorded_room_train_v3 as augmentation
    core.environment()
    config = bound(full_configuration)
    template = bound(config['CPU_template_configuration'])
    source = template['source_contract_template']
    session = bound(config['session'])
    # Verify the immutable numerical provenance once. The resource evaluator
    # below is explicitly new; no frozen source/environment is edited.
    contract = core.verify_contract(source['numerical_CPU_contract'], session, guard=policy)
    validated = core.bridge.validate_DATA(config['planning_arguments']['DATA_evidence'], policy)
    core.admit = policy.admit
    import fastmss_BPE9223_cpu_common_v3 as augmentation_resource_calls
    augmentation_resource_calls.admit = policy.admit
    np, torch = core.numpy_runtime(frontend=True)
    from transformers import AutoFeatureExtractor
    extractor = AutoFeatureExtractor.from_pretrained(contract['frontend']['base_path'], local_files_only=True)
    require(type(extractor).__name__ == 'ParakeetFeatureExtractor' and
            pin(inspect.getfile(type(extractor))) == contract['frontend']['source'], 'Original numerical frontend')
    from fastmss_original_clip_geometry_overlay_v6 import OriginalRows
    reader = OriginalRows(validated, geometry_overlay=config['original_clip_geometry_overlay'], raw_contract=contract['raw_lifetime_review'],
        private_limit_bytes=policy.private_limit, frontend_enabled=True, guard=policy)
    reader.ledger.finish = types.MethodType(_raw_finish_without_full_heap_collection, reader.ledger)
    augmenter = augmentation.RecordedRoomTrainingAdapter(source['augmentation_policy'], reader=reader,
        guard=policy, ROOT_review=config['ROOT_CPU_execution_review'], configuration=config['CPU_template_configuration'])
    require(augmenter.identity == source['augmentation_identity'], 'Unchanged noise/room numerical policy')
    def clean_reader(row, expected, phase):
        wave = reader(row, expected, phase)
        return wave, core.reference_entry(row, expected, reader)
    clean_reader.finish_row = reader.finish_raw_row
    def original_frontend(wave):
        inputs = dict(extractor([wave], **core.sink.FRONTEND_ARGS))
        require(set(inputs) == {'input_features', 'attention_mask'}, 'Original complete frontend output')
        mask = inputs['attention_mask'].bool()
        arrays = {'input_features': inputs['input_features'].masked_fill(~mask.unsqueeze(-1), 0).detach().cpu().contiguous().numpy(),
                  'attention_mask': mask.detach().cpu().contiguous().numpy()}
        core.check_frontend(arrays, len(wave), np)
        return arrays
    return core, single, source, validated, reader, augmenter, clean_reader, original_frontend


def worker_main(configuration_path, private_limit_bytes):
    """One serial CPU process until explicit normal close, no GPU imports."""
    output = sys.stdout.buffer
    commands = sys.stdin.buffer
    sys.stdout = sys.stderr  # Libraries may print; binary stdout is protocol only.
    began = time.monotonic()
    profile=checked_resource_profile(bound(configuration_path))
    require(private_limit_bytes==profile['CPU_private_limit_bytes'],'Same explicit CPU private budget')
    policy = AvailableMemoryPolicy(private_limit_bytes)
    reader = None
    completed = 0
    try:
        core, single, source, validated, reader, augmenter, clean_reader, frontend = _startup(configuration_path, policy)
        producer = single.identity()
        provenance = {'protocol': pin(__file__), 'original_clip_geometry_overlay': bound(configuration_path)['original_clip_geometry_overlay'],
                      'numerical_singlepass_source': pin(single.__file__),
                      'producer': producer, 'augmentation_identity': source['augmentation_identity'],
                      'resource_policy': 'available_RAM_explicit_process_local_resource_evaluator_v1'}
        _send(output, {'kind': 'ready', 'producer_identity': producer, 'source': provenance,
                       'startup_seconds': time.monotonic() - began})
        previous_final = None
        while True:
            command = _header(commands)
            if command['kind'] == 'close':
                break
            require(command['kind'] == 'prepare_plan', 'Only ordered preparation or normal close')
            plan = bound(command['plan'])
            plan_sha = digest(plan)
            require(plan_sha == command['plan_sha256'], 'Exact submitted plan digest')
            if previous_final is not None:
                require(plan['initial_sampler_sha256'] == previous_final,
                        'Private next plan must continue the prior exact sampler state')
            normalized = core.bridge.normalize_updates(plan, population=validated['retained_rows'], guard=policy)
            store = _MemoryStore({'augmentation_identity': source['augmentation_identity']})
            live_owner = {'kind': 'resident_CPU_producer_identity_v1', 'source': provenance['protocol'],
                          'producer': producer, 'plan_sha256': plan_sha, 'process_exit_claimed': False}
            prepare = single.SinglePassProducer(live_owner, store, normalized, clean_reader=clean_reader,
                weak_transform=augmenter.transform, frontend=lambda wave: augmenter.frontend(frontend, wave), guard=policy)
            for index in range(len(normalized)):
                update_began = time.monotonic()
                store.arrays = []
                metadata = prepare.prepare_update(index)
                require(not reader.ledger.jobs and reader.active_job is None, 'Every full raw row released before transport')
                metadata['kind'] = 'FastMSS_resident_RAM_prepared_update_v1'
                arrays = store.arrays
                total = sum(a.nbytes for a in arrays)
                require(len(arrays) == 2 * len(metadata['rows']) and 0 < total <= PACKET_LIMIT,
                        'Bounded full update packet with all features/masks')
                token = '%s:%s:%s' % (producer['pid'], plan_sha, index)
                descriptors = [{'shape': list(a.shape), 'dtype': str(a.dtype), 'bytes': a.nbytes,
                                'sha256': hashlib.sha256(memoryview(a).cast('B')).hexdigest()} for a in arrays]
                packet = {'kind': 'update', 'token': token, 'plan_sha256': plan_sha, 'update_index': index,
                          'metadata': metadata, 'metadata_sha256': digest(metadata), 'source': provenance,
                          'array_descriptors': descriptors, 'array_bytes': total,
                          'timings': {'preparation_seconds': time.monotonic() - update_began}}
                _send(output, packet, arrays)
                completed += 1
                # Counters survive; per-row diagnostic event lists and already
                # transferred arrays do not accumulate over a full training run.
                reader.ledger.events.clear()
                store.arrays = []
                del arrays, metadata, packet, descriptors
            previous_final = digest(plan['final_private_planning_sampler'])
            _send(output, {'kind': 'plan_complete', 'plan_sha256': plan_sha,
                           'updates': len(normalized), 'final_sampler_sha256': previous_final,
                           'process_exit_claimed': False})
            del prepare, store, normalized, plan
        reader.close()
        reader = None
        _send(output, {'kind': 'closed', 'completed_updates': completed,
                       'wall_seconds': time.monotonic() - began,
                       'peak_private_bytes': policy.peak_private,
                       'peak_commit_fraction': policy.peak_commit_fraction,
                       'allocation_wait_count':policy.allocation_wait_count,
                       'allocation_wait_seconds':policy.allocation_wait_seconds_total,
                       'prospective_commit_reserve_bytes':policy.prospective_reserve_bytes})
        return 0
    except BaseException as exc:
        traceback.print_exc(file=sys.stderr)
        try:
            _send(output, {'kind': 'error', 'message': repr(exc), 'completed_updates': completed,
                'actual_resource_telemetry':{'last_observed_memory':policy.last_observed,
                    'peak_private_bytes':policy.peak_private,'allocation_wait_count':policy.allocation_wait_count,
                    'allocation_wait_seconds':policy.allocation_wait_seconds_total,
                    'prospective_commit_reserve_bytes':policy.prospective_reserve_bytes}})
        except BaseException:
            pass
        return 1
    finally:
        if reader is not None:
            reader.close()


class ResidentFeatureProducer:
    """Parent-side bounded receiver, asynchronous plan sender and release credit."""
    def __init__(self, configuration_pin, output_directory, queue_bytes=DEFAULT_QUEUE,
                 private_limit_bytes=8_000_000_000, guard=lambda: None):
        require(type(queue_bytes) is int and PACKET_LIMIT <= queue_bytes <= 4 * 1024**3, 'Bounded RAM feature queue')
        self.queue_bytes, self.guard = queue_bytes, guard
        self.folder = Path(output_directory).resolve()
        require(self.folder.is_relative_to(ROOT) and not self.folder.exists(), 'Fresh E resident producer log directory')
        self.folder.mkdir(parents=True)
        self.configuration = configuration_pin
        self.protocol_source = pin(__file__)
        config = bound(configuration_pin)
        profile=checked_resource_profile(config)
        require(queue_bytes==profile['queue_capacity_bytes'] and private_limit_bytes==profile['CPU_private_limit_bytes'],
                'Unchanged full packet/queue byte and CPU private reservations')
        self.max_live_packets=profile['maximum_live_receiver_packets']
        self.peak_live_packets=0;self.credit_wait_seconds=0.
        contract = bound(bound(config['CPU_template_configuration'])['source_contract_template']['numerical_CPU_contract'])
        environment = os.environ.copy()
        environment.update(CUDA_VISIBLE_DEVICES='-1', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
        for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            environment[name] = '1'
        for name in ('TEMP', 'TMP', 'HF_HOME', 'TORCH_EXTENSIONS_DIR'):
            require(name in environment and Path(environment[name]).resolve().is_relative_to(ROOT), 'Apply E use_storage.ps1 before launching')
        self.log = (self.folder / 'CPU_stderr.log').open('xb')
        self.process = subprocess.Popen([contract['python']['path'], '-Xutf8', '-B', str(Path(__file__).resolve()),
            '--worker', '--configuration', configuration_pin['path'] if isinstance(configuration_pin, dict) else str(configuration_pin),
            '--private-limit', str(private_limit_bytes)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.log, env=environment, bufsize=0, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        self.condition = threading.Condition()
        self.packets = deque()
        self.live = {}
        self.used_bytes = 0
        self.peak_queue_bytes = 0
        self.error = None
        self.producer_identity = None
        self.source = None
        self.ready = threading.Event()
        self.closed_message = None
        self.plan_completions = {}
        self.commands = queue.Queue(maxsize=2)
        self.submitted = set()
        self._sender = threading.Thread(target=self._send_commands, daemon=True)
        self._receiver = threading.Thread(target=self._receive, daemon=True)
        self._sender.start()
        self._receiver.start()

    def _send_commands(self):
        try:
            while True:
                command = self.commands.get()
                if command['kind']=='abort_sender':return
                _send(self.process.stdin, command)
                if command['kind'] == 'close':
                    return
        except BaseException as exc:
            self._fail(exc)

    def _fail(self, exc):
        with self.condition:
            self.error = exc
            self.ready.set()
            self.condition.notify_all()

    def _receive(self):
        try:
            import numpy as np
            while True:
                frame = _header(self.process.stdout)
                kind = frame['kind']
                if kind == 'ready':
                    from fastmss_owned_process_closure_v2 import identity
                    observed = identity(frame['producer_identity']['pid'])
                    require(observed == frame['producer_identity'] and
                            (observed['pid'] == self.process.pid or observed['ppid'] == self.process.pid),
                            'Actual CPU child or its directly owned Windows interpreter shim')
                    self.producer_identity, self.source = frame['producer_identity'], frame['source']
                    (self.folder / 'actual_CPU_ready.json').write_bytes(canonical(frame))
                    require(frame['source']['protocol'] == self.protocol_source, 'Unchanged CPU protocol source')
                    self.ready.set()
                elif kind == 'update':
                    require(frame['metadata_sha256'] == digest(frame['metadata']), 'RAM metadata digest')
                    count = frame['array_bytes']
                    require(type(count) is int and 0 < count <= PACKET_LIMIT and
                            count == sum(d['bytes'] for d in frame['array_descriptors']), 'Bounded complete frame bytes')
                    with self.condition:
                        credit_began=time.monotonic()
                        while (self.used_bytes+count>self.queue_bytes or len(self.live)>=self.max_live_packets) and self.error is None:
                            self.condition.wait(.25)
                        self.credit_wait_seconds+=time.monotonic()-credit_began
                        if self.error is not None:
                            return
                        self.used_bytes += count
                        self.peak_queue_bytes = max(self.peak_queue_bytes, self.used_bytes)
                    arrays = []
                    for spec in frame['array_descriptors']:
                        dtype = np.dtype(spec['dtype'])
                        require(dtype in (np.dtype('float32'), np.dtype('bool')) and
                                all(type(n) is int and n > 0 for n in spec['shape']) and
                                math.prod(spec['shape']) * dtype.itemsize == spec['bytes'], 'Exact non-object array geometry')
                        data = _read_exact(self.process.stdout, spec['bytes'])
                        require(hashlib.sha256(data).hexdigest() == spec['sha256'], 'Complete RAM feature bytes SHA')
                        array = np.frombuffer(data, dtype=dtype).reshape(spec['shape'])
                        arrays.append(array)
                    rows = frame['metadata']['rows']
                    require(len(arrays) == 2 * len(rows), 'Every row feature and mask')
                    packed = []
                    for i, row in enumerate(rows):
                        a, m = arrays[2 * i:2 * i + 2]
                        require(a.dtype == np.float32 and m.dtype == np.bool_ and
                                list(a.shape) == row['input_features']['shape'] and
                                list(m.shape) == row['attention_mask']['shape'] and
                                row['input_features']['array_sha256'] == frame['array_descriptors'][2 * i]['sha256'] and
                                row['attention_mask']['array_sha256'] == frame['array_descriptors'][2 * i + 1]['sha256'],
                                'Row metadata and RAM payload identity')
                        packed.append({'input_features': a, 'attention_mask': m})
                    packet = {**frame, 'update': frame['metadata'], 'arrays': packed}
                    with self.condition:
                        require(packet['token'] not in self.live, 'Never duplicate a live packet token')
                        self.live[packet['token']] = {'bytes': count, 'delivered': False}
                        self.peak_live_packets=max(self.peak_live_packets,len(self.live))
                        self.packets.append(packet)
                        self.condition.notify_all()
                    del arrays, packed, packet, data
                elif kind == 'plan_complete':
                    self.plan_completions[frame['plan_sha256']] = frame
                elif kind == 'closed':
                    self.closed_message = frame
                    return
                elif kind == 'error':
                    (self.folder/'actual_CPU_failure.json').write_bytes(canonical(frame))
                    raise RuntimeError('Resident CPU preparation failed: ' + frame['message'])
                else:
                    raise ValueError('Unknown resident preparation frame')
        except BaseException as exc:
            self._fail(exc)

    def submit(self, plan_pin):
        plan = bound(plan_pin)
        sha = digest(plan)
        require(sha not in self.submitted and self.error is None, 'Distinct ordered plan on healthy producer')
        self.commands.put_nowait({'kind': 'prepare_plan', 'plan': plan_pin, 'plan_sha256': sha})
        self.submitted.add(sha)
        return sha

    def get(self, timeout=300):
        deadline = time.monotonic() + timeout
        with self.condition:
            while not self.packets:
                if self.error is not None:
                    raise self.error
                if self.process.poll() is not None:
                    raise RuntimeError('CPU process ended without the requested packet; see CPU_stderr.log')
                if time.monotonic() > deadline:
                    raise TimeoutError('Resident CPU update packet timed out')
                self.condition.wait(.25)
                self.guard()
            packet = self.packets.popleft()
            self.live[packet['token']]['delivered'] = True
            return packet

    def next_update(self, plan_sha, update_index, timeout=300):
        packet = self.get(timeout)
        require(packet['plan_sha256'] == plan_sha and packet['update_index'] == update_index,
                'Actual next plan/update, never speculative authoritative consumption')
        return packet

    def release(self, token):
        with self.condition:
            value = self.live.pop(token)
            require(value['delivered'], 'Cannot release an undelivered RAM packet')
            self.used_bytes -= value['bytes']
            self.condition.notify_all()

    def close(self, timeout=90):
        require(not self.live and not self.packets, 'Release or explicitly abort prepared packets before normal close')
        if self.error is not None:
            error = self.error
            self.abort()
            raise error
        self.commands.put({'kind': 'close'}, timeout=10)
        code = self.process.wait(timeout=timeout)
        self._receiver.join(timeout=10)
        self._sender.join(timeout=10)
        require(not self._receiver.is_alive() and not self._sender.is_alive(),'Actual resident CPU transport threads joined')
        self.process.stdin.close();self.process.stdout.close()
        self.log.close()
        require(code == 0 and self.closed_message is not None and self.error is None,
                'Actual CPU wait0 plus complete normal report required')
        result = {**self.closed_message, 'actual_Popen_wait_code': code,
                  'producer_identity': self.producer_identity, 'source': self.source,
                  'queue_capacity_bytes': self.queue_bytes, 'peak_queue_bytes': self.peak_queue_bytes,
                  'maximum_live_receiver_packets':self.max_live_packets,'peak_live_receiver_packets':self.peak_live_packets,
                  'receiver_credit_wait_seconds':self.credit_wait_seconds,'transport_threads_joined':True,'pipe_handles_closed':True}
        (self.folder / 'actual_CPU_process0.json').write_bytes(canonical(result))
        return result

    def abort(self):
        self._fail(RuntimeError('Resident producer explicitly aborted; no normal exit claimed'))
        try:self.commands.put_nowait({'kind':'abort_sender'})
        except queue.Full:pass
        if self.producer_identity is not None and self.producer_identity['pid'] != self.process.pid:
            import psutil
            try:
                child = psutil.Process(self.producer_identity['pid'])
                if abs(child.create_time() - self.producer_identity['create_time']) < .001:
                    child.terminate()
                    child.wait(timeout=30)
            except psutil.NoSuchProcess:
                pass
        if self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=30)
        self._receiver.join(timeout=10);self._sender.join(timeout=10)
        require(not self._receiver.is_alive() and not self._sender.is_alive(),'Aborted transport threads genuinely joined')
        self.process.stdin.close();self.process.stdout.close()
        self.log.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--configuration', required=True)
    parser.add_argument('--private-limit', type=int, default=8_000_000_000)
    args = parser.parse_args()
    require(args.worker, 'Explicit CPU resident worker entry point')
    raise SystemExit(worker_main(args.configuration, args.private_limit))
