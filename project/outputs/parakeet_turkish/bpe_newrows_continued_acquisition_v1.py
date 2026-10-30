"""INACTIVE acquired512-row continuation; no source admission or GPU launch.

The frozen phase1 loader audits/loads all207 weights and the actual3-table
AdamK at sourceLR1e-4 before only its live LR changes. Encoder LoRA and the
other204 learned tensors stay frozen. This is a new declared row-LR phase,
not an original207 or phase1 exact resume. Actual CLOSED/data/ASR admission
and opaque sampler/RNG restoration remain caller responsibilities.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path

TABLE_NAMES = ('decoder.embedding.weight', 'joint.head.weight', 'joint.head.bias')
BPE_ROWS = (8199, 8711)
PHASE = 'HF_TSC_BPE512_new_rows_continued_acquisition_v1'
STATE_KIND = 'HF_TSC_BPE512_new_rows_continued_acquisition_state_v1'
PHASE_KIND = STATE_KIND
IDENTITY_KIND = 'HF_TSC_BPE512_new_rows_continued_acquisition_identity_v1'
SOURCE_UTILITY_SHA = 'bc26638f4874aaa5e92029e86c30bef0624a477514957d8f8d054d7e92088724'


def require(ok, message):
    if not ok: raise ValueError(message)


class BPEContinuedAcquisition:
    def __init__(self, model, phase1_state, original207_state, *, original_source_record,
                 phase1_record, active_row_lr=1e-3, torch_module=None):
        if torch_module is None:
            import torch as torch_module
        self.torch = torch_module
        require(type(active_row_lr) in (int, float) and active_row_lr in (1e-3, 3e-4),
            'Only declared rowLR1e-3 or3e-4 is admitted')
        path = Path(__file__).with_name('bpe_newrows_acquisition_v1.py')
        require(hashlib.sha256(path.read_bytes()).hexdigest() == SOURCE_UTILITY_SHA,
            'Frozen strict phase1 loader changed')
        from bpe_newrows_acquisition_v1 import BPEOnlyAcquisition, checked_parked_record
        self.original_record = checked_parked_record(original_source_record)
        self.phase1_record = checked_parked_record(phase1_record)
        require(self.original_record['path'] != self.phase1_record['path'], 'Separate immutable paired sources required')
        self.source_step = original207_state.get('step')
        self.source_acquisition_updates = phase1_state.get('acquisition_optimizer_updates')
        require(original207_state.get('schema') == 2 and type(self.source_step) is int and self.source_step > 0
            and original207_state.get('selected_step') == self.source_step
            and type(self.source_acquisition_updates) is int and self.source_acquisition_updates > 0,
            'Actual original selected207 plus positive acquired K required')
        self.names = tuple(original207_state['weights'])
        require(len(self.names) == len(set(self.names)) == 207
            and tuple(phase1_state['phase_identity']['source_weight_names']) == self.names
            and original207_state['lora_metadata'] == phase1_state['lora_metadata'],
            'Paired207 names/LoRA layout differ')
        self.model = model
        self.parameters = dict(model.named_parameters())
        self._compare_source_weights(phase1_state['weights'], original207_state['weights'])
        runtime = phase1_state.get('runtime_state', {})
        require(runtime.get('source_selected_step') == self.source_step
            and runtime.get('step') == self.source_step + self.source_acquisition_updates,
            'Phase1 runtime S/K frontier differs')
        self.source_phase_identity = copy.deepcopy(phase1_state['phase_identity'])
        self._base = BPEOnlyAcquisition(model, self.names, self.original_record,
            source_lora_metadata=original207_state['lora_metadata'], lr=1e-4, torch_module=self.torch)
        # Strict source group1e-4,207 weights, complete finite/protected3 moments
        # and scalar K clocks are validated BEFORE any live LR change.
        self.inherited_runtime = self._base.load_phase_state(phase1_state)
        require(self._base.optimizer_updates_completed == self.source_acquisition_updates,
            'Actual inherited table Adam clock differs')
        self.tables = self._base.tables
        self.optimizer = self._base.optimizer
        self.lora_metadata = self._base.lora_metadata
        self.active_row_lr = float(active_row_lr)
        self.optimizer.param_groups[0]['lr'] = self.active_row_lr
        self._base.base_lr = self.active_row_lr
        self._base.declared_optimizer_group = copy.deepcopy({key:value for key,value in
            self.optimizer.param_groups[0].items() if key != 'params'})
        self._local_updates = 0
        self._incomplete_update = False
        self.fixed_weight_sha256 = self._fixed_digest(phase1_state['weights'])
        self._check_contract()

    @property
    def optimizer_updates_completed(self): return self._local_updates

    @property
    def cumulative_step(self): return self.source_step + self.source_acquisition_updates + self._local_updates

    def _bytes(self, tensor):
        return tensor.detach().cpu().contiguous().reshape(-1).view(self.torch.uint8).numpy().tobytes()

    def _same_bytes(self, left, right):
        return left.shape == right.shape and left.dtype == right.dtype and self._bytes(left) == self._bytes(right)

    def _compare_source_weights(self, acquired, original):
        require(set(acquired) == set(original) == set(self.names), 'Both complete207 weight sets required')
        for name in self.names:
            before, after = original[name], acquired[name]
            require(self.torch.is_tensor(before) and self.torch.is_tensor(after)
                and before.dtype == after.dtype == self.torch.float32
                and bool(self.torch.isfinite(before).all()) and bool(self.torch.isfinite(after).all()),
                'Paired learned tensor FP32/finiteness differs: ' + name)
            if name not in TABLE_NAMES:
                require(self._same_bytes(before, after), 'Frozen original204 weight bytes changed: ' + name)
            else:
                require(before.shape[0] == (8199 if name == TABLE_NAMES[0] else 8204)
                    and after.shape[0] == (8711 if name == TABLE_NAMES[0] else 8716)
                    and before.shape[1:] == after.shape[1:]
                    and self._same_bytes(before[:8199], after[:8199])
                    and self._same_bytes(before[8199:], after[8711:]),
                    'Protected old/six/blank/duration bytes changed: ' + name)

    def _fixed_digest(self, weights):
        digest = hashlib.sha256()
        for name in self.names:
            value = weights[name]
            digest.update(name.encode()); digest.update(str(value.dtype).encode()); digest.update(str(tuple(value.shape)).encode())
            if name in TABLE_NAMES:
                digest.update(self._bytes(value[:8199])); digest.update(self._bytes(value[8711:]))
            else: digest.update(self._bytes(value))
        return digest.hexdigest()

    def _check_contract(self):
        self._base._check_contract()
        require(self.optimizer.param_groups[0]['lr'] == self.active_row_lr
            and self.active_row_lr in (1e-3, 3e-4)
            and self._base.optimizer_updates_completed == self.source_acquisition_updates + self._local_updates
            and set(self.optimizer.state) == set(self.tables), 'Continued row-only LR/state/count contract differs')

    def _require_complete(self):
        require(not self._incomplete_update, 'Failed/partial Adam update; only validated own-phase reload may recover')

    def zero_grad(self):
        self._require_complete(); self._check_contract(); self._base.zero_grad()

    def step(self, max_norm=1.):
        self._require_complete(); self._check_contract()
        require(max_norm == 1., 'Declared one three-table global clip norm is1')
        norm = self._base.mask_then_clip(max_norm)
        self._incomplete_update = True
        self.optimizer.step()
        self._local_updates += 1
        self._base.optimizer_updates_completed = self.source_acquisition_updates + self._local_updates
        self._check_contract(); self._incomplete_update = False
        return norm

    def phase_identity(self):
        identity = self._base.phase_identity()
        return {**identity, 'kind': IDENTITY_KIND,
            'utility_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'strict_source_loader_sha256': SOURCE_UTILITY_SHA,
            'source_step': self.source_step, 'source_acquisition_updates': self.source_acquisition_updates,
            'source_phase1_artifact': dict(self.phase1_record), 'source_row_lr':1e-4,
            'active_row_lr':self.active_row_lr, 'source_moments_or_clock_reset':False,
            'source_phase_identity_sha256':hashlib.sha256(json.dumps(self.source_phase_identity,
                sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest(),
            'frozen_original204_and_protected_rows_sha256':self.fixed_weight_sha256,
            'local_N_distinct_from_Adam_K_plus_N':True, 'encoder_LoRA_active':False,
            'same_phase_utility_restore_supported':True, 'external_worker_resume_admission_supported':False}

    def _audit_optimizer(self, optimizer, total):
        groups = optimizer.get('param_groups', [])
        require(len(groups) == 1 and groups[0]['params'] == [0,1,2]
            and {k:v for k,v in groups[0].items() if k!='params'} == self._base.declared_optimizer_group
            and set(optimizer.get('state',{})) == {0,1,2}, 'Complete same three-table optimizer required')
        for identifier, parameter in zip((0,1,2), self.tables):
            moments = optimizer['state'][identifier]
            require(set(moments) == {'step','exp_avg','exp_avg_sq'}, 'Standard three Adam fields required')
            clock = moments['step']
            require(self.torch.is_tensor(clock) and clock.dtype == self.torch.float32 and clock.ndim == 0
                and bool(self.torch.isfinite(clock)) and float(clock) == total, 'Actual row Adam clock must equal K+N')
            for key in ('exp_avg','exp_avg_sq'):
                value = moments[key]
                require(self.torch.is_tensor(value) and value.dtype == self.torch.float32 and value.shape == parameter.shape
                    and bool(self.torch.isfinite(value).all()) and not bool(self.torch.count_nonzero(value[:8199]))
                    and not bool(self.torch.count_nonzero(value[8711:])), 'Finite same-shape protected-zero row moments required')
            require(bool((moments['exp_avg_sq'] >= 0).all()), 'Negative Adam variance')

    def state_dict(self, *, runtime_state=None):
        self._require_complete(); self._check_contract()
        from bpe_newrows_acquisition_v1 import checked_parked_record
        checked_parked_record(self.original_record); checked_parked_record(self.phase1_record)
        weights = self._base.serialize_weights()
        require(all(bool(self.torch.isfinite(value).all()) for value in weights.values())
            and self._fixed_digest(weights) == self.fixed_weight_sha256, 'Frozen204/protected rows or finite weights changed')
        optimizer = self._base._cpu_clone(self.optimizer.state_dict())
        total = self.source_acquisition_updates + self._local_updates
        self._audit_optimizer(optimizer, total)
        return {'schema':1,'kind':STATE_KIND,'phase_identity':self.phase_identity(),'weights':weights,
            'lora_metadata':copy.deepcopy(self.lora_metadata),'acquisition_optimizer':optimizer,
            'acquisition_optimizer_updates':total,'continued_acquisition_updates':self._local_updates,
            'source_step':self.source_step,'source_acquisition_updates':self.source_acquisition_updates,
            'cumulative_step':self.cumulative_step,'complete_optimizer_update_boundary':True,
            'original207_optimizer_parked':dict(self.original_record),'phase1_optimizer_source':dict(self.phase1_record),
            'runtime_state':self._base._cpu_clone(runtime_state)}

    def load_phase_state(self, state):
        require(state.get('schema') == 1 and state.get('kind') == STATE_KIND
            and state.get('phase_identity') == self.phase_identity() and state.get('lora_metadata') == self.lora_metadata
            and state.get('complete_optimizer_update_boundary') is True
            and state.get('original207_optimizer_parked') == self.original_record
            and state.get('phase1_optimizer_source') == self.phase1_record
            and state.get('source_step') == self.source_step
            and state.get('source_acquisition_updates') == self.source_acquisition_updates,
            'Only the same declared continued-row phase is admitted')
        local = state.get('continued_acquisition_updates')
        require(type(local) is int and local >= 0
            and state.get('acquisition_optimizer_updates') == self.source_acquisition_updates + local
            and state.get('cumulative_step') == self.source_step + self.source_acquisition_updates + local,
            'Own LOCAL N/Adam K+N/cumulative S+K+N differ')
        weights = state.get('weights', {})
        require(set(weights) == set(self.names), 'All207 learned weights required')
        for name,value in weights.items():
            require(self.torch.is_tensor(value) and value.dtype == self.torch.float32
                and value.shape == self.parameters[name].shape and bool(self.torch.isfinite(value).all()),
                'Learned weight shape/FP32/finiteness differs')
        require(self._fixed_digest(weights) == self.fixed_weight_sha256, 'Frozen204/protected weight bytes differ')
        self._audit_optimizer(state['acquisition_optimizer'], self.source_acquisition_updates + local)
        from bpe_newrows_acquisition_v1 import checked_parked_record
        checked_parked_record(self.original_record); checked_parked_record(self.phase1_record)
        self._incomplete_update = True
        with self.torch.no_grad():
            for name,value in weights.items(): self.parameters[name].copy_(value)
        self.optimizer.load_state_dict(copy.deepcopy(state['acquisition_optimizer']))
        self._local_updates = local
        self._base.optimizer_updates_completed = self.source_acquisition_updates + local
        self._check_contract(); self._incomplete_update = False
        return self._base._cpu_clone(state.get('runtime_state'))

    def load_state_dict(self, state): return self.load_phase_state(state)

    def unfreeze(self):
        raise ValueError('Any encoder-LoRA/shared-head/old-row unfreeze requires a separately declared phase')
