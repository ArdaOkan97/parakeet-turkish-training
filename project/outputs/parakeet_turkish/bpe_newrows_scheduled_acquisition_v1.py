"""INACTIVE scheduled acquisition of the existing512 rows; no source admission.

The strict frozen continuation constructor first audits/loads source phase1
weights and three Adam states at LR1e-4/K. No moments or clocks are reset.
A new localN schedule uses warmup50 + cosine to0.1peak over1000 updates.
All204 other learned tensors and old/six/blank/duration rows stay frozen.
The caller owns actual CLOSED admission, native ASR, runtime/RNG restoration,
and an exclusive GPU slot. This utility neither launches nor exports a model.
"""
from __future__ import annotations
import copy
import hashlib
import math
from pathlib import Path

from bpe_newrows_continued_acquisition_v1 import BPEContinuedAcquisition, require

PHASE = 'HF_TSC_BPE512_new_rows_scheduled_acquisition_v1'
STATE_KIND = 'HF_TSC_BPE512_new_rows_scheduled_acquisition_state_v1'
PHASE_KIND = STATE_KIND
IDENTITY_KIND = 'HF_TSC_BPE512_new_rows_scheduled_acquisition_identity_v1'
CONTINUED_UTILITY_SHA = 'ee25eb365a3bd578c48ca95aa66be4429146ef96319ac5614d12c8a6248049b7'
SOURCE_ROW_LR = 1e-4
WARMUP_UPDATES = 50
HORIZON_UPDATES = 1000
MIN_LR_RATIO = .1
ALLOWED_PEAKS = (1e-3, 3e-4)


def schedule_identity(peak_row_lr):
    require(type(peak_row_lr) in (int, float) and peak_row_lr in ALLOWED_PEAKS,
            'Only declared matched peak1e-3 or3e-4 is admitted')
    return {'kind':'linear_warmup_cosine_floor_v1', 'warmup_updates':WARMUP_UPDATES,
            'horizon_updates':HORIZON_UPDATES, 'min_lr_ratio':MIN_LR_RATIO,
            'peak_row_lr':float(peak_row_lr), 'clock':'local_scheduled_updates_N',
            'apply_before_Adam':True, 'warmup_resets_Adam_moments_or_clock':False}


def lr_for_update(update, peak_row_lr):
    """The FIRST next local update is1; source K never enters this formula."""
    schedule_identity(peak_row_lr)
    require(type(update) is int and 1 <= update <= HORIZON_UPDATES,
            'Scheduled local update must be an integer1..1000')
    if update <= WARMUP_UPDATES:
        return float(peak_row_lr) * update / WARMUP_UPDATES
    progress = (update - WARMUP_UPDATES) / (HORIZON_UPDATES - WARMUP_UPDATES)
    return float(peak_row_lr) * (MIN_LR_RATIO + (1. - MIN_LR_RATIO) * .5 * (1. + math.cos(math.pi * progress)))


class BPEScheduledAcquisition(BPEContinuedAcquisition):
    """Same masked three-table Adam; only the declared local LR schedule is new."""
    def __init__(self, model, phase1_state, original207_state, *, original_source_record,
                 phase1_record, peak_row_lr=1e-3, torch_module=None):
        path = Path(__file__).with_name('bpe_newrows_continued_acquisition_v1.py')
        require(hashlib.sha256(path.read_bytes()).hexdigest() == CONTINUED_UTILITY_SHA,
                'Frozen complete207/three-Adam continuation implementation changed')
        self.peak_row_lr = float(schedule_identity(peak_row_lr)['peak_row_lr'])
        self._schedule_initialized = False
        # That frozen constructor audits original204/protected bytes, source
        # groupLR1e-4, full207 weights, all3 moments and scalarK before LR change.
        super().__init__(model, phase1_state, original207_state,
            original_source_record=original_source_record, phase1_record=phase1_record,
            active_row_lr=self.peak_row_lr, torch_module=torch_module)
        self._static_group = copy.deepcopy({key:value for key,value in
            self.optimizer.param_groups[0].items() if key not in ('params','lr')})
        self.last_applied_lr = None
        self._set_dynamic_lr(SOURCE_ROW_LR)
        self._schedule_initialized = True
        self._check_contract()

    @property
    def current_row_lr(self): return self.optimizer.param_groups[0]['lr']

    def lr_for_update(self, update): return lr_for_update(update, self.peak_row_lr)

    def _expected_group_lr(self, local):
        require(type(local) is int and 0 <= local <= HORIZON_UPDATES,
                'Scheduled local counter must be0..1000')
        return SOURCE_ROW_LR if local == 0 else self.lr_for_update(local)

    def _set_dynamic_lr(self, value):
        # Internal declared group metadata is updated with the exact schedule;
        # never relax non-LR group fields, gradient mask or frozen-weight policy.
        require(math.isfinite(value) and 0 < value <= self.peak_row_lr,
                'Positive schedule-derived LR at or below peak required')
        self.optimizer.param_groups[0]['lr'] = value
        self._base.declared_optimizer_group['lr'] = value

    def _check_contract(self):
        if not self._schedule_initialized:
            return super()._check_contract()
        self._base._check_contract()
        require(self.active_row_lr == self.peak_row_lr in ALLOWED_PEAKS
            and self._base.base_lr == self.peak_row_lr
            and {key:value for key,value in self.optimizer.param_groups[0].items()
                 if key not in ('params','lr')} == self._static_group
            and self.current_row_lr == self._expected_group_lr(self._local_updates)
            and self.last_applied_lr == (None if self._local_updates == 0 else self.current_row_lr)
            and self._base.optimizer_updates_completed == self.source_acquisition_updates + self._local_updates
            and set(self.optimizer.state) == set(self.tables),
            'Scheduled phase/group/appliedLR/localN/AdamK+N contract differs')

    def step(self, max_norm=1.):
        self._require_complete(); self._check_contract()
        require(max_norm == 1. and self._local_updates < HORIZON_UPDATES,
                'One3-table clip norm1 and remaining scheduled horizon required')
        norm = self._base.mask_then_clip(max_norm)
        self._incomplete_update = True
        applied = self.lr_for_update(self._local_updates + 1)
        self._set_dynamic_lr(applied)
        self.last_applied_lr = applied
        self.optimizer.step()
        self._local_updates += 1
        self._base.optimizer_updates_completed = self.source_acquisition_updates + self._local_updates
        self._check_contract(); self._incomplete_update = False
        return norm

    def phase_identity(self):
        identity = super().phase_identity()
        # A static phase identity cannot contain the current schedule point.
        identity.pop('declared_optimizer_group', None)
        identity.update(kind=IDENTITY_KIND,
            utility_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            continued_utility_sha256=CONTINUED_UTILITY_SHA,
            optimizer_group_static_hyperparameters=copy.deepcopy(self._static_group),
            row_schedule=schedule_identity(self.peak_row_lr), peak_row_lr=self.peak_row_lr,
            active_row_lr_is_schedule_peak=True, dynamic_applied_lr_is_not_scientific_identity=True,
            same_phase_utility_restore_supported=True, external_worker_resume_admission_supported=False)
        return identity

    def _audit_optimizer(self, optimizer, total):
        local = total - self.source_acquisition_updates
        expected = self._expected_group_lr(local)
        groups = optimizer.get('param_groups', [])
        require(len(groups) == 1 and groups[0].get('lr') == expected
            and {key:value for key,value in groups[0].items() if key not in ('params','lr')} == self._static_group,
            'Saved applied LR/static group must match saved scheduledN')
        # Reuse the strict original three-moment/finite/protected-zero/clock
        # auditor with an expected saved group, without changing live weights,
        # moments or groupLR while admission is still validating.
        previous = self._base.declared_optimizer_group
        self._base.declared_optimizer_group = {**copy.deepcopy(self._static_group), 'lr':expected}
        try:
            return super()._audit_optimizer(optimizer, total)
        finally:
            self._base.declared_optimizer_group = previous

    def state_dict(self, *, runtime_state=None):
        state = super().state_dict(runtime_state=runtime_state)
        state.update(kind=STATE_KIND, row_schedule=schedule_identity(self.peak_row_lr),
            scheduled_acquisition_updates=self._local_updates,
            last_applied_lr=self.last_applied_lr, current_row_lr=self.current_row_lr,
            scheduled_next_update=(self._local_updates + 1 if self._local_updates < HORIZON_UPDATES else None))
        return state

    def load_phase_state(self, state):
        local = state.get('scheduled_acquisition_updates')
        require(state.get('schema') == 1 and state.get('kind') == STATE_KIND
            and state.get('phase_identity') == self.phase_identity()
            and state.get('row_schedule') == schedule_identity(self.peak_row_lr)
            and type(local) is int and 0 <= local <= HORIZON_UPDATES
            and state.get('continued_acquisition_updates') == local
            and state.get('last_applied_lr') == (None if local == 0 else self.lr_for_update(local))
            and state.get('current_row_lr') == self._expected_group_lr(local)
            and state.get('scheduled_next_update') == (local + 1 if local < HORIZON_UPDATES else None),
            'Only same scheduled phase with exact localN/appliedLR/next frontier admitted')
        # The inherited durable loader keeps complete207/frozen rawbytes,
        # all3 Adam moments/clockK+N and paired artifact audits before copies.
        # Translate only the known schema tag; no caller state is modified.
        translated = dict(state)
        from bpe_newrows_continued_acquisition_v1 import STATE_KIND as CONTINUED_STATE_KIND
        translated['kind'] = CONTINUED_STATE_KIND
        previous_group = self._base.declared_optimizer_group
        previous_applied = self.last_applied_lr
        self._base.declared_optimizer_group = {**copy.deepcopy(self._static_group), 'lr':self._expected_group_lr(local)}
        self.last_applied_lr = state['last_applied_lr']
        try:
            runtime = super().load_phase_state(translated)
        except BaseException:
            self._base.declared_optimizer_group = previous_group
            self.last_applied_lr = previous_applied
            # If actual weight/optimizer loading began, the frozen loader's
            # poison stays set. Only a complete validated reload can clear it.
            raise
        self._check_contract()
        return runtime

    def load_state_dict(self, state): return self.load_phase_state(state)
