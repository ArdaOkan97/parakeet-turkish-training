"""Two exact broad passes, then one exact cleaner pass, with carried replay.

This pure state coordinator cannot load a model, arm production or select data.
Preparation is provisional; only one observed Adam commit can advance either
cursor and replay. Journals bind a compact state digest, avoiding a repeated
copy of all replay queue IDs in every update event.
"""
from copy import deepcopy
from bpe_broad_production_state_v1 import CompletePassCursor, digest, require

WEAK = 'weak_label_continued_training'
CLEANER = 'cleaner_data_fine_tuning'
COMPLETE = 'all_required_passes_complete'
KIND = 'ASR18_two_broad_then_one_cleaner_schedule_v1'

class TrainingSchedule:
    def __init__(self, *, broad_population, broad_identity, broad_mean_U,
                 cleaner_population, cleaner_identity, cleaner_mean_U,
                 source_turkish_rng, replay):
        require(cleaner_population == 352768 and cleaner_mean_U == 11972886/352768,
                'Exact existing cleaner population and full BPE denominator required')
        require(type(broad_population) is int and broad_population > 0 and broad_mean_U > 0,
                'Validated final broad denominator required before constructing schedule')
        self.broad = CompletePassCursor(population=broad_population, identity=broad_identity,
            seed=digest(source_turkish_rng), passes=2)
        self.cleaner = CompletePassCursor(population=cleaner_population, identity=cleaner_identity,
            seed=digest([source_turkish_rng, CLEANER]), passes=1)
        self.means = {WEAK: broad_mean_U, CLEANER: cleaner_mean_U}
        self.replay = replay
        self.poisoned = False

    @property
    def phase(self):
        return WEAK if self.broad.remaining else CLEANER if self.cleaner.remaining else COMPLETE

    @property
    def active_cursor(self):
        require(self.phase != COMPLETE, 'All declared passes are already complete')
        return self.broad if self.phase == WEAK else self.cleaner

    def state_dict(self):
        return {'kind': KIND, 'phase': self.phase, 'broad_cursor': self.broad.state_dict(),
                'cleaner_cursor': self.cleaner.state_dict(), 'carried_replay': deepcopy(self.replay.state_dict()),
                'population_mean_U': deepcopy(self.means)}

    def committed_updates(self):
        total = 0
        for cursor in (self.broad, self.cleaner):
            passes, prefix = divmod(cursor.draws, cursor.population)
            require(prefix % 14 == 0, 'Checkpoint frontier is not an actual complete optimizer batch boundary')
            total += passes*((cursor.population+13)//14)+prefix//14
        return total

    def load_state_dict(self, value):
        require(not self.poisoned and value['kind'] == KIND and value['population_mean_U'] == self.means,
                'Exact unpoisoned schedule/population identities required')
        require(value['cleaner_cursor']['draws'] == 0 or value['broad_cursor']['draws'] == self.broad.population*2,
                'Cleaner draws cannot precede both complete broad passes')
        try:
            self.broad.load_state_dict(value['broad_cursor']); self.cleaner.load_state_dict(value['cleaner_cursor'])
            self.replay.load_state_dict(value['carried_replay'])
            require(value['phase'] == self.phase and self.replay.updates == self.committed_updates(),
                    'Saved phase/frontiers/actual replay update counts disagree')
        except BaseException:
            self.poisoned = True
            raise

    def propose(self, *, optimizer_updates, resolve_TR):
        return ScheduledTransaction(self, optimizer_updates=optimizer_updates, resolve_TR=resolve_TR)

class ScheduledTransaction:
    def __init__(self, schedule, *, optimizer_updates, resolve_TR):
        require(not schedule.poisoned and schedule.phase != COMPLETE, 'Completed/poisoned schedule has no next draw')
        self.schedule = schedule; self.phase = schedule.phase; self.cursor = schedule.active_cursor
        self.before = schedule.state_dict(); self.optimizer_updates = optimizer_updates
        self.closed = False; self.poisoned = False
        require(schedule.replay.updates == optimizer_updates == schedule.committed_updates(),
                'One carried replay pair per actual Adam update and complete phase batch boundary required')
        boundary = self.cursor.population-self.cursor.draws % self.cursor.population
        self.TR_draws = self.cursor.preview(min(14, self.cursor.remaining, boundary))
        try:
            self.replay_rows = tuple(schedule.replay.next())
            self.after_replay = deepcopy(schedule.replay.state_dict())
        finally:
            schedule.replay.load_state_dict(self.before['carried_replay'])
        self.TR_identity = [resolve_TR(self.phase, draw['ordinal']) for draw in self.TR_draws]
        require(len(self.TR_identity) == len(self.TR_draws) and all(
            value['language'] == 'tr' and value['ordinal'] == draw['ordinal'] and value['source_id']
            for value, draw in zip(self.TR_identity, self.TR_draws)), 'Independent stage/ordinal/source identity required')
        require(schedule.state_dict() == self.before, 'Preparation must not advance either stage or replay RNG')
        self.identity_digest = digest({'phase': self.phase, 'before': self.before,
            'TR': self.TR_identity, 'draws': self.TR_draws,
            'replay': [{'source_id': row['source_id'], 'language': row['language']} for row in self.replay_rows],
            'after_replay': self.after_replay})

    def abandon(self):
        require(not self.closed and not self.poisoned and self.schedule.state_dict() == self.before,
                'Only an unchanged provisional transaction can be abandoned')
        self.closed = True
        return {'draws_consumed': 0, 'both_phase_cursors_and_replay_RNG_unchanged': True}

    def commit(self, *, optimizer, journal, native_receipt, prepared_evidence_binding):
        schedule = self.schedule
        require(not self.closed and not self.poisoned and not schedule.poisoned
                and schedule.state_dict() == self.before, 'Concurrent or reused sampler transaction prohibited')
        require(not optimizer.poisoned and optimizer.updates == self.optimizer_updates+1,
                'Exactly one actual positive Adam commit required')
        try:
            expected = self.TR_identity+list(self.replay_rows)
            require(native_receipt['update'] == optimizer.updates and native_receipt['phase'] == self.phase
                    and native_receipt['source_ids'] == [row['source_id'] for row in expected]
                    and native_receipt['languages'] == [row['language'] for row in expected],
                    'Actual native ordered stage/source IDs differ')
            require(set(prepared_evidence_binding) == {'path', 'bytes', 'sha256'}
                    and prepared_evidence_binding['bytes'] > 0, 'Caller-verified durable preparation evidence required')
            self.cursor.commit(self.TR_draws)
            schedule.replay.load_state_dict(self.after_replay)
            state = schedule.state_dict()
            # Full queue/RNG state belongs in bounded checkpoints. A compact
            # event binds it without hundreds of kilobytes of repeated IDs.
            event = {'kind': 'ASR18_committed_training_update_v1', 'phase': self.phase,
                'optimizer_updates': optimizer.updates, 'transaction_sha256': self.identity_digest,
                'sampler_state_sha256': digest(state), 'broad_draws': schedule.broad.draws,
                'cleaner_draws': schedule.cleaner.draws, 'replay_updates': schedule.replay.updates,
                'prepared_evidence': prepared_evidence_binding, 'native': native_receipt}
            journal.append(event)
            self.closed = True
            return state
        except BaseException:
            self.poisoned = True; schedule.poisoned = True; optimizer.poisoned = True
            raise
