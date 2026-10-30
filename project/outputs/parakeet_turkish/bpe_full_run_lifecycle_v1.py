"""Inactive two-broad/one-cleaner orchestration; no runtime or launch side effects.

The native owner supplies separately reviewed, source-bound callbacks. Existing
native engines, finite callers, cursor math, scorer and cache helpers are not
modified. Metadata protocol controls cannot admit a native/full-DATA run.
"""
from copy import deepcopy
import inspect
import json
from pathlib import Path
import shutil
import threading

from bpe_broad_production_state_v1 import canonical, digest, permutation, require
from bpe_training_schedule_v1 import WEAK, CLEANER, COMPLETE, KIND as SCHEDULE_KIND
from bpe_production_window_coordinator_v4 import make_plan
from bpe_broad_audio_v1 import record

KIND = 'ASR18_two_broad_one_cleaner_full_run_lifecycle_v1'
ROOT = Path(r'E:\gpu_training_scratchpad')
HERE = Path(__file__).resolve().parent
REFERENCES = dict(tr=825, en=1045, es=1289, it=1158, fr=1072, de=1004, pt=1095)
LIMITS = dict(en=69, es=52, it=30, fr=74, de=70, pt=66)
SCORER_SHA = '433c31d9cbc4975c42d3a0fe659cd0bbd6c1caf2e1eb3d85169baafec7f61047'
GATE_SHA = 'cf0f9c66fe5f1046b3ae151026bb19be3b286233a83391055e89dc450fb3c6af'
CALLBACKS = ('admit_operation', 'prepare_window', 'execute_window', 'verify_checkpoint',
             'serial_DEV', 'verify_DEV', 'preserve_winner', 'verify_winner', 'verify_resume',
             'checkpoint_storage_boundary')


def serial():
    require(threading.current_thread() is threading.main_thread(), 'Serial lifecycle only')


def bound(pin, *, maximum_bytes=16*1024**2):
    require(type(pin) is dict and set(pin) == {'path', 'bytes', 'sha256'}
            and type(pin['bytes']) is int and 0 < pin['bytes'] <= maximum_bytes
            and record(pin['path']) == pin, 'Exact bounded artifact/source binding required')
    return pin


def read_bound(pin, *, maximum_bytes=16*1024**2):
    bound(pin, maximum_bytes=maximum_bytes)
    return json.loads(Path(pin['path']).read_text(encoding='utf-8-sig'))


def integer(value, low=0):
    require(type(value) is int and value >= low, 'Exact nonboolean integer required')
    return value


def progress(state):
    """Validate exact schedule counters; no schedule or RNG mutation."""
    require(state['kind'] == SCHEDULE_KIND, 'Original full two-broad/one-cleaner schedule required')
    broad, clean = state['broad_cursor'], state['cleaner_cursor']
    n, m = integer(broad['population'], 1), integer(clean['population'], 1)
    b, c = integer(broad['draws']), integer(clean['draws'])
    require(type(broad['passes']) is int and broad['passes'] == 2
            and type(clean['passes']) is int and clean['passes'] == 1
            and m == 352768 and 0 <= b <= 2*n and 0 <= c <= m
            and (not c or b == 2*n), 'Exact pass budgets and broad-before-cleaner order')
    require(state['population_mean_U'][CLEANER] == 11972886/352768,
            'Original complete cleaner target denominator must carry unchanged')
    updates = 0
    for draws, population in ((b, n), (c, m)):
        passes, prefix = divmod(draws, population)
        require(prefix % 14 == 0, 'Only complete14-or-exact-tail Adam boundaries')
        updates += passes*((population+13)//14)+prefix//14
    phase = WEAK if b < 2*n else CLEANER if c < m else COMPLETE
    require(state['phase'] == phase and integer(state['carried_replay']['updates']) == updates,
            'Phase/replay/actual Adam counters disagree')
    return {'broad_population': n, 'cleaner_population': m, 'broad_draws': b,
            'cleaner_draws': c, 'optimizer_updates': updates, 'phase': phase}


def milestones(state):
    p = progress(state); n, m = p['broad_population'], p['cleaner_population']
    u = (n+13)//14
    return [dict(name='broad_pass_1', phase=WEAK, epoch=0, broad_draws=n, cleaner_draws=0, optimizer_updates=u),
            dict(name='broad_pass_2', phase=WEAK, epoch=1, broad_draws=2*n, cleaner_draws=0, optimizer_updates=2*u),
            dict(name='cleaner_pass_1', phase=CLEANER, epoch=0, broad_draws=2*n, cleaner_draws=m,
                 optimizer_updates=2*u+(m+13)//14)]


def next_boundary_request(state, completed_boundaries):
    serial(); p = progress(state)
    require(type(completed_boundaries) is int and 0 <= completed_boundaries <= 3, 'Exact acknowledged boundary prefix')
    if completed_boundaries == 3:
        require(p['phase'] == COMPLETE, 'Completion cannot shorten declared passes')
        return {'kind': 'training_and_declared_DEV_complete', 'TEST_complete': False, 'export_complete': False}
    boundary = milestones(state)[completed_boundaries]
    require(p['optimizer_updates'] <= boundary['optimizer_updates'],
            'Training crossed an unacknowledged pass boundary without complete DEV')
    if p['optimizer_updates'] == boundary['optimizer_updates']:
        require(all(p[k] == boundary[k] for k in ('broad_draws', 'cleaner_draws')),
                'Actual tail counters differ from exact pass boundary')
        return {'kind': 'checkpoint_then_serial350DEV', 'boundary': boundary, 'TEST_used': False}
    require(p['phase'] == boundary['phase'], 'Implicit phase jump prohibited')
    remaining = boundary['optimizer_updates']-p['optimizer_updates']
    return {'kind': 'bounded_training_window', 'phase': p['phase'], 'boundary': boundary,
            'maximum_updates': min(20, remaining), 'before_sampler_sha256': digest(state),
            'initial_optimizer_updates': p['optimizer_updates'], 'no_pass_crossing': True}


def validate_window_plan(state, request, plan):
    """Reject pass1->pass2 windows even though both have the same WEAK phase."""
    require(request['kind'] == 'bounded_training_window' and plan['kind'] ==
            'ASR18_phase_aware_provisional_production_window_plan_v1'
            and plan['production_enabled'] is False and plan['phase'] == request['phase']
            and plan['initial_sampler_sha256'] == digest(state)
            and plan['initial_committed_updates'] == request['initial_optimizer_updates']
            and type(plan['updates']) is list and 1 <= len(plan['updates']) <= request['maximum_updates'],
            'Exact nonempty bounded prepared plan required before any native draw')
    cursor = state['broad_cursor'] if request['phase'] == WEAK else state['cleaner_cursor']
    n, start = cursor['population'], cursor['draws']; epoch = start//n
    require(epoch == request['boundary']['epoch'], 'Wrong current declared pass')
    previous = digest(state)
    for index, update in enumerate(plan['updates']):
        count = min(14, n-start % n)
        require(update['phase'] == request['phase'] and update['global_update'] == request['initial_optimizer_updates']+index+1
                and update['before_sampler_sha256'] == previous and len(update['TR_draws']) == count
                and len(update['rows']) == count+2 and len(update['replay_rows']) == 2,
                'Every complete/tail update includes exact original replay pair')
        expected = [{'ordinal': permutation((start+j)%n, n, [cursor['seed'], epoch])+1,
                     'epoch': epoch, 'draw_index': start+j} for j in range(count)]
        require(update['TR_draws'] == expected and digest(update['TR_draws']) == digest(expected),
                'Exact original bijection/draw prefix; no reseed, exclusion or next-pass leakage')
        start += count; previous = update['after_sampler_sha256']
        require(start <= (epoch+1)*n and (start < (epoch+1)*n or index == len(plan['updates'])-1),
                'Every pass tail is a checkpoint/complete DEV barrier, including weak pass1')
    final = plan['final_private_planning_sampler']; q = progress(final)
    require(digest(final) == previous and q['optimizer_updates'] == request['initial_optimizer_updates']+len(plan['updates'])
            and q['broad_draws'] == (start if request['phase'] == WEAK else state['broad_cursor']['draws'])
            and q['cleaner_draws'] == (start if request['phase'] == CLEANER else state['cleaner_cursor']['draws']),
            'Full final private plan state must match exact pass counters')
    return plan


def make_pass_bounded_plan(schedule, *, completed_boundaries, **planning):
    before = deepcopy(schedule.state_dict()); request = next_boundary_request(before, completed_boundaries)
    require(request['kind'] == 'bounded_training_window', 'Actual tail DEV must complete before next plan')
    require('maximum_updates' not in planning, 'Cannot override exact remaining-pass cap')
    plan = make_plan(schedule, maximum_updates=request['maximum_updates'], **planning)
    require(digest(schedule.state_dict()) == digest(before), 'Planning changed original schedule/RNG')
    return validate_window_plan(before, request, plan)


def official_DEV(report, pointer, scorer_policy):
    """Recompute fixed350 saved predictions with the immutable official scorer."""
    require(report['kind'] == 'ASR18_serial_complete350_boundary_DEV_v1' and report['passed'] is True
            and report['serial_optimizer_free'] is True and report['full350_DEV_complete'] is True
            and report['split'] == 'validation' and report['TEST_used'] is False
            and report['checkpoint_pointer'] == pointer and report['scorer_policy'] == scorer_policy,
            'Complete actual serial DEV of exactly the coherent boundary checkpoint required')
    require(scorer_policy['official_scorer'] == record(HERE/'metrics.py')
            and scorer_policy['official_scorer']['sha256'] == SCORER_SHA
            and scorer_policy['integer_gate_contract'] == record(HERE/'tsc_bpe_full_decoder_transition_v1.py')
            and scorer_policy['integer_gate_contract']['sha256'] == GATE_SHA
            and scorer_policy['reference_words'] == REFERENCES and scorer_policy['retention_limits'] == LIMITS,
            'No diagnostic normalizer, denominator, integer gate or official scorer replacement')
    from metrics import score_transcriptions
    require(set(report['predictions']) == set(scorer_policy['DEV_manifests']) == set(REFERENCES), 'All seven fixed DEV languages')
    scores, counts = {}, {}
    for language, words in REFERENCES.items():
        def lines(pin):
            bound(pin); return [json.loads(v) for v in Path(pin['path']).read_text(encoding='utf-8-sig').splitlines() if v.strip()]
        manifest, predictions = lines(scorer_policy['DEV_manifests'][language]), lines(report['predictions'][language])
        require(len(manifest) == len(predictions) == 50 and len({r['source_id'] for r in manifest}) == 50
                and [r['source_id'] for r in manifest] == [r['source_id'] for r in predictions], 'Exact fixed50 order/membership')
        for source, result in zip(manifest, predictions):
            require(source['language'] == result['language'] == language and result['split'] == 'validation'
                    and result['reference'] == source['text'], 'Fixed official reference text/language/split changed')
            for key in ('sentence_id', 'dataset_revision', 'audio_sha256', 'wav_sha256'):
                require(result.get(key) == source.get(key), 'Saved prediction source provenance differs')
        score = score_transcriptions(predictions, language=language); score.pop('details')
        require(score['utterances'] == 50 and score['reference_words'] == words
                and report['scores']['validation_'+language]['asr'] == score,
                'Official saved-text score/count summary differs')
        scores[language] = score; counts[language] = integer(score['word_errors'])
    return {'word_errors': counts, 'reference_words': deepcopy(REFERENCES), 'retention_limits': deepcopy(LIMITS),
            'strict_all_six_passed': all(counts[l] <= v for l, v in LIMITS.items()),
            'turkish_wer': counts['tr']/825, 'full350_DEV_complete': True, 'TEST_used': False}


def checkpoint_storage_gate(proof, policy_binding, *, actual_free_bytes):
    """Accounting boundary only. This module grants no checkpoint deletion."""
    require(proof['kind'] == 'ASR18_reviewed_owned_checkpoint_storage_boundary_v1'
            and proof['passed'] is True and proof['policy'] == policy_binding
            and proof['owner'] == '/root/memory_diagnosis'
            and proof['bounded_publication_and_retirement_implementation_reviewed'] is True
            and proof['preserve_qualified_winner_coherent_latest_source_history_DEV_and_journal'] is True
            and proof['existing_or_original_artifact_deletion_authorized'] is False
            and proof['unbounded_per_window_checkpoint_retention'] is False,
            'Separately reviewed bounded checkpoint storage implementation/policy remains mandatory')
    additional = integer(proof['maximum_additional_owned_bytes'], 1)
    require(type(actual_free_bytes) is int and actual_free_bytes-additional >= 100_000_000_000,
            'Prospective owned publication would violate E100GB reserve; no cleanup authority or fallback')
    return {'passed': True, 'projected_free_bytes_lower_bound': actual_free_bytes-additional,
            'maximum_additional_owned_bytes': additional, 'deletion_performed_or_authorized': False}


class _LifecycleProtocol:
    """Pure metadata state machine; internal controls are not execution admission."""
    def __init__(self, schedule_state, identity, qualified_winner, *, restored=None):
        serial(); progress(schedule_state)
        self.identity = deepcopy(identity); self.poisoned = False
        self.state = {'kind': KIND, 'identity_sha256': digest(identity), 'completed_boundaries': [],
                      'last_sampler_sha256': digest(schedule_state), 'last_pointer': None,
                      'winner': deepcopy(qualified_winner), 'training_complete': False,
                      'TEST_complete': False, 'export_complete': False, 'production_enabled': False}
        if restored is not None:
            require(restored['kind'] == KIND and restored['identity_sha256'] == digest(identity)
                    and restored['last_sampler_sha256'] == digest(schedule_state)
                    and restored['TEST_complete'] is False and restored['export_complete'] is False
                    and restored['production_enabled'] is False, 'Exact restored lifecycle/scientific identity required')
            self.state = deepcopy(restored)
        completed = self.state['completed_boundaries']
        require(type(completed) is list and len(completed) <= 3 and [v['boundary'] for v in completed] ==
                milestones(schedule_state)[:len(completed)], 'Exact contiguous declared boundary history')
        request = self.request(schedule_state)
        require(self.state['training_complete'] == (request['kind'] == 'training_and_declared_DEV_complete'),
                'No completion claim before all actual pass tails and complete DEV')

    def request(self, schedule_state):
        require(not self.poisoned and digest(schedule_state) == self.state['last_sampler_sha256'],
                'Unreconciled actual native frontier; restore coherent checkpoint/lifecycle')
        return next_boundary_request(schedule_state, len(self.state['completed_boundaries']))

    def accept_window(self, before, plan, after, pointer_binding, pointer):
        request = self.request(before); validate_window_plan(before, request, plan)
        require(digest(after) == digest(plan['final_private_planning_sampler'])
                and pointer['kind'] == 'ASR18_production_window_completed_checkpoint_pointer_v2'
                and pointer['sampler_state_sha256'] == digest(after)
                and integer(pointer['committed_updates'], 1) == progress(after)['optimizer_updates']
                and pointer['all_window_updates_committed'] is True
                and pointer['writer_returned_and_payload_verified'] is True
                and pointer['saved_semantic_state_verified']['passed'] is True
                and pointer['saved_semantic_state_verified']['optimizer_updates'] == pointer['committed_updates']
                and pointer['saved_semantic_state_verified']['sampler_state_sha256'] == digest(after)
                and pointer['saved_semantic_state_verified']['journal_prefix'] == pointer['journal_prefix']
                and pointer['diagnostic_only'] is False,
                'Actual positive full-run checkpoint/prefix, never a diagnostic20/21 promotion')
        self.state['last_sampler_sha256'] = digest(after); self.state['last_pointer'] = deepcopy(pointer_binding)
        return self.request(after)

    def accept_DEV(self, schedule_state, report_binding, score, winner_binding):
        request = self.request(schedule_state)
        require(request['kind'] == 'checkpoint_then_serial350DEV' and self.state['last_pointer'] is not None,
                'Actual coherent checkpoint exactly at declared pass tail required')
        require(score['full350_DEV_complete'] is True and score['reference_words'] == REFERENCES
                and score['retention_limits'] == LIMITS and score['TEST_used'] is False,
                'No incomplete/diagnostic/TEST gate substitution')
        old = self.state['winner']; candidate_errors = integer(score['word_errors']['tr'])
        expected_change = score['strict_all_six_passed'] and (old is None or candidate_errors < old['word_errors']['tr'])
        require(type(score['strict_all_six_passed']) is bool and score['strict_all_six_passed'] ==
                all(integer(score['word_errors'][l]) <= v for l, v in LIMITS.items()), 'Strict integer all6 gates')
        if expected_change:
            self.state['winner'] = {'artifact': deepcopy(winner_binding), 'word_errors': deepcopy(score['word_errors']),
                                    'reference_words': deepcopy(REFERENCES), 'DEV': deepcopy(report_binding),
                                    'checkpoint_pointer': deepcopy(self.state['last_pointer']), 'strict_qualified': True}
        else:
            require(winner_binding == (old['artifact'] if old else None), 'Failed/equal retention candidate cannot replace qualified winner')
        self.state['completed_boundaries'].append({'boundary': request['boundary'], 'DEV': deepcopy(report_binding),
            'checkpoint_pointer': deepcopy(self.state['last_pointer']), 'score': deepcopy(score), 'winner_changed': bool(expected_change)})
        action = self.request(schedule_state); self.state['training_complete'] = action['kind'] == 'training_and_declared_DEV_complete'
        return {'next': action, 'winner_changed': bool(expected_change), 'retention_early_stop': False,
                'declared_training_shortened': False, 'TEST_complete': False, 'export_complete': False}


class SourceBoundOwnerCallbacks:
    """Identity binding, not authority to bypass ROOT/source/resource admission."""
    def __init__(self, sources, callbacks):
        require(set(sources) == set(callbacks) == set(CALLBACKS), 'Complete explicit native-owner callback interface')
        self.sources, self.callbacks = deepcopy(sources), dict(callbacks)
        for name, function in callbacks.items():
            require(callable(function), 'Callable native owner interface required'); bound(sources[name])
            require(str(Path(inspect.getsourcefile(function)).resolve()) == sources[name]['path'], 'Callback implementation source mismatch')

    def call(self, name, request):
        serial(); bound(self.sources[name])
        return self.callbacks[name](deepcopy(request))


class FullRunLifecycle:
    """One admitted window OR one serial DEV boundary per call; no launch CLI.

    The owner still implements phase-compatible native/DEV execution, memory
    admission, journal branching, exact all207/threeAdam/RNG restore and durable
    lifecycle checkpoint integration. None is inferred from these metadata checks.
    """
    def __init__(self, schedule, *, identity, callbacks, qualified_winner=None, restored=None, restore_proof=None):
        require(isinstance(callbacks, SourceBoundOwnerCallbacks), 'Exact source-bound owner callbacks required')
        self.schedule, self.callbacks, self.identity = schedule, callbacks, deepcopy(identity)
        bound(identity['full_run_source_contract']); bound(identity['scorer_policy'])
        bound(identity['checkpoint_storage_policy'])
        require(identity['kind'] == 'ASR18_reviewed_full_two_broad_one_cleaner_run_identity_v1'
                and identity['native_owner'] == '/root/memory_diagnosis'
                and identity['required_broad_passes'] == 2 and identity['required_cleaner_passes'] == 1
                and identity['source_original_clocks'] == [64250, 2000, 62250]
                and identity['source_optimizer_RNG_sampler_reset'] is False,
                'Explicit full-run identity; no diagnostic or counter reset seed')
        state = schedule.state_dict()
        if qualified_winner is not None:
            require(qualified_winner['strict_qualified'] is True and qualified_winner['reference_words'] == REFERENCES
                    and all(integer(qualified_winner['word_errors'][l]) <= v for l, v in LIMITS.items())
                    and type(qualified_winner['word_errors']['tr']) is int
                    and callbacks.call('verify_winner',
                    {'winner': qualified_winner, 'identity': identity}) is True, 'Preserve independently qualified source winner')
        if restored is not None:
            require(callbacks.call('verify_resume', {'identity': identity, 'lifecycle': restored, 'schedule': state,
                    'proof': restore_proof, 'all207_threeAdam_allRNG_and_journal_prefix_required': True}) is True,
                    'Actual exact native restore plus lifecycle/winner/checkpoint prefix required')
        self.protocol = _LifecycleProtocol(state, identity, qualified_winner, restored=restored)

    def advance(self):
        serial(); before = deepcopy(self.schedule.state_dict()); action = self.protocol.request(before)
        if action['kind'] == 'training_and_declared_DEV_complete': return action
        storage_pin = self.callbacks.call('checkpoint_storage_boundary', {'action': action, 'identity': self.identity,
            'lifecycle': deepcopy(self.protocol.state), 'preserve_all_existing_and_original_checkpoints': True})
        checkpoint_storage_gate(read_bound(storage_pin), self.identity['checkpoint_storage_policy'],
                                actual_free_bytes=shutil.disk_usage(ROOT).free)
        require(self.callbacks.call('admit_operation', {'action': action, 'identity': self.identity,
                'lifecycle': deepcopy(self.protocol.state)}) is True, 'Independent current owner/lease/resource/caller admission required')
        require(digest(self.schedule.state_dict()) == digest(before), 'Admission changed scientific state')
        try:
            if action['kind'] == 'bounded_training_window':
                prepared = self.callbacks.call('prepare_window', {'action': action, 'identity': self.identity, 'schedule': before})
                plan = read_bound(prepared['coordinator_plan']); validate_window_plan(before, action, plan)
                require(digest(self.schedule.state_dict()) == digest(before), 'Preparation advanced sampler/RNG/Adam')
                completion_pin = self.callbacks.call('execute_window', prepared); completion = read_bound(completion_pin)
                pointer_pin = completion['latest_pointer']; pointer = read_bound(pointer_pin)
                after = self.schedule.state_dict()
                require(completion['passed'] is True and completion['actual_new_updates'] == len(plan['updates'])
                        and completion['global_new_updates'] == pointer['committed_updates']
                        and completion['sampler_state_sha256'] == pointer['sampler_state_sha256']
                        and completion['checkpoints'][-1]['exact_restore']['all207_threeAdam_all4RNG_match_actual_pre_save'] is True
                        and self.callbacks.call('verify_checkpoint', {'pointer': pointer_pin, 'completion': completion_pin,
                            'expected_schedule': after, 'identity': self.identity}) is True,
                        'Actual owner-verified saved sampler/all207/threeAdam/RNG/coherent checkpoint required')
                return self.protocol.accept_window(before, plan, after, pointer_pin, pointer)
            pointer_pin = self.protocol.state['last_pointer']
            report_pin = self.callbacks.call('serial_DEV', {'action': action, 'checkpoint_pointer': pointer_pin,
                'identity': self.identity, 'optimizer_updates': progress(before)['optimizer_updates'], 'split': 'validation', 'clips': 350})
            require(digest(self.schedule.state_dict()) == digest(before), 'Serial DEV changed original sampler/replay state')
            require(self.callbacks.call('verify_DEV', {'report': report_pin, 'checkpoint_pointer': pointer_pin,
                    'boundary': action['boundary'], 'identity': self.identity,
                    'all207_weights_and_all_source_clocks_unchanged_required': True,
                    'all4RNG_and_replay_queues_unchanged_required': True}) is True,
                    'Actual complete serial optimizer-free DEV boundary proof required')
            report = read_bound(report_pin); policy = read_bound(self.identity['scorer_policy'])
            score = official_DEV(report, pointer_pin, policy)
            old = self.protocol.state['winner']
            change = score['strict_all_six_passed'] and (old is None or score['word_errors']['tr'] < old['word_errors']['tr'])
            winner = self.callbacks.call('preserve_winner', {'old_winner': old, 'candidate_pointer': pointer_pin,
                'DEV': report_pin, 'score': score, 'replace': bool(change), 'identity': self.identity})
            if change:
                bound(winner)
                require(self.callbacks.call('verify_winner', {'winner': winner, 'DEV': report_pin, 'score': score,
                        'checkpoint_pointer': pointer_pin, 'identity': self.identity}) is True, 'Durable actual strict-qualified winner proof')
            require(digest(self.schedule.state_dict()) == digest(before), 'Selection changed native schedule/RNG')
            return self.protocol.accept_DEV(before, report_pin, score, winner)
        except BaseException:
            self.protocol.poisoned = True
            raise

    def state_dict(self):
        require(not self.protocol.poisoned, 'Failure needs exact coherent owner reconciliation, not automatic retry')
        return deepcopy(self.protocol.state)
