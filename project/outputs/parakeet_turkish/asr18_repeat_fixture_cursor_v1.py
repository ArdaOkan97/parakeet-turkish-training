"""Fixed14-source20-update resource fixture, explicitly not corpus coverage."""
from copy import deepcopy
from bpe_broad_production_state_v1 import require, permutation

class RepeatedFixtureCursor:
    def __init__(self, *, ordinals, identity, seed, repeats=20):
        require(len(ordinals) == len(set(ordinals)) == 14 and repeats == 20,
                'Exact14-source bounded20-update memory diagnostic required')
        self.ordinals = sorted(ordinals); self.identity = deepcopy(identity); self.seed = seed
        self.repeats = repeats; self.population = 14; self.draws = 0

    @property
    def remaining(self):
        return self.population*self.repeats-self.draws

    def preview(self, count):
        require(type(count) is int and count == 14 and self.remaining >= count,
                'Only one full14-source diagnostic cycle at each update')
        return [{'ordinal': self.ordinals[permutation((self.draws+j) % 14, 14,
                    [self.seed, (self.draws+j)//14])],
                 'epoch': (self.draws+j)//14, 'draw_index': self.draws+j} for j in range(count)]

    def commit(self, prepared):
        require(prepared == self.preview(len(prepared)), 'Only exact prepared prefix may advance after Adam')
        self.draws += len(prepared)

    def state_dict(self):
        return {'kind': 'ASR18_repeat14_resource_diagnostic_cursor_v1', 'ordinals': list(self.ordinals),
                'identity': deepcopy(self.identity), 'seed': self.seed, 'repeats': self.repeats, 'draws': self.draws}

    def load_state_dict(self, value):
        require({k: v for k, v in value.items() if k != 'draws'} ==
                {k: v for k, v in self.state_dict().items() if k != 'draws'}, 'Exact diagnostic cursor identity differs')
        require(type(value['draws']) is int and 0 <= value['draws'] <= 280 and value['draws'] % 14 == 0,
                'Only actual complete14-source optimizer boundaries are admitted')
        self.draws = value['draws']

    def coverage(self):
        return {'diagnostic_source_rows': 14, 'diagnostic_repeated_cycles_completed': self.draws//14,
                'required_diagnostic_repeated_cycles': self.repeats, 'complete': self.remaining == 0,
                'committed_diagnostic_draws': self.draws, 'full_broad_corpus_passes': 0,
                'scientific_complete_pass_claimed': False}
