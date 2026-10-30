"""New ASR-18 branch: retain BPE histories and unpark original encoder Adam."""
from copy import deepcopy
import math

from bpe_full_decoder_optimizer_v1 import TABLES, partition_plan, require

KIND = 'BPE_broad_full207_partitioned_AdamW_v1'


def bits_equal(torch, left, right):
    return (left.shape == right.shape and left.dtype == right.dtype and
            torch.equal(left.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                        right.detach().cpu().contiguous().reshape(-1).view(torch.uint8)))


def encoder_lr(clock):
    require(type(clock) is int and clock >= 62250, 'Declared encoder clock required')
    if clock >= 65948:
        return 3e-5
    fraction = (clock - 40750 - 1) / (25198 - 1)
    return 3e-5 + (1e-4 - 3e-5) * .5 * (1 + math.cos(math.pi * fraction))


class BPEBroadFull207Optimizer:
    def __init__(self, model, original_source, bpe_source, *, source_records, torch_module):
        self.torch = torch_module
        self.source_records = deepcopy(source_records)
        require(original_source['step'] == original_source['selected_step'] == 62250,
                'Qualified original full207 source62250 required')
        require(bpe_source['step'] == 64250, 'Existing completed BPE64250 warm start required')
        self.names = tuple(original_source['weights'])
        self.head_names, self.lora_names, self.mature_plan, self.fresh_plan = partition_plan(self.names, 8711)
        self.parameters = {n: p for n, p in model.named_parameters() if n in self.names}
        require(set(self.parameters) == set(self.names) == set(bpe_source['weights']), 'All207 names required')
        for n, p in model.named_parameters():
            p.requires_grad_(n in self.parameters)
            p.grad = None
        for n, p in self.parameters.items():
            saved = bpe_source['weights'][n]
            require(p.shape == saved.shape and p.dtype == saved.dtype == self.torch.float32
                    and bits_equal(self.torch, p, saved), 'BPE warm-start weights differ: ' + n)
            if n in self.lora_names:
                require(bits_equal(self.torch, saved, original_source['weights'][n]),
                        'Parked encoder weights must still equal original source: ' + n)
        self.heads = tuple(self.parameters[n] for n in self.head_names)
        self.adapters = tuple(self.parameters[n] for n in self.lora_names)
        self.mature = tuple(self.view(item) for item in self.mature_plan)
        self.fresh = tuple(self.view(item) for item in self.fresh_plan)
        bpe = bpe_source['optimizer']
        require(bpe['updates'] == 2000 and bpe['mature_Adam_clock'] == 64250
                and bpe['new_row_Adam_clock'] == 2000 and bpe['parked_encoder_Adam_clock'] == 62250,
                'Actual BPE/mature/parked clocks differ')
        identity = bpe['identity']
        require(identity['vocabulary'] == 8711 and identity['blank_id'] == 8192
                and identity['mature_optimizer_partitions'] == list(self.mature_plan)
                and identity['fresh_optimizer_partitions'] == list(self.fresh_plan),
                'Exact BPE table/duration partitions required')
        original = original_source['optimizer']
        require(original['param_groups'][0]['params'] == list(range(15))
                and original['param_groups'][1]['params'] == list(range(15, 207)),
                'Original named207 Adam ID order required')
        encoder_group = {k: deepcopy(v) for k, v in original['param_groups'][1].items() if k != 'params'}
        require(encoder_group['lr'] == encoder_lr(62250), 'Original encoder LR differs')
        encoder_saved = {'state': {i: original['state'][i + 15] for i in range(192)},
                         'param_groups': [{**encoder_group, 'params': list(range(192))}]}
        self.optimizers = [self.restore(self.mature, bpe['mature_optimizer'], 64250),
                           self.restore(self.fresh, bpe['fresh_optimizer'], 2000),
                           self.restore(self.adapters, encoder_saved, 62250, expected_weight_decay=0.)]
        require(self.optimizers[0].param_groups[0]['lr'] == 1e-5
                and self.optimizers[1].param_groups[0]['lr'] == 1e-4, 'Inherited head/new-row LRs differ')
        self.base_clocks = (64250, 2000, 62250)
        self.updates = 0
        self.poisoned = False
        self.group_policies = [{k: deepcopy(v) for k, v in o.param_groups[0].items() if k != 'params'}
                               for o in self.optimizers]
        self.validate(full=True)

    def view(self, item):
        parent = self.parameters[item['name']]
        if item['target_rows'] is None:
            return parent
        low, high = item['target_rows']
        result = self.torch.nn.Parameter(parent.detach()[low:high], requires_grad=True)
        require(result.data_ptr() == parent[low:high].data_ptr(), 'Storage-sharing partition required')
        return result

    def restore(self, parameters, saved, clock, *, expected_weight_decay=.01):
        require(len(saved['param_groups']) == 1 and saved['param_groups'][0]['params'] == list(range(len(parameters)))
                and set(saved['state']) == set(range(len(parameters))), 'Complete named Adam partitions required')
        group = {k: deepcopy(v) for k, v in saved['param_groups'][0].items() if k != 'params'}
        require(group['weight_decay'] == expected_weight_decay and tuple(group['betas']) == (.9, .999)
                and group['eps'] == 1e-8 and not group['amsgrad'], 'Inherited AdamW recipe differs')
        optimizer = self.torch.optim.AdamW([{'params': list(parameters), **group}])
        for i, parameter in enumerate(parameters):
            old = saved['state'][i]
            require(old['step'].item() == clock and old['step'].dtype == self.torch.float32,
                    'No moment clock reset allowed')
            require(all(old[k].shape == parameter.shape and old[k].dtype == self.torch.float32
                        and bool(self.torch.isfinite(old[k]).all()) for k in ('exp_avg', 'exp_avg_sq'))
                    and bool((old['exp_avg_sq'] >= 0).all()), 'Finite FP32 source moments required')
            optimizer.state[parameter] = {'step': old['step'].detach().cpu().clone(),
                **{k: old[k].detach().to(device=parameter.device, copy=True) for k in ('exp_avg', 'exp_avg_sq')}}
            require(all(bits_equal(self.torch, optimizer.state[parameter][k], old[k])
                        for k in ('step', 'exp_avg', 'exp_avg_sq')), 'Byte-exact moment transfer failed')
        return optimizer

    def validate(self, full=False):
        for optimizer, base in zip(self.optimizers, self.base_clocks):
            for parameter in optimizer.param_groups[0]['params']:
                state = optimizer.state[parameter]
                require(state['step'].item() == base + self.updates
                        and state['exp_avg'].shape == state['exp_avg_sq'].shape == parameter.shape,
                        'Partition clocks/geometry diverged')
                if full:
                    require(all(bool(self.torch.isfinite(state[k]).all()) for k in ('exp_avg', 'exp_avg_sq'))
                            and bool((state['exp_avg_sq'] >= 0).all()), 'Nonfinite/negative Adam moments')

    def zero_grad(self, set_to_none=True):
        require(not self.poisoned and set_to_none is True, 'Incomplete optimizer update is poisoned')
        for p in (*self.parameters.values(), *self.mature, *self.fresh):
            p.grad = None

    def step(self):
        require(not self.poisoned and all(p.grad is not None for p in self.parameters.values()),
                'All207 gradients required before one global clip')
        norm = self.torch.nn.utils.clip_grad_norm_([*self.adapters, *self.heads], 1., error_if_nonfinite=True)
        for p, item in zip((*self.mature, *self.fresh), (*self.mature_plan, *self.fresh_plan)):
            gradient = self.parameters[item['name']].grad
            rows = item['target_rows']
            p.grad = gradient.detach() if rows is None else gradient[rows[0]:rows[1]].detach()
        six_before = [self.parameters[n][8193:8199].detach().clone() for n in TABLES]
        self.poisoned = True
        self.optimizers[2].param_groups[0]['lr'] = encoder_lr(62250 + self.updates + 1)
        for optimizer in self.optimizers:
            optimizer.step()
        with self.torch.no_grad():
            for n, before in zip(TABLES, six_before):
                rows = self.parameters[n][8193:8199]
                rows.copy_(before + 6. * (rows - before))
        self.updates += 1
        self.validate()
        self.poisoned = False
        return norm

    def identity(self):
        return {'kind': KIND, 'source_records': self.source_records, 'source_clocks': list(self.base_clocks),
                'active_model_tensors': 207, 'head_names': list(self.head_names),
                'encoder_LoRA_names': list(self.lora_names), 'mature_plan': list(self.mature_plan),
                'fresh_plan': list(self.fresh_plan), 'vocabulary': 8711, 'blank': 8192,
                'duration_rows': [8711, 8716], 'mature_lr': 1e-5, 'new_rows_lr': 1e-4,
                'encoder_lr_policy': 'original_cosine_on_unparked_encoder_clock_until65948_then3e-5',
                'partition_weight_decay': [.01, .01, 0.],
                'source_moments_or_clocks_reset': False, 'six_glyph_gain': 6., 'global_clip_once': 1.}

    def state_dict(self):
        require(not self.poisoned, 'Cannot save partially applied update')
        self.validate(full=True)
        return {'kind': KIND, 'identity': self.identity(), 'updates': self.updates,
                'clocks': [c + self.updates for c in self.base_clocks],
                'optimizers': [o.state_dict() for o in self.optimizers]}

    def load_state_dict(self, saved):
        require(not self.poisoned and saved['kind'] == KIND and saved['identity'] == self.identity(),
                'Own optimizer source/geometry identity differs')
        count = saved['updates']
        require(type(count) is int and count >= 0 and saved['clocks'] == [c + count for c in self.base_clocks],
                'Own exact clocks differ')
        require(len(saved['optimizers']) == 3, 'All3 optimizer histories required')
        for index, (optimizer, state, policy) in enumerate(zip(self.optimizers, saved['optimizers'], self.group_policies)):
            actual = {k: v for k, v in state['param_groups'][0].items() if k != 'params'}
            expected = {**policy, 'lr': encoder_lr(62250 + count)} if index == 2 else policy
            require(actual == expected, 'Own exact AdamW hyperparameters differ')
            require(state['param_groups'][0]['params'] == list(range(len(optimizer.param_groups[0]['params'])))
                    and set(state['state']) == set(range(len(optimizer.param_groups[0]['params']))),
                    'Own exact optimizer parameter IDs differ')
        self.poisoned = True
        for optimizer, state in zip(self.optimizers, saved['optimizers']):
            optimizer.load_state_dict(state)
        self.updates = count
        self.validate(full=True)
        self.poisoned = False
