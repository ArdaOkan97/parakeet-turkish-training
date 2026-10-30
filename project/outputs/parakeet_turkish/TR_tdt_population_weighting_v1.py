"""Inactive TR-only sequence NLL/fixed population U candidate; no ML imports.

The caller first computes its frozen native/PT-preservation objective once.
``clip`` returns that exact object. ``token_population`` changes only TR TDT
by U/meanU; RP TDT and every preservation vector retain their original graph.
Native clamp/blank/duration/reduction and global207 gradient clipping are owned
by the frozen caller and are never changed here. Positive target lengths are
required, as in the admitted carrier/collator; no empty-target or weight clamp
is introduced. Mean weight1 over the population does not imply equal average
loss, gradients, clipping frequency or WER.
"""
from __future__ import annotations

from dataclasses import replace
import math
import struct

KIND = 'TR_only_TDT_fixed_population_U_weighting_v1'
MODES = ('clip', 'token_population')
POPULATION_CLIPS = 253798
POPULATION_MEAN_U = 26.642061797177284
POPULATION_REVIEW_SHA256 = 'c9dfe011d408846916ede9c35523d8bca083ff95624935901e118aecda54a181'
REPLAY_LANGUAGES = frozenset(('en', 'es', 'it', 'fr', 'de', 'pt'))


def weights_for_metadata(rows, target_tokens, *, mode, population_mean_U=POPULATION_MEAN_U):
    if mode not in MODES or type(population_mean_U) not in (float, int):
        raise ValueError('Declared clip or token_population mode and finite fixed mean required')
    if not math.isfinite(population_mean_U) or population_mean_U != POPULATION_MEAN_U:
        raise ValueError('Frozen full253798 population mean U changed')
    if not rows or len(rows) != len(target_tokens):
        raise ValueError('One positive target length per admitted clip required')
    weights, tr_indices = [], []
    for i, (row, u) in enumerate(zip(rows, target_tokens)):
        if type(u) is not int or u < 1 or row.get('target_tokens') != u:
            raise ValueError('Positive original target length required; no empty-target clamp')
        language = row.get('language')
        if language != 'tr' and language not in REPLAY_LANGUAGES:
            raise ValueError('Only original TR and declared six replay languages supported')
        if language == 'tr' and u > 180:
            raise ValueError('Original admitted TR length cap changed')
        if language == 'tr': tr_indices.append(i)
        weights.append(u / population_mean_U if mode == 'token_population' and language == 'tr' else 1.)
    if not tr_indices and mode != 'clip':
        raise ValueError('Weighted mode is scoped to a physical batch containing TR')
    # Original per_tdt is FP32. Log the exact FP32 coefficients applied on
    # device without an extra CUDA read, rather than unrounded Python ratios.
    weights = [struct.unpack('<f', struct.pack('<f', w))[0] for w in weights]
    tr_weights = [weights[i] for i in tr_indices]
    return {
        'kind': KIND, 'mode': mode, 'population_mean_U': population_mean_U,
        'population_clips': POPULATION_CLIPS, 'population_review_sha256': POPULATION_REVIEW_SHA256,
        'TR_indices': tr_indices, 'TR_target_tokens': [target_tokens[i] for i in tr_indices],
        'all_clip_weights': weights, 'actual_TR_weights': tr_weights,
        'TR_weight_min': min(tr_weights) if tr_weights else None,
        'TR_weight_mean': math.fsum(tr_weights) / len(tr_weights) if tr_weights else None,
        'TR_weight_max': max(tr_weights) if tr_weights else None, 'replay_weights_exactly_one': True, 'applied_weight_dtype':'IEEE754_FP32',
        'empty_target_or_weight_clamp': False,
    }


def apply_to_objective(original, rows, *, mode, torch_module=None, population_mean_U=POPULATION_MEAN_U):
    """Return (objective, CPU exposure metadata), without recomputing native loss.

    All tensors remain differentiable. Off-mode returns the original object,
    not a reconstructed expression. Inject Torch only in the future native
    worker; source imports and scalar fixtures do not import it.
    """
    exposure = weights_for_metadata(rows, original.target_tokens, mode=mode, population_mean_U=population_mean_U)
    if mode == 'clip': return original, exposure
    if torch_module is None: raise ValueError('Injected native tensor runtime required for weighted mode')
    unweighted = original.per_clip['tdt']
    weight = torch_module.tensor(exposure['all_clip_weights'], dtype=unweighted.dtype, device=unweighted.device)
    weighted = unweighted * weight
    # RP delta is literal zero. Keep the existing PT/old-five preservation
    # expression and its graph; do not rebuild, detach or renormalize it.
    total = original.per_clip['total'] + unweighted * (weight - 1.)
    result = replace(original, total=total.mean(), tdt=weighted.mean(), per_clip={
        **original.per_clip, 'total': total, 'tdt': weighted,
        'tdt_unweighted': unweighted, 'tdt_weighted': weighted,
    })
    return result, exposure


def detached_metrics(objective, exposure, *, torch_module=None):
    """One small native stack/read; both TDT components have explicit names.

    ``tdt_loss`` denotes the applied TDT component. Off-mode uses the original
    callback and adds scalar aliases without another tensor operation/read.
    The worker must also publish ``exposure.actual_TR_weights`` aligned to its
    sampled source IDs, across8TR then6TR+2RP; no per-batch meanU is substituted.
    """
    if exposure['mode'] == 'clip':
        values = objective.detached_metrics()
        return {**values, 'tdt_loss_unweighted': values['tdt_loss'], 'tdt_loss_weighted': values['tdt_loss']}
    if torch_module is None: raise ValueError('Injected native tensor runtime required')
    unweighted = objective.per_clip['tdt_unweighted']
    weighted = objective.per_clip['tdt_weighted']
    indices = exposure['TR_indices']
    scalars = (objective.total, objective.tdt, objective.token_kl, objective.duration_kl,
               objective.feature_distillation, unweighted.mean(), weighted.mean(),
               unweighted[indices].mean(), weighted[indices].mean())
    values = torch_module.stack(scalars).detach().float().cpu().tolist()
    keys = ('loss', 'tdt_loss', 'replay_kl', 'replay_duration_kl', 'replay_feature_distillation',
            'tdt_loss_unweighted', 'tdt_loss_weighted', 'TR_tdt_unweighted_mean', 'TR_tdt_weighted_mean')
    return dict(zip(keys, values))


def declaration(mode):
    if mode not in MODES: raise ValueError('Unsupported normalization mode')
    return {
        'kind': KIND, 'mode': mode, 'population_mean_U': POPULATION_MEAN_U,
        'population_clips': POPULATION_CLIPS, 'population_review_sha256': POPULATION_REVIEW_SHA256,
        'TR_TDT': 'raw_NLL/fixed_population_mean_U' if mode == 'token_population' else 'raw_NLL/U',
        'RP_TDT': 'original_raw_NLL/U', 'RP_TDT_and_all_preservation_unchanged': True,
        'whole_population_mean_TR_weight': 1., 'mean_weight_scope':'Mathematical population factors before FP32 casting',
        'weight_cast_policy':'Unclamped FP32 factor, exact CPU coefficient log', 'per_microbatch_or_update_weight_renormalization': False,
        'effective_group_weight_changes': {'fleurs': 2.368809370542985, 'cv': .7465169837230463, 'tsc': 1.0876223855792688} if mode == 'token_population' else {'fleurs':1.,'cv':1.,'tsc':1.},
        'natural_TR_draws_and_future_RP_sequence_unchanged': True,
        'native_blank_duration_sigma_omega_FastEmit_clamp_reduction_unchanged': True,
        'global207_gradient_clip_unchanged': True, 'empty_target_or_weight_clamp': False,
        'all207_Adam_history_sampler_RNG_clocks_preserved': True,
        'no_optimizer_or_coverage_or_grace_or_failure_reset': True,
        'equal_gradient_or_loss_or_quality_claim': False,
        'activation_requires_completed_head_policy_and_real_source350_native_admission': True,
        'actual_GPU_tested': False,
    }
