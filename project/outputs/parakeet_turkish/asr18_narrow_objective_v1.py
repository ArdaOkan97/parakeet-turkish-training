"""Masked true-batch metadata-only-TR Parakeet LoRA objective, isolated from the pilot helpers.

The total is the mean of per-clip target-length-normalized TDT and weighted
preservation losses. Turkish clips contribute zero preservation terms. Divide
``result.total`` by gradient accumulation in the caller, not by replay count.
CUDA uses BF16 autocast while model/adapter/optimizer storage remains FP32.
"""
from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from efficient_tdt import chunked_tdt_loss
import soundfile as sf

def read_audio(row):
    audio, sr = sf.read(row['audio_filepath'], dtype='float32')
    if sr != 16000 or audio.ndim != 1:
        raise ValueError('Prepared data must be mono 16 kHz')
    return audio


@dataclass
class TrainingBatch:
    rows: list[dict]
    inputs: dict[str, torch.Tensor]
    labels: torch.Tensor
    decoder_input_ids: torch.Tensor
    target_lengths: torch.Tensor  # CPU int64 metadata; no repeated CUDA reads.
    frame_lengths: torch.Tensor   # CPU original-cache lengths.
    replay_base_features: torch.Tensor
    replay_indices: tuple[int, ...]
    blank_id: int
    old_size: int

    @property
    def batch_size(self):
        return len(self.rows)

    def to(self, device, *, non_blocking=False):
        """Return a moved shallow copy; original CPU prefetch batch is retained."""
        device = torch.device(device)
        return replace(self, inputs={name: value.to(device, non_blocking=non_blocking)
                                    for name, value in self.inputs.items()},
                       labels=self.labels.to(device, non_blocking=non_blocking),
                       decoder_input_ids=self.decoder_input_ids.to(device, non_blocking=non_blocking),
                       replay_base_features=self.replay_base_features.to(device, non_blocking=non_blocking))

    @property
    def audio_seen_seconds(self):
        return math.fsum(float(row["duration"]) for row in self.rows)

    @property
    def lattice_padding_factor(self):
        valid = int((self.frame_lengths * (self.target_lengths + 1)).sum())
        padded = self.batch_size * int(self.frame_lengths.max()) * (int(self.target_lengths.max()) + 1)
        return padded / valid


@dataclass
class BatchObjective:
    total: torch.Tensor
    tdt: torch.Tensor
    token_kl: torch.Tensor
    duration_kl: torch.Tensor
    feature_distillation: torch.Tensor
    per_clip: dict[str, torch.Tensor]
    batch_size: int
    encoder_frames: tuple[int, ...]
    target_tokens: tuple[int, ...]
    debug_encoder_features: torch.Tensor | None = None

    @property
    def loss(self):
        return self.total

    @property
    def tdt_loss(self):
        return self.tdt

    @property
    def replay_kl(self):
        return self.token_kl

    @property
    def replay_duration_kl(self):
        return self.duration_kl

    @property
    def replay_feature_distillation(self):
        return self.feature_distillation

    def detached_metrics(self):
        """One small metadata read, instead of one GPU sync per component."""
        values = torch.stack((self.total, self.tdt, self.token_kl, self.duration_kl,
                              self.feature_distillation)).detach().float().cpu().tolist()
        return dict(zip(("loss", "tdt_loss", "replay_kl", "replay_duration_kl",
                         "replay_feature_distillation"), values))


def collate_training_batch(processor, rows, cache_items=None, *, blank_id: int,
                           old_size: int, device="cuda", audio_reader: Callable = read_audio,
                           non_blocking=False) -> TrainingBatch:
    """Collate explicit TR metadata plus genuine packed replay teacher features."""
    from tr_frame_metadata_v1 import load_training_item, KIND
    rows=list(rows)
    if not rows or not 0 <= blank_id < old_size:raise ValueError("Positive batch/unchanged blank required")
    if cache_items is None:cache_items=[load_training_item(row) for row in rows]
    if len(cache_items)!=len(rows):raise ValueError("Training item count differs")
    labels, frame_lengths, replay_features, replay_indices = [], [], [], []
    for index,(row,item) in enumerate(zip(rows,cache_items)):
        if item["source_id"]!=row["source_id"]:raise ValueError("Source identity differs")
        target=item["labels"].detach().cpu().long()
        if target.ndim!=1 or not target.numel() or bool((target<0).any()):raise ValueError("Positive target IDs required")
        if not math.isfinite(float(row["duration"])) or row["duration"]<=0:raise ValueError("Positive duration required")
        if row["language"]=="tr":
            if item.get("kind")!=KIND or "features" in item:raise ValueError("TR requires explicit metadata; teacher features absent")
            frames=int(item["encoder_frames"])
            if frames<1 or frames!=row["encoder_frames"] or target.numel()!=row["target_tokens"]:raise ValueError("TR frame/target metadata differs")
        else:
            if "kind" in item or int(target.max())>=old_size:raise ValueError("Replay requires unchanged original teacher cache/IDs")
            original=item["features"].detach().cpu()
            if original.ndim!=2 or original.shape[0]<1 or not torch.isfinite(original).all():raise ValueError("Finite genuine replay features required")
            frames=original.shape[0];replay_features.append(original);replay_indices.append(index)
        labels.append(target);frame_lengths.append(frames)
    target_lengths=torch.tensor([target.numel() for target in labels],dtype=torch.long)
    frame_lengths=torch.tensor(frame_lengths,dtype=torch.long)
    labels_padded=pad_sequence(labels,batch_first=True,padding_value=blank_id)
    ids=torch.cat((torch.full((len(rows),1),blank_id,dtype=torch.long),labels_padded),dim=1)
    if replay_features and len({feature.shape[1] for feature in replay_features})!=1:raise ValueError("Replay hidden dimensions differ")
    # Only genuine replay rows are stored; TR has no teacher-feature placeholder.
    maximum_t=int(frame_lengths.max());hidden=replay_features[0].shape[1] if replay_features else 0
    packed_replay=torch.zeros((len(replay_features),maximum_t,hidden),dtype=torch.float32)
    for index,feature in enumerate(replay_features):packed_replay[index,:feature.shape[0]]=feature
    audio=[audio_reader(row) for row in rows]
    inputs=dict(processor(audio=audio,sampling_rate=16000,return_tensors="pt",padding="longest"))
    if "attention_mask" not in inputs or "input_features" not in inputs:raise ValueError("Masked mel inputs required")
    mask=inputs["attention_mask"].bool()
    if mask.ndim!=2 or mask.shape[0]!=len(rows) or bool((mask.sum(-1)<=0).any()):raise ValueError("Positive input masks required")
    right=torch.arange(mask.shape[1])[None,:]<mask.sum(-1)[:,None]
    if not torch.equal(mask.cpu(),right):raise ValueError("Right padded masks required")
    inputs["input_features"]=inputs["input_features"].masked_fill(~mask.unsqueeze(-1),0)
    destination=torch.device(device)
    inputs={name:value.to(destination,non_blocking=non_blocking) for name,value in inputs.items()}
    return TrainingBatch(rows=rows,inputs=inputs,labels=labels_padded.to(destination,non_blocking=non_blocking),
        decoder_input_ids=ids.to(destination,non_blocking=non_blocking),target_lengths=target_lengths,frame_lengths=frame_lengths,
        replay_base_features=packed_replay.to(destination,non_blocking=non_blocking),replay_indices=tuple(replay_indices),blank_id=blank_id,old_size=old_size)


def _sample_grid(lengths, device):
    """Host-created per-example linspace grids and masks, preserving B1 policy."""
    count = max(min(8, length) for length in lengths)
    indices = torch.zeros((len(lengths), count), dtype=torch.long)
    masks = torch.zeros((len(lengths), count), dtype=torch.bool)
    for row, length in enumerate(lengths):
        size = min(8, length)
        values = torch.linspace(0, length - 1, size).long()
        indices[row, :size] = values
        indices[row, size:] = values[-1]
        masks[row, :size] = True
    return indices.to(device), masks.to(device)


def sampled_preservation(model, teacher, batch, features, encoder_projected,
                         decoder_projected, *, old_size):
    """Per-clip old-token KL, duration KL and masked feature cosine losses.

    Only replay rows enter the original decoder, so Turkish extension IDs can
    never reach its embedding. Variable-size sample grids use valid-cell masks.
    The return vectors have full batch length, with literal zero Turkish terms.
    """
    batch_size = batch.batch_size
    zeros = features.new_zeros((batch_size,), dtype=torch.float32)
    if not batch.replay_indices:
        return zeros, zeros.clone(), zeros.clone()
    if any(module.training for module in teacher.values()):
        raise ValueError("Preservation teacher modules must remain in eval mode")
    device = features.device
    replay = torch.tensor(batch.replay_indices, dtype=torch.long, device=device)
    t_lengths = [int(batch.frame_lengths[index]) for index in batch.replay_indices]
    d_lengths = [int(batch.target_lengths[index]) + 1 for index in batch.replay_indices]
    ts, t_mask = _sample_grid(t_lengths, device)
    us, u_mask = _sample_grid(d_lengths, device)
    example = torch.arange(len(batch.replay_indices), device=device)[:, None]
    original_features = batch.replay_base_features
    ids = batch.decoder_input_ids.index_select(0, replay)
    with torch.no_grad():
        original_ep = teacher["encoder_projector"](original_features)[example, ts]
        original_dp = teacher["decoder"](ids)[example, us]
        original_logits = teacher["joint"](original_ep[:, :, None], original_dp[:, None])
        tokens = original_logits[..., :old_size].float().softmax(-1)
        durations = original_logits[..., old_size:].float().softmax(-1)
    ep = encoder_projected.index_select(0, replay)[example, ts]
    dp = decoder_projected.index_select(0, replay)[example, us]
    logits = model.joint(ep[:, :, None], dp[:, None])
    log_tokens = logits[..., :model.config.vocab_size].float().log_softmax(-1)
    log_durations = logits[..., model.config.vocab_size:].float().log_softmax(-1)
    grid_mask = t_mask[:, :, None] & u_mask[:, None, :]
    grid_count = grid_mask.sum((1, 2)).float()
    token_cells = (tokens * (tokens.clamp_min(1e-12).log() - log_tokens[..., :old_size])).sum(-1)
    duration_cells = (durations * (durations.clamp_min(1e-12).log() - log_durations)).sum(-1)
    token_per_replay = (token_cells * grid_mask).sum((1, 2)) / grid_count
    duration_per_replay = (duration_cells * grid_mask).sum((1, 2)) / grid_count
    feature_cells = 1 - F.cosine_similarity(features.index_select(0, replay).float(), original_features, dim=-1)
    feature_mask = torch.arange(feature_cells.shape[1], device=device)[None, :] < torch.tensor(t_lengths, device=device)[:, None]
    feature_per_replay = (feature_cells * feature_mask).sum(-1) / torch.tensor(t_lengths, device=device)
    return (zeros.index_copy(0, replay, token_per_replay),
            zeros.index_copy(0, replay, duration_per_replay),
            zeros.index_copy(0, replay, feature_per_replay))


def batch_objective(model, teacher, batch: TrainingBatch, *, old_size: int,
                    replay_kl=5.0, duration_kl=0.5, feature_distillation=1.0,
                    chunk_size=16, tdt_loss_fn=chunked_tdt_loss,
                    validate_lengths=True, use_autocast=True,
                    tdt_host_lengths=False, return_debug_features=False) -> BatchObjective:
    """Run a true encoder/decoder/joint batch and return a mean per-clip loss.

    CPU uses FP32. GPU uses CUDA BF16 autocast. Cached original lengths are
    kept on CPU; strict encoder-mask agreement is checked once. Set
    ``tdt_host_lengths=True`` only for a loss implementation supporting CPU
    metadata (the original loss requires device-local final-length indexing).
    ``validate_lengths=False`` is reserved for an already verified frozen
    frontend/cache combination, for explicitly profiled synchronization costs.
    """
    if model.encoder.training:
        raise ValueError("Keep the adapted encoder in eval mode to freeze statistics/dropout")
    if old_size != batch.old_size or model.config.blank_token_id != batch.blank_id:
        raise ValueError("Batch tokenizer/teacher/blank identities disagree")
    device = batch.labels.device
    autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                if device.type == "cuda" and use_autocast else contextlib.nullcontext())
    with autocast:
        encoded = model.encoder(**batch.inputs)
        if encoded.attention_mask is None:
            raise ValueError("Encoder omitted its output attention mask")
        maximum_t = int(batch.frame_lengths.max())
        if encoded.last_hidden_state.shape[1] < maximum_t:
            raise ValueError("Student encoder cannot represent every original teacher frame")
        if validate_lengths:
            actual_lengths = encoded.attention_mask.sum(-1).detach().cpu().long()
            if not torch.equal(actual_lengths, batch.frame_lengths):
                raise ValueError("Batched student encoder lengths differ from original singleton teacher cache")
        features = encoded.last_hidden_state[:, :maximum_t]
        ep = model.encoder_projector(features)
        dp = model.decoder(batch.decoder_input_ids)
        loss_frame_lengths = batch.frame_lengths if tdt_host_lengths else batch.frame_lengths.to(device)
        loss_target_lengths = batch.target_lengths if tdt_host_lengths else batch.target_lengths.to(device)
        raw_tdt = tdt_loss_fn(model.joint, ep, dp, batch.labels,
                             model.config.vocab_size, batch.blank_id, model.config.durations,
                             logit_lengths=loss_frame_lengths, target_lengths=loss_target_lengths,
                             chunk_size=chunk_size, reduction="none")
        if raw_tdt.shape != (batch.batch_size,):
            raise ValueError("TDT reduction=none must return one raw loss per clip")
        per_tdt = raw_tdt.float() / batch.target_lengths.to(device)
        token, duration, feature = sampled_preservation(model, teacher, batch, features, ep, dp, old_size=old_size)
        total_per_clip = per_tdt + replay_kl * token + duration_kl * duration + feature_distillation * feature
    return BatchObjective(total=total_per_clip.mean(), tdt=per_tdt.mean(), token_kl=token.mean(),
                          duration_kl=duration.mean(), feature_distillation=feature.mean(),
                          per_clip={"tdt": per_tdt, "token_kl": token, "duration_kl": duration,
                                    "feature_distillation": feature, "total": total_per_clip},
                          batch_size=batch.batch_size,
                          encoder_frames=tuple(batch.frame_lengths.tolist()),
                          target_tokens=tuple(batch.target_lengths.tolist()),
                          debug_encoder_features=features.detach() if return_debug_features else None)


def preservation_row_scales(rows):
    """Exact per-clip density declaration; TR preservation remains literal0."""
    scales=[]
    for row in rows:
        language=row["language"]
        if language=="tr":scales.append(0.)
        elif language=="pt":scales.append(1.)
        elif language in ("en","es","it","fr","de"):scales.append(5/3)
        else:raise ValueError("Unapproved preservation language: "+str(language))
    return scales


def pt_focused_batch_objective(model,teacher,batch,*,replay_kl,duration_kl,feature_distillation,portuguese_token_kl_multiplier,**kwargs):
    # Frozen native forward, per-targetU TDT and full-vocab teacher losses run
    # exactly once. Reweight their differentiable per-clip outputs, mean16 via
    # the unchanged physical-batch mean/accumulation2 convention.
    from dataclasses import replace
    result=batch_objective(model,teacher,batch,replay_kl=0.,duration_kl=0.,feature_distillation=0.,**kwargs)
    scale=torch.tensor(preservation_row_scales(batch.rows),dtype=torch.float32,device=result.per_clip["tdt"].device)
    token_scale=scale*torch.tensor([portuguese_token_kl_multiplier if row["language"]=="pt" else 1. for row in batch.rows],dtype=torch.float32,device=scale.device)
    total=(result.per_clip["tdt"]+replay_kl*token_scale*result.per_clip["token_kl"]
        +duration_kl*scale*result.per_clip["duration_kl"]+feature_distillation*scale*result.per_clip["feature_distillation"])
    return replace(result,total=total.mean(),per_clip={**result.per_clip,"total":total})
