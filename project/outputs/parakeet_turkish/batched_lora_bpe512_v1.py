"""Candidate TR8711 + ORIGINAL replay8193 mixed collation.

Only this collator boundary changes. TrainingBatch and the complete objective
are imported unchanged from the frozen metadata helper at actual runtime.
No ML or audio module is imported merely by importing this file.
"""
from __future__ import annotations
import math
from typing import Callable

def collate_training_batch(processor, rows, cache_items=None, *, blank_id: int,
                           old_size: int, device="cuda", audio_reader=None,
                           non_blocking=False):
    """Collate explicit TR metadata plus genuine packed replay teacher features."""
    import torch
    from torch.nn.utils.rnn import pad_sequence
    from batched_lora_tr_metadata_v1 import TrainingBatch
    from hf_bpe512_tr_metadata_v1 import load_candidate_tr_item, KIND, VOCAB_SIZE
    from tr_frame_metadata_v1 import load_training_item as load_original_replay
    def load_training_item(row):
        if row["language"] == "tr":
            return load_candidate_tr_item(row,torch_module=torch)
        if row.get("training_input_type") is not None:
            raise ValueError("Replay must use original unannotated labels/cache")
        return load_original_replay(row)
    if audio_reader is None:
        from runtime import read_audio
        audio_reader = read_audio
    rows=list(rows)
    if not rows or blank_id != 8192 or old_size != 8193:raise ValueError("Positive batch/unchanged blank required")
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
            checked = load_candidate_tr_item(row,torch_module=torch)
            if not torch.equal(target,checked["labels"]):raise ValueError("Candidate item labels differ from checked row")
            frames=int(item["encoder_frames"])
            if frames<1 or frames!=row["encoder_frames"] or target.numel()!=row["target_tokens"] or int(target.max())>=VOCAB_SIZE or bool((target==blank_id).any()):raise ValueError("TR frame/target metadata differs")
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
