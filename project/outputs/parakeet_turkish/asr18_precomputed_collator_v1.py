"""Verified CPU-produced inputs; original label/replay/mask collation retained."""
import torch
from torch.nn.utils.rnn import pad_sequence
from asr18_narrow_objective_v1 import TrainingBatch
import math

def collate_precomputed_row(row,prepared_inputs,full_target_ids,cache_items=None,*,blank_id=8192,
                            old_size=8193,device='cuda',non_blocking=False):
    from tr_frame_metadata_v1 import load_training_item, KIND
    rows=[row]
    if not rows or not 0 <= blank_id < old_size:raise ValueError("Positive batch/unchanged blank required")
    if cache_items is None:cache_items=[load_training_item(row) for row in rows]
    if len(cache_items)!=len(rows):raise ValueError("Training item count differs")
    if (type(full_target_ids) is not list or not full_target_ids
        or cache_items[0]['labels'].detach().cpu().long().tolist()!=full_target_ids):
        raise ValueError('Every independently bound full precomputed target ID must match actual metadata/cache')
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
    if set(prepared_inputs)!={'input_features','attention_mask'}:
        raise ValueError('Only complete original precomputed feature/mask inputs are admitted')
    inputs={name:torch.from_numpy(value) for name,value in prepared_inputs.items()}
    if (inputs['input_features'].dtype!=torch.float32 or inputs['attention_mask'].dtype!=torch.bool
        or inputs['input_features'].ndim!=3 or inputs['input_features'].shape[0]!=1 or inputs['input_features'].shape[-1]!=128
        or inputs['attention_mask'].shape!=inputs['input_features'].shape[:2]
        or not bool(torch.isfinite(inputs['input_features']).all())):
        raise ValueError('Exact serial original feature/mask dtype/geometry required')
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
