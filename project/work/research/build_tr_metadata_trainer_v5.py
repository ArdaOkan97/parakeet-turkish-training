"""Create NEW metadata-aware variants; never overwrite existing versions."""
from pathlib import Path
import ast, hashlib, json
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'outputs/parakeet_turkish'
def create(name,text):
    path=D/name
    if path.exists():raise FileExistsError(path)
    ast.parse(text);path.write_text(text,encoding='utf-8',newline='\n')
    return hashlib.sha256(path.read_bytes()).hexdigest()
collate='''def collate_training_batch(processor, rows, cache_items=None, *, blank_id: int,
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


'''
text=(D/'batched_lora.py').read_text(encoding='utf-8')
start=text.index('def collate_training_batch(');end=text.index('def _sample_grid(',start)
text=text[:start]+collate+text[end:]
text=text.replace('base_features','replay_base_features')
# The newly inserted name already contains the prefix, avoid duplicating it.
text=text.replace('replay_replay_base_features','replay_base_features')
assert text.count('batch.replay_base_features.index_select(0, replay)')==1
text=text.replace('batch.replay_base_features.index_select(0, replay)','batch.replay_base_features')
text=text.replace('Masked true-batch Parakeet LoRA objective','Masked true-batch metadata-only-TR Parakeet LoRA objective')
collator_sha=create('batched_lora_tr_metadata_v1.py',text)
text=(D/'train_lora_batched_data_v4.py').read_text(encoding='utf-8')
text=text.replace('from batched_lora import collate_training_batch, batch_objective','from batched_lora_tr_metadata_v1 import collate_training_batch, batch_objective; from tr_frame_metadata_v1 import annotate_tr_frame_metadata, confirm_tr_metadata_acceptance')
text=text.replace('tr = annotate_cache_lengths(make_cache(model, processor, tr, cache, args.max_tokens))','tr = annotate_tr_frame_metadata(processor, tr, args.tr_frame_metadata, args.tr_frame_metadata_proof, args.max_tokens)')
text=text.replace('confirm_acoustic_cache_acceptance(tr, public_data_audit)','confirm_tr_metadata_acceptance(tr, public_data_audit)')
text=text.replace('train_lora_batched_data_v4.py','train_lora_batched_data_v5.py')
text=text.replace('"batched_lora.py", "train_lora.py"','"batched_lora_tr_metadata_v1.py", "tr_frame_metadata_v1.py", "batched_lora.py", "train_lora.py"')
text=text.replace('args.data_v3_proof, OUTPUT / "tr_public_v3_tokenizer_acceptance.json"','args.data_v3_proof, args.tr_frame_metadata, args.tr_frame_metadata_proof, OUTPUT / "tr_public_v3_tokenizer_acceptance.json"')
text=text.replace('"teacher":"original frozen heads conditioned only on original frozen singleton encoder cache; appended IDs excluded from replay"','"teacher":"original frozen heads conditioned only on genuine original replay singleton feature cache; TR has explicit lengths/labels and no teacher features; appended IDs excluded from replay"')
text=text.replace('p.add_argument("--run-id",required=True)','p.add_argument("--tr-frame-metadata",type=Path,default=DATA/"tr_public_v3_frame_metadata_v1.jsonl"); p.add_argument("--tr-frame-metadata-proof",type=Path,default=OUTPUT/"TR_frame_metadata_cpu_verification_v1.json"); p.add_argument("--run-id",required=True)')
tree=ast.parse(text)
run=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='run')
sync=[node.lineno for node in ast.walk(run) if isinstance(node,ast.Call) and ast.unparse(node.func)=='torch.cuda.synchronize']
if 539 not in sync:raise ValueError('Exact optimizer cache hook539 must remain unchanged')
trainer_sha=create('train_lora_batched_data_v5.py',text)
print(json.dumps({'collator_sha256':collator_sha,'trainer_sha256':trainer_sha,'run_sync539':True}))
