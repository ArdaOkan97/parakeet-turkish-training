"""Mature8711 -> append-only9223 model migration; no diagnostic ancestry.

Old lexical rows retain their IDs and bytes. New rows use original mature-leaf
means once, with one bias -2 penalty. Duration rows move as a separate block.
All numerical imports occur only inside the admitted GPU owner's call.
"""
import copy, hashlib, json, random
from pathlib import Path
from bpe_broad_audio_v1 import record, require

OLD, VOCAB, BLANK, DURATIONS = 8711, 9223, 8192, 5
TOKENIZER_SHA = 'f2d80ffee7ed1b006a83b52cc77bfb28bed4ba850352b5222bb3a49d6cfbac4a'


def bound(pin, guard):
    guard(); path=Path(pin['path']); actual=record(path)
    require(actual=={k:pin[k] for k in ('path','bytes','sha256')}
        and ('mtime_ns' not in pin or path.stat().st_mtime_ns==pin['mtime_ns']), 'Exact migration metadata pin')
    require(path.stat().st_size<=2_000_000, 'Bounded donor/tokenizer metadata')
    return json.loads(path.read_text(encoding='utf-8'))


def checked_plan(pin, guard):
    p=bound(pin,guard)
    require(p['kind']=='BPE9223_original_mature_leaf_row_initialization_plan_v1'
        and [p[k] for k in ('source_lexical_rows','target_lexical_rows','source_head_rows','target_head_rows','blank_id_unchanged')]
            ==[OLD,VOCAB,OLD+DURATIONS,VOCAB+DURATIONS,BLANK]
        and p['recursive_averaging_or_repeated_bias_penalty'] is False
        and p['training_token_ID_permutation'] is False, 'Exact append-only reviewed9223 donor policy')
    require(p['old_duration_to_new_rows']==[{'old':OLD+i,'new':VOCAB+i} for i in range(DURATIONS)], 'Exact duration relocation')
    require(len(p['donors'])==VOCAB-OLD and [v['HF_id'] for v in p['donors']]==list(range(OLD,VOCAB)), 'All512 ordered new rows')
    candidate=bound(p['candidate_tokenizer'],guard)
    require(p['candidate_tokenizer']['sha256']==TOKENIZER_SHA and len(candidate['model']['vocab'])==VOCAB, 'Exact approved corpus-driven tokenizer')
    by_id={v:k for k,v in candidate['model']['vocab'].items()}
    require(set(by_id)==set(range(VOCAB)), 'Contiguous unique9223 IDs')
    for row in p['donors']:
        leaves=row['original_leaf_donor_ids']
        require(type(leaves) is list and 2<=len(leaves)<=256 and all(type(i) is int and 0<=i<OLD and i!=BLANK for i in leaves)
            and row['piece']==by_id[row['HF_id']]==''.join(by_id[i] for i in leaves), 'Complete flattened mature leaves, multiplicities retained')
    return p,candidate


def expand_model(model, processor, plan_pin, *, torch, guard, admit):
    from extend_vocab import _make_tokenizer
    from tokenizers import Tokenizer
    import numpy as np
    from bpe_broad_full207_optimizer_v1 import bits_equal
    admit('BPE9223_bounded_migration_metadata',32_000_000)
    p,candidate=checked_plan(plan_pin,guard)
    embedding,head=model.decoder.embedding,model.joint.head
    require(model.config.vocab_size==OLD and embedding.num_embeddings==OLD and head.out_features==OLD+DURATIONS
        and model.config.blank_token_id==BLANK and embedding.weight.dtype==head.weight.dtype==head.bias.dtype==torch.float32,
        'Actual mature8711 student only, unchanged training blank')
    old_vocab=processor.tokenizer.get_vocab()
    require(all(candidate['model']['vocab'].get(piece)==i for piece,i in old_vocab.items()) and len(old_vocab)==OLD,
        'Every mature token retains exact spelling and ID')
    new_parameter_bytes=4*(VOCAB*embedding.embedding_dim+(VOCAB+DURATIONS)*(head.in_features+1))
    added_parameter_bytes=4*(VOCAB-OLD)*(embedding.embedding_dim+head.in_features+1)
    leaf_scratch=4*max(len(v['original_leaf_donor_ids']) for v in p['donors'])*max(embedding.embedding_dim,head.in_features)
    require(4*added_parameter_bytes+leaf_scratch+8*1024**2<=32_000_000,
        'Explicit32M new-row weights/moments/gradients/readback/metadata steady allowance')
    admit('BPE9223_model_replacement',new_parameter_bytes+leaf_scratch+8*1024**2)
    py=random.getstate();np_state=np.random.get_state();cpu=torch.get_rng_state().clone();cuda=[v.clone() for v in torch.cuda.get_rng_state_all()]
    try:
        new_embedding=torch.nn.Embedding(VOCAB,embedding.embedding_dim,padding_idx=embedding.padding_idx,
            device=embedding.weight.device,dtype=embedding.weight.dtype)
        new_head=torch.nn.Linear(head.in_features,VOCAB+DURATIONS,device=head.weight.device,dtype=head.weight.dtype)
    finally:
        torch.set_rng_state(cpu);torch.cuda.set_rng_state_all(cuda)
    with torch.no_grad():
        new_embedding.weight[:OLD].copy_(embedding.weight)
        new_head.weight[:OLD].copy_(head.weight[:OLD]);new_head.bias[:OLD].copy_(head.bias[:OLD])
        new_head.weight[VOCAB:].copy_(head.weight[OLD:]);new_head.bias[VOCAB:].copy_(head.bias[OLD:])
        for row in p['donors']:
            guard();i=row['HF_id'];leaves=row['original_leaf_donor_ids']
            new_embedding.weight[i].copy_(embedding.weight[leaves].mean(0))
            new_head.weight[i].copy_(head.weight[leaves].mean(0));new_head.bias[i].copy_(head.bias[leaves].mean()-2.)
        for old,new in ((embedding.weight,new_embedding.weight[:OLD]),(head.weight[:OLD],new_head.weight[:OLD]),
                        (head.bias[:OLD],new_head.bias[:OLD]),(head.weight[OLD:],new_head.weight[VOCAB:]),(head.bias[OLD:],new_head.bias[VOCAB:])):
            require(bits_equal(torch,old,new),'Every old lexical and relocated duration byte preserved')
    model.decoder.embedding,model.joint.head=new_embedding,new_head
    model.config.vocab_size=model.joint.vocab_size=VOCAB
    if hasattr(model.decoder,'vocab_size'):model.decoder.vocab_size=VOCAB
    if hasattr(model.generation_config,'vocab_size'):model.generation_config.vocab_size=VOCAB
    suppressions={}
    for field in ('suppress_tokens','begin_suppress_tokens'):
        old=copy.deepcopy(getattr(model.generation_config,field,None))
        require(old is None or all(type(v) is int and 0<=v<OLD+DURATIONS for v in old),'Known mature suppression IDs')
        new=None if old is None else [v+(VOCAB-OLD) if OLD<=v<OLD+DURATIONS else v for v in old]
        require(not any(OLD<=v<VOCAB for v in new or []),'All512 new rows unsuppressed')
        setattr(model.generation_config,field,new);suppressions[field]={'before':old,'after':new}
    processor.tokenizer=_make_tokenizer(processor.tokenizer,Tokenizer.from_str(json.dumps(candidate,ensure_ascii=False)))
    actual_np=np.random.get_state()
    require(py==random.getstate() and np_state[0]==actual_np[0] and np.array_equal(np_state[1],actual_np[1])
        and np_state[2:]==actual_np[2:] and torch.equal(cpu,torch.get_rng_state())
        and all(torch.equal(a,b) for a,b in zip(cuda,torch.cuda.get_rng_state_all())), 'All4 RNG domains preserved by migration')
    return {'kind':'ASR18_actual_append_only_BPE9223_model_migration_v1','plan':copy.deepcopy(plan_pin),
        'candidate_tokenizer':p['candidate_tokenizer'],'source_vocabulary':OLD,'vocabulary':VOCAB,'blank':BLANK,
        'duration_rows':[VOCAB,VOCAB+DURATIONS],'old_rows_and_duration_bytes_exact':True,'all4_RNG_preserved':True,
        'initialization':'ORIGINAL mature8711 flattened leaf mean once; bias mean minus2 once',
        'new_parameter_bytes':new_parameter_bytes,'added_parameter_bytes':added_parameter_bytes,
        'maximum_leaf_scratch_bytes':leaf_scratch,'steady_new_row_allowance_bytes':32_000_000,'suppression':suppressions,
        'old_Parameter_objects_replaced_only_for_three_expanded_tables':True,'production_source_used':True,'diagnostic_checkpoint_used':False}
