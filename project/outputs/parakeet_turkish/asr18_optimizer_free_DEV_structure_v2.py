"""Inactive BPE8711/LoRA207 structure for a fresh, optimizer-free DEV worker.

No original training checkpoint, Adam tensor, sampler or replay object is read.
Every temporary active tensor MUST be replaced by selective saved weights before
the model is usable. This module has no import-time numerical imports or CLI.
"""
from pathlib import Path
import hashlib
import json
import threading

from bpe_broad_production_state_v1 import require, digest

HERE = Path(__file__).resolve().parent


def _hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()


def expected_names(model):
    """Independent original fifteen decoder/joint plus 24*4*2 adapter layout."""
    parameters = dict(model.named_parameters())
    heads = tuple(name for name in parameters if name.startswith(('encoder_projector.', 'decoder.', 'joint.')))
    adapters = tuple(f'encoder.layers.{layer}.self_attn.{projection}.{suffix}'
        for layer in range(24) for projection in ('q_proj', 'k_proj', 'v_proj', 'o_proj')
        for suffix in ('lora_A', 'lora_B'))
    require(len(heads) == 15 and len(adapters) == 192 and len(set(heads + adapters)) == 207
            and set(adapters).issubset(parameters), 'Original named15+192 exact layout required')
    return heads + adapters


def prepare_structure(model, processor, *, torch, guard, admit):
    require(threading.current_thread() is threading.main_thread()
            and len(model.encoder.layers) == 24
            and hasattr(model, '_diagnostic_frozen_backing')
            and len(model._diagnostic_frozen_backing) == 294
            and model.config.vocab_size == model.decoder.embedding.num_embeddings == 8193
            and model.joint.head.out_features == 8198 and model.config.blank_token_id == 8192,
            'Fresh original8193 model, exact readonly294 bank and serial main thread required')
    guard(); admit('optimizer_free_DEV_BPE_LoRA_structure', 250_000_000)
    from tsc_bpe_acquisition_transition_v1 import candidate_assets
    from extend_vocab import _backend, _make_tokenizer
    from tokenizers import Tokenizer
    from encoder_lora import install_encoder_lora, encoder_lora_metadata
    candidate, _unused_donors = candidate_assets()
    old_backend = _backend(processor.tokenizer)
    original_vocab = old_backend.get_vocab()
    require(len(original_vocab) == 8193 and set(original_vocab.values()) == set(range(8193)),
            'Exact official original8193 tokenizer required')
    backend = Tokenizer.from_str(json.dumps(candidate, ensure_ascii=False))
    require(backend.get_vocab_size() == 8711
            and all(backend.token_to_id(piece) == index for piece, index in original_vocab.items())
            and backend.token_to_id(processor.blank_token) == 8192,
            'Frozen TRAIN-only candidate must preserve every original token and blank ID')
    tokenizer = _make_tokenizer(processor.tokenizer, backend)
    require(len(tokenizer) == 8711 and json.loads(_backend(tokenizer).to_str()) == candidate,
            'Exact existing8711 candidate tokenizer, never retrained or simplified')
    embedding, head = model.decoder.embedding, model.joint.head
    suppression = {}
    for field in ('suppress_tokens', 'begin_suppress_tokens'):
        old = getattr(model.generation_config, field, None)
        require(old is None or all(type(i) is int and 0 <= i < 8198 for i in old),
                'Unknown original generation suppression')
        new = None if old is None else [i + 518 if 8193 <= i < 8198 else i for i in old]
        require(not any(8193 <= i < 8711 for i in new or []), 'New lexical rows must remain unsuppressed')
        suppression[field] = {'before': old, 'after': new}
        setattr(model.generation_config, field, new)
    # No donor/source weights are needed: all207 parameters are overwritten by
    # the sole saved checkpoint. Meta allocation avoids a temporary random GPU
    # head initialization. Empty tensors cannot be evaluated before restoration.
    with torch.device('meta'):
        new_embedding = torch.nn.Embedding(8711, embedding.embedding_dim, padding_idx=embedding.padding_idx,
                                           dtype=torch.float32)
        new_head = torch.nn.Linear(head.in_features, 8716, bias=True, dtype=torch.float32)
    model.decoder.embedding = new_embedding.to_empty(device='cuda')
    model.joint.head = new_head.to_empty(device='cuda')
    model.config.vocab_size = model.joint.vocab_size = 8711
    if hasattr(model.decoder, 'vocab_size'): model.decoder.vocab_size = 8711
    if hasattr(model.generation_config, 'vocab_size'): model.generation_config.vocab_size = 8711
    processor.tokenizer = tokenizer
    layout = install_encoder_lora(model, rank=8, alpha=16, last_n_layers=24)
    for name, parameter in model.named_parameters():
        if name.endswith(('lora_A', 'lora_B')):
            parameter.data = parameter.data.to(device='cuda', dtype=torch.float32)
    names = expected_names(model); selected = set(names)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in selected); parameter.grad = None
    require(all(p.device.type == 'cuda' and p.dtype == torch.float32
                for name, p in model.named_parameters() if name in selected),
            'Only named207 active FP32 GPU tensors')
    model.config.encoder_lora_tracking = {'layout': layout, 'kind': 'unmerged',
        'implementation_sha256': hashlib.sha256((HERE/'encoder_lora.py').read_bytes()).hexdigest()}
    model._ASR18_DEV_weights_restored = False
    def require_restored(current, inputs):
        require(current._ASR18_DEV_weights_restored is True,
                'No forward until every saved207 weight has been restored and byte checked')
    model._ASR18_DEV_restore_gate = model.register_forward_pre_hook(require_restored)
    model.eval()
    return {'kind': 'ASR18_optimizer_free_uninitialized207_DEV_structure_v1',
        'vocabulary': 8711, 'blank': 8192, 'duration_rows': [8711, 8716],
        'named_parameters': list(names), 'named_parameters_sha256': digest(list(names)),
        'tokenizer_sha256': _hash(candidate), 'candidate_exact': True,
        'suppression': suppression, 'LoRA': encoder_lora_metadata(model),
        'optimizer_objects_constructed': 0, 'Adam_tensor_views_created': 0,
        'sampler_or_replay_objects_constructed': 0, 'all_saved_weights_required_before_forward': True}


def assert_ready(model, *, torch, names):
    from bpe_broad_production_v3 import frozen_parameter
    parameters = dict(model.named_parameters())
    require(model._ASR18_DEV_weights_restored is True and set(names) == set(expected_names(model))
            and len(names) == 207 and not any(p.requires_grad or p.grad is not None for p in parameters.values())
            and all(parameters[name].device.type == 'cuda' and parameters[name].dtype == torch.float32
                    for name in names), 'Restored, frozen optimizer-free207 GPU model required')
    require(all((p := frozen_parameter(model, name)).device.type == 'cpu' and not p.requires_grad
                and p.data_ptr() == backing.data_ptr()
                for name, backing in model._diagnostic_frozen_backing.items())
            and model._diagnostic_frozen_metrics['logical_current_matrix_bytes'] == 0,
            'Original readonly backing and empty serial hooks required')
