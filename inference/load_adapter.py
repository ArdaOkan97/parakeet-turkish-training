"""Load the native NeMo LoRA + fully trained heads in this repository.

This is a custom NeMo adapter format, not a PEFT checkpoint. The base model
is pinned and downloaded through huggingface_hub; no remote code is executed.
"""
from pathlib import Path
import copy
import json
import tarfile
import tempfile

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    """Unmerged rank-r update while retaining ordinary native parameter names."""
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.weight = linear.weight
        self.bias = linear.bias
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features, dtype=self.weight.dtype, device=self.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, dtype=self.weight.dtype, device=self.weight.device))
        self.scaling = float(alpha) / rank

    def forward(self, x):
        return F.linear(x, self.weight, self.bias) + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling


def load_model(adapter_dir, *, base_nemo=None, device='cpu', cache_dir=None, memory_mapped=True, matrix_dtype=None):
    """Return a frozen EncDecRNNTBPEModel with unmerged adapters.

    Set cache_dir / HF_HOME / TMPDIR to storage appropriate for your system.
    base_nemo optionally supplies the exact pinned NVIDIA native .nemo locally.
    Default FP32 execution is the validated inference setting. This is an
    inference release; it does not restore training optimizer/sampler state.
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from omegaconf import open_dict
    from nemo.collections.asr.models import EncDecRNNTBPEModel
    from nemo.collections.asr.losses.rnnt import RNNTLoss
    from nemo.core.connectors.save_restore_connector import SaveRestoreConnector
    root = Path(adapter_dir)
    cfg = json.loads((root / 'adapter_config.json').read_text(encoding='utf-8'))
    if cfg['format'] != 'parakeet_nemo_lora_and_full_heads_v1':
        raise ValueError('Unsupported adapter format')
    if base_nemo is None:
        base_nemo = hf_hub_download(cfg['base_model'], 'parakeet-tdt-0.6b-v3.nemo',
                                   revision=cfg['base_revision'], cache_dir=cache_dir)
    class MemoryMappedRestore(SaveRestoreConnector):
        @staticmethod
        def _load_state_dict_from_disk(model_weights, map_location='cpu'):
            return torch.load(model_weights, map_location=map_location, mmap=True, weights_only=False)
    connector = MemoryMappedRestore() if memory_mapped else SaveRestoreConnector()
    if memory_mapped:
        # On Windows a live mmap cannot be unlinked by NeMo's temporary-dir
        # cleanup. Keep an extracted base cache; all members must be flat files.
        from huggingface_hub.constants import HF_HUB_CACHE
        extracted_cache = Path(cache_dir or HF_HUB_CACHE) / 'nemo_native_extracted'
        extracted_cache.mkdir(parents=True, exist_ok=True)
        extracted = Path(tempfile.mkdtemp(prefix=cfg['base_revision'][:12] + '_', dir=extracted_cache))
        with tarfile.open(base_nemo, 'r:*') as archive:
            for member in archive.getmembers():
                if member.isdir() and member.name.rstrip('/') in ('.', ''):
                    continue
                if not member.isfile() or len(Path(member.name).parts) != 1:
                    raise ValueError(f'Unexpected native archive member: {member.name}')
            archive.extractall(extracted, filter='data')
        connector.model_extracted_dir = str(extracted)
    model = EncDecRNNTBPEModel.restore_from(str(base_nemo), map_location='cpu', save_restore_connector=connector)
    loss_cfg = copy.deepcopy(model.cfg.loss)
    loss_name, loss_kwargs = model.extract_rnnt_loss_cfg(loss_cfg)
    if loss_name != 'tdt':
        raise ValueError('Expected the NVIDIA Parakeet TDT base')
    model.change_vocabulary(str(root / 'tokenizer'), 'bpe')
    with open_dict(model.cfg):
        model.cfg.loss = loss_cfg
        model.cfg.train_ds = None
        model.cfg.validation_ds = {}
        model.cfg.test_ds = None
        model.cfg.preprocessor.dither = 0.0
        model.cfg.decoding.strategy = 'greedy_batch'
        model.cfg.decoding.greedy.use_cuda_graph_decoder = False
    # change_vocabulary defaults to the RNNT loss; retain the original TDT.
    model.loss = RNNTLoss(num_classes=cfg['native_vocabulary'], loss_name=loss_name,
                         loss_kwargs=loss_kwargs, reduction=model.cfg.get('rnnt_reduction', 'mean_batch'))
    model.change_decoding_strategy(copy.deepcopy(model.cfg.decoding), verbose=False)
    if model.joint.fuse_loss_wer:
        model.joint.set_loss(model.loss)
        model.joint.set_wer(model.wer)
    # Training used these frozen base matrices in BF16. Preserve their values
    # (rounded to BF16, represented in FP32), rather than silently using FP32 originals.
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name in cfg['frozen_matrix_names']:
            parameters[name].copy_(parameters[name].to(torch.bfloat16).float())
    weights = load_file(str(root / 'adapter_model.safetensors'), device='cpu')
    parents = sorted({name.rsplit('.', 1)[0] for name in weights if name.endswith('.lora_A')})
    if len(parents) != 96:
        raise ValueError('Expected 96 attention projection adapters')
    for name in parents:
        parent_name, child = name.rsplit('.', 1)
        parent = model.get_submodule(parent_name)
        setattr(parent, child, LoRALinear(getattr(parent, child), cfg['rank'], cfg['alpha']))
    parameters = dict(model.named_parameters())
    if len(weights) != 207:
        raise ValueError('Expected 192 LoRA tensors and 15 fully trained head tensors')
    with torch.no_grad():
        for name, value in weights.items():
            if name not in parameters or parameters[name].shape != value.shape:
                raise ValueError(f'Adapter shape/name mismatch: {name}')
            parameters[name].copy_(value)
    model.eval()
    model.requires_grad_(False)
    if matrix_dtype is not None:
        parameters = dict(model.named_parameters())
        for name in cfg['frozen_matrix_names']:
            parameters[name].data = parameters[name].data.to(matrix_dtype)
    return model.to(device)
