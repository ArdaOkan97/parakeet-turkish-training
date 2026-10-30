"""Inactive ASR-18 bootstrap: GPU admission required; base weights load directly to GPU."""
from pathlib import Path
import copy
import gc
import hashlib
import json
import os
import random

PROJECT = Path(__file__).resolve().parents[2]
STORAGE = Path(r'E:\gpu_training_scratchpad')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_broad_BPE_model(*, native_admission, source_records, run_id, stage_callback):
    """Only a freshly admitted ROOT native worker may call this function."""
    lease = json.loads((PROJECT / 'work/gpu_training_owner.json').read_text(encoding='utf-8-sig'))
    require(lease['owner'] == '/root' and lease['actual_GPU_running'] is True
            and lease['actual_owned_worker_pid'] == os.getpid(), 'Fresh ROOT GPU ownership required')
    require(native_admission['passed'] is True and native_admission['pid'] == os.getpid()
            and native_admission['run_id'] == run_id and native_admission['CUDA_version'] == '12.9'
            and native_admission['native_batch_cap_bytes'] == 8000000000
            and native_admission['maximum_full_joint_bytes'] == 1700000000
            and native_admission['native_end_cache_bytes'] == 3000000000,
            'Unchanged native runtime and memory bounds required')
    for key in ('TEMP', 'TMP', 'HF_HOME', 'TORCH_EXTENSIONS_DIR'):
        require(Path(os.environ[key]).resolve().is_relative_to(STORAGE.resolve()), 'E-only process environment required')
    import torch
    from transformers import AutoModelForTDT, AutoProcessor
    from runtime import corrected_tdt_loss
    from encoder_lora import install_encoder_lora, encoder_lora_metadata
    from extend_vocab import extend_processor_and_model
    from train_lora import checkpoint_encoder_blocks
    from train_lora_batched_tsc_bpe_full_decoder_v1 import expand_candidate
    from bpe_broad_full207_optimizer_v1 import BPEBroadFull207Optimizer

    base = Path((PROJECT / 'work/base_model_path.txt').read_text().strip()).resolve()
    require(base.is_relative_to(PROJECT.resolve()) or base.is_relative_to(STORAGE.resolve()),
            'Existing project or E: base model required')
    stage_callback('before_direct_GPU_base_load')
    processor = AutoProcessor.from_pretrained(base, local_files_only=True)
    # Avoid the former complete FP32 CPU model followed by a second GPU copy.
    model = AutoModelForTDT.from_pretrained(base, dtype=torch.float32, local_files_only=True,
                                          low_cpu_mem_usage=True, device_map={'': 'cuda:0'})
    require(all(p.device.type == 'cuda' for p in model.parameters()), 'Direct GPU placement incomplete')
    require(all(p.dtype == torch.float32 for p in model.parameters() if p.is_floating_point()),
            'Preserve original FP32 parameter storage')
    model.loss_function = corrected_tdt_loss
    teacher = {n: copy.deepcopy(getattr(model, n)).eval().requires_grad_(False)
               for n in ('encoder_projector', 'decoder', 'joint')}
    extend_processor_and_model(processor, model)
    layout = install_encoder_lora(model, rank=8, alpha=16, last_n_layers=24)
    model.config.encoder_lora_tracking = {'layout': layout, 'kind': 'unmerged',
        'implementation_sha256': hashlib.sha256((PROJECT / 'outputs/parakeet_turkish/encoder_lora.py').read_bytes()).hexdigest()}
    stage_callback('after_direct_GPU_base_and_original_teacher')
    for item in source_records.values():
        path = Path(item['path']).resolve()
        require(path.is_relative_to(PROJECT.resolve()) or path.is_relative_to(STORAGE.resolve()), 'Source path escapes allowed roots')
        require(path.stat().st_size == item['bytes'], 'Pinned checkpoint size differs')
        with path.open('rb') as stream:
            require(hashlib.file_digest(stream, 'sha256').hexdigest() == item['sha256'], 'Pinned checkpoint SHA differs')
    original = torch.load(source_records['original']['path'], map_location='cpu', mmap=True, weights_only=True)
    bpe = torch.load(source_records['bpe']['path'], map_location='cpu', mmap=True, weights_only=True)
    for name, parameter in model.named_parameters():
        if name in original['weights']:
            with torch.no_grad():
                parameter.copy_(original['weights'][name])
    expansion = expand_candidate(model, processor, original)
    require(expansion['vocabulary'] == 8711 and expansion['blank'] == 8192, 'Frozen8711 BPE geometry required')
    result = model.load_state_dict(bpe['weights'], strict=False)
    require(not result.unexpected_keys and not any(n in bpe['weights'] for n in result.missing_keys),
            'All207 BPE warm-start tensors must load')
    optimizer = BPEBroadFull207Optimizer(model, original, bpe, source_records=source_records, torch_module=torch)
    checkpoint_encoder_blocks(model)
    require(sum(p.requires_grad for p in model.parameters()) == 207, 'All207 active tensors required')
    # Preserve source RNG after construction. The new dataset sampler is a separately declared phase transition.
    random.setstate(bpe['python_rng'])
    torch.set_rng_state(bpe['torch_rng'])
    torch.cuda.set_rng_state_all(bpe['cuda_rng'])
    provenance = {'optimizer': optimizer.identity(), 'parent_step': bpe['step'],
                  'parent_run_id': bpe['run_id'], 'parent_selected_step': bpe['selected_step'],
                  'parent_history_storage': 'immutable_pinned_parent_checkpoint_not_copied_into_new_phase_history',
                  'LoRA': encoder_lora_metadata(model), 'BPE_expansion_geometry': expansion,
                  'direct_GPU_base_loading': True, 'full_CPU_base_model_copy_created': False,
                  'positive_native_training_update_proved': False}
    del original, bpe
    gc.collect()
    stage_callback('after_source_moment_transfer_and_CPU_checkpoint_release')
    model.encoder.eval()
    return model, processor, teacher, optimizer, provenance
