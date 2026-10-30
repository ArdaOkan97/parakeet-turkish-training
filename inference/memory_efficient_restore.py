"""Optional Windows-friendly mmap restore; ordinary NeMo model, no custom layers.

The .nemo also loads with EncDecRNNTBPEModel.restore_from without this helper
when sufficient host memory is available. This helper retains an extracted
archive cache. Call transcribe under torch.autocast for BF16 matrix placement.
"""
import json
from pathlib import Path
import tarfile
import tempfile
import torch


def restore_model(nemo_path, *, cache_dir, device='cpu', matrix_dtype=None):
    from nemo.collections.asr.models import EncDecRNNTBPEModel
    from nemo.core.connectors.save_restore_connector import SaveRestoreConnector
    class MemoryMappedRestore(SaveRestoreConnector):
        @staticmethod
        def _load_state_dict_from_disk(model_weights, map_location='cpu'):
            return torch.load(model_weights, map_location=map_location, mmap=True, weights_only=False)
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    extracted = Path(tempfile.mkdtemp(prefix='parakeet_tr_', dir=cache))
    with tarfile.open(nemo_path, 'r:*') as archive:
        for member in archive.getmembers():
            if member.isdir() and member.name.rstrip('/') in ('.', ''):
                continue
            if not member.isfile() or len(Path(member.name).parts) != 1:
                raise ValueError(f'Unexpected native archive member: {member.name}')
        archive.extractall(extracted, filter='data')
    connector = MemoryMappedRestore()
    connector.model_extracted_dir = str(extracted)
    model = EncDecRNNTBPEModel.restore_from(str(nemo_path), map_location='cpu', save_restore_connector=connector)
    model.eval()
    model.requires_grad_(False)
    if matrix_dtype is not None:
        config = json.loads((Path(__file__).parent / 'inference_config.json').read_text())
        parameters = dict(model.named_parameters())
        for name in config['frozen_matrix_names']:
            parameters[name].data = parameters[name].data.to(matrix_dtype)
    return model.to(device)
