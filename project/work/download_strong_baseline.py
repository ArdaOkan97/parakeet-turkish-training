import json
import os
from pathlib import Path
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_DISABLE_XET'] = '1'
from huggingface_hub import HfApi, snapshot_download
root = Path(__file__).resolve().parent.parent
repo = 'openai/whisper-large-v3'
info = HfApi().model_info(repo)
target = root / 'work/models/whisper-large-v3'
lock = {'repository': repo, 'revision': info.sha, 'local_path': str(target)}
(root / 'work/strong_baseline_model.lock.json').write_text(json.dumps(lock, indent=2)+'\n', encoding='utf-8')
print(json.dumps(lock), flush=True)
path = snapshot_download(repo_id=repo, revision=info.sha, local_dir=target,
                         allow_patterns=['model.safetensors', '*.json', '*.txt', '*.model'],
                         ignore_patterns=['onnx/*', 'coreml/*', 'flax*', 'tf*'], max_workers=4)
print(json.dumps({'complete': True, 'path': path, **lock}), flush=True)
