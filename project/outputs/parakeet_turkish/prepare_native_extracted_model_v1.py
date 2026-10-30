"""Extract the pinned native model into normalized workspace paths without tar mode changes."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SOURCE = ROOT / 'work/models/hf_native_cache/models--nvidia--parakeet-tdt-0.6b-v3/snapshots/541d1f99c6b0c3cd0b11a95167540bb8edefd82b/parakeet-tdt-0.6b-v3.nemo'
TARGET = ROOT / 'work/nemo_direct_trial_v1/native_base_extracted'
PROOF = OUT / 'native_workspace_extraction_v1.json'


def file_sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    if TARGET.exists() or PROOF.exists():
        raise FileExistsError('Preserve prior extraction and proof')
    target = TARGET.resolve()
    if not target.is_relative_to(ROOT.resolve()):
        raise ValueError('Extraction target outside workspace')
    expected = json.loads((OUT / 'runs/full_nemo_turkish_20261011_pilot_01/preflight.json').read_text(encoding='utf-8'))['native_base_sha256']
    source_sha = file_sha(SOURCE)
    if source_sha != expected:
        raise ValueError('Native archive changed since actual pilot preflight')
    target.mkdir(parents=True)
    files = {}
    with tarfile.open(SOURCE, 'r:*') as archive:
        for member in archive:
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or '..' in relative.parts or any(':' in part for part in relative.parts):
                raise ValueError('Unsafe native archive member: ' + member.name)
            destination = target.joinpath(*relative.parts).resolve()
            if not destination.is_relative_to(target):
                raise ValueError('Archive member escapes destination')
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError('Native archive has an unsupported link or special member')
            destination.parent.mkdir(parents=True, exist_ok=True)
            value = hashlib.sha256()
            count = 0
            with archive.extractfile(member) as stream, destination.open('xb') as output:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                    output.write(block)
                    value.update(block)
                    count += len(block)
            if count != member.size:
                raise ValueError('Truncated native archive member')
            files[str(destination)] = {'sha256': value.hexdigest(), 'bytes': count}
    if not (target / 'model_config.yaml').is_file() or not (target / 'model_weights.ckpt').is_file():
        raise ValueError('Native model config or weights missing')
    proof = {'kind': 'native_workspace_extraction_v1', 'passed': True,
             'source_path': str(SOURCE), 'source_sha256': source_sha,
             'extracted_directory': str(target), 'files': files,
             'archive_file_bytes_preserved': True, 'archive_permissions_not_applied': True,
             'created_at_utc': datetime.now(timezone.utc).isoformat()}
    with PROOF.open('x', encoding='utf-8') as stream:
        json.dump(proof, stream, indent=2)
    print(json.dumps({'passed': True, 'directory': str(target), 'files': len(files)}))


if __name__ == '__main__':
    main()
