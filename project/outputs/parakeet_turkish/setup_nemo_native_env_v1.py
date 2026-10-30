"""Isolated NeMo dependency overlay; never run pip transactions in training-env.

prepare creates a NEW venv with a read-only import path to existing packages.
install writes only NEW site-packages via pip --target/--no-deps, using the exact
successful constrained resolver report. CUDA is hidden for every subprocess.
No GPU training or inference is implemented here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parents[1]
PARENT = ROOT/'work/training-env'
ENV = ROOT/'work/nemo-native-env-v1'
WORK = ROOT/'work/nemo_native_setup_v1'
PLAN = ROOT/'work/research/nemo_reference/nemo_native_windows_dependency_plan_v4.json'
PARENT_SITE = PARENT/'Lib/site-packages'
SITE = ENV/'Lib/site-packages'
PYTHON = ENV/'Scripts/python.exe'
EXCLUDED = {'torch', 'torchvision', 'torchaudio', 'pip'}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, data):
    temporary = path.with_name(path.name+'.new-'+str(os.getpid()))
    temporary.write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8')
    os.replace(temporary, path)


def process_env():
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES='-1', PYTHONUTF8='1',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               TMP=str(WORK/'tmp'), TEMP=str(WORK/'tmp'),
               HF_HUB_DISABLE_TELEMETRY='1', PIP_DISABLE_PIP_VERSION_CHECK='1')
    env.pop('PYTHONPATH', None)
    env.pop('PIP_TARGET', None)
    env.pop('PIP_PREFIX', None)
    return env


def run(command, log=None):
    if log:
        require(not log.exists(), 'Do not overwrite an earlier setup log: '+str(log))
        with log.open('w', encoding='utf-8') as stream:
            result = subprocess.run(command, cwd=ROOT, env=process_env(),
                                    stdout=stream, stderr=subprocess.STDOUT,
                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        require(result.returncode == 0, 'Setup subprocess failed; inspect '+str(log))
        return None
    result = subprocess.run(command, cwd=ROOT, env=process_env(), text=True,
                            encoding='utf-8', capture_output=True,
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    require(result.returncode == 0, result.stdout+result.stderr)
    return result.stdout


def parent_evidence():
    source = """import importlib.metadata as m, json, pathlib, sys, torch
print(json.dumps({'torch_version':torch.__version__,'torch_module_path':str(pathlib.Path(torch.__file__).resolve()),
'torch_cuda_build':torch.version.cuda,'cuda_initialized':torch.cuda.is_initialized(),
'base_site_packages':str((pathlib.Path(sys.base_prefix)/'Lib/site-packages').resolve()),
'distribution_versions':{x:m.version(x) for x in ('torch','transformers','numpy','safetensors')},
'torch_module_sha_path':str(pathlib.Path(torch.__file__).resolve())}))
"""
    data = json.loads(run([str(PARENT/'Scripts/python.exe'), '-X', 'utf8', '-c', source]))
    require(data['torch_version'] == '2.7.1+cu128' and data['torch_cuda_build'] == '12.8', 'Unexpected parent Torch build')
    require(data['cuda_initialized'] is False, 'CPU evidence initialized CUDA')
    require(any(Path(data['torch_module_path']).is_relative_to(path) for path in
                (PARENT_SITE, Path(data['base_site_packages']))), 'Parent Torch origin differs')
    data['torch_init_sha256'] = sha(data['torch_module_path'])
    for filename in ('train_lora_batched.py', 'efficient_tdt.py', 'efficient_tdt_v2.py', 'runtime.py', 'batched_lora.py'):
        data[filename+'_sha256'] = sha(OUTPUT/filename)
    return data


def validated_plan():
    require(PLAN.is_file(), 'Successful constrained v4 resolver plan missing')
    plan = read(PLAN)
    require(plan['environment']['sys_platform'] == 'win32' and plan['environment']['python_version'] == '3.11', 'Resolver platform differs')
    packages = {}
    lines = []
    for item in plan['install']:
        metadata = item['metadata']
        name, version = metadata['name'], metadata['version']
        normalized = name.lower().replace('_', '-').replace('.', '-')
        require(re.fullmatch(r'[A-Za-z0-9_.-]+', name) and re.fullmatch(r'[A-Za-z0-9_.+!-]+', version), 'Unsafe resolver requirement')
        require(normalized not in EXCLUDED, 'Resolver would replace parent Torch/pip: '+normalized)
        require(normalized not in packages and not item.get('is_yanked'), 'Duplicate/yanked package')
        digest = item['download_info']['archive_info']['hashes']['sha256']
        require(re.fullmatch(r'[a-f0-9]{64}', digest), 'Package archive SHA missing')
        url = item['download_info']['url']
        require(url.startswith('https://files.pythonhosted.org/'), 'Unexpected non-PyPI archive origin')
        packages[normalized] = {'name': name, 'version': version, 'sha256': digest, 'url': url}
        lines.append(f'{name}=={version} --hash=sha256:{digest}')
    require(packages['nemo-toolkit']['version'] == '3.0.0', 'Unexpected NeMo version')
    require(packages['pytorch-lightning']['version'] == packages['lightning']['version'] == '2.4.0', 'Both Lightning distributions must be2.4.0')
    return packages, '\n'.join(lines)+'\n'


def prepared():
    marker = WORK/'preparation.json'
    require(marker.is_file() and PYTHON.is_file(), 'Run --stage prepare first')
    state = read(marker)
    require(state['helper_sha256'] == sha(__file__) and state['resolver_plan_sha256'] == sha(PLAN), 'Prepared helper/plan changed')
    require((SITE/'read_training_packages.pth').read_text(encoding='utf-8') == state['parent_import_path_text'], 'Read-only parent import path changed')
    require(state['parent_evidence'] == parent_evidence(), 'Parent protected sources/Torch changed')
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('prepare', 'install', 'cpu-preflight'))
    args = parser.parse_args()
    require(ENV.resolve().is_relative_to((ROOT/'work').resolve()) and ENV.resolve() != PARENT.resolve(), 'Environment target leaves own workspace')
    packages, requirements = validated_plan()
    if args.stage == 'prepare':
        require(not ENV.exists() and not (WORK/'preparation.json').exists(), 'New environment/preparation already exists')
        if WORK.exists():
            require({p.name for p in WORK.iterdir()} <= {'tmp', 'pip-cache'}, 'Unexpected prior setup files')
        WORK.mkdir(exist_ok=True)
        (WORK/'tmp').mkdir(exist_ok=True)
        (WORK/'pip-cache').mkdir(exist_ok=True)
        before = parent_evidence()
        run([str(PARENT/'Scripts/python.exe'), '-m', 'venv', '--without-pip', str(ENV)], WORK/'prepare.log')
        require(SITE.is_dir(), 'New venv site-packages missing')
        import_paths = str(PARENT_SITE.resolve())+'\n'+before['base_site_packages']+'\n'
        (SITE/'read_training_packages.pth').write_text(import_paths, encoding='utf-8')
        (WORK/'requirements-resolved.txt').write_text(requirements, encoding='utf-8')
        (WORK/'constraints.txt').write_text('torch==2.7.1\npytorch-lightning==2.4.0\nlightning==2.4.0\n', encoding='utf-8')
        save(WORK/'preparation.json', {'schema':1, 'helper_sha256':sha(__file__),
            'environment_path':str(ENV), 'parent_import_path':str(PARENT_SITE),
            'parent_import_path_text':import_paths,
            'parent_environment_is_import_only':True, 'parent_evidence':before,
            'resolver_plan_path':str(PLAN), 'resolver_plan_sha256':sha(PLAN),
            'resolved_packages':packages,
            'requirements_sha256':sha(WORK/'requirements-resolved.txt'),
            'installation_policy':'pip --target NEW/site-packages --no-deps --ignore-installed --require-hashes; no transaction in parent environment',
            'cuda_visible_devices':'-1'})
        require(parent_evidence() == before, 'Parent source changed during prepare')
        print(json.dumps({'prepared': True, 'path':str(ENV), 'helper_sha256':sha(__file__), 'next_stage':'install'}))
        return
    state = prepared()
    if args.stage == 'install':
        require(not (WORK/'installation.json').exists(), 'Installation receipt already exists')
        require((WORK/'requirements-resolved.txt').read_text(encoding='utf-8') == requirements and
                sha(WORK/'requirements-resolved.txt') == state['requirements_sha256'], 'Resolved requirements changed')
        command = [str(PYTHON), '-m', 'pip', '--isolated', 'install',
            '--target', str(SITE), '--no-deps', '--ignore-installed', '--require-hashes',
            '--no-compile', '--no-warn-script-location', '--disable-pip-version-check',
            '--cache-dir', str(WORK/'pip-cache'), '-r', str(WORK/'requirements-resolved.txt')]
        run(command, WORK/'install.log')
        after = parent_evidence()
        require(after == state['parent_evidence'], 'Parent protected sources/Torch changed during install')
        save(WORK/'installation.json', {'schema':1, 'status':'complete', 'helper_sha256':sha(__file__),
            'resolver_plan_sha256':sha(PLAN), 'command':command, 'parent_evidence_before':state['parent_evidence'],
            'parent_evidence_after':after, 'parent_environment_mutated':False,
            'pip_target':str(SITE), 'torch_install_requested':False, 'cuda_visible_devices':'-1'})
        print(json.dumps({'installed':True, 'path':str(ENV), 'next_stage':'cpu-preflight'}))
        return
    require((WORK/'installation.json').is_file(), 'Install receipt missing')
    require(not (WORK/'cpu_preflight.json').exists(), 'Preserve existing CPU preflight')
    source = """import importlib, importlib.metadata as m, json, pathlib, traceback, torch
torch.set_num_threads(2)
data={'torch_version':torch.__version__,'torch_module_path':str(pathlib.Path(torch.__file__).resolve()),'cuda_initialized_before':torch.cuda.is_initialized(),'modules':{},'versions':{}}
for name in ('nemo_toolkit','pytorch_lightning','lightning','numba_cuda'):
 try:data['versions'][name]=m.version(name)
 except Exception as error:data['versions'][name]=type(error).__name__+': '+str(error)
for name in ('nemo','lightning.pytorch','pytorch_lightning','nemo.collections.asr.models','nemo.collections.asr.losses.rnnt','nemo.collections.asr.parts.numba.rnnt_loss'):
 try:
  module=importlib.import_module(name);data['modules'][name]={'passed':True,'path':getattr(module,'__file__',None)}
 except Exception as error:data['modules'][name]={'passed':False,'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()}
data['cuda_initialized_after']=torch.cuda.is_initialized()
data['passed']=all(x['passed'] for x in data['modules'].values()) and not data['cuda_initialized_after'] and data['torch_version']=='2.7.1+cu128'
print(json.dumps(data))
"""
    text = run([str(PYTHON), '-X', 'utf8', '-c', source])
    # NeMo imports may log to stdout; the final nonempty line is the JSON result.
    data = json.loads(text.strip().splitlines()[-1])
    (WORK/'cpu_preflight_stdout.txt').write_text(text, encoding='utf-8')
    require(data['cuda_initialized_before'] is False and data['cuda_initialized_after'] is False, 'CPU import initialized CUDA')
    require(data['torch_module_path'] == state['parent_evidence']['torch_module_path'], 'Torch was replaced in overlay')
    data.update(helper_sha256=sha(__file__), resolver_plan_sha256=sha(PLAN),
                scope='Native imports with CUDA hidden, not CUDA loss execution or model training',
                parent_environment_mutated=False, cuda_visible_devices='-1')
    save(WORK/'cpu_preflight.json', data)
    print(json.dumps({'passed':data['passed'],'report':str(WORK/'cpu_preflight.json')}))
    if not data['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
