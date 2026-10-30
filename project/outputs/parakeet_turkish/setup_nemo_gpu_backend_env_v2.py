"""NEW read-only layered NeMo CUDA compiler environment; parent owns network.

The existing training/native-v1 environments are import-only. Resolver and pip
target only the new v2 site, with exact shared Torch/Numba/llvmlite constraints.
Every subprocess has CUDA hidden. No GPU kernel/training is executed here.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import setup_nemo_native_env_v1 as prior

ROOT, OUTPUT = prior.ROOT, prior.OUTPUT
ENV = ROOT/'work/nemo-native-env-v2'
SITE = ENV/'Lib/site-packages'
PYTHON = ENV/'Scripts/python.exe'
WORK = ROOT/'work/nemo_gpu_backend_setup_v2'
PLAN = WORK/'dependency_plan.json'
EXCLUDE = {'torch','torchaudio','torchvision','pip','nemo-toolkit','numba','llvmlite','numpy','lightning','pytorch-lightning'}
require, sha = prior.require, prior.sha


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:stream.write(json.dumps(value,indent=2)+'\n')


def environment():
    value=dict(os.environ)
    value.update(CUDA_VISIBLE_DEVICES='-1',PYTHONUTF8='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                 TMP=str(WORK/'tmp'),TEMP=str(WORK/'tmp'),PIP_DISABLE_PIP_VERSION_CHECK='1')
    for key in ('PYTHONPATH','PIP_TARGET','PIP_PREFIX'):value.pop(key,None)
    return value


def run(command,log=None):
    if log:
        require(not log.exists(),'Preserve earlier setup log')
        with log.open('x',encoding='utf-8') as stream:
            result=subprocess.run(command,cwd=ROOT,env=environment(),stdout=stream,stderr=subprocess.STDOUT,
                                  creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        require(result.returncode==0,'Setup failed: '+str(log));return None
    result=subprocess.run(command,cwd=ROOT,env=environment(),capture_output=True,text=True,encoding='utf-8',
                          creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    require(result.returncode==0,result.stdout+result.stderr)
    return result.stdout


def protected():
    value=prior.parent_evidence()
    modules=('nemo/collections/asr/losses/rnnt.py','nemo/collections/asr/losses/rnnt_pytorch.py',
             'nemo/collections/asr/parts/numba/rnnt_loss/rnnt_pytorch.py',
             'nemo/collections/asr/parts/numba/rnnt_loss/utils/cuda_utils/gpu_rnnt.py',
             'nemo/collections/asr/parts/numba/rnnt_loss/utils/cuda_utils/gpu_rnnt_kernel.py')
    value['native_v1_files_sha256']={name:sha(prior.SITE/name) for name in modules}
    value['native_v1_install_receipt_sha256']=sha(prior.WORK/'installation.json')
    return value


def prepared():
    value=read(WORK/'preparation.json')
    require(value['helper_sha256']==sha(__file__) and value['protected']==protected(),'Source/parent changed')
    require((SITE/'read_native_packages.pth').read_text(encoding='utf-8')==value['import_path_text'],'Layer import paths changed')
    return value


def plan():
    require(PLAN.is_file(),'Successful NEW resolver plan required')
    value=read(PLAN)
    require(value['environment']['sys_platform']=='win32' and value['environment']['python_version']=='3.11','Resolver platform')
    packages={};lines=[]
    for item in value['install']:
        name=item['metadata']['name'];version=item['metadata']['version']
        key=name.lower().replace('_','-').replace('.','-')
        require(key not in EXCLUDE and key not in packages and not item.get('is_yanked'),'Forbidden/duplicate/yanked dependency:'+key)
        require(re.fullmatch(r'[A-Za-z0-9_.-]+',name) and re.fullmatch(r'[A-Za-z0-9_.+!-]+',version),'Unsafe requirement')
        digest=item['download_info']['archive_info']['hashes']['sha256']
        require(re.fullmatch('[a-f0-9]{64}',digest),'Archive hash missing')
        require(item['download_info']['url'].startswith('https://files.pythonhosted.org/'),'Unexpected archive origin')
        packages[key]={'name':name,'version':version,'sha256':digest,'url':item['download_info']['url']}
        lines.append(f'{name}=={version} --hash=sha256:{digest}')
    require('numba-cuda' in packages and 'cuda-python' in packages,'Officialcu12 compiler/bindings missing')
    return packages,'\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=('prepare','resolve','install','cpu-preflight'))
    args=parser.parse_args()
    require(ENV.resolve().is_relative_to((ROOT/'work').resolve()) and ENV not in (prior.ENV,prior.PARENT),'Own NEW environment only')
    if args.stage=='prepare':
        require(not ENV.exists() and not (WORK/'preparation.json').exists(),'Preserve prior environment')
        WORK.mkdir(exist_ok=True);(WORK/'tmp').mkdir(exist_ok=True);(WORK/'pip-cache').mkdir(exist_ok=True)
        before=protected()
        run([str(prior.PYTHON),'-m','venv','--without-pip',str(ENV)],WORK/'prepare.log')
        paths=[prior.SITE,prior.PARENT_SITE,Path(before['base_site_packages'])]
        text=''.join(str(path.resolve())+'\n' for path in paths)
        (SITE/'read_native_packages.pth').write_text(text,encoding='utf-8')
        (WORK/'constraints.txt').write_text('torch==2.7.1\nnumba==0.68.0\nllvmlite==0.50.0\npytorch-lightning==2.4.0\nlightning==2.4.0\n',encoding='utf-8')
        save(WORK/'preparation.json',{'schema':1,'kind':'read_only_native_cuda_backend_overlay',
            'helper_sha256':sha(__file__),'environment_path':str(ENV),'import_path_text':text,'protected':before,
            'cuda_visible_devices':'-1','parent_environments_import_only':True,
            'official_requirements':'Pinned NeMo cu12 extra requires numba-cuda[cu12] and cuda-python>=12,<13 on Windows; GPU success still unmeasured'})
        require(before==protected(),'Protected sources changed')
        print(json.dumps({'prepared':True,'environment':str(ENV)}));return
    state=prepared()
    if args.stage=='resolve':
        require(not PLAN.exists(),'Preserve original dependency plan')
        command=[str(PYTHON),'-m','pip','--isolated','install','--dry-run','--prefer-binary',
            '--constraint',str(WORK/'constraints.txt'),'--report',str(PLAN),
            '--cache-dir',str(WORK/'pip-cache'),'numba-cuda[cu12]','cuda-python>=12,<13']
        run(command,WORK/'resolver.log')
        packages,requirements=plan()
        (WORK/'requirements-resolved.txt').write_text(requirements,encoding='utf-8')
        save(WORK/'resolution.json',{'schema':1,'passed':True,'command':command,'plan_sha256':sha(PLAN),
            'resolved_packages':packages,'helper_sha256':sha(__file__),'parent_environments_mutated':False})
        require(state['protected']==protected(),'Resolver changed protected environment')
        print(json.dumps({'resolved':True,'packages':list(packages)}));return
    packages,requirements=plan()
    resolution=read(WORK/'resolution.json')
    require(resolution['plan_sha256']==sha(PLAN) and (WORK/'requirements-resolved.txt').read_text(encoding='utf-8')==requirements,'Resolver identity differs')
    if args.stage=='install':
        require(not (WORK/'installation.json').exists(),'Preserve installation receipt')
        command=[str(PYTHON),'-m','pip','--isolated','install','--target',str(SITE),'--no-deps','--ignore-installed',
                 '--require-hashes','--no-compile','--no-warn-script-location','--cache-dir',str(WORK/'pip-cache'),
                 '-r',str(WORK/'requirements-resolved.txt')]
        run(command,WORK/'install.log')
        require(state['protected']==protected(),'Install changed protected environments')
        save(WORK/'installation.json',{'schema':1,'status':'complete','command':command,'helper_sha256':sha(__file__),
            'plan_sha256':sha(PLAN),'packages':packages,'parent_environments_mutated':False,'GPU_executed':False})
        print(json.dumps({'installed':True,'environment':str(ENV)}));return
    require((WORK/'installation.json').exists(),'Installation receipt required')
    source="""import json,pathlib,importlib.metadata as m,torch,numba,numba.cuda,llvmlite,nemo
from nemo.collections.asr.parts.numba.rnnt_loss import TDTLossNumba
data={'torch_version':torch.__version__,'torch_path':str(pathlib.Path(torch.__file__).resolve()),'numba_version':numba.__version__,'numba_path':str(pathlib.Path(numba.__file__).resolve()),'cuda_backend_path':str(pathlib.Path(numba.cuda.__file__).resolve()),'llvmlite_version':llvmlite.__version__,'nemo_path':str(pathlib.Path(nemo.__file__).resolve()),'numba_cuda_version':m.version('numba-cuda'),'cuda_python_version':m.version('cuda-python'),'cuda_initialized':torch.cuda.is_initialized()}
print(json.dumps(data))
"""
    output=run([str(PYTHON),'-X','utf8','-c',source]);data=json.loads(output.strip().splitlines()[-1])
    require(data['torch_path']==state['protected']['torch_module_path'] and data['torch_version']=='2.7.1+cu128','Shared Torch replaced')
    require(data['numba_version']=='0.68.0' and data['llvmlite_version']=='0.50.0','Shared core compiler replaced')
    require(Path(data['cuda_backend_path']).is_relative_to(SITE),'CUDA compiler still comes from deprecated inherited backend')
    require(Path(data['nemo_path']).is_relative_to(prior.SITE) and data['cuda_initialized'] is False,'NeMo scope/CUDA CPU preflight')
    data.update(passed=True,helper_sha256=sha(__file__),plan_sha256=sha(PLAN),
                scope='NativeCUDA compiler import paths only; no kernel/GPU/training success claim')
    save(WORK/'cpu_preflight.json',data)
    require(state['protected']==protected(),'CPU imports changed protected environment')
    print(json.dumps({'passed':True,'cuda_backend_path':data['cuda_backend_path']}))


if __name__=='__main__':main()
