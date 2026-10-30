"""Install the parent's already successful cu12 plan into NEW native-env-v2 only.

Uses the completed CPU preparation from setup_nemo_gpu_backend_env_v2 unchanged.
No resolver repeat, parent-package transactions, CUDA initialization or training.
"""
import argparse
import json
from pathlib import Path
import setup_nemo_gpu_backend_env_v2 as layer

SETUP_SHA='e68e4964aa65566404ae5da3a3c192f8b06e9b71c7819206e3abb1a036a0ece6'
PLAN=layer.ROOT/'work/nemo_native_setup_v1/cuda_backend_dependency_plan_v1.json'
PLAN_SHA='607ee0e7ea55bcdbca42b74842dae5430689fdf90582015c33d086c34d203e76'
require,sha=layer.require,layer.sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=('install','cpu-preflight'))
    args=parser.parse_args()
    require(sha(layer.__file__)==SETUP_SHA and sha(PLAN)==PLAN_SHA,'Frozen layer/authoritative resolver changed')
    state=layer.prepared()
    layer.PLAN=PLAN
    packages,requirements=layer.plan()
    installation=layer.WORK/'installation_root_plan_v3.json'
    if args.stage=='install':
        require(not installation.exists() and not (layer.SITE/'numba_cuda').exists(),'Preserve existing compiler install')
        requirement_path=layer.WORK/'requirements-resolved-root-v3.txt'
        require(not requirement_path.exists(),'Preserve exact requirements')
        requirement_path.write_text(requirements,encoding='utf-8')
        command=[str(layer.PYTHON),'-m','pip','--isolated','install','--target',str(layer.SITE),
                 '--no-deps','--ignore-installed','--require-hashes','--no-compile','--no-warn-script-location',
                 '--cache-dir',str(layer.WORK/'pip-cache'),'-r',str(requirement_path)]
        layer.run(command,layer.WORK/'install_root_plan_v3.log')
        require(state['protected']==layer.protected(),'Parent/native-v1 protected files changed')
        layer.save(installation,{'schema':1,'status':'complete','helper_sha256':sha(__file__),
            'preparation_helper_sha256':SETUP_SHA,'parent_plan_path':str(PLAN),'parent_plan_sha256':PLAN_SHA,
            'command':command,'packages':packages,'requirements_sha256':sha(requirement_path),
            'pip_target':str(layer.SITE),'parent_environments_mutated':False,'CUDA_executed':False,
            'source_and_protected_evidence_before':state['protected'],'source_and_protected_evidence_after':layer.protected()})
        print(json.dumps({'installed':True,'environment':str(layer.ENV),'next_stage':'cpu-preflight'}));return
    require(installation.exists(),'Install root plan first')
    receipt=layer.read(installation)
    require(receipt['helper_sha256']==sha(__file__) and receipt['parent_plan_sha256']==PLAN_SHA,'Installation provenance')
    source="""import json,pathlib,importlib.metadata as m,torch,numba,numba.cuda,llvmlite,nemo
from nemo.collections.asr.parts.numba.rnnt_loss import TDTLossNumba
data={'torch_version':torch.__version__,'torch_path':str(pathlib.Path(torch.__file__).resolve()),'numba_version':numba.__version__,'numba_path':str(pathlib.Path(numba.__file__).resolve()),'cuda_backend_path':str(pathlib.Path(numba.cuda.__file__).resolve()),'llvmlite_version':llvmlite.__version__,'nemo_path':str(pathlib.Path(nemo.__file__).resolve()),'numba_cuda_version':m.version('numba-cuda'),'cuda_python_version':m.version('cuda-python'),'cuda_initialized':torch.cuda.is_initialized()}
print(json.dumps(data))
"""
    text=layer.run([str(layer.PYTHON),'-X','utf8','-c',source])
    data=json.loads(text.strip().splitlines()[-1])
    require(data['torch_path']==state['protected']['torch_module_path'] and data['torch_version']=='2.7.1+cu128','Shared Torch replaced')
    require(data['numba_version']=='0.68.0' and data['llvmlite_version']=='0.50.0','Shared core compiler replaced')
    require(Path(data['cuda_backend_path']).is_relative_to(layer.SITE),'Deprecated inherited CUDA compiler remains active')
    require(Path(data['nemo_path']).is_relative_to(layer.prior.SITE) and data['cuda_initialized'] is False,'Read-onlyNeMo/CPU scope')
    require(data['numba_cuda_version']==packages['numba-cuda']['version'] and data['cuda_python_version']==packages['cuda-python']['version'],'Resolved compiler/bindings version differs')
    require(state['protected']==layer.protected(),'Protected files changed during CPU preflight')
    data.update(passed=True,helper_sha256=sha(__file__),preparation_helper_sha256=SETUP_SHA,
                parent_plan_sha256=PLAN_SHA,installation_sha256=sha(installation),
                scope='CPU compiler import paths/versions only; actual CUDA loss gradient still needs root solo proof')
    layer.save(layer.WORK/'cpu_preflight_root_plan_v3.json',data)
    print(json.dumps({'passed':True,'cuda_backend_path':data['cuda_backend_path']}))


if __name__=='__main__':main()
