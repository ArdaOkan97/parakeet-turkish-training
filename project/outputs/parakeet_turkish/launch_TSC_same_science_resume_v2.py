"""Fresh native preload for SAME-run exact TSC resume with immutable original ready.

The original ready file remains the scientific identity token. It is not fresh
PID/runtime evidence. A separate NEW proof/completion records this process's
actual preload and current frozen isolated TSC chain. No source is patched.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent
PINS = {'train_lora_batched_data_v11.py': '4b4c9e3008205c61bb5a881972187b5b132881dc7a1d9249583d1adcd52a7d20', 'native_tdt_checkpointed_v3.py': '839194dddf83c3b08bf608d01cb540b2f160feee426a91373d14cfc426ab7494', 'cuda_cache_reclaim_v2.py': '869b9ffaac70c9b7b6e7590ba1d330363100d6b2ddb9f8f7af9b2b9865c1c7f9', 'atomic_io_retry_v1.py': '2d3fe379d51021fd15679e562c71286f7f9501180d28dde77893e8f72c80a1b0', 'resource_watchdog_v2.py': '64d1393e1e45cf65dac14f78ac955331c91b4a3f0eafbf7535196e2b61778641', 'paired_windows_commit_telemetry_v1.py': '2ae9f2cdfc388a294914e000c60b62fe09cddac147700c5762851d626a95a683', 'public_turkish_data_policy_v3.py': '08eee3896bf8571c9e97dae1be688ecef7766816c019741dd7631d70f640ce41', 'preload_nemo_linker_v1.py': 'f8787b0f3346c9f526903ff0c06de4c266f940ca3b0f41c55907cc795e7cc2ba', 'batched_lora_tr_metadata_v1.py': 'b78236a851a9739c30423bb00863973d571cf9fee948be4473bcd1cb34f05075', 'tr_frame_metadata_v1.py': 'e74bc92b4c089ca92ca408397267eecfb665c51b43874909f6367def64d760af', 'training_input_audit_reuse_v1.py': 'f8cd68791b54e061e4012cfe2f715c19cb390c14229a1a76ee18fb5d2e60d07f', 'launch_nemo_linker_v4.py': 'bdd7078dff81adc961c17588ed7fd84b152da71c619c2ff114994fa68d308af0', 'tsc_supplemental_data_policy_v1.py': 'a10b1cbaa94b4e6810f6a5fd749de4b7743bbaf68504c63bbcd5070e8cdd7cdb', 'prepare_TSC_TRAIN_v1.py': '08785c2446229ae6716feee5e9103d77c468faf0eaf68a602e0477bc2d36063e', 'cuda_cache_caller_tsc_v2.py': 'fabf69ed5a3b80f5e68dcd34c096914bd8cf5a3070e6107b010eb5978f8243e9', 'launch_with_io_retry_tsc_v2.py': 'adf004d39cab233158d8da96c43bc70105d2fa3fcd0d551077ed43c0954b5238', 'launch_with_cache_reclaim_tsc_v2.py': '42c1757ae0563c56a9b47f6a3cdae15e2b90313161c978e20f3714c4e82445ec', 'launch_with_memory_guard_tsc_v2.py': '9f15b82108fe88b957cf2f35466a37eaa150ad1f3a457aad204c8dbaf6833eab', 'launch_native_training_tsc_v2.py': '6cbe40ee32fb4f73125d61ad6b6c11550178ebac9d9028470d1e05891613dafc', 'train_lora_batched_tsc_v2.py': 'e3df50a3b308a4db2cc39ca83c1a85772709c305576b03d812b6112a70b52484', 'tsc_training_inputs_v2.py': 'e1175f6a5e8d4cb8371052afe16b3a539d598e87ec0e1ceb4d2da0314274fbcf', 'tsc_supplemental_data_policy_v2.py': 'f4cdeb53ef0358fd0a6cdb819e6455127d718e8a263127419dc538294057a520', 'prepare_TSC_quarantine_v2.py': '545840caa8d99fab158621a0b86c6559cd96005e07ddec30776f3f8f81d105fc'}
RUNTIME_MEMORY_NAME='launch_with_memory_guard_TSC_resume_v2.py'
RUNTIME_MEMORY_SHA = '67a99f29575fbc1d765167b754463d34738879d060d41b1b5486ae3acaa37db5'


def require(ok,message):
    if not ok:raise ValueError(message)
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def scoped(path):
    value=Path(path).resolve();require(value.is_relative_to(ROOT),'Workspace-only evidence required');return value
def option(argv,name):
    require(argv.count(name)==1 and argv.index(name)+1<len(argv),'Exactly one '+name+' required')
    return argv[argv.index(name)+1]
def argument_sha(argv):return hashlib.sha256(json.dumps(argv,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def write_new(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,ensure_ascii=False)


def validate(args):
    for name,digest in PINS.items():require(sha(HERE/name)==digest,'Frozen same-science source changed: '+name)
    require(sha(HERE/RUNTIME_MEMORY_NAME)==RUNTIME_MEMORY_SHA,'Reviewed exact-only runtime memory admission changed')
    original=scoped(args.original_native_provenance)
    require(sha(original)==args.original_native_provenance_sha256,'Original first-start identity changed')
    old=json.loads(original.read_text(encoding='utf-8'))
    values=list(args.memory_arguments)
    if values[:1]==['--']:values=values[1:]
    require(values.count('--')==1,'One memory-wrapper/trainer separator required')
    memory,worker=values[:values.index('--')],values[values.index('--')+1:]
    require(option(worker,'--run-id')==old.get('run_id') and old.get('kind')=='native_tdt_training_backend_v3'
        and old.get('status')=='ready' and old.get('tdt_implementation')=='nemo'
        and old.get('trainer_sha256')==PINS['train_lora_batched_tsc_v2.py']
        and old.get('adapter_sha256')==PINS['native_tdt_checkpointed_v3.py'],
        'Preserve SAME run/original trainer/adapter startup identity')
    require(scoped(option(worker,'--native-backend-provenance'))==original
        and option(worker,'--tdt-implementation')=='nemo' and '--resume-weights' not in worker
        and '--resume-runtime-migration' not in worker,'Only SAME-run exact resume with original ready and no migration allowed')
    scoped(option(worker,'--resume-exact'))
    scoped(option(worker,'--source-closure'))
    for flag in ('--tsc-input-proof','--tsc-training-approval','--source-closure','--tr-manifest','--replay-manifest','--tr-frame-metadata','--tr-frame-metadata-proof','--data-v3-proof'):
        path=scoped(option(worker,flag));key=flag.removeprefix('--').replace('-','_')
        require(old.get('TSC_explicit_asset_records',{}).get(key)=={'path':str(path),'sha256':sha(path)},'Original approved TSC asset differs: '+flag)
    require(old.get('launcher_sha256')==PINS['launch_native_training_tsc_v2.py']
        and old.get('memory_launcher_sha256')==PINS['launch_with_memory_guard_tsc_v2.py'],
        'Original first TSC backend-ready source identity differs')
    require(option(memory,'--cap-gb')=='8' and option(memory,'--threshold-gb')=='4'
        and old.get('declared_allocator_policy')=={'cap_gb':8,'end_cache_gb':4,'native_inactive_threshold_bytes':750000000},
        'Preserve original cap8/end4/native750MB policy')
    proof_paths=[scoped(option(memory,flag)) for flag in
        ('--provenance','--cache-provenance','--memory-provenance','--worker-identity')]
    require(len(set(proof_paths))==4 and all(not p.exists() for p in proof_paths),
        'Four distinct NEW process evidence paths required')
    fresh=scoped(args.fresh_runtime_provenance);completion=fresh.with_suffix('.completion.json')
    protected={original,scoped(option(worker,'--source-closure')),scoped(option(worker,'--resume-exact'))}
    require(not protected.intersection([fresh,completion,*proof_paths])
        and fresh not in proof_paths and completion not in proof_paths,
        'Fresh evidence must not alias immutable identity, closure, checkpoint or other evidence')
    require(not fresh.exists() and not completion.exists() and fresh!=original and completion!=original,
        'NEW separate actual runtime proof/completion required')
    return values,worker,original,old,fresh,completion


def main(args):
    values,worker,original,old,fresh,completion=validate(args)
    if args.validate_only:
        print(json.dumps({'passed':True,'run_id':old['run_id'],'Torch_imported':'torch' in sys.modules,
            'CUDA_initialized':False,'actual_fresh_preload_pending':True,'original_ready_is_not_fresh_PID_evidence':True}))
        return
    require(Path(sys.executable).resolve()==ROOT/'work/nemo-native-env-v2/Scripts/python.exe','Use isolated native-env-v2')
    require('torch' not in sys.modules,'Fresh preload must precede Torch import')
    evidence={'schema':1,'kind':'TSC_same_science_native_preload_resume_v2','status':'starting',
        'run_id':old['run_id'],'pid':os.getpid(),'ppid':os.getppid(),'started_at_utc':datetime.now(timezone.utc).isoformat(),
        'source_sha256':PINS,'runtime_memory_launcher':{'path':str(HERE/RUNTIME_MEMORY_NAME),'sha256':RUNTIME_MEMORY_SHA},'launcher_sha256':sha(__file__),'worker_arguments_sha256':argument_sha(worker),
        'original_first_start_ready':{'path':str(original),'sha256':sha(original),'original_first_start_pid':old.get('pid')},
        'original_ready_scope':'Immutable scientific identity token only; its old PID is not this process',
        'scientific_config_identity_changed':False,'fresh_runtime_proof_separate_from_original_ready':True}
    outcome='failed';error=None;previous_argv=sys.argv;wrote=False
    try:
        from preload_nemo_linker_v1 import preload
        result=preload() # Actual NEW process CUDA12.9 linkage before memory/Torch.
        require(result.get('kind')=='coherent_native_cuda12_9_preload_v1'
            and result.get('module_sha256')==PINS['preload_nemo_linker_v1.py']
            and result.get('actual_binding_linker',{}).get('version')==[12,9]
            and result.get('cuda_initialized') is False,'Actual fresh coherent preload differs')
        evidence.update(status='ready',preload=result,ready_at_utc=datetime.now(timezone.utc).isoformat())
        write_new(fresh,evidence);wrote=True
        sys.argv=[str(HERE/RUNTIME_MEMORY_NAME),*values]
        runpy.run_path(str(HERE/RUNTIME_MEMORY_NAME),run_name='__main__')
        outcome='complete'
    except BaseException as exception:
        if isinstance(exception,SystemExit) and exception.code in (None,0):outcome='complete'
        else:error=repr(exception)
        raise
    finally:
        sys.argv=previous_argv
        if not wrote:
            evidence.update(status='failed',error=error);write_new(fresh,evidence)
        unchanged=(sha(__file__)==evidence['launcher_sha256'] and sha(original)==args.original_native_provenance_sha256
            and all(sha(HERE/n)==h for n,h in PINS.items()) and sha(HERE/RUNTIME_MEMORY_NAME)==RUNTIME_MEMORY_SHA)
        write_new(completion,{'schema':1,'kind':'TSC_same_science_native_preload_resume_completion_v2','status':outcome,
            'run_id':old['run_id'],'pid':os.getpid(),'ppid':os.getppid(),'fresh_runtime_proof_path':str(fresh),
            'fresh_runtime_proof_sha256':sha(fresh),'launcher_sha256_after':sha(__file__),'sources_unchanged':unchanged,
            'original_first_start_ready_sha256_after':sha(original),'error':error,'finished_at_utc':datetime.now(timezone.utc).isoformat()})
        require(unchanged,'Original identity or frozen runtime source changed')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original-native-provenance',type=Path,required=True)
    p.add_argument('--original-native-provenance-sha256',required=True)
    p.add_argument('--fresh-runtime-provenance',type=Path,required=True)
    p.add_argument('--validate-only',action='store_true')
    p.add_argument('memory_arguments',nargs=argparse.REMAINDER)
    main(p.parse_args())
