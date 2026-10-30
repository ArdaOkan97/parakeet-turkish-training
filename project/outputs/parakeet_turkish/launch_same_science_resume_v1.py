"""Fresh native preload for SAME-run exact HF resume with immutable original ready.

The original ready file remains the scientific identity token. It is not fresh
PID/runtime evidence. A separate NEW proof/completion records this process's
actual preload and current frozen memory15/trainer11 chain. No source is patched.
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
PINS={'preload_nemo_linker_v1.py':'f8787b0f3346c9f526903ff0c06de4c266f940ca3b0f41c55907cc795e7cc2ba',
    'native_tdt_checkpointed_v3.py':'839194dddf83c3b08bf608d01cb540b2f160feee426a91373d14cfc426ab7494',
    'train_lora_batched_data_v11.py':'4b4c9e3008205c61bb5a881972187b5b132881dc7a1d9249583d1adcd52a7d20',
    'launch_with_memory_guard_v15.py':'00213ae0a1af7ec5a24c9a2912bbab43a173c40a48448885cc971e7c4d77e3aa',
    'launch_with_cache_reclaim_v12.py':'9febf980a1eb51dd35e5d7768afe73664774ca0445763f6cf1e2caccc5db802b',
    'launch_with_io_retry_v9.py':'8d49db792cafa88341a64d893eeae4f09233edde36cdca3b53954f6da102cd3e'}


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
    original=scoped(args.original_native_provenance)
    require(sha(original)==args.original_native_provenance_sha256,'Original first-start identity changed')
    old=json.loads(original.read_text(encoding='utf-8'))
    values=list(args.memory_arguments)
    if values[:1]==['--']:values=values[1:]
    require(values.count('--')==1,'One memory-wrapper/trainer separator required')
    memory,worker=values[:values.index('--')],values[values.index('--')+1:]
    require(option(worker,'--run-id')==old.get('run_id') and old.get('kind')=='native_tdt_training_backend_v3'
        and old.get('status')=='ready' and old.get('tdt_implementation')=='nemo'
        and old.get('trainer_sha256')==PINS['train_lora_batched_data_v11.py']
        and old.get('adapter_sha256')==PINS['native_tdt_checkpointed_v3.py'],
        'Preserve SAME run/original trainer/adapter startup identity')
    require(scoped(option(worker,'--native-backend-provenance'))==original
        and option(worker,'--tdt-implementation')=='nemo' and '--resume-weights' not in worker
        and '--resume-runtime-migration' not in worker,'Only SAME-run exact resume with original ready and no migration allowed')
    scoped(option(worker,'--resume-exact'))
    require(option(memory,'--cap-gb')=='8' and option(memory,'--threshold-gb')=='4'
        and old.get('declared_allocator_policy')=={'cap_gb':8,'end_cache_gb':4,'native_inactive_threshold_bytes':750000000},
        'Preserve original cap8/end4/native750MB policy')
    fresh=scoped(args.fresh_runtime_provenance);completion=fresh.with_suffix('.completion.json')
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
    evidence={'schema':1,'kind':'HF_same_science_native_preload_resume_v1','status':'starting',
        'run_id':old['run_id'],'pid':os.getpid(),'ppid':os.getppid(),'started_at_utc':datetime.now(timezone.utc).isoformat(),
        'source_sha256':PINS,'launcher_sha256':sha(__file__),'worker_arguments_sha256':argument_sha(worker),
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
        sys.argv=[str(HERE/'launch_with_memory_guard_v15.py'),*values]
        runpy.run_path(str(HERE/'launch_with_memory_guard_v15.py'),run_name='__main__')
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
            and all(sha(HERE/n)==h for n,h in PINS.items()))
        write_new(completion,{'schema':1,'kind':'HF_same_science_native_preload_resume_completion_v1','status':outcome,
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
