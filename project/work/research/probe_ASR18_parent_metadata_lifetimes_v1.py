"""Finite metadata-only retained-frame/released-helper comparison, no owner run."""
import argparse,gc,hashlib,importlib,json,os,sys,time
from pathlib import Path
P=Path(__file__).resolve().parents[2];sys.path[:0]=[str(P/'outputs/parakeet_turkish'),str(P/'work/research')]
ROOT=Path(r'E:\gpu_training_scratchpad');N=ROOT/'analysis/BPE9223_training_launch_v1'
RUN=ROOT/'runs/BPE9223_weak2_clean1_20261018_02'
PLAN=RUN/'checkpoints/transient/p000000002/coordinator_plan_copy.json'
BRIDGE=RUN/'checkpoints/transient/p000000002/bridge.json'
RESPONSE=N/'full_training_genesis_actual_02/native_000001/response.json'
CAP=512_000_000


def require(value,message):
    if not value:raise ValueError(message)


def binding(path):
    path=Path(path).resolve();require(path.is_relative_to(ROOT) or path.is_relative_to(P),'Permitted E/source scope')
    before=path.stat();require(before.st_size<=8*1024**2,'Small metadata/source file only')
    h=hashlib.sha256()
    with path.open('rb') as stream:
        while block:=stream.read(65536):h.update(block)
    after=path.stat();require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Immutable metadata input')
    return {'path':str(path),'bytes':before.st_size,'sha256':h.hexdigest()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);a=p.parse_args()
    began=time.monotonic();folder=Path(a.folder).resolve()
    require(folder.is_relative_to(N) and not folder.exists(),'Fresh E fixture output only')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1' and all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')),'CPU only, four numerical caps1')
    parent=importlib.import_module('run_ASR18_BPE9223_full_run_owner_v2')
    controller=importlib.import_module('asr18_BPE9223_full_run_controller_v2')
    require(not any(k=='torch' or k.startswith('torch.') for k in sys.modules),'No Torch/model imports')
    from resource_watchdog_v2 import sample_memory
    from fastmss_owned_process_closure_v2 import identity
    from bpe_broad_production_state_v1 import digest
    parent.storage();sources={'parent2':binding(parent.__file__),'controller2':binding(controller.__file__)}
    require(sources['parent2']['sha256']=='78d7d97a4719a7d95bd54ea6060f5339f7643daa451eb7c222b6529bdc6f82d8','Exact original parent2 import')
    completion=json.loads(RESPONSE.read_text(encoding='utf-8'))['window_completion'];pins={'plan':binding(PLAN),'bridge':binding(BRIDGE),'response':binding(RESPONSE),'completion':binding(completion['path'])}
    require(pins['completion']==completion and sum(v['bytes'] for v in pins.values())<=16*1024**2,'Exact bounded prior completion and four metadata inputs')
    snapshots=[]
    def snapshot(label):
        limits=parent.private_lifetime_limit(CAP);sample=sample_memory(os.getpid(),'parent_metadata_lifetime_fixture')
        require(sample['paired_sampling_within_one_second'] and sample['system_memory']['available'] and sample['worker_private_memory']['available'],'Available fresh paired telemetry')
        system=sample['system_memory'];require(system['commit_fraction']<.90 and system['commit_available_bytes']>2_000_000_000,'Same system90/2GB resource admission')
        require(time.monotonic()-began<180,'Finite metadata fixture')
        value={'label':label,**limits,'current_within_64MB_CPU_gate':limits['current_private_bytes']<=64_000_000,'sample':sample}
        snapshots.append(value);return value
    def parse():
        return {key:json.loads(Path(pin['path']).read_text(encoding='utf-8')) for key,pin in pins.items()}
    def inspect_trees(trees):
        require(trees['bridge']['coordinator_plan']['sha256']==pins['plan']['sha256'] and trees['bridge']['coordinator_plan']['bytes']==pins['plan']['bytes'],'Same copied coordinator plan/bridge')
        require(trees['response']['window_completion']==pins['completion'] and trees['plan']['initial_committed_updates']==20
            and len(trees['plan']['updates'])==len(trees['bridge']['updates'])==20,'Actual p1 complete and p2 planned20 metadata only')
        return {key:digest(value) for key,value in trees.items()}
    gc.collect();snapshot('baseline_after_same_parent_controller_imports')
    held=parse();digests=inspect_trees(held);snapshot('actual_four_metadata_trees_held_in_frame')
    del held;collected=gc.collect();snapshot('after_explicit_delete_and_gc')
    def helper():
        local=parse();proof=inspect_trees(local);snapshot('same_four_trees_held_inside_helper')
        return proof
    helper_digests=helper();helper_collected=gc.collect();snapshot('after_helper_return_and_gc')
    require(helper_digests==digests and all(binding(v['path'])==v for v in pins.values()) and all(binding(v['path'])==v for v in sources.values()),'All file identities and parsed value digests preserved')
    require(not any(k=='torch' or k.startswith('torch.') for k in sys.modules),'No Torch/model import throughout')
    report={'kind':'ASR18_readonly_parent_metadata_lifetime_fixture_v1','passed':True,'producer':identity(),'source':binding(__file__),'sources':sources,
        'input_metadata':pins,'parsed_value_digests':digests,'helper_value_digests':helper_digests,'snapshots':snapshots,'worker_current_and_OS_lifetime_cap_bytes':CAP,
        'current_64MB_gate_unchanged':True,'explicit_delete_collected_objects':collected,'helper_return_collected_objects':helper_collected,
        'measured_release_bytes':{'delete_gc':snapshots[1]['current_private_bytes']-snapshots[2]['current_private_bytes'],'helper_return_gc':snapshots[3]['current_private_bytes']-snapshots[4]['current_private_bytes']},
        'original_run_cache_store_writes':0,'GPU_audio_model_jobs':0,'production_sources_edited':False,'full_owner_rollover_or_CPU64_fit_proven':False,
        'limitations':'Four saved metadata trees and same imports only; excludes full owner/context/store/native lifetime and allocator history. Current private release is measured, never inferred from gc or RSS.',
        'wall_seconds':time.monotonic()-began}
    folder.mkdir();path=folder/'report.json'
    with path.open('x',encoding='utf-8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
    print(json.dumps(binding(path)),flush=True)


if __name__=='__main__':main()
