"""Paired CPU-only import cost and exact actual-source carried-replay control."""
import argparse,copy,gc,hashlib,json,os,sys,time
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';T=P/'work/training-env/Lib/site-packages'
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')
sys.path.insert(0,str(T))
import numpy as np
sys.path.remove(str(T))
import torch
assert torch.__version__=='2.7.1+cpu' and torch.version.cuda is None
sys.path.extend([str(T),str(S)])
from bpe_broad_audio_v1 import require,record
from bpe_broad_production_state_v1 import CarriedReplay,canonical
from atomic_io_retry_v1 import write_json
from resource_watchdog_v2 import sample_memory

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['full','narrow'],required=True);p.add_argument('--folder',required=True);a=p.parse_args()
    folder=Path(a.folder);require(folder.is_relative_to(A),'E-only finite receipt');folder.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    before=sample_memory(os.getpid(),'after_CPU_Torch_before_arm_import')['worker_private_memory'];began=time.monotonic()
    if a.arm=='full':
        from batched_lora_tr_metadata_v1 import collate_training_batch
        from train_lora_batched_tsc_expanded_TRAIN_v1 import pt_focused_batch_objective
        from train_lora import checkpoint_encoder_blocks
        from train_lora_batched_tsc_bpe_full_decoder_v1 import expand_candidate
        from train_lora_batched_data_v11 import BucketCycler
    else:
        from asr18_narrow_objective_v1 import collate_training_batch,pt_focused_batch_objective
        from asr18_narrow_training_primitives_v1 import checkpoint_encoder_blocks,expand_candidate,BucketCycler
    elapsed=time.monotonic()-began;after=sample_memory(os.getpid(),'after_arm_import_before_source_archive')['worker_private_memory']
    write_json(folder/'import.json',{'before':before,'after':after,'seconds':elapsed})
    proof=json.loads((A/'source_projection_v4/report.json').read_text());binding=proof['projections']['bpe']['projected']
    require(record(binding['path'])==binding,'Pinned original-source projection changed')
    saved=torch.load(binding['path'],map_location='cpu',weights_only=True,mmap=True)
    sampler=copy.deepcopy(saved['sampler']['continued_parent_sampler']);del saved;gc.collect()
    snapshot={'source_step':64250,'source_identity':proof['projections']['bpe']['parent'],**{key:sampler[key] for key in ('replay','replay_language_counts','turkish_rng','replay_rng','Portuguese_rng')}}
    pool_path=P/'outputs/parakeet_turkish/runs/adaptive_20261014_TR_WorldSpeech_expanded_01/training_pool.json'
    require(record(pool_path)['sha256']=='00b839c717762aba74326e9867969d85b7ffaaabd7ead7ddbe732fe6be8c6cf4','Exact original1914 replay pool')
    rows=json.loads(pool_path.read_text())['replay'];require(len(rows)==1914,'Exact full original replay pool')
    replay=CarriedReplay(rows,snapshot,BucketCycler);start=copy.deepcopy(replay.state_dict());sequence=[]
    for index in range(1000):sequence.append([(row['source_id'],row['language']) for row in replay.next()])
    first_hash=hashlib.sha256(canonical(sequence)).hexdigest();end_hash=hashlib.sha256(canonical(replay.state_dict())).hexdigest()
    replay.load_state_dict(start);again=[]
    for index in range(1000):again.append([(row['source_id'],row['language']) for row in replay.next()])
    require(sequence==again and end_hash==hashlib.sha256(canonical(replay.state_dict())).hexdigest(),'Fresh restore actual1000 replay choices/queues/RNG exactly repeated')
    report={'passed':True,'kind':'ASR18_paired_CPU_import_and_carried_replay_control_v1','arm':a.arm,'torch':torch.__version__,
        'before_import':before,'after_import':after,'import_seconds':elapsed,'import_private_increment_bytes':after['private_bytes']-before['private_bytes'],
        'actual_original_replay_pool':record(pool_path),'actual_source_projection':binding,'replay_choices':1000,'replay_rows':2000,
        'replay_sequence_sha256':first_hash,'final_carried_replay_state_sha256':end_hash,'fresh_restore_exact':True,
        'librosa_loaded':('librosa' in sys.modules),'scipy_signal_loaded':('scipy.signal' in sys.modules),
        'unused_CLI_modules_loaded':sorted(n for n in sys.modules if n.startswith(('runtime','train_lora_batched_tsc','benchmark','train_lora'))),
        'source':record(__file__),'derivation':record(A/'narrow_training_primitives_derivation_v1.json'),'ASR_weights_or_model_loaded':False,'native_optimizer_updates':0}
    write_json(folder/'report.json',report)

if __name__=='__main__':main()
