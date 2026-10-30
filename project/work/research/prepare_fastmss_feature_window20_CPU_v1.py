"""Serial bounded fixed14+2 replay feature preparation BEFORE GPU residency.

No model, optimizer, native update or authoritative draw object is created.
Frozen decoder/PCM cache policy is reused; the feature window stays provisional.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish'
ROOT=Path(r'E:\gpu_training_scratchpad');AREA=ROOT/'analysis/ASR18_memory_diagnosis_v1'
CPU_PYTHON=ROOT/'scratch/BPE_broad_CPU_preflight_env_v2/Scripts/python.exe'
sys.path.insert(0,str(S))
from atomic_io_retry_v1 import write_json
from bpe_broad_production_v5 import decode_prepared,cleanup_prepared,CommitJournal
from fastmss_feature_window_v1 import (record,read_bound,load_plan,create_window,owner_folder,verify_feature_closure,
    observe_feature_child,validate_feature_entry,finalize_window,FRONTEND_ARGS,scoped)
from fastmss_ordered_draw_binding_v1 import AugmentationDrawContext
from fastmss_train_augmentation_v1 import require
from resource_watchdog_v2 import sample_memory

PLAN=AREA/'feature_window20_plan_01/plan_v2.json'
REPLAY=AREA/'window_replay_PCM_actual_01/report_plan_v2.json'
EXPECTED_PLAN='336617348e3296f270e1af48e7613443f016a75e77859cb6a3b203cdf829db9f'
EXPECTED_REPLAY='085fe85e85b3787e9d86650881c3e8a9c21094fdb49de1bb974e41592fd2e126'


def main(output):
    folder=scoped(output);folder.mkdir(parents=True,exist_ok=False)
    require(os.environ['CUDA_VISIBLE_DEVICES']=='-1','No GPU visibility for feature coordinator')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        require(os.environ[key]=='1','Four fixed CPU library caps required')
    plan_binding=record(PLAN);replay_binding=record(REPLAY)
    require(plan_binding['sha256']==EXPECTED_PLAN and replay_binding['sha256']==EXPECTED_REPLAY,'Pinned planning/replay references changed')
    plan=load_plan(plan_binding,lambda:None);replay=read_bound(replay_binding)
    require(len(plan['updates'])==20 and plan['per_update_TR']==14 and plan['per_update_carried_replay']==2 and
        replay['passed'] and replay['plan']==plan_binding,'Only exact bounded20 fixed14+2 source plan')
    config=json.loads((S/'bpe_broad_production_v5.disabled.json').read_text())
    require(config['production_enabled'] is False,'No production enablement')
    journal=CommitJournal(folder/'provisional_PREPARATION_ONLY_EMPTY.jsonl')
    started=time.monotonic();window_owner=None;closures=[];updates=[];child=None;child_folder=None
    source_paths=[Path(__file__),S/'fastmss_feature_window_v1.py',P/'work/research/produce_fastmss_feature_update_CPU_v1.py',
        P/'work/research/ASR18_bounded_CPU_phase_v3.py',P/'work/research/decode_ASR18_broad_batch_v6.py',
        S/'bpe_broad_audio_FLAC_records_v1.py',S/'bpe_broad_production_v5.py',S/'fastmss_cpu_prepared_transport_v1.py',
        S/'fastmss_ordered_draw_binding_v1.py',S/'fastmss_train_augmentation_v1.py',S/'fastmss_broad_optin_v1.json']
    pinned_sources=[record(path) for path in source_paths]
    write_json(folder/'launch_binding.json',{'PID':os.getpid(),'PPID':os.getppid(),'source_bindings':pinned_sources,
        'plan':plan_binding,'replay_reference':replay_binding,'max_updates':20,'actual_optimizer_updates':0,
        'authoritative_frontiers_advanced':False,'GPU_visible':False,'feature_window_capacity_bytes':512*1024**2,
        'PCM_frozen_capacity_bytes':128*1024**2,'PCM_frozen_max_jobs':2})
    def guard():
        require(not (folder/'stop.json').exists(),'Owned cooperative CPU preparation stop')
        require(time.monotonic()-started<900,'Bounded feature preparation wall budget')
        require(shutil.disk_usage(ROOT).free>=100_000_000_000,'E100GB reserve')
        lease=json.loads((P/'work/gpu_training_owner.json').read_text())
        require(not lease['actual_GPU_running'],'Prepare features before GPU residency; no overlap or GPU-owner edit')
        require(all(record(v['path'])==v for v in pinned_sources),'Admitted coordinator/decoder/transport/frontend source changed')
    def admit(name,budget):
        guard();sample=sample_memory(os.getpid(),'CPU_window_stage_admission');system=sample['system_memory']
        require(system['available'] and sample['worker_private_memory']['available'] and sample['paired_sampling_within_one_second'],'Fresh paired admission required')
        require(system['commit_used_bytes']+budget<.90*system['commit_limit_bytes'] and
                system['commit_available_bytes']-budget>2_000_000_000,'Fresh bounded CPU workload admission denied; no unchanged retry')
        write_json(folder/'stage_admission.json',{'name':name,'budget_bytes':budget,'sample':sample,'passed':True})
    base=Path((P/'work/base_model_path.txt').read_text().strip())
    frontend={'source':record(P/'work/training-env/Lib/site-packages/transformers/models/parakeet/feature_extraction_parakeet.py'),
        'base_path':str(base),'processor_config':record(base/'processor_config.json'),'arguments':FRONTEND_ARGS,
        'Torch':'2.7.1+cpu','torch_threads':4,
        'CPU_GPU_byte_equivalence_proof':record(AREA/'original_parent_frontend_compare_01/frontend_byte_equivalence.json')}
    try:
        for index,update in enumerate(plan['updates']):
            guard();part=folder/f'update_{index+1:02}';part.mkdir()
            write_json(folder/'status.json',{'status':'decoding_PCM','PID':os.getpid(),'completed_feature_updates':len(closures),
                'current_update':index+1,'actual_optimizer_updates':0,'authoritative_frontiers_advanced':False})
            contexts=tuple(AugmentationDrawContext(**value) for value in update['contexts'])
            prepared=decode_prepared(update['TR_draws'],contexts[:14],part/'PCM',config,batch_id=update['batch_id'],
                admit=admit,stop_check=guard,diagnostic_only=True)
            transport=prepared['augmentation_transport'];pcm_manifest=read_bound(transport['manifest'],guard)
            if window_owner is None:
                window_owner=create_window(plan_binding,window_id=folder.name,augmentation_identity=pcm_manifest['identity'],frontend=frontend,guard=guard)
                write_json(folder/'window_owner.json',window_owner)
            request={'kind':'FastMSS_CPU_feature_update_request_v1','owner':window_owner,'update_index':index,
                'prior_closures':closures,'decoder_report':prepared['decoder_report'],
                'PCM_actual_closure':prepared['CPU_actual_closure'],'replay_reference_report':replay_binding}
            request_path=part/'feature_request.json';write_json(request_path,request)
            admit('CPU_original_frontend_update_'+str(index+1),700_000_000)
            child_folder=part/'frontend_guard'
            command=[str(CPU_PYTHON),'-X','utf8','-B',str(P/'work/research/ASR18_bounded_CPU_phase_v3.py'),
                '--folder',str(child_folder),'--seconds','120','--script',str(P/'work/research/produce_fastmss_feature_update_CPU_v1.py'),
                '--','--request',str(request_path),'--output',str(part/'feature_result.json')]
            with (part/'frontend.log').open('x',encoding='utf-8') as log:
                child=subprocess.Popen(command,cwd=P,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW)
                write_json(folder/'status.json',{'status':'CPU_frontend','PID':os.getpid(),'owned_handle_PID':child.pid,
                    'completed_feature_updates':len(closures),'current_update':index+1,'actual_optimizer_updates':0})
                began=time.monotonic()
                while child.poll() is None:
                    guard();require(time.monotonic()-began<130,'Owned frontend CPU deadline');time.sleep(.2)
                code=child.wait()
            proof=observe_feature_child(child,owner_binding=window_owner,index=index,proof_path=part/'actual_feature_closure.json',guard=guard)
            require(code==0,'Actual frontend child failed; preserve partial window and no retry')
            actual=verify_feature_closure(proof,window_owner,success=True,guard=guard)
            feature_update=read_bound(actual['update_record'],guard)
            _,feature_folder=owner_folder(window_owner,guard)
            for row in feature_update['rows']:
                validate_feature_entry(row,feature_folder,guard)
            # Evidence and all feature arrays are already fsynced and verified;
            # this discards reproducible provisional PCM, never a training draw.
            cleanup_prepared(prepared,part/'PCM',batch_id=update['batch_id'],journal=journal,discard=True,stop_check=guard)
            closures.append(proof);updates.append({'index':index,'feature_update':actual['update_record'],'closure':proof,
                'guard_closure':record(child_folder/'closure.json'),'original_PCM_discarded_after_closed_verified_feature_handoff':True})
            child=None
            write_json(folder/'status.json',{'status':'preparing_next_update','PID':os.getpid(),'completed_feature_updates':len(closures),
                'actual_optimizer_updates':0,'authoritative_frontiers_advanced':False,'window_owner':window_owner})
            del prepared,feature_update,pcm_manifest
        manifest=finalize_window(window_owner,closures,guard=guard)
        final=read_bound(manifest,guard)
        require(journal.binding()['offset']==0,'Preparation must not append any authoritative commit')
        report={'passed':True,'kind':'FastMSS_actual_closed20_CPU_feature_window_preparation_v1','owner':window_owner,
            'manifest':manifest,'plan':plan_binding,'replay_reference':replay_binding,'updates':updates,'source_bindings':pinned_sources,
            'aggregate_feature_window_bytes':final['aggregate_files_bytes'],'actual_closed_feature_children':20,
            'actual_optimizer_updates':0,'authoritative_frontiers_advanced':False,'full_corpus_passes':0,
            'feature_window_retained_for_native_control_and_checkpoint1_10_20':True,'production_enabled':False,
            'native_feature_consumer_validated':False,'wall_seconds':time.monotonic()-started}
        write_json(folder/'report.json',report)
        write_json(folder/'status.json',{'status':'ACTUAL_CPU_preparation_complete','PID':os.getpid(),'completed_feature_updates':20,
            'manifest':manifest,'actual_optimizer_updates':0,'authoritative_frontiers_advanced':False})
        print(json.dumps({'passed':True,'updates':20,'bytes':final['aggregate_files_bytes'],'manifest':manifest}),flush=True)
    except BaseException as error:
        write_json(folder/'failure.json',{'error':repr(error),'completed_feature_updates':len(closures),'owner':window_owner,
            'no_automatic_retry':True,'actual_optimizer_updates':0,'authoritative_frontiers_advanced':False})
        raise
    finally:
        if child is not None and child.poll() is None:
            write_json(child_folder/'stop.json',{'reason':'owned_feature_coordinator_leaving'})
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                import psutil
                boot=json.loads((child_folder/'boot.json').read_text());process=psutil.Process(boot['pid'])
                require((boot['pid']==child.pid or boot['ppid']==child.pid) and process.ppid()==boot['ppid'],'Exact owned frontend identity before stop')
                process.terminate();process.wait(timeout=5);child.wait(timeout=5)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);main(parser.parse_args().output)
