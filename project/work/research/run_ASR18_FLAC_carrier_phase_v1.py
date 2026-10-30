"""One finite pinned carrier phase; one guarded native CPU child at a time."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil
from ASR18_continuous_data_guard_v4 import DataIOGuard,ROOT,AREA
from ASR18_FLAC_carrier_guard_v3 import CACHE,metadata,signature
from cpu_preparation_guard_v2 import now,write_json,memory

PROJECT=Path(__file__).resolve().parents[2]
WORKER=PROJECT/'work/research/prepare_ASR18_KIRAAT_FLAC_records_v3.py'
RUNTIME=PROJECT/'work/nemo-native-env-v2/Scripts/python.exe'


def binding(path):
    p=Path(path);return {'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def cache_bytes():
    return sum(p.stat().st_size for p in CACHE.rglob('*') if p.is_file())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',required=True);args=parser.parse_args()
    phase=Path(args.phase).resolve();assert phase.is_relative_to(ROOT.resolve())
    guard=DataIOGuard('FLAC_phase_supervision',heavy=False,maximum_bytes=32_000_000)
    guard.wait();phase.mkdir(exist_ok=False)
    approved=AREA/'KIRAAT_FLAC_sweep_plan_prepared_v1.json'
    assert binding(approved)['sha256']=='cb2d65d9a09132846d9a47ae1099b5147e702fb8aed2b5fd258467b617513361'
    plan=metadata(AREA/'preparation_plan_v1.json');CACHE.mkdir(exist_ok=True)
    jobs=[j for j in plan['download_jobs'] if j['repository']=='serdarcaglar/kiraat' and Path(j['path']).name.startswith(('train-','rest-'))]
    assert len(jobs)==729
    reused=[];selected=[];size_estimate=0
    for job in sorted(jobs,key=lambda j:j['path']):
        source=Path(job['destination']);source_receipt=source.with_suffix(source.suffix+'.receipt.json')
        if not(source.exists() and source_receipt.exists()):continue
        r=metadata(source_receipt)
        assert source.stat().st_size==job['size'] and r['sha256']==job['sha256'] and r['revision']==job['revision']
        folder=CACHE/job['sha256'];receipt_path=folder/'receipt.json';audit_path=folder/'validation.json'
        if receipt_path.exists() and audit_path.exists():
            receipt=metadata(receipt_path);audit=metadata(audit_path)
            assert receipt['complete'] is True and audit['passed'] is True and audit['source_sha256']==job['sha256']
            assert audit['carrier_receipt']==binding(receipt_path)
            for field in ('records','index'):
                actual=signature(receipt[field]['path'])
                assert actual['bytes']==receipt[field]['bytes'] and actual['mtime_ns']==receipt[field]['mtime_ns']
            reused.append({'job':job,'receipt':binding(receipt_path),'validation':binding(audit_path),
                           'physical_rows':receipt['physical_rows'],'records_bytes':receipt['records']['bytes']})
            continue
        if len(selected)>=32 or size_estimate+job['size']>15_000_000_000:continue
        existing=(folder/'records.bin').stat().st_size if (folder/'records.bin').exists() else 0
        selected.append({'job':job,'source_signature':signature(source),'source_receipt':binding(source_receipt),
                         'initial_payload_bytes':existing})
        size_estimate+=job['size']
    assert len(reused)>=5 and selected
    sources={str(p):binding(p) for p in (Path(__file__),WORKER,PROJECT/'work/research/ASR18_FLAC_carrier_guard_v3.py')}
    pinned={'kind':'ASR18_bounded_original_FLAC_carrier_phase_v1','created_at_utc':now(),'owner':'/root/cpu_data_preparation',
        'supervisor_PID':os.getpid(),'source_pins':sources,'approved_plan':binding(approved),'selected':selected,
        'reused_verified_fixtures':reused,'maximum_new_sources':32,'maximum_new_payload_bytes':16_000_000_000,
        'maximum_total_paced_IO_bytes':96_000_000_000,'control_metadata_reservation_bytes':32_000_000,
        'aggregate_cache_cap_bytes':400_000_000_000,'minimum_E_reserve_bytes':100_000_000_000,
        'reserve_all_remaining_fixed_downloads':True,'maximum_carrier_IO_bytes_s':32*1024**2,
        'one_native_CPU_carrier_child':True,'runtime':str(RUNTIME),'TRAIN_admitted':False}
    write_json(phase/'pinned_plan.json',pinned)
    finished=[];charged=0;child_runs=[];started=time.monotonic()
    def growth():
        return sum(max(0,(CACHE/item['job']['sha256']/'records.bin').stat().st_size-item['initial_payload_bytes'])
                   for item in selected if (CACHE/item['job']['sha256']/'records.bin').exists())
    def publish(status,**extra):
        state={'kind':'ASR18_bounded_original_FLAC_carrier_phase_status_v1','at_utc':now(),
            'supervisor_PID':os.getpid(),'status':status,'selected_sources':len(selected),
            'new_sources_prepared_and_audited':len(finished),'reused_sources':len(reused),
            'new_payload_bytes':growth(),'actual_completed_child_paced_IO_bytes':charged,
            'elapsed_seconds':time.monotonic()-started,'one_CPU_carrier_child':True,
            'full729_coverage':False,'TRAIN_admitted':False,**extra}
        write_json(phase/'status.json',state);write_json(AREA/'KIRAAT_FLAC_carrier_current_status_v1.json',state)
    publish('pinned_before_preparation',active_child_PID=None)
    for source_index,item in enumerate(selected):
        for mode in ('prepare','audit'):
            guard.wait()
            for path,pin in sources.items():assert binding(path)==pin,'Do not mutate admitted phase sources'
            run=phase/f'{source_index:03d}_{mode}';run.mkdir()
            config={**item,'mode':mode,'run_folder':str(run),'cache_bytes_before':cache_bytes(),
                'remaining_download_bytes_reserved':sum(j['size'] for j in plan['download_jobs'] if not Path(j['destination']).exists()),
                'remaining_IO_bytes':96_000_000_000-32_000_000-charged,
                'remaining_new_payload_bytes':16_000_000_000-growth()}
            assert config['remaining_IO_bytes']>0 and config['remaining_new_payload_bytes']>=0
            config_path=run/'config.json';write_json(config_path,config)
            with (run/'stdout.log').open('wb') as log:
                child=subprocess.Popen([str(RUNTIME),'-X','utf8','-B',str(WORKER),'--config',str(config_path)],
                    stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=os.environ.copy(),cwd=PROJECT)
                publish(mode,active_child_PID=child.pid,current_source=item['job']['path'],run_folder=str(run))
                last=0
                while child.poll() is None:
                    if time.monotonic()-last>5:
                        publish(mode,active_child_PID=child.pid,current_source=item['job']['path'],run_folder=str(run))
                        last=time.monotonic()
                    time.sleep(.25)
                actual=child.wait()
            native_path=run/'native_closure.json';native=metadata(native_path) if native_path.exists() else None
            actual_proof={'kind':'ASR18_actual_owned_FLAC_child_exit_v1','observed_at_utc':now(),'native_PID':child.pid,
                'mode':mode,'actual_exit_code':actual,'source_sha256':item['job']['sha256'],
                'worker_source':sources[str(WORKER)],'native_closure':binding(native_path) if native else None}
            write_json(run/'actual_owned_exit.json',actual_proof);child_runs.append(actual_proof)
            if actual!=0 or native is None:
                publish('stopped_preserved_prefix',active_child_PID=None,failed_mode=mode,actual_exit_code=actual)
                raise RuntimeError('Owned carrier child stopped; preserved source and append-only prefix; no automatic retry')
            assert native['native_return_code']==0 and native['watchdog_clean'] is True
            charged+=native['actual_paced_IO_bytes']
            assert charged+32_000_000<=96_000_000_000 and growth()<=16_000_000_000 and cache_bytes()<=400_000_000_000
            result=metadata(run/'result.json')
            if mode=='audit':
                finished.append({'job':item['job'],'source_signature':item['source_signature'],
                    'source_receipt':item['source_receipt'],'prepared_actual_exit':binding(phase/f'{source_index:03d}_prepare/actual_owned_exit.json'),
                    'audited_actual_exit':binding(run/'actual_owned_exit.json'),'audit_result':result})
                write_json(phase/'completed_sources.json',finished)
                publish('source_prepared_and_all_bytes_audited',active_child_PID=None,current_source=item['job']['path'])
                print(json.dumps({'source':item['job']['path'],'prepared_and_audited':len(finished),
                    'payload_bytes':growth(),'actual_paced_IO_bytes':charged}),flush=True)
    publish('completed_declared_bounded_phase',active_child_PID=None,actual_exit_code=0)
    write_json(phase/'native_phase_CLOSED0.json',{'kind':'ASR18_native_bounded_FLAC_phase_CLOSED0_v1',
        'at_utc':now(),'supervisor_PID':os.getpid(),'native_return_code':0,'pinned_plan':binding(phase/'pinned_plan.json'),
        'completed_sources':binding(phase/'completed_sources.json'),'new_sources':len(finished),
        'new_payload_bytes':growth(),'actual_paced_IO_bytes':charged,'full729_coverage':False,'TRAIN_admitted':False})


if __name__=='__main__':main()
