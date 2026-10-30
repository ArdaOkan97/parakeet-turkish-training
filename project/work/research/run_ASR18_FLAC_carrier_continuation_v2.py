"""Finite continuation of the exact original32-source pilot after actual old closure."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from ASR18_continuous_data_guard_v4 import DataIOGuard,ROOT,AREA
from ASR18_FLAC_carrier_guard_v3 import CACHE,metadata,signature
from cpu_preparation_guard_v2 import now,write_json

PROJECT=Path(__file__).resolve().parents[2]
WORKER=PROJECT/'work/research/prepare_ASR18_KIRAAT_FLAC_records_v4.py'
RUNTIME=PROJECT/'work/nemo-native-env-v2/Scripts/python.exe'


def binding(path):
    p=Path(path);return {'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def cache_bytes():return sum(p.stat().st_size for p in CACHE.rglob('*') if p.is_file())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',required=True);args=parser.parse_args()
    phase=Path(args.phase).resolve();assert phase.is_relative_to(ROOT.resolve())
    guard=DataIOGuard('FLAC_phase_supervision_v2',heavy=False,maximum_bytes=32_000_000);guard.wait()
    phase.mkdir(exist_ok=False)
    old=AREA/'KIRAAT_FLAC_carrier_phase_v1';retirement_path=AREA/'carrier_phase_v1_actual_retirement_v1.json'
    retirement=metadata(retirement_path)
    assert retirement['actual_exit_code']==1 and retirement['no_owned_carrier_child_running'] is True
    assert retirement['actual_tool_chunk_id']=='c4afef'
    original=metadata(old/'pinned_plan.json');complete=metadata(old/'completed_sources.json')
    all_selected=original['selected'];done_sha={x['job']['sha256'] for x in complete}
    remaining=[(i,x) for i,x in enumerate(all_selected) if x['job']['sha256'] not in done_sha]
    assert len(all_selected)==32 and len(remaining)==31 and len(complete)==1
    selected=remaining[:2]  # Finish the prepared source, then measure one fresh complete prepare+audit pair.
    inherited=int(retirement['actual_paced_IO_bytes']);charged=inherited
    plan=metadata(AREA/'preparation_plan_v1.json')
    sources={str(p):binding(p) for p in (Path(__file__),WORKER,
        PROJECT/'work/research/prepare_ASR18_KIRAAT_FLAC_records_v3.py',
        PROJECT/'work/research/ASR18_FLAC_carrier_guard_v3.py',PROJECT/'work/research/ASR18_FLAC_carrier_guard_v4.py')}
    started=time.monotonic()
    def growth():
        return sum(max(0,(CACHE/x['job']['sha256']/'records.bin').stat().st_size-x['initial_payload_bytes'])
            for x in all_selected if (CACHE/x['job']['sha256']/'records.bin').exists())
    initial_growth=growth()
    pinned={'kind':'ASR18_bounded_FLAC_original_pilot_continuation_v2','created_at_utc':now(),
        'owner':'/root/cpu_data_preparation','supervisor_PID':os.getpid(),'source_pins':sources,
        'original_pinned_plan':binding(old/'pinned_plan.json'),'actual_retirement':binding(retirement_path),
        'ROOT_acknowledgment':binding(AREA/'ROOT_KIRAAT_FLAC_sweep_acknowledgment_v1.json'),
        'remaining_original_selected':selected,'initial_completed_sources':complete,
        'declared_added_audits_at_tuning_boundary':2,'original_pilot_sources_remaining_after_boundary':29,
        'initial_new_payload_bytes':initial_growth,'inherited_actual_paced_IO_bytes':inherited,
        'maximum_total_original_pilot_sources':32,'maximum_cumulative_pilot_payload_bytes':16_000_000_000,
        'maximum_cumulative_pilot_IO_bytes':96_000_000_000,'control_metadata_reservation_bytes':32_000_000,
        'aggregate_cache_cap_bytes':400_000_000_000,'minimum_E_reserve_bytes':100_000_000_000,
        'reserve_all_remaining_fixed_downloads':True,'maximum_carrier_IO_bytes_s':64*1024**2,
        'expected_peak_private_bytes':1_000_000_000,'extra_projected_memory_margin_bytes':250_000_000,
        'one_native_CPU_carrier_child':True,'runtime':str(RUNTIME),'full729_coverage':False,'TRAIN_admitted':False}
    write_json(phase/'pinned_plan.json',pinned)
    finished=list(complete)
    def publish(status,**extra):
        state={'kind':'ASR18_bounded_FLAC_carrier_continuation_status_v2','at_utc':now(),
            'supervisor_PID':os.getpid(),'status':status,'remaining_selected_sources':len(selected),
            'original_pilot_prepared_and_audited_sources':len(finished),'reused_prior_fixtures':5,
            'original_pilot_new_payload_bytes':growth(),'continuation_payload_growth_bytes':growth()-initial_growth,
            'cumulative_completed_child_paced_IO_bytes':charged,'inherited_actual_paced_IO_bytes':inherited,
            'elapsed_seconds':time.monotonic()-started,'carrier_IO_ceiling_bytes_s':64*1024**2,
            'full729_coverage':False,'TRAIN_admitted':False,**extra}
        write_json(phase/'status.json',state);write_json(AREA/'KIRAAT_FLAC_carrier_current_status_v1.json',state)
    publish('pinned_continuation_before_work',active_child_PID=None,active_launcher_PID=None)
    for original_index,item in selected:
        receipt=CACHE/item['job']['sha256']/'receipt.json'
        prior_prep=old/f'{original_index:03d}_prepare/actual_owned_exit.json'
        if receipt.exists():
            proof=metadata(prior_prep)
            assert proof['actual_exit_code']==0 and proof['source_sha256']==item['job']['sha256']
            preparation_proof=prior_prep;modes=('audit',)
        else:preparation_proof=None;modes=('prepare','audit')
        for mode in modes:
            guard.wait()
            for path,pin in sources.items():assert binding(path)==pin,'Do not mutate admitted continuation sources'
            run=phase/f'{original_index:03d}_{mode}';run.mkdir()
            config={**item,'mode':mode,'run_folder':str(run),'cache_bytes_before':cache_bytes(),
                'remaining_download_bytes_reserved':sum(j['size'] for j in plan['download_jobs'] if not Path(j['destination']).exists()),
                'remaining_IO_bytes':96_000_000_000-32_000_000-charged,
                'remaining_new_payload_bytes':16_000_000_000-growth(),'expected_peak_private_bytes':1_000_000_000}
            assert config['remaining_IO_bytes']>0 and config['remaining_new_payload_bytes']>=0
            config_path=run/'config.json';write_json(config_path,config)
            with (run/'stdout.log').open('wb') as log:
                child=subprocess.Popen([str(RUNTIME),'-X','utf8','-B',str(WORKER),'--config',str(config_path)],
                    stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=os.environ.copy(),cwd=PROJECT)
                last=0
                while child.poll() is None:
                    if time.monotonic()-last>2:
                        live=metadata(run/'live.json') if (run/'live.json').exists() else {}
                        publish(mode,active_child_PID=live.get('PID'),active_launcher_PID=child.pid,
                            current_source=item['job']['path'],run_folder=str(run),child_live=live)
                        last=time.monotonic()
                    time.sleep(.25)
                actual=child.wait()
            native_path=run/'native_closure.json';native=metadata(native_path) if native_path.exists() else None
            proof={'kind':'ASR18_actual_owned_FLAC_child_exit_v1','observed_at_utc':now(),
                'native_PID':native['PID'] if native else None,'launcher_PID':child.pid,
                'mode':mode,'actual_exit_code':actual,'source_sha256':item['job']['sha256'],
                'worker_source':sources[str(WORKER)],'native_closure':binding(native_path) if native else None}
            write_json(run/'actual_owned_exit.json',proof)
            if actual!=0 or native is None:
                publish('stopped_preserved_prefix',active_child_PID=None,active_launcher_PID=None,
                    failed_mode=mode,actual_exit_code=actual)
                raise RuntimeError('Carrier stopped; source/prefix preserved; no automatic retry')
            assert native['native_return_code']==0 and native['watchdog_clean'] is True
            charged+=native['actual_paced_IO_bytes']
            assert charged+32_000_000<=96_000_000_000 and growth()<=16_000_000_000 and cache_bytes()<=400_000_000_000
            if mode=='prepare':preparation_proof=run/'actual_owned_exit.json'
            else:
                result=metadata(run/'result.json')
                finished.append({'job':item['job'],'source_signature':item['source_signature'],
                    'source_receipt':item['source_receipt'],'prepared_actual_exit':binding(preparation_proof),
                    'audited_actual_exit':binding(run/'actual_owned_exit.json'),'audit_result':result})
                write_json(phase/'completed_sources.json',finished)
                publish('source_prepared_and_all_bytes_audited',active_child_PID=None,active_launcher_PID=None,current_source=item['job']['path'])
                print(json.dumps({'source':item['job']['path'],'original_pilot_audited_sources':len(finished),
                    'original_pilot_payload_bytes':growth(),'cumulative_paced_IO_bytes':charged}),flush=True)
    assert len(finished)==3
    publish('completed_declared_two_source_tuning_boundary',active_child_PID=None,active_launcher_PID=None,actual_exit_code=0)
    write_json(phase/'native_phase_CLOSED0.json',{'kind':'ASR18_native_bounded_FLAC_continuation_CLOSED0_v2',
        'at_utc':now(),'supervisor_PID':os.getpid(),'native_return_code':0,'pinned_plan':binding(phase/'pinned_plan.json'),
        'completed_sources':binding(phase/'completed_sources.json'),'original_pilot_sources':len(finished),
        'original_pilot_new_payload_bytes':growth(),'cumulative_original_pilot_paced_IO_bytes':charged,
        'full729_coverage':False,'TRAIN_admitted':False})


if __name__=='__main__':main()
