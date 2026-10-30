"""Whole-phase arithmetic/stop controls, no new native/forecast fit evidence."""
import argparse,json,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from atomic_io_retry_v1 import write_json
from bpe_broad_audio_v1 import record,require
from asr18_precomputed_worker_forecast_v1 import whole_worker_forecast,phase_projection
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')
def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);a=p.parse_args();f=Path(a.folder);require(f.is_relative_to(A),'E-only policy arithmetic');f.mkdir(exist_ok=False)
    prior_path=A/'geometry_forecast_CPU_actual_04/report.json';prior=json.loads(prior_path.read_text(encoding='utf-8'));require(prior['passed'],'Exact proposed native metadata controls')
    native=prior['proposed_initial_budget'];forecast=whole_worker_forecast(native);checks=[]
    require(forecast['native_private_forecast_bytes']==5_560_813_440 and forecast['entire_worker_private_forecast_bytes']==5_910_813_440,'Whole worker must include350MB beyond native ceiling')
    checks.append('Entire worker5,910,813,440B includes unchanged350MB restore beyond native5,560,813,440B')
    for phase,private,increment in (('native',native['current_private_bytes'],native['incremental_bytes']),
        ('checkpoint_write',native['absolute_private_forecast_bytes'],64_000_000),
        ('checkpoint_restore',native['absolute_private_forecast_bytes'],350_000_000)):
        phase_projection(forecast,phase=phase,current_private_bytes=private,increment_bytes=increment);checks.append(phase+' retains its full allowance without assumed residency drop')
    def reject(name,call):
        try:call()
        except ValueError:checks.append(name);return
        raise ValueError('Expected rejection: '+name)
    reject('Native growth beyond future restore envelope stops before native',lambda:phase_projection(forecast,phase='native',current_private_bytes=5_560_813_441,increment_bytes=1))
    reject('Write reserve cannot shrink',lambda:phase_projection(forecast,phase='checkpoint_write',current_private_bytes=1,increment_bytes=63_000_000))
    reject('Restore reserve cannot shrink',lambda:phase_projection(forecast,phase='checkpoint_restore',current_private_bytes=1,increment_bytes=349_000_000))
    reject('Unknown phase has no fallback',lambda:phase_projection(forecast,phase='CPU_frontend',current_private_bytes=1,increment_bytes=1))
    require(not any(n in sys.modules for n in ('torch','numpy','scipy','transformers','pyarrow')),'No numeric/native imports');checks.append('No GPU/native/CPU frontend or production approval')
    write_json(f/'report.json',{'passed':True,'check_count':len(checks),'checks':checks,'source':record(__file__),
        'forecast_source':record(S/'asr18_precomputed_worker_forecast_v1.py'),'native_proposal':record(prior_path),'proposal':forecast,
        'prospective_actual_success_claimed':False,'actual_optimizer_updates':0,'production_forecast_approved':False})
    print(json.dumps({'passed':True,'checks':len(checks),'entire_worker_private_proposal':forecast['entire_worker_private_forecast_bytes'],'actual_native_updates':0}))
if __name__=='__main__':main()
