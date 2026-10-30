"""Exact failing resident-forecast constructor under upfront UTF8; CPU metadata only."""
import argparse,ctypes,json,os,sys,time,threading,traceback
from pathlib import Path
from ctypes import wintypes
P=Path(__file__).resolve().parents[2];sys.path[:0]=[str(P/'outputs/parakeet_turkish'),str(P/'work/research')]
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import digest
from run_ASR18_BPE9223_full_run_owner_v19 import fresh_admit,private_lifetime_limit
from run_ASR18_frozen_CNN_lifetime_pair_v8 import Watch,pid_identity,storage
ROOT=Path('E:/gpu_training_scratchpad')
def main(a):
    storage();folder=Path(a.folder).resolve();require(folder.is_relative_to(ROOT) and not folder.exists(),'Fresh E metadata output')
    failure_pin=record(a.failure);failure=read_json(a.failure);review=read_json(a.ROOT_review)
    require(review['accepted'] is True and review['reviewer']=='/root'
        and review['scope']=='metadata_BPE9223_prepared60_exact_UTF8_resident_forecast_v1'
        and review['source']==record(__file__) and review['actual_failure']==failure_pin
        and review['production_approval_claimed'] is False,'Exact ROOT source/resource scope')
    require(os.environ.get('PYTHONUTF8')=='1' and sys.flags.utf8_mode==1 and sys.getdefaultencoding()=='utf-8',
        'UTF8 process-local launch policy active BEFORE imports/constructor, never runtime environment patch')
    kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.GetCurrentProcess.restype=wintypes.HANDLE
    kernel.SetPriorityClass.argtypes=[wintypes.HANDLE,wintypes.DWORD];kernel.SetPriorityClass.restype=wintypes.BOOL
    require(kernel.SetPriorityClass(kernel.GetCurrentProcess(),0x4000),'BelowNormal metadata job')
    samples=[]
    for _ in range(3):
        x=fresh_admit(512000000);require(x['passed'],'Fresh system512MB prospective');samples.append(x);time.sleep(.2)
    folder.mkdir();write_json(folder/'boot.json',{'source':record(__file__),'producer':pid_identity(),'actual_failure':failure_pin})
    write_json(folder/'admission.json',samples);watch=Watch(folder,128000000,worker=True);watch.phase='exact_failing_forecast_UTF8'
    timer=threading.Timer(60,lambda:write_json(folder/'stop.json',{'reason':'reviewed60s'}));timer.daemon=True;timer.start();code=1;began=time.monotonic()
    def guard():watch.check();private_lifetime_limit(128000000)
    try:
        keys=('storage_state','durable_latest','current_registered_journal','pool_registry','pool_index','pool_archive_prefix','storage_operation_witnesses')
        require(all(record(failure[k]['path'])==failure[k]for k in keys),'All7 current prepared60 authority pins')
        cp=failure['attempted_native_configuration'];require(record(cp['path'])==cp,'Exact actual failed worker configuration')
        c=read_json(cp['path']);require(c['worker_mode']=='resident_training' and c['restore_parent_pointer']==failure['checkpoint_pointer'],
            'Exact original prepared60 resident restore, no GPU process')
        from asr18_BPE9223_geometry_forecast_v1 import CalibratedGeometryForecast,resident_worker_forecast
        calibration=CalibratedGeometryForecast(**c['forecast_calibration'],guard=guard)
        inputs=read_json(c['first_feature_update']['path']);require(record(c['first_feature_update']['path'])==c['first_feature_update'],'Same actual source-bound first feature update')
        expected=resident_worker_forecast(calibration.budget(rows=inputs['rows'],current_private_bytes=3000000000,update_in_process=1))
        require(digest(expected)==digest(c['entire_worker_forecast']),'Exact full forecast matches admitted worker policy after real constructor')
        require(all(record(failure[k]['path'])==failure[k]for k in keys),'All7 unchanged after constructor/budget')
        guard();write_json(folder/'report.json',{'kind':'ASR18_actual_prepared60_UTF8_exact_resident_forecast_preflight_v1','passed':True,
            'source':record(__file__),'producer':pid_identity(),'actual_failure':failure_pin,'attempted_native_configuration':cp,
            'sys_flags_utf8_mode':sys.flags.utf8_mode,'PYTHONUTF8':os.environ['PYTHONUTF8'],'default_encoding':sys.getdefaultencoding(),
            'actual_calibration_source':record(sys.modules[CalibratedGeometryForecast.__module__].__file__),
            'full_forecast':expected,'full_forecast_sha256':digest(expected),'all7_authoritative_pins_unchanged':True,
            'actual_GPU_audio_or_CPU_feature_jobs':0,'wall_seconds':time.monotonic()-began,'private':private_lifetime_limit(128000000)})
        code=0;print(json.dumps({'passed':True,'report':record(folder/'report.json')}),flush=True)
    except BaseException:
        with (folder/'traceback.log').open('a',encoding='utf8')as stream:traceback.print_exc(file=stream)
        raise
    finally:timer.cancel();watch.close(code)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--failure',required=True);p.add_argument('--ROOT-review',required=True);p.add_argument('--folder',required=True);main(p.parse_args())
