"""Real owned native wait0 and exact checkpoint continuation; no hotpath supervisor."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
import psutil
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path[:0]=[str(S),str(P/'work/research')]
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
from asr18_memory_diagnosis_v2 import storage


def main(a):
    storage();require(sys.flags.utf8_mode==1 and os.environ.get('PYTHONUTF8')=='1','UTF8 before owner entry')
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    review=read_json(a.review);driver=P/'work/research/run_ASR18_BPE9223_resident_RAM_training_v5.py'
    require(review['accepted'] is True and review['reviewer']=='/root'
        and review['driver_source']==record(driver)
        and review['RAM_native_source']==record(S/'asr18_BPE9223_grouped_RAM_native_v1.py')
        and review['CPU_source']==record(S/'fastmss_resident_feature_stream_v1.py')
        and review['DEV_helper_source']==record(S/'asr18_resident_serial350_DEV_v2.py')
        and review['source_configuration']==record(a.configuration)
        and review['initial_checkpoint']==record(a.pointer) and review['latest_failure']==record(a.latest_failure),
        'Exact one ROOT source/science/real CPU output execution binding')
    require(a.queue_bytes==review['bounded_RAM_queue_bytes']==268435456 and review['periodic_DEV_interval_updates']==4000,
        'Unchanged bounded queue and periodic cadence')
    configuration=read_json(a.configuration);policy=configuration['private_scalar_prefetch_policy']
    require(review['private_scalar_prefetch_source']==policy['source']==record(S/'asr18_BPE9223_private_scalar_prefetch_v1.py')
        and review['private_scalar_prefetch_actual_CPU_review']==policy['actual_CPU_control_review']
        and record(policy['actual_CPU_control_review']['path'])==policy['actual_CPU_control_review']
        and review['additional_prefetch_host_reservation_bytes']==policy['additional_host_reservation_bytes']==268435456,
        'Exact integrated helper, finite equality review and full added host reservation')
    if a.prior_process is not None:
        require(a.DEV_frontend_resume is None,'New normal-boundary entry requires genuine prior wait0, not historical failure relabel')
    else:
        require(a.DEV_frontend_resume is not None and review['DEV_frontend_resume']==record(a.DEV_frontend_resume),
            'Only the already-supported typed4000 failure lane when no genuine prior process0')
    branch=Path(a.branch).resolve();control=Path(a.output).resolve()
    require(control.is_relative_to(Path(r'E:\gpu_training_scratchpad')) and not control.exists(),'Fresh E loop records');control.mkdir()
    pointer=Path(a.pointer);prior=None if a.prior_process is None else Path(a.prior_process);invocation=a.start_invocation;root_pid=os.getpid()
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0';env['PYTHONUTF8']='1'
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
    while True:
        rows=subprocess.check_output(['nvidia-smi','--query-gpu=memory.total,memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines()
        require(len(rows)==1,'Exactly one admitted training GPU')
        total,free=[int(v.strip())*1024**2 for v in rows[0].split(',')]
        require(total>=17000000000 and free>=14000000000,'Physical GPU capacity plus unchanged12GB allocator/2GBfree reserve')
        command=[str(P/'work/nemo-native-env-v2/Scripts/python.exe'),'-X','utf8','-B',str(driver),
            '--configuration',a.configuration,'--stream-configuration',a.configuration,'--pointer',str(pointer),
            '--latest-failure',a.latest_failure,'--folder',str(branch),'--invocation',str(invocation),
            '--queue-bytes',str(a.queue_bytes),'--maximum-wall-seconds','86400']
        if prior is not None:command+=['--prior-process',str(prior)]
        elif invocation==a.start_invocation:command+=['--DEV-frontend-resume',a.DEV_frontend_resume]
        if invocation==1:command+=['--validation-stop','460']
        log=control/('native_'+str(invocation).zfill(6)+'.log')
        with log.open('xb')as stream:
            child=subprocess.Popen(command,cwd=P,env=env,stdout=stream,stderr=subprocess.STDOUT)
            launcher={'pid':child.pid,'create_time':psutil.Process(child.pid).create_time(),'ppid':root_pid}
            write_json(control/'current_child.json',{'invocation':invocation,'launcher':launcher,'command':command,
                'log':str(log),'branch':str(branch),'source_checkpoint':record(pointer)})
            print(json.dumps({'actual_native_launch':launcher,'invocation':invocation,'log':str(log),'saved_source_updates':read_json(pointer)['committed_updates']}),flush=True)
            code=child.wait()
        inv=branch/'invocations'/('owner_'+str(invocation).zfill(6))
        require(code==0,'Actual native process failed; preserve all state/logs, no unchanged retry')
        report=read_json(inv/'normal_completion.json');who=report['producer']
        try:alive=abs(psutil.Process(who['pid']).create_time()-who['create_time'])<.001
        except psutil.NoSuchProcess:alive=False
        require(not alive and report['passed'] is True and report['all_live_updates_saved'] is True,'Real normal0 and all live updates coherently saved')
        closure={'kind':'ASR18_actual_resident_RAM_GPU_process_wait0_v1','actual_Popen_wait_code':code,'launcher':launcher,
            'producer':who,'producer_observed_dead':True,'worker_report':record(inv/'normal_completion.json'),
            'source':record(__file__),'worker_source':record(driver),'log':record(log)}
        prior=control/('native_'+str(invocation).zfill(6)+'_actual0.json');write_json(prior,closure)
        pointer=Path(report['latest_checkpoint']['pointer']['path'])
        print(json.dumps({'actual_normal0':record(prior),'saved_updates':report['actual_optimizer_updates'],'automatic_continuation':report['completed_boundaries']<3}),flush=True)
        if report['completed_boundaries']==3:break
        invocation+=1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--review',required=True);p.add_argument('--configuration',required=True)
    p.add_argument('--pointer',required=True);p.add_argument('--latest-failure',required=True);p.add_argument('--branch',required=True)
    p.add_argument('--output',required=True);p.add_argument('--queue-bytes',type=int,default=268435456);p.add_argument('--start-invocation',type=int,default=1);p.add_argument('--prior-process');p.add_argument('--DEV-frontend-resume');main(p.parse_args())
