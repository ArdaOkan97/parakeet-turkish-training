"""Metadata-only fixed400 continued-production speed observer, no benchmark/job.
Requires genuine progress beyond159020 and complete original timing rows; never
stops training, reads tensors/audio, or changes sources/environment/resources.
"""
import hashlib,json,math,os,shutil,sys,time
from pathlib import Path
import psutil
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
from resource_watchdog_v2 import sample_memory
N=Path('E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1');A=Path(__file__).parent
F=N/'resident_RAM_two_logical_packet_lookahead_v1';B=Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01');I=B/'invocations/owner_000021'
START=158620;END=159020;BEGIN=time.monotonic();PEAK=0

def guard():
    global PEAK
    x=sample_memory(os.getpid(),'source29_fixed400_existing_production_records');m=x['system_memory'];PEAK=max(PEAK,x['worker_private_memory']['private_bytes'])
    require(x['paired_sampling_within_one_second']and PEAK<536870912 and time.monotonic()-BEGIN<180
        and m['commit_available_bytes']>=2268435456 and m['physical_available_bytes']>=1768435456
        and shutil.disk_usage('E:/').free>=100000000000,'Observer512MiB180s/fullprefetch/originalfloors/E100GB')
    return x

def main():
    me=psutil.Process();me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);parent=me.parent()
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'and os.environ.get('PYTHONUTF8')=='1'
        and all(os.environ.get(k)=='1'for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'))
        and Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad')),
        'Separate metadata CPU process-local E environment only')
    initial=guard();m=initial['system_memory'];require(m['commit_available_bytes']>=2805306368 and m['physical_available_bytes']>=2305306368,
        'Full prospective512MiB plus full256MiB and original floors')
    progress=read_json(B/'progress/latest.json');require(progress['producer']=={'pid':37836,'create_time':1792733651.0335317,'ppid':4344}
        and progress['actual_optimizer_updates']>=END and not progress['optimizer_poisoned']and not progress['schedule_poisoned']
        and progress['last_saved_updates']>=159000 and progress['planned_native_updates']==401072
        and progress['administrative_update_stop']is None and not(I/'actual_failure.json').exists(),
        'Same healthy continued source29 fixed interval naturally complete')
    out=A/'actual_fixed400_speed158620_159020_01';require(not out.exists(),'ONE fixed interval metadata seal');out.mkdir()
    endpoint_raw={};endpoints={};endpoint_offsets={};log=A/'native_000021.log'
    with log.open('rb')as f:
        while f.tell()<2*1024**2:
            guard();begin=f.tell();line=f.readline(262144);finish=f.tell()
            require(line and len(line)<262144,'Bounded existing native log includes fixed endpoints')
            try:value=json.loads(line)
            except (ValueError,UnicodeDecodeError):continue
            if value.get('kind')=='ASR18_actual_resident_RAM_training_progress_v1'and value.get('actual_optimizer_updates')in(START,END):
                u=value['actual_optimizer_updates'];require(u not in endpoints,'One real emitted fixed progress endpoint')
                require(value['producer']==progress['producer']and value['source']==progress['source'],'Same producer/source across interval')
                endpoints[u]=value;endpoint_raw[u]=line;endpoint_offsets[u]={'start':begin,'end':finish,'bytes':len(line),'sha256':hashlib.sha256(line).hexdigest()}
            if len(endpoints)==2:break
    require(set(endpoints)=={START,END},'Both actual emitted endpoints, not startup cumulative estimate')
    timing_path=I/'private_metadata_window_timings.jsonl';rows=[];row_raw=[];row_start=None;row_end=None
    with timing_path.open('rb')as f:
        while f.tell()<2*1024**2:
            guard();begin=f.tell();line=f.readline(131072);finish=f.tell()
            require(line.endswith(b'\n')and len(line)<131072,'Complete original bounded timing row')
            value=json.loads(line)
            if value['initial_updates']>=START and value['committed_updates']<=END:
                if row_start is None:row_start=begin
                rows.append(value);row_raw.append(line);row_end=finish
            if value['committed_updates']>=END:break
    require(len(rows)==20 and [(r['initial_updates'],r['committed_updates'])for r in rows]==[(u,u+20)for u in range(START,END,20)],
        'ALL20 contiguous full production windows, no cherry picking')
    checkpoints=[r['committed_updates']for r in rows if r['checkpoint_included']]
    require(checkpoints==[158800,159000],'Fixed400 naturally includes BOTH200-save windows')
    a,b=endpoints[START],endpoints[END];updates=END-START
    wall=b['active_pipeline_seconds']-a['active_pipeline_seconds'];audio=b['source_audio_seconds']-a['source_audio_seconds'];native=b['native_seconds']-a['native_seconds']
    require(wall>0 and audio>0 and native>0 and all(y-x==400 for x,y in zip(a['clocks'],b['clocks']))
        and b['native_updates_this_process']-a['native_updates_this_process']==400,'Genuine400 updates and three clocks')
    sums={k:math.fsum(r.get(k,0)for r in rows)for k in ('window_pipeline_seconds','native_seconds','source_audio_seconds',
        'checkpoint_seconds','native_commit_pressure_wait_seconds','native_commit_pressure_waited_updates',
        'end_window_wait_count','serial_fallback_count','worker_preparation_seconds','main_adoption_and_submission_seconds')}
    require(abs(sums['native_seconds']-native)<.005 and abs(sums['source_audio_seconds']-audio)<.01,
        'Exact timing sums agree with full emitted cumulative endpoint deltas')
    endpoint_copy=out/'actual_emitted_progress_endpoints.jsonl';endpoint_copy.write_bytes(endpoint_raw[START]+endpoint_raw[END])
    timing_copy=out/'all20_original_window_timing_rows.jsonl';timing_copy.write_bytes(b''.join(row_raw))
    write_json(out/'current_positive_progress.json',progress)
    durability={}
    for u in (158800,159000):
        ck=B/('checkpoints/update_'+str(u).zfill(9));pointer=read_json(ck/'pointer.json');p=read_json(ck/'persistence_verified.json')
        require(pointer['committed_updates']==u and p['passed']is True and p['optimizer_updates']==u
            and p['named_tensors_verified']==854 and p['weights_verified']==207 and p['Adam_states_verified']==215
            and p['independent_pre_save_state_sha256']==p['saved_state_sha256']and p['saved_semantic_state_verified']['passed']is True,
            'Natural full854/207/215 PRE-SAVED semantic checkpoint remains genuine')
        durability[str(u)]={'pointer':record(ck/'pointer.json'),'persistence':record(ck/'persistence_verified.json')}
    family=read_json(A/'actual_source29_firstpositive_runtime_family.json')['actual_runtime_identities'];observed=[]
    for role,who in family.items():
        p=psutil.Process(who['pid']);require(abs(p.create_time()-who['create_time'])<.001,'Same exact current owned identity '+role)
        observed.append({'role':role,**who,'exact_alive':True})
    configuration=read_json(I/'configuration.json');review=read_json(F/'ROOT_source29_approved_two_logical_lookahead_saved158600_final_execution_review_v1.json')
    require(all(configuration[k]==v for k,v in review['future_seven_runtime_bindings'].items())
        and configuration['resident_RAM_packet_credit_policy']==review['packet_credit_policy']
        and configuration['native_update_commit_pressure_policy']==review['native_update_commit_pressure_policy'],
        'Source29 actual max2/receiver1/225/all7 bindings still unchanged')
    report={'kind':'ASR18_actual_source29_fixed400_checkpoint_inclusive_continued_streaming_observation_v1','passed':True,
        'owned_session':87302,'fixed_interval':[START,END],'updates':400,'complete_emitted_wall_seconds':wall,
        'source_audio_seconds':audio,'audio_minutes_per_wall_minute':audio/wall,'training_and_commit_seconds':native,
        'native_training_and_commit_seconds_per_update':native/updates,'outside_training_and_commit_seconds_per_update':(wall-native)/updates,
        'audio_seconds_per_update':audio/updates,'updates_per_hour':updates*3600/wall,
        'actual_progress_endpoints':{str(START):a,str(END):b},'endpoint_source_offsets':{'path':str(log),'rows':endpoint_offsets},
        'actual_endpoint_copy':record(endpoint_copy),'all20_original_timing_row_copy':record(timing_copy),
        'timing_source_segment':{'path':str(timing_path),'start_offset':row_start,'end_offset':row_end,
            'bytes':row_end-row_start,'sha256':record(timing_copy)['sha256']},'all20_window_rows':rows,'window_sums':sums,
        'complete_endpoint_wall_minus_sum_window_seconds':wall-sums['window_pipeline_seconds'],
        'checkpoint_updates_included':checkpoints,'natural_durability':durability,'current_progress':record(out/'current_positive_progress.json'),
        'same_actual_five_identities_alive':observed,'configuration':record(I/'configuration.json'),
        'actual_credit_and_225_policies_unchanged':True,'full_goal401072_continues_no_finite_stop':True,
        'baseline_original400':{'audio_minutes_per_wall_minute':76.569475,'scope':'400 continued updates with2saves',
            'ROOT_receipt':record(N/'ROOT_prefetch_actual_400_update_verification_v1.json')},
        'recent_source27_ordinary80':{'audio_minutes_per_wall_minute':53.61977345,'scope':'80updates without saves, different ordinals',
            'owner_receipt':record(N/'resident_RAM_source27_speed_review_v1/actual_post_Discord_Steam_quit_existing_windows157080_157160_v1.json')},
        'limits':['Native timer includes training+transaction/journal commit, not pure GPU kernels',
            'Fixed400 excludes first20 startup but a fresh process/fullDB hashing may warm OS cache; permanence not proved',
            'Actual CPU feature preparation/transport time is not separately retained by driver; no invented split',
            'Full source29 policy/source checks and actual throughput observed; no memory-fit/saving/future-speed guarantee'],
        'no_audio_model_GPU_probe_benchmark_or_new_DEV':True,'no_child_environment_source_or_resource_change':True,
        'ROOT_actual_acceptance_pending':True,'sealer_identity':{'pid':me.pid,'create_time':me.create_time(),'ppid':me.ppid()},
        'sealer_launcher':{'pid':parent.pid,'create_time':parent.create_time(),'ppid':parent.ppid()},
        'guard_initial':initial,'guard_final':guard(),'sampled_observer_private_peak':PEAK,'sealer_source':record(__file__)}
    require('numpy'not in sys.modules and 'torch'not in sys.modules,'No array/model/GPU imports')
    write_json(out/'actual_fixed400_speed_status.json',report)
    print(json.dumps({'status':record(out/'actual_fixed400_speed_status.json'),'rate':audio/wall,'wall':wall,'audio':audio,
        'native_per_update':native/400,'gate_wait_seconds':sums['native_commit_pressure_wait_seconds'],'checkpoint_seconds':sums['checkpoint_seconds']}),flush=True)
if __name__=='__main__':main()
