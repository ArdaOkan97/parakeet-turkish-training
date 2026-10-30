"""Finite E-only hybrid simulated RIR generation, no speech/model/GPU import."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
ROOT = Path(r'E:\gpu_training_scratchpad')
sys.path.insert(0, str(PROJECT/'outputs/parakeet_turkish'))

ROOMS = [
 ('treated_small', 'Small treated room', [3.5,3.,2.5], .15),
 ('office', 'Office', [5.,4.,2.8], .3),
 ('meeting', 'Meeting room', [7.,5.,3.], .5),
 ('classroom', 'Classroom', [9.,7.,3.2], .8),
 ('small_hall', 'Small hall', [14.,9.,4.], 1.2),
 ('large_hall', 'Large hall', [20.,12.,6.], 1.8),
 ('tall_atrium', 'Tall atrium', [14.,12.,9.], 2.2),
 ('reflective_hall', 'Large reflective hall', [24.,16.,8.], 2.5)]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--start',type=int,required=True);p.add_argument('--end',type=int,required=True)
    args=p.parse_args(); assert 0 <= args.start < args.end <= 8
    out=Path(args.output).resolve(); assert out.is_relative_to(ROOT.resolve()) and not out.exists()
    assert os.environ['CUDA_VISIBLE_DEVICES']=='-1'
    sys.path.insert(0,str(ROOT/'scratch/FastMSS_pyroom_runtime_v1/isolated_imports'))
    import numpy as np
    import scipy
    import soundfile as sf
    import pyroomacoustics as pra
    import fastmss_room_bank_preview_v1 as helper
    from resource_watchdog_v2 import sample_memory
    def guard():
        sample=sample_memory(os.getpid(),'diverse_hybrid_RIR_generation')
        w,s=sample['worker_private_memory'],sample['system_memory']
        assert w['available'] and s['available'] and sample['paired_sampling_within_one_second']
        assert w['private_bytes'] < 2_000_000_000 and s['commit_available_bytes']>=2_000_000_000
        until=time.monotonic()+15
        while s['commit_used_bytes'] >= .9*s['commit_limit_bytes']:
            assert s['commit_used_bytes'] < .95*s['commit_limit_bytes'] and time.monotonic()<until
            time.sleep(.2);s=sample_memory(os.getpid(),'diverse_hybrid_RIR_yield')['system_memory']
    guard();out.mkdir(parents=True,exist_ok=False);assets=[]
    for index in range(args.start,args.end):
        room_id,label,dims,target=ROOMS[index];seed=2026101700+index;pra.random.seed(seed)
        guard();began=time.monotonic();alpha,unused_order=pra.inverse_sabine(target,dims)
        assert 0 < alpha < 1
        horizon=min(6.,max(.75,2.5*target))
        room=pra.ShoeBox(dims,fs=16000,materials=pra.Material(alpha,scattering=.2),
            max_order=3,ray_tracing=True,air_absorption=True)
        room.set_ray_tracing(n_rays=8192,receiver_radius=.5,energy_thres=1e-7,time_thres=horizon,hist_bin_size=.004)
        mic=np.array([.23*dims[0],.37*dims[1],1.5]);room.add_microphone(mic)
        positions=[('near',mic+np.array([.8,.5,.1])),('far',np.array([.80*dims[0],.76*dims[1],1.6]))]
        for name,position in positions:room.add_source(position)
        room.compute_rir();guard()
        for source_index,(name,position) in enumerate(positions):
            rir=np.asarray(room.rir[0][source_index],dtype=np.float32)
            assert 0 < len(rir) <= 7*16000 and np.isfinite(rir).all()
            path=out/f'{room_id}_{name}.wav';sf.write(path,rir,16000,subtype='FLOAT')
            restored,rate=sf.read(path,dtype='float32');assert rate==16000 and np.array_equal(restored,rir)
            distance=float(np.linalg.norm(position-mic));delay=int(pra.constants.get('frac_delay_length')//2)
            measurement=helper.measure_decay(restored,expected_direct_arrival_sample=delay+distance/room.c*16000,
                horizon_seconds=horizon,guard=guard)
            assets.append({'id':f'{room_id}_{name}','room_id':room_id,'room_label':label,'position':name,
                'audio':helper.pin(path,guard=guard),'sample_rate':16000,'samples':len(rir),
                'room_dimensions_m':dims,'microphone_xyz_m':mic.tolist(),'source_xyz_m':position.tolist(),
                'source_microphone_distance_m':distance,'target_RT60_seconds':target,
                'wall_uniform_energy_absorption':float(alpha),'wall_scattering':.2,
                'global_fractional_filter_delay_samples':delay,'sound_speed_m_per_s':room.c,
                'method':'experimental_pyroomacoustics_hybrid_ISM_order3_and_finite_ray_tracing',
                'ray_settings':dict(room.rt_args),'seed':seed,'measured_decay':measurement,
                'actual_room_generation_wall_seconds':time.monotonic()-began,
                'real_room_recording':False,'production_enabled':False})
        del room
        (out/'progress.json').write_text(json.dumps({'completed_room':index,'assets':len(assets),'PID':os.getpid()}),encoding='utf-8')
    result={'kind':'FastMSS_diverse_hybrid_simulated_room_RIR_bank_v1','production_enabled':False,
        'generator_seed_base':2026101700,'room_indices':[args.start,args.end],'assets':assets,
        'runtime':{'NumPy':np.__version__,'SciPy':scipy.__version__,'SoundFile':sf.__version__,'pyroomacoustics':pra.__version__},
        'source':helper.pin(__file__,guard=guard),'helper_source':helper.pin(helper.__file__,guard=guard),
        'method_docs':'https://pyroomacoustics.readthedocs.io/en/stable/pyroomacoustics.room.html',
        'method_caveat':'Experimental hybrid simulation with finite rays; target Sabine RT60 is not measured decay or a real-room recording.',
        'no_GPU_model_speech_or_training_used':not any(x in sys.modules for x in ('torch','nemo'))}
    (out/'room_RIR_registry_v1.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'status':'generated','assets':len(assets),'registry':str(out/'room_RIR_registry_v1.json'),'PID':os.getpid()}))

if __name__=='__main__':main()
