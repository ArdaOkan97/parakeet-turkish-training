"""Focused synthetic metadata/NumPy/owned-Popen successor controls only."""
import argparse,ast,gc,hashlib,importlib.util,json,os,subprocess,sys,time
from copy import deepcopy
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
import fastmss_row_serial_cpu_v2 as c
from fastmss_raw_lifetime_v1 import *
from asr18_owned_family_observation_v2 import owned_family_sample,same_process,terminate_known_family
from atomic_io_retry_v1 import write_json,read_json

def leaf(folder):
    folder.mkdir(exist_ok=False);write_json(folder/'identity.json',c.identity());deadline=time.monotonic()+25
    while not (folder/'release.json').exists():c.resource_guard();require(time.monotonic()<deadline,'Bounded intentional CPU orphan');time.sleep(.05)

def outer(folder):
    folder.mkdir(exist_ok=False)
    with (folder/'leaf.log').open('xb') as stream:
        child=subprocess.Popen([str(c.CPU_ENV/'Scripts/python.exe'),'-B',str(P/'work/research/ASR18_bounded_CPU_phase_v3.py'),
            '--folder',str(folder/'leaf_guard'),'--seconds','35','--script',str(Path(__file__).resolve()),'--','--mode','leaf','--folder',str(folder/'leaf')],
            stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        deadline=time.monotonic()+15
        while not (folder/'leaf/identity.json').exists():c.resource_guard();require(child.poll() is None and time.monotonic()<deadline,'Guarded leaf startup');time.sleep(.05)
        write_json(folder/'ready.json',{'producer':c.identity(),'leaf':read_json(folder/'leaf/identity.json')})
        while not (folder/'exit_outer.json').exists():c.resource_guard();require(time.monotonic()<deadline,'Finite outer protocol');time.sleep(.05)

def main(folder):
    c.environment();guard=c.resource_guard;folder=c.scoped(folder);folder.mkdir(exist_ok=False)
    before=list(c.sink.CACHE.iterdir()) if c.sink.CACHE.exists() else [];checks=[]
    def check(name,value=True):require(value,name);checks.append(name)
    def reject(name,fn):
        try:fn()
        except (ValueError,KeyError,TypeError):checks.append(name);return
        raise AssertionError('Expected rejection: '+name)
    files=[S/'fastmss_raw_lifetime_v1.py',S/'fastmss_bounded_original_audio_v1.py',S/'fastmss_row_serial_cpu_v2.py',
        P/'work/research/produce_FastMSS_independent_rows_CPU_v2.py',P/'work/research/produce_FastMSS_serial_features_CPU_v2.py',
        P/'work/research/prepare_FastMSS_phase_window_CPU_v2.py']
    for path in files:ast.parse(path.read_text());check('parse_'+path.name)
    planner_path=S/'bpe_final_DATA_window_planner_v4.py';tree=ast.parse(planner_path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='FinalBroadRows')
    row_method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='row')
    assignment=next(n for n in ast.walk(row_method) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='value' for t in n.targets) and isinstance(n.value,ast.Dict))
    for dataset,sr,n in (('kiraat',24000,24000),('alania_yodas3',48000,48000)):
        raw={'ordinal':1,'dataset':dataset,'unique_id':'synthetic:'+dataset,'source_id':'original-recording',
            'text':'Köşeyi, kenarı burayı kullanıyorum.','original_flags_json':'{"review_alarm":true}','start':0,'end':1,
            'shard':str(c.ROOT/'datasets/kiraat/synthetic.parquet') if dataset=='kiraat' else '0000'}
        declared={'ordinal':1,'source_component':1 if dataset=='kiraat' else 0,'native_sr':sr,'native_samples':n,
            'clean16k_samples':16000,'mel_frames':100,'encoder_frames':13,'target_tokens':3,'status':'declared'}
        ids=[4,8100,8700];env={'raw':deepcopy(raw),'declared':declared,'u':len(ids),'ids':ids,'deepcopy':deepcopy,'from_samples':c.from_samples}
        code=compile(ast.fix_missing_locations(ast.Module(body=[deepcopy(assignment)],type_ignores=[])),str(planner_path),'exec');exec(code,env)
        actual=c.normalize_declared_TR(raw,declared,ids)
        check('exact_planner4_row_'+dataset,c.same(actual,env['value']))
        check('all_raw_flags_and_targets_preserved_'+dataset,actual['broad_original']==raw and actual['tr_metadata_labels']==ids)
        missing=deepcopy(actual);missing.pop('declared_frontend_geometry')
        check('historical_missing_geometry_fails_exact_row_'+dataset,not c.same(actual,missing))
        bad=deepcopy(actual);bad['declared_frontend_geometry']['physical_feature_frames']-=1
        check('cropped_geometry_cannot_equal_planner_'+dataset,not c.same(actual,bad))
        path=canonical_source_path(raw,c.ROOT)
        check('canonical_source_'+dataset,path==(c.ROOT/'datasets/yodas3/audio/0000.tar' if dataset=='alania_yodas3' else Path(raw['shard'])))
        pin={'path':str(path),'bytes':100,'sha256':'a'*64,'mtime_ns':1,'receipt':{'synthetic':True},'index':None if dataset=='kiraat' else {'synthetic':True}}
        obs={'actual_original_source_path':str(path),'source_file_sha256':'a'*64}
        check('inventory_source_SHA_'+dataset,source_identity_binding(raw,pin,obs,c.ROOT)['sha256']=='a'*64)
        reject('wrong_decoder_SHA_'+dataset,lambda:source_identity_binding(raw,pin,{**obs,'source_file_sha256':'b'*64},c.ROOT))
    reject('noncanonical_YODAS_shard',lambda:canonical_source_path({'dataset':'alania_yodas3','shard':'../0000'},c.ROOT))
    reject('KIRAAT_outside_E',lambda:canonical_source_path({'dataset':'kiraat','shard':str(P/'synthetic.parquet')},c.ROOT))
    reject('unknown_native_count',lambda:raw_plan(native_samples=None,native_rate=48000,clean_samples=16000,phase=c.bridge.WEAK,role='student_tr',channels_bound=8,frame_samples_bound=16384))
    reject('unsupported_unknown_channels',lambda:raw_plan(native_samples=48000,native_rate=48000,clean_samples=16000,phase=c.bridge.WEAK,role='student_tr',channels_bound=9,frame_samples_bound=16384))
    reject('declared_resample_count_mismatch',lambda:raw_plan(native_samples=48000,native_rate=48000,clean_samples=16001,phase=c.bridge.WEAK,role='student_tr',channels_bound=8,frame_samples_bound=16384))
    plan=raw_plan(native_samples=48000,native_rate=48000,clean_samples=16000,phase=c.bridge.WEAK,role='student_tr',channels_bound=8,frame_samples_bound=16384)
    ledger=RawLifetimeLedger(guard=guard);ledger.begin('one',plan);ledger.begin('two',plan)
    reject('third_live_job_rejected',lambda:ledger.begin('three',plan))
    check('raw128_twojobs_exact',ledger.peak<=CAPACITY and ledger.job_peak==2)
    ledger.finish('one');ledger.finish('two')
    large=raw_plan(native_samples=48000*1200,native_rate=48000,clean_samples=16000*1200,phase=c.bridge.WEAK,role='student_tr',channels_bound=8,frame_samples_bound=16384)
    allocated=[]
    def too_large():ledger.begin('oversized',large);allocated.append('never')
    reject('oversized_fails_BEFORE_allocator_callback',too_large);check('allocator_not_called',not allocated)
    from fastmss_bounded_original_audio_v1 import BorrowedCompressedIO
    media=bytes(range(256))*1000
    with BorrowedCompressedIO(media) as borrowed:
        check('borrowed_compressed_read_bytes_exact',borrowed.read(12345)==media[:12345])
        borrowed.seek(-20,2);check('borrowed_compressed_seek_end_exact',borrowed.read(20)==media[-20:])
        borrowed.seek(77);buffer=bytearray(512);count=borrowed.readinto(buffer)
        check('borrowed_compressed_readinto_bytes_exact',count==512 and bytes(buffer)==media[77:589])
        reject('whole_compressed_duplicate_read_forbidden',lambda:borrowed.read(-1))
        reject('oversized_compressed_callback_forbidden',lambda:borrowed.read(1024**2+1))
    np,_=c.numpy_runtime();local=np.random.default_rng(813)
    wave=(local.standard_normal(140003,dtype=np.float32)*np.float32(.8)).astype(np.float32);wave[:6]=[-1.5,-1.,-0.,0.,1.,1.5]
    f32,i16=chunked_hashes(wave,guard=guard)
    old=np.clip(np.rint(wave*32768),-32768,32767).astype('<i2')
    check('chunked_full_i16_hash_byteexact',i16==hashlib.sha256(old.tobytes()).hexdigest())
    check('chunked_full_f32_hash_byteexact',f32==hashlib.sha256(wave.tobytes()).hexdigest());del old,wave
    ledger.begin('actual_numpy',plan)
    with ledger.stage('actual_numpy','decode_resample'):
        wave=np.zeros(16000,dtype=np.float32);ledger.observe('actual_numpy',wave,name='clean',maximum_bytes=64000)
    reject('cannot_release_live_clean_buffer',lambda:ledger.finish('actual_numpy'))
    with ledger.stage('actual_numpy','weak_transform'):
        prepared=(wave+np.float32(.01)).astype(np.float32);ledger.observe('actual_numpy',prepared,name='prepared',maximum_bytes=64000)
    del wave;gc.collect();reject('cannot_release_live_prepared_buffer',lambda:ledger.finish('actual_numpy'))
    del prepared;gc.collect();ledger.finish('actual_numpy');raw_receipt=ledger.receipt();check('all_actual_raw_buffers_released',raw_receipt['passed'])
    poisoned=RawLifetimeLedger(guard=guard);poisoned.begin('p',plan)
    def overlap():
        with poisoned.stage('p','decode_resample'):
            with poisoned.stage('p','canonical_hash'):pass
    reject('overlapping_unplanned_stage_rejected',overlap);reject('failed_ledger_cannot_claim_success',poisoned.receipt)
    bypass=raw_plan(native_samples=48000*121,native_rate=48000,clean_samples=16000*121,phase=c.bridge.WEAK,role='student_tr',channels_bound=8,frame_samples_bound=16384)
    check('over120_retained_clean_policy',bypass['weak_transform_possible'] is False and bypass['source_excluded_or_cropped'] is False)
    for phase,role in ((c.bridge.CLEANER,'student_tr'),(c.bridge.WEAK,'replay')):
        protected=raw_plan(native_samples=16000,native_rate=16000,clean_samples=16000,phase=phase,role=role,channels_bound=1,frame_samples_bound=16384)
        check('protected_no_transform_'+role+phase,protected['weak_transform_possible'] is False)
    # Actual Windows orphan failure tests NEW inner-controller retirement,
    # without ever dispatching genuine DATA or touching shared feature CACHE.
    spec=importlib.util.spec_from_file_location('new_inner_controller_under_test',files[-1]);ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
    known={};samples=[]
    with (folder/'outer.log').open('xb') as stream:
        child=subprocess.Popen([str(c.CPU_ENV/'Scripts/python.exe'),'-B',str(P/'work/research/ASR18_bounded_CPU_phase_v3.py'),
            '--folder',str(folder/'outer_guard'),'--seconds','40','--script',str(Path(__file__).resolve()),'--','--mode','outer','--folder',str(folder/'outer')],stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        launcher=c.identity(child.pid);deadline=time.monotonic()+20
        try:
            while not (folder/'outer/ready.json').exists():
                guard();require(child.poll() is None and time.monotonic()<deadline,'Bounded actual outer child')
                samples.append(owned_family_sample(child,launcher,known=known,guard=guard));time.sleep(.05)
            samples.append(owned_family_sample(child,launcher,known=known,guard=guard));ready=read_json(folder/'outer/ready.json')
            require(max(v['private_bytes_sum'] for v in samples)<400_000_000,'Tiny actual Popen family400MB bound')
            write_json(folder/'outer/exit_outer.json',{'controlled_exit':True});check('actual_outer_handle_exit0',child.wait(timeout=10)==0)
            check('leaf_live_AFTER_launcher0',same_process(ready['leaf']))
            retired=ctl.retire_owned_failure(child,launcher,known,folder)
            check('NEW_inner_cleanup_observed_all_dead',retired['retirement']['all_captured_owned_identities_observed_dead'] and not same_process(ready['leaf']))
            check('failure_cleanup_never_fake_normal_leaf0',retired['normal_leaf_exit0_claimed'] is False and retired['passed'] is False)
        finally:
            if child.poll() is None or any(same_process(v) for v in known.values()):terminate_known_family(child,launcher,known)
    check('actual_shared_feature_CACHE_untouched',(list(c.sink.CACHE.iterdir()) if c.sink.CACHE.exists() else [])==before)
    check('no_Torch_frontend_model_import',not any(k=='torch' or k.startswith(('torch.','transformers','librosa','pyarrow')) for k in sys.modules))
    result={'kind':'FastMSS_raw_lifetime_and_row_contract_control_fixture_v1','passed':True,'checks':checks,'check_count':len(checks),
        'sources':[c.record(path,guard) for path in files],'planner_source':c.record(planner_path,guard),'raw_receipt':raw_receipt,
        'actual_owned_orphan_family_peak_bytes':max(v['private_bytes_sum'] for v in samples),'actual_orphan_failure_retirement':retired,
        'real_audio_frontend_model_GPU_or_DATA_prepared':False,'actual_shared_CACHE_unchanged':True,'actual_optimizer_updates':0,
        'actual_sampler_or_draw_frontier_advanced':False,'full_corpus_raw_fit_or_production_approval_claimed':False}
    c.write_new(folder/'report.json',result);print(json.dumps({'passed':True,'checks':len(checks)}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('main','outer','leaf'),default='main');p.add_argument('--folder',required=True)
    a=p.parse_args();globals()[a.mode](Path(a.folder).resolve())
