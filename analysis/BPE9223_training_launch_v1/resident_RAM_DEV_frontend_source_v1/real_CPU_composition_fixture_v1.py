"""One real fixed DEV row and original frontend; CUDA transfer alone is intercepted."""
import hashlib,json,os,random,sys,threading,time,traceback
from pathlib import Path
P=Path(r'C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import digest
out=Path(sys.argv[1]);require(not out.exists(),'Fresh owned CPU fixture');out.mkdir()
require(os.environ['CUDA_VISIBLE_DEVICES']=='-1' and sys.flags.utf8_mode==1,'CPU-only upfront UTF8')
import psutil
psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
began=time.monotonic();stop=threading.Event();errors=[];peak=[0];samples=[0]
def guard():
    require(not errors,'Actual CPU fixture monitor: '+repr(errors))
    require(time.monotonic()-began<180,'Bounded180s real CPU composition')
def observe():
    from resource_watchdog_v2 import sample_memory
    while not stop.wait(.1):
        try:
            s=sample_memory(os.getpid(),'real_DEV_frontend_composition_CPU');samples[0]+=1
            peak[0]=max(peak[0],s['worker_private_memory']['private_bytes'])
            require(peak[0]<=2100000000 and s['system_memory']['commit_available_bytes']>=2000000000
                and s['system_memory']['physical_available_bytes']>=1500000000,'Same DEV incremental resource/headroom scope')
        except BaseException as e:errors.append(repr(e));return
t=threading.Thread(target=observe,daemon=True);t.start()
try:
    import torch,numpy as np
    torch.set_num_threads(1)
    from native_tdt_minimal_import_v1 import prepare
    prepare()
    from transformers import AutoProcessor
    from tokenizers import Tokenizer
    from extend_vocab import _make_tokenizer
    from asr18_token_only_carrier_v1 import load
    from asr18_BPE9223_model_migration_v2 import checked_plan
    from asr18_resident_serial350_DEV_v2 import compose_DEV_processor
    import runtime
    configuration=Path(r'E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1/resident_RAM_periodic4000_source_v1/configuration.json')
    c=read_json(configuration);base=Path((P/'work/base_model_path.txt').read_text().strip())
    training=load(base);_,candidate=checked_plan(c['row_migration_plan'],guard)
    training.tokenizer=_make_tokenizer(training.tokenizer,Tokenizer.from_str(json.dumps(candidate,ensure_ascii=False)))
    require(len(training.tokenizer)==9223 and training.blank_token_id==8192,'Exact candidate9223/immutable blank')
    candidate_vocab=candidate['model']['vocab'];require(training.tokenizer.get_vocab()==candidate_vocab,'All exact9223 IDs')
    old_tokenizer=training.tokenizer;old_vocab=digest(old_tokenizer.get_vocab());old_specials=dict(old_tokenizer.special_tokens_map)
    rng=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
    processor,composition=compose_DEV_processor(training,guard=guard)
    policy=read_json(c['full_run_identity']['scorer_policy']['path']);manifest=policy['DEV_manifests']['tr']
    require(record(manifest['path'])=={k:manifest[k]for k in ('path','bytes','sha256')},'Pinned actual fixed Turkish DEV membership')
    row=json.loads(next(v for v in Path(manifest['path']).read_text(encoding='utf-8-sig').splitlines()if v.strip()))
    original_cuda=torch.Tensor.cuda;calls=[]
    def CPU_transfer_only(tensor,*args,**kwargs):
        require(tensor.device.type=='cpu','Actual frontend produced CPU tensor');calls.append({'shape':list(tensor.shape),'dtype':str(tensor.dtype)})
        return tensor
    try:
        torch.Tensor.cuda=CPU_transfer_only
        actual=runtime.inputs_for(processor,row)
    finally:torch.Tensor.cuda=original_cuda
    guard();audio=runtime.read_audio(row)
    baseline=AutoProcessor.from_pretrained(base,local_files_only=True)
    expected=baseline(audio=[audio],sampling_rate=16000,return_tensors='pt')
    require(set(actual)==set(expected) and calls and all(torch.equal(actual[k],expected[k])for k in actual),
        'Original runtime.inputs_for/full original frontend exact features/masks')
    require(training.tokenizer is old_tokenizer and digest(training.tokenizer.get_vocab())==old_vocab
        and training.tokenizer.special_tokens_map==old_specials,'Token-only caller and all inherited specials unchanged')
    ids=training.tokenizer(row['text'],add_special_tokens=False)['input_ids']
    require(processor.decode(ids,skip_special_tokens=True)==training.tokenizer.decode(ids,skip_special_tokens=True),
        'Original processor/tokenizer decode delegation agrees; no decode policy changed')
    try:training(audio=[audio],sampling_rate=16000,return_tensors='pt')
    except ValueError as e:require('Audio frontend unavailable' in str(e),'Truthful original failed composition')
    else:raise ValueError('Original token-only carrier must still reject audio')
    random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2])
    require(random.getstate()==rng[0] and np.array_equal(np.random.get_state()[1],rng[1][1])
        and torch.equal(torch.get_rng_state(),rng[2]) and not torch.cuda.is_initialized(),'CPU RNG preserved; no CUDA/model/native work')
    report={'kind':'ASR18_real_original_DEV_frontend_live9223_composition_CPU_fixture_v1','passed':True,
        'source':record(__file__),'helper_source':record(S/'asr18_resident_serial350_DEV_v2.py'),'configuration':record(configuration),
        'producer':{'pid':os.getpid(),'create_time':psutil.Process().create_time(),'ppid':os.getppid()},
        'scorer_manifest':manifest,'source_id':row['source_id'],'language':row['language'],'sample_count':len(audio),
        'row_sha256':digest(row),'migrated_tokenizer_vocabulary_sha256':old_vocab,'composition':composition,
        'actual_original_runtime_inputs_for_executed':True,'actual_AutoProcessor_call_and_frontend_executed':True,
        'only_final_Tensor_cuda_transfer_intercepted_and_restored':True,'GPU_transfer_or_model_generation_tested':False,
        'feature_tensors':{k:{'shape':list(v.shape),'dtype':str(v.dtype),'sha256':hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest()}for k,v in actual.items()},
        'transfer_calls':calls,'original_frontend_baseline_byte_equal':True,'training_carrier_unchanged':True,
        'decode_delegation_agrees_without_policy_change':True,'CUDA_initialized':torch.cuda.is_initialized(),
        'native_updates_or_sampler_draws':0,'wall_seconds':time.monotonic()-began}
    write_json(out/'report.json',report);print(json.dumps({'passed':True,'report':record(out/'report.json')}),flush=True)
except BaseException as e:
    write_json(out/'failure.json',{'error':repr(e),'traceback':traceback.format_exc()});raise
finally:
    stop.set();t.join(3)
    write_json(out/'monitor_completion.json',{'samples':samples[0],'peak_private_bytes':peak[0],'errors':errors,'joined':not t.is_alive()})
