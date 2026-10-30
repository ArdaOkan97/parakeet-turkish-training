"""Bounded DE/PT causal component evaluation: one model, two diagnostic hybrids.

No optimizer, export, TEST, corpus preparation or training. Only CLOSED8250
BEST and8750 latest trainable207 maps are paired for fixed100 DEV per hybrid.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,ctypes,gc,hashlib,json,math,os,random,sys,threading,time
from collections import deque
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent
SOURCE='adaptive_20261012_TR_uniform_majority_PT_focus_integer_01'
PINS={'train_lora_batched_tsc_uniform_majority_ptfocus_v3.py':'cbe277fb8464008d755f7f88f0c26cc2662bcc871efd271b5759e99f7352ec5b',
      'run_TSC_uniform_majority_ptfocus_v3.py':'1de96cba7a708f461e8340ae39ea64a89b9653fecac6b76ca4ccde5330578e9f',
      'preload_nemo_linker_v1.py':'f8787b0f3346c9f526903ff0c06de4c266f940ca3b0f41c55907cc795e7cc2ba',
      'native_tdt_checkpointed_v4.py':'24b9cf5b8f3e11d5b043c22d0940da3735a46f8f70b6f5810b97f17d545cd90e'}
def require(ok,message):
    if not ok:raise ValueError(message)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def record(path):return {'path':str(Path(path).resolve()),'sha256':sha(path),'bytes':Path(path).stat().st_size}
def write(path,value):
    path=Path(path);temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temporary.replace(path)
def now():return datetime.now(timezone.utc).isoformat()

def preflight(args):
    require(args.max_seconds>0 and args.max_seconds<=600,'At most600 seconds diagnostic bound')
    run=HERE/'runs'/SOURCE;selection=read(run/'selection.json');config=read(run/'run_config.json')
    closure=read(HERE/'PT_focus_integer_actual_CLOSED8750_retention_review_v1.json')
    require(closure['actual_owned_exit']=={'session_id':43870,'exit_code':0,'tool_chunk':'df6a79'}
        and closure['sources_unchanged'] is True and not closure['resource_failure'],'Actual owned CLOSED8750 required')
    require(selection['selected_step']==8250 and selection['completed_steps']==8750 and selection['accepted'] is True,'Coherent actual BEST8250/latest8750 pointers required')
    for name,digest in PINS.items():require(sha(HERE/name)==digest,'Frozen source changed:'+name)
    for name,digest in config['config_identity']['source_sha256'].items():require(sha(HERE/name)==digest,'Source scientific chain changed:'+name)
    require(not args.output.exists(),'Fresh diagnostic output')
    require(read(ROOT/'work/gpu_training_owner.json')['owner']=='/root/benchmark_design','Exclusive delegated GPU lease required')
    return config,selection,closure

def main(args):
    config,selection,closure=preflight(args)
    if args.validate_only:print(json.dumps({'passed':True,'GPU_launched':False,'diagnostic_utterances':200}));return
    require(args.gpu_window_authorized,'Explicit exclusive GPU authorization required')
    from resource_watchdog_v2 import ResourceGuard,sample_memory
    initial=sample_memory(os.getpid(),'retention_components_before_GPU')['system_memory'];peak=closure['observed_peak_owned_private_bytes']
    require(initial['available'] and initial['commit_used_bytes']+peak<.9*initial['commit_limit_bytes']
        and initial['commit_limit_bytes']-initial['commit_used_bytes']-peak>2000000000,'Fresh baseline+actual predecessor peak fails memory admission')
    priority_api=ctypes.WinDLL('kernel32',use_last_error=True);priority_api.GetCurrentProcess.restype=ctypes.c_void_p
    priority_api.SetPriorityClass.argtypes=[ctypes.c_void_p,ctypes.c_ulong]
    require(priority_api.SetPriorityClass(priority_api.GetCurrentProcess(),0x4000),'BelowNormal priority failed')
    started=time.monotonic();guard=ResourceGuard();stop=threading.Event();failure=[];observations=deque(maxlen=160);phase=['starting']
    report={'kind':'actual_CLOSED8750_DE_PT_component_diagnostic_v1','status':'running','pid':os.getpid(),
        'source_run':SOURCE,'script':record(__file__),'started_at_utc':now(),'source_pins':PINS,
        'memory_admission':initial,'previous_actual_private_peak':peak,'max_seconds':args.max_seconds,
        'one_GPU_model':True,'optimizer_created':False,'weights_exported':False,'TEST_evaluated':False,
        'native_loss_kernel_used':False,'diagnostic_not_inference_speed_benchmark':True,
        'corners_saved':{'A_best8250':{'de':70,'pt':66},'B_latest8750':{'de':71,'pt':71}},'hybrids':{}}
    write(args.output,report)
    def monitor():
        while not stop.is_set():
            sample=sample_memory(os.getpid(),phase[0]);reason,evidence=guard.observe(sample['system_memory'],sample['worker_private_memory'],observed_monotonic=sample['observed_monotonic'])
            observations.append({'at_utc':now(),'phase':phase[0],'evidence':evidence})
            if reason:failure.append(reason)
            stop.wait(5)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    def check():
        if failure:raise RuntimeError(failure[0])
        if time.monotonic()-started>args.max_seconds:raise RuntimeError('diagnostic600second_bound')
    try:
        require('torch' not in sys.modules,'Preload before ML import')
        from preload_nemo_linker_v1 import preload
        report['actual_preload']=preload()
        import torch
        torch.cuda.set_per_process_memory_fraction(8000000000/torch.cuda.get_device_properties(0).total_memory,0)
        from runtime import load_model,DATA
        from encoder_lora import install_encoder_lora,encoder_lora_metadata
        from extend_vocab import extend_processor_and_model
        from train_lora import checkpoint_encoder_blocks
        import benchmark
        phase[0]='load_one_model';check();model,processor=load_model()
        vocab=extend_processor_and_model(processor,model)
        layout=install_encoder_lora(model,rank=config['rank'],alpha=config['alpha'],last_n_layers=config['last_n_layers'])
        model.config.encoder_lora_tracking={'layout':layout,'implementation_sha256':sha(HERE/'encoder_lora.py'),'kind':'unmerged'}
        if not config['no_encoder_checkpoint']:checkpoint_encoder_blocks(model)
        expected={name:p for name,p in model.named_parameters() if p.requires_grad}
        require(len(expected)==207 and model.config.vocab_size==8199 and model.config.blank_token_id==8192,'Original8199/all207 layout required')
        maps={}
        for role,key,step in (('best','weights',8250),('latest','latest_weights',8750)):
            path=Path(selection[key]);report[role+'_checkpoint']=record(path)
            saved=torch.load(path,map_location='cpu',weights_only=True)
            require(saved['schema']==2 and saved['step']==step and saved['run_id']==SOURCE
                and saved['config_identity']==config['config_identity'] and saved['lora_metadata']==encoder_lora_metadata(model),'Actual matching checkpoint scientific layout/frontier required')
            weights=saved['weights'];require(set(weights)==set(expected),'Exact all207 names required')
            for name,tensor in weights.items():require(tensor.dtype==torch.float32 and tensor.shape==expected[name].shape and torch.isfinite(tensor).all(),'Finite matching source tensor:'+name)
            maps[role]=weights;del saved;gc.collect();check()
        adapters=[name for name in expected if name.endswith(('lora_A','lora_B'))];heads=[name for name in expected if name not in adapters]
        require(len(adapters)==192 and len(heads)==15,'Exact192 encoder/15 head partition required')
        report['partition']={'encoder_names':adapters,'head_names':heads,'encoder_tensors':192,'head_tensors':15}
        original_inputs=benchmark.inputs_for
        def guarded_inputs(*a,**kw):check();return original_inputs(*a,**kw)
        benchmark.inputs_for=guarded_inputs
        for tag,encoder_role,head_role in (('C_latest_encoder_best_heads','latest','best'),('D_best_encoder_latest_heads','best','latest')):
            phase[0]=tag;check()
            with torch.no_grad():
                for name,p in expected.items():p.copy_(maps[encoder_role if name in adapters else head_role][name])
            model.eval();random.seed(config['seed']);torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed'])
            result={};paths={}
            for language in ('de','pt'):
                original_data=benchmark.DATA;benchmark.DATA=DATA/'portuguese_retention_v1' if language=='pt' else DATA
                result_tag=args.output.stem+'_'+tag+'_'+language
                require(not (HERE/'results'/result_tag).exists(),'Fresh prediction namespace required')
                try:score=benchmark.evaluate(model,processor,result_tag,languages=(language,),splits=('validation',))
                finally:benchmark.DATA=original_data
                require(score['validation_'+language]['asr']['utterances']==50,'Exactly fixed50 each')
                result[language]=score['validation_'+language]['asr'];folder=HERE/'results'/result_tag
                paths[language]={key:record(folder/name) for key,name in (('summary','summary.json'),('metadata','metadata.json'),('predictions','validation_'+language+'.jsonl'))}
            report['hybrids'][tag]={'encoder':encoder_role,'heads':head_role,'ASR':result,'artifacts':paths}
            write(args.output,report);print(json.dumps({'hybrid':tag,'word_errors':{k:v['word_errors'] for k,v in result.items()}}),flush=True)
            torch.cuda.empty_cache()
        report.update(status='complete',passed=True,actual_GPU_decode=True,actual_diagnostic_utterances=200,
            interpretation_limit='Two conditional hybrids localize component effects at fixed weights; interactions/finite sample prevent a universal causal capacity claim',
            peak_allocated_GPU_bytes=torch.cuda.max_memory_allocated(),peak_reserved_GPU_bytes=torch.cuda.max_memory_reserved())
    except BaseException as error:
        report.update(status='failed',passed=False,error=repr(error));raise
    finally:
        stop.set();thread.join(7);report.update(finished_at_utc=now(),elapsed_seconds=time.monotonic()-started,
            guard_observations=list(observations),guard_thread_joined=not thread.is_alive(),resource_stop=failure)
        write(args.output,report)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-seconds',type=float,default=600);parser.add_argument('--validate-only',action='store_true')
    parser.add_argument('--gpu-window-authorized',action='store_true');main(parser.parse_args())
