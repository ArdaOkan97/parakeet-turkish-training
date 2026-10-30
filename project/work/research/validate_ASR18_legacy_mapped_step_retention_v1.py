"""Actual legacy checkpoint COW-mapping ownership, without ASR or GPU loads."""
import argparse, gc, json, os, subprocess, sys, time
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[2];ROOT=Path(r'E:\gpu_training_scratchpad')
sys.path.insert(0,str(PROJECT/'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import write_json
from bpe_broad_audio_v1 import record,require
from resource_watchdog_v2 import sample_memory

def arm(mode,folder):
    folder.mkdir(exist_ok=False)
    numpy_path=PROJECT/'work/training-env/Lib/site-packages'
    sys.path.insert(0,str(numpy_path));import numpy;sys.path.remove(str(numpy_path))
    import torch
    require(torch.__version__.endswith('+cpu') and not torch.cuda.is_available(),'Actual isolated CPU-only Torch required')
    torch.set_num_threads(1)
    config=json.loads((PROJECT/'outputs/parakeet_turkish/bpe_broad_production_v6.disabled.json').read_text())
    binding=config['coherent_diagnostic_parent']['checkpoint'];require(record(binding['path'])==binding,'Actual immutable legacy checkpoint1 required')
    samples=[]
    def observe(stage):
        gc.collect()
        item=sample_memory(os.getpid(),stage)
        require(item['worker_private_memory']['available'] and item['system_memory']['available'],'Paired actual process/system memory required')
        samples.append(item);return item['worker_private_memory']['private_bytes']
    baseline=observe('before_legacy_mmap_load')
    saved=torch.load(binding['path'],map_location='cpu',weights_only=True,mmap=True)
    loaded=observe('after_legacy_mmap_load')
    # Each scalar is a tiny view, but its storage may retain the whole mapping.
    steps=[state['step'] for partition in saved['optimizer']['optimizers'] for state in partition['state'].values()]
    require(steps and all(step.numel()==1 and step.dtype==torch.float32 and step.device.type=='cpu' for step in steps),'Actual CPU float32 Adam clocks required')
    count=len(steps);tensor_bytes=sum(v.numel()*v.element_size() for v in steps)
    clocks=[step.item() for step in steps]
    if mode=='cloned':steps=[step.clone() for step in steps]
    del saved;gc.collect()
    held=observe('after_drop_archive_except_'+mode+'_steps')
    for step in steps:step.add_(1)
    del step
    written=observe('after_mutable_CPU_steps_increment')
    require([step.item() for step in steps]==[clock+1 for clock in clocks],'Actual step writes differ')
    del steps;gc.collect()
    released=observe('after_drop_all_steps')
    require(record(binding['path'])==binding,'Copy-on-write control modified original checkpoint')
    write_json(folder/'report.json',{'passed':True,'mode':mode,'CPU_only':True,'PID':os.getpid(),
        'source':record(__file__),'checkpoint':binding,'CPU_step_tensors':count,'CPU_step_tensor_bytes':tensor_bytes,
        'baseline_private_bytes':baseline,'loaded_private_bytes':loaded,'held_private_bytes':held,
        'after_step_write_private_bytes':written,'released_private_bytes':released,
        'held_minus_released_private_bytes':held-released,'original_checkpoint_SHA_unchanged':True,
        'torch_version':str(torch.__version__),'default_mmap_options':torch.serialization.get_default_mmap_options(),
        'samples':samples,'GPU_or_ASR_loaded':False})

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--arm',choices=('borrowed','cloned'));a=p.parse_args()
    folder=Path(a.output).resolve();require(folder.is_relative_to(ROOT.resolve()),'All new retention evidence E-only')
    if a.arm:return arm(a.arm,folder)
    folder.mkdir(exist_ok=False);results=[]
    import psutil
    for mode in ('borrowed','cloned'):
        guard=folder/(mode+'_guard');target=folder/mode
        command=[sys.executable,'-X','utf8','-B',str(PROJECT/'work/research/ASR18_bounded_CPU_phase_v3.py'),
            '--folder',str(guard),'--seconds','60','--script',str(Path(__file__).resolve()),'--','--output',str(target),'--arm',mode]
        with (folder/(mode+'.log')).open('x',encoding='utf-8') as log:
            child=subprocess.Popen(command,cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT)
            code=child.wait(timeout=70)
        require(code==0,'Actual isolated CPU retention arm failed')
        boot=json.loads((guard/'boot.json').read_text());closure=json.loads((guard/'closure.json').read_text())
        require((boot['pid']==child.pid or boot['ppid']==child.pid) and not psutil.pid_exists(boot['pid'])
            and closure['actual_exit_code']==0 and closure['monitor_clean'] and closure['observed']['samples']>=3,
            'Actual owned CPU arm closure/guard required')
        result=json.loads((target/'report.json').read_text());require(result['passed'],'Actual complete arm required')
        results.append({'mode':mode,'actual_exit_code':code,'actual_worker_PID':boot['pid'],'owned_handle_PID':child.pid,
                        'parent_observed_wait':True,'report':record(target/'report.json'),'result':result,
                        'sampled_private_peak_bytes':closure['observed']['peak_private_bytes']})
    borrowed,cloned=[value['result'] for value in results]
    difference=borrowed['held_minus_released_private_bytes']-cloned['held_minus_released_private_bytes']
    write_json(folder/'report.json',{'passed':True,'CPU_only':True,'source':record(__file__),'results':results,
        'only_live_CPU_steps_controlled':True,'same_actual_checkpoint_and_CPU_runtime':True,
        'retained_private_difference_bytes':difference,
        'whole_legacy_archive_retention_supported':difference>.9*borrowed['checkpoint']['bytes'],
        'historical_78852_attribution_claimed':False,'full_integrated02_growth_attribution_claimed':False,
        'next_control':'sameprocess multiupdate with NEWstreamingcheckpoint/READ_ONLYmapping/ownedCPUstepclones',
        'production_lifecycle_choice_made':False})
    print(json.dumps({'passed':True,'retained_private_difference_bytes':difference,
        'borrowed_held_minus_released':borrowed['held_minus_released_private_bytes'],
        'cloned_held_minus_released':cloned['held_minus_released_private_bytes']}))

if __name__=='__main__':main()
