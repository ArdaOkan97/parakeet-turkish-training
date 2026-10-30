"""Lightweight completion seal for the single actual170600 metadata observer."""
import hashlib,json,os,shutil,sys
from pathlib import Path
import psutil
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from resource_watchdog_v2 import sample_memory
F=Path(__file__).resolve().parent;A=F.parent/'resident_RAM_training_actual_25';OUT=A/'actual_saved170600_recovery_01'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def write(p,v):
 with Path(p).open('x',encoding='utf-8',newline='\n')as f:json.dump(v,f,indent=2);f.write('\n')
def who(p):return {'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()}
def dead(w):
 try:return abs(psutil.Process(w['pid']).create_time()-w['create_time'])>=.001
 except psutil.NoSuchProcess:return True
psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'and Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad'))
sample=sample_memory(os.getpid(),'source33_recovery_metadata_completion');m=sample['system_memory']
assert sample['paired_sampling_within_one_second']and sample['worker_private_memory']['private_bytes']<536870912
assert m['commit_available_bytes']>=2805306368 and m['physical_available_bytes']>=2305306368 and shutil.disk_usage('E:/').free>=100000000000
statuspath=OUT/'actual_recovery_status.json';status=read(statuspath)
assert status['passed']is True and status['new70_count']==70 and status['new70_range']==[170401,170470]
segment=status['new70_segment'];assert segment['start_offset']==344297457 and segment['end_offset']==344435096
assert segment['bytes']==137639 and segment['sha256']=='229d9e261c9def1a36c42fe78087aa47fef4e4f8f1e421bf13953f995d215380'
deaths=[{'identity':status[k],'exact_dead':dead(status[k])}for k in ('sealer_identity','sealer_launcher')]
assert all(x['exact_dead']for x in deaths)
snapshot=Path(status['actual70_snapshot']['path']);assert pin(snapshot)==status['actual70_snapshot']
tool=OUT/'actual_observer_TOOL_v1.json';write(tool,{'kind':'ASR18_source33_actual170600_recovery_observer_TOOL_v1',
 'actual_TOOL_exit_code':0,'actual_TOOL_chunk':'d8d948','actual_TOOL_wall_seconds':1.0735561,'status':pin(statuspath),
 'no_retry_or_training_change':True})
rootdirect=F/'ROOT_source33_saved170400_direct70_new_positive_verification_v1.json'
completion={'kind':'ASR18_actual_source33_recovery_metadata_completion_v1','status':pin(statuspath),'actual_TOOL':pin(tool),
 'observer_scope':pin(F/'ROOT_source33_natural170600_eighth_recovery_metadata_observer_scope_v1.json'),
 'observer_exact_deaths':deaths,'ROOT_closed_direct70':pin(rootdirect),
 'ROOT_closed_initial_boot7':pin(F/'ROOT_source33_initial_boot_and_actual7_configuration_binding_v1.json'),
 'same_new70_segment_and_snapshot':segment,'new70_snapshot':pin(snapshot),
 'corrected170470_row6_excerpt':status['evidence']['actual170470_row6'],
 'new170600_archive_pin_reused_without_reread':status['evidence']['standard170600_preparation_archive'],
 'no_original_journal_archive_tensor_audio_reread_in_completion':True,'no_fit_speed_saving_prevention_or_recurrent_fix_success_claim':True,
 'completion_sealer':who(psutil.Process()),'completion_sealer_launcher':who(psutil.Process().parent()),
 'completion_pair_live_at_write_exact_deaths_checked_after_return':True,'completion_source':pin(__file__),'sample':sample,
 'owned_training_session':65067,'child_source_environment_reader_and_resource_policies_unchanged':True,
 'ROOT_actual_recovery_acceptance_pending':True}
p=OUT/'metadata_completion_v1.json';write(p,completion)
print(json.dumps({'completion':pin(p),'actual_TOOL':pin(tool),'observer_deaths':deaths,
 'completion_sealer':completion['completion_sealer'],'completion_launcher':completion['completion_sealer_launcher']}),flush=True)
