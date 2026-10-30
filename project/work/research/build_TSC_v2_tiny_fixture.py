"""Make one NEW focused v2 routing fixture from the reviewed v1 fixture."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'work/research'
source=(HERE/'TSC_same_science_tiny_check_v1.py').read_text(encoding='utf-8')
for name in ('run_hf_TSC_arm','resume_TSC_same_science','launch_TSC_same_science_resume','supervise_TSC_same_science'):
    source=source.replace(name+'_v1',name+'_v2')
for old,new in {'train_lora_batched_tsc_v1':'train_lora_batched_tsc_v2','launch_native_training_tsc_v1':'launch_native_training_tsc_v2',
    'launch_with_memory_guard_tsc_v1':'launch_with_memory_guard_tsc_v2','launch_with_cache_reclaim_tsc_v1':'launch_with_cache_reclaim_tsc_v2',
    'memory_guard_training_launcher_tsc_v1':'memory_guard_training_launcher_tsc_v2','memory_guard_training_launcher_TSC_resume_v1':'memory_guard_training_launcher_TSC_resume_v2',
    'cache_reclamation_training_launcher_tsc_v1':'cache_reclamation_training_launcher_tsc_v2','atomic_io_retry_tsc_v1':'atomic_io_retry_tsc_v2',
    'isolated_HF_TSC_guarded_quality_arm_v1':'isolated_HF_TSC_guarded_quality_arm_v2','TSC_same_science_exact_resume_tiny_verification_v1':'TSC_same_science_exact_resume_tiny_verification_v2',
    'len(chain)==21':'len(chain)==22','Pinned original 21 science sources':'Pinned original 22 v2 science sources'}.items():source=source.replace(old,new)
source=source.replace("save(source,{'config_identity':{'training_arguments':recipe}})",
    "save(source,{'config_identity':{'training_arguments':{**recipe,'sampling_mode':'domain30','fleurs_fraction':.3,'replay_seed':20261012}}})")
needle="for key,path in assets.items():save(path,{'source_run_config':exact.record(source)} if key=='source_closure' else {'synthetic_only':True})"
replacement=needle+"\n            quarantine=d/'quarantine.json';save(quarantine,{'synthetic_only':True})\n            save(assets['tsc_input_proof'],{'synthetic_only':True,'references':{'quarantine_proof':exact.record(quarantine)}})"
assert needle in source;source=source.replace(needle,replacement)
needle="'TSC_input_proof':exact.record(assets['tsc_input_proof']),'TSC_training_approval':exact.record(assets['tsc_training_approval'])}"
replacement="'TSC_input_proof':exact.record(assets['tsc_input_proof']),'TSC_training_approval':exact.record(assets['tsc_training_approval']),'TSC_quarantine_proof':exact.record(quarantine)}"
assert needle in source;source=source.replace(needle,replacement)
with (HERE/'TSC_same_science_tiny_check_v2.py').open('x',encoding='utf-8',newline='\n') as stream:stream.write(source)
print('Prepared v2 small fixture; execution waits for final new source pins.')
