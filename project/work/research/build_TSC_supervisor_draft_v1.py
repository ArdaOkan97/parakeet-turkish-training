"""New source-only dynamic-horizon supervisor draft; no runtime/data imports."""
from pathlib import Path
import ast,hashlib
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'outputs/parakeet_turkish'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
source=(HERE/'supervise_domain30_same_science_v1.py').read_text(encoding='utf-8')
for old,new in [('run_hf_domain30_arm_v1','run_hf_TSC_arm_v1'),('resume_domain30_same_science_v1','resume_TSC_same_science_v1'),
 ('launch_domain30_same_science_resume_v1','launch_TSC_same_science_resume_v1'),
 ('isolated_HF_domain30_guarded_quality_arm_v1','isolated_HF_TSC_guarded_quality_arm_v1'),
 ('domain_same_run_exact_science_continuation_v1','TSC_same_run_exact_science_continuation_v1'),
 ('train_lora_batched_domain30_v1.py','train_lora_batched_tsc_v1.py'),
 ('launch_native_training_domain30_v1.py','launch_native_training_tsc_v1.py'),
 ('launch_with_memory_guard_domain30_v1.py','launch_with_memory_guard_tsc_v1.py'),
 ('original_domain_ready','original_TSC_ready'),('automatic_domain_same_science_window_supervisor_v1','automatic_TSC_same_science_window_supervisor_v1'),
 ('pinned_domain_controller','pinned_TSC_controller'),('minimum_domain_epochs_completed','uniform_combined_pool_epochs_completed'),
 ('minimum_domain_epochs_horizon','uniform_combined_pool_epoch_horizon'),('configured_domain_horizon','configured_TSC_horizon')]:source=source.replace(old,new)
source=source.replace('429daedee13ad14ee803c197eee597078940ef63525b093fe881c2af88ac772f',sha(HERE/'run_hf_TSC_arm_v1.py'))
source=source.replace('3a6b4f885b4fd4786c903f017408ad78d82bed518a7977a7068746dbe5cbfa87',sha(HERE/'resume_TSC_same_science_v1.py'))
source=source.replace('d192743f89fe4c0197267fe8701ddcc724229191f224e12baa9d80fa1dfcbb73',sha(HERE/'launch_TSC_same_science_resume_v1.py'))
source=source.replace('HORIZON=13009',"# Dynamic horizon is pinned by actual approved combined-pool config, never copied from another arm.")
source=source.replace("require(config.get('steps')==HORIZON and identity['training_arguments']['epochs']==1.,\n        'Preserve the actual one minimum-domain-pass13009-update horizon')", "require(config.get('steps')==math.ceil(config['turkish_clips_used']/8) and identity['training_arguments']['epochs']==1.,\n        'Preserve actual one combined-pool pass ceil(N/8) horizon')")
source=source.replace("'configured_optimizer_horizon_unchanged':HORIZON", "'configured_optimizer_horizon_unchanged':initial['configured_steps']")
source=source.replace("'scientific_config_identity_sha256':canonical_sha(science),\n        'controller_close_publish_order'", "'scientific_config_identity_sha256':canonical_sha(science),'configured_steps':state['configured_steps'],\n        'controller_close_publish_order'")
source=source.replace("closed['completed_steps']>=HORIZON", "closed['completed_steps']>=initial['configured_steps']")
source=source.replace("'scientific_config_identity_sha256':canonical_sha(science),'completed_steps'", "'scientific_config_identity_sha256':canonical_sha(science),'configured_steps':state['configured_steps'],'completed_steps'")
source=source.replace("    return chain['source_sha256']", "    require(sha(HERE/preload.RUNTIME_MEMORY_NAME)==preload.RUNTIME_MEMORY_SHA,'Exact-only memory admission source changed')\n    return chain['source_sha256']")
source=source.replace('unchanged original21-source domain chain','unchanged original21-source TSC science chain')
source=source.replace('a domain30 arm','an approved TSC combined-pool arm').replace('Original domain science','Original TSC science')
source=source.replace('domain-ready','TSC-ready').replace('original domain arm','original TSC arm').replace('SAME-run exact windows','SAME-run exact windows')
source=source.replace('first-domain','first-TSC').replace('First-domain','First-TSC').replace('first domain window','first TSC window')
source=source.replace('first-domain','first-TSC').replace('domain sampler','combined-pool sampler')
source=source.replace('Original immutable domain-ready','Original immutable TSC-ready').replace('Keep exact domain state','Keep exact TSC state')
source=source.replace('actual one minimum-domain','actual one combined-pool')
path=HERE/'supervise_TSC_same_science_v1.py'
if path.exists():raise FileExistsError(path)
ast.parse(source);path.write_text(source,encoding='utf-8',newline='\n');print(path.name,sha(path))
