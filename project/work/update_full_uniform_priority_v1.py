from pathlib import Path
from datetime import datetime, timezone
import json

root = Path(__file__).resolve().parents[1]
out = root / 'outputs/parakeet_turkish'
path = out / 'experiment_plan.json'
plan = json.loads(path.read_text(encoding='utf-8'))
archive = root / 'work/experiment_plan_before_full_uniform_priority_20261012.json'
if not archive.exists():
    archive.write_bytes(path.read_bytes())
items = {x['id']: x for x in plan['ordered_experiments']}
plan['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
plan['latest_user_priority'] = 'Complete a uniform pass over all available Turkish TRAIN data with the best actually tested recipe before further short controls; add Portuguese retention.'
plan['retention_languages'] = ['en', 'es', 'it', 'fr', 'de', 'pt']
plan['evaluation_languages'] = ['tr', 'en', 'es', 'it', 'fr', 'de', 'pt']
plan['Portuguese_gate_status'] = 'Separate original-v3 baseline and BEST6500 Portuguese DEV50 measurement pending; original six-language baseline remains immutable.'
plan['status_authority'] = 'Actual controller/native closure and complete fixed DEV results determine status. New full run requires seven DEV sets; historical trials retain their original six-set scope.'
policy = plan['automatic_execution_policy']
policy['after_trial'] = 'Review all seven complete DEV sets for the new full run, actual LR, throughput and new-pass unique coverage. Historical trials are scored against the original six sets. Preserve the qualifying champion.'
policy['promising_recipe'] = 'User now prioritizes a full uniform pass from the already tested original8199 BEST6500. Do not wait for new token-control improvement and do not restart short screens. Continue exact weights/Adam/RNG/history and declared sampler migration through all253798 unique Turkish clips, with DEV250.'
policy['full_data_priority'] = 'Full uniform pass first; future ideas only after its actual complete coverage and learning-curve review. Retention/resource failures require cause repair and explicit continuation review, never unchanged retries.'
policy['Turkish_sampling'] = 'Uniform shuffled Turkish TRAIN clips; natural2037/253798 FLEURS share. No30% FLEURS quota. Historical domain30 draws remain separate lineage.'
policy['Portuguese'] = '319 public FLEURS TRAIN replay clips plus separate fixed50DEV and100reservedTEST; original five replay boundary states preserved. Fresh original-v3 Portuguese baseline required before full training.'
item = items['04']
item.update(status='preparing_full_data_priority', code_ready=True,
    recipe='Original8199 BEST6500: all207 learned weights and Adam moments retained; heads LR1e-5; encoder LoRA r8/alpha16/all24 attention layers LR3e-4; native NeMo4; physicalB8 accumulation2; replay50%, KL10/durationKL0.5/features1. Uniform shuffled full253798 Turkish clips; add319 Portuguese TRAIN replay clips. DEV every250 and at complete unique coverage. No fresh optimizer or short screen.',
    entry_rule='Immediate next GPU training run after actual Portuguese baseline/source gate and changed host-memory admission. The user explicitly chose the best existing recipe for a full data pass.',
    source_checkpoint='work/closed-sources/TSC_headlr_BPE512_6500_6844_main_v1/best_checkpoint.pt',
    source_checkpoint_sha256='df299dd5e1918f37fc1fb83fe7b428b458daf10a759a021cb6fae07c4eb71a16',
    source_step=6500, new_uniform_pass_updates=31725, new_uniform_end_step=38225,
    full_unique_TR_required=253798, final_extra_next_cycle_draws=2,
    actual_full_pass_completed=False, actual_GPU_training_started=False,
    prepared_worker='train_lora_batched_tsc_uniform_full_v1.py',
    prepared_boundary='tsc_uniform_full_transition_v1.py',
    prepared_sampler='tsc_uniform_full_sampler_v1.py',
    pending=['Portuguese actual data preparation and base/source DEV50', 'Full worker/controller source review and actual state admission', 'Changed host memory policy and actual whole-update fit'])
items['03'].update(status='partial_training_resource_stop_deferred',
    actual_updates=10, actual_memory04_updates=10, current_family_smoke_passed=True,
    actual_smoke_passed=True, actual_GPU_trial_completed=False, full1000_claimed=False,
    actual_ROOT_queue_session=50137, actual_queue_ROOT_closed_session=50137,
    actual_queue_ROOT_closed_exit_chunk='a260c4',
    actual_controller_native_closure='Native0 with saved checkpoint/complete DEV; controller and queue1 under system-commit90% guard at step7010.',
    deferred_reason='User requests full uniform training with best tested original recipe before more diagnostic screens.',
    actual_GPU_training_running=False)
for key in ('05', '06', '07', '08', '09', '10'):
    items[key]['execution_priority'] = 'After actual full uniform data pass and its DEV review'
    items[key]['deferred_reason'] = 'Latest explicit user full-data priority'
items['07']['recipe'] = items['07']['recipe'].replace('allfive', 'all six retention')
items['09']['recipe'] = 'After the complete uniform Turkish pass, audit CV/TSC source errors and suspicious transcripts. Apply justified TRAIN-only filtering or mild augmentation as separate controls. Research additional public domains only if the learning curve and error analysis justify them. The user rejected the30% FLEURS repeat quota.'
items['11']['recipe'] = items['11']['recipe'].replace('allfive languages', 'all six retention languages')
plan['ordered_experiments'] = [items[x] for x in ('04','01','02','03','05','06','07','08','09','10','11')]
plan['completed_evidence'][-1] = 'ACTIVE15-minute recurring review remains enabled. The last local queue is actually CLOSED under the memory guard; full uniform run and Portuguese baseline are being prepared. No full expanded-data pass has completed.'
path.write_text(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
print(json.dumps({'updated':str(path),'first_item':plan['ordered_experiments'][0]['id'],'full_GPU_running':False,'retention_languages':plan['retention_languages']}))
