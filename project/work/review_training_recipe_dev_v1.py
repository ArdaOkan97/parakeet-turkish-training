"""Short stdlib-only source/DEV recipe snapshot; never reads TEST predictions."""
from pathlib import Path
import hashlib
import json
import math
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/parakeet_turkish'
RUNS = OUT / 'runs'
RESULTS = OUT / 'results'
FILES = {}


def read(path):
    data = path.read_bytes()
    FILES[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def dev_summary(tag):
    path = RESULTS / tag / 'summary.json'
    if not path.exists():
        return None
    value = read(path)
    dev = {lang: value[f'validation_{lang}']['asr'] for lang in ('en','es','it','fr','de','tr')}
    if not all(row['utterances'] == 50 for row in dev.values()):
        raise ValueError('Expected only the fixed 50-per-language completed DEV')
    return dev


baseline = dev_summary('base_v3')
runs = {}
names = (
    'adaptive_20261010_continuation_03_resume_B8',
    'adaptive_20261011_native_data_01_public_rank8_control',
    'adaptive_20261011_native_memory_fix_01_native_resume',
    'adaptive_20261011_native_memory_fix_01_public_rank8_replay25_screen',
    'adaptive_20261011_native_memory_fix_01_public_winner_extension',
)
for name in names:
    config = read(RUNS / name / 'run_config.json')
    history = read(RUNS / name / 'training_history.json')
    latest = history[-1]
    evaluations = {}
    for step in [0] + [row['step'] for row in history if 'validation' in row]:
        summary = dev_summary(f'{name}_step_{step}')
        if summary is None:
            continue
        old = {}
        for lang in ('en','es','it','fr','de'):
            now = summary[lang]
            previous = baseline[lang]
            tolerance = max(.005, .1 * previous['wer'])
            old[lang] = {
                'word_errors': now['word_errors'], 'baseline_word_errors': previous['word_errors'],
                'reference_words': now['reference_words'],
                'extra_errors': now['word_errors'] - previous['word_errors'],
                'wer_delta': now['wer'] - previous['wer'], 'allowed_wer_delta': tolerance,
                'remaining_word_error_margin': (previous['wer'] + tolerance - now['wer']) * now['reference_words'],
                'passed': now['wer'] <= previous['wer'] + tolerance + 1e-12,
            }
        evaluations[step] = {'turkish': summary['tr'], 'old_languages': old,
                             'retention_passed': all(row['passed'] for row in old.values())}
    total = latest['audio_seen_seconds']
    tr_seconds = latest['turkish_audio_seen_seconds']
    kd = latest['replay_kl'] * config['replay_kl'] + latest['replay_duration_kl'] * config['duration_kl'] + latest['replay_feature_distillation'] * config['feature_distillation']
    runs[name] = {
        'config': {key: config.get(key) for key in (
            'steps','epochs','replay_fraction','head_lr','lora_lr','new_token_lr_multiplier',
            'replay_kl','duration_kl','feature_distillation','warmup_steps','schedule',
            'rank','alpha','batch_size','accumulation','tr_clips_per_update','replay_clips_per_update',
            'turkish_clips_used','turkish_unique_pool_hours','replay_clips_used','replay_unique_pool_hours',
            'early_stop_patience','max_retention_failures','warm_start','teacher')},
        'latest_completed_update': latest['step'],
        'actual_turkish_clips_drawn': latest['turkish_clips_seen'],
        'actual_unique_turkish_clips_seen': latest['turkish_unique_clips_seen'],
        'actual_turkish_clip_epochs': latest['turkish_epoch'],
        'actual_turkish_audio_hours': tr_seconds / 3600,
        'actual_replay_audio_hours': (total - tr_seconds) / 3600,
        'actual_turkish_audio_fraction': tr_seconds / total,
        'latest_loss': {key: latest[key] for key in ('loss','tdt_loss','replay_kl','replay_duration_kl','replay_feature_distillation','gradient_norm')},
        'latest_weighted_preservation_loss': kd,
        'latest_preservation_fraction_of_loss': kd / latest['loss'],
        'replay_language_clip_counts': latest.get('replay_language_clip_counts'),
        'development': evaluations,
    }

control = read(RUNS / names[2] / 'training_history.json')
extension = read(RUNS / names[4] / 'training_history.json')
first = min(250, len(control), len(extension))
same_samples = all(a['sampled_source_ids'] == b['sampled_source_ids'] for a,b in zip(control[:first], extension[:first]))
first_control_metadata = read(RESULTS / f'{names[2]}_step_0' / 'metadata.json')
first_extension_metadata = read(RESULTS / f'{names[4]}_step_0' / 'metadata.json')

source_locations = {
    'teacher_original_before_extension_and_resume': ['train_lora_batched_data_v9.py',286,288],
    'ungrouped_roundtrip_and_replay_old_IDs': ['train_lora_batched_data_v9.py',301,305],
    'all_head_and_attention_adapter_Adam_groups': ['train_lora_batched_data_v9.py',325,330],
    'literal_blank_SOS_and_padding_valid_target_lengths': ['batched_lora_tr_metadata_v1.py',128,148],
    'replay_only_teacher_old_token_KL_and_moved_durations': ['batched_lora_tr_metadata_v1.py',166,209],
    'raw_none_then_exactly_one_target_length_normalization': ['batched_lora_tr_metadata_v1.py',250,259],
    'native_kernel_valid_lexical_targets_exclude_blank': ['native_tdt_checkpointed_v2.py',154,180],
    'native_full_logits_sigma0_and_bound_recompute_closure': ['native_tdt_checkpointed_v2.py',184,232],
    'new_six_rows_only_actual_Adam_delta_multiplier': ['train_lora_batched_data_v9.py',532,538],
    'plateau_patience_counts_only_full_TR_epochs': ['train_lora_batched_data_v9.py',477,483],
}
for filename in set(row[0] for row in source_locations.values()) | {'extend_vocab.py','encoder_lora.py'}:
    path = OUT / filename
    FILES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()

report = {
    'schema':1, 'kind':'training_recipe_source_and_DEV_review_v1',
    'created_at_utc': datetime.now(timezone.utc).isoformat(),
    'CPU_only':True, 'Torch_NeMo_model_imports':False, 'GPU_execution':False,
    'TEST_predictions_inspected':False, 'training_source_or_run_mutation':False,
    'snapshot_warning':'Run history was a completed on-disk snapshot; this is not an assertion of later live state.',
    'runs':runs,
    'same_initial_and_repeated_control_prefix': {
        'updates_compared':first,'sampled_source_ids_exactly_equal':same_samples,
        'initial_model_state_sha256_equal': first_control_metadata['model_fingerprint']['identity']['model_state_sha256'] == first_extension_metadata['model_fingerprint']['identity']['model_state_sha256'],
        'initial_model_state_sha256':first_extension_metadata['model_fingerprint']['identity']['model_state_sha256'],
        'interpretation':'New-optimizer extension restarts from selected step0 and repeats the control sampler prefix. This is deliberate winner warm-start behavior, not exact continuation of latest optimizer/cursor.'},
    'source_findings':[
        'No target/SOS/blank/duration indexing error found: blank8192 retained, lexical width8199 includes six appended8193..8198, durations8199..8203; decoder receives [blank]+labels while raw target lengths omit SOS/padding.',
        'Native2 computes raw per-clip NLL with full lexical normalization and sigma0/omega0; collator divides each raw NLL by target length once, then averages clips/accumulation. Different published NeMo settings are a new scientific arm, not a correction to duplicate normalization.',
        'Teacher heads are copied from pinned base before extension and source loading; only genuine replay frames/old target IDs enter KD. Turkish KD is zero. Student old-token KL normalizes over all8199 labels, appropriately penalizing new labels on replay.',
        'All24 rank8 attention adapters plus15 head tensors remain trainable (207 tensors). New six embedding/head rows use actual Adam delta multiplier6; moved duration rows are outside this slice. No zero-gradient assertion is made by this CPU source review.',
        'Replay fraction is a clip fraction. Actual50%-TR-clip sampling supplies only~21%TR audio;75%-TR-clip supplies~45%TR audio, so short equal250-update screens are not equal TR exposure.',
        'Plateau patience already advances only at completed TR clip epochs. Retention early stops remain immediate after the configured consecutive failures; do not waive the Spanish gate.'
    ],
    'source_locations':source_locations,
    'recommendation':{
        'first_controlled_change':'Give the75%-TR-clip recipe substantially more Turkish exposure with the same rank8/alpha16, heads5e-5, LoRA3e-4, newrows multiplier6, native loss and KD coefficients. Keep50%control available; select only checkpoints passing all five DEV gates.',
        'equal_TR_exposure_updates':{str(fraction):{'TR75_clip_updates':math.ceil(fraction*74883/12),'TR50_clip_updates':math.ceil(fraction*74883/8)} for fraction in (.25,.5,1.)},
        'continuation_policy':'For a chosen same-config arm, resume its latest Adam/sampler/RNG rather than repeatedly restarting source1871 and its first2000 TR recordings. Retain its best qualifying checkpoint independently for selection.',
        'retention_risk':'Inherited source1871 Spanish is already6 word errors above original with<1error gate headroom. If sustained75%-TR updates lose Spanish, a targeted Spanish replay weight/teacher penalty or lower shared-head LR is a separate next arm; avoid simultaneous LR/rank/vocabulary changes.',
        'LR_change_now':'No evidence for increasing shared-head LR: Turkish loss is finite/falling and inherited Spanish margin is tight. Keep initial LR for exposure comparison; targeted-head LR reduction is a fallback if retention is the limiting factor.',
        'accuracy_limit':'~65%TR DEV WER remains poor and far from the verified strong-baseline target; qualified retention/improvement does not mean competitive or usable Turkish ASR.',
        'evidence_limits':'No per-component gradient-norm diagnosis, new GPU test, full NeMo Lightning equivalence, or causal proof that extra epochs will close the gap. Short DEV differences of1–7 word errors do not prove saturation.'
    },
    'input_files_sha256':FILES,
}
target = OUT / 'training_recipe_dev_review_v1.json'
if target.exists():
    raise FileExistsError(target)
target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'report':str(target),'runs':{name:{'step':value['latest_completed_update'],'TR_audio_fraction':value['actual_turkish_audio_fraction'],'TR_clip_epoch':value['actual_turkish_clip_epochs']} for name,value in runs.items()},'same_prefix':same_samples},indent=2))
