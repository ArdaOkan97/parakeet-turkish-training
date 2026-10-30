"""Pin at most eight cached-only future review examples at a closed boundary."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(r'E:\gpu_training_scratchpad')
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scratch/CPU_validation_v2_env/Lib/site-packages'))
import cpu_preparation_guard_v2 as base
from run_large_v3_sanity_cpu_v6 import hard_process_memory_job

SEED = 2026101611
KOREAN_ALARM = 'c3b4607fd8e60c557d5815621b4f192b_00074'
AREA = ROOT/'analysis/CPU_validation_v11'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_cached(path, guard):
    """Pace the small, already cached text inputs without opening any audio."""
    data = bytearray()
    with path.open('rb') as source:
        while True:
            guard.wait()
            time.sleep(65536/1048576)
            block = source.read(65536)
            if not block:
                break
            data.extend(block)
            if len(data) > 10_000_000:
                raise RuntimeError('Declared 10MB metadata input bound reached')
    return bytes(data)


def rank(row, group):
    return sha(f'{SEED}|{group}|{row["segment_id"]}'.encode('utf-8'))


def select(rows, group, count):
    assert len(rows) >= count
    return [(group, row) for row in sorted(rows, key=lambda row: (rank(row, group), row['segment_id']))[:count]]


def main():
    base.AREA = AREA
    guard = base.Guard('future_sanity8_cached_metadata_v11')
    guard.wait()
    job = hard_process_memory_job()
    boundary = ROOT/'analysis/CPU_validation_v10/actual_next10k_reference_CLOSED0_v10.json'
    assert json.loads(boundary.read_text())['exit_code'] == 0
    v8 = ROOT/'analysis/CPU_validation_v8'
    manifest_path = v8/'new256_tiny_primary_manifest_v8.jsonl'
    ledger_path = v8/'new256_tiny_primary_ledger_v8.jsonl'
    assert manifest_path.stat().st_size+ledger_path.stat().st_size < 10_000_000
    manifest_raw, ledger_raw = read_cached(manifest_path, guard), read_cached(ledger_path, guard)
    assert sha(manifest_raw) == '75124a49c199a94dfbee14c4a78e50c167066cd2763aad795011df2d2f07cddc'
    assert sha(ledger_raw) == 'ba6c74f1ea22e803e168261c134966fb7b0885f6691a459c5f659a7f2ba8865a'
    assert len(manifest_raw)+len(ledger_raw) < 10_000_000
    supplied = {row['segment_id']: row for row in map(json.loads, manifest_raw.decode('utf-8').splitlines())}
    rows = list(map(json.loads, ledger_raw.decode('utf-8').splitlines()))
    assert len(rows) == len(supplied) == 256
    uncertain = [row for row in rows if row['full_clip_primary']['uncertain_under70']]
    retained = [row for row in rows if row['retains_primary_language_screen_eligibility']]
    window_flagged = [row for row in retained if row['supplemental_any_confident_nontr'] or row['supplemental_any_uncertain_under70']]
    clean = [row for row in retained if row not in window_flagged]
    korean = next(row for row in rows if row['segment_id'] == KOREAN_ALARM)
    assert korean['full_clip_primary']['detected_language'] == 'ko'
    assert korean['full_clip_primary']['confident_nontr']
    assert (len(uncertain), len(window_flagged), len(clean)) == (15, 5, 235)
    chosen = [('fixed_short_confident_nontr_alarm_requires_listening', korean)]
    chosen += select(uncertain, 'seeded_fullclip_uncertain_control', 3)
    chosen += select(window_flagged, 'seeded_primary_pass_with_supplemental_window_alarm', 2)
    chosen += select(clean, 'seeded_primary_pass_control', 2)
    assert len(chosen) == len({row['segment_id'] for group, row in chosen}) == 8
    output = []
    for group, row in chosen:
        selected = dict(supplied[row['segment_id']])
        selected.update(
            QC_group='future_sanity8_from_tiny_v8_metadata_v11',
            future_review_group=group,
            future_review_selection_seed=SEED,
            future_review_seeded_rank=None if row['segment_id'] == KOREAN_ALARM else rank(row, group),
            historical_tiny_full_clip_primary=row['full_clip_primary'],
            historical_primary_screening_category=row['primary_screening_category'],
            historical_retains_primary_language_screen_eligibility=row['retains_primary_language_screen_eligibility'],
            historical_supplemental_window_observations=row['supplemental_window_observations'],
            historical_supplemental_any_confident_nontr=row['supplemental_any_confident_nontr'],
            historical_supplemental_any_uncertain_under70=row['supplemental_any_uncertain_under70'],
            raw_tiny_flags_preserved=True,
            selection_uses_cached_tiny_groups=True,
            future_transcription_outputs_read_or_generated=False,
            audio_read_or_decoded_for_this_selection=False,
            accepted_TRAIN=False, source_label_corrected=False,
            source_dataset_revision='8b0fdd976ed9668b64f5ee0696317a8f241ce39c',
            audio_dataset_revision='81d802f45a1c1e1f5331260484dacf1fbea81981',
        )
        output.append(selected)
    target = AREA/'future_sanity8_from_tiny_v8_manifest_v11.jsonl'
    raw = ''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in output).encode('utf-8')
    if target.exists():
        assert target.read_bytes() == raw
    else:
        target.write_bytes(raw)
    receipt = {
        'kind':'future_sanity8_metadata_pin_v11', 'pinned_at_utc':base.now(),
        'source':str(Path(__file__)), 'source_SHA256':sha(Path(__file__).read_bytes()),
        'closed_reference_boundary':str(boundary), 'closed_reference_boundary_SHA256':sha(boundary.read_bytes()),
        'selection_seed':SEED, 'selection_method':'Ascending SHA256(seed|group|segment_id) within declared cached Tiny groups; fixed Korean alarm included separately',
        'source_manifest':str(manifest_path), 'source_manifest_SHA256':sha(manifest_raw),
        'source_ledger':str(ledger_path), 'source_ledger_SHA256':sha(ledger_raw),
        'manifest_path':str(target), 'manifest_SHA256':sha(raw), 'clips':8,
        'duration_s':sum(row['duration_s'] for row in output),
        'group_counts':{group:sum(g == group for g, row in chosen) for group, row in chosen},
        'IDs':[row['segment_id'] for row in output],
        'not_a_random256_accuracy_estimate':True,
        'selection_conditioned_on_existing_tiny_predictions':True,
        'no_new_audio_model_download_or_inference':True,
        'ROOT_owns_any_later_audio_or_GPU_review_admission':True,
        'threshold70_is_uncalibrated_screening_not_truth':True,
        'raw_flags_or_primary_eligibility_changed':False,
        'accepted_TRAIN_hours':0,
    }
    base.write_json(AREA/'future_sanity8_from_tiny_v8_receipt_v11.json', receipt)
    guard.status(phase='metadata_only_future_sanity8_PINNED', manifest_SHA256=sha(raw), clips=8)
    print(json.dumps({'manifest':str(target),'SHA256':sha(raw),'groups':receipt['group_counts'],'duration_s':receipt['duration_s']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
