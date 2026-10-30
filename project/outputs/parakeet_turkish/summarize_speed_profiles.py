"""Select an audited faster path without changing the pilot training files."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent

def read(name):
    return json.loads((OUTPUT/name).read_text(encoding='utf-8'))

def sha(name):
    return hashlib.sha256((OUTPUT/name).read_bytes()).hexdigest()

def main():
    old_name, new_name = 'batch_profile_original_b1_median.json', 'batch_profile_v2_b1_median.json'
    old, new = read(old_name), read(new_name)
    audit = read('batch_profile_b1_throughput_independent_audit.json')
    high = read('batch_profile_v2_b1_high.json')
    for report in (old, new, high):
        if report['status'] != 'passed' or not report['parity']['passed']:
            raise RuntimeError('A required GPU profile failed.')
    if not audit['passed']:
        raise RuntimeError('Independent median timing audit failed.')
    for key in ('weights_sha256','source_ids','audio_seconds_per_update','code_sha256','precision'):
        if old[key] != new[key]:
            raise RuntimeError('Compared profile identity differs: '+key)
    for report in (new, high):
        p = report['parity']
        if not all(g['relative_l2_difference'] < .005 and g['cosine_similarity'] > .99999
                   for g in p['gradient_groups'].values()):
            raise RuntimeError('Single-clip objective group gradient check failed.')
        if report['batch_size'] != 1 or report['accumulation'] != 4 or not report['encoder_checkpointing']:
            raise RuntimeError('Profile selection must describe the verified configuration.')
    before = old['timing']['end_to_end']
    after = new['timing']['end_to_end']
    if before['updates'] < 20 or after['updates'] < 20 or high['timing']['end_to_end']['updates'] < 10:
        raise RuntimeError('Insufficient timed updates.')
    ratio = after['audio_min_per_wall_min']/before['audio_min_per_wall_min']
    report = {
        'passed':True, 'throughput_improved':ratio > 1.1,
        'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'selected_configuration':{'batch_size':1,'accumulation':4,'tdt_implementation':'v2',
                                  'chunk_size':16,'encoder_checkpointing':True},
        'before_audio_min_per_wall_min':before['audio_min_per_wall_min'],
        'after_audio_min_per_wall_min':after['audio_min_per_wall_min'],
        'throughput_ratio':ratio, 'throughput_increase_percent':100*(ratio-1),
        'before_median_seconds_per_update':before['median_seconds_per_update'],
        'after_median_seconds_per_update':after['median_seconds_per_update'],
        'before_peak_allocated_vram_gb':before['peak_allocated_vram_gb'],
        'after_peak_allocated_vram_gb':after['peak_allocated_vram_gb'],
        'historical_full_run_train_audio_min_per_wall_min':7.740766065374341,
        'timing_scope':'Matched fixed median TRAIN quartet; each20updates after3warmup; includes WAV read, CPU frontend, H2D, forward/backward, optimizer; excludes teacher-cache torch.load, data selection, DEV, initialization and checkpoint IO.',
        'long_clip_safety':{'source_ids':high['source_ids'], 'parity_passed':True,
                            'timing':high['timing']['end_to_end']},
        'physical_batching_status':'B2 not selected: original BF16 gradient gate failed and strict full-model FP32 oracle remains unexplained. B1 exact-loss optimization supplies the speed gain.',
        'expanded_run_rate':'Measure from actual logged audio seconds / training wall seconds; fixed-quartet speed is not an expanded-run average.',
        'input_sha256':{name:sha(name) for name in (old_name,new_name,'batch_profile_b1_throughput_independent_audit.json',
                                                   'batch_profile_v2_b1_high.json','batch_profile_original_b1_high_diagnostic.json')}
    }
    if not report['throughput_improved']:
        raise RuntimeError('Verified throughput gain is not sufficient.')
    (OUTPUT/'batching_speed_comparison.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report))

if __name__ == '__main__':
    main()
