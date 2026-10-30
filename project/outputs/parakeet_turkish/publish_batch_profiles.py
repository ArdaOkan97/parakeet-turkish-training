"""Publish independently audited batch profiles to the existing local Trackio project."""
import hashlib
import json
import os
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parents[1]
os.environ['TRACKIO_DIR'] = str(OUTPUT / 'trackio')
os.environ['HF_HOME'] = str(ROOT / 'work/monitor-hf-cache')
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
for key in ('TRACKIO_SPACE_ID','TRACKIO_SERVER_URL','TRACKIO_DATASET_ID','TRACKIO_BUCKET_ID','TRACKIO_WEBHOOK_URL'):
    os.environ.pop(key, None)

import trackio

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    audit_path = OUTPUT / 'large_batch_results_verification.json'
    audit = json.loads(audit_path.read_text())
    assert audit['passed'] is True, 'Publish only independently audited timing evidence.'
    matrix_path = OUTPUT / 'large_batch_matrix.json'
    matrix = json.loads(matrix_path.read_text())
    marker = OUTPUT / 'large_batch_trackio_publication.json'
    assert not marker.exists(), 'Existing publication must be reviewed before republishing.'
    publication = {'project':'parakeet-turkish', 'runs':[], 'audit_sha256':sha(audit_path),
                   'matrix_sha256':sha(matrix_path), 'dashboard':'http://127.0.0.1:7860/?project=parakeet-turkish'}
    for family in ('typical','long'):
        rows = []
        for entry in matrix['cases']:
            if entry['family'] != family or entry['mode'] != 'timing':
                continue
            path = Path(entry['report_path'])
            assert sha(path) == entry['report_sha256']
            row = json.loads(path.read_text())
            if row.get('timing',{}).get('end_to_end'):
                rows.append(row)
        rows.sort(key=lambda row:row['batch_size'])
        if not rows:
            continue
        baseline = next(row for row in rows if row['batch_size'] == 1)['timing']['end_to_end']['audio_min_per_wall_min']
        name = 'batch_profile_effective16_' + family
        trackio.init(project='parakeet-turkish',name=name,group='Physical batch benchmarks',resume='allow',
                     config={'purpose':'Matched physical batch throughput; fixed effective batch16',
                             'split':'TRAIN','family':family,'precision':'BF16','attention_backend':'default',
                             'matrix_sha256':publication['matrix_sha256'],'audit_sha256':publication['audit_sha256'],
                             'adoption':'Experimental timing; current training configuration unchanged.'})
        for row in rows:
            t = row['timing']['end_to_end']
            trackio.log({'benchmark/physical_batch_size':row['batch_size'],
                         'benchmark/audio_min_per_wall_min':t['audio_min_per_wall_min'],
                         'benchmark/gain_over_b1':t['audio_min_per_wall_min']/baseline,
                         'benchmark/median_update_seconds':t['median_seconds_per_update'],
                         'benchmark/p90_update_seconds':t['p90_seconds_per_update'],
                         'benchmark/peak_allocated_vram_gb':t['peak_allocated_vram_gb'],
                         'benchmark/peak_reserved_vram_gb':t['peak_reserved_vram_gb'],
                         'benchmark/lattice_padding_factor':row['padding']['lattice_padding_factor'],
                         'benchmark/same_shape_numerical_gate_passed':int(row['parity']['passed'] is True),
                         'benchmark/adoption_eligible':int(row.get('adoption_eligible') is True)},
                        step=row['batch_size'])
        trackio.finish()
        publication['runs'].append({'name':name,'batches':[row['batch_size'] for row in rows]})
    marker.write_text(json.dumps(publication,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(publication),flush=True)

if __name__ == '__main__':
    main()
