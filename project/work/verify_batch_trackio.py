"""Read back local Trackio records and match the audited speed table."""
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / 'outputs/parakeet_turkish'
db = OUT / 'trackio/parakeet-turkish.db'
publication = OUT / 'large_batch_trackio_publication.json'
speed = json.loads((OUT / 'large_batch_speed_report.json').read_text())
assert speed['integrity_passed'] is True
with sqlite3.connect(db.as_uri() + '?mode=ro', uri=True) as connection:
    results = []
    for family in ('typical', 'long'):
        name = 'batch_profile_effective16_' + family
        stored = connection.execute('SELECT step,metrics FROM metrics WHERE run_name=? ORDER BY step', (name,)).fetchall()
        assert [step for step, _ in stored] == [1, 2, 4, 8, 16]
        expected = {row['batch_size']: row for row in speed['table'] if row['family'] == family and row['timing_scope'] == 'end_to_end'}
        for step, value in stored:
            metric = json.loads(value)
            assert metric['benchmark/physical_batch_size'] == step
            for key, field in [('audio_min_per_wall_min', 'audio_min_per_wall_min'),
                               ('peak_allocated_vram_gb', 'peak_allocated_vram_gb'),
                               ('peak_reserved_vram_gb', 'peak_reserved_vram_gb')]:
                assert metric['benchmark/' + key] == expected[step][field]
            assert metric['benchmark/adoption_eligible'] == 0
            assert metric['benchmark/same_shape_numerical_gate_passed'] == 0
        results.append({'name': name, 'steps': [step for step, _ in stored], 'rows': len(stored)})
report = {'passed': True, 'verified_at_utc': datetime.now(timezone.utc).isoformat(),
          'scope': 'Read-only SQLite readback of all10 local Trackio benchmark records against the independently audited speed table.',
          'project': 'parakeet-turkish', 'database': str(db), 'runs': results,
          'publication_sha256': hashlib.sha256(publication.read_bytes()).hexdigest()}
(OUT / 'large_batch_trackio_verification.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'passed': True, 'runs': results}))
