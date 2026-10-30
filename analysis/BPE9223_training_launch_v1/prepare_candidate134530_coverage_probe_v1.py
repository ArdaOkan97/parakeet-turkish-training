"""Derive the same finite frozen-reader observation for the exact new3rows."""
from pathlib import Path
import ast,hashlib,json
N=Path(__file__).parent
old=N/'resident_RAM_original_interval_post88426_failure_actual_v1/observe_ordered_remaining_original_intervals_v1.py'
t=old.read_text(encoding='utf-8')
before="read_json(N/'actual_failure_closure.json')['actual_current_preparation_plan']['path']"
after="read_json(N/'actual_ordered_remaining_plan_metadata.json')['source_plan']['path']"
assert t.count(before)==2;t=t.replace(before,after)
for before,after in [('range(6,20)','(9,)'),('len(candidates)==65','len(candidates)==3'),
    ('Exactly65 ordinary candidates','Exactly3 current ordered YODAS candidates'),
    ('88421+update_index',"update['global_update']"),('post88426_CPU_coverage_probe','post134529_CPU_coverage_probe')]:
    assert before in t;t=t.replace(before,after)
ast.parse(t);p=N/'resident_RAM_original_interval_post134529_failure_actual_v1/observe_candidate134530_original_intervals_v1.py'
assert not p.exists();p.write_text(t,encoding='utf-8')
print(json.dumps({'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))
