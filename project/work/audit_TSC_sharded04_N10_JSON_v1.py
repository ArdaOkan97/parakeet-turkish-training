from pathlib import Path
import json

source = Path(__file__).with_name('audit_TSC_memory03_N11_JSON_v1.py').read_text(encoding='utf-8')
source = source.replace('adaptive_20261011_hf_TSC_BPE512_row_schedule_memory_03_main', 'adaptive_20261011_hf_TSC_BPE512_row_schedule_sharded_04_main')
source = source.replace('7011', '7010').replace('42301', '50137').replace('28a6d4', 'a260c4')
source = source.replace('2026-10-12T05:25:42Z', '2026-10-12T05:51:36Z')
source = source.replace('scheduled_memory03_N11_closed_resource_DEV_JSON_audit_v1', 'scheduled_sharded04_N10_closed_resource_DEV_JSON_audit_v1')
source = source.replace('TSC_row_schedule_memory03_N11_resource_DEV_JSON_audit_v1.json', 'TSC_row_schedule_sharded04_N10_resource_DEV_JSON_audit_v1.json')
source = source.replace('DEV7011_independent_saved_rescore', 'DEV7010_independent_saved_rescore')
source = source.replace('11-update cold rates', '10-update cold rates')
source = source.replace("'specific_external_PID_or_mirror_cause_proven':False", "'specific_external_PID_or_mirror_cause_proven':False, 'V9_sharding_did_not_prevent_this_resource_stop':True")
source = source.replace("e.stable();path=", "e.stable();assert all(dev[l]['ASR']['word_errors']==n for l,n in {'tr':322,'en':56,'es':48,'it':27,'fr':63,'de':68}.items());path=")
exec(compile(source, str(Path(__file__).resolve()), 'exec'))
