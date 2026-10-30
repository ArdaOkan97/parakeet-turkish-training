import json,sys
from pathlib import Path
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from bpe_broad_audio_v1 import record
from fastmss_BPE9223_cpu_common_v4 import environment,resource_guard
environment();resource_guard()
O=Path(__file__).parent;N=O.parent
old=json.loads((N/'CPU_preparation_v1/source_handoff_v3.json').read_text());g=json.loads((O/'generation.json').read_text())
import fastmss_BPE9223_recorded_room_train_v4 as adapter
_,identity=adapter.policy_identity(old['policy'],resource_guard)
target=O/'augmentation_identity_v4.json';assert not target.exists();target.write_text(json.dumps(identity,indent=2),encoding='utf-8')
h={'kind':'BPE9223_generic_CPU_frozen_source_handoff_v4','owner':'/root/cpu_data_preparation',
    'source_scope':'frozen for ROOT review; no audio admission','sources':g['sources'],
    'prior_CPU3_handoff':record(N/'CPU_preparation_v1/source_handoff_v3.json'),'old_sources_preserved':g['baseline_sources'],
    'exact_delta_map':g['checks'],'hard_context':g['hard_context'],'policy':old['policy'],'augmentation_identity':record(target),
    'complete_9223_population_evidence':old['complete_9223_population_evidence'],
    'configuration_helper':g['configuration_deriver'],'configuration_helper_API':old['configuration_helper_API'],
    'CLI':old['CLI'],'reader_API':old['reader_API'],'configuration_kind':old['configuration_kind'],
    'template_kind':old['template_kind'],'review_scope':old['review_scope'],'context_source_path':old['context_source_path'],
    'resource_caps':old['resource_caps'],'full_branch_math_or_resource_changes':False,
    'metadata_validation':'All8 sources compile and exact import/path substitutions verified; bridge has sole extra hard context name/hash change; no broad control rerun',
    'remaining':['ROOT exact source/config/resource review and genuine hard CPU/native preparation',
        'For future derived full templates, memory-owned context2 lazy protocol3 callback needs its separately named protocol4 import successor; direct exact-reviewed configs work unchanged'],
    'no_audio_or_model_or_GPU_job_started':True,'production_enabled':False,'old_admitted_sources_or_environments_edited':False}
target=O/'source_handoff_v4.json';assert not target.exists();target.write_text(json.dumps(h,indent=2),encoding='utf-8');print(json.dumps(record(target)))
# Repair only illustrative record pins in a NEW handoff; preserve prior file.
F=N/'finite_serial_DEV_source_v2';p=F/'source_handoff_v2.json';f=json.loads(p.read_text());mapping={v['path']:v for v in f['sources'].values()}
def refresh(v):
    if type(v) is dict:
        if v.get('path') in mapping:return mapping[v['path']]
        return {k:refresh(x) for k,x in v.items()}
    if type(v) is list:return [refresh(x) for x in v]
    if type(v) is str:return v.replace('asr18_finite_BPE9223_checkpoint_evidence_v1.build_binding','asr18_finite_BPE9223_checkpoint_evidence_v2.build_binding')
    return v
f=refresh(f);f['illustrative_source_records_corrected_from']=record(p)
target=F/'source_handoff_v2_current.json';assert not target.exists();target.write_text(json.dumps(f,indent=2),encoding='utf-8');print(json.dumps(record(target)))
