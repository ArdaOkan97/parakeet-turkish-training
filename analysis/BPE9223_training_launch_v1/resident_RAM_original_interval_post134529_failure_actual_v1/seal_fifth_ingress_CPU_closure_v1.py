"""Exact external actual0/death binding for the real CPU ingress control."""
import json,hashlib
from pathlib import Path
import psutil
N=Path(__file__).parent;A=N/'ingress_frontend_actual_01'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def pin(p):
    b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
report=read(A/'report.json');completion=read(A/'CPU_completion.json');who=report['producer'];now=None
try:now=psutil.Process(who['pid']).create_time()
except psutil.NoSuchProcess:pass
assert now is None or abs(now-who['create_time'])>=.001
assert report['passed']and report['actual_native_updates']==report['draws_consumed']==0 and report['no_GPU_or_model']
assert report['actual_geometry']=={'clean16k_samples':200064,'valid_mel_frames':1250,'physical_feature_frames':1251,'encoder_frames':157}
assert len(report['reference']['full_target_ids'])==82 and len(report['negative_controls'])==7 and all(x['rejected']for x in report['negative_controls'])
assert report['previous_policy_delegation']['passed']and report['previous_policy_delegation']['parent_method_restored']
assert completion['producer']==who
result={'kind':'ASR18_actual_fifth_full_clip_ingress_CPU0_closure_v1','actual_TOOL_exit_code':0,'actual_TOOL_chunk_id':'24cb7b',
    'actual_owned_CPU_session':63889,'report':pin(A/'report.json'),'completion':pin(A/'CPU_completion.json'),
    'source':pin(N/'verify_fifth_full_clip_ingress_frontend_v1.py'),'producer':who,'observed_current_create_time':now,'exact_original_CPU_identity_dead':True,
    'full_features_mask_finite_normalization_and82labels_passed':True,'seven_strict_negatives_passed':True,'prior4_policy_and_metadata_delegation_preserved':True,
    'actual_native_updates':0,'actual_draws':0,'actual_GPU_or_model_job':False,'production_admitted':False,
    'elapsed_seconds':report['elapsed_seconds'],'peak_private_bytes':report['peak_private_bytes'],
    'prior_metadata_generator1_failed_TOOL_exit_code':1,'prior_metadata_generator1_failed_TOOL_chunk_id':'f82067',
    'prior_failed_generator_source':pin(N/'derive_owner16_and_stream7_v1.py'),'corrected_generator_source':pin(N/'derive_owner16_and_stream7_v2.py')}
out=N/'actual_fifth_ingress_CPU_closure.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(pin(out)),flush=True)
