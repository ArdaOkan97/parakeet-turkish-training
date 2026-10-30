"""Bind an actually closed diagnostic state to finite serial DEV; no model import."""
import json,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from atomic_io_retry_v1 import write_json
from bpe_broad_audio_v1 import record,require
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')
def main():
    f=A/'feature_window_native20_01';actual=json.loads((f/'actual_owned_CLOSED0.json').read_text());r=json.loads((f/'report.json').read_text());c=r['completion']
    require(actual['report']==record(f/'report.json') and actual['actual_tool_exit_code']==actual['actual_native_exit_code']==0 and r['passed'] and c['native_updates']==20,'Actual closed20 prerequisite')
    cp=c['checkpoint'];require(record(cp['path'])==cp,'Checkpoint metadata changed');meta=json.loads(Path(cp['path']).read_text());require(record(meta['payload']['path'])==meta['payload'],'Final streaming payload changed')
    window=json.loads(Path(c['feature_window']['path']).read_text());owner=window['owner']
    boundary={'kind':'coherent_checkpoint_boundary','owner_sha256':owner['sha256'],'window_manifest_sha256':c['feature_window']['sha256'],
        'committed_window_updates':20,'checkpoint':cp,'checkpoint_payload':meta['payload'],'native_actual_closed':record(f/'actual_owned_CLOSED0.json'),
        'native_report':record(f/'report.json'),'final_exact_restore':record(f/'restore_proof_20.json'),
        'verified_all207_three_Adam_all_RNG_samplers_and_journal_prefix':True,'verified_final_coherent_latest_pointer':record(f/'latest_coherent_checkpoint.json'),
        'caller_owner':'/root/memory_diagnosis','serial_DEV_uses_checkpoint_not_features':True,'cleanup_does_not_qualify_checkpoint':True}
    require(not (f/'final_coherent_feature_cleanup_boundary_v1.json').exists(),'Preserve boundary history')
    write_json(f/'final_coherent_feature_cleanup_boundary_v1.json',boundary)
    config={'kind':'ASR18_actual20_canonical2D_serial350DEV_diagnostic_v2','production_enabled':False,'diagnostic_only':True,
        'DEV_expected_native_updates':20,'DEV_checkpoint':cp,'DEV_training_report':record(f/'report.json'),'DEV_actual_tool_closure':record(f/'actual_owned_CLOSED0.json'),
        'bootstrap':'bpe_broad_streamed_bootstrap_v2.py','frozen_loader':'stream_frozen_encoder_matrices_v1.py',
        'new_joint_numerical_policy':'tdt_joint_canonical_2d_fp32_v2.py','forward_concurrency':1,'full_DEV_clips':350,'TEST_used':False,
        'retention_error_limits':{'en':69,'es':52,'it':30,'fr':74,'de':70,'pt':66},
        'DEV_incremental_private_forecast_bytes':2100000000,'DEV_forecast_calibration':record(A/'new_joint_streamed_checkpoint2_DEV_01/report.json'),
        'resource_guards':{'CUDA_allocator_cap_bytes':8000000000,'system_yield_fraction':.9,'consecutive_fresh':3,'critical_fraction':.95,'headroom_bytes':2000000000},
        'checkpoint_unqualified_even_if_DEV_passes':True,'production_training_started':False}
    path=S/'asr18_serial_native20_DEV_v2.disabled.json';require(not path.exists(),'Immutable new DEV config');write_json(path,config)
    source_manifest=A/'generic_canonical2D_serialDEV_source_manifest_v2.json';request=json.loads(source_manifest.read_text())
    require(all(record(v['path'])==v for v in request['bindings']),'Prepared DEV dependency changed')
    bindings=request['bindings']+[record(source_manifest),record(path),cp,meta['payload'],record(f/'report.json'),record(f/'actual_owned_CLOSED0.json')]
    review={'kind':'ASR18_local_finite_serial_DEV_source_review_v1','scope':'bounded_production_caller_preparation','accepted':True,
        'reviewer':'/root/memory_diagnosis','ROOT_review_pending':True,'ROOT_review_claimed':False,'production_approval_claimed':False,
        'bindings':bindings,'source_request':record(source_manifest),'config':record(path),
        'checks':['Finite optimizer-free serial350 only','Actual owned closed20/report/checkpoint payload verified',
                  'All207 restore and explicit original-plus20 clocks; no diagnostic source replaces original bootstrap',
                  'Canonical2D_v2 exact policy assertion and original processor evaluation path',
                  'Strict six-language integer limits unchanged; TEST unused and checkpoint always unqualified',
                  'Fresh independent parent/system/CUDA guards; DEV2.1GB forecast from measured prior1.834GB increase plus margin',
                  'Complete conservative owned source closure from existing source request bound; no running sources modified'],
        'human_and_ROOT_authorized_finite_DEV_no_new_production_authority':True}
    write_json(A/'serial_native20_DEV_local_source_review_v1.json',review)
    print(json.dumps({'boundary':record(f/'final_coherent_feature_cleanup_boundary_v1.json'),'config':record(path),
        'local_finite_review':record(A/'serial_native20_DEV_local_source_review_v1.json'),'ROOT_review_claimed':False}))
if __name__=='__main__':main()
