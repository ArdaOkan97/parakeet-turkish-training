"""Inactive fifth exact full-clip certificate and prior-preserving delegation."""
import ast,difflib,hashlib,json
from pathlib import Path
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish';N=Path(__file__).parent
def pin(p):
    p=Path(p);r=p.read_bytes();return {'path':str(p.resolve()),'bytes':len(r),'sha256':hashlib.sha256(r).hexdigest()}
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(p,v):assert not p.exists();p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
report_path=N/'full_member_actual_01/report.json';r=json.loads(report_path.read_text());u=json.loads((N/'probe_actual_01/actual_failed_update_metadata.json').read_text());row=u['rows'][3]
geometry={k:r['actual_interval_geometry'][k]for k in ('native_sr','native_samples','clean16k_samples','mel_frames','encoder_frames','target_tokens')}
c={'kind':'ASR18_source_bound_one_original_full_clip_geometry_certificate_v1','original_row':r['row'],
    'original_row_sha256':digest(r['row']),'original_declaration':r['declared_geometry'],'full_target_ids':row['tr_metadata_labels'],
    'full_target_ids_sha256':digest(row['tr_metadata_labels']),'raw_label_sha256':hashlib.sha256(r['row']['text'].encode()).hexdigest(),
    'original_TAR':r['original_TAR'],'recording_index_entry':r['recording_index_entry'],'member_sha256':r['member_sha256'],
    'actual_geometry':geometry,'decoded_frames':r['decoded_frames'],'max_abs_PTS_residual_samples':r['maximum_abs_PTS_residual_samples'],
    'native_f32_sha256':r['full_source_native_f32_sha256'],'clean_f32_sha256':r['clean_f32_sha256'],'clean_i16_sha256':r['clean_i16_sha256'],
    'independent_full_member_observation':pin(report_path),'actual_failure':pin(N/'actual_failure_closure.json'),
    'ordered_failure_localization':pin(N/'probe_actual_01/report.json'),'actual_causal_CPU_closure':pin(N/'actual_causal_CPU_closure_receipt.json'),
    'original_annotations_indexes_and_plans_unchanged':True,'all_original_physical_samples_once':True,
    'full_labels_order_and_full_pass_denominator_unchanged':True,'no_generic_EOF_tolerance_or_clamp':True,
    'historical_native_reservation_retained':600192,'actual_physical_native_samples':600190,
    'unchanged_clean200064_mel1250_features1251_T157_U82':True}
certificate=N/'fifth_original_full_clip_geometry_certificate_v1.json';write(certificate,c)
old_path=S/'fastmss_original_clip_geometry_overlay_v3.py';old=old_path.read_text();new=old
replacements=[('fastmss_original_clip_geometry_overlay_v2','fastmss_original_clip_geometry_overlay_v4'),
    ('Third exact','Fifth exact'),('self.third_overlay_policy','self.fifth_overlay_policy'),('self.third_certificate','self.fifth_certificate'),
    ('its two prior certified rows','its four prior certified rows'),('ASR18_source_bound_third_original_full_clip_geometry_overlay_v1','ASR18_source_bound_fifth_original_full_clip_geometry_overlay_v1'),
    ('alania:2c21c91407c3dc1cb4111da29af1c53e_00000','alania:9105d11d7d12c9c166028f98935dc107_00000'),
    ('ORDINAL=382298','ORDINAL=417094'),('14.977','12.504'),('native_samples=718896','native_samples=600192'),
    ('native_samples=718890','native_samples=600190'),('clean16k_samples=239632','clean16k_samples=200064'),
    ('clean16k_samples=239630','clean16k_samples=200064'),('mel_frames=1497','mel_frames=1250'),
    ('encoder_frames=188','encoder_frames=157'),('target_tokens=91','target_tokens=82'),
    ("== 91, 'Exact original row/annotation/full labels'","== 82, 'Exact original row/annotation/full labels'"),
    ("certificate['decoded_frames'] == 750","certificate['decoded_frames'] == 626"),
    ('mel1497 versus declared1497, with T188/U91','mel1250 versus declared1250, with T157/U82'),
    ('decode_full_member=prior.decode_full_member','decode_full_member=prior.prior.decode_full_member')]
for before,after in replacements:assert before in new,before;new=new.replace(before,after)
before="""    require(observed['full_original_frames_used_once'] is True
            and observed['synthesized_or_skipped_samples'] == 0
            and observed['actual_geometry'] == actual
            and all(observed[k] == certificate[k] for k in ('member_sha256', 'native_f32_sha256',
                'clean_f32_sha256', 'clean_i16_sha256', 'recording_index_entry', 'original_TAR')),
            'Exact independent full-member physical PCM observation')"""
after="""    require(observed['all_original_physical_samples_used_once'] is True
            and observed['synthesized_or_skipped_samples'] == 0
            and {k:observed['actual_interval_geometry'][k]for k in actual} == actual
            and observed['full_source_native_f32_sha256'] == certificate['native_f32_sha256']
            and observed['full_target_ids'] == certificate['full_target_ids']
            and all(observed[k] == certificate[k] for k in ('member_sha256',
                'clean_f32_sha256', 'clean_i16_sha256', 'recording_index_entry', 'original_TAR')),
            'Exact independent full-member physical PCM and complete target observation')"""
assert before in new;new=new.replace(before,after);ast.parse(new)
path=S/'fastmss_original_clip_geometry_overlay_v5.py';assert not path.exists();path.write_text(new,encoding='utf-8')
delta=N/'overlay3_to5_full_clip_diff.txt';delta.write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=str(old_path),tofile=str(path))),encoding='utf-8')
handoff={'kind':'ASR18_INACTIVE_fifth_exact_full_clip_overlay_source_handoff_v1','production_admitted':False,
    'overlay_source':pin(path),'certificate':pin(certificate),'delta_from_frozen_full_clip_overlay3':pin(delta),
    'actual_failure':pin(N/'actual_failure_closure.json'),'actual_causal_CPU_closure':pin(N/'actual_causal_CPU_closure_receipt.json'),
    'independent_full_member':pin(report_path),'scope':'ASR18_source_bound_fifth_original_full_clip_geometry_overlay_v1',
    'prior_overlay4_source_preserved':pin(S/'fastmss_original_clip_geometry_overlay_v4.py'),
    'resource_scheduling_CPU6_and_numerical_native_unchanged':True,'CPU_ingress_frontend_control_pending':True}
out=N/'fifth_full_clip_overlay_source_handoff_v1.json';write(out,handoff);print(json.dumps({'handoff':pin(out),**handoff}),flush=True)
