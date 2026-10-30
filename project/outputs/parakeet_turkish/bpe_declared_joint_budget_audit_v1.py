"""Stream every retained declared row through the unchanged joint budget.

This proves only the stated joint/DP logical allowance. Encoder/decoder memory,
full worker forecasts, actual decoded geometry, and CUDA admission are separate.
Unsupported rows are counted and preserved; they are never skipped as draws.
"""
from pathlib import Path
from bpe_broad_production_state_v1 import require
from bpe_broad_audio_v1 import record
from bpe_declared_geometry_index_v1 import open_final,signature
from tdt_full_native_allocation_budget_v1 import allocation_budget

def audit_index_joint_budget(path,*,expected_rows,expected_tokens,guard=lambda:None):
    require(type(expected_rows) is int and expected_rows>0 and type(expected_tokens) is int and expected_tokens>=expected_rows,'Exact complete retained denominator required')
    path=Path(path).resolve();before=signature(path);connection=open_final(path);count=tokens=unsupported=0;examples=[];largest=None
    maximum={'encoder_frames':0,'target_tokens':0,'full16k_samples':0,'clean_PCM_float32_bytes_per_row':0,'original_feature_and_mask_bytes_per_row':0}
    try:
        connection.set_progress_handler(lambda:(guard() or 0),10000)
        rows=connection.execute('SELECT ordinal,encoder_frames,target_tokens,clean16k_samples,mel_frames,status FROM geometry ORDER BY ordinal')
        for ordinal,t,u,n,mel,status in rows:
            guard();count+=1;require(type(ordinal) is int and ordinal==count<=expected_rows and type(u) is int and u>0,'Every exact contiguous retained ordinal/full target count required')
            tokens+=u
            try:
                require(status=='declared' and all(type(v) is int and v>0 for v in (t,u,n,mel)) and mel==n//160 and t==(mel+7)//8,
                    'Unresolved/missing/incorrect declared full geometry remains blocked')
                for key,value in (('encoder_frames',t),('target_tokens',u),('full16k_samples',n),('clean_PCM_float32_bytes_per_row',4*n),
                    ('original_feature_and_mask_bytes_per_row',mel*128*4+mel)):
                    maximum[key]=max(maximum[key],value)
                budget=allocation_budget(t,u,projected_elements=(t+u+1)*640,head_elements=8716*641,hidden_width=640)
                if largest is None or budget['total_budget_bytes']>largest['budget']['total_budget_bytes']:largest={'ordinal':ordinal,'budget':budget}
            except ValueError as exc:
                unsupported+=1
                if len(examples)<16:examples.append({'ordinal':ordinal,'declared_T':t,'full_U':u,'full16k_samples':n,'status':status,'reason':str(exc)})
        require(count==expected_rows and tokens==expected_tokens,'Whole retained row/target denominator differs')
    finally:connection.close()
    guard();require(before==signature(path),'Final declared index changed during logical budget audit')
    return {'kind':'ASR18_all_retained_declared_joint_DP_logical_budget_audit_v1','complete':True,'passed':unsupported==0,
        'retained_rows':count,'total_target_tokens':tokens,'unsupported_rows':unsupported,'first_unsupported_rows':examples,
        'maximum_declared_geometry':maximum,'largest_total_joint_budget':largest,
        'all_original_targets_counted_no_quality_or_language_exclusions':True,'actual_audio_decoded':False,'actual_training_draws_consumed':0,
        'encoder_decoder_or_worker_peak_fit_approved':False,'production_native_forecast_approved':False,'TRAIN_admitted':False,
        'allocation_policy_source':record(Path(__file__).with_name('tdt_full_native_allocation_budget_v1.py')),'source':record(__file__)}
