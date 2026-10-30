"""Saved closed-control scalar/source audit; no ML/checkpoint/audio/TEST reads."""
from pathlib import Path
import codecs
import hashlib
import json
import math
import sys

ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'outputs/parakeet_turkish'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def history_rows(path, receipt):
    """One JSON-array row at a time; at most4MiB decoder buffer, no fullhistory load."""
    path=Path(path);before=(path.stat().st_size,path.stat().st_mtime_ns)
    digest=hashlib.sha256();decoder=codecs.getincrementaldecoder('utf-8')();parsed=json.JSONDecoder()
    with path.open('rb') as stream:
        buffer=''; position=0; eof=False; started=False
        while True:
            while position<len(buffer) and buffer[position].isspace():position+=1
            if not started and position<len(buffer):
                if buffer[position]!='[':raise ValueError('History must be JSON array')
                position+=1;started=True;continue
            if started and position<len(buffer):
                if buffer[position]==',':position+=1;continue
                if buffer[position]==']':
                    position+=1
                    # Finish hash and verify no extra JSON outside the one array.
                    remainder=buffer[position:]
                    while True:
                        data=stream.read(65536)
                        if not data:break
                        digest.update(data);remainder+=decoder.decode(data)
                        if len(remainder)>4*1024*1024:raise ValueError('Trailing data exceeds bound')
                    remainder+=decoder.decode(b'',final=True)
                    if remainder.strip():raise ValueError('Trailing JSON data')
                    break
                try:row,end=parsed.raw_decode(buffer,position)
                except json.JSONDecodeError:
                    if eof:raise
                else:
                    if type(row) is not dict:raise ValueError('History rows must be dictionaries')
                    yield row
                    position=end
                    if position>65536:buffer=buffer[position:];position=0
                    continue
            if eof:raise ValueError('Truncated saved history')
            data=stream.read(65536)
            if data:digest.update(data);buffer+=decoder.decode(data)
            else:buffer+=decoder.decode(b'',final=True);eof=True
            if len(buffer)-position>4*1024*1024:raise ValueError('One saved row exceeds bounded decoder')
    after=(path.stat().st_size,path.stat().st_mtime_ns)
    if before!=after:raise ValueError('Closed evidence changed during audit')
    receipt.update(path=str(path.resolve()),sha256=digest.hexdigest(),bytes=before[0])


def audit_run(run_id, inherited):
    folder=HERE/'runs'/run_id
    selection=json.loads((folder/'selection.json').read_text(encoding='utf-8'))
    config=json.loads((folder/'run_config.json').read_text(encoding='utf-8'))
    recipe=config['config_identity']['training_arguments']
    coefficients=(recipe['replay_kl'],recipe['duration_kl'],recipe['feature_distillation'])
    if coefficients!=(10,.5,1):raise ValueError('Matched saved controls must use10/.5/1')
    last=[];points={};receipt={};residual=0.;count=0; sample_prefix=hashlib.sha256()
    for row in history_rows(folder/'training_history.json',receipt):
        if row['step']<=inherited:continue
        n=row['step']-inherited;count+=1
        if n!=count:raise ValueError('Saved phase prefix is not contiguous')
        expected=row['tdt_loss']+10*row['replay_kl']+.5*row['replay_duration_kl']+row['replay_feature_distillation']
        residual=max(residual,abs(row['loss']-expected))
        scalar={key:row[key] for key in ('loss','tdt_loss','replay_kl','replay_duration_kl','replay_feature_distillation','gradient_norm')}
        scalar.update(local_update=n,step=row['step'],weighted_token_KD=10*row['replay_kl'],
            weighted_duration_KD=.5*row['replay_duration_kl'],feature_term=row['replay_feature_distillation'])
        scalar['token_KD_fraction_of_scalar_loss']=scalar['weighted_token_KD']/row['loss'] if row['loss'] else None
        if n in (1,50,250,500,750,993,1000):points[str(n)]=scalar
        if 'validation' in row:
            points.setdefault(str(n),scalar).update(validation=row['validation'])
        if n<=250:sample_prefix.update(json.dumps(row['sampled_source_ids'],separators=(',',':')).encode())
        last.append(scalar)
        if len(last)>100:last.pop(0)
    if count!=selection['completed_steps']-inherited:raise ValueError('Selection/phase history differs')
    means={key:math.fsum(row[key] for row in last)/len(last) for key in
        ('loss','tdt_loss','weighted_token_KD','weighted_duration_KD','feature_term','gradient_norm')}
    means['token_KD_fraction_of_scalar_loss']=means['weighted_token_KD']/means['loss']
    return {'run_id':run_id,'inherited_steps':inherited,'phase_updates':count,'history':receipt,
      'saved_points':points,'last100_means':means,'max_scalar_decomposition_roundoff':residual,
      'same_first250_draw_ids_sha256':sample_prefix.hexdigest(),'selected_step':selection['selected_step'],
      'selected_turkish_WER':selection['turkish_validation_wer'],'target_denominator':'actual candidate target U',
      'coefficients':{'token_KD':10,'duration_KD':.5,'feature':1},
      'selection_record':{'path':str((folder/'selection.json').resolve()),'sha256':sha(folder/'selection.json')}}


stats_path=HERE/'TSC_BPE512_TRAIN_target_metadata_preparation_v1.json'
stats=json.loads(stats_path.read_text(encoding='utf-8'))
lengths={}
for group,value in {'whole_pool':stats,**stats['per_source']}.items():
    old,new=value['baseline_target_tokens'],value['candidate_target_tokens']
    lengths[group]={'clips':old['clips'],'baseline_mean_U':old['mean'],'candidate_mean_U':new['mean'],
      'old_sum_U':old['sum'],'new_sum_U':new['sum'],'ratio_of_mean_old_U_to_new_U':old['sum']/new['sum'],
      'ratio_is_not_mean_of_individual_clip_inverse_U_ratios':True,
      'token_sum_reduction_fraction':value['target_token_sum_reduction_fraction']}
runs=[audit_run('adaptive_20261011_hf_TSC_BPE512_acquire_01',6500),
      audit_run('adaptive_20261011_hf_TSC_BPE512_row_lr1e3_01_main',7000),
      audit_run('adaptive_20261011_hf_TSC_BPE512_row_lr3e4_01_main',7000)]
report={'kind':'saved_BPE_target_length_loss_scaling_audit_v1','passed':True,
 'scope':'Read-only pinned source, completed TRAIN preparation aggregates and CLOSED saved histories; no fullpool load/ML/checkpoints/audio/GPU/DEV inference/TEST.',
 'source_records':{name:{'path':str((HERE/name).resolve()),'sha256':sha(HERE/name)} for name in
   ('batched_lora_tr_metadata_v1.py','train_lora_batched_tsc_bpe_row_schedule_memory_v3.py','native_tdt_checkpointed_v4.py')},
 'mathematics':{'effective_batch_size':16,'batch_size':8,'accumulation':2,
  'objective':'(1/16) sum_i [raw_native_TDT_NLL_i / candidate_U_i + replay_i * (10*mean_valid_grid_KL_i + 0.5*mean_duration_KL_i + mean_frame_cosine_i)]',
  'native_reduction_none_then_external_U_normalization_once':True,
  'KD_grid':'min(8,T) x min(8,U_replay+1) masked cells; each replay clip grid-mean',
  'TR_KD_terms_zero':True,'RP_target_lengths_and_teacher8193_unchanged':True,
  'RP_fraction':.5,'mixed_batch_token_KD_equals_5_times_mean_per_replay_clip_KL':True,
  'student_token_softmax_full8711_teacher_old8193_extended_by_zero':True,
  'new_class_gradient_per_KD_cell':'10/16 * p_student(new_class) / valid_grid_count; summed over replay clips (before accumulation equivalent16)',
  'no_new_token_mass_escape':'KL(q_old || p_full[:old]) = KL(q_old || p_old/sum_old_p) - log(sum_old_p)',
  'shorter_TR_U_holding_raw_gradient_fixed':'coefficient grows by U_old/U_new; this does not prove actual gradient magnitude because lattice and targets change',
  'per_batch_divisor_change_with_shorter_targets':False,'per_example_TDT_divisor_change_with_shorter_targets':True,
  'scalar_metrics_do_not_measure_per_language_losses_or_component_gradient_norms':True},
 'TRAIN_length_aggregates':lengths,'TRAIN_length_report':{'path':str(stats_path),'sha256':sha(stats_path)},
 'CLOSED_control_metrics':runs,'constant_control_first250_draws_match':runs[1]['same_first250_draw_ids_sha256']==runs[2]['same_first250_draw_ids_sha256'],
 'recommendation':{'normalization_bug_or_KD_bypass_found':False,
  'current_action':'Keep shared-head trial unchanged; judge saved WER/emissions/allfive gates first. Detached scalar KL does not establish gradient dominance.',
  'conditional_one_variable_challenger':'If a clean normalization comparison is later needed: keep candidate labels/native lattice/KD/LRs/sampler/clip16 identical, replace ONLY TR NLL divisor candidate_U with its already-recorded bpe512_baseline_target_tokens; replay divisor unchanged.',
  'why':'This removes the BPE-induced TR loss-unit rescaling while preserving the same candidate prediction task; it decreases nominal TR TDT weight, so it is not an assumed acquisition boost.',
  'new_family_implemented':False,'tested_quality_or_causality_claimed':False}}
out=HERE/'BPE_target_length_loss_scaling_saved_control_audit_v1.json'
with out.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({'report':str(out),'sha256':sha(out),'TRAIN_ratio':lengths['whole_pool']['ratio_of_mean_old_U_to_new_U'],
 'last100':[{k:v for k,v in r.items() if k in ('run_id','last100_means','max_scalar_decomposition_roundoff')} for r in runs]}))
