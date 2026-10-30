"""Inactive stdlib-only metric/counter fixture; never imports/logs Trackio."""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'outputs/parakeet_turkish'


def main(output):
    path=HERE/'track_continuation_v4.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    functions={node.name:node for node in tree.body if isinstance(node,ast.FunctionDef)}
    old=ast.parse((HERE/'track_continuation_v3.py').read_text(encoding='utf-8'))
    old_functions={node.name:node for node in old.body if isinstance(node,ast.FunctionDef)}
    checks=[]
    def require(ok,name):
        if not ok:raise AssertionError(name)
        checks.append(name)
    def rejected(name,callback):
        try:callback()
        except ValueError:checks.append(name);return
        raise AssertionError(name+' did not reject')
    for name in ('finite_number','resume_context','values_for'):
        require(ast.dump(functions[name],include_attributes=False)==ast.dump(old_functions[name],include_attributes=False),
                name+'_exact_resume_source_unchanged')
    metric_tree=ast.parse((HERE/'track_experiment.py').read_text(encoding='utf-8'))
    metric=next(node for node in metric_tree.body if isinstance(node,ast.FunctionDef) and node.name=='metrics_for')
    metric_env={}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[metric],type_ignores=[])),'original_metrics_AST','exec'),metric_env)
    reads=[]
    summary={'validation_'+language:{'asr':{'wer':.05}} for language in ('tr','en','es','it','fr','de')}
    def read_json(path,default):
        reads.append(path)
        return summary if path.parent.name in ('fixture_step_1250','fixture_step_1252') else default
    env={'math':math,'Path':Path,'ROOT':ROOT,'OUTPUT':HERE,'argparse':argparse,
         'original':SimpleNamespace(metrics_for=metric_env['metrics_for']),
         'read_json':read_json,'ACQUISITION_PHASE':'HF_TSC_BPE512_new_rows_only_acquisition_v1'}
    names=('finite_number','resume_context','values_for','acquisition_context','acquisition_values_for',
           'development_cycle','cycle_rows','parser')
    exec(compile(ast.fix_missing_locations(ast.Module(body=[functions[name] for name in names],type_ignores=[])),str(path),'exec'),env)
    phase=env['ACQUISITION_PHASE']
    config={'phase':phase,'source_selected_step':1250,'inherited_history_steps':1250,
            'config_identity':{'acquisition_phase':{'kind':phase,'source_selected_step':1250,
              'active_optimizer_tables':3,'acquisition_lr':1e-4,'source_Adam_retained_dormant':True,'unfreeze_allowed':False}}}
    gate={'retention_gate_passed':True,'retention_wer_deltas':{'en':0}}
    base={'step':1250,'tdt_loss':1.,'replay_kl':.1,'gradient_norm':.2,'elapsed_seconds':5.,'peak_allocated_vram_gb':1.,
          'train_wall_seconds':100.,'audio_seen_seconds':9000.,'turkish_audio_seen_seconds':2000.,
          'replay_audio_seen_seconds':7000.,'audio_minutes_per_wall_minute':90.,'validation':gate}
    inherited={**base,'step':1200}
    new={**base,'step':1251,'acquisition_updates_completed':1,'acquisition_lr':1e-4,'head_lr':0.,'lora_lr':0.,
         'new_token_update_multiplier':1.,'old_six_character_update_multiplier_applied':False,
         'train_wall_seconds':102.,'audio_seen_seconds':9300.,'turkish_audio_seen_seconds':2100.,
         'replay_audio_seen_seconds':7200.,'audio_minutes_per_wall_minute':9300./102.}
    new.pop('validation')
    final={**new,'step':1252,'acquisition_updates_completed':2,'validation':gate,
           'train_wall_seconds':104.,'audio_seen_seconds':9600.,'turkish_audio_seen_seconds':2200.,
           'replay_audio_seen_seconds':7400.,'audio_minutes_per_wall_minute':9600./104.}
    history=[inherited,base,new,final];before=copy.deepcopy(history)
    context=env['acquisition_context'](config,history)
    values=env['acquisition_values_for'](new,context)
    require(values['resources/post_acquisition_audio_minutes_per_wall_minute']==150.
            and values['resources/post_acquisition_turkish_audio_minutes_per_wall_minute']==50.
            and values['resources/post_acquisition_replay_audio_minutes_per_wall_minute']==100.,
            'post_phase_rates_subtract_actual_parent_audio_and_train_wall')
    require(values['phase/acquisition_updates_completed']==1 and values['phase/acquisition_lr']==1e-4
            and values['phase/lora_active']==0,'actual_worker_phase_fields_use_local_fresh_Adam_clock_and_constant_LR')
    require(env['acquisition_values_for'](inherited,context)=={}
            and env['acquisition_values_for'](base,context)['phase/acquisition_updates_completed']==0,
            'inherited_training_rows_are_not_relabelled_as_acquisition_updates')
    rejected('wrong_local_Adam_clock_rejected',lambda:env['acquisition_values_for']({**new,'acquisition_updates_completed':1251},context))
    rejected('nonincreasing_phase_wall_rejected',lambda:env['acquisition_values_for']({**new,'train_wall_seconds':100.},context))
    rejected('LoRA_active_recipe_rejected',lambda:env['acquisition_values_for']({**new,'lora_lr':3e-4},context))
    development=env['development_cycle']('fixture',history,config)
    require({target.parent.name for target in reads}=={'fixture_step_1250','fixture_step_1252'},
            'only_actual_newrun_source300_and_fresh_validation_folders_read_no_inherited_DEV_aliases')
    rows=list(env['cycle_rows'](config,history,development))
    require(rows[1][0]==1250 and rows[1][1]['dev/wer_en_percent']==5.
            and rows[2][1]['resources/audio_minutes_per_wall_minute']==9300./102.
            and rows[2][1]['resources/post_acquisition_audio_minutes_per_wall_minute']==150.,
            'actual_source_baseline_axis_and_cumulative_metrics_preserved')
    require(history==before,'saved_history_counters_validation_and_ids_unchanged')
    args=env['parser']().parse_args([])
    require(args.state_path==ROOT/'work/trackio_continuation_mirror_state_v4.json'
            and args.source_state_path==ROOT/'work/trackio_continuation_mirror_state_v3.json',
            'new_state_defaults_to_separate_v4_and_preserved_v3_source')
    # Execute just the actual state-load prefix with in-memory records, never follow/log.
    follow=functions['follow'];prefix=copy.deepcopy(follow.body[:2])
    state_func=ast.FunctionDef(name='read_initial_state',args=copy.deepcopy(follow.args),
             body=prefix+[ast.Return(value=ast.Name(id='state',ctx=ast.Load()))],decorator_list=[])
    state={'old_run':{'finished':'heldout-evaluation-complete','reference_base_v3':True,'3500':{'train/tdt_loss':2.}}}
    state_reads=[]
    class StatePath:
        def exists(self):return False
    env['read_json']=lambda target,default:(state_reads.append(target),copy.deepcopy(state))[1]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[state_func],type_ignores=[])),str(path),'exec'),env)
    loaded=env['read_initial_state'](SimpleNamespace(state_path=StatePath(),source_state_path='PRESERVED_V3_STATE'))
    require(loaded==state and state_reads==['PRESERVED_V3_STATE'],
            'actual_state_migration_prefix_preserves_finished_reference_and_point_flags')
    report={'kind':'inactive_Trackio_continuation_v4_fields_CPU_verification_v1','passed':True,
            'source':{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()},
            'worker':{'path':str(HERE/'train_lora_batched_tsc_bpe_acquire_v1.py'),
                      'sha256':hashlib.sha256((HERE/'train_lora_batched_tsc_bpe_acquire_v1.py').read_bytes()).hexdigest()},
            'assertions':len(checks),'checks':checks,'new_phase_metric_keys':sorted(values),
            'real_Trackio_imported_or_logged':False,'actual_shared_JSON_or_mirror_state_written':False,
            'GPU_or_ML_executed':False,'actual_phase_training_measured':False,
            'timing_scope':'Differences of worker sum-of-update train_wall/audio counters; excludes startup/DEV/checkpoint/history serialization',
            'activation':'ROOT must close owned v3 mirror before starting v4 with separate state; no activation performed'}
    with Path(output).open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps({'passed':True,'assertions':len(checks),'report':str(output),'source_sha256':report['source']['sha256']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
