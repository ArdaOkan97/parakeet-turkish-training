"""Focused stdlib/AST checks. All ML/closure/runtime values are explicit mocks."""
import argparse
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / 'outputs/parakeet_turkish/train_lora_batched_tsc_bpe_acquire_v1.py'
SOURCE = ROOT / 'outputs/parakeet_turkish/train_lora_batched_tsc_headlr_v1.py'


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True)
    cli = p.parse_args()
    if cli.output.exists(): raise ValueError('Fresh receipt required')
    text = WORKER.read_text(encoding='utf-8'); tree = ast.parse(text)
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    source_tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    source_parser = next(n for n in source_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'parser')
    checks, rejected = [], []
    def check(condition, name):
        if not condition: raise AssertionError(name)
        checks.append(name)
    with tempfile.TemporaryDirectory(prefix='TSC_bpe_worker_fixture_', dir=ROOT / 'work') as temp:
        sandbox = Path(temp)
        calls = []
        env = dict(argparse=argparse, Path=Path, DATA=sandbox/'data', OUTPUT=sandbox/'output', ROOT=sandbox,
                   METADATA_PATH=sandbox/'candidate.jsonl', REPORT_PATH=sandbox/'candidate_proof.json',
                   math=math, __doc__='fixture', acquisition_preflight=lambda args:calls.append(('closure',args.run_id)),
                   native_loss_identity=lambda args:calls.append(('ready',args.run_id)))
        exec(compile(ast.Module(body=[source_parser], type_ignores=[]), str(SOURCE), 'exec'), env)
        env['source_parser'] = env.pop('parser')
        exec(compile(ast.Module(body=[functions['require'],functions['parser'],functions['validate_arguments']],type_ignores=[]),str(WORKER),'exec'),env)
        argv = ['--run-id','synthetic_acquisition','--tsc-input-proof',str(sandbox/'inputs.json'),
                '--tsc-training-approval',str(sandbox/'approval.json'),'--source-closure',str(sandbox/'closure.json'),
                '--replay-seed','123','--eval-every-updates','250','--steps','44958',
                '--resume-acquisition',str(sandbox/'opaque_best.pt'),
                '--acquisition-source-receipt',str(sandbox/'closed_receipt.json')]
        args = env['parser']().parse_args(argv); env['validate_arguments'](args)
        check(args.acquisition_max_updates==500 and args.acquisition_lr==1e-4,'actual_parser_phase500_constant_LR_defaults')
        check(args.native_max_joint_bytes==1850000000,'actual_parser_declared_widened_native_bound')
        check(calls==[('closure','synthetic_acquisition'),('ready','synthetic_acquisition')],'actual_validator_routes_closure_then_native_ready')
        for dest,value in [('resume_exact',sandbox/'phase.pt'),('resume_weights',sandbox/'weights.pt'),
                           ('resume_optimization',sandbox/'legacy.pt'),('resume_runtime_migration',sandbox/'migration.json'),
                           ('acquisition_lr',2e-4),('acquisition_max_updates',0),('eval_every_updates',500)]:
            broken=SimpleNamespace(**vars(args)); setattr(broken,dest,value)
            try: env['validate_arguments'](broken)
            except ValueError: rejected.append(dest)
            else: raise AssertionError('Rejected option accepted: '+dest)
        smoke=env['parser']().parse_args(argv+['--smoke-updates','1'])
        env['validate_arguments'](smoke)
        check(smoke.steps==44958 and smoke.smoke_updates==1,'one_update_smoke_separate_from_inherited_horizon')
        captured={}
        class PhaseStub:
            def state_dict(self,*,runtime_state):
                captured['runtime']=runtime_state
                return {'schema':1,'kind':'HF_TSC_BPE512_new_rows_only_acquisition_state_v1',
                        'weights':{str(i):'opaque_frozen_or_active_tensor' for i in range(207)},
                        'acquisition_optimizer':{'three_tables':'opaque'},'acquisition_optimizer_updates':2,
                        'parked_source_optimizer':{'loaded_into_acquisition_optimizer':False},'runtime_state':runtime_state}
        def save(state,path): captured['saved']=state;Path(path).write_bytes(b'stdlib opaque serialization fixture only')
        env.update(torch=SimpleNamespace(save=save,get_rng_state=lambda:'mock_CPU_rng',
                        cuda=SimpleNamespace(get_rng_state_all=lambda:['mock_CUDA_rng'])),
                   random=SimpleNamespace(getstate=lambda:'mock_Python_rng'))
        exec(compile(ast.Module(body=[functions['save_training_state']],type_ignores=[]),str(WORKER),'exec'),env)
        target=sandbox/'phase.pt'
        env['save_training_state'](target,None,PhaseStub(),SimpleNamespace(state_dict=lambda:{'updates':152}),
            run_id='synthetic_acquisition',step=152,config_identity={'phase':'synthetic'},history=[{'step':152}],
            best_score=.4,selected_step=150,patience=1,retention_failures=0,next_eval_epoch=.1,
            train_wall_seconds=20.,audio_seconds=100.,language_audio_seconds={'tr':50.},
            evaluation_wall_seconds=2.,initial_gate={'historical':True},source_selected_step=150)
        state=captured['saved']
        check(target.read_bytes()==b'stdlib opaque serialization fixture only' and not Path(str(target)+'.tmp').exists(),
              'actual_serializer_temporary_replace_in_isolated_workspace')
        check(len(state['weights'])==207 and 'optimizer' not in state and 'acquisition_optimizer' in state,
              'distinct_explicit207_phase_serializer_no_legacy_source_Adam')
        check(state['step']==152 and state['source_selected_step']==150 and state['acquisition_optimizer_updates']==2,
              'cumulative_step_and_phase_optimizer_count_are_distinct')
        check(captured['runtime']['sampler']=={'updates':152} and captured['runtime']['retention_failures']==0
            and captured['runtime']['initial_gate']=={'historical':True},'runtime_frontier_counters_and_initial_gate_retained')
        tensor=lambda value:SimpleNamespace(shape=(1,),dtype='opaque_FP32',payload=value)
        phase_weights={str(i):tensor('weight'+str(i)) for i in range(207)}
        moments={i:{key:tensor(key+str(i)) for key in ('step','exp_avg','exp_avg_sq')} for i in range(3)}
        optimizer_state={'param_groups':[{'params':[0,1,2],'lr':1e-4,'weight_decay':0.}], 'state':moments}
        phase_saved={'weights':copy.deepcopy(phase_weights),'acquisition_optimizer':copy.deepcopy(optimizer_state)}
        phase=SimpleNamespace(names=tuple(phase_weights),parameters=phase_weights,
                              optimizer=SimpleNamespace(state_dict=lambda:optimizer_state))
        env['_table_bytes_sha']=lambda value:hashlib.sha256(str(getattr(value,'payload',value)).encode()).hexdigest()
        exec(compile(ast.Module(body=[functions['_verify_smoke_phase_reload']],type_ignores=[]),str(WORKER),'exec'),env)
        env['_verify_smoke_phase_reload'](phase,phase_saved,captured['runtime'])
        check(True,'actual_smoke_reload_checker_accepts_all207_and_three_Adam_opaque_bytes')
        for label,alter in [('reloaded_frozen_weight',lambda state:setattr(state['weights']['206'],'payload','corrupt')),
                            ('reloaded_Adam_moment',lambda state:setattr(state['acquisition_optimizer']['state'][2]['exp_avg'],'payload','corrupt'))]:
            broken=copy.deepcopy(phase_saved);alter(broken)
            try:env['_verify_smoke_phase_reload'](phase,broken,captured['runtime'])
            except ValueError:rejected.append(label)
            else:raise AssertionError('Corrupt smoke state accepted: '+label)
    objective=functions['batch_objective']
    no_grad=[n for n in ast.walk(objective) if isinstance(n,ast.With) and any(
        isinstance(i.context_expr,ast.Call) and isinstance(i.context_expr.func,ast.Attribute)
        and i.context_expr.func.attr=='no_grad' for i in n.items)]
    check(len(no_grad)==1,'one_explicit_encoder_projector_no_grad_scope')
    inside=ast.dump(no_grad[0]); full=ast.dump(objective)
    check('attr=\'encoder\'' in inside and 'attr=\'encoder_projector\'' in inside and 'attr=\'decoder\'' not in inside,
          'encoder_and_projector_inside_no_grad_decoder_outside')
    check('attr=\'decoder\'' in full and "id='sampled_preservation'" in full and "id='tdt_loss_fn'" in full,
          'decoder_native_TDT_and_KD_remain_differentiable_call_paths')
    run=functions['run']; run_calls=[n for n in ast.walk(run) if isinstance(n,ast.Call)]
    synchronize=[n for n in run_calls if isinstance(n.func,ast.Attribute) and n.func.attr=='synchronize']
    check(len(synchronize)==1,'one_optimizer_end_run_synchronize_target')
    acq_step=[n for n in run_calls if isinstance(n.func,ast.Attribute) and n.func.attr=='step'
              and isinstance(n.func.value,ast.Name) and n.func.value.id=='acquisition']
    check(len(acq_step)==1 and acq_step[0].lineno<synchronize[0].lineno,'row_only_mask_then_clip_step_precedes_measured_synchronize')
    check(not any(isinstance(n.func,ast.Attribute) and n.func.attr=='AdamW' for n in run_calls),
          'worker_constructs_no_source207_optimizer')
    check('new_token_lr_multiplier*' not in text and 'optimizer.load_state_dict(resume' not in text,
          'no_old_six_boost_or_source_Adam_restore')
    run_dump=ast.dump(run)
    check("id='prepare_sampler_state'" in run_dump and "attr='load_state_dict'" in run_dump
        and "id='validate_training_item'" in run_dump,'real_declared_sampler_migration_native_load_and_sampled_candidate_carrier_check')
    ordered=lambda name:min(n.lineno for n in run_calls if isinstance(n.func,ast.Name) and n.func.id==name)
    check(ordered('initial_source_equivalence')<ordered('BPEOnlyAcquisition'),'actual300_equality_before_fresh_row_only_Adam_construction')
    bound=functions['native_loss_function']
    check('8716' in ast.unparse(bound) and '253798 + 1595' in ast.unparse(bound),
          'all_TR_and_original_replay_widened_arithmetic_bound')
    report={'schema':1,'kind':'TSC_BPE_acquisition_worker_stdlib_source_verification_v1','passed':True,
        'worker_sha256':hashlib.sha256(WORKER.read_bytes()).hexdigest(),
        'fixture_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'checks':checks,'check_count':len(checks),'rejected_options':rejected,
        'optimizer_end_synchronize_line':synchronize[0].lineno,
        'scope':'Actual parser/validator/serializer with explicit opaque ML/closure/native mocks, plus source AST; no real source receipt/checkpoint/corpus/ASR/CUDA/300-equivalence/hardware execution',
        'actual_GPU_gate_passed':False,'actual_CLOSED_source_admitted':False,'activation_performed':False}
    cli.output.parent.mkdir(parents=True,exist_ok=True)
    with cli.output.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2)
    print(json.dumps({'passed':True,'checks':len(checks),'rejections':len(rejected),'sync':synchronize[0].lineno}))


if __name__=='__main__':main()
