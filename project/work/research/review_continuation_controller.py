"""Read-only real paused-state checks and mocked continuation lifecycle guards.

All child processes, GPU work, TEST inference, Backlog calls and clock waits in
lifecycle fixtures are mocked. The real check loads the existing pause state
on CPU and does not remove markers or modify source/training artifacts.
"""
from __future__ import annotations
import argparse
import copy
import contextlib
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import types
import uuid
from pathlib import Path

ROOT = Path.cwd()
OUTPUT = ROOT / 'outputs/parakeet_turkish'
sys.path.insert(0, str(OUTPUT))
import run_experiment_loop_continued as controller
import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(1)

SOURCE = Path(controller.__file__).read_text(encoding='utf-8')
SOURCE_SHA = controller.sha(Path(controller.__file__))
AREA = ROOT / 'work/research/continuation_guard_fixtures' / SOURCE_SHA[:12] / uuid.uuid4().hex[:8]
AREA.mkdir(parents=True, exist_ok=True)
RESULTS = []


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def check(name, condition, detail=None):
    RESULTS.append({'name':name, 'passed':bool(condition), 'detail':detail})
    if not condition:
        raise AssertionError(name + ': ' + str(detail))


def real_preflight():
    manifest = OUTPUT / 'runs/adaptive_20261010_01/gpu_benchmark_pause.json'
    saved = controller.read(manifest)
    profile_evidence = OUTPUT / 'large_batch_typical_fp32_reference_math_b1.json'
    benchmark_evidence = [{'path':str(profile_evidence), 'sha256':controller.sha(profile_evidence)}]
    mock_manifest = copy.deepcopy(saved)
    # The GPU benchmark is not complete yet. Mock only its authorization flags
    # and evidence binding so we can audit the existing exact-resume state now.
    # This result cannot be used as an actual benchmark-completion assertion.
    mock_manifest.update(benchmark_completed=True, original_controller_exited=True,
        benchmark_evidence=benchmark_evidence)
    original_read = controller.read
    def patched_read(path, default=None):
        return copy.deepcopy(mock_manifest) if Path(path).resolve() == manifest.resolve() else original_read(path, default)
    controller.read = patched_read
    args = argparse.Namespace(loop_id=saved['loop_id'], pause_manifest=manifest, validate_only=True)
    try:
        verified = controller.validate_pause(args)
        check('real_paused_checkpoint_optimizer_sampler_rng_config', verified['checkpoint_step'] == 1159,
            {'step':verified['checkpoint_step'], 'checkpoint_sha256':saved['checkpoint_sha256'],
             'scope':'Actual CPU checkpoint/data/source audit; benchmark-completion flags are mocked only.'})
        command = controller.exact_resume_command(verified['entry'], verified['checkpoint'], 1234.5)
        check('exact_argv_keeps_inferred_steps_identity', '--steps' not in command and '--resume-weights' not in command
            and command.count('--resume-exact') == 1 and command[command.index('--max-wall-seconds')+1] == '1234.5')
        from train_lora_batched import parser
        parsed = parser().parse_args(command[5:])
        check('actual_trainer_cli_accepts_exact_resume', parsed.steps is None and parsed.batch_size == 1
            and parsed.accumulation == 4 and parsed.epochs == .25 and parsed.resume_exact == verified['checkpoint'])
        for name, edit in (
            ('benchmark_not_finished', lambda p:p.update(benchmark_completed=False)),
            ('controller_not_exited', lambda p:p.update(original_controller_exited=False)),
            ('checkpoint_hash_mismatch', lambda p:p.update(checkpoint_sha256='0'*64)),
            ('paused_history_hash_mismatch', lambda p:p.update(paused_history_sha256='0'*64)),
            ('outside_stop_marker', lambda p:p['owned_stop_markers'][0].update(path=str(ROOT.parent/'foreign.stop'))),
            ('unowned_stop_marker_hash', lambda p:p['owned_stop_markers'][0].update(sha256='0'*64)),
            ('original_controller_changed', lambda p:p.update(original_controller_sha256='0'*64)),
            ('wrong_pause_run', lambda p:p.update(run_id='other_run')),
            ('unhashed_benchmark_evidence', lambda p:p.update(benchmark_evidence=[]))):
            mock_manifest = copy.deepcopy(saved)
            mock_manifest.update(benchmark_completed=True, original_controller_exited=True,
                benchmark_evidence=benchmark_evidence)
            edit(mock_manifest)
            try:
                controller.validate_pause(args)
            except (RuntimeError, ValueError, OSError) as error:
                check(name, True, str(error))
            else:
                check(name, False, 'Corruption was accepted')
        return verified
    finally:
        controller.read = original_read


class Clock:
    def __init__(self): self.value = 0.
    def monotonic(self): return self.value
    def sleep(self, seconds):
        self.value += seconds
        if self.value > 50000:
            raise RuntimeError('Fixture wait exceeded bounded lifetime')


def lifecycle(name, mode='normal'):
    area = AREA / name
    fake_output = area / 'outputs'
    fake_work = area / 'work'
    fake_output.mkdir(parents=True)
    fake_work.mkdir()
    loop_id = 'continued_fixture_' + name
    loop_dir = fake_output / 'runs' / loop_id
    run_id = loop_id + '_screen_A'
    snapshot = fake_output / 'run_experiment_loop_continued.py'
    snapshot.write_text(SOURCE, encoding='utf-8')
    module = types.ModuleType('continued_' + name)
    module.__file__ = str(snapshot)
    exec(compile(SOURCE, str(snapshot), 'exec'), module.__dict__)
    module.ROOT, module.OUTPUT = area, fake_output
    clock = Clock()
    module.time = clock
    from datetime import datetime, timedelta, timezone
    anchor = datetime.now(timezone.utc)
    start_at = anchor - timedelta(hours=1)
    deadline_at = anchor + timedelta(hours=11)
    selected = {'batch_size':1,'accumulation':4,'tdt_implementation':'v2','chunk_size':16,'encoder_checkpointing':True}
    profile = {'passed':True,'throughput_improved':True,'selected_configuration':selected}
    put(fake_output/'batching_speed_comparison.json',profile)
    manifest = fake_work / 'data/tr_expanded_train.jsonl'
    manifest.parent.mkdir()
    manifest.write_text('{}\n')
    put(fake_output/'expanded_turkish_verification.json',{'passed':True})
    baseline = {'metrics':{'wer':.0993939393939394}}
    baseline_provenance = {'scope':'mock DEV evidence'}
    pilot_weights = fake_work/'pilot.pt'
    pilot_weights.write_bytes(b'CPU fixture; not a model')
    checkpoint = fake_work/'pause_step_12.pt'
    checkpoint.write_bytes(b'CPU fixture; not a model')
    parameters = copy.deepcopy(module.SCREENING[0][1])
    stop_file = fake_work/(run_id+'.stop')
    command = module.fresh_command(run_id, parameters, pilot_weights, .25, 10303.9335,
        selected, manifest, .1193939393939394, stop_file)
    entry = {'run_id':run_id,'name':'screen_A','parameters':parameters,'epochs_planned':.25,
        'initialization_weights':str(pilot_weights),'initialization_sha256':module.sha(pilot_weights),
        'optimizer_resume':False,'started_at_utc':start_at.isoformat(),'stage_wall_quota_seconds':10303.9335,
        'command':command,'status':'complete','exit_code':0,'selection':{'accepted':True},
        'selection_provenance':{'stale':True},'epochs_completed':.0009}
    origin = {'loop_id':loop_id,'started_at_utc':start_at.isoformat(),'deadline_at_utc':deadline_at.isoformat(),
        'initial_window_hours':12.,'screen_epochs_requested':.25,'finalization_reserve_seconds':1800.,
        'stop_grace_seconds':180.,'finalization_cap_seconds':1800.,'selected_configuration':selected,
        'profile_sha256':module.sha(fake_output/'batching_speed_comparison.json'),
        'strong_baseline_provenance':baseline_provenance,'competitive_threshold_wer':.1193939393939394,
        'expanded_manifest_sha256':module.sha(manifest),
        'expanded_data_verification_sha256':module.sha(fake_output/'expanded_turkish_verification.json')}
    state = {**origin,'status':'failed','failure_type':'RuntimeError','failure_message':'Intentional interlock',
        'controller_sha256':module.ORIGINAL_CONTROLLER_SHA256,'selection_split':'FLEURS validation only',
        'trials':[entry],'warnings':[],'best_development_candidate':{'run_id':run_id,'wer':.77,'provenance':{'stale':True}}}
    put(loop_dir/'loop_status.json',state)
    markers = []
    for role, screen in [('trainer_pause','screen_A'),('next_stage_interlock','screen_B')]:
        path = fake_work/(loop_id+'_'+screen+'.stop')
        path.write_text('Fixture owned marker '+role)
        markers.append({'role':role,'path':str(path),'sha256':module.sha(path)})
    pause = {'run_id':run_id,'checkpoint_sha256':module.sha(checkpoint),'owned_stop_markers':markers,'benchmark_evidence':[]}
    pause_path = loop_dir/'gpu_benchmark_pause.json'
    put(pause_path,pause)
    identity = {'fixed':'same paused config'}
    preflight = {'pause':pause,'pause_path':pause_path,'origin':origin,'state':state,'entry':entry,
        'checkpoint':checkpoint,'checkpoint_step':12,'config':{'config_identity':identity},
        'benchmark_pause_path':pause_path,'benchmark_pause_sha256':module.sha(pause_path),
        'start_at':start_at,'deadline_at':deadline_at}
    module.validate_pause = lambda args:copy.deepcopy(preflight)
    module.validate_baseline = lambda :(baseline,baseline_provenance)
    module.keep_awake = lambda enable:{'enabled':enable,'mock':True}
    module.backlog_note = lambda message:None
    commands, children, post_calls = [], [], []

    def mock_selection(rid):
        if rid == module.PILOT_RUN:
            selection = {'run_id':rid,'accepted':True,'selected_step':1200,'turkish_validation_wer':.8242,
                'weights':str(pilot_weights)}
            return selection, {'pilot':'fresh metadata'}
        path = fake_output/'runs'/rid/'selection.json'
        selection = module.read(path,{})
        if not selection.get('accepted'):
            raise RuntimeError('Unaccepted mock selection')
        return selection, {'selection_sha256':module.sha(path),'history_sha256':module.sha(path.parent/'training_history.json'),
            'run_config_sha256':module.sha(path.parent/'run_config.json'),'weights_sha256':module.sha(Path(selection['weights'])),
            'selected_step':selection['selected_step']}
    module.validate_selection = mock_selection

    class Child:
        def __init__(self, argv, **kwargs):
            self.argv, self.pid, self.returncode, self.birth = argv, 12345, None, clock.value
            self.terminated = False
            commands.append(argv);children.append(self)
            self.rid = argv[argv.index('--run-id')+1]
            self.stop = Path(argv[argv.index('--stop-file')+1])
            directory = fake_output/'runs'/self.rid
            directory.mkdir(parents=True,exist_ok=True)
            score = .77 if self.rid.endswith('screen_A') else .76
            if mode == 'same_score':score = .77
            completed = 3204 if self.rid.endswith('screen_A') else 3204
            selected_step = completed
            weights = fake_work/(self.rid+'_best.pt')
            weights.write_bytes(b'CPU fixture selected weights '+self.rid.encode())
            put(directory/'selection.json',{'run_id':self.rid,'accepted':True,'selected_step':selected_step,
                'turkish_validation_wer':score,'weights':str(weights),'completed_steps':completed,'turkish_epochs_completed':.25})
            put(directory/'training_history.json',[{'step':completed,'turkish_epoch':.25,
                'validation':{'turkish_wer':score,'retention_gate_passed':True}}])
            config = {'run_id':self.rid,'config_identity':identity}
            if '--resume-exact' in argv:
                config['warm_start'] = {'mode':'exact_resume','sha256':pause['checkpoint_sha256'],'source_step':12}
                if mode == 'bad_resume':config['warm_start']['source_step'] = 11
            put(directory/'run_config.json',config)
        def poll(self):
            if self.returncode is not None:return self.returncode
            if mode == 'poll_error':raise RuntimeError('Mock process-poll failure')
            if mode == 'stage_grace' and not self.stop.exists():return None
            if mode == 'stage_grace':return None
            self.returncode = 0
            return 0
        def terminate(self):self.terminated=True;self.returncode=-15
        def kill(self):self.terminated=True;self.returncode=-9
        def wait(self,timeout=None):return self.returncode

    class Subprocess:
        CREATE_NO_WINDOW=0;STDOUT=-2;DEVNULL=-3;PIPE=-1
        TimeoutExpired=subprocess.TimeoutExpired
        Popen=Child
        def run(self,argv,**kwargs):
            post_calls.append({'argv':argv,'timeout':kwargs.get('timeout')})
            if mode == 'post_timeout':
                clock.value += kwargs['timeout']
                raise self.TimeoutExpired(argv,kwargs['timeout'])
            clock.value += 30
            if any(str(x).endswith('finalize_lora.py') for x in argv):
                rid = argv[argv.index('--run-id')+1]
                path = fake_output/'runs'/rid/'selection.json'
                selected = module.read(path)
                selected.update(merged=True,heldout_tests_evaluated=True,
                    merged_development_gate={'passed':True,'turkish_wer':.7})
                put(path,selected)
            elif any(str(x).endswith('benchmark_strong_tr_test.py') for x in argv):
                frozen = Path(argv[argv.index('--frozen-selection')+1])
                selected = module.read(frozen)
                put(fake_output/'adaptive_strong_baseline_test_comparison.json',{'passed':True,'status':'complete',
                    'run_id':selected['run_id'],'selected_step':selected['selected_step'],'frozen_selection_sha256':module.sha(frozen)})
            else:
                frozen = Path(argv[argv.index('--frozen-selection')+1])
                selected = module.read(frozen)
                put(loop_dir/'adaptive_results_verification.json',{'passed':mode!='bad_audit','run_id':selected['run_id'],
                    'loop_id':loop_id,'selected_step':selected['selected_step'],'frozen_selection_sha256':module.sha(frozen)})
            return types.SimpleNamespace(returncode=0)
    module.subprocess = Subprocess()
    for filename in ('finalize_lora.py','benchmark_strong_tr_test.py','verify_adaptive_results.py'):
        (fake_output/filename).write_text('# mocked CPU lifecycle worker\n')
    error = None
    try:
        with (area/'mock_lifecycle.log').open('w',encoding='utf-8') as stream, contextlib.redirect_stdout(stream):
            module.main(argparse.Namespace(loop_id=loop_id,pause_manifest=pause_path,validate_only=False))
    except BaseException as exception:error = {'type':type(exception).__name__,'message':str(exception)}
    final = module.read(loop_dir/'loop_status.json',{})
    return {'name':name,'error':error,'final':final,'commands':commands,'children_cleaned':all(child.returncode is not None for child in children),
        'terminated_children':sum(child.terminated for child in children),'post_calls':post_calls,
        'markers_removed':not any(Path(marker['path']).exists() for marker in markers),
        'initial_stage_remaining_seconds':10303.9335-3600,'clock_seconds':clock.value,
        'expected_absolute_deadline':origin['deadline_at_utc']}


if __name__ == '__main__':
    import subprocess
    original_hash = controller.sha(OUTPUT/'run_experiment_loop.py')
    real = real_preflight()
    normal = lifecycle('normal')
    check('mocked_continuation_completes_required_pipeline', normal['error'] is None and normal['final']['status']=='complete')
    check('original_absolute_deadline_preserved', normal['final']['deadline_at_utc'] == normal['expected_absolute_deadline'])
    commands = normal['commands']
    check('same_A_exact_then_fresh_B_C', '--resume-exact' in commands[0] and
        all('--resume-weights' in command for command in commands[1:3]) and
        all(command[command.index('--resume-weights')+1].endswith('pilot.pt') for command in commands[1:3]))
    check('A_budget_not_reset', float(commands[0][commands[0].index('--max-wall-seconds')+1]) <= normal['initial_stage_remaining_seconds']+2)
    check('partial_pause_remains_explicit', normal['final']['trials'][0]['pause_segment']['epochs_completed'] < .25
        and normal['final']['trials'][0]['optimizer_resume'] is True)
    check('owned_markers_only_removed_after_preflight',normal['markers_removed'])
    check('finalizer_strong_test_auditor_share_decreasing_cap',len(normal['post_calls'])==3
        and all(0<call['timeout']<=1800 for call in normal['post_calls'])
        and normal['post_calls'][0]['timeout']>normal['post_calls'][-1]['timeout'])
    check('awake_restored_after_success',normal['final']['awake_restoration']['enabled'] is False)
    same = lifecycle('same','same_score')
    check('same_score_winner_provenance_refreshed',same['final']['best_development_candidate']['run_id'].endswith('screen_A')
        and same['final']['best_development_candidate']['provenance'].get('history_sha256')
        and not same['final']['best_development_candidate']['provenance'].get('stale'))
    failed = lifecycle('failed_resume','bad_resume')
    check('incorrect_resume_evidence_cannot_win',failed['final']['trials'][0]['status']=='failed'
        and not failed['final']['best_development_candidate']['run_id'].endswith('screen_A'))
    cleanup = lifecycle('exception','poll_error')
    check('owned_child_cleanup_on_exception',cleanup['children_cleaned'] and cleanup['terminated_children']==1
        and cleanup['final']['status']=='failed' and cleanup['final']['awake_restoration']['enabled'] is False)
    bounded = lifecycle('bounded','stage_grace')
    check('stage_grace_is_bounded',bounded['children_cleaned'] and bounded['terminated_children']>=1
        and bounded['clock_seconds']<12*3600)
    timed = lifecycle('timed','post_timeout')
    check('post_selection_timeout_is_failure',timed['final']['status']=='finalization_timed_out')
    bad_audit = lifecycle('audit','bad_audit')
    check('failed_independent_audit_prevents_complete',bad_audit['final']['status']=='final_audit_failed')
    check('frozen_original_controller_unchanged',controller.sha(OUTPUT/'run_experiment_loop.py')==original_hash)
    report = {'passed':all(row['passed'] for row in RESULTS),'cpu_only':True,'controller_sha256':SOURCE_SHA,
        'original_controller_sha256':original_hash,'checks':RESULTS,'fixture_root':str(AREA),
        'scope':'Real paused optimizer/sampler/RNG checkpoint preflight on CPU, with benchmark-completion flags mocked. Every lifecycle child/GPU/TEST/Backlog call is mocked. No owned live marker removed.',
        'real_checkpoint_step':real['checkpoint_step'],'lifecycle_cases':[{key:value for key,value in row.items() if key not in ('final','commands')}
            for row in (normal,same,failed,cleanup,bounded,timed,bad_audit)]}
    put(OUTPUT/'continuation_controller_cpu_verification.json',report)
    print(json.dumps({'passed':report['passed'],'checks':len(RESULTS),'report':str(OUTPUT/'continuation_controller_cpu_verification.json')}))
