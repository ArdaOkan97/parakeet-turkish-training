"""NEW source/layout and synthetic two-stage metadata controls; no ML/payloads."""
import argparse
import ast
from copy import deepcopy
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT/'outputs/parakeet_turkish'
ROOT = Path(r'E:\gpu_training_scratchpad')
sys.path.insert(0,str(SOURCE))


class NoNumerical(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','numpy','scipy','transformers','tokenizers','soundfile','av','pyarrow'}:
            raise ImportError('Forbidden numerical/audio import in source-only controls: '+fullname)


def record(path):
    path = Path(path).resolve()
    assert path.suffix in ('.py','.json','.txt') and (path.is_relative_to(PROJECT) or path.is_relative_to(ROOT))
    raw = path.read_bytes()
    return {'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def main(folder):
    folder = Path(folder).resolve(); assert folder.is_relative_to(ROOT.resolve()) and not folder.exists()
    from resource_watchdog_v2 import sample_memory
    samples = []
    for index in range(3):
        value = sample_memory(os.getpid(),'optimizer_free_DEV_metadata_admission')
        system = value['system_memory']; worker = value['worker_private_memory']
        assert value['paired_sampling_within_one_second'] and system['available'] and worker['available']
        assert worker['private_bytes']+64_000_000 < 2_000_000_000
        assert (system['commit_used_bytes']+64_000_000)/system['commit_limit_bytes'] < .90
        assert system['commit_limit_bytes']-system['commit_used_bytes']-64_000_000 > 2_000_000_000
        assert shutil.disk_usage(ROOT).free > 100_000_000_000
        samples.append(value); time.sleep(.05)
    sys.meta_path.insert(0,NoNumerical())
    import asr18_full_run_serial_DEV_v5 as dev
    import asr18_optimizer_free_DEV_structure_v2 as structure
    import asr18_full_run_DEV_runtime_v2 as runtime
    from bpe_broad_production_state_v1 import digest, require
    from bpe_streaming_checkpoint_semantics_v2 import select
    checks = []
    def check(value,name):
        assert value,name
        checks.append(name)
    def reject(operation,name):
        try: operation()
        except (ValueError,KeyError,TypeError,IndexError,AssertionError): checks.append(name); return
        raise AssertionError('Unexpected acceptance: '+name)
    names = ('asr18_optimizer_free_DEV_structure_v2.py','asr18_full_run_serial_DEV_v5.py','asr18_full_run_DEV_runtime_v2.py')
    pins = {name:record(SOURCE/name) for name in names}
    for name in names:
        tree = ast.parse((SOURCE/name).read_text(encoding='utf-8'))
        compile(tree,name,'exec')
        calls = [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node,ast.Call)]
        check(not any(call.endswith(('.AdamW','.Adam','.load')) and call.startswith(('torch.optim','torch.'))
                      or call in ('BPEBroadFull207Optimizer','CompletePassCursor','CarriedReplay','BucketCycler')
                      for call in calls),name+': no optimizer/sampler constructors or torch.load')
        check(not any(isinstance(node,(ast.Import,ast.ImportFrom)) and
                      (getattr(node,'module','') or '').split('.')[0] in ('torch','transformers','numpy')
                      for node in tree.body),name+': no import-time ML initialization')
    historical = ROOT/'analysis/ASR18_memory_diagnosis_v1/feature_window_native20_01/after_update20.state.json'
    actual_pin = record(historical)
    check(actual_pin['sha256']=='a2d9c869f23d93b776b93b290cb432ea18e1138c901594ad614b837b3d42d318',
          'Actual closed20 small metadata pin; never tensor payload')
    document = json.loads(historical.read_text(encoding='utf-8'))
    weight_pairs = select(document['tree'],'weights')[1]
    actual_names = tuple(pair[0][1] for pair in weight_pairs)
    fake_model = SimpleNamespace(named_parameters=lambda:[(name,None) for name in actual_names])
    expected = structure.expected_names(fake_model)
    check(set(expected)==set(actual_names) and len(expected)==207,'NEW structural names match actual original named207 metadata')
    check(sum(name.startswith(('encoder_projector.','decoder.','joint.')) for name in expected)==15,'Exact fifteen original head/decoder tensors')
    check(sum(name.endswith(('lora_A','lora_B')) for name in expected)==192,'Exact 24*4*2 adapter tensors')
    reject(lambda:structure.expected_names(SimpleNamespace(named_parameters=lambda:[(n,None) for n in actual_names[:-1]])),
           'Missing original tensor layout rejected')
    reject(lambda:structure.expected_names(SimpleNamespace(named_parameters=lambda:[*( (n,None) for n in actual_names),('joint.extra',None)])),
           'Unexpected head tensor layout rejected')
    p = runtime.resource_proposal()
    check(p['whole_worker_private_forecast_bytes']==p['maximum_pre_restore_private_bytes']+
          p['weight_RNG_restore_increment_bytes']+p['serial350DEV_increment_bytes']==5_450_000_000,
          'Candidate whole-worker arithmetic retains restore allowance during DEV')
    check(p['actual_new_bootstrap_or_full350_fit_proved'] is False and p['post_restore_private_release_assumed'] is False,
          'Candidate arithmetic does not claim resource fit or restoration release')
    # Execute the exact NEW boundary function with only IO/process-verifier
    # adapters substituted. Every positive object is synthetic in-memory only.
    # Real process/death/guard validators are NOT exercised or attested here.
    tree = ast.parse((SOURCE/names[1]).read_text(encoding='utf-8'))
    function = next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='checked_native_boundary')
    function.body = [node for node in function.body if not isinstance(node,ast.ImportFrom)]
    source_pins = {k:{'synthetic_source':k} for k in ('boundary_verifier','native_worker','native_parent','recovery_worker','recovery_parent')}
    def process(synthetic,**kwargs):
        require(synthetic.get('synthetic_process0') is True,'Synthetic process adapter rejects live/missing proof')
        if 'report' in kwargs: require(dev.same(synthetic['report'],kwargs['report']),'Synthetic exact report adapter')
    objects = {
        'policy':{'native_boundary_sources':source_pins},
        'native0':{'synthetic_process0':True,'worker_source':source_pins['native_worker'],
                   'parent_source':source_pins['native_parent'],'report':'native'},
        'native':{'passed':True,'diagnostic_only':False,'run_identity_sha256':digest({'synthetic_identity':1}),
                  'all_exact_GPU_checkpoint_restores_deferred':True,'windows':['window']},
        'window':{'passed':True,'diagnostic_only':False,'all_checkpoint_persistence_verified':True,
                  'exact_GPU_restores_deferred':True,'checkpoints':[]},
        'recovery0':{'synthetic_process0':True,'worker_source':source_pins['recovery_worker'],
                     'parent_source':source_pins['recovery_parent'],'report':'recovery'},
        'recovery':{'kind':'ASR18_full_run_actual_all_captured_checkpoints_recovered_v1','passed':True,
            'diagnostic_only':False,'all_captured_checkpoints_recovered':True,'native_process_completions':['native0'],
            'run_identity_sha256':digest({'synthetic_identity':1}),'actual_native_updates_in_recovery_process':0,
            'actual_draws_consumed_in_recovery':0,'recovered_checkpoints':[]}}
    entries=[]
    for update in (1,10,20):
        pointer='pointer'+str(update); recovered='restored'+str(update)
        item={'pointer':pointer,'checkpoint':'metadata'+str(update),'pre_save_fingerprint':'pre'+str(update),
              'next_draw_packet':'next'+str(update)}; entries.append(item)
        objects[pointer]={'checkpoint':item['checkpoint'],'committed_updates':update}
        objects[recovered]={'passed':True,**item,'all207_threeAdam_all4RNG_match_actual_pre_save':True,
            'both_samplers_journal_prefix_and_next_draw_exact':True,'native_updates_performed':0,
            'draws_consumed':0,'separate_fresh_GPU_recovery':True}
        objects['recovery']['recovered_checkpoints'].append(recovered)
    objects['window']['checkpoints']=deepcopy(entries)
    pointer={'sampler_state_sha256':'sampler','journal_prefix':{'sha256':'prefix'}}
    request={'native_boundary_sources':source_pins,'ROOT_policy_review':'policy','checkpoint_pointer':'pointer20',
             'identity':{'synthetic_identity':1},'optimizer_updates':20}
    objects['evidence']={'kind':'ASR18_full_run_native_exit_and_all_checkpoint_recovery_boundary_for_serial_DEV_v1',
        'owner':dev.OWNER,'passed':True,'diagnostic_only':False,'source':source_pins['boundary_verifier'],
        'checkpoint_pointer':'pointer20','run_identity_sha256':digest(request['identity']),'optimizer_updates':20,
        **pointer,'all207_threeAdam_all4RNG_exact_checkpoint_restore_verified':True,
        'native_process_completions':['native0'],'captured_checkpoints':deepcopy(entries),
        'actual_recovery_process0':'recovery0','recovery_report':'recovery'}
    globals_={'read_bound':lambda pin:objects[pin],'verify':process,'bound':lambda pin:pin,
              'require':require,'same':dev.same,'digest':digest,'integer':dev.integer,'OWNER':dev.OWNER}
    exec(compile(ast.Module(body=[function],type_ignores=[]),'<exact boundary AST with synthetic IO adapters>','exec'),globals_)
    checked=globals_['checked_native_boundary']
    check(checked('evidence',request=request,pointer=pointer) is objects['evidence'],
          'Synthetic two-stage metadata accepted only with all three1/10/20 recovery entries')
    def rejected_change(key,field,value,label):
        old=deepcopy(objects[key]);objects[key][field]=value
        try: reject(lambda:checked('evidence',request=request,pointer=pointer),label)
        finally:objects[key]=old
    rejected_change('evidence','diagnostic_only',True,'Diagnostic boundary cannot enter fullrun DEV')
    rejected_change('evidence','kind','ASR18_resident_native_persistence_verified_idle_boundary_v2','Live persistence idle cannot substitute for actual native/recovery exits')
    rejected_change('native0','synthetic_process0',False,'Missing actual native0 adapter rejected')
    rejected_change('recovery0','synthetic_process0',False,'Missing actual recovery0 adapter rejected')
    rejected_change('recovery','recovered_checkpoints',['restored20'],'Final-only recovery cannot replace every captured checkpoint')
    rejected_change('recovery','recovered_checkpoints',['restored20','restored10','restored1'],'Wrong recovery order rejected')
    rejected_change('recovery','diagnostic_only',True,'Finite diagnostic recovery cannot be promoted')
    rejected_change('recovery','actual_native_updates_in_recovery_process',False,'Boolean recovery clock rejected')
    rejected_change('restored10','all207_threeAdam_all4RNG_match_actual_pre_save',False,'Missing exact all-state middle recovery rejected')
    rejected_change('restored10','draws_consumed',1,'Recovery consumed draw rejected')
    rejected_change('restored20','next_draw_packet','wrong','Wrong exact next-draw binding rejected')
    rejected_change('evidence','optimizer_updates',True,'Boolean boundary optimizer clock rejected')
    rejected_change('evidence','journal_prefix',{'sha256':'wrong'},'Wrong authoritative journal prefix rejected')
    rejected_change('native','diagnostic_only',True,'Diagnostic native source rejected before DEV')
    rejected_change('native0','worker_source',{'synthetic_source':'wrong'},'Wrong native source rejected')
    rejected_change('evidence','captured_checkpoints',[entries[-1]],'Captured checkpoint omission rejected')
    check(all(record(SOURCE/name)==pin for name,pin in pins.items()) and record(historical)==actual_pin,
          'All NEW sources and actual old metadata unchanged by controls')
    check(not any(name.split('.')[0] in {'numpy','torch','scipy','transformers','tokenizers','soundfile','av','pyarrow'}
                  for name in sys.modules),'No numerical/audio/model runtime imported')
    folder.mkdir()
    report={'kind':'ASR18_optimizer_free_DEV_synthetic_source_contract_controls_v1','passed':True,
        'checks':checks,'check_count':len(checks),'producer':{'pid':os.getpid(),'ppid':os.getppid()},
        'sources':pins,'historical_small_metadata':actual_pin,'prospective_increment_bytes':64_000_000,
        'fresh_three_samples':samples,'resource_policy_proposal':p,
        'synthetic_process_and_IO_adapters_only':True,'actual_process_validator_exercised':False,
        'actual_saved_weight_payload_opened':False,'actual_model_or_tokenizer_instantiated':False,
        'actual_checkpoint_restore_or_DEV_executed':False,'GPU_used':False,'production_enabled':False,
        'fullrun_or_resource_approval_claimed':False}
    (folder/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'passed':True,'checks':len(checks),'report':record(folder/'report.json')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--folder',required=True)
    main(parser.parse_args().folder)
