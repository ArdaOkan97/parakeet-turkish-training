"""Narrow source-fraction regression; stdlib metadata only, no state/GPU reads."""
import argparse,ast,copy,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'outputs/parakeet_turkish'
sys.path.insert(0,str(HERE))
import resume_domain30_same_science_v2 as exact
import supervise_domain30_same_science_v2 as supervisor
import run_hf_domain30_arm_v2 as first
def save(path,value):path.write_text(json.dumps(value),encoding='utf-8')
def rejected(call):
    try:call()
    except ValueError:return True
    raise AssertionError('Bad source science was accepted')
def run(output):
    assert not output.exists()
    tree=ast.parse((ROOT/'work/research/domain_same_science_tiny_check_v1.py').read_text())
    recipe=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='recipe' for t in n.targets)))
    recipe['eval_epoch_fraction']=.026708331664062605
    checks=[]
    with tempfile.TemporaryDirectory(dir=ROOT/'work/research',prefix='domain_fraction_v2_') as directory:
        d=Path(directory);src=d/'source_config.json';closure=d/'closure.json';ready=d/'original_ready.json'
        save(src,{'config_identity':{'training_arguments':recipe}});save(closure,{'source_run_config':exact.record(src)});save(ready,{'fixture_only':True})
        argv=first.recipe_arguments(recipe)+['--run-id','synthetic_domain_v2','--source-closure',str(closure),
            '--native-backend-provenance',str(ready),'--sampling-mode','domain30','--fleurs-fraction','.3','--replay-seed','20261012']
        identity={'training_arguments':{**recipe,'sampling_mode':'domain30','fleurs_fraction':.3,'replay_seed':20261012},
            'extended_tokenizer_sha256':first.TOKENIZER_SHA,'input_sha256':{str(ready):exact.sha(ready),str(closure):exact.sha(closure)},
            'domain_sampling_identity':{'sampling_mode':'domain30','replay_seed':20261012}}
        config={**recipe,'run_id':'synthetic_domain_v2','sampling_mode':'domain30','replay_seed':20261012,
            'config_identity':identity,'code_sha256':{}}
        exact.validate_config(config,argv);checks.append('Actual nonquarter copied source fraction accepted')
        wrong=copy.deepcopy(config);wrong['config_identity']['training_arguments']['eval_epoch_fraction']=.25
        wrong['eval_epoch_fraction']=.25;wrong_argv=list(argv);exact.replace_option(wrong_argv,'--eval-epoch-fraction',.25)
        assert rejected(lambda:exact.validate_config(wrong,wrong_argv));checks.append('Selfconsistent but altered quarter fraction rejected versus original source')
        wrong_argv=list(argv);exact.replace_option(wrong_argv,'--head-lr',1e-4)
        assert rejected(lambda:exact.validate_config(config,wrong_argv));checks.append('LR change rejected')
        save(src,{'tampered':True})
        assert rejected(lambda:exact.validate_config(config,argv));checks.append('Changed original closed-source config rejected')
    proof=supervisor.controller_order_proof();assert set(proof['controllers'])=={'first','exact'}
    checks.append('New pinned first/exact owned-close-completion-publish order verified')
    assert supervisor.automatic_budget_handoff({'stop_early_reason':'soft_max_wall_seconds'},
        {'resource_review_required':False,'stop_category':'budget'})
    assert not supervisor.automatic_budget_handoff({'stop_early_reason':'controller_stop_file'},
        {'resource_review_required':False,'stop_category':'manual'})
    checks.append('Scheduled budget accepted and manual stop excluded')
    assert not any(n in sys.modules for n in ('torch','nemo','numpy','soundfile','transformers'))
    checks.append('No ML/audio imports or GPU execution')
    report={'kind':'domain_source_fraction_exact_resume_v2_tiny_verification','passed':True,'checks':checks,
        'source_sha256':{p.name:exact.sha(p) for p in (Path(exact.__file__),Path(supervisor.__file__),Path(first.__file__))},
        'source_order_proof':proof,'actual_source_config_or_checkpoint_read':False,'GPU_execution':False}
    output=first.scoped(output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2)
    print(json.dumps({'passed':True,'checks':len(checks),'output':str(output)}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
