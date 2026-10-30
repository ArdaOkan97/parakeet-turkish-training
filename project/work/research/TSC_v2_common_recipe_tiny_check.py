"""Focused stdlib source-sampler transition fixture; no real data/model/state reads."""
import argparse,ast,copy,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'outputs/parakeet_turkish'
def run(output):
    tree=ast.parse((HERE/'run_hf_TSC_arm_v2.py').read_text(encoding='utf-8'))
    namespace={'math':math}
    names={'RECIPE_KEYS','SOURCE_SAMPLER_KEYS'}
    nodes=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in n.targets)]
    nodes += [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('require','common_source_recipe','recipe_arguments')]
    # Actual source functions with only their literal scientific constants; no runtime imports.
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'TSC_v2_actual_source_fixture','exec'),namespace)
    baseline=ast.parse((ROOT/'work/research/domain_same_science_tiny_check_v1.py').read_text(encoding='utf-8'))
    recipe=ast.literal_eval(next(n.value for n in baseline.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='recipe' for t in n.targets)))
    recipe['eval_epoch_fraction']=.026708331664062605
    checks=[]
    common,transition=namespace['common_source_recipe'](recipe)
    assert common==recipe and not transition['independent_to_shared_replay_rng'];checks.append('Legacy uniform common recipe copied exactly')
    domain={**recipe,'sampling_mode':'domain30','fleurs_fraction':.3,'replay_seed':20261012}
    common,transition=namespace['common_source_recipe'](domain)
    assert common==recipe and transition['TR_domain_weighting_removed'] and transition['independent_to_shared_replay_rng']
    assert transition['source_sampler']=={k:domain[k] for k in namespace['SOURCE_SAMPLER_KEYS']};checks.append('Exactly three domain fields excluded with explicit sampler/RNG transition')
    argv=namespace['recipe_arguments'](common)
    assert '--sampling-mode' not in argv and '--replay-seed' not in argv and '--fleurs-fraction' not in argv
    assert argv[argv.index('--eval-epoch-fraction')+1]==str(recipe['eval_epoch_fraction']);checks.append('Actual command keeps copied source DEV fraction and common science only')
    for delta in ({'unreviewed_key':True},{'sampling_mode':'domain30'},{'sampling_mode':'domain30','fleurs_fraction':.3,'replay_seed':'bad'}):
        try:namespace['common_source_recipe']({**recipe,**delta})
        except ValueError:checks.append('Rejected unknown/incomplete/invalid sampler fields '+str(delta))
        else:raise AssertionError('Unexpected recipe accepted')
    assert not any(n in sys.modules for n in ('torch','nemo','numpy','soundfile'));checks.append('No ML/audio/real checkpoint/data imports')
    output=output.resolve();assert output.is_relative_to(ROOT) and not output.exists()
    report={'kind':'TSC_v2_common_recipe_source_transition_tiny_verification','passed':True,'checks':checks,'check_count':len(checks),
        'synthetic_only':True,'runtime_chain_pending':True,'GPU_or_checkpoint_execution':False}
    with output.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2)
    print(json.dumps({'passed':True,'checks':len(checks),'output':str(output)}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
