"""Root-invoked CPU fork of a closed full HF training state; only run_id changes.

No CUDA is initialized. All Adam/sampler/RNG/history/selection counters and
config identity remain intact. Latest is archived; only BEST is preseeded.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil

import fork_native_resume_state_v1 as base

HERE, ROOT = base.HERE, base.ROOT


def main(args):
    base.require(args.source_closed_confirmed, 'Root must confirm the owned source worker is closed')
    for value in (args.source_run_id, args.target_run_id):
        base.require(re.fullmatch(r'[A-Za-z0-9_-]+', value), 'Safe source/target run IDs required')
    base.require(args.source_run_id != args.target_run_id, 'Use a NEW target run ID')
    folder = base.scoped(args.output_dir)
    source_run = HERE/'runs'/args.source_run_id
    target_run = HERE/'runs'/args.target_run_id
    source_status = base.scoped(args.source_controller_status)
    status = json.loads(source_status.read_text(encoding='utf-8'))
    base.require(status.get('run_id') == args.source_run_id and status.get('child_exit_code') == 0
        and status.get('status') in ('train_window_ended', 'resource_review_required'), 'Closed successful child status required')
    selection = json.loads((source_run/'selection.json').read_text(encoding='utf-8'))
    config = json.loads((source_run/'run_config.json').read_text(encoding='utf-8'))
    history = json.loads((source_run/'training_history.json').read_text(encoding='utf-8'))
    base.require(selection['run_id'] == config['run_id'] == args.source_run_id
        and selection['completed_steps'] == args.source_step and len(history) == args.source_step,
        'Actual source step/config/history differs')
    base.require([row['step'] for row in history] == list(range(1, args.source_step+1)), 'Contiguous completed optimizer history required')
    selected = selection['selected_step']
    base.require(not folder.exists() and not target_run.exists(), 'Fresh fork/run namespaces required')
    for role in ('latest','best'):
        base.require(not (ROOT/'work'/(args.target_run_id+'_'+role+'.pt')).exists(), 'Fresh target checkpoint paths required')
    sources = {key: base.record(path) for key,path in {
        'controller_status':source_status, 'run_config':source_run/'run_config.json',
        'history':source_run/'training_history.json', 'selection':source_run/'selection.json',
        'latest':ROOT/'work'/(args.source_run_id+'_latest.pt'),
        'best':ROOT/'work'/(args.source_run_id+'_best.pt')}.items()}
    import torch
    torch.set_num_threads(2)
    base.require(not torch.cuda.is_initialized(), 'CPU fork must never initialize CUDA')
    folder.mkdir(parents=True)
    manifest = {'schema':1,'kind':'HF_exact_state_run_id_fork_v1','status':'CPU_prepared','passed':True,
        'source_run_id':args.source_run_id,'target_run_id':args.target_run_id,'source_step':args.source_step,
        'inherited_selected_step':selected,'only_changed_state_path':'run_id',
        'source_closed_confirmed_by_root':True,'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'source_evidence':sources,'source_run_config':sources['run_config'],'source_checkpoint':{},
        'fork_checkpoint':{},'development_reuse':{},'fork_helper_sha256':base.sha(Path(__file__)),
        'tree_auditor_sha256':base.sha(Path(base.__file__)), 'heldout_TEST_read_or_copied':False}
    for role,step in (('latest',args.source_step),('best',selected)):
        source = Path(sources[role]['path'])
        state = torch.load(source,map_location='cpu',weights_only=True)
        base.require(state['run_id'] == args.source_run_id and state['step'] == step
            and state['selected_step'] == selected and len(state['weights']) == 207
            and len(state['history']) == step,
            'Full source checkpoint/run/config/selection mismatch')
        if role == 'latest':
            base.require(state['schema'] == 2 and len(state['optimizer']['state']) == 207,
                'Full 207-state Adam checkpoint required')
            base.require(state['config_identity'] == config['config_identity'], 'Latest config identity differs')
            base.require(state['history'] == history, 'Checkpoint and durable history differ')
            manifest['source_config_identity'] = state['config_identity']
            manifest['source_config_identity_sha256'] = hashlib.sha256(json.dumps(state['config_identity'],
                sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
        original = base.identity(state,torch)
        state['run_id'] = args.target_run_id
        archive = folder/(args.target_run_id+'_'+role+'.pt')
        torch.save(state,archive)
        del state
        actual = torch.load(archive,map_location='cpu',weights_only=True)
        base.require(actual['run_id'] == args.target_run_id and base.identity(actual,torch) == original,
            'Fork changed weights/Adam/sampler/RNG/history/counters/config')
        del actual
        target = ROOT/'work'/(args.target_run_id+'_'+role+'.pt')
        if role == 'best':
            shutil.copyfile(archive,target)
            base.require(base.sha(target) == base.sha(archive), 'BEST preseed byte mismatch')
        manifest['source_checkpoint'][role] = {**sources[role], 'step':step}
        manifest['fork_checkpoint'][role] = {**base.record(archive),'step':step,
            'mutable_work_path':str(target),'mutable_work_copy_created':role == 'best',
            'identity_without_top_run_id':original}
    for step in sorted(set((selected,args.source_step))):
        source = HERE/'results'/(args.source_run_id+'_step_'+str(step))
        target = HERE/'results'/(args.target_run_id+'_step_'+str(step))
        base.require(source.is_dir() and not target.exists(), 'Actual DEV and fresh target required')
        files = {str(p.relative_to(source)):base.sha(p) for p in source.rglob('*') if p.is_file()}
        base.require(not any(Path(p).name.startswith('test_') for p in files), 'No heldout copying')
        shutil.copytree(source,target)
        base.require(all(base.sha(target/p) == digest for p,digest in files.items()), 'Inherited DEV changed')
        manifest['development_reuse'][str(step)] = {'source_path':str(source),'target_path':str(target),
            'files_sha256':files,'evaluation_reused_without_new_inference':True,'heldout_test_copied':False}
    target_run.mkdir()
    initial = source_run/'initial_development.json'
    shutil.copyfile(initial,target_run/'initial_development.json')
    manifest['initial_development_reuse'] = {'source':base.record(initial),'target':base.record(target_run/'initial_development.json')}
    for item in sources.values():
        base.require(base.sha(Path(item['path'])) == item['sha256'], 'Closed source changed during fork')
    base.require(not torch.cuda.is_initialized(), 'Unexpected CUDA initialization')
    manifest['CUDA_initialized'] = False
    path = folder/'fork_manifest.json'
    with path.open('x',encoding='utf-8') as stream: json.dump(manifest,stream,ensure_ascii=False,indent=2)
    print(json.dumps({'passed':True,'manifest':str(path),'manifest_sha256':base.sha(path),
        'source_step':args.source_step,'selected_step':selected,'CUDA_initialized':False}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run-id',required=True)
    p.add_argument('--source-step',required=True,type=int)
    p.add_argument('--target-run-id',required=True)
    p.add_argument('--source-controller-status',required=True,type=Path)
    p.add_argument('--source-closed-confirmed',action='store_true')
    p.add_argument('--output-dir',required=True,type=Path)
    main(p.parse_args())
