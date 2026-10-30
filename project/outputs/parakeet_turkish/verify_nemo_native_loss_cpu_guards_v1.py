"""CPU synthetic guards/reference-gradient regression; never executes CUDA."""
import copy
import hashlib
import json
from pathlib import Path

import torch
import verify_nemo_native_loss_v1 as worker


def main():
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    torch.manual_seed(294311)
    output = Path(__file__).with_name('nemo_native_loss_cpu_guards_v1.json')
    assert not output.exists()
    checks = 0
    cases = []
    for batch in (1,2):
        t,u,v = 7,4,8199
        lengths = torch.tensor([t-i for i in range(batch)], dtype=torch.long)
        targets = torch.tensor([u-i for i in range(batch)], dtype=torch.long)
        labels = torch.randint(1,8192,(batch,u),dtype=torch.long)
        labels[:,0] = 8198
        logits = torch.randn(batch,t,u+1,v+5)
        data = {'schema':1, 'kind':'actual_fleurs_training_joint_logits',
            'joint_logits':logits, 'labels':labels, 'frame_lengths':lengths,
            'target_lengths':targets, 'blank_id':8192, 'durations':[0,1,2,3,4],
            'provenance':{'split':'train','source_ids':['synthetic_guard_fixture_'+str(i) for i in range(batch)],
                'model_training':False,'dropout_disabled':True,'checkpoint_sha256':'a'*64,'tokenizer_sha256':'b'*64}}
        # This metadata is a synthetic validation fixture, not an actual-audio proof.
        worker.validate_input(data)
        checks += 1
        ref, _ = worker.reference_class('TDTLossPytorch')
        for sigma in (0., .02):
            left, right = logits.clone().requires_grad_(), logits.clone().requires_grad_()
            expected = ref(blank=8192,durations=[0,1,2,3,4],reduction=None,sigma=sigma)(left,labels,lengths,targets)
            actual = worker.custom_tdt(right,labels,lengths,targets,[0,1,2,3,4],8192,v,sigma)
            upstream = torch.arange(1,batch+1).float()/3
            (expected*upstream).sum().backward()
            (actual*upstream).sum().backward()
            grad = worker.grad_stats(left.grad,right.grad)
            delta = float((expected-actual).abs().max())
            assert delta < 3e-5 and grad['finite'] and grad['relative_l2'] < 1e-4 and grad['cosine'] > .999999
            cases.append({'batch':batch,'sigma':sigma,'max_loss_delta':delta,'gradient':grad})
            checks += 4
        corruptions = {
            'wrong_split':lambda x:x['provenance'].update(split='test'),
            'nonactual_kind':lambda x:x.update(kind='synthetic'),
            'wrong_blank':lambda x:x.update(blank_id=8198),
            'reordered_durations':lambda x:x.update(durations=[0,2,1,3,4]),
            'wrong_dtype':lambda x:x.update(joint_logits=x['joint_logits'].double()),
            'zero_target_length':lambda x:x['target_lengths'].__setitem__(0,0),
            'wrong_max_frames':lambda x:x['frame_lengths'].__setitem__(0,1),
            'blank_positive_label':lambda x:x['labels'].__setitem__((0,0),8192),
            'missing_checkpoint_sha':lambda x:x['provenance'].pop('checkpoint_sha256'),
            'training_dropout':lambda x:x['provenance'].update(model_training=True)}
        rejected = []
        for name, mutation in corruptions.items():
            changed = copy.deepcopy(data)
            mutation(changed)
            try:
                worker.validate_input(changed)
            except (RuntimeError,KeyError):
                rejected.append(name)
            else:
                raise AssertionError('Accepted corrupted native input: '+name)
        checks += len(rejected)
    assert not torch.cuda.is_initialized()
    report = {'schema':1,'passed':True,'checks':checks,'synthetic_fp32_cases':cases,
        'rejected_corruptions_per_batch':rejected,'cuda_initialized':False,
        'scope':'Synthetic CPU validation/reference/custom recurrence only; no actual logits or native CUDA execution',
        'worker_sha256':worker.sha(worker.__file__),
        'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'passed':True,'checks':checks,'worker_sha256':report['worker_sha256']}))


if __name__ == '__main__':
    main()
