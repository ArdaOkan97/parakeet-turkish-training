"""Finite public AdamW implementation comparison; no training source mutation."""
import gc
from pathlib import Path
import torch
from atomic_io_retry_v1 import write_json
from asr18_diagnostic_bootstrap_v1 import AREA,record,require


def probe(folder,*,single,stage,admit):
    admit('AdamW_fixed_fixture_startup',1_600_000_000)
    stage('before_AdamW_fixture')
    parameters=[torch.nn.Parameter((torch.arange(size,device='cuda',dtype=torch.float32).remainder(997)/997).reshape(shape))
                for size,shape in ((1048576,(4096,256)),(131072,(1024,128)),(64,(64,)))]
    optimizer=torch.optim.AdamW(parameters,lr=1e-5,weight_decay=.01,foreach=False if single else None)
    for parameter in parameters:
        optimizer.state[parameter]={'step':torch.tensor(64250.,dtype=torch.float32),
            'exp_avg':torch.full_like(parameter,.125),'exp_avg_sq':torch.full_like(parameter,.0625)}
    stage('after_AdamW_fixture_source_state')
    for update in range(20):
        for index,parameter in enumerate(parameters):parameter.grad=torch.full_like(parameter,(index+1)*(update+1)*.0001)
        optimizer.step();optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();torch.cuda.empty_cache()
        stage('AdamW_fixture_update_complete',diagnostic_update=update+1)
    saved={'weights':[p.detach().cpu() for p in parameters],
           'moments':[{key:value.detach().cpu() for key,value in optimizer.state[p].items()} for p in parameters]}
    equivalent=None
    if single:
        reference=AREA/'AdamW_auto_reference_01/fixture.pt'
        prior=torch.load(reference,map_location='cpu',weights_only=True,mmap=True)
        def equal(a,b):return torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
        equivalent=all(equal(a,b) for a,b in zip(saved['weights'],prior['weights'])) and all(
            equal(a[key],b[key]) for a,b in zip(saved['moments'],prior['moments']) for key in ('step','exp_avg','exp_avg_sq'))
        require(equivalent,'Public single-tensor AdamW differs from source CUDA auto implementation')
        write_json(folder/'AdamW_equivalence.json',{'passed':True,'reference':record(reference),'updates':20,
            'weights_and_moments_byte_exact':True,'public_foreach_false':True,'probe_source':record(__file__)})
    else:torch.save(saved,folder/'fixture.pt')
    del parameters,optimizer,saved;gc.collect();torch.cuda.empty_cache();stage('after_AdamW_fixture_release')
    return {'passed':True,'training_run_started':False,'positive_native_update':False,'fixture_AdamW_updates':20,
            'weights_and_moments_byte_exact':equivalent,'public_foreach':False if single else None}
