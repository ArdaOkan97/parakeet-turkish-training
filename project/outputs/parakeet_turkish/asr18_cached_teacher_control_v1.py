"""Exact fixed-replay teacher constants for a bounded memory diagnostic.

The normal sampled_preservation implementation still computes all student
losses and gradients. Only frozen teacher forward results are cached. This
cache covers the declared two replay controls, not the broad training pool.
"""
import gc
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from atomic_io_retry_v1 import write_json
from asr18_diagnostic_bootstrap_v1 import AREA,record,require


def bits(a,b):
    return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))


def teacher_fingerprint(teacher):
    digest=hashlib.sha256()
    for head,module in sorted(teacher.items()):
        for name,value in sorted(module.state_dict().items()):
            digest.update((head+'.'+name+str(value.dtype)+str(tuple(value.shape))).encode())
            digest.update(memoryview(value.detach().cpu().contiguous().view(torch.uint8).numpy()))
    return digest.hexdigest()


class CachedHead(torch.nn.Module):
    def __init__(self,owner,kind):
        super().__init__();self.owner=owner;self.kind=kind;self.eval()
    def forward(self,*args):
        value=self.owner.active
        require(value is not None,'Exact source-ID cache activation required')
        if self.kind=='encoder_projector':
            require(bits(args[0],value['features']),'Cached teacher features differ')
            return value['ep']
        if self.kind=='decoder':
            require(bits(args[0],value['ids']),'Cached teacher decoder IDs differ')
            return value['dp']
        require(bits(args[0],value['sampled_ep']) and bits(args[1],value['sampled_dp']),'Cached teacher sampling grid differs')
        return value['logits']


class CachedTeacher:
    def __init__(self,records):
        self.records={key:{name:value.cuda() if isinstance(value,torch.Tensor) else value for name,value in row.items()}
                      for key,row in records.items()}
        self.active=None
        self.heads={name:CachedHead(self,name) for name in ('encoder_projector','decoder','joint')}
    def activate(self,source_id):
        require(source_id in self.records,'Replay control is outside the pinned teacher cache')
        self.active=self.records[source_id]


def prepare(model,folder,*,stage,admit):
    from tr_frame_metadata_v1 import load_training_item
    from batched_lora_tr_metadata_v1 import _sample_grid,sampled_preservation
    manifest=AREA/'control_manifest_v3/manifest.json'
    rows=json.loads(manifest.read_text())['rows'][14:]
    require(len(rows)==2 and [row['language'] for row in rows]==['pt','en'],'Exact two replay controls required')
    model.eval();model.requires_grad_(False)
    teacher={name:getattr(model,name) for name in ('encoder_projector','decoder','joint')}
    fingerprint=teacher_fingerprint(teacher)
    admit('fixed_teacher_replay_forward_and_gradient_proof',1_000_000_000)
    records={};comparisons=[]
    for row in rows:
        item=load_training_item(row)
        features=item['features'][None].cuda();labels=item['labels'].long()[None].cuda()
        ids=torch.cat((torch.full((1,1),8192,device='cuda',dtype=torch.long),labels),dim=1)
        frames=features.shape[1];tokens=labels.shape[1]
        ts,_=_sample_grid([frames],'cuda');us,_=_sample_grid([tokens+1],'cuda')
        example=torch.arange(1,device='cuda')[:,None]
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            ep=teacher['encoder_projector'](features);dp=teacher['decoder'](ids)
            sampled_ep=ep[example,ts][:,:,None];sampled_dp=dp[example,us][:,None]
            logits=teacher['joint'](sampled_ep,sampled_dp)
        value={'features':features,'ids':ids,'ep':ep,'dp':dp,'sampled_ep':sampled_ep,'sampled_dp':sampled_dp,'logits':logits,
               'frames':frames,'tokens':tokens,'source_id':row['source_id']}
        records[row['source_id']]={key:tensor.detach().cpu() if isinstance(tensor,torch.Tensor) else tensor for key,tensor in value.items()}
        cached=CachedTeacher({row['source_id']:records[row['source_id']]});cached.activate(row['source_id'])
        def calculate(heads):
            student=(features.detach()*1.01).requires_grad_()
            batch=SimpleNamespace(batch_size=1,replay_indices=(0,),frame_lengths=torch.tensor([frames]),
                target_lengths=torch.tensor([tokens]),replay_base_features=features,decoder_input_ids=ids)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                student_ep=model.encoder_projector(student)
                student_dp=(model.decoder(ids).detach()*1.001).requires_grad_()
                losses=sampled_preservation(model,heads,batch,student,student_ep,student_dp,old_size=8193)
                total=40*losses[0].sum()+2*losses[1].sum()+20*losses[2].sum()
            grads=torch.autograd.grad(total,(student,student_ep,student_dp))
            return [tensor.detach().cpu() for tensor in (*losses,*grads)]
        normal=calculate(teacher);actual=calculate(cached.heads)
        exact=all(bits(a,b) for a,b in zip(normal,actual))
        require(exact,'Cached teacher changed actual preservation losses/student gradients')
        comparisons.append({'source_id':row['source_id'],'all3_losses_and3_student_gradients_byte_exact':exact,
                            'frames':frames,'tokens':tokens})
        del item,features,labels,ids,ep,dp,sampled_ep,sampled_dp,logits,value,cached,normal,actual
        gc.collect();torch.cuda.empty_cache();stage('cached_teacher_control_case_complete',source_id=row['source_id'])
    path=folder/'teacher_constants.pt';torch.save(records,path)
    report={'passed':True,'cache':record(path),'control_manifest':record(manifest),'teacher_fingerprint':fingerprint,
        'preparer':record(__file__),'comparisons':comparisons,'cached_replay_sources':len(records),
        'full_broad_replay_coverage_claimed':False,'model_weights_changed':False,'loss_or_preservation_weight_changed':False}
    write_json(folder/'teacher_cache_equivalence.json',report)
    return {'passed':True,'positive_native_update':False,'frozen_teacher_cache_proved':True,'replay_controls':len(records)}


def restore_for_training(model):
    proof_path=AREA/'teacher_control_cache_03/teacher_cache_equivalence.json'
    proof=json.loads(proof_path.read_text())
    require(proof['passed'] and proof['preparer']==record(__file__),'Pinned actual cache equivalence required')
    require(proof['cache']==record(proof['cache']['path']),'Pinned teacher constants changed')
    normal={name:getattr(model,name) for name in ('encoder_projector','decoder','joint')}
    require(teacher_fingerprint(normal)==proof['teacher_fingerprint'],'Exact original pretrained teacher weights required')
    require(proof['control_manifest']==record(AREA/'control_manifest_v3/manifest.json'),'Fixed replay controls changed')
    records=torch.load(proof['cache']['path'],map_location='cpu',weights_only=True,mmap=True)
    cache=CachedTeacher(records)
    return cache,{'proof':record(proof_path),'cache':proof['cache'],'teacher_fingerprint':proof['teacher_fingerprint'],
        'actual_loss_and_student_gradients_byte_exact':True,'scope':'exact two fixed replay controls; no broad coverage claim'}
