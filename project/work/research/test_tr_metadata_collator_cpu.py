from pathlib import Path
import sys, json, hashlib, ast, copy
import torch
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'outputs/parakeet_turkish';sys.path.insert(0,str(D));torch.set_num_threads(2)
import batched_lora as old
import batched_lora_tr_metadata_v1 as new
from tr_frame_metadata_v1 import KIND
from efficient_tdt_v2 import chunked_tdt_loss
checks=[]
def check(name,truth):
    assert truth,name
    checks.append(name)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
before_rng=torch.get_rng_state().clone()
torch.manual_seed(1984)
class Processor:
    def __call__(self,*,audio,**kw):
        frames=[len(x) for x in audio];width=max(frames)
        data=torch.zeros(len(audio),width,3);mask=torch.zeros(len(audio),width,dtype=torch.bool)
        for i,(x,t) in enumerate(zip(audio,frames)):
            data[i,:t]=x;mask[i,:t]=True
        return {'input_features':data,'attention_mask':mask}
class Encoder(torch.nn.Module):
    def __init__(self):super().__init__();self.projection=torch.nn.Linear(3,3)
    def forward(self,input_features,attention_mask):
        from types import SimpleNamespace
        return SimpleNamespace(last_hidden_state=self.projection(input_features),attention_mask=attention_mask)
class Joint(torch.nn.Module):
    def __init__(self,n):super().__init__();self.head=torch.nn.Linear(3,n)
    def forward(self,a,b):return self.head(torch.tanh(a+b))
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__();from types import SimpleNamespace
        self.encoder=Encoder();self.encoder_projector=torch.nn.Linear(3,3);self.decoder=torch.nn.Embedding(9,3);self.joint=Joint(12)
        self.config=SimpleNamespace(vocab_size=9,blank_token_id=7,durations=[0,1,2])
        self.encoder.eval()
model=Model();teacher={'encoder_projector':copy.deepcopy(model.encoder_projector).eval().requires_grad_(False),
    'decoder':torch.nn.Embedding(8,3).eval().requires_grad_(False),'joint':Joint(11).eval().requires_grad_(False)}
audio={};rows=[];old_items=[];new_items=[]
for i,(lang,t,ids) in enumerate([('tr',4,[8,2]),('en',2,[2]),('tr',3,[1,8]),('de',3,[1,2])]):
    source=f'CPUfixture{i}';audio[source]=torch.randn(t,3)
    row={'source_id':source,'language':lang,'duration':float(t),'encoder_frames':t,'target_tokens':len(ids)}
    target=torch.tensor(ids);features=torch.randn(t,3)
    rows.append(row);old_items.append({'source_id':source,'features':features,'labels':target})
    new_items.append({'source_id':source,'kind':KIND,'encoder_frames':t,'labels':target} if lang=='tr' else old_items[-1])
def reader(row):return audio[row['source_id']]
old_batch=old.collate_training_batch(Processor(),rows,old_items,blank_id=7,old_size=8,device='cpu',audio_reader=reader)
new_batch=new.collate_training_batch(Processor(),rows,new_items,blank_id=7,old_size=8,device='cpu',audio_reader=reader)
for field in ['labels','decoder_input_ids','target_lengths','frame_lengths']:
    check('exact_'+field,torch.equal(getattr(old_batch,field),getattr(new_batch,field)))
for key in old_batch.inputs:check('exact_encoder_input_'+key,torch.equal(old_batch.inputs[key],new_batch.inputs[key]))
check('genuine_replay_only_feature_storage',new_batch.replay_base_features.shape==(2,4,3) and torch.equal(new_batch.replay_base_features,old_batch.base_features[list(old_batch.replay_indices)]))
check('TR_feature_field_absent',all('features' not in new_items[i] for i in [0,2]) and not hasattr(new_batch,'base_features'))
old_result=old.batch_objective(model,teacher,old_batch,old_size=8,tdt_loss_fn=chunked_tdt_loss,tdt_host_lengths=True,use_autocast=False)
old_result.total.backward();old_grads={name:p.grad.clone() for name,p in model.named_parameters()};model.zero_grad(set_to_none=True)
new_result=new.batch_objective(model,teacher,new_batch,old_size=8,tdt_loss_fn=chunked_tdt_loss,tdt_host_lengths=True,use_autocast=False)
new_result.total.backward()
for name,p in model.named_parameters():check('exact_gradient_'+name,torch.equal(p.grad,old_grads[name]))
for field in ['total','tdt','token_kl','duration_kl','feature_distillation']:check('exact_objective_'+field,torch.equal(getattr(new_result,field),getattr(old_result,field)))
for field,value in old_result.per_clip.items():check('exact_per_clip_'+field,torch.equal(value,new_result.per_clip[field]))
check('literal_zero_TR_preservation',all(torch.equal(new_result.per_clip[field][[0,2]],torch.zeros(2)) for field in ['token_kl','duration_kl','feature_distillation']))
try:
    new.collate_training_batch(Processor(),rows,old_items,blank_id=7,old_size=8,device='cpu',audio_reader=reader)
except ValueError:checks.append('reject_TR_fake_teacher_features')
else:raise AssertionError('TR feature cache must be rejected')
all_tr=new.collate_training_batch(Processor(),rows[:1],new_items[:1],blank_id=7,old_size=8,device='cpu',audio_reader=reader)
check('all_TR_empty_replay_storage',all_tr.replay_base_features.shape==(0,4,0))
all_tr_result=new.batch_objective(model,teacher,all_tr,old_size=8,tdt_loss_fn=chunked_tdt_loss,tdt_host_lengths=True,use_autocast=False)
check('all_TR_finite_objective_no_teacher_features',torch.isfinite(all_tr_result.total).item())
def function_tree(path,name):return next(node for node in ast.parse(path.read_text()).body if isinstance(node,ast.FunctionDef) and node.name==name)
check('batch_objective_AST_unchanged',ast.dump(function_tree(D/'batched_lora.py','batch_objective'),include_attributes=False)==ast.dump(function_tree(D/'batched_lora_tr_metadata_v1.py','batch_objective'),include_attributes=False))
old_run=function_tree(D/'train_lora_batched_data_v4.py','run');new_run=function_tree(D/'train_lora_batched_data_v5.py','run')
old_update=next(node for node in old_run.body if isinstance(node,ast.For) and ast.unparse(node.target)=='step')
new_update=next(node for node in new_run.body if isinstance(node,ast.For) and ast.unparse(node.target)=='step')
check('full_training_update_AST_unchanged',ast.dump(old_update,include_attributes=False)==ast.dump(new_update,include_attributes=False))
check('exact_update_sync_line539',any(isinstance(n,ast.Call) and ast.unparse(n.func)=='torch.cuda.synchronize' and n.lineno==539 for n in ast.walk(new_update)))
torch.set_rng_state(before_rng)
check('CUDA_not_initialized',not torch.cuda.is_initialized())
target=D/'TR_metadata_collator_cpu_verification_v1.json'
if target.exists():raise FileExistsError(target)
target.write_text(json.dumps({'schema':1,'passed':True,'CPU_only':True,'check_count':len(checks),'checks':checks,
    'collator_sha256':sha(D/'batched_lora_tr_metadata_v1.py'),'metadata_helper_sha256':sha(D/'tr_frame_metadata_v1.py'),
    'trainer_sha256':sha(D/'train_lora_batched_data_v5.py'),'scope':'Exact mixed/allTR CPU objective/gradient/schema and unchanged full update AST; actual GPU frame verification pending'},indent=2)+'\n')
print(json.dumps({'passed':True,'checks':len(checks)}))
