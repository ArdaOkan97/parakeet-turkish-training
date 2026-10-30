"""Three original Adam objects, with independent step0 appended9223 views.

Existing moments are transferred by ownership, never reset or numerically
copied/reconstructed. The original optimizer is poisoned after transfer.
"""
from copy import deepcopy
from bpe_broad_full207_optimizer_v1 import BPEBroadFull207Optimizer, encoder_lr
from bpe_full_decoder_optimizer_v1 import TABLES, require
from asr18_BPE9223_model_migration_v2 import bounded_equal

KIND='BPE9223_broad_full207_three_AdamW_mixed_fresh_clock_v1'


class BPE9223Full207Optimizer(BPEBroadFull207Optimizer):
    def __init__(self, model, original, *, migration, torch_module, guard):
        self.torch=torch_module
        def bits_equal(torch,left,right):return bounded_equal(torch,left,right,guard=guard)
        require(not original.poisoned and original.updates==0 and original.base_clocks==(64250,2000,62250)
            and len(original.optimizers)==3 and migration['old_rows_and_duration_bytes_exact'] is True,
            'Original production genesis/mature threeAdam only; never diagnostic adoption')
        original.validate(full=True)
        self.source_records=deepcopy(original.source_records);self.migration=deepcopy(migration)
        self.names=original.names;self.head_names=original.head_names;self.lora_names=original.lora_names
        self.parameters={n:p for n,p in model.named_parameters() if n in self.names}
        require(set(self.parameters)==set(original.parameters) and len(self.parameters)==207,'All207 active named tensors')
        for name,old in original.parameters.items():
            new=self.parameters[name]
            if name in TABLES:
                require(bits_equal(self.torch,old[:8711],new[:8711]),'Mature lexical bytes unchanged: '+name)
                if name!=TABLES[0]:require(bits_equal(self.torch,old[8711:],new[9223:]),'Duration bytes relocated: '+name)
            else:require(new is old and bits_equal(self.torch,new,old),'All non-table Parameter objects unchanged')
            new.requires_grad_(True);new.grad=None
        self.mature_plan=tuple({**item,'target_rows':[9223,9228]} if item['part']=='duration' else deepcopy(item) for item in original.mature_plan)
        old_fresh=tuple(deepcopy(item) for item in original.fresh_plan)
        self.fresh_plan=old_fresh+tuple({'name':n,'part':'appended512_BPE9223','source_id':None,'source_rows':None,'target_rows':[8711,9223]} for n in TABLES)
        self.heads=tuple(self.parameters[n] for n in self.head_names);self.adapters=tuple(self.parameters[n] for n in self.lora_names)
        self.mature=tuple(self.view(item) for item in self.mature_plan);self.fresh=tuple(self.view(item) for item in self.fresh_plan)
        self.optimizers=original.optimizers;object_ids=[id(v) for v in self.optimizers]
        # A failure after any transfer cannot leave the old caller reusable.
        original.poisoned=True
        # Existing three objects, state tensors and CPU clocks retain identity.
        for optimizer,old_params,new_params in ((self.optimizers[0],original.mature,self.mature),
                (self.optimizers[1],original.fresh,self.fresh[:3])):
            require(tuple(optimizer.param_groups[0]['params'])==tuple(old_params),'Exact existing named partition order')
            for old,new in zip(old_params,new_params):
                state=optimizer.state.pop(old)
                require(new.shape==old.shape and bits_equal(self.torch,new,old),'Storage-sharing new parent slice has exact old bytes')
                optimizer.state[new]=state
            optimizer.param_groups[0]['params']=list(new_params)
        for parameter in self.fresh[3:]:
            self.optimizers[1].param_groups[0]['params'].append(parameter)
            self.optimizers[1].state[parameter]={'step':self.torch.tensor(0.,dtype=self.torch.float32,device='cpu'),
                'exp_avg':self.torch.zeros_like(parameter),'exp_avg_sq':self.torch.zeros_like(parameter)}
        require(object_ids==[id(v) for v in self.optimizers] and tuple(self.optimizers[2].param_groups[0]['params'])==self.adapters,
            'Exactly the original three Adam objects and original LoRA parameter identities')
        self.base_clocks=(64250,2000,62250);self.updates=0;self.poisoned=False
        self.group_policies=deepcopy(original.group_policies)
        self.validate(full=True)

    def validate(self, full=False):
        for index,optimizer in enumerate(self.optimizers):
            for offset,parameter in enumerate(optimizer.param_groups[0]['params']):
                base=0 if index==1 and offset>=3 else self.base_clocks[index]
                state=optimizer.state[parameter]
                require(state['step'].device.type=='cpu' and state['step'].dtype==self.torch.float32
                    and state['step'].item()==base+self.updates
                    and state['exp_avg'].shape==state['exp_avg_sq'].shape==parameter.shape,
                    'Independent existing/new row clocks and exact moment geometry')
                if full:require(all(bool(self.torch.isfinite(state[k]).all()) for k in ('exp_avg','exp_avg_sq'))
                    and bool((state['exp_avg_sq']>=0).all()),'Finite nonnegative complete moments')
        require([len(o.param_groups[0]['params']) for o in self.optimizers]==[17,6,192], 'Exact mature17/fresh6/LoRA192 state coverage')

    def identity(self):
        v=super().identity();v.update(kind=KIND,vocabulary=9223,duration_rows=[9223,9228],
            migration=self.migration,existing_fresh512_Adam_initial_clock=2000,appended512_Adam_initial_clock=0,
            fresh_parameter_state_base_clocks=[2000]*3+[0]*3,original_three_Adam_objects_preserved=True)
        return v

    def state_dict(self):
        require(not self.poisoned,'Cannot save a partial update');self.validate(full=True)
        return {'kind':KIND,'identity':self.identity(),'updates':self.updates,
            'clocks':[c+self.updates for c in self.base_clocks],'appended512_Adam_clock':self.updates,
            'optimizers':[o.state_dict() for o in self.optimizers]}

    def load_state_dict(self, saved):
        require(not self.poisoned and saved['kind']==KIND and saved['identity']==self.identity(),'Exact9223 own optimizer identity')
        count=saved['updates'];require(type(count) is int and count>=0 and saved['clocks']==[c+count for c in self.base_clocks]
            and saved['appended512_Adam_clock']==count and len(saved['optimizers'])==3,'No clock reset on exact own resume')
        for index,(optimizer,state,policy) in enumerate(zip(self.optimizers,saved['optimizers'],self.group_policies)):
            expected={**policy,'lr':encoder_lr(62250+count)} if index==2 else policy
            require(len(state['param_groups'])==1 and {k:v for k,v in state['param_groups'][0].items() if k!='params'}==expected
                and state['param_groups'][0]['params']==list(range(len(optimizer.param_groups[0]['params'])))
                and set(state['state'])==set(state['param_groups'][0]['params']),'Exact9223 Adam policies and indexed states')
        self.poisoned=True
        for optimizer,state in zip(self.optimizers,saved['optimizers']):optimizer.load_state_dict(state)
        self.updates=count;self.validate(full=True);self.poisoned=False
