"""Original two-group Adam207, continuous clocks and inherited six-glyph gain."""
from copy import deepcopy
import hashlib
from tsc_three_epoch_transition_v1 import SOURCE_STEP, encoder_lr, require
from bpe_full_decoder_optimizer_v1 import TABLES


class Full207Optimizer:
    def __init__(self,model,state,context,torch_module):
        self.torch=torch_module
        self.parameters={n:p for n,p in model.named_parameters() if p.requires_grad}
        require(set(self.parameters)==set(state["weights"]) and len(self.parameters)==207,"All207 actual trainable tensors required")
        self.names=tuple(self.parameters)
        self.head_names=tuple(n for n in self.names if not n.endswith(("lora_A","lora_B")))
        self.lora_names=tuple(n for n in self.names if n.endswith(("lora_A","lora_B")))
        self.heads=[self.parameters[n] for n in self.head_names]
        self.adapters=[self.parameters[n] for n in self.lora_names]
        self.actual=[*self.heads,*self.adapters]
        require(len(self.heads)==15 and len(self.adapters)==192,"Original15/192 group ordering required")
        groups=state["optimizer"]["param_groups"]
        self.group_policy=[{k:v for k,v in g.items() if k not in ("params","lr")} for g in groups]
        initial=[{**deepcopy(groups[0]),"params":self.heads},{**deepcopy(groups[1]),"params":self.adapters}]
        self.optimizer=self.torch.optim.AdamW(initial)
        self.global_step=state["step"]
        self.load_state_dict(state["optimizer"],self.global_step)

    def load_state_dict(self,state,step):
        groups=state["param_groups"]
        require(len(groups)==2 and groups[0]["params"]==list(range(15))
            and groups[1]["params"]==list(range(15,207)) and set(state["state"])==set(range(207)),"Original Adam IDs/order required")
        for i,g in enumerate(groups):
            require({k:v for k,v in g.items() if k not in ("params","lr")}==self.group_policy[i]
                and g["lr"]==(1e-5 if i==0 else encoder_lr(step)),"Own exact Adam hyperparameters/LR changed")
        for index,p in enumerate(self.actual):
            m=state["state"][index]
            require(set(m)=={"step","exp_avg","exp_avg_sq"} and m["step"].item()==step
                and m["step"].dtype==self.torch.float32
                and all(m[k].shape==p.shape and m[k].dtype==self.torch.float32
                        and bool(self.torch.isfinite(m[k]).all()) for k in ("exp_avg","exp_avg_sq"))
                and bool((m["exp_avg_sq"]>=0).all()),"All207 saved moment tensors/clocks must be exact finite FP32")
        self.optimizer.load_state_dict(state)
        self.global_step=step

    def zero_grad(self,**kwargs):self.optimizer.zero_grad(**kwargs)

    def step(self):
        step=self.global_step+1
        require(all(p.grad is not None for p in self.actual),"All207 gradients required")
        self.optimizer.param_groups[0]["lr"]=1e-5
        self.optimizer.param_groups[1]["lr"]=encoder_lr(step)
        # Preserve the frozen producer's adapter-then-head clipping order;
        # Adam serialization remains head15 followed by adapter192.
        norm=self.torch.nn.utils.clip_grad_norm_([*self.adapters,*self.heads],1.,error_if_nonfinite=True)
        before=[self.parameters[n][8193:8199].detach().clone() for n in TABLES]
        self.optimizer.step()
        with self.torch.no_grad():
            for n,old in zip(TABLES,before):
                rows=self.parameters[n][8193:8199]
                rows.copy_(old+6*(rows-old))
        self.global_step=step
        require(all(self.optimizer.state[p]["step"].item()==step for p in self.actual),"All207 actual Adam clocks must advance")
        return norm

    def state_dict(self):return self.optimizer.state_dict()

    def fingerprints(self):
        return {n:hashlib.sha256(memoryview(self.parameters[n].detach().cpu().contiguous().reshape(-1).view(self.torch.uint8).numpy())).hexdigest() for n in self.names}
