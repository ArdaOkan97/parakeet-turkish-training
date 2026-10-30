"""UNADMITTED target-class sufficient statistic, for mathematical fixtures only.

At lattice state(t,u), target-constrained TDT paths can emit blank or y[u].
Every other lexical logit contributes only to the normalization denominator.
Combine those other logits with logsumexp into an unreachable sink. This keeps
the complete vocabulary normalization; it neither truncates nor renormalizes a
selected vocabulary. DP/TDT runtime integration is intentionally not present.
"""
def compress_lexical(logits,target_by_u,blank_id,*,torch_module):
    torch=torch_module
    if logits.ndim!=4 or target_by_u.ndim!=1 or target_by_u.numel()!=logits.shape[2]:raise ValueError('B,T,U,V logits and exact per-state target IDs required')
    if logits.shape[-1]<3 or not (0<=blank_id<logits.shape[-1]):raise ValueError('Vocabulary/blank geometry')
    if not bool(((target_by_u>=0)&(target_by_u<logits.shape[-1])&(target_by_u!=blank_id)).all()):raise ValueError('Positive lexical target, excluding blank')
    indices=target_by_u.to(device=logits.device,dtype=torch.long)[None,None,:,None].expand(*logits.shape[:-1],1)
    target=logits.gather(-1,indices).squeeze(-1);blank=logits[...,blank_id]
    other=logits.clone();other[...,blank_id]=-float('inf');other.scatter_(-1,indices,-float('inf'))
    sink=torch.logsumexp(other,dim=-1)
    # Blank-last keeps the official kernel's documented blank convention.
    return torch.stack((target,sink,blank),dim=-1)
