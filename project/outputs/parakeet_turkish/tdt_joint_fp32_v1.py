"""Opt-in instance-only FP32 joint policy; a NEW numerical recipe.

Encoder and decoder remain under their declared BF16 autocast policy. Only
their projected tensors enter FP32 addition, ReLU and the FP32 joint head.
This does not claim the original BF16 trajectory or alter active tensors.
"""
import threading
from types import MethodType
import torch
from torch.nn import functional as F

POLICY={'kind':'ASR18_joint_add_relu_linear_FP32_v1','addition':'float32','activation':'float32_relu',
        'linear_forward_and_backward':'float32','allow_tf32':False,'full_vocabulary':8711,'durations':5,
        'encoder_decoder_autocast_unchanged':True,'original_BF16_trajectory_equivalence_claimed':False,
        'active_weights_dtype_or_optimizer_changed':False}

def forward(self,encoder_hidden_states,decoder_hidden_states):
    if threading.current_thread() is not threading.main_thread():raise ValueError('Serial joint policy only')
    with torch.autocast('cuda',enabled=False):
        return F.linear(F.relu(encoder_hidden_states.float()+decoder_hidden_states.float()),self.head.weight,self.head.bias)

def apply(joint):
    if hasattr(joint,'_ASR18_fp32_joint_policy'):raise ValueError('Joint policy already applied')
    if not isinstance(joint.head,torch.nn.Linear) or joint.head.out_features!=8716:raise ValueError('Pinned linear full8711+5 head required')
    if len(tuple(joint.parameters()))!=2 or any(p.dtype!=torch.float32 for p in joint.parameters()):raise ValueError('Existing two FP32 head parameters required')
    if torch.backends.cuda.matmul.allow_tf32:raise ValueError('FP32 policy requires TF32 disabled explicitly')
    joint.forward=MethodType(forward,joint);joint._ASR18_fp32_joint_policy=dict(POLICY)
    return dict(POLICY)
