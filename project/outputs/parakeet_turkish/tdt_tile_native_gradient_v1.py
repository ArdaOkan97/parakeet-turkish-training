# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Copyright 2018-2019, Mingkun Huang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Owned tile-index adapter of pinned official TDT gradient; equations unchanged.
Only acts/lexical-gradient physical addressing and global(t,u) origin differ.
Duration, norm, alpha/beta/likelihood indexing remains the full original DP.
"""
import math
import torch
from numba import cuda
from nemo.collections.asr.parts.numba.rnnt_loss.utils.cuda_utils.gpu_rnnt_kernel import GPU_RNNT_THREAD_SIZE

@cuda.jit()
def tile_tdt_grad_kernel(
    label_grads: torch.Tensor,
    duration_grads: torch.Tensor,
    acts: torch.Tensor,
    duration_acts: torch.Tensor,
    denom: torch.Tensor,
    sigma: float,
    alphas: torch.Tensor,
    betas: torch.Tensor,
    logll: torch.Tensor,
    xlen: torch.Tensor,
    ylen: torch.Tensor,
    mlabels: torch.Tensor,  # [B, U]
    minibatch: int,
    maxT: int,
    maxU: int,
    alphabet_size: int,
    tile_t_begin: int,
    tile_u_begin: int,
    tile_u_size: int,
    blank_: int,
    durations: torch.Tensor,
    num_durations: int,
    fastemit_lambda: float,
    clamp: float,
):
    """
    Compute gradients over the transduction step.

    Args:
        grads: Zero Tensor of shape [B, T, U, V] to store gradients for tokens.
        duration_grads: Zero Tensor of shape [B, T, U, D] to store gradients for durations.

        acts: Tensor of shape [B, T, U, V] flattened. Represents the logprobs activation tensor for tokens.
        duration_acts: Tensor of shape [B, T, U, D] flattened. Represents the logprobs activation tensor for durations.
        denom: Tensor of shape [B, T, U] flattened. Represents the denominator of the logprobs activation tensor
            across entire vocabulary.
        alphas: Alpha variable, contains forward probabilities. A tensor of shape [B, T, U].
        betas: Beta varoable, contains backward probabilities. A tensor of shape [B, T, U].
        logll: Log-likelihood of the forward variable, represented as a vector of shape [B].
            Represents the log-likelihood of the forward pass.
        xlen: Vector of length B which contains the actual acoustic sequence lengths in the padded
            activation tensor.
        ylen: Vector of length B which contains the actual target sequence lengths in the padded
            activation tensor.
        mlabels: Matrix of shape [B, U+1] (+1 here is due to <SOS> token - usually the RNNT blank).
            The matrix contains the padded target transcription that must be predicted.
        minibatch: Int representing the batch size.
        maxT: The maximum possible acoustic sequence length. Represents T in the logprobs tensor.
        maxU: The maximum possible target sequence length. Represents U in the logprobs tensor.
        alphabet_size: The vocabulary dimension V+1 (inclusive of RNNT blank).
        blank_: Index of the RNNT blank token in the vocabulary. Generally the first or last token in the vocab.
        fastemit_lambda: Float scaling factor for FastEmit regularization. Refer to
            FastEmit: Low-latency Streaming ASR with Sequence-level Emission Regularization.
        clamp: Float value. When set to value >= 0.0, will clamp the gradient to [-clamp, clamp].

    Updates:
        Kernel inplace updates the following inputs:
        -   grads: Gradients with respect to the log likelihood (logll).
    """
    # Kernel call:
    # blocks_per_grid = minibatch (b) * maxT (t) * maxU (u)
    # threads_per_block = constant buffer size of parallel threads (v :: Constant)
    tid = cuda.threadIdx.x  # represents v, taking steps of some constant size
    idx = tid  # index of v < V+1; in steps of constant buffer size
    local_col = cuda.blockIdx.x
    col = (tile_t_begin + local_col // tile_u_size) * maxU + tile_u_begin + local_col % tile_u_size

    # Decompose original indices from fused `col`
    u = col % maxU  # (b * t * u) % u = u
    bt = (col - u) // maxU  # (b * t * u - u) // U = b * t
    t = bt % maxT  # (b * t) % t = t
    mb = (bt - t) // maxT  # (b * t - t) // T = b

    # constants
    T = xlen[mb]  # select AM length of current sample
    U = ylen[mb] + 1  # select target length of current sample, +1 for the blank token
    labels: torch.Tensor = mlabels[mb]  # labels = mlabels + mb * (maxU - 1);

    # Buffered gradient calculations, broadcast across B=b, T=t and U=u, looped over V with some constant stride.
    # Look up gradient calculation from rnnt_numpy.compute_gradient()

    if t < T and u < U:
        logpk_blank = (
            denom[col] + acts[local_col * alphabet_size + blank_] - sigma
        )  # whenever sigma is used, it is for logit under-normalization.

        if idx < num_durations:
            grad = 0.0
            if t + durations[idx] < T and u < U - 1:  # for label
                logpk_label = denom[col] + acts[local_col * alphabet_size + labels[u]] - sigma
                grad -= math.exp(alphas[col] + betas[col + 1 + durations[idx] * maxU] + logpk_label - logll[mb])

            if t + durations[idx] < T and durations[idx] > 0:  # for blank in the middle
                grad -= math.exp(alphas[col] + betas[col + durations[idx] * maxU] + logpk_blank - logll[mb])

            if t + durations[idx] == T and u == U - 1 and durations[idx] > 0:  # for blank as the last symbol
                grad -= math.exp(alphas[col] + logpk_blank - logll[mb])

            grad = grad * math.exp(duration_acts[col * num_durations + idx])
            duration_grads[col * num_durations + idx] = grad

        # For cuda kernels, maximum number of threads per block is limited to some value.
        # However, it may be the case that vocabulary size is larger than this limit
        # To work around this, an arbitrary thread buffer size is chosen such that,
        # 1) each element within the thread pool operates independently of the other
        # 2) An inner while loop moves the index of each buffer element by the size of the buffer itself,
        #    such that all elements of the vocabulary size are covered in (V + 1 // thread_buffer) number of steps.
        # As such, each thread will perform the while loop at least (V + 1 // thread_buffer) number of times
        while idx < alphabet_size:
            # remember, `col` represents the tri-index [b, t, u]
            # therefore; logpk = denom[b, t, u] + acts[b, t, u, v]
            logpk = denom[col] + acts[local_col * alphabet_size + idx]
            # initialize the grad of the sample acts[b, t, u, v]
            grad = math.exp(alphas[col] + betas[col] + logpk - logll[mb])

            # If FastEmit regularization is enabled, calculate the gradeint of probability of predicting the next label
            # at the current timestep.
            # The formula for this is Equation 9 in https://arxiv.org/abs/2010.11148, multiplied by the log probability
            # of the current step (t, u), normalized by the total log likelihood.
            # Once the gradient has been calculated, scale it by `fastemit_lambda`, as in Equation 10.
            if fastemit_lambda > 0.0 and u < U - 1:
                fastemit_grad = 0.0

                for i in range(num_durations):
                    if t + durations[i] < T:
                        fastemit_grad += fastemit_lambda * math.exp(
                            alphas[col]  # alphas(t, u)
                            + (denom[col] + acts[local_col * alphabet_size + labels[u]])  # log prob of token emission
                            + duration_acts[col * num_durations + i]  # duration log-prob
                            + betas[col + 1 + durations[i] * maxU]  # betas(t, u+1)
                            + logpk  # log Pr(k|t, u)
                            - sigma  # for logit under-normalization
                            - logll[mb]  # total log likelihood for normalization
                        )
            else:
                fastemit_grad = 0.0

            # Update the gradient of act[b, t, u, v] with the gradient from FastEmit regularization
            grad = grad + fastemit_grad

            # grad to last blank transition
            # grad[b, T-1, U-1, v=blank] -= exp(alphas[b, t, u] + logpk - sigma - logll[b] + logp(duration) for all possible non-zero durations.
            if idx == blank_ and u == U - 1:
                for i in range(num_durations):
                    if durations[i] == 0:
                        continue
                    if t == T - durations[i]:
                        grad -= math.exp(
                            alphas[col] + logpk - sigma - logll[mb] + duration_acts[col * num_durations + i]
                        )

            # grad of blank across t < T;
            # grad[b, t<T-1, u, v=blank] -= exp(alphas[b, t, u] + logpk - sigma + logp_duration - logll[b] + betas[b, t + duration, u]) for all non-zero durations
            if idx == blank_:
                for i in range(num_durations):
                    if durations[i] == 0:
                        continue
                    if t < T - durations[i]:
                        grad -= math.exp(
                            alphas[col]
                            + logpk
                            - sigma
                            - logll[mb]
                            + betas[col + maxU * durations[i]]
                            + duration_acts[col * num_durations + i]
                        )

            # grad of correct token across u < U;
            # grad[b, t, u<U-1, v=label[u]] -= exp(alphas[b, t, u] + logpk - sigma + logp_duration - logll[b] + betas[b, t + duration, u + 1]) for all blank durations.
            # Scale the gradient by (1.0 + FastEmit_lambda) in log space, then exponentiate
            if u < U - 1 and idx == labels[u]:
                # exp(log(1 + fastemit_lambda) + ...) is numerically more stable than
                # multiplying (1.0 + fastemit_lambda) with result.
                for i in range(num_durations):
                    if t + durations[i] < T:
                        grad -= math.exp(
                            math.log1p(fastemit_lambda)
                            + alphas[col]
                            + logpk
                            - sigma
                            - logll[mb]
                            + betas[col + 1 + maxU * durations[i]]
                            + duration_acts[col * num_durations + i]
                        )

            # update grads[b, t, u, v] = grad
            label_grads[local_col * alphabet_size + idx] = grad

            # clamp gradient (if needed)
            if clamp > 0.0:
                g = label_grads[local_col * alphabet_size + idx]
                g = min(g, clamp)
                g = max(g, -clamp)
                label_grads[local_col * alphabet_size + idx] = g

            # update internal index through the thread_buffer;
            # until idx < V + 1, such that entire vocabulary has been updated.
            idx += GPU_RNNT_THREAD_SIZE

