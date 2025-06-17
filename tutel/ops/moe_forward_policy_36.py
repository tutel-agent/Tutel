# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from tutel import ops

from torch.utils.cpp_extension import IS_HIP_EXTENSION

import triton
import triton.language as tl

scale_width = 128

@triton.jit
def weighted_sum_kernel(x_ptr,
               w_ptr,
               y_ptr,
               N, M,
               BLOCK_SIZE: tl.constexpr,
               ):
    bx = tl.program_id(axis=0)
    tx = tl.arange(0, BLOCK_SIZE)
    n0, m0 = bx // tl.cdiv(M, 1024), bx % tl.cdiv(M, 1024)

    T: tl.constexpr = 8
    mask = (m0 * 1024 + tx < M)
    x = tl.load(x_ptr + n0.to(tl.int64) * M * T + (m0 * 1024 + tx), mask=mask).to(tl.float32) * tl.load(w_ptr + (n0 + tx // BLOCK_SIZE) * T, mask=mask)
    for i in tl.static_range(1, T):
      x = x + tl.load(x_ptr + n0.to(tl.int64) * M * T + (m0 * 1024 + tx) + (i * M), mask=mask).to(tl.float32) * tl.load(w_ptr + (n0 + tx // BLOCK_SIZE) * T + i, mask=mask)
    tl.store(y_ptr + n0 * M + (m0 * 1024 + tx), x.to(tl.bfloat16), mask=mask)

def weighted_sum(x, w):
    y = torch.empty([x.size(0), x.size(2)], dtype=x.dtype, device=x.device)
    grid = lambda meta: (x.size(0) * triton.cdiv(x.size(2), meta['BLOCK_SIZE']),)
    weighted_sum_kernel[grid](x, w, y, x.size(0), x.size(2), BLOCK_SIZE=1024)
    return y


@triton.jit
def fused_moe_kernel_ex(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bse,
    stride_bsk,
    stride_bsn,
    # Block size for block-wise quantization
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    use_a_scale: tl.constexpr,
    compute_type: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    per_channel_quant: tl.constexpr,
    even_Ks: tl.constexpr,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.

    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    offs_token = offs_token.to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )

    off_experts = tl.load(expert_ids_ptr + pid_m)
    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )
    if use_int8_w8a16:
        b_scale_ptrs = (
            b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
        )
        b_scale = tl.load(b_scale_ptrs)

    if use_fp8_w8a8 or use_int8_w8a8:
        # block-wise
        if group_k > 0 and group_n > 0:
            if use_a_scale:
                a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            offs_bsn = offs_bn // group_n
            b_scale_ptrs = (
                b_scale_ptr + off_experts * stride_bse + offs_bsn * stride_bsn
            )
        # channel-wise
        elif per_channel_quant:
            b_scale_ptrs = (
                b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
            )
            b_scale = tl.load(b_scale_ptrs)
            # Load per-token scale for activations
            if use_a_scale:
                a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
                a_scale = tl.load(a_scale_ptrs, mask=token_mask, other=0.0)[:, None]
        # tensor-wise
        else:
            if use_a_scale:
                a_scale = tl.load(a_scale_ptr)
            b_scale = tl.load(b_scale_ptr + off_experts)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.
        if even_Ks:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None],
                other=0.0,
            )
            b = tl.load(b_ptrs)
        else:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                other=0.0,
            )
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        # We accumulate along the K dimension.
        if use_int8_w8a16:
            accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
        elif use_fp8_w8a8 or use_int8_w8a8:
            if group_k > 0 and group_n > 0:
                k_start = k * BLOCK_SIZE_K
                offs_ks = k_start // group_k
                b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)

                if use_a_scale:
                    a_scale = tl.load(a_scale_ptrs + offs_ks * stride_ask, mask=token_mask, other=0.0)
                    accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
                else:
                    accumulator += tl.dot(a, b.to(compute_type)) * b_scale[None, :]
            else:
                # fix out of shared memory issue
                if use_fp8_w8a8:
                    accumulator = tl.dot(a, b, acc=accumulator)
                else:
                    accumulator += tl.dot(a, b)
        else:
            accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]
    if use_int8_w8a16:
        accumulator = (accumulator * b_scale).to(compute_type)
    elif use_fp8_w8a8 or use_int8_w8a8:
        if group_k > 0 and group_n > 0:
            accumulator = accumulator.to(compute_type)
        else:
            if use_a_scale:
                accumulator = (accumulator * a_scale * b_scale).to(compute_type)
            else:
                accumulator = (accumulator * b_scale).to(compute_type)
    else:
        accumulator = accumulator.to(compute_type)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def moe_one_layer(A, A_scale, B, B_scale, topk_ids, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded, config):
    assert sorted_token_ids is None or config['BLOCK_SIZE_M'] == sorted_token_ids.block_size_M

    if A.dtype == torch.uint8:
        A = A.view(torch.float8_e4m3fnuz)
    C = torch.empty([*topk_ids.shape, B.size(1)], device=A.device, dtype=torch.bfloat16)
    padded_size = 0
    grid = lambda META: (triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"]) * triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"]),)

    fused_moe_kernel_ex[grid](A, B, C, A_scale, B_scale, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded,
            B.shape[1],
            B.shape[2] - padded_size,
            sorted_token_ids.shape[0],
            topk_weights.numel(),
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(2),
            B.stride(1),
            C.stride(1),
            C.stride(2),
            A_scale.stride(0) if A_scale is not None and A_scale.ndim == 2 else 0,
            A_scale.stride(1) if A_scale is not None and A_scale.ndim == 2 else 0,
            B_scale.stride(0) if B_scale is not None and B_scale.ndim >= 2 else 0,
            B_scale.stride(2) if B_scale is not None and B_scale.ndim == 3 else 0,
            B_scale.stride(1) if B_scale is not None and B_scale.ndim >= 2 else 0,
            scale_width, scale_width,
            MUL_ROUTED_WEIGHT=False,
            top_k=topk_ids.numel() // A.size(0),
            compute_type=tl.bfloat16,
            use_a_scale=(A.dtype != torch.bfloat16),
            use_fp8_w8a8=True,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            per_channel_quant=False,
            even_Ks=((B.shape[2] - padded_size) % config["BLOCK_SIZE_K"] == 0),
            **config,
        )
    return C

@torch.compile
def silu_mul(x):
    return torch.nn.functional.silu(x.narrow(-1, 0, x.size(-1) // 2)) * x.narrow(-1, x.size(-1) // 2, x.size(-1) // 2)

def moe_forward(x, topk_ids, topk_weights, B, B_post, C_prev, C, moe_align_sort_fn=None):
    assert triton.__version__ >= '3.3.0', 'Please use triton>=3.3.0 to ensure reproducible Triton performance.'

    if moe_align_sort_fn is None:
        try:
            from sglang.srt.layers.moe.fused_moe_triton.fused_moe import moe_align_block_size as moe_align_sort_fn
        except:
            from external_moe_sorting import moe_align_block_size as moe_align_sort_fn

    block_size_M = 16
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_sort_fn(topk_ids, block_size_M, B.size(0))
    sorted_token_ids.block_size_M = block_size_M

    x, x_scal = ops.to_float8_per_token(x, scale_width)

    config = {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 4, 'num_warps': 2, 'num_stages': 1, 'waves_per_eu': 2}
    B_out = moe_one_layer(x, x_scal, B, B.scale_inv, topk_ids, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded, config=config).flatten(0, 1)

    config = {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 2, 'num_warps': 2, 'num_stages': 1, 'waves_per_eu': 2}
    B_post_out = moe_one_layer(B_out, B_out, B_post, B_post.scale_inv, topk_ids, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded, config=config).flatten(0, 1)

    B_post_act = silu_mul(B_post_out)

    config = {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 2, 'num_warps': 2, 'num_stages': 1, 'waves_per_eu': 2}
    C_prev_out = moe_one_layer(B_post_act, B_post_act, C_prev, C_prev.scale_inv, topk_ids, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded, config=config).flatten(0, 1)

    x, x_scal = ops.to_float8_per_token(C_prev_out, scale_width)

    config = {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 4, 'num_warps': 4, 'num_stages': 1, 'waves_per_eu': 2}
    C_out = moe_one_layer(x, x_scal, C, C.scale_inv, topk_ids, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded, config=config).flatten(0, 1)

    out = weighted_sum(C_out.view(*topk_ids.shape, -1), topk_weights)
    return out

