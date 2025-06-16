#!/usr/bin/env python3

import functools
import json
import logging
import os, sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from tutel import ops

torch.manual_seed(0)


def ref_moe_full(x, topk_ids, topk_weights, B, B_post, C_prev, C, **kwargs):
 
    def ref_to_bfloat16(x, x_scal):
        if x.dtype == torch.bfloat16:
            return x
        if x.dtype == torch.uint8:
            x = x.view(torch.float8_e4m3fnuz)
        assert x.dim() == x_scal.dim() and x.dim() >= 2
        if x.size(-2) == x_scal.size(-2):
            x_bf16 = (x.view(-1, 128).float() * x_scal.view(-1, 1)).bfloat16().view(x.shape)
        else:
            B = x
            x_bf16 = (B.view(B.size(0), B.size(1) // 128, 128, B.size(2) // 128, 128).float() * x_scal.view(B.size(0), B.size(1) // 128, 1, B.size(2) // 128, 1)).flatten(1, 2).flatten(2, 3).bfloat16()
        return x_bf16

    A = ref_to_bfloat16(*ops.to_float8_per_token(x, 128))

    B_out = torch.einsum('bk,bmk->bm', [A.repeat_interleave(8, dim=0), ref_to_bfloat16(B, B.scale_inv).index_select(0, topk_ids.flatten()).view(-1, B.size(-2), B.size(-1))])

    B_post_out = torch.einsum('bk,bmk->bm', [B_out, ref_to_bfloat16(B_post, B_post.scale_inv).index_select(0, topk_ids.flatten()).view(-1, B_post.size(-2), B_post.size(-1))])

    B_post_act = torch.nn.functional.silu(B_post_out.narrow(-1, 0, B_post_out.size(-1) // 2)) * B_post_out.narrow(-1, B_post_out.size(-1) // 2, B_post_out.size(-1) // 2)

    C_prev_out = torch.einsum('bk,bmk->bm', [B_post_act, ref_to_bfloat16(C_prev, C_prev.scale_inv).index_select(0, topk_ids.flatten()).view(-1, C_prev.size(-2), C_prev.size(-1))])

    C_prev_out = ref_to_bfloat16(*ops.to_float8_per_token(C_prev_out, 128))

    C_out = torch.einsum('bk,bmk->bm', [C_prev_out, ref_to_bfloat16(C, C.scale_inv).index_select(0, topk_ids.flatten()).view(-1, C.size(-2), C.size(-1))])

    return (C_out.view(*topk_ids.shape, -1) * topk_weights.unsqueeze(-1)).sum(dim=1)

def compare(L, R):  
    print('Opt:', L.shape, L.sum(-1).flatten())
    print('Ref:', R.shape, R.sum(-1).flatten())
    if L.shape != R.shape:
        return print('Shapes differ', L.shape, R.shape)
    diff = (R - L).abs().argmax().item()
    print('Max Diff:', (R - L).abs().max().item(), L.flatten()[diff], 'v.s.', R.flatten()[diff])

def fp8_gen(shape):
    v = (torch.randn(shape, device='cuda') / 100).to(torch.float8_e4m3fnuz)
    v.scale_inv = torch.randn(list(shape[:-2]) + [(shape[-2] + 128 - 1) // 128, (shape[-1] + 128 - 1) // 128], device='cuda')
    return v


def main():
    E = 256
    policy = os.environ.get('POLICY', 'moe_forward_policy_32')
    batch = int(os.environ.get('BATCH', 32))

    A = torch.randn([batch, 7168], device='cuda', dtype=torch.bfloat16)
    B = fp8_gen([E, 256, 7168])
    B_post = fp8_gen([E, 512, 256])
    C_prev = fp8_gen([E, 128, 256])
    C = fp8_gen([E, 7168, 128])
    topk_ids = torch.randint(0, E, [batch, 8], device='cuda', dtype=torch.int32)
    topk_weights = torch.randn(topk_ids.shape, device='cuda')

    if policy == 'moe_forward_policy_32':
      from tutel.ops.moe_forward_policy_32 import moe_forward as opt_moe_full
    elif policy == 'moe_forward_policy_3200':
      from tutel.ops.moe_forward_policy_3200 import moe_forward as opt_moe_full
    else:
      raise Exception(f'Unrecognized policy typr: {policy}')

    L = opt_moe_full(A, topk_ids, topk_weights, B, B_post, C_prev, C)

    from torch.profiler import profile, record_function, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
      with record_function("model_inference"):
        for i in range(100):
          opt_moe_full(A, topk_ids, topk_weights, B, B_post, C_prev, C)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))

    if batch <= 3200:
      R = ref_moe_full(A, topk_ids, topk_weights, B, B_post, C_prev, C)
      compare(L, R)

if __name__ == '__main__':
    main()

