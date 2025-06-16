# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from tutel import ops

from torch.utils.cpp_extension import IS_HIP_EXTENSION


def moe_forward(x, topk_ids, topk_weights, B, B_post, C_prev, C, moe_align_sort_fn=None):
    return ops.glu_expert_bf16xf8_noshared(x, topk_ids, topk_weights, B, B.scale_inv, B_post, B_post.scale_inv, C_prev, C_prev.scale_inv, C, C.scale_inv)

