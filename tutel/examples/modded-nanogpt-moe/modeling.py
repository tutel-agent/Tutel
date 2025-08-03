import os
import sys
sys.dont_write_bytecode = True

import uuid
import time
import copy
import glob
import json
from dataclasses import dataclass
from functools import lru_cache, partial # Added partial for hook registration
from pathlib import Path

from tutel import system, net, moe

use_moe = int(os.environ.get("USE_MOE", 1))

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch

from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# -----------------------------------------------------------------------------

class ManagedGemm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        z = x @ y.t()
        ctx.save_for_backward(x, y)
        if hasattr(ManagedGemm, 'lr'):
           ctx.lr = ManagedGemm.lr
        return z

    @staticmethod
    def backward(ctx, dz):
        x, y = ctx.saved_tensors
        dx = dz @ y
        dy = dz.view(-1, dz.size(-1)).t() @ x.view(-1, x.size(-1))
        dy, _ = None, apply_gradient(y, y.state, dy, group={"lr": ctx.lr})
        return (dx, dy,)

    @staticmethod
    def call(x, y, naive=False):
        if naive or not hasattr(y, 'state'):
           return x @ y.t()
        else:
           return ManagedGemm.apply(x, y)

device = "cuda"
batch_size, max_length = 1, 4096

class Qwen3(torch.nn.Module):
    def __init__(self, state_dict, config):
        super(Qwen3, self).__init__()
        load = lambda key: state_dict[key]
        param = lambda t, trainable=True: torch.nn.Parameter(t.to(device)) if trainable else t.to(device)

        self.n_layers = int(os.environ.get('LAYER', config['num_hidden_layers']))
        self.head_dim = config['head_dim']
        self.q_head_dim = config['num_attention_heads']
        self.kv_head_dim = config['num_key_value_heads']

        self.token_emb = param(load('model.embed_tokens.weight'), False)
        self.lm_head = param(load('lm_head.weight'))

        self.rms_att_w = param(torch.cat([
            load(f'model.layers.{l}.input_layernorm.weight').unsqueeze(0) for l in range(self.n_layers)
        ] + [load('model.norm.weight').unsqueeze(0),]))

        self.rms_ffn_w = param(torch.cat([load(f'model.layers.{l}.post_attention_layernorm.weight').unsqueeze(0) for l in range(self.n_layers)]))

        self.qk_norm = param(torch.cat([torch.cat([
            load(f'model.layers.{l}.self_attn.q_norm.weight').unsqueeze(0),
            load(f'model.layers.{l}.self_attn.k_norm.weight').unsqueeze(0),
        ]).unsqueeze(0) for l in range(self.n_layers)]))

        self.qkv_proj = torch.nn.ParameterList([param(torch.cat([
            load(f'model.layers.{l}.self_attn.q_proj.weight'),
            load(f'model.layers.{l}.self_attn.k_proj.weight'),
            load(f'model.layers.{l}.self_attn.v_proj.weight'),
        ])) for l in range(self.n_layers)])

        self.o_proj = torch.nn.ParameterList([param(load(f'model.layers.{l}.self_attn.o_proj.weight')) for l in range(self.n_layers)])

        self.gate_up_p = torch.nn.ParameterList([param(torch.cat([
            load(f'model.layers.{l}.mlp.gate_proj.weight'),
            load(f'model.layers.{l}.mlp.up_proj.weight'),
        ]).unsqueeze(0)) for l in range(self.n_layers)])

        self.down_p = torch.nn.ParameterList([param(torch.cat([
            load(f'model.layers.{l}.mlp.down_proj.weight'),
        ]).unsqueeze(0)) for l in range(self.n_layers)])

        freqs = 1 / (config['rope_theta'] ** (torch.arange(0, self.head_dim // 2, device=device) / (self.head_dim / 2.0))).flatten()
        self.freq_emb = torch.cat((freqs, freqs), dim=-1)
        self.kv_cache = torch.zeros([2, self.n_layers, batch_size, max_length, self.kv_head_dim, self.head_dim], dtype=self.qkv_proj[0].dtype, device=self.qkv_proj[0].device)

        '''
        self.down_p_lorank_a = param(torch.rand(self.n_layers, 1, 32, 3072, dtype=self.down_p.dtype, device=self.down_p.device) / math.sqrt(3072))
        self.down_p_lorank_b = param(torch.rand(self.n_layers, 1, 1024, 32, dtype=self.down_p.dtype, device=self.down_p.device) / math.sqrt(32))
        with torch.no_grad():
            V = (self.down_p_lorank_b @ self.down_p_lorank_a)
            self.down_p.copy_(self.down_p - V) '''


    def gemm(self, x, y, out=None):
        return torch.matmul(x, y.t(), out=out)

    def rms_norm(self, x, weight, eps=1e-6):
        input_dtype = x.dtype
        x = x.float()
        variance = (x * x).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        return weight * x.to(input_dtype)

    def add_norm(self, x, xb, weight, eps=1e-6):
        x = x + xb
        return x, self.rms_norm(x, weight, eps)

    def apply_rotary_pos_emb(self, q, k, position_ids):
        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        emb = position_ids.to(self.freq_emb.dtype) @ self.freq_emb.view(1, -1)
        cos, sin = emb.cos().to(q.dtype), emb.sin().to(k.dtype)
        cos = cos.view(*q.shape[:-2], -1, q.size(-1))
        sin = sin.view(*q.shape[:-2], -1, q.size(-1))
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    def attn(self, q, k_buffer, v_buffer, sm_scale, layer_id, offset):
        # return torch.nn.functional.scaled_dot_product_attention(q_states.transpose(1, 2), k_states.transpose(1, 2), v_states.transpose(1, 2), scale=1 / math.sqrt(self.head_dim), enable_gqa=True, is_causal=True).transpose(1, 2)

        def fn(grad):
            print(grad.shape, grad.reshape(grad.size(1), -1).sum(-1)); exit(0)
            return grad

        def fn_noexit(grad):
            print(grad.shape, grad.reshape(grad.size(1), -1).sum(-1))
            return grad

        # q.register_hook(fn)
        SI, S = q.size(1), k_buffer.size(1)
        # 内存占用变大
        temp_mask = torch.ones(SI, S, dtype=torch.bool, device=q.device).tril(diagonal=0)
        attn_bias = torch.zeros(SI, S, dtype=q.dtype, device=q.device). \
            masked_fill_(temp_mask.logical_not(), float("-inf"))

        ''' if L == 0 and R == 32:
            torch.save(k_buffer.detach(), f'cache/k_in_{layer_id}.pt')
            torch.save(v_buffer.detach(), f'cache/v_in_{layer_id}.pt')
            k_buffer.register_hook(fn)
        elif L == 32 and R == 64:
            k_in = torch.load(f'cache/k_in_{layer_id}.pt').requires_grad_()
            v_in = torch.load(f'cache/v_in_{layer_id}.pt').requires_grad_()
            attn_bias = torch.cat([torch.zeros_like(attn_bias), attn_bias], dim=-1)

            k_buffer.register_hook(fn)  # 分散的一部分grad （除去v_in）
            k_in.register_hook(fn_noexit)

            k_buffer = torch.cat([k_in, k_buffer], dim=1)
            v_buffer = torch.cat([v_in, v_buffer], dim=1)

            k_buffer.register_hook(fn_noexit)
        else:
            k_buffer.register_hook(fn) '''

        if not q.requires_grad:
            self.kv_cache[0, layer_id, :, offset:offset + k_buffer.size(1)] = k_buffer
            self.kv_cache[1, layer_id, :, offset:offset + k_buffer.size(1)] = v_buffer

        k_buffer = torch.cat([self.kv_cache[0, layer_id, :, :offset], k_buffer], dim=1)
        v_buffer = torch.cat([self.kv_cache[1, layer_id, :, :offset], v_buffer], dim=1)
        attn_bias = torch.cat([torch.zeros([SI, offset], dtype=attn_bias.dtype, device=attn_bias.device), attn_bias], dim=-1)

        qk = torch.einsum('bthHm,bshm->bthHs', [
            q.view(q.size(0), q.size(1), k_buffer.size(2), -1, q.size(-1)), k_buffer])
        qk = torch.softmax(qk * sm_scale + attn_bias.view(1, SI, 1, 1, -1), dim=-1)

        o = torch.einsum('bthHs,bshm->bthHm', [qk, v_buffer]).reshape(q.size())
        # o.register_hook(fn)
        return o

    def glu_ffn(self, x, layer_id):
        gate_up, down = self.gate_up_p[layer_id][0], self.down_p[layer_id][0]

        x = (x @ gate_up.t())
        x = torch.nn.functional.silu(x.narrow(-1, 0, x.size(-1) // 2)) * x.narrow(-1, x.size(-1) // 2, x.size(-1) // 2)

        if hasattr(self, 'down_p_lorank_a'):
            x = (x @ down.t()) + (x @ self.down_p_lorank_a[layer_id][0].t()) @ self.down_p_lorank_b[layer_id][0].t()
        else:
            x = x @ down.t()
        return x

    def rotary_emb(self, qkv_out, layer_id, offset):
        b, s, l = qkv_out.size(0), qkv_out.size(1), layer_id
        q_states, k_states, v_states = \
            self.rms_norm(qkv_out.narrow(-2, 0, self.q_head_dim), self.qk_norm[l][0]), \
            self.rms_norm(qkv_out.narrow(-2, self.q_head_dim, self.kv_head_dim), self.qk_norm[l][1]), \
            qkv_out.narrow(-2, self.q_head_dim + self.kv_head_dim, self.kv_head_dim)

        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        def apply_rotary_pos_emb(q, k, freq_emb, position_ids):
            emb = position_ids.to(freq_emb.dtype) @ freq_emb.view(1, -1)
            cos, sin = emb.cos().to(q.dtype), emb.sin().to(k.dtype)
            cos = cos.view(*q.shape[:-2], -1, q.size(-1))
            sin = sin.view(*q.shape[:-2], -1, q.size(-1))
            q_embed = (q * cos) + (rotate_half(q) * sin)
            k_embed = (k * cos) + (rotate_half(k) * sin)
            return q_embed, k_embed

        position_ids = (torch.arange(0, b * s, dtype=torch.int32, device=qkv_out.device) % s).view(b, s, 1) + offset
        q_states, k_states = apply_rotary_pos_emb(q_states, k_states, self.freq_emb, position_ids)

        # q_states.register_hook(fn)
        # v_states.register_hook(fn)
        return q_states, k_states, v_states

    def gemm(self, x, y):
        return x @ y.t()
        # return ManagedGemm.call(xb, self.qkv_proj[l])

    def forward(self, token_in, target_seq, offset=0, out=None):
        token_in = token_in.view(1, -1)
        x = self.token_emb.index_select(0, token_in.flatten()).view(*token_in.shape, self.token_emb.size(-1))
        xb = self.rms_norm(x, self.rms_att_w[0])
        # xb.register_hook(fn)

        for l in range(self.n_layers):
            qkv_out = self.gemm(xb, self.qkv_proj[l]).view(x.size(0), x.size(1), -1, self.head_dim)

            # q_states, k_states, v_states = self.rotary_emb(qkv_out, l, offset)
            q_states, k_states, v_states = checkpoint(self.rotary_emb, qkv_out, l, offset, use_reentrant=True)

            import math
            scores = self.attn(q_states, k_states, v_states, sm_scale=1 / math.sqrt(self.head_dim), layer_id=l, offset=offset)
            xb = self.gemm(scores.flatten(-2), self.o_proj[l])

            x, xb = self.add_norm(x, xb, self.rms_ffn_w[l])
            xb = self.glu_ffn(xb, l)
            x, xb = self.add_norm(x, xb, self.rms_att_w[l + 1])

        out = self.gemm(xb, self.lm_head)
        # out.register_hook(fn)
        # return out

        logits = out.view(-1, out.size(-1))
        log_probs = F.log_softmax(logits, dim=1)
        true_class_log_probs = log_probs.gather(1, target_seq.unsqueeze(1)).squeeze(1)
        loss = -torch.mean(true_class_log_probs)
        # loss2 = F.cross_entropy(logits.view(-1, logits.size(-1)), target_seq) # , reduction="sum" if self.training else "mean")
        # loss.l_aux = 0.0
        return loss

def get_model():
    # 加载模型参数
    try:
        model_path = f'Qwen/Qwen3-0.6B'

        from safetensors.torch import safe_open, save_file
        state_dict = {}
        for f in os.listdir(model_path):
          if f.endswith('.safetensors'):
            with safe_open(f'{model_path}/{f}', framework='pt') as f:
              for k in f.keys():
                 state_dict[k] = f.get_tensor(k)

        with open(f'{model_path}/config.json', 'r') as fp:
          config = json.loads(fp.read())

    except Exception as e:
        print(f"Failed to load model from {model_path}: {e}")
        raise

    graph = Qwen3(state_dict, config).to(device)
    return graph
