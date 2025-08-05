import os
import sys
sys.dont_write_bytecode = True

import uuid
import time
import copy
import glob
from dataclasses import dataclass
from functools import lru_cache, partial # Added partial for hook registration
from pathlib import Path

from tutel import system, net, moe
import autort

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
torch.empty(1, device="cuda", requires_grad=True).backward() # prevents a bug on some systems

torch.manual_seed(0)
import random
random.seed(0)
import numpy as np
np.random.seed(0)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.use_deterministic_algorithms(True)

from torch import Tensor, nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Distributed data loader

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

# find world_size starting indicies, such that each begins with token 50256 and local_batches don't overlap
def find_batch_starts(tokens: Tensor, pos: int, local_batch_size: int, max_batch_span: int):
    boundary_mask = tokens[pos : pos + max_batch_span] == 50256
    boundary_positions = torch.nonzero(boundary_mask, as_tuple=False).squeeze(-1) + pos
    start = boundary_positions[0].item()
    starts = []
    for i in range(1, len(boundary_positions)):
        end = boundary_positions[i].item() 
        if end - start >= local_batch_size:
            starts.append(start) # append start once end pos is confirmed
            if len(starts) == net.get_world_size():
                return starts, end - pos
            start = end
    assert False # increase max_batch_span if necessary

def distributed_data_generator(filename_pattern: str, batch_size: int, align_to_bos: bool):
    rank = net.get_world_rank()
    world_size = net.get_world_size()
    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size
    file_iter = iter(files) # use itertools.cycle(files) instead if you want to do multi-epoch training
    tokens, pos = _load_data_shard(next(file_iter)), 0
    max_batch_span = 2 * batch_size if align_to_bos else batch_size # provide buffer to handle samples up to length local_batch_size
    while True:
        if pos + max_batch_span + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        if align_to_bos:
            batch_starts, batch_span = find_batch_starts(tokens, pos, local_batch_size, max_batch_span)
            start_idx = batch_starts[rank]
        else:
            batch_span = batch_size
            start_idx = pos + rank * local_batch_size
        buf = tokens[start_idx:][:local_batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True) # no sync on host side;
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True) # H2D in another stream isn't helpful.
        pos += batch_span
        yield inputs, targets

# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    # data
    train_files = "data/fineweb10B/fineweb_train_*.bin" # input .bin to train on
    val_files = "data/fineweb10B/fineweb_val_*.bin" # input .bin to eval validation loss on
    val_tokens = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    train_seq_len = 1024 # FlexAttention sequence length
    val_seq_len = train_seq_len # FlexAttention sequence length for validation
    # optimization
    num_iterations = 1750 # number of iterations to run
    cooldown_frac = 0.45 # fraction of training spent cooling down the learning rate
    # evaluation and logging
    val_loss_every = 50 # every how many steps to evaluate val loss? 0 for only at the end
    save_checkpoint = False
    balancing_importance = 0.2
args = Hyperparameters()

# torchrun sets these env variables
assert torch.cuda.is_available()
parallel_env = system.init_data_model_parallel(group_count=1, backend='nccl')
rank = parallel_env.global_rank
world_size = parallel_env.global_size

device = parallel_env.local_device
torch.cuda.set_device(device)
master_process = (rank == 0)
print0 = parallel_env.dist_print

# begin by printing this file (the Python code)
print0("="*100)
# log information about the hardware/software environment this is running on
print0(f"Running Python {sys.version}")
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")
print0("="*100)

from modeling import get_model
model = get_model()

for name, param in model.named_parameters():
    param.param_name = name
    if not hasattr(param, 'expert'):
        net.simple_broadcast(param.detach(), 0)

# optimizers = [torch.optim.Adam(model.parameters(), lr=0.008)]
from rmsprop import RMSprop
optimizers = [RMSprop(model.parameters(), lr=0.008)]

# model: nn.Module = torch.compile(model, dynamic=False)


########################################
#        Training and validation       #
########################################

train_loader = distributed_data_generator(args.train_files, world_size * args.train_seq_len, align_to_bos=False)
training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.perf_counter()
# begin training
train_steps = args.num_iterations
for step in range(train_steps + 1):
    last_step = (step == train_steps)

    # --------------- VALIDATION SECTION -----------------
    if False and (last_step or (step > 0 and step % args.val_loss_every == 0)):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms = 1000 * (time.perf_counter() - t0)
        model.eval()
        val_batch_size = world_size * args.val_seq_len
        assert args.val_tokens % val_batch_size == 0
        val_steps = args.val_tokens // val_batch_size
        val_loader = distributed_data_generator(args.val_files, val_batch_size, align_to_bos=False)
        val_loss, val_l_aux = 0, 0
        with torch.no_grad():
            for _ in range(val_steps):
                inputs, targets = next(val_loader)
                val_loss_step = model(inputs, targets)
                val_loss += val_loss_step
                val_l_aux += getattr(val_loss_step, 'l_aux', 0)
        val_loss /= val_steps
        val_l_aux /= val_steps
        del val_loader
        # net.simple_all_reduce(val_loss, op=torch.distributed.ReduceOp.AVG, inplace=True)
        # if isinstance(val_l_aux, torch.Tensor):
        #     net.simple_all_reduce(val_l_aux, op=torch.distributed.ReduceOp.AVG, inplace=True)
        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.4f} val_balance_loss:{val_l_aux:.4f} step_time:{training_time_ms/args.val_loss_every:.2f}ms")
        model.train()
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    if last_step:
        if master_process and args.save_checkpoint:
            log = dict(step=step, code=code, model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
            os.makedirs(f"logs/{run_id}", exist_ok=True)
            torch.save(log, f"logs/{run_id}/state_step{step:06d}.pt")
        # the last step only has the validation loop, so break to avoid training
        break

    train_start = autort.wait()
    # --------------- TRAINING SECTION -----------------
    inputs, targets = next(train_loader)

    def param_scanner(p_scanner_fn):
        for opt in optimizers:
            for group in opt.param_groups:
                for param in group["params"]:
                    p_scanner_fn(param)
 
    def CHECK_STATES(name, p_scanner_fn, weak_match=False):
        print0(f'[DEBUG({name})] Validating all parameters are in proper states..')
        with torch.no_grad():
          for opt in optimizers:
            for group in opt.param_groups:
                for param in group["params"]:
                    p = p_scanner_fn(param)
                    if p is None:
                        continue
                    v_validate = net.all_gather(p.sum().flatten(), 0)
                    all_matched = (v_validate == v_validate[0]).all()
                    if (all_matched and not weak_match and hasattr(param, 'expert')) or (not all_matched and not hasattr(param, 'expert')):
                         print0(v_validate)
                         print0(f"[DEBUG({name})] Validation failed on the param `{param.param_name}` with Shape({param.shape}). ❌")
                         exit(1)

        print0(f'[DEBUG({name})] Validation passed. ✅')

    if step >= 0:
        # Disable CHECK_STATES for the following steps
        CHECK_STATES = lambda *args, **kwargs: None

    CHECK_STATES('weights-init', lambda p: p)

    step_tokens = inputs.numel()
    blockSize = 256
    blockSize = blockSize if blockSize > 0 else step_tokens

    model.zero_grad(set_to_none=True)
    loss = model(inputs, targets)
    loss = loss.view(-1, blockSize).sum(-1).sum() / step_tokens

    (loss + args.balancing_importance * getattr(loss, 'l_aux', 0)).backward()
    param_scanner(lambda p: net.simple_all_reduce(p.grad, op=torch.distributed.ReduceOp.AVG, inplace=True) if getattr(p, 'grad', None) is not None and not hasattr(param, 'expert') else None)

    CHECK_STATES('gradients', lambda p: p.grad if hasattr(p, 'grad') else None, weak_match=True)

    for opt in optimizers:
        opt.step()

    CHECK_STATES('weights-upd', lambda p: p)
    train_stop = autort.wait()

    # logging
    approx_training_time_ms = (train_stop - train_start) * 1000
    print0(f"step:{step+1}/{train_steps} loss:{loss.item()} train_time:{approx_training_time_ms:.2f}ms")

print0(f"peak memory allocated per-device: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
       f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB")
