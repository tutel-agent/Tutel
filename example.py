#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import math
import os
import statistics
import sys
import time

import torch


FP4_E2M1 = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Correctness and CPU performance example for NVFP4 GEMV/fused MoE."
    )
    parser.add_argument("--experts", type=int)
    parser.add_argument(
        "--output-dim",
        type=int,
        metavar="N",
        help="Output N; in --fused-swiglu this is total W13 rows (gate+up).",
    )
    parser.add_argument("--input-dim", type=int, metavar="K")
    parser.add_argument("--batch", type=int, default=9)
    parser.add_argument("--tokens", type=int, default=1, metavar="M")
    parser.add_argument("--topk", type=int, metavar="T")
    parser.add_argument("--intermediate-dim", type=int, metavar="I")
    parser.add_argument("--threads", type=int, default=torch.get_num_threads())
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output-scale", type=float, default=1.0)
    parser.add_argument("--w13-output-scale", type=float, default=1.0)
    parser.add_argument("--w2-output-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--expert-ids",
        type=int,
        nargs="+",
        help="Exactly batch IDs, or M*T IDs in --end-to-end mode.",
    )
    parser.add_argument("--check-rows", type=int, default=3)
    parser.add_argument("--check-cols", type=int, default=4)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--fused-swiglu",
        action="store_true",
        help="Treat output-dim as total W13 rows [gate, up]; returns half as many hidden values.",
    )
    mode.add_argument(
        "--w2",
        action="store_true",
        help="Run W2 with per-expert activations; defaults to N=4096 and K=2048.",
    )
    mode.add_argument(
        "--w2-reduce",
        action="store_true",
        help="Run W2 and FP32 routing-weight reduction; defaults to N=4096 and K=2048.",
    )
    mode.add_argument(
        "--end-to-end",
        action="store_true",
        help="Run single-call W13+SwiGLU+W2+top-k reduction (default K=4096, I=2048, N=4096).",
    )
    mode.add_argument(
        "--end-to-end-mxfp4",
        action="store_true",
        help="Run Kimi K3 MXFP4 W13+SiTU-GLU+W2 (default E=896, K=3584, I=3072, N=3584, T=16).",
    )
    parser.add_argument(
        "--diagnose-parallel",
        action="store_true",
        help="Compare short 1-thread and requested-thread timing probes.",
    )
    parser.add_argument(
        "--diagnose-stages",
        action="store_true",
        help="For M=1 end-to-end, time equivalent standalone W13 and W2 stages.",
    )
    parser.add_argument(
        "--diagnose-internal",
        action="store_true",
        help="For --end-to-end, report actual fused CPU phase timings and selected kernels.",
    )
    parser.add_argument(
        "--compare-row-tiles",
        action="store_true",
        help="For --end-to-end, alternate row tiles 1/4 on the same tensors in one process.",
    )
    parser.add_argument(
        "--compare-w2-schedules",
        action="store_true",
        help="For --end-to-end, alternate output/routes W2 scheduling on the same tensors.",
    )
    parser.add_argument(
        "--small",
        action="store_true",
        help="Use mode-specific small dimensions for a quick correctness run.",
    )
    return parser.parse_args()


def validate_args(args):
    if args.diagnose_internal and not args.end_to_end:
        raise ValueError("--diagnose-internal requires --end-to-end")
    if args.compare_row_tiles and not args.end_to_end:
        raise ValueError("--compare-row-tiles requires --end-to-end")
    if args.compare_w2_schedules and not args.end_to_end:
        raise ValueError("--compare-w2-schedules requires --end-to-end")
    if args.experts is None:
        args.experts = 896 if args.end_to_end_mxfp4 else 256
    if args.topk is None:
        args.topk = 16 if args.end_to_end_mxfp4 else 9
    if args.intermediate_dim is None:
        args.intermediate_dim = 3072 if args.end_to_end_mxfp4 else 2048
    if args.output_dim is None:
        if args.end_to_end_mxfp4:
            args.output_dim = 3584
        else:
            args.output_dim = 4096
    if args.input_dim is None:
        if args.end_to_end_mxfp4:
            args.input_dim = 3584
        else:
            args.input_dim = 2048 if args.w2 or args.w2_reduce else 4096
    if args.small:
        if args.end_to_end_mxfp4:
            (
                args.experts,
                args.output_dim,
                args.input_dim,
                args.intermediate_dim,
                args.tokens,
                args.topk,
            ) = (4, 64, 64, 64, 2, 3)
        elif args.end_to_end:
            (
                args.experts,
                args.output_dim,
                args.input_dim,
                args.intermediate_dim,
                args.tokens,
                args.topk,
            ) = (4, 64, 64, 32, 2, 3)
        elif args.w2 or args.w2_reduce:
            args.experts, args.output_dim, args.input_dim, args.batch = 4, 64, 32, 5
        else:
            args.experts, args.output_dim, args.input_dim, args.batch = 4, 64, 80, 5
    positive = ["experts", "output_dim", "input_dim", "batch", "threads"]
    if args.end_to_end or args.end_to_end_mxfp4:
        positive.extend(("intermediate_dim", "tokens", "topk"))
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError("{} must be positive".format(name.replace("_", "-")))
    if args.input_dim % 16:
        raise ValueError("input-dim/K must be divisible by 16")
    if args.end_to_end_mxfp4 and args.input_dim % 32:
        raise ValueError("MXFP4 input-dim/K must be divisible by 32")
    if args.fused_swiglu and args.output_dim % 2:
        raise ValueError("output-dim must be even in --fused-swiglu mode")
    if args.end_to_end and args.intermediate_dim % 16:
        raise ValueError("intermediate-dim/I must be divisible by 16")
    if args.end_to_end_mxfp4 and args.intermediate_dim % 32:
        raise ValueError("MXFP4 intermediate-dim/I must be divisible by 32")
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be nonnegative and iterations must be positive")
    if args.check_rows <= 0 or args.check_cols <= 0:
        raise ValueError("check-rows and check-cols must be positive")
    if args.end_to_end_mxfp4 and (
        args.output_scale != 1.0
        or args.w13_output_scale != 1.0
        or args.w2_output_scale != 1.0
    ):
        raise ValueError("Kimi K3 MXFP4 does not use checkpoint output scales")
    scale_names = (
        ("w13-output-scale", args.w13_output_scale),
        ("w2-output-scale", args.w2_output_scale),
    ) if args.end_to_end else () if args.end_to_end_mxfp4 else (
        ("output-scale", args.output_scale),
    )
    for name, value in scale_names:
        if not math.isfinite(value):
            raise ValueError("{} must be finite".format(name))
        if abs(value) > torch.finfo(torch.float32).max:
            raise ValueError("{} must be representable as a finite float".format(name))
    if args.expert_ids is not None:
        expected_ids = (
            args.tokens * args.topk
            if args.end_to_end or args.end_to_end_mxfp4
            else args.batch
        )
        if len(args.expert_ids) != expected_ids:
            raise ValueError(
                "expert-ids must contain exactly {} values".format(expected_ids)
            )
        invalid = [value for value in args.expert_ids if value < 0 or value >= args.experts]
        if invalid:
            raise ValueError(
                "expert IDs must be in [0, {}), got {}".format(args.experts, invalid)
            )


def load_extension():
    try:
        import tutel_custom_kernel  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Unable to import tutel_custom_kernel. Build it first with "
            "`NO_CUDA=1 python setup.py build_ext --inplace --force "
            "--enable_cpu_moe`."
        ) from exc
    if not hasattr(torch.ops.tutel_ops, "nvfp4_batched_gemv"):
        raise RuntimeError(
            "CPU MoE operators are disabled. Rebuild with "
            "`NO_CUDA=1 python setup.py build_ext --inplace --force "
            "--enable_cpu_moe`."
        )


def format_bytes(size):
    return "{:.3f} GiB ({:.3f} GB)".format(size / 2**30, size / 1e9)


def standalone_result_dim(args):
    return args.output_dim // 2 if args.fused_swiglu else args.output_dim


def make_inputs(args):
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    weight_rows = args.output_dim
    activation_rows = args.batch if args.w2 or args.w2_reduce else 1
    A = torch.randn(
        (activation_rows, args.input_dim),
        dtype=torch.bfloat16,
        generator=generator,
    ).contiguous()
    W = torch.randint(
        0,
        256,
        (args.experts, weight_rows, args.input_dim // 2),
        dtype=torch.uint8,
        generator=generator,
    ).contiguous()
    # Codes [0, 126] cover finite positive E4M3FN values and exclude NaN code 0x7f.
    W_scale = torch.randint(
        0,
        127,
        (args.experts, weight_rows, args.input_dim // 16),
        dtype=torch.uint8,
        generator=generator,
    ).contiguous()
    if args.expert_ids is None:
        expert_ids = torch.randint(
            0, args.experts, (args.batch,), dtype=torch.int32, generator=generator
        )
        if args.batch > 1:
            expert_ids[-1] = expert_ids[0]
    else:
        expert_ids = torch.tensor(args.expert_ids, dtype=torch.int32)
    expert_weights = None
    if args.w2_reduce:
        expert_weights = torch.softmax(
            torch.randn(args.batch, dtype=torch.float32, generator=generator), dim=0
        ).contiguous()
    return A, W, W_scale, expert_ids.contiguous(), expert_weights


def make_end_to_end_inputs(args):
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    x = torch.randn(
        (args.tokens, args.input_dim),
        dtype=torch.bfloat16,
        generator=generator,
    ).contiguous()
    w13 = torch.randint(
        0,
        256,
        (args.experts, 2 * args.intermediate_dim, args.input_dim // 2),
        dtype=torch.uint8,
        generator=generator,
    ).contiguous()
    w13_scale = torch.randint(
        0,
        127,
        (args.experts, 2 * args.intermediate_dim, args.input_dim // 16),
        dtype=torch.uint8,
        generator=generator,
    ).contiguous()
    w2 = torch.randint(
        0,
        256,
        (args.experts, args.output_dim, args.intermediate_dim // 2),
        dtype=torch.uint8,
        generator=generator,
    ).contiguous()
    w2_scale = torch.randint(
        0,
        127,
        (args.experts, args.output_dim, args.intermediate_dim // 16),
        dtype=torch.uint8,
        generator=generator,
    ).contiguous()
    if args.expert_ids is None:
        topk_ids = torch.randint(
            0,
            args.experts,
            (args.tokens, args.topk),
            dtype=torch.int64,
            generator=generator,
        )
        if topk_ids.numel() > 1:
            topk_ids[-1, -1] = topk_ids[0, 0]
    else:
        topk_ids = torch.tensor(args.expert_ids, dtype=torch.int64).view(
            args.tokens, args.topk
        )
    topk_weights = torch.softmax(
        torch.randn(
            (args.tokens, args.topk), dtype=torch.float32, generator=generator
        ),
        dim=1,
    ).contiguous()
    return (
        x,
        w13,
        w13_scale,
        w2,
        w2_scale,
        topk_ids.contiguous(),
        topk_weights,
    )


def make_mxfp4_inputs(args):
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    x = torch.randn(
        (args.tokens, args.input_dim),
        dtype=torch.bfloat16,
        generator=generator,
    ).contiguous()
    w13 = torch.randint(
        0,
        256,
        (args.experts, 2 * args.intermediate_dim, args.input_dim // 2),
        dtype=torch.uint8,
        generator=generator,
    ).contiguous()
    w13_scale = torch.randint(
        120,
        132,
        (args.experts, 2 * args.intermediate_dim, args.input_dim // 32),
        dtype=torch.uint8,
        generator=generator,
    ).contiguous()
    w2 = torch.randint(
        0,
        256,
        (args.experts, args.output_dim, args.intermediate_dim // 2),
        dtype=torch.uint8,
        generator=generator,
    ).contiguous()
    w2_scale = torch.randint(
        120,
        132,
        (args.experts, args.output_dim, args.intermediate_dim // 32),
        dtype=torch.uint8,
        generator=generator,
    ).contiguous()
    if args.expert_ids is None:
        topk_ids = torch.randint(
            0,
            args.experts,
            (args.tokens, args.topk),
            dtype=torch.int64,
            generator=generator,
        )
        if topk_ids.numel() > 1:
            topk_ids[-1, -1] = topk_ids[0, 0]
    else:
        topk_ids = torch.tensor(args.expert_ids, dtype=torch.int64).view(
            args.tokens, args.topk
        )
    topk_weights = torch.softmax(
        torch.randn(
            (args.tokens, args.topk),
            dtype=torch.float32,
            generator=generator,
        ),
        dim=1,
    ).contiguous()
    return (
        x,
        w13,
        w13_scale,
        w2,
        w2_scale,
        topk_ids.contiguous(),
        topk_weights,
    )


def decode_e4m3fn(raw):
    bits = raw.to(torch.int16)
    exponent = (bits >> 3) & 0x0F
    mantissa = bits & 0x07
    mantissa_f = mantissa.float()
    subnormal = mantissa_f / 512.0
    normal = (1.0 + mantissa_f / 8.0) * torch.pow(2.0, exponent.float() - 7.0)
    values = torch.where(exponent == 0, subnormal, normal)
    values = torch.where(
        (exponent == 0x0F) & (mantissa == 0x07),
        torch.full_like(values, float("nan")),
        values,
    )
    return torch.where((bits & 0x80) != 0, -values, values)


def decode_e8m0(raw):
    return torch.ldexp(
        torch.ones_like(raw, dtype=torch.float32),
        raw.to(torch.int32) - 127,
    )


def sample_indices(size, count):
    count = min(size, count)
    if count == 1:
        return [0]
    return sorted(
        {round(index * (size - 1) / (count - 1)) for index in range(count)}
    )


def sampled_projection(A, W, W_scale, activation_row, expert, column):
    activation = A[activation_row].float()
    K = A.size(1)
    packed = W[expert, column]
    codes = torch.empty(K, dtype=torch.long)
    codes[0::2] = (packed & 0x0F).long()
    codes[1::2] = (packed >> 4).long()
    scales = decode_e4m3fn(W_scale[expert, column]).repeat_interleave(16)
    return (activation * (FP4_E2M1[codes] * scales)).sum()


def projection_rows_reference(activation, packed_rows, scale_rows, chunk_rows=64):
    K = activation.numel()
    outputs = []
    activation_f = activation.float()
    for begin in range(0, packed_rows.size(0), chunk_rows):
        packed = packed_rows[begin : begin + chunk_rows]
        scales_raw = scale_rows[begin : begin + chunk_rows]
        codes = torch.empty((packed.size(0), K), dtype=torch.long)
        codes[:, 0::2] = (packed & 0x0F).long()
        codes[:, 1::2] = (packed >> 4).long()
        scales = decode_e4m3fn(scales_raw).repeat_interleave(16, dim=-1)
        outputs.append(
            (activation_f.unsqueeze(0) * (FP4_E2M1[codes] * scales)).sum(dim=1)
        )
    return torch.cat(outputs)


def sampled_end_to_end_reference(args, tensors, rows, columns):
    x, w13, w13_scale, w2, w2_scale, topk_ids, topk_weights = tensors
    hidden_cache = {}
    expected = []
    positions = []
    for token in rows:
        reduced = torch.zeros(len(columns), dtype=torch.float32)
        for topk in range(args.topk):
            expert = int(topk_ids[token, topk])
            cache_key = (token, expert)
            if cache_key not in hidden_cache:
                projected = projection_rows_reference(
                    x[token], w13[expert], w13_scale[expert]
                )
                gate, up = projected.chunk(2)
                hidden_cache[cache_key] = (
                    torch.nn.functional.silu(gate * args.w13_output_scale)
                    * (up * args.w13_output_scale)
                ).bfloat16()
            hidden = hidden_cache[cache_key].view(1, -1)
            for index, column in enumerate(columns):
                projected = sampled_projection(
                    hidden, w2, w2_scale, 0, expert, column
                )
                reduced[index] += (
                    topk_weights[token, topk]
                    * (projected * args.w2_output_scale)
                )
        for index, column in enumerate(columns):
            expected.append(reduced[index].bfloat16().float())
            positions.append((token, column))
    return positions, torch.stack(expected)


def check_end_to_end_correctness(args, tensors, output):
    rows = sample_indices(args.tokens, args.check_rows)
    columns = sample_indices(args.output_dim, args.check_cols)
    positions, expected = sampled_end_to_end_reference(args, tensors, rows, columns)
    actual = torch.stack([output[row, column].float() for row, column in positions])
    absolute = (actual - expected).abs()
    relative = absolute / expected.abs().clamp_min(1e-12)
    max_abs = float(absolute.max())
    max_rel = float(relative.max())
    passed = bool(torch.all(absolute <= 0.05 + 0.04 * expected.abs()))
    print(
        "Correctness: samples={}, max_abs={:.6g}, max_rel={:.6g}, {}".format(
            len(positions), max_abs, max_rel, "PASS" if passed else "FAIL"
        )
    )
    return passed


def mxfp4_sampled_projection(A, W, W_scale, activation_row, expert, column):
    activation = A[activation_row].float()
    K = A.size(1)
    packed = W[expert, column]
    codes = torch.empty(K, dtype=torch.long)
    codes[0::2] = (packed & 0x0F).long()
    codes[1::2] = (packed >> 4).long()
    scales = decode_e8m0(W_scale[expert, column]).repeat_interleave(32)
    return (activation * (FP4_E2M1[codes] * scales)).sum()


def mxfp4_projection_rows_reference(
    activation, packed_rows, scale_rows, chunk_rows=64
):
    K = activation.numel()
    outputs = []
    activation_f = activation.float()
    for begin in range(0, packed_rows.size(0), chunk_rows):
        packed = packed_rows[begin : begin + chunk_rows]
        scales_raw = scale_rows[begin : begin + chunk_rows]
        codes = torch.empty((packed.size(0), K), dtype=torch.long)
        codes[:, 0::2] = (packed & 0x0F).long()
        codes[:, 1::2] = (packed >> 4).long()
        scales = decode_e8m0(scales_raw).repeat_interleave(32, dim=-1)
        outputs.append(
            (activation_f.unsqueeze(0) * (FP4_E2M1[codes] * scales)).sum(dim=1)
        )
    return torch.cat(outputs)


def sampled_mxfp4_reference(args, tensors, rows, columns):
    x, w13, w13_scale, w2, w2_scale, topk_ids, topk_weights = tensors
    hidden_cache = {}
    expected = []
    positions = []
    for token in rows:
        reduced = torch.zeros(len(columns), dtype=torch.float32)
        for topk in range(args.topk):
            expert = int(topk_ids[token, topk])
            cache_key = (token, expert)
            if cache_key not in hidden_cache:
                projected = mxfp4_projection_rows_reference(
                    x[token], w13[expert], w13_scale[expert]
                )
                gate, up = projected.chunk(2)
                gate_term = 4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)
                up_term = 25.0 * torch.tanh(up / 25.0)
                hidden_cache[cache_key] = (gate_term * up_term).bfloat16()
            hidden = hidden_cache[cache_key].view(1, -1)
            for index, column in enumerate(columns):
                projected = mxfp4_sampled_projection(
                    hidden, w2, w2_scale, 0, expert, column
                )
                reduced[index] += topk_weights[token, topk] * projected
        for index, column in enumerate(columns):
            expected.append(reduced[index].bfloat16().float())
            positions.append((token, column))
    return positions, torch.stack(expected)


def check_mxfp4_correctness(args, tensors, output):
    rows = sample_indices(args.tokens, args.check_rows)
    columns = sample_indices(args.output_dim, args.check_cols)
    positions, expected = sampled_mxfp4_reference(args, tensors, rows, columns)
    actual = torch.stack([output[row, column].float() for row, column in positions])
    absolute = (actual - expected).abs()
    relative = absolute / expected.abs().clamp_min(1e-12)
    max_abs = float(absolute.max())
    max_rel = float(relative.max())
    passed = bool(torch.all(absolute <= 0.05 + 0.04 * expected.abs()))
    print(
        "Correctness: samples={}, max_abs={:.6g}, max_rel={:.6g}, {}".format(
            len(positions), max_abs, max_rel, "PASS" if passed else "FAIL"
        )
    )
    return passed


def sampled_reference(args, A, W, W_scale, expert_ids, rows, columns):
    expected = []
    positions = []
    if args.w2_reduce:
        for column in columns:
            value = torch.tensor(0.0, dtype=torch.float32)
            for batch_row in range(args.batch):
                expert = int(expert_ids[batch_row])
                projected = sampled_projection(
                    A, W, W_scale, batch_row, expert, column
                )
                projected = projected * args.output_scale
                value += projected * args.expert_weights[batch_row]
            expected.append(value.bfloat16().float())
            positions.append((0, column))
        return positions, torch.stack(expected)

    for batch_row in rows:
        expert = int(expert_ids[batch_row])
        activation_row = batch_row if args.w2 else 0
        for column in columns:
            gate = sampled_projection(
                A, W, W_scale, activation_row, expert, column
            )
            gate = gate * args.output_scale
            if args.fused_swiglu:
                up = sampled_projection(
                    A,
                    W,
                    W_scale,
                    activation_row,
                    expert,
                    column + standalone_result_dim(args),
                )
                value = torch.nn.functional.silu(gate) * (up * args.output_scale)
            else:
                value = gate
            expected.append(value.bfloat16().float())
            positions.append((batch_row, column))
    return positions, torch.stack(expected)


def check_correctness(args, A, W, W_scale, expert_ids, output):
    rows = [0] if args.w2_reduce else sample_indices(args.batch, args.check_rows)
    columns = sample_indices(standalone_result_dim(args), args.check_cols)
    positions, expected = sampled_reference(
        args, A, W, W_scale, expert_ids, rows, columns
    )
    actual = torch.stack([output[row, column].float() for row, column in positions])
    absolute = (actual - expected).abs()
    relative = absolute / expected.abs().clamp_min(1e-12)
    max_abs = float(absolute.max())
    max_rel = float(relative.max())
    passed = bool(torch.all(absolute <= 0.02 + 0.02 * expected.abs()))
    print(
        "Correctness: samples={}, max_abs={:.6g}, max_rel={:.6g}, {}".format(
            len(positions), max_abs, max_rel, "PASS" if passed else "FAIL"
        )
    )
    return passed


def operator_latencies(op, inputs, scalar_args, iterations):
    latencies = []
    output = None
    for _ in range(iterations):
        start = time.perf_counter()
        output = op(*inputs, *scalar_args)
        latencies.append(time.perf_counter() - start)
    return output, latencies


def diagnose_parallelism(args, op, inputs, scalar_args):
    if args.threads == 1:
        print("Parallel diagnostic: skipped because --threads=1")
        return
    requested_threads = args.threads
    medians = []
    entry_threads = torch.get_num_threads()
    try:
        for thread_count in (1, requested_threads):
            torch.set_num_threads(thread_count)
            op(*inputs, *scalar_args)
            _, latencies = operator_latencies(op, inputs, scalar_args, 3)
            medians.append(statistics.median(latencies))
    finally:
        torch.set_num_threads(entry_threads)
    print(
        "Parallel diagnostic: 1 thread={:.3f} ms, {} threads={:.3f} ms, "
        "speedup={:.2f}x".format(
            medians[0] * 1e3,
            requested_threads,
            medians[1] * 1e3,
            medians[0] / medians[1],
        )
    )


def print_parallel_info():
    print("torch.get_num_threads(): {}".format(torch.get_num_threads()))
    details = torch.__config__.parallel_info().splitlines()
    prefixes = ("at::get_num_threads()", "omp_get_max_threads()", "ATen parallel backend:")
    for line in details:
        stripped = line.strip()
        if stripped.startswith(prefixes):
            print("  {}".format(stripped))


def diagnose_end_to_end_stages(args, tensors, end_to_end_median):
    if args.tokens != 1:
        print("Stage diagnostic: skipped because it currently requires M=1")
        return
    x, w13, w13_scale, w2, w2_scale, topk_ids, topk_weights = tensors
    expert_ids = topk_ids.flatten().to(torch.int32).contiguous()
    stage_iterations = min(10, max(3, args.iterations))

    w13_op = torch.ops.tutel_ops.nvfp4_batched_gemv_swiglu
    w13_inputs = (x, w13, w13_scale, expert_ids)
    hidden = w13_op(*w13_inputs, args.w13_output_scale)
    _, w13_latencies = operator_latencies(
        w13_op, w13_inputs, (args.w13_output_scale,), stage_iterations
    )

    w2_op = torch.ops.tutel_ops.nvfp4_batched_gemv_w2_reduce
    w2_inputs = (
        hidden,
        w2,
        w2_scale,
        expert_ids,
        topk_weights.flatten().contiguous(),
    )
    w2_op(*w2_inputs, args.w2_output_scale)
    _, w2_latencies = operator_latencies(
        w2_op, w2_inputs, (args.w2_output_scale,), stage_iterations
    )
    w13_median = statistics.median(w13_latencies)
    w2_median = statistics.median(w2_latencies)
    print(
        "Standalone stage diagnostic: W13+SwiGLU={:.3f} ms, "
        "W2+reduction={:.3f} ms, sum={:.3f} ms".format(
            w13_median * 1e3, w2_median * 1e3, (w13_median + w2_median) * 1e3
        )
    )
    stage_sum = w13_median + w2_median
    overhead = end_to_end_median - stage_sum
    overhead_percent = 100.0 * overhead / stage_sum
    print(
        "E2E difference vs standalone sum (not internal overhead): "
        "{:+.3f} us ({:+.2f}%)".format(
            overhead * 1e6, overhead_percent
        )
    )


def diagnose_nvfp4_internal(args, tensors):
    op = torch.ops.tutel_ops.fused_nvfp4_moe_swiglu_cpu_profile
    scalar_args = (args.w13_output_scale, args.w2_output_scale)
    for _ in range(args.warmup):
        op(*tensors, *scalar_args)
    samples = []
    backends = set()
    for _ in range(args.iterations):
        _, phase_seconds, w13_backend, w2_backend = op(*tensors, *scalar_args)
        samples.append(phase_seconds)
        backends.add((w13_backend, w2_backend))

    print("Actual kernel dispatch: {}".format(
        "; ".join(
            "W13={}, W2={}".format(w13, w2) for w13, w2 in sorted(backends)
        )
    ))
    phase_names = (
        "setup", "W13+SwiGLU", "hidden_prepare",
        "W2_setup", "W2+reduction", "cleanup",
    )
    medians = [
        statistics.median(sample[index] for sample in samples)
        for index in range(len(phase_names))
    ]
    print("Internal fused profile (median us; instrumented): {}".format(
        ", ".join(
            "{}={:.3f}".format(name, value * 1e6)
            for name, value in zip(phase_names, medians)
        )
    ))
    totals = [sum(sample) for sample in samples]
    print(
        "Internal native total: median={:.3f} us, mean={:.3f} us "
        "(excludes dispatcher, result packaging and Python)".format(
            statistics.median(totals) * 1e6, statistics.mean(totals) * 1e6
        )
    )


def compare_nvfp4_row_tiles(args, tensors):
    compare_nvfp4_modes(args, tensors, row_tiles=True)


def compare_nvfp4_w2_schedules(args, tensors):
    compare_nvfp4_modes(args, tensors, row_tiles=False)


def compare_nvfp4_modes(args, tensors, *, row_tiles):
    op = torch.ops.tutel_ops.fused_nvfp4_moe_swiglu_cpu
    profile_op = torch.ops.tutel_ops.fused_nvfp4_moe_swiglu_cpu_profile
    scalar_args = (args.w13_output_scale, args.w2_output_scale)
    if row_tiles:
        env_name, modes = "TUTEL_NVFP4_ROW_TILE", (1, 4)
        title, mode_label, names = "Row-tile A/B", "tile", ("tile1", "tile4")
    else:
        env_name, modes = "TUTEL_NVFP4_W2_SCHEDULE", ("output", "routes")
        title, mode_label, names = "W2-schedule A/B", "schedule", modes
    original_mode = os.environ.get(env_name)
    outputs = {}
    backends = {}
    latencies = {mode: [] for mode in modes}
    try:
        for mode in modes:
            os.environ[env_name] = str(mode)
            output, _, w13_backend, w2_backend = profile_op(*tensors, *scalar_args)
            outputs[mode] = output
            backends[mode] = (w13_backend, w2_backend)
            if row_tiles:
                honored = all(
                    ("rows4" in backend.split("/")) == (mode == 4)
                    for backend in backends[mode]
                    if backend.split("/")[0] == "AVX512-BF16"
                )
            else:
                honored = ("routes" in w2_backend.split("/")) == (mode == "routes")
            if not honored:
                raise RuntimeError(
                    "The extension did not honor {}={}. "
                    "Rebuild the CPU extension with --enable_cpu_moe.".format(env_name, mode)
                )
        if row_tiles and not any(
            "rows4" in backend.split("/") for backend in backends[4]
        ):
            print(
                "Row-tile A/B skipped: no four-row AVX512-BF16 path selected "
                "(W13={}, W2={}).".format(*backends[4])
            )
            return
        torch.testing.assert_close(
            outputs[modes[1]], outputs[modes[0]], rtol=0, atol=0, equal_nan=True
        )
        for iteration in range(args.warmup):
            for mode in (modes if iteration % 2 == 0 else modes[::-1]):
                os.environ[env_name] = str(mode)
                op(*tensors, *scalar_args)
        for iteration in range(args.iterations):
            for mode in (modes if iteration % 2 == 0 else modes[::-1]):
                os.environ[env_name] = str(mode)
                start = time.perf_counter()
                output = op(*tensors, *scalar_args)
                elapsed = time.perf_counter() - start
                latencies[mode].append(elapsed)
                outputs[mode] = output
        torch.testing.assert_close(
            outputs[modes[1]], outputs[modes[0]], rtol=0, atol=0, equal_nan=True
        )
    finally:
        if original_mode is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = original_mode

    bytes_read = tensors[5].numel() * sum(
        tensor.numel() * tensor.element_size() // tensor.size(0)
        for tensor in tensors[1:5]
    )
    print(
        "{}: same process/tensors, alternating {}->{} / {}->{}, "
        "{} samples per mode, uninstrumented calls".format(
            title, *modes, *modes[::-1], args.iterations
        )
    )
    for mode in modes:
        median = statistics.median(latencies[mode])
        print(
            "  {}={}: median={:.3f} us, mean={:.3f} us, "
            "effective Weight+scale={:.3f} GB/s; W13={}, W2={}".format(
                mode_label, mode, median * 1e6, statistics.mean(latencies[mode]) * 1e6,
                bytes_read / median / 1e9, *backends[mode],
            )
        )
    ratios = [old / new for old, new in zip(latencies[modes[0]], latencies[modes[1]])]
    wins = sum(new < old for old, new in zip(latencies[modes[0]], latencies[modes[1]]))
    print(
        "  Paired median speedup ({}/{})={:.3f}x; "
        "{} faster in {}/{} pairs; outputs match.".format(
            *names, statistics.median(ratios), names[1], wins, args.iterations
        )
    )


def run_mxfp4_end_to_end(args):
    routes = args.tokens * args.topk
    w13_weight_bytes = (
        args.experts * 2 * args.intermediate_dim * (args.input_dim // 2)
    )
    w13_scale_bytes = (
        args.experts * 2 * args.intermediate_dim * (args.input_dim // 32)
    )
    w2_weight_bytes = (
        args.experts * args.output_dim * (args.intermediate_dim // 2)
    )
    w2_scale_bytes = (
        args.experts * args.output_dim * (args.intermediate_dim // 32)
    )
    input_bytes = (
        w13_weight_bytes
        + w13_scale_bytes
        + w2_weight_bytes
        + w2_scale_bytes
        + 2 * args.tokens * args.input_dim
        + 12 * routes
    )
    print(
        "Mode: end-to-end Kimi K3 MXFP4 SiTU-GLU MoE; shape: "
        "E={}, M={}, T={}, K={}, W13 total rows={}, hidden/W2 K I={}, "
        "N={}, threads={}".format(
            args.experts,
            args.tokens,
            args.topk,
            args.input_dim,
            2 * args.intermediate_dim,
            args.intermediate_dim,
            args.output_dim,
            args.threads,
        )
    )
    print(
        "Estimated allocated input memory: {} "
        "(W13={}, W13 scales={}, W2={}, W2 scales={})".format(
            format_bytes(input_bytes),
            format_bytes(w13_weight_bytes),
            format_bytes(w13_scale_bytes),
            format_bytes(w2_weight_bytes),
            format_bytes(w2_scale_bytes),
        )
    )
    try:
        tensors = make_mxfp4_inputs(args)
    except (MemoryError, RuntimeError) as exc:
        print(
            "error: input allocation failed ({}). Use --small or reduce dimensions.".format(
                exc
            ),
            file=sys.stderr,
        )
        return 2

    torch.set_num_threads(args.threads)
    print_parallel_info()
    print(
        "Kernel dispatch: compile/runtime-selected AVX512-BF16 when supported; "
        "otherwise AVX2/scalar."
    )
    print(
        "MXFP4: packed E2M1, group32 E8M0 scale=2^(byte-127), no global scale."
    )
    print(
        "SiTU-GLU: hidden=BF16((4*tanh(gate/4)*sigmoid(gate))"
        "*(25*tanh(up/25)))"
    )
    print("Top-k expert IDs: {}".format(tensors[5].tolist()))
    print("Top-k routing weights: {}".format(tensors[6].tolist()))
    op = torch.ops.tutel_ops.fused_mxfp4_moe_situ_cpu
    output = op(*tensors)
    if not check_mxfp4_correctness(args, tensors, output):
        return 1
    if args.diagnose_parallel:
        diagnose_parallelism(args, op, tensors, ())
    if args.diagnose_stages:
        print("Stage diagnostic: unavailable because MXFP4 exposes only the fused API")

    for _ in range(args.warmup):
        output = op(*tensors)
    output, latencies = operator_latencies(op, tensors, (), args.iterations)
    median = statistics.median(latencies)
    mean = statistics.mean(latencies)
    w13_packed_read = routes * 2 * args.intermediate_dim * (args.input_dim // 2)
    w13_scale_read = routes * 2 * args.intermediate_dim * (args.input_dim // 32)
    w2_packed_read = routes * args.output_dim * (args.intermediate_dim // 2)
    w2_scale_read = routes * args.output_dim * (args.intermediate_dim // 32)
    packed_read = w13_packed_read + w2_packed_read
    scale_read = w13_scale_read + w2_scale_read
    w13_flops = 2 * routes * 2 * args.intermediate_dim * args.input_dim
    w2_flops = 2 * routes * args.output_dim * args.intermediate_dim
    logical_flops = w13_flops + w2_flops
    print(
        "Latency (single-call MXFP4 W13+SiTU-GLU+W2+reduction): "
        "median={:.3f} ms, mean={:.3f} ms".format(median * 1e3, mean * 1e3)
    )
    print(
        "Throughput: {:.3f} routes/s, {:.3f} tokens/s, "
        "{:.3f} logical GFLOP/s".format(
            routes / median, args.tokens / median, logical_flops / median / 1e9
        )
    )
    print(
        "Logical work/call: W13={:.3f} GFLOP, W2={:.3f} GFLOP, "
        "total={:.3f} GFLOP".format(
            w13_flops / 1e9, w2_flops / 1e9, logical_flops / 1e9
        )
    )
    print(
        "Bytes/call: W13 packed={}, W13 scales={}, W2 packed={}, W2 scales={}".format(
            format_bytes(w13_packed_read),
            format_bytes(w13_scale_read),
            format_bytes(w2_packed_read),
            format_bytes(w2_scale_read),
        )
    )
    print(
        "Effective bandwidth: Weight={:.3f} GB/s, Weight+scale={:.3f} GB/s".format(
            packed_read / median / 1e9,
            (packed_read + scale_read) / median / 1e9,
        )
    )
    print("Checksum: {:.9g}".format(output.float().sum().item()))
    return 0


def run_end_to_end(args):
    routes = args.tokens * args.topk
    w13_weight_bytes = (
        args.experts * 2 * args.intermediate_dim * (args.input_dim // 2)
    )
    w13_scale_bytes = (
        args.experts * 2 * args.intermediate_dim * (args.input_dim // 16)
    )
    w2_weight_bytes = (
        args.experts * args.output_dim * (args.intermediate_dim // 2)
    )
    w2_scale_bytes = (
        args.experts * args.output_dim * (args.intermediate_dim // 16)
    )
    input_bytes = (
        w13_weight_bytes
        + w13_scale_bytes
        + w2_weight_bytes
        + w2_scale_bytes
        + 2 * args.tokens * args.input_dim
        + 12 * routes
    )
    print(
        "Mode: end-to-end fused NVFP4 MoE; shape: E={}, M={}, T={}, "
        "K={}, W13 total rows={}, hidden/W2 K I={}, N={}, threads={}".format(
            args.experts,
            args.tokens,
            args.topk,
            args.input_dim,
            2 * args.intermediate_dim,
            args.intermediate_dim,
            args.output_dim,
            args.threads,
        )
    )
    print(
        "Estimated allocated input memory: {} "
        "(W13={}, W13 scales={}, W2={}, W2 scales={})".format(
            format_bytes(input_bytes),
            format_bytes(w13_weight_bytes),
            format_bytes(w13_scale_bytes),
            format_bytes(w2_weight_bytes),
            format_bytes(w2_scale_bytes),
        )
    )
    try:
        tensors = make_end_to_end_inputs(args)
    except (MemoryError, RuntimeError) as exc:
        print(
            "error: input allocation failed ({}). Use --small or reduce dimensions.".format(
                exc
            ),
            file=sys.stderr,
        )
        return 2

    torch.set_num_threads(args.threads)
    print_parallel_info()
    print(
        "Kernel dispatch: each stage selects AVX512-BF16 at compile/runtime "
        "when supported and its K is divisible by 32; otherwise AVX2/scalar."
    )
    print(
        "Fused math: hidden=BF16(SiLU(w13_scale*gate_dot)*(w13_scale*up_dot)); "
        "output=BF16(sum_t route_weight*(w2_scale*W2_dot(hidden)))"
    )
    print("Top-k expert IDs: {}".format(tensors[5].tolist()))
    print("Top-k routing weights: {}".format(tensors[6].tolist()))
    op = torch.ops.tutel_ops.fused_nvfp4_moe_swiglu_cpu
    scalar_args = (args.w13_output_scale, args.w2_output_scale)
    output = op(*tensors, *scalar_args)
    if not check_end_to_end_correctness(args, tensors, output):
        return 1
    if args.diagnose_parallel:
        diagnose_parallelism(args, op, tensors, scalar_args)

    for _ in range(args.warmup):
        output = op(*tensors, *scalar_args)
    output, latencies = operator_latencies(
        op, tensors, scalar_args, args.iterations
    )
    median = statistics.median(latencies)
    mean = statistics.mean(latencies)
    w13_packed_read = routes * 2 * args.intermediate_dim * (args.input_dim // 2)
    w13_scale_read = routes * 2 * args.intermediate_dim * (args.input_dim // 16)
    w2_packed_read = routes * args.output_dim * (args.intermediate_dim // 2)
    w2_scale_read = routes * args.output_dim * (args.intermediate_dim // 16)
    packed_read = w13_packed_read + w2_packed_read
    scale_read = w13_scale_read + w2_scale_read
    w13_flops = 2 * routes * 2 * args.intermediate_dim * args.input_dim
    w2_flops = 2 * routes * args.output_dim * args.intermediate_dim
    logical_flops = w13_flops + w2_flops
    print(
        "Latency (single-call W13+SwiGLU+W2+reduction for all tokens): "
        "median={:.3f} ms, mean={:.3f} ms".format(median * 1e3, mean * 1e3)
    )
    print(
        "Throughput: {:.3f} routes/s, {:.3f} tokens/s, "
        "{:.3f} logical GFLOP/s".format(
            routes / median, args.tokens / median, logical_flops / median / 1e9
        )
    )
    print(
        "Logical work/call: W13={:.3f} GFLOP, W2={:.3f} GFLOP, total={:.3f} GFLOP".format(
            w13_flops / 1e9, w2_flops / 1e9, logical_flops / 1e9
        )
    )
    print(
        "Bytes/call: W13 packed={}, W13 scales={}, W2 packed={}, W2 scales={}".format(
            format_bytes(w13_packed_read),
            format_bytes(w13_scale_read),
            format_bytes(w2_packed_read),
            format_bytes(w2_scale_read),
        )
    )
    print(
        "Effective bandwidth: Weight={:.3f} GB/s, Weight+scale={:.3f} GB/s".format(
            packed_read / median / 1e9, (packed_read + scale_read) / median / 1e9
        )
    )
    if args.diagnose_stages:
        diagnose_end_to_end_stages(args, tensors, median)
    if args.diagnose_internal:
        diagnose_nvfp4_internal(args, tensors)
    if args.compare_row_tiles:
        compare_nvfp4_row_tiles(args, tensors)
    if args.compare_w2_schedules:
        compare_nvfp4_w2_schedules(args, tensors)
    print("Checksum: {:.9g}".format(output.float().sum().item()))
    return 0


def main():
    args = parse_args()
    try:
        validate_args(args)
        load_extension()
        if (
            args.diagnose_internal or args.compare_row_tiles or args.compare_w2_schedules
        ) and not hasattr(
            torch.ops.tutel_ops, "fused_nvfp4_moe_swiglu_cpu_profile"
        ):
            raise RuntimeError(
                "NVFP4 internal profiling is unavailable. Rebuild with "
                "`NO_CUDA=1 python setup.py build_ext --inplace --force "
                "--enable_cpu_moe`."
            )
        if args.end_to_end_mxfp4 and not hasattr(
            torch.ops.tutel_ops, "fused_mxfp4_moe_situ_cpu"
        ):
            raise RuntimeError(
                "MXFP4 CPU MoE operator is unavailable. Rebuild with "
                "`NO_CUDA=1 python setup.py build_ext --inplace --force "
                "--enable_cpu_moe`."
            )
    except (ValueError, RuntimeError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2
    if args.end_to_end:
        return run_end_to_end(args)
    if args.end_to_end_mxfp4:
        return run_mxfp4_end_to_end(args)

    weight_rows = args.output_dim
    result_dim = standalone_result_dim(args)
    weight_bytes = args.experts * weight_rows * (args.input_dim // 2)
    scale_bytes = args.experts * weight_rows * (args.input_dim // 16)
    activation_rows = args.batch if args.w2 or args.w2_reduce else 1
    input_bytes = (
        weight_bytes
        + scale_bytes
        + 2 * activation_rows * args.input_dim
        + 4 * args.batch
        + (4 * args.batch if args.w2_reduce else 0)
    )
    mode_name = (
        "fused W13 SwiGLU"
        if args.fused_swiglu
        else "W2 expert projection"
        if args.w2
        else "W2 routing reduction"
        if args.w2_reduce
        else "W-only GEMV"
    )
    if args.fused_swiglu:
        shape = "E={}, W13 total rows={}, hidden I={}, K={}, batch={}, threads={}".format(
            args.experts,
            weight_rows,
            result_dim,
            args.input_dim,
            args.batch,
            args.threads,
        )
    else:
        shape = "E={}, output N={}, W rows={}, K={}, batch={}, threads={}".format(
            args.experts,
            result_dim,
            weight_rows,
            args.input_dim,
            args.batch,
            args.threads,
        )
    print("Mode: {}; shape: {}".format(mode_name, shape))
    print(
        "Estimated allocated input memory: {} (W={}, scales={})".format(
            format_bytes(input_bytes), format_bytes(weight_bytes), format_bytes(scale_bytes)
        )
    )

    try:
        A, W, W_scale, expert_ids, expert_weights = make_inputs(args)
    except (MemoryError, RuntimeError) as exc:
        print(
            "error: input allocation failed ({}). Use --small or reduce dimensions.".format(
                exc
            ),
            file=sys.stderr,
        )
        return 2

    torch.set_num_threads(args.threads)
    print_parallel_info()
    print(
        "Kernel dispatch: compile/runtime-selected AVX512-BF16 when supported "
        "with K divisible by 32; otherwise AVX2/scalar."
    )
    if args.fused_swiglu:
        print("Fused math: SiLU(output_scale * gate_dot) * (output_scale * up_dot)")
        op = torch.ops.tutel_ops.nvfp4_batched_gemv_swiglu
    elif args.w2:
        op = torch.ops.tutel_ops.nvfp4_batched_gemv_w2
    elif args.w2_reduce:
        print(
            "Reduction math: BF16(sum_b routing_weight[b] * "
            "(output_scale * W2_dot[b]))"
        )
        op = torch.ops.tutel_ops.nvfp4_batched_gemv_w2_reduce
    else:
        op = torch.ops.tutel_ops.nvfp4_batched_gemv
    print("Expert IDs: {}".format(expert_ids.tolist()))
    if args.w2_reduce:
        args.expert_weights = expert_weights
        print("Routing weights: {}".format(expert_weights.tolist()))
        inputs = (A, W, W_scale, expert_ids, expert_weights)
    else:
        inputs = (A, W, W_scale, expert_ids)
    output = op(*inputs, args.output_scale)
    if not check_correctness(args, A, W, W_scale, expert_ids, output):
        return 1
    if args.diagnose_parallel:
        diagnose_parallelism(args, op, inputs, (args.output_scale,))

    for _ in range(args.warmup):
        output = op(*inputs, args.output_scale)
    output, latencies = operator_latencies(
        op, inputs, (args.output_scale,), args.iterations
    )

    median = statistics.median(latencies)
    mean = statistics.mean(latencies)
    packed_read = args.batch * weight_rows * (args.input_dim // 2)
    scale_read = args.batch * weight_rows * (args.input_dim // 16)
    logical_flops = 2 * args.batch * weight_rows * args.input_dim
    print(
        "Latency (all batch expert {}): median={:.3f} ms, mean={:.3f} ms".format(
            "W13+SwiGLU"
            if args.fused_swiglu
            else "W2 GEMVs"
            if args.w2
            else "W2+reduction"
            if args.w2_reduce
            else "GEMVs",
            median * 1e3,
            mean * 1e3,
        )
    )
    if args.w2_reduce:
        print(
            "Throughput: {:.3f} expert GEMVs/s, {:.3f} reduced outputs/s, "
            "{:.3f} logical GFLOP/s".format(
                args.batch / median, 1.0 / median, logical_flops / median / 1e9
            )
        )
    else:
        print(
            "Throughput: {:.3f} vectors/s, {:.3f} logical GFLOP/s".format(
                args.batch / median, logical_flops / median / 1e9
            )
        )
    print(
        "Bytes/call: packed weights={}, scales={}".format(
            format_bytes(packed_read), format_bytes(scale_read)
        )
    )
    print(
        "Bandwidth: Weight={:.3f} GB/s, Weight+scale={:.3f} GB/s".format(
            packed_read / median / 1e9, (packed_read + scale_read) / median / 1e9
        )
    )
    print("Checksum: {:.9g}".format(output.float().sum().item()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
