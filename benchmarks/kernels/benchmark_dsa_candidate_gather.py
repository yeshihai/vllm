# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure the SM90 candidate-gather decode path against dense scoring.

The candidate-only decode path (``VLLM_DSA_CANDIDATE_GATHER_DECODE``) trades
a per-step fixed cost for scoring work that no longer scales with the context
length. Its profitability model needs exactly one measured input, the fixed
cost in microseconds, which this script reports for the current device:

    python benchmarks/kernels/benchmark_dsa_candidate_gather.py

Set the reported value as ``VLLM_DSA_CANDIDATE_GATHER_FIXED_COST_US``. The
row and context dependence is derived from shapes and the device's memory
bandwidth, so nothing else needs per-device tuning.

Only the candidate path is timed, which is enough to fit the fixed cost and
keeps the footprint to a few hundred MB: the gather path reads a fixed
candidate width per row, so its cost is almost independent of the context
length, while dense cost is a well-understood linear stream over the context.
"""

import argparse

import torch

from vllm.model_executor.kernels.attention.dsa.candidate_gather import (
    _ACHIEVABLE_BANDWIDTH_FRACTION,
    _GATHER_PASSES,
    _INDEXER_SCALE_BYTES,
    _device_bandwidth_bytes_per_us,
    candidate_gather_score_decode,
    candidate_gather_workspace_specs,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import has_deep_gemm
from vllm.v1.worker import workspace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=2048,
        help="Candidate blocks per row; must be a power of two.",
    )
    parser.add_argument("--candidate-block-size", type=int, default=8)
    parser.add_argument("--topk", type=int, default=512)
    parser.add_argument("--cache-block-size", type=int, default=64)
    parser.add_argument(
        "--context",
        type=int,
        default=1 << 18,
        help="Compressed indexer context backing the cache. Only has to be "
        "long enough to scatter the candidates across pages.",
    )
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 6, 12, 24, 48, 96])
    parser.add_argument("--topk-backend", type=str, default="auto")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=20)
    return parser.parse_args()


def build_inputs(args: argparse.Namespace, rows: int, device: torch.device):
    fp8 = current_platform.fp8_dtype()
    num_blocks = args.context // args.cache_block_size
    entry_bytes = args.head_dim + _INDEXER_SCALE_BYTES
    page_bytes = args.cache_block_size * entry_bytes
    backing = torch.zeros((num_blocks, page_bytes), dtype=torch.uint8, device=device)
    payload = torch.randn(
        num_blocks, args.cache_block_size * args.head_dim, device=device
    ).to(fp8)
    scales = torch.ones(num_blocks, args.cache_block_size, device=device)
    flat = backing.view(num_blocks, args.cache_block_size, entry_bytes)
    flat[:, :, : args.head_dim] = payload.view(
        num_blocks, args.cache_block_size, args.head_dim
    ).view(torch.uint8)
    flat[:, :, args.head_dim :] = scales.view(torch.uint8).view(
        num_blocks, args.cache_block_size, _INDEXER_SCALE_BYTES
    )
    # Scatter pages and candidates so the gather streams from DRAM rather
    # than replaying one cache-resident page.
    block_table = torch.randperm(num_blocks, device=device).to(torch.int32)[None]
    candidates = torch.randperm(
        args.context // args.candidate_block_size, device=device
    )[: args.num_candidates].to(torch.int32)[None]
    return dict(
        q=torch.randn(rows, args.num_heads, args.head_dim, device=device).to(fp8),
        kv_cache=flat,
        weights=torch.rand(rows, args.num_heads, device=device),
        seq_lens=torch.full((rows, 1), args.context, device=device, dtype=torch.int32),
        block_table=block_table.expand(rows, -1),
        candidate_blocks=candidates.expand(rows, -1),
        topk_indices=torch.empty(rows, args.topk, device=device, dtype=torch.int32),
        fp8_dtype=fp8,
    )


def time_rows(args: argparse.Namespace, rows: int, device: torch.device) -> float:
    inputs = build_inputs(args, rows, device)
    workspace.current_workspace_manager().get_simultaneous(
        *candidate_gather_workspace_specs(
            rows,
            args.num_candidates,
            args.candidate_block_size,
            args.head_dim,
            args.topk,
        )
    )

    def once() -> None:
        candidate_gather_score_decode(
            inputs["q"],
            inputs["kv_cache"],
            inputs["weights"],
            inputs["seq_lens"],
            inputs["block_table"],
            inputs["candidate_blocks"],
            args.candidate_block_size,
            1,
            args.topk,
            args.topk_backend,
            inputs["topk_indices"],
            args.head_dim,
            args.context,
            inputs["fp8_dtype"],
        )

    for _ in range(args.warmup):
        once()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(args.iters):
        once()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1.0e3 / args.iters


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required")
    if not current_platform.is_device_capability_family(90) or not has_deep_gemm():
        print("warning: this path is only selected on SM90 with DeepGEMM\n")
    device = torch.device("cuda", torch.cuda.current_device())
    workspace.init_workspace_manager(device)

    candidate_width = args.num_candidates * args.candidate_block_size
    bytes_per_position = args.head_dim + _INDEXER_SCALE_BYTES
    bandwidth = _device_bandwidth_bytes_per_us()
    gather_bytes_per_row = _GATHER_PASSES * candidate_width * bytes_per_position

    print(f"device:          {torch.cuda.get_device_name(device)}")
    print(
        f"bandwidth:       {bandwidth / 1e6:.2f} TB/s assumed achievable "
        f"({_ACHIEVABLE_BANDWIDTH_FRACTION:.0%} of peak)"
    )
    print(f"candidate width: {candidate_width} compressed positions")
    print(f"context:         {args.context}\n")
    print(f"{'rows':>6}{'measured us':>14}{'modelled us':>14}{'fixed us':>12}")

    fixed_estimates = []
    for rows in args.rows:
        measured = time_rows(args, rows, device)
        streaming = rows * gather_bytes_per_row / bandwidth
        fixed = measured - streaming
        fixed_estimates.append(fixed)
        print(f"{rows:>6}{measured:>14.1f}{streaming:>14.1f}{fixed:>12.1f}")

    # The fixed term is what remains once the row-proportional streaming cost
    # is removed; take the max so the recommendation stays conservative.
    recommended = max(fixed_estimates)
    print(
        f"\nVLLM_DSA_CANDIDATE_GATHER_FIXED_COST_US={recommended:.0f}\n"
        "Rounding up is the safe direction: it withholds the optimization "
        "from batches it would have helped, rather than regressing them."
    )
    dense_per_row_us = bytes_per_position / bandwidth
    breakeven = _GATHER_PASSES * candidate_width
    threshold = recommended / dense_per_row_us
    example_context = 1 << 20
    example_rows = max(1, round(threshold / (example_context - breakeven)))
    print(
        "\nWith this value the path turns on once\n"
        f"  rows * (context - {breakeven}) >= {threshold:.3g}\n"
        f"e.g. at context={example_context} that is {example_rows} rows."
    )


if __name__ == "__main__":
    main()
