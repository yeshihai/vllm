# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark script to compare W4A16 (Marlin) vs W8A8 (INT8 GEMM)
"""

import argparse
import itertools
import os
import copy
import torch
import torch.utils.benchmark as benchmark
from weight_shapes import WEIGHT_SHAPES

from vllm import _custom_ops as ops
from vllm._custom_ops import cutlass_scaled_mm as vllm_scaled_mm
from vllm._custom_ops import scaled_int8_quant as vllm_scaled_int8_quant
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    GPTQ_MARLIN_MAX_PARALLEL,
    GPTQ_MARLIN_MIN_THREAD_N,
    MARLIN_SUPPORTED_GROUP_SIZES,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
    MarlinWorkspace,
    marlin_quantize,
)
from vllm.scalar_type import scalar_types

# Default batch sizes for benchmarking
DEFAULT_BATCH_SIZES = [1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]


def quantize_int8_weight(b, w_type, device):
    """Quantize weight to INT8."""
    if w_type == "tensor":
        scale_b = torch.ones(1, device=device, dtype=torch.float32)
        b_int8, scale_b_int8, _ = vllm_scaled_int8_quant(b, scale_b)
        assert scale_b_int8.numel() == 1
    else:  # channel
        b_int8, scale_b_int8, _ = vllm_scaled_int8_quant(b)
        assert scale_b_int8.numel() == b.shape[0]
    return b_int8.t(), scale_b_int8


def quantize_int8_activation(a, scale=None):
    """Quantize activation to INT8."""
    if scale is None:
        a_int8, scale_a, _ = vllm_scaled_int8_quant(a)
    else:
        a_int8, scale_a, _ = vllm_scaled_int8_quant(a, scale)
    return a_int8, scale_a


def prepare_w8a8_data(a, b, device):
    """Prepare data for W8A8 INT8 GEMM."""
    # Per-channel weight quantization
    b_int8_t, scale_b = quantize_int8_weight(b, "channel", device)
    # Per-token activation quantization
    a_int8, scale_a = quantize_int8_activation(a)
    return a_int8, b_int8_t, scale_a, scale_b


def run_benchmark(
    results: list,
    size_m: int,
    size_k: int,
    size_n: int,
    group_size: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    """Run benchmark comparing W4A16 and W8A8."""
    
    # Generate random input data
    a = torch.randn((size_m, size_k), device=device, dtype=dtype)
    b = torch.randn((size_n, size_k), device=device, dtype=dtype)
    
    label = "Quant GEMM Comparison"
    sub_label = f"MKN=({size_m}x{size_k}x{size_n}), g={group_size}"
    
    # ========== BF16 Baseline ==========
    globals_dict = {"a": a, "b": b}
    results.append(
        benchmark.Timer(
            stmt="torch.matmul(a, b.t())",
            globals=globals_dict,
            label=label,
            sub_label=sub_label,
            description="BF16_gemm",
        ).blocked_autorange(min_run_time=0.5)
    )
    
    # ========== W8A8 INT8 GEMM ==========
    a_int8, b_int8_t, scale_a, scale_b = prepare_w8a8_data(a, b, device)
    
    globals_w8a8 = {
        "a_int8": a_int8,
        "b_int8_t": b_int8_t,
        "scale_a": scale_a,
        "scale_b": scale_b,
        "dtype": dtype,
        "vllm_scaled_mm": vllm_scaled_mm,
    }
    
    # W8A8 with per-token activation quantization (dynamic quant)
    results.append(
        benchmark.Timer(
            stmt="vllm_scaled_mm(a_int8, b_int8_t, scale_a, scale_b, dtype)",
            globals=globals_w8a8,
            label=label,
            sub_label=sub_label,
            description="W8A8_int8_gemm",
        ).blocked_autorange(min_run_time=0.5)
    )
    
    # ========== W4A16 Marlin ==========
    # Use uint4b8 which is the standard GPTQ-style uint4 with bias=8
    quant_type = scalar_types.uint4b8
    
    # Check if group_size is valid for Marlin
    if group_size not in MARLIN_SUPPORTED_GROUP_SIZES and group_size != -1:
        return  # Skip silently
    
    # For group_size != -1, check divisibility
    if group_size != -1 and size_k % group_size != 0:
        return  # Skip silently
    
    try:
        w_ref, marlin_q_w, marlin_s, marlin_g_idx, marlin_sort_indices, _ = (
            marlin_quantize(b, quant_type, group_size, act_order=False)
        )
    except Exception as e:
        print(f"  Warning: W4 quantization failed: {e}")
        return
    
    marlin_workspace = MarlinWorkspace(
        size_n, GPTQ_MARLIN_MIN_THREAD_N, GPTQ_MARLIN_MAX_PARALLEL
    )
    
    # Direct function call instead of string stmt
    def run_marlin_gemm():
        return ops.marlin_gemm(
            a,  # a
            None,  # a_tmp
            marlin_q_w,  # b_q_weight
            marlin_s,  # b_scales
            None,  # b_scales_2
            None,  # workspace
            None,  # b_zeros
            marlin_g_idx,  # g_idx
            marlin_sort_indices,  # sort_indices
            marlin_workspace.scratch,  # marlin_workspace
            quant_type,  # b_q_type - ScalarType object
            size_m,  # size_m
            size_n,  # size_n
            size_k,  # size_k
            True,  # is_k_full
            False,  # has_zp
            False,  # use_fp32_reduce
            False,  # is_zp_float
        )
    
    # Warmup
    for _ in range(3):
        run_marlin_gemm()
    
    # Benchmark using manual timing
    import time
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        run_marlin_gemm()
    torch.cuda.synchronize()
    end = time.perf_counter()
    avg_time_ms = (end - start) / 10 * 1000
    
    # Store as a simple result
    class SimpleResult:
        def __init__(self, median_time, description, sub_label):
            self.median = median_time
            self.description = description
            self.sub_label = sub_label
    
    results.append(SimpleResult(avg_time_ms / 1000, "W4A16_marlin", sub_label))
    
    # ========== INT8 activation quantization overhead ==========
    globals_w4a8_quant = {
        "a": a,
        "vllm_scaled_int8_quant": vllm_scaled_int8_quant,
    }
    
    results.append(
        benchmark.Timer(
            stmt="vllm_scaled_int8_quant(a)",
            globals=globals_w4a8_quant,
            label=label,
            sub_label=sub_label,
            description="INT8_act_quant_only",
        ).blocked_autorange(min_run_time=0.5)
    )


def compute_tflops(m, n, k, time_ms):
    """Compute TFLOPS from dimensions and time in milliseconds."""
    flops = 2 * m * n * k  # multiply-add
    return flops * 1e-12 / (time_ms * 1e-3)


def print_results_table(results: list, batch_sizes: list):
    """Print results in a formatted table."""
    print("\n" + "=" * 120)
    print(f"{'Shape (M,K,N)':<25} {'BF16':>12} {'W8A8':>12} {'W4A16':>12} {'INT8_quant':>12}")
    print("=" * 120)
    
    # Group results by shape
    grouped = {}
    for r in results:
        key = r.sub_label
        if key not in grouped:
            grouped[key] = {}
        grouped[key][r.description] = r
    
    for sub_label, measurements in grouped.items():
        row = [sub_label]
        for desc in ["BF16_gemm", "W8A8_int8_gemm", "W4A16_marlin", "INT8_act_quant_only"]:
            if desc in measurements:
                time_ms = measurements[desc].median * 1000
                row.append(f"{time_ms:.4f}ms")
            else:
                row.append("N/A")
        print(f"{row[0]:<25} {row[1]:>12} {row[2]:>12} {row[3]:>12} {row[4]:>12}")
    
    print("=" * 120)


def print_tflops_table(results: list, batch_sizes: list):
    """Print TFLOPS results in a formatted table."""
    print("\n" + "=" * 140)
    print(f"{'Shape (M,K,N)':<30} {'BF16 TFLOPS':>15} {'W8A8 TFLOPS':>15} {'W4A16 TFLOPS':>15} {'W8A8/BF16':>12} {'W4A16/BF16':>12}")
    print("=" * 140)
    
    # Group results by shape
    grouped = {}
    for r in results:
        key = r.sub_label
        if key not in grouped:
            grouped[key] = {}
        grouped[key][r.description] = r
    
    for sub_label, measurements in grouped.items():
        # Parse M, K, N from sub_label
        parts = sub_label.split(",")
        mkn_part = parts[0].replace("MKN=(", "").replace(")", "")
        m, k, n = map(int, mkn_part.split("x"))
        
        row = [sub_label]
        times = {}
        for desc in ["BF16_gemm", "W8A8_int8_gemm", "W4A16_marlin"]:
            if desc in measurements:
                time_ms = measurements[desc].median * 1000
                tflops = compute_tflops(m, n, k, time_ms)
                times[desc] = time_ms
                row.append(f"{tflops:.2f}")
            else:
                row.append("N/A")
        
        # Calculate speedup vs BF16
        if "BF16_gemm" in times:
            bf16_time = times["BF16_gemm"]
            for desc in ["W8A8_int8_gemm", "W4A16_marlin"]:
                if desc in times:
                    speedup = bf16_time / times[desc]
                    row.append(f"{speedup:.2f}x")
                else:
                    row.append("N/A")
        else:
            row.extend(["N/A", "N/A"])
        
        print(f"{row[0]:<30} {row[1]:>15} {row[2]:>15} {row[3]:>15} {row[4]:>12} {row[5]:>12}")
    
    print("=" * 140)


def main(args):
    print("=" * 80)
    print("W4A16 vs W8A8 Performance Benchmark")
    print("=" * 80)
    print(f"Group size: {args.group_size}")
    print(f"Supported group sizes: {MARLIN_SUPPORTED_GROUP_SIZES + [-1]}")
    print("=" * 80)
    
    # Prepare shapes from model configurations
    shapes = []
    for model, tp_size in itertools.product(args.models, args.tp_sizes):
        for KN, tp_dim in copy.deepcopy(WEIGHT_SHAPES[model]):
            KN[tp_dim] //= tp_size
            shapes.append((KN[0], KN[1], model))  # (K, N, model_name)
    
    # Add custom shapes if provided
    for k, n in args.shapes:
        shapes.append((k, n, "custom"))
    
    # Remove duplicates and sort
    seen = set()
    unique_shapes = []
    for k, n, model in shapes:
        if (k, n) not in seen:
            seen.add((k, n))
            unique_shapes.append((k, n, model))
    
    results: list = []
    
    for k, n, model in unique_shapes:
        print(f"\nBenchmarking: {model}, K={k}, N={n}")
        for m in args.batch_sizes:
            print(f"  M={m}...", end="", flush=True)
            try:
                run_benchmark(results, m, k, n, args.group_size)
                print(" done")
            except Exception as e:
                print(f" failed: {e}")
                import traceback
                traceback.print_exc()
    
    # Print results
    print("\n\n" + "=" * 80)
    print("BENCHMARK RESULTS (Time in ms)")
    print("=" * 80)
    print_results_table(results, args.batch_sizes)
    
    print("\n\n" + "=" * 80)
    print("BENCHMARK RESULTS (TFLOPS)")
    print("=" * 80)
    print_tflops_table(results, args.batch_sizes)
    
    # Save results to file if specified
    if args.output:
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
        with open(args.output, "w") as f:
            f.write("Shape,Description,Median_ms,Mean_ms,Std_ns\n")
            for r in results:
                f.write(f"{r.sub_label},{r.description},{r.median * 1000},{r.median * 1000},0\n")
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark W4A16 vs W8A8 GEMM performance"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=["meta-llama/Llama-3.1-8B-Instruct"],
        choices=list(WEIGHT_SHAPES.keys()),
        help="Models to benchmark (for shape extraction)",
    )
    parser.add_argument(
        "--tp-sizes",
        nargs="+",
        type=int,
        default=[1],
        help="Tensor parallel sizes",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=DEFAULT_BATCH_SIZES,
        help="Batch sizes (M dimension) to benchmark",
    )
    parser.add_argument(
        "--shapes",
        nargs="+",
        type=lambda x: tuple(map(int, x.split(","))),
        default=[],
        help="Custom shapes in K,N format (e.g., --shapes 4096,4096 8192,8192)",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=128,
        choices=MARLIN_SUPPORTED_GROUP_SIZES + [-1],
        help="Group size for W4 quantization",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output file to save results (CSV format)",
    )
    
    args = parser.parse_args()
    main(args)