"""
Marlin/Cutlass kernel standalone benchmark: W4A16 -> W8A8 -> W4A8
  W4A16 : ops.marlin_gemm       (fp16 activation, int4 weight)
  W8A8  : ops.cutlass_scaled_mm (int8 activation, int8 weight)
  W4A8  : ops.marlin_gemm       (int8 activation, int4 weight)

Usage:
  CUDA_VISIBLE_DEVICES=3 python marlin_w4a8_bench.py --mode all --shapes qwen --m 1 4 8 16 32 64 128
"""
import argparse
import torch
import torch.utils.benchmark as benchmark

from vllm import _custom_ops as ops
from vllm.scalar_type import scalar_types
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_make_workspace_new,
    marlin_quant_input,
    marlin_act_int8_process_scales,
    should_use_atomic_add_reduce,
    USE_FP32_REDUCE_DEFAULT,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
    marlin_quantize,
)

QWEN25_7B_SHAPES = [
    (3584, 3584),
    (3584, 512),
    (3584, 18944),
    (18944, 3584),
]

LLAMA2_7B_SHAPES = [
    (4096, 4096),
    (4096, 11008),
    (11008, 4096),
]

L40_BW_GBS = 864.0


def make_inputs(M: int, K: int, N: int, mode: str, group_size: int = 128):
    torch.manual_seed(42)
    w = torch.randn(K, N, dtype=torch.float16, device="cuda")
    a_fp16 = torch.randn(M, K, dtype=torch.float16, device="cuda")

    if mode == "w8a8":
        w_scale = w.abs().max() / 127.0
        w_int8 = (w / w_scale).clamp(-128, 127).round().to(torch.int8)
        w_ref = w_int8.to(torch.float16) * w_scale

        a_scale = a_fp16.abs().max() / 127.0
        a_int8 = (a_fp16 / a_scale).clamp(-128, 127).round().to(torch.int8)

        # cutlass_scaled_mm requires b.stride(0) == 1
        w_int8_col = torch.empty(N, K, dtype=torch.int8, device="cuda").t()
        w_int8_col.copy_(w_int8)

        return dict(
            mode=mode,
            a_fp16=a_fp16,
            a_int8=a_int8,
            a_scale=a_scale.reshape(1).to(torch.float32),
            w_int8=w_int8_col,
            w_scale=w_scale.reshape(1).to(torch.float32),
            w_ref=w_ref,
            M=M,
            K=K,
            N=N,
        )

    if mode == "w4a16":
        w_ref, marlin_q_w, marlin_s, g_idx, sort_indices, _ = marlin_quantize(
            w, scalar_types.uint4b8, group_size, act_order=False, input_dtype=None
        )
        workspace = marlin_make_workspace_new(a_fp16.device)
        return dict(
            mode=mode,
            a_fp16=a_fp16,
            marlin_q_w=marlin_q_w,
            marlin_s=marlin_s,
            g_idx=g_idx,
            sort_indices=sort_indices,
            workspace=workspace,
            w_ref=w_ref,
            M=M,
            K=K,
            N=N,
            group_size=group_size,
        )

    if mode == "w4a8":
        # input_dtype=int8 gives W4A8-compatible packed weight/scales
        w_ref, marlin_q_w, marlin_s_raw, g_idx, sort_indices, _ = marlin_quantize(
            w, scalar_types.uint4b8, group_size, act_order=False, input_dtype=torch.int8
        )

        # IMPORTANT: process marlin_s for W4A8 path
        # returns (processed_scales_int16_view, input_global_scale_fp32_scalar)
        marlin_s, input_global_scale = marlin_act_int8_process_scales(marlin_s_raw)

        workspace = marlin_make_workspace_new(a_fp16.device)
        return dict(
            mode=mode,
            a_fp16=a_fp16,
            marlin_q_w=marlin_q_w,
            marlin_s=marlin_s,
            input_global_scale=input_global_scale,
            g_idx=g_idx,
            sort_indices=sort_indices,
            workspace=workspace,
            w_ref=w_ref,
            M=M,
            K=K,
            N=N,
            group_size=group_size,
        )

    raise ValueError(f"unknown mode: {mode}")


def run_kernel(inp: dict) -> torch.Tensor:
    mode = inp["mode"]
    M, K, N = inp["M"], inp["K"], inp["N"]

    if mode == "w8a8":
        return ops.cutlass_scaled_mm(
            inp["a_int8"],
            inp["w_int8"],
            inp["a_scale"],
            inp["w_scale"],
            torch.float16,
        )

    if mode == "w4a16":
        use_atomic_add = should_use_atomic_add_reduce(
            m=M, n=N, k=K, device=inp["a_fp16"].device, dtype=inp["a_fp16"].dtype
        )
        return ops.marlin_gemm(
            inp["a_fp16"],
            None,
            inp["marlin_q_w"],
            None,
            inp["marlin_s"],
            None,
            None,
            None,
            inp["g_idx"],
            inp["sort_indices"],
            inp["workspace"],
            scalar_types.uint4b8,
            size_m=M,
            size_n=N,
            size_k=K,
            is_k_full=True,
            use_atomic_add=use_atomic_add,
            use_fp32_reduce=USE_FP32_REDUCE_DEFAULT,
            is_zp_float=False,
        )

    if mode == "w4a8":
        x_int8, a_scales = marlin_quant_input(inp["a_fp16"], torch.int8)

        # CRITICAL FIX:
        # final per-token scales must include global factor from marlin_act_int8_process_scales
        a_scales_final = a_scales * inp["input_global_scale"]

        use_atomic_add = should_use_atomic_add_reduce(
            m=M, n=N, k=K, device=inp["a_fp16"].device, dtype=x_int8.dtype
        )
        return ops.marlin_gemm(
            x_int8,
            None,
            inp["marlin_q_w"],
            None,
            inp["marlin_s"],
            a_scales_final,
            None,
            None,
            inp["g_idx"],
            inp["sort_indices"],
            inp["workspace"],
            scalar_types.uint4b8,
            size_m=M,
            size_n=N,
            size_k=K,
            is_k_full=True,
            use_atomic_add=use_atomic_add,
            use_fp32_reduce=USE_FP32_REDUCE_DEFAULT,
            is_zp_float=False,
        )

    raise ValueError(f"unknown mode: {mode}")


def check_accuracy(inp: dict) -> dict:
    out_k = run_kernel(inp).float()
    out_ref = torch.matmul(inp["a_fp16"].float(), inp["w_ref"].float())

    diff = (out_k - out_ref).abs()
    max_abs = diff.max().item()
    denom = out_ref.abs().max().item()
    rel_err = max_abs / denom if denom > 1e-6 else float("nan")
    cos = torch.nn.functional.cosine_similarity(out_k, out_ref, dim=-1).mean().item()
    return dict(max_abs=max_abs, rel_err=rel_err, cos_sim=cos)


def bench_latency_us(inp: dict, warmup: int = 20, repeat: int = 200) -> float:
    for _ in range(warmup):
        run_kernel(inp)
    torch.cuda.synchronize()
    t = benchmark.Timer(
        stmt="run_kernel(inp); torch.cuda.synchronize()",
        globals={"run_kernel": run_kernel, "inp": inp, "torch": torch},
        num_threads=1,
    )
    return t.timeit(repeat).mean * 1e6


def compute_tflops(M, K, N, lat_us):
    return 2 * M * K * N / (lat_us * 1e-6) / 1e12


def estimate_bytes_per_gemm(M, K, N, mode, group_size=128):
    # Effective traffic model (for comparability), not raw DRAM counter.
    a_bytes_fp16 = M * K * 2
    a_bytes_int8 = M * K * 1
    c_bytes_fp16 = M * N * 2

    if mode == "w8a8":
        w_bytes = K * N
        s_bytes = 8
        a_bytes = a_bytes_int8
    else:
        w_bytes = (K * N) // 2
        s_bytes = (K // group_size) * N * 2
        a_bytes = a_bytes_fp16 if mode == "w4a16" else a_bytes_int8

    return a_bytes + w_bytes + s_bytes + c_bytes_fp16


def compute_eff_bw_gbs(M, K, N, lat_us, mode, group_size=128):
    total_bytes = estimate_bytes_per_gemm(M, K, N, mode, group_size)
    return total_bytes / (lat_us * 1e-6) / 1e9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, nargs="+", default=[1, 4, 8, 16, 32, 64, 128])
    parser.add_argument("--shapes", choices=["qwen", "llama", "all"], default="qwen")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--mode", choices=["w4a16", "w8a8", "w4a8", "all"], default="all")
    parser.add_argument("--no-accuracy", action="store_true")
    args = parser.parse_args()

    shapes = (
        QWEN25_7B_SHAPES
        if args.shapes == "qwen"
        else LLAMA2_7B_SHAPES
        if args.shapes == "llama"
        else QWEN25_7B_SHAPES + LLAMA2_7B_SHAPES
    )
    modes = ["w4a16", "w8a8", "w4a8"] if args.mode == "all" else [args.mode]

    print(f"\n{'='*98}")
    print(
        f"  Marlin/Cutlass Benchmark | group_size={args.group_size} "
        f"| GPU: {torch.cuda.get_device_name(0)}"
    )
    print("  W4A16/W4A8 -> ops.marlin_gemm | W8A8 -> ops.cutlass_scaled_mm")
    print(f"{'='*98}")

    for mode in modes:
        print(f"\n-- {mode.upper()} {'-'*68}")
        print(
            f"  {'Shape(K,N)':18s} {'M':>5s} {'Lat(us)':>10s} {'TFLOPS':>8s}"
            f" {'EffBW':>10s} {'BW%':>8s} {'MaxErr':>10s} {'CosSim':>10s}"
        )
        print(f"  {'-'*92}")

        for K, N in shapes:
            for M in args.m:
                inp = make_inputs(M, K, N, mode, args.group_size)
                lat = bench_latency_us(inp)
                tflops = compute_tflops(M, K, N, lat)
                eff_bw = compute_eff_bw_gbs(M, K, N, lat, mode, args.group_size)
                bw_ratio = eff_bw / L40_BW_GBS * 100.0
                bw_flag = "*" if eff_bw > L40_BW_GBS else " "

                if not args.no_accuracy:
                    acc = check_accuracy(inp)
                    max_err_str = f"{acc['max_abs']:.5f}"
                    cos_str = f"{acc['cos_sim']:.6f}"
                else:
                    max_err_str = "-"
                    cos_str = "-"

                print(
                    f"  ({K:5d},{N:5d}) {M:6d} {lat:10.2f} {tflops:8.3f}"
                    f" {eff_bw:10.1f}{bw_flag} {bw_ratio:7.1f}%"
                    f" {max_err_str:>10s} {cos_str:>10s}"
                )
            print()

    print(f"L40 theoretical peak: INT8 362 TOPS | Mem BW {L40_BW_GBS:.0f} GB/s")
    print("decode (small M): usually memory-bound, watch EffBW")
    print("prefill (large M): usually compute-bound, watch TFLOPS")
    print("* EffBW is model-estimated effective bandwidth, not raw DRAM counter.\n")


if __name__ == "__main__":
    main()