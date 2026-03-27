"""
Marlin/Cutlass kernel standalone benchmark: W4A16 → W8A8 → W4A8 三档对比
  W4A16 : ops.marlin_gemm  (fp16 activation, int4 weight)
  W8A8  : ops.cutlass_scaled_mm  (int8 activation, int8 weight)  ← vLLM 真实路径
  W4A8  : ops.marlin_gemm  (int8 activation, int4 weight)        ← 你 PR 修的路径

用法：
    CUDA_VISIBLE_DEVICES=3 python marlin_w4a8_bench.py --mode all --shapes qwen --m 1 4 8 16 32 64 128
    CUDA_VISIBLE_DEVICES=3 python marlin_w4a8_bench.py --mode w4a16
    CUDA_VISIBLE_DEVICES=3 python marlin_w4a8_bench.py --mode w4a8
"""
import argparse
import torch
import torch.utils.benchmark as benchmark

from vllm import _custom_ops as ops
from vllm.scalar_type import scalar_types
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_make_workspace_new,
    marlin_quant_input,
    marlin_permute_scales,
    marlin_act_int8_process_scales,
    should_use_atomic_add_reduce,
    USE_FP32_REDUCE_DEFAULT,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
    marlin_quantize,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    quantize_weights,
)

# ── Weight shapes ────────────────────────────────────────────────────────────

QWEN25_7B_SHAPES = [
    (3584,  3584),   # q_proj / o_proj
    (3584,   512),   # k_proj / v_proj (GQA)
    (3584, 18944),   # gate_proj / up_proj
    (18944, 3584),   # down_proj
]

LLAMA2_7B_SHAPES = [
    (4096,  4096),
    (4096, 11008),
    (11008, 4096),
]


# ── 数据准备 ─────────────────────────────────────────────────────────────────

def make_inputs(M: int, K: int, N: int, mode: str, group_size: int = 128):
    """
    mode='w4a16' : Marlin W4A16
    mode='w8a8'  : Cutlass W8A8 (int8 x int8, per-tensor scale)
    mode='w4a8'  : Marlin W4A8  (int8 activation x int4 weight)
    """
    torch.manual_seed(42)
    w = torch.randn(K, N, dtype=torch.float16, device='cuda')
    a_fp16 = torch.randn(M, K, dtype=torch.float16, device='cuda')

    if mode == 'w8a8':
        # W8A8: weight 量化成 int8，activation 量化成 int8
        # cutlass_scaled_mm 要求 weight layout: [K, N] int8，column-major 或 row-major
        # per-tensor scale：scale_a [1], scale_b [1]
        w_scale = w.abs().max() / 127.0
        w_int8 = (w / w_scale).clamp(-128, 127).round().to(torch.int8)
        w_ref = w_int8.to(torch.float16) * w_scale  # dequant reference

        a_scale = a_fp16.abs().max() / 127.0
        a_int8 = (a_fp16 / a_scale).clamp(-128, 127).round().to(torch.int8)

        # cutlass_scaled_mm 要求 b 满足 b.stride(0) == 1（column-major）
        # 正确做法：先分配 [N, K] 再 .t() 得到 shape=[K,N], stride=(1,N)
        # .t().contiguous() 是 row-major [N,K]，stride(0)=K，不满足
        w_int8_col = torch.empty(N, K, dtype=torch.int8, device='cuda').t()
        w_int8_col.copy_(w_int8)   # shape=[K,N], stride=(1,N) ← column-major ✓

        return dict(
            mode=mode,
            a_fp16=a_fp16,
            a_int8=a_int8,
            a_scale=a_scale.reshape(1).to(torch.float32),   # [1]
            w_int8=w_int8_col,                               # [N, K] for cutlass
            w_scale=w_scale.reshape(1).to(torch.float32),   # [1]
            w_ref=w_ref,
            M=M, K=K, N=N,
        )

    elif mode == 'w4a16':
        w_ref, marlin_q_w, marlin_s, g_idx, sort_indices, _ = marlin_quantize(
            w, scalar_types.uint4b8, group_size,
            act_order=False, input_dtype=None,
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
            M=M, K=K, N=N,
            group_size=group_size,
        )

    elif mode == 'w4a8':
        w_ref, marlin_q_w, marlin_s, g_idx, sort_indices, _ = marlin_quantize(
            w, scalar_types.uint4b8, group_size,
            act_order=False, input_dtype=torch.int8,
        )

        # W4A8 关键：weight scale 需要经过 marlin_act_int8_process_scales 变换
        # 它把 fp16 scale 压缩为 int16 表示（除以 max，乘以 4096，round），
        # 并返回 input_global_scale = 1/4096 * scale_max
        # 推理时：a_scales_final = per_token_scale * input_global_scale
        num_groups = K // group_size
        if num_groups > 1:
            marlin_s, input_global_scale = marlin_act_int8_process_scales(marlin_s)
        else:
            input_global_scale = None

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
            M=M, K=K, N=N,
            group_size=group_size,
        )

    else:
        raise ValueError(f"unknown mode: {mode}")


# ── Kernel 调用 ───────────────────────────────────────────────────────────────

def run_kernel(inp: dict) -> torch.Tensor:
    mode = inp['mode']
    M, K, N = inp['M'], inp['K'], inp['N']

    if mode == 'w8a8':
        # cutlass_scaled_mm(a, b, scale_a, scale_b, out_dtype)
        # a: [M, K] int8 row-major
        # b: [K, N] int8 column-major (stride=(1,N)，即原始 w_int8 的 column-major view)
        # 计算: a @ b * scale_a * scale_b → [M, N] fp16
        return ops.cutlass_scaled_mm(
            inp['a_int8'],    # [M, K] int8, row-major
            inp['w_int8'],    # [K, N] int8, column-major (stride(0)==1)
            inp['a_scale'],   # [1] fp32
            inp['w_scale'],   # [1] fp32
            torch.float16,
        )

    elif mode == 'w4a16':
        use_atomic_add = should_use_atomic_add_reduce(
            m=M, n=N, k=K, device=inp['a_fp16'].device, dtype=inp['a_fp16'].dtype,
        )
        return ops.marlin_gemm(
            inp['a_fp16'], None,
            inp['marlin_q_w'], None,
            inp['marlin_s'], None, None, None,
            inp['g_idx'], inp['sort_indices'],
            inp['workspace'],
            scalar_types.uint4b8,
            size_m=M, size_n=N, size_k=K,
            is_k_full=True,
            use_atomic_add=use_atomic_add,
            use_fp32_reduce=USE_FP32_REDUCE_DEFAULT,
            is_zp_float=False,
        )

    elif mode == 'w4a8':
        x_int8, a_scales = marlin_quant_input(inp['a_fp16'], torch.int8)
        # 当 input_global_scale 存在时（num_groups > 1），需要乘上去
        # 这样 a_scales 的物理意义从 fp32 per-token scale 变成了
        # "已经被 weight scale 的 max 归一化过的" 联合 scale
        if inp['input_global_scale'] is not None:
            a_scales = a_scales * inp['input_global_scale']
        use_atomic_add = should_use_atomic_add_reduce(
            m=M, n=N, k=K, device=inp['a_fp16'].device, dtype=x_int8.dtype,
        )
        return ops.marlin_gemm(
            x_int8, None,
            inp['marlin_q_w'], None,
            inp['marlin_s'], a_scales, None, None,
            inp['g_idx'], inp['sort_indices'],
            inp['workspace'],
            scalar_types.uint4b8,
            size_m=M, size_n=N, size_k=K,
            is_k_full=True,
            use_atomic_add=use_atomic_add,
            use_fp32_reduce=USE_FP32_REDUCE_DEFAULT,
            is_zp_float=False,
        )


# ── 精度验证 ──────────────────────────────────────────────────────────────────

def check_accuracy(inp: dict) -> dict:
    out_k = run_kernel(inp).float()
    # reference: a_fp16 @ w_ref (w_ref 是 dequant 后的 fp16 weight)
    out_ref = torch.matmul(inp['a_fp16'].float(), inp['w_ref'].float())

    diff = (out_k - out_ref).abs()
    max_abs = diff.max().item()
    denom = out_ref.abs().max().item()
    rel_err = max_abs / denom if denom > 1e-6 else float('nan')
    cos = torch.nn.functional.cosine_similarity(
        out_k, out_ref, dim=-1
    ).mean().item()
    return dict(max_abs=max_abs, rel_err=rel_err, cos_sim=cos)


# ── 性能测试 ──────────────────────────────────────────────────────────────────

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


def compute_bw_gbs(K, N, lat_us, mode, group_size=128):
    if mode == 'w8a8':
        w_bytes = K * N          # int8: 1 byte/elem
        s_bytes = 0              # per-tensor scale，忽略
    else:
        w_bytes = K * N // 2     # int4: 0.5 byte/elem
        s_bytes = (K // group_size) * N * 2  # fp16 scale
    return (w_bytes + s_bytes) / (lat_us * 1e-6) / 1e9


# ── 主程序 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--m', type=int, nargs='+',
                        default=[1, 4, 8, 16, 32, 64, 128])
    parser.add_argument('--shapes', choices=['qwen', 'llama', 'all'], default='qwen')
    parser.add_argument('--group-size', type=int, default=128)
    parser.add_argument('--mode', choices=['w4a16', 'w8a8', 'w4a8', 'all'],
                        default='all')
    parser.add_argument('--no-accuracy', action='store_true')
    args = parser.parse_args()

    shapes = (QWEN25_7B_SHAPES if args.shapes == 'qwen'
              else LLAMA2_7B_SHAPES if args.shapes == 'llama'
              else QWEN25_7B_SHAPES + LLAMA2_7B_SHAPES)
    modes = ['w4a16', 'w8a8', 'w4a8'] if args.mode == 'all' else [args.mode]

    print(f"\n{'='*90}")
    print(f"  Marlin/Cutlass Benchmark  |  group_size={args.group_size}"
          f"  |  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  W4A16/W4A8 → ops.marlin_gemm  |  W8A8 → ops.cutlass_scaled_mm")
    print(f"{'='*90}")

    for mode in modes:
        print(f"\n── {mode.upper()} {'─'*62}")
        print(f"  {'Shape(K,N)':18s} {'M':>5s} {'Lat(us)':>10s} {'TFLOPS':>8s}"
              f" {'BW(GB/s)':>10s} {'MaxErr':>10s} {'CosSim':>10s}")
        print(f"  {'-'*78}")

        for K, N in shapes:
            for M in args.m:
                inp = make_inputs(M, K, N, mode, args.group_size)
                lat = bench_latency_us(inp)
                tflops = compute_tflops(M, K, N, lat)
                bw = compute_bw_gbs(K, N, lat, mode, args.group_size)

                if not args.no_accuracy:
                    acc = check_accuracy(inp)
                    max_err, cos = acc['max_abs'], acc['cos_sim']
                else:
                    max_err, cos = float('nan'), float('nan')

                print(f"  ({K:5d},{N:5d})      {M:5d} {lat:10.2f} {tflops:8.3f}"
                      f" {bw:10.1f} {max_err:10.5f} {cos:10.6f}")
            print()

    print(f"L40 理论峰值: INT8 362 TOPS | 内存带宽 864 GB/s")
    print(f"decode (小 M) → memory-bound，BW 是核心指标")
    print(f"prefill (大 M) → compute-bound，TFLOPS 是核心指标\n")


if __name__ == '__main__':
    main()