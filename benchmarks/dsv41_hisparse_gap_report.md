# DSv4.1-Flash HiSparse：ON/OFF prefill gap 优化报告

日期：2026-11-04
分支：`feat/dsv41-hisparse-gap` @ `2877d8ea5d`（已推送 yeshihai/vllm）
基线交付分支：`feat/dsv41-c2-host-back` @ `098251c782`（tree 与备份 `backup/dsv41-c2-preconverge-20261104` @ `0b33846e15` 完全一致）

## 0. 结论（TL;DR）

- 优化内容：prefill 时同一个 KV source 的压缩 KV gather，从"每个消费层各 gather 一次"改成"每个 KV source 每个 chunk 只 gather 一次，消费层复用 workspace 尾部保留区"。
- 效果（8×H20，TP8/EP8，cold prefill，c1，OSL 64）：
  - ON prefill TTFT：131k 16.73s → **10.52s（−37.1%）**；262k 47.76 → **23.39（−51.0%）**；524k 152.7 → **55.96（−63.4%）**；1M 536.9 → **153.45（−71.4%）**。
  - HiSparse ON−OFF gap：131k 6.86s → **0.68s（−90.1%）**；262k 26.78 → **2.50（−90.7%）**；524k 105.7 → **9.46（−91.0%）**；1M 418.4 → **36.72（−91.2%）**。
- 不变式全部保持：
  - 容量：47,870,492 tokens = **45.65×**（与优化前一致）。
  - 全量 GSM8K 1319：ON **91.51%**（优化前 90.98%），OFF **91.13%**（优化前 91.13%）。
  - 1M 共享前缀 decode：c1 ITL med/p99 **89.15/91.18ms**（优化前 89.34/91.58）；c4 **89.01/91.76**（优化前 89.29/91.65），无回归。

## 1. 固定测试标准

### 硬件/配置
8×H20（SM90），TP8/EP8，`max_model_len=1048576`，`max_num_batched_tokens=4096`，`max_num_seqs=32`，`gpu_memory_utilization=0.92`，`--enforce-eager`，`kv-cache-dtype=fp8_ds_mla`，无投机解码。
- ON：`HiSparseConnector`，`host_pool_gib=128`。
- OFF：纯 GPU baseline，其余 flags 相同，`--kernel-config {"linear_backend":"marlin","moe_backend":"marlin","enable_jit_warmup":false}`。

### Prefill TTFT（gap 主指标）
- exact ISL ∈ {131072, 262144, 524288, 1048000}，OSL 64，concurrency 1，temperature 0，random dataset，`random-range-ratio 0`。
- 1 warmup + measured（131k 点 measured 4 条，seed 201；其余 seed 701/702/703）。
- 服务端 `--no-enable-prefix-caching`，冷前缀；指标取 `vllm bench serve` 的 median TTFT。

### Decode（部署相关场景）
- 1M 共享前缀：`prefix_repetition`，prefix 1,040,384 + suffix 100 + output 4096，seed 12345，显式 prewarm，c1/c4。
- 指标：ITL median/p99、output tok/s、TTFT。

### 精度
- GSM8K 8-shot，全量 1319，concurrency 16，temperature 0（脚本默认）。
- 另跑 salted shared-prefix GSM8K-200（prefix-cache replay 路径守卫）。

### 不变式/验收
- 容量（startup `GPU + host KV cache size`）不得下降。
- 全量 GSM8K 与 baseline 差异 ≤ 噪声（±1pp 量级）；salted 不得出现 replay bug 的坍塌（~82%）。
- 1M 共享前缀 decode ITL 变化 ≤ ±2%。

## 2. Profile 证据（131k 单请求，成对 torch profiler）

| 指标 | ON | OFF |
|---|---|---|
| TTFT | 16.816 s | 9.925 s |
| `_dequantize_and_gather_k_kernel` | **6.593 s（1292 calls，40.5% device time）** | 不在 OFF 热点表中（device cache 走不同 dispatch） |
| `_rope_quant_insert_kernel`（compressor host 写） | 47.2 ms | 1.1 ms |
| `cp_gather_indexer_k_quant_cache`（indexer gather） | 2.5 ms | 2.5 ms |
| `Memcpy DtoD` | 70.7 ms | 70.4 ms |

结论：6.89s 的 ON−OFF gap 中约 6.59s（96%）来自 `_dequantize_and_gather_k_kernel` 读取 host-backed 压缩源 cache；compressor host 写、indexer gather、spill memcpy 全部可忽略。

根因：C1+C2 host-back 后，压缩源层的 `kv_cache` 是 `host_pool.allocate()` 的 host 视图（`vllm/v1/hisparse/binding.py::allocate_hisparse_kv_caches`）。prefill 的每个 cr>0 层都从该 host 视图 gather 同一份压缩 KV：C1 source 20 被 20 层消费，C2 source 2/8/14 各被 6 层消费 → 每步 38 次 gather，实际只需 4 次（4 个 source 各一次），约 11.6× 冗余字节。

## 3. 改动（4 个文件，+150/−21）

1. `vllm/v1/worker/workspace.py`
   - 新增 `WorkspaceManager.get_retained(key, shape, dtype) -> (view, intact)`：workspace 尾部保留区；`intact` 仅在"同 key、同 size、期间无更大的 `get_simultaneous` 触及尾部、未 realloc"时为真。
   - `get_simultaneous` 在请求越过保留区起点时清空 `_retained`；resize 时清空；per ubatch/lane slot。
2. `vllm/models/deepseek_v41/nvidia/flashmla.py::_forward_prefill`
   - 压缩 KV gather 写入 retained 尾部；`is_kv_source` 层每步必 gather（本步刚写入新行），消费层在 `intact` 时复用，只重写自己的 SWA 列。
   - key = (压缩源 prefix, chunk_start, chunk_end)。
3. `vllm/model_executor/layers/sparse_attn_indexer.py`
   - index scratch 改为按 chunk 最大值分配（原来按整个 batch 的 `total_seq_lens`），保证头部 scratch 不会碰到尾部保留区。
4. `tests/v1/worker/test_workspace.py`
   - 新增 retained span 的单测：小 scratch 不破坏、四类扰动（overlap/other key/other size/realloc）报告失效、per-slot 隔离。

来源：移植 upstream #59131（SM100 mega-attn 路径）到 SM90 实际使用的 FlashMLA 路径；语义与其 `get_retained` 一致。

正确性说明：
- 每个 KV source 层在同一 forward 内总是先于其消费者执行且强制 gather，因此消费者复用的永远是本步刚写入的数据，跨 step 不会复用旧数据。
- 多 chunk（>4 个请求）批次时，单一 retained slot 会在 chunk 间失效，消费者自动回退到重新 gather——正确性不受影响，只是收益只落在最后一个 chunk（与 upstream 行为一致）。

## 4. 结果

### 4.1 Prefill TTFT（秒，median）

| ISL | pre ON | post ON | ON 变化 | pre OFF | post OFF | gap pre | gap post | gap 变化 |
|---|---|---|---|---|---|---|---|---|
| 131,072 | 16.73 | **10.52** | −37.1% | 9.87 | 9.84 | 6.86 | **0.68** | −90.1% |
| 262,144 | 47.76 | **23.39** | −51.0% | 20.98 | 20.89 | 26.78 | **2.50** | −90.7% |
| 524,288 | 152.7 | **55.96** | −63.4% | 47.01 | 46.50 | 105.7 | **9.46** | −91.0% |
| 1,048,000 | 536.9 | **153.45** | −71.4% | 118.49 | 116.73 | 418.4 | **36.72** | −91.2% |

### 4.2 1M 共享前缀 decode（prefix cache，4096 output）

| point | ITL med/p99 pre | ITL med/p99 post | out tok/s pre | post | TTFT pre | post |
|---|---|---|---|---|---|---|
| c1 | 89.34 / 91.58 ms | **89.15 / 91.18** | 10.94 | 11.06 | 7.86 s | **4.90 s** |
| c4 | 89.29 / 91.65 ms | **89.01 / 91.76** | 42.50 | 42.92 | 13.57 s | **10.61 s** |

### 4.3 精度

| 测试 | pre-patch | post-patch |
|---|---|---|
| GSM8K 全量 1319（ON） | 90.98% | **91.51%** |
| GSM8K 全量 1319（OFF） | 91.13% | **91.13%** |
| GSM8K-200 普通（ON） | 89.0% | 88.0% |
| GSM8K-200 salted shared-prefix | 91.5% | 90.0% |

salted 200 的 −1.5pp 在 200 样本噪声内（σ≈2.1pp）；若 replay bug 复现会坍塌到 ~82%，未出现。

### 4.4 容量
- 47,870,492 tokens / 1M = **45.65×**，与优化前一致（workspace 预留不变）。

## 5. 残留 gap 与下一步

- 1M 残余 gap 36.7s：每步仍有 4 次 host gather（C1 source 20 ×1 + C2 sources 2/8/14 ×3），仍是 O(N²)。
- 下一步候选（按收益/风险）：
  1. **append-only 持久 gather**：prefill 各 step 复用上一步的 gathered 结果，只 gather 新增压缩行 → host 读从 O(N²) 降到 O(N)，预计 1M 残余 gap 从 36.7s 降到个位数秒。
  2. 对 resident 行从 GPU resident mirror 读取，未 resident 行回退 host（需要 per-page 源选择，改动较大）。
  3. 当前结果已可作为第一阶段交付；PR 前需 rebase main、整理提交历史、补 evals/文档。

## 6. 复现

- 服务与脚本：
  - ON 启动：`/root/run_dsv41_hisparse_bench_128.sh`
  - OFF 启动：`EXTRA_ARGS='--no-enable-prefix-caching --scheduler-reserve-full-isl --kernel-config {"linear_backend":"marlin","moe_backend":"marlin","enable_jit_warmup":false}' bash /root/run_dsv41_baseline.sh`
  - prefill A/B：`bash /root/ab_gap_prefill.sh <tag>` → `/root/bench_results/<tag>/`
  - 1M 前缀 decode：`bash /root/ab_gap_prefix.sh <tag>`
  - 全量精度：`/root/miniconda3/envs/vllm_dev/bin/python -I /root/dsv41_gsm8k.py --limit 1319 --concurrency 16 --tag <tag>`
- 本次 A/B 日志：`/root/ab_on_patch.log`、`/root/ab_off_patch.log`、`/root/ab_prefix_patch.log`、`/root/ab_off_gsm.log`
- 本次结果目录：`/root/bench_results/on_patch`、`/root/bench_results/off_patch`
- profiler trace：`/root/torch_profiles/on`、`/root/torch_profiles/off`（含 `.profiler_out.txt` kernel 表）
- 补丁 commit：`2877d8ea5d` `[Perf][DSv4.1] Reuse the gathered compressed KV across consumers in prefill`
- 单元测试：`tests/v1/worker/test_workspace.py`（环境无 pytest，用等价 driver `/root/test_retained_driver.py` 跑通全部断言）

## 7. 数据口径与未测项（2026-11-04 整理）

- 优化前 = 2026-11-03，commit `098251c782`（tree 与 `0b33846e15` 一致）；优化后 = 2026-11-04，commit `2877d8ea5d`。
- 优化后重新实测：prefill c1 四个点、decode 1M 共享前缀 c1/c4、GSM8K 全量 1319（ON 和 OFF）、salted GSM8K-200。
- 未重测（表中沿用优化前，已标注）：decode c8/c16/c32（补丁只改 prefill）；优化后 OFF decode（同一原因）。
- 未测：优化后 cold prefill 高并发（c8/16/32，ISL 131k）；decode 成对 profile。
- 高并发运行项（prefill c8/16/32、decode c8/16/32）在补丁后没有重跑，任何引用 c8+ decode 的表都必须标注为 pre-patch 数据。

## 8. 后续计划（按顺序）

1. **Decode 成对 profile（先做，不改代码）**：1M 共享前缀，c1 和 c32，ON/OFF 各截 ~512 个 decode step，拆 resolver/hot-resident 索引间接、host-miss copy、kernel/stride 各占多少。
2. **Decode 优化（视 profile 结果）**：
   - 若间接层占主导：做“全 resident 快路径”（top-k 行全在 resident 时跳过 hot/resident 解析，直接读 resident block table）；decode 路径有 prefix-cache NaN 前科，必须用 salted GSM8K 守。
   - 若 miss copy 占主导：试 `device_buffer_size` 4096→8192（约 +150 MB/rank，容量影响 <0.5%），看 miss 率和 c32 p99。
   - 预期收益：median ITL 的 4–7%（约 3–6 ms/token），高并发 p99 已经 ON 更好，按 ROI 决定是否动 kernel。
3. **Bounded append-only prefill（可选，收益最大的一截）**：
   - 复用现有 4 行 prefill workspace（~4.5 GiB/rank，已计入预算，**零额外显存**）。
   - key = (source prefix, 有序 request ids, 每个 request 已有效前缀长度)；只 append 新算出的压缩行；需要 kernel 支持 `dst_col_offset` / `src_row_start`（或 append variant）。
   - 只在一个 chunk（≤4 个并发长请求，c1–c4）时生效；多 chunk 自动回退全量 gather，容量/正确性不变。
   - 目标：1M 残余 gap 36.7s → ~1–2s。**不要**给 max_num_seqs 分配 per-request 持久 buffer：32 行 ≈ 36 GiB/rank，会把容量从 45.65× 打到 ~8×。
4. **改动后重测矩阵**：prefill c1 131k/262k/524k/1M；cold prefill c8/16/32 @131k；decode 1M 前缀 c1/4/8/16/32；容量；全量 GSM8K 1319（ON+OFF）；salted 200。
5. **PR 准备（你来做主）**：rebase 最新 main（3 个 prep commit 干净；C1 host-back commit 在 `vllm/v1/hisparse/runtime.py` 和 `csrc/` 与 upstream #60103/#61680/#60618 冲突），整理历史（去掉 reject→allow 对），补 docs/evals，写 PR 描述；不做任何自主 PR 动作。


## 9. Rebase 到最新 main 后的回归（2026-11-04）

- Base：upstream main `8a702b18a1`（原 base `fa3b9d1377`）。
- 分支：`feat/dsv41-hisparse-gap` @ `7c9a158471`（已推送）；rebase 前的头 `2877d8ea5d` 保留在 `backup/gap-prerebase-20261104`。
- 编译：CMake 增量构建（`cmake-build-release`），`cmake --build --preset release --target install`，BUILD_EXIT=0。
- Rebase 适配（已折进 C1 提交 `60780efd70`）：
  1. `hisparse_resolve_residency` 签名合并：upstream 新增 `num_valid_rows`（CUDA-graph padding 行），我们保留 `row_value_bytes`（host-backed 分块行宽），两个参数都在。
  2. `HiSparseRuntime.invalidate_written_slots` 薄包装：upstream #60103 用 `HiSparseSlotInvalidator` 取代旧方法；包装内部用新 invalidator，保住 compressor 直写的 hot 副本失效。
- 环境注意：本机 Anaconda Python 3.12.0 没有 `os.memfd_create`，upstream `SharedOffloadRegion` 多 segment 路径会失败；已在 venv 的 `site-packages/sitecustomize.py` 加 libc shim（不属于仓库代码，upstream CI 正常 Python 不需要）。

### 9.1 容量
- ON：47,868,379 tokens = **45.65×**（不变）。
- OFF：23,074,544 = **22.01×**（旧 base 22.31×，upstream 内存变化导致的 base drift）。

### 9.2 Prefill TTFT（c1，同协议）

| ISL | ON | OFF | gap | 旧 base gap |
|---|---|---|---|---|
| 131k | 10.52s | 9.82s | **0.70s** | 0.68s |
| 262k | 23.37s | 20.93s | **2.44s** | 2.50s |
| 524k | 55.93s | 46.51s | **9.42s** | 9.46s |
| 1M | 153.21s | 116.78s | **36.43s** | 36.72s |

rebase 后 ON 与 gap 均与旧 base 逐位持平（≤0.2%）。

### 9.3 GSM8K

200 题（8-shot，c16）：

| 配置 | rebase base | 旧 base |
|---|---|---|
| ON（无 prefix cache） | 90.0% | 88.0% |
| OFF（无 prefix cache） | 89.0% | 90.5% |
| ON + prefix cache + salted | 89.0% | 90.0% |

全量 1319：

| base | ON | OFF |
|---|---|---|
| 旧 base | 91.51%（perf 后）/ 90.98%（perf 前） | 91.13% |
| rebase base | **90.90%** | **90.45%** |

同 base ON−OFF = +0.45pp（1319 样本 σ≈0.8pp，持平）。

### 9.4 1M 共享前缀 decode（rebase base）

| 点 | ITL med/p99 | 旧 base | TTFT | output tok/s |
|---|---|---|---|---|
| c1 | 89.79 / 92.07 ms | 89.15 / 91.18 | 4.89s | 10.98 |
| c4 | 89.86 / 92.50 ms | 89.01 / 91.76 | 10.26s | 42.66 |

decode 差异 ≤1%（base drift），无实质回归。

### 9.5 结论
- rebase 后性能、gap、容量、精度全部保持；
- 两个适配已进 C1 提交，历史可直接用于 PR；
- 下一步仅剩 PR 准备（`pr-checklist`、描述、评测摘要），PR 动作由你执行。

### 9.6 Prefix-cache 全量（补充，2026-11-04）

- rebase base，salted 共享前缀 + prefix caching，全量 GSM8K 1319 = **91.28%**（777s）。
- 对照：普通 ON 全量 90.90%，OFF 全量 90.45%；三者差异 ≤0.8pp（1319 样本 σ），在噪声内。
- 结论：SWA bounded replay + resident mirror 的 prefix-cache 路径在全量上没有退化，未出现历史 bug 的坍塌（~82%/1%）。

## 10. 高并发性能与 decode p99 归因（2026-11-04，rebase base `7c9a158471`）

### 10.1 Prefill（cold 131k，OSL 1，单波）

| 并发 | ON median/p99 TTFT | OFF median/p99 TTFT | gap |
|---|---|---|---|
| c8 | 45.68s / 80.06s | 42.58s / 74.63s | +7.3% / +7.3% |
| c16 | 85.69s / 159.22s | 79.97s / 148.57s | +7.2% / +7.2% |
| c32 | 165.91s / 317.92s | 154.73s / 296.39s | +7.2% / +7.3% |

4k/128 c32 吞吐：ON 186.91 output tok/s，OFF 195.05（−4.2%）。高并发 cold prefill 的 ON/OFF gap 稳定在 +7.2%，没有随并发放大。

### 10.2 Decode（1M 共享前缀 + 100 new + 4096 output）

| 并发 | ON ITL med/p99 | OFF ITL med/p99 | 旧 base ON/OFF p99 |
|---|---|---|---|
| c8 | 89.81 / 93.35 | 86.96 / 90.44 | 92.42 / 88.61 |
| c16 | 89.93 / 113.90 | 86.45 / 100.17 | 94.43 / 99.64 |
| c32 | 90.81 / 213.26 | 87.11 / 136.12 | 100.58 / 137.34 |

输出 tok/s：ON c8/c16/c32 = 81.75 / 149.75 / 259.92；OFF = 85.03 / 159.27 / 279.01。
ON hot-buffer miss：c8 1.14%、c16 1.80%、c32 3.02%；host→device 0.25 / 0.68 / 2.29GB（旧 base 约 0.66/0.97/1.14%，0.08/0.18/0.97GB）。

### 10.3 p99 归因（排除法）

| 假设 | 实验 | 结果 | 结论 |
|---|---|---|---|
| ON GPU pool 变小 | 启动 HMA 行 | GPU 1,268,500 blocks vs 旧 1,268,556（−0.004%），host 1,398,101 不变，45.65× | 排除 |
| base-wide drift | OFF c16/c32 p99 | 100.17 / 136.12 vs 旧 99.64 / 137.34 | 排除；是 ON 专属 |
| eager_host_mirror（upstream 新默认 True） | 设 `eager_host_mirror=False` | c32 91.64 / 213.06ms，miss 2.82%，host 2.14GB | 排除 |
| hot buffer 太小 | `device_buffer_size=8192` | c32 90.41 / 200.70ms（p99 −6%），显存影响 ~0.004% | 不是根因，可作缓解 knob |
| 我们的 invalidator 包装 | 临时关闭 compressor invalidate | c32 91.16 / 244.24ms，miss 3.08%，host 2.36GB | 排除 |
| invalidate kernel 语义变化 | 旧/新 kernel 对比 | 都是把 state row 中匹配 written_slot 的 entry 置 -1；新版仅 int4 向量化/多 group | 排除 |

结论：ON 在新 base 上从 c8 起 hot miss 率翻倍（0.66→1.14% 起），到 c32 达 3.0%，大量 host→device DMA 插入 decode 流，导致 c16/c32 的 p99 从旧 base 的 94/101ms 升到 114/213ms。根因是 **upstream #60103 重写的 resolver/all-resident 路径的 ON 专属行为**，与 C1/C2 host-back、prefill gather 补丁、GPU pool、eager_host_mirror、invalidator 包装均无关；median ITL 不受影响（~90ms）。

缓解与后续：
- 部署侧若在意高并发 c32 的 p99，可把 `device_buffer_size` 从 4096 提到 8192（p99 −6%，显存影响 ~0.004%），但无法完全消除。
- 彻底修复需要回看 upstream #60103 的 resolver/hot 策略（为何同 GPU pool/hot 容量下 miss 率翻倍），不在本 PR 范围内。

### 10.4 A/B 更正（2026-11-05）：miss 率差异不是 rebase/#60103 引入

用 pre-rebase 分支 `2877d8ea5d`（旧 base `fa3b9d1377`）重编后，跑与 10.2 完全相同的 1M 前缀 c32：

| 构建 | c32 miss | host→device | ITL med/p99 |
|---|---|---|---|
| pre-rebase 旧构建（今天） | 2.94% | 2.38GB | 90.77 / 207.19ms |
| rebase 新构建 | 3.02% | 2.29GB | 90.81 / 213.26ms |
| 旧 base 历史记录（11-03） | 1.14% | 0.97GB | 92.00 / 100.58ms |

结论：旧构建今天也复现 ~2.9%/207ms，与新构建一致；11-03 的 1.14%/100.58ms 是当时的测量条件差异（热缓冲状态/统计窗口/前序点），**不是 rebase 或 #60103 的代码回归**。10.3 中"upstream #60103 导致"的归因已被这个 A/B 推翻；高并发 decode 的 p99 ~210ms 是当前 workload/构建的真实表现，与我们的 feature 和 rebase 无关。恢复后分支为 `feat/dsv41-hisparse-gap` @`7c9a158471`，`.so` 已还原为验收时备份的版本。

## 11. device_buffer_size 扫描（2026-11-05）

同一协议（1M 前缀 prewarm + c32 4096 output，seed 12345，rebase build `7c9a158471`）：

| device_buffer_size | ITL med/p99 | miss | host→device | out tok/s | HMA shared blocks |
|---|---|---|---|---|---|
| 4096（当前） | 90.96 / 214.90ms | 2.92% | 2.39GB | 268.57 | 1,268,500 |
| 8192 | 90.78 / 234.34ms | 3.16% | 2.54GB | 260.11 | 1,268,444 |
| 16384 | 91.07 / 260.44ms | 2.82% | 2.21GB | 266.61 | 1,268,219 |
| 32768（int16 上限） | 90.89 / 212.78ms | 3.14% | 2.51GB | 260.05 | 1,267,883 |

结论：median 不变；p99/miss/host 无单调趋势，波动属 run-to-run noise（±10–20%）；更大的 hot buffer 只让 shared pool 少 0.05%（−617 blocks），没有性能收益。**保持 `device_buffer_size=4096`，不建议调大。** 1M 前缀 c32 的 p99 ~210–260ms 是 host-miss workload 的固有波动，这个参数解决不了。
