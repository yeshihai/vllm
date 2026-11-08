# DSv4.1 P/D 冷启动性能基线（ON vs OFF）

日期：2026-11-08；机器：8x H20 96GB（P：GPU 0-3，D：GPU 4-7，各 TP4+EP）。
代码：`feat/dsv41-hisparse-pd`（含 position-paired NIXL pull）。

## 配置

| | Prefill (P) | Decode (D) |
|---|---|---|
| 端口 | 8100 | 8200 |
| KV 传输 | NixlConnector | ON: `MultiConnector[NixlConnector, HiSparseConnector]`；OFF: `NixlConnector` |
| HiSparse | 关闭 | ON: `device_buffer_size=4096`，`host_pool_gib=128` |
| MBT | 8192 | 4096 |
| prefix caching | on | on |

请求经 `tests/v1/kv_connector/nixl_integration/toy_proxy_server.py`（:8192）
先发 P 再发 D，TTFT 含 P 的完整 prefill、KV 拉取和 D 首 token。

## 测量方法（重要）

`vllm bench serve --num-warmups 1` 会用**同一条 prompt** 预热；在 prefix
caching 开启时，被测量的请求命中 P/D 的 prefix cache，TTFT 反映的是
cache-hit 而不是冷 prefill。本次早期矩阵因此失真（例如 OFF 8k TTFT
260 ms，冷启动为 957 ms；OFF 131k longdecode TTFT 1.8 s，冷启动 13.6 s）。

正确做法（`harness/bench_pd_dsv41.sh`）：

- 先用 seed 9999 单独预热 kernel（不计入结果）；
- 所有测量点 `--num-warmups 0`；
- ON/OFF 使用不同的 `SEED_BASE`（10000/20000），保证两次运行的随机
  prompt 不同，P 的 prefix cache 不会跨配置命中。
- 每次测量都必须用**全新的 SEED_BASE**：失败/半途而废的 run 仍然会把
  prefill 请求发给 P，P 的 prefix cache 会被预热；用同一个 SEED_BASE
  重跑会全部变成 cache hit。本次 MBT 实验就踩到了这个坑。

## MBT 实验（P 侧）

P 的 prefill 时间决定 TTFT，尝试把 P 的 `--max-num-batched-tokens`
从 8192 提高：

| P MBT | 结果 |
|---:|---|
| 32768 | 启动失败：KV cache 只剩 0.08 GiB，估算最大模型长度 3232（激活显存挤占 KV） |
| 16384 | 可启动。冷 64k TTFT 6.52–6.54 s（新 seed）vs 8192 的 6.77 s，差 ~3.7%，在波动内 |
| 8192 | 保持 |

注意：MBT16k 首次测量出现 3–7x 的“加速”，是因为之前一次失败的 run
已经把同一批 prompt 的 KV 留在了 P 的 prefix cache 中；更换新 seed 后
加速消失。结论：在当前 0.92 显存预算下，提高 P 的 MBT 不是有效的 TTFT
杠杆；P 的冷 prefill 速率约 9–10k tok/s，要压 TTFT 需要算法/并行度层面
的改动（chunk 调度、gather workers 或更多 prefill GPU）。

## 结果（冷启动，随机 prompt，temperature 0）

`SEED_BASE=10000` ON / `20000` OFF；TTFT/ITL 单位 ms，p50。

| point | OFF TTFT | ON TTFT | ΔTTFT | OFF ITL | ON ITL | ΔITL | ON miss% | H2D GB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| prefill 8k | 957.3 | 961.8 | +0.5% | 82.42 | 85.73 | +4.0% | 0 | 0 |
| prefill 32k | 3423.5 | 3466.9 | +1.3% | 83.38 | 87.11 | +4.5% | 0 | 0 |
| prefill 64k | 6767.1 | 6856.1 | +1.3% | 82.45 | 85.80 | +4.1% | 0 | 0 |
| prefill 128k | 13755.4 | 13831.2 | +0.6% | 82.81 | 86.87 | +4.9% | 0 | 0 |
| prefill 256k | 28365.8 | 28688.8 | +1.1% | 82.84 | 85.75 | +3.5% | 25.7 | 0.09 |
| longdecode 128k c1 | 13621.6 | 13813.5 | +1.4% | 83.07 | 87.22 | +5.0% | 22.5 | 0.14 |
| longdecode 128k c2 | 17818.8 | 18233.3 | +2.3% | 83.26 | 87.26 | +4.8% | 18.5 | 0.52 |
| longdecode 128k c4 | 35240.3 | 36142.6 | +2.6% | 82.83 | 87.84 | +6.1% | 19.7 | 0.55 |
| throughput 4k c32 | 1861.4 | 1971.5 | +5.9% | 83.97 | 88.20 | +5.0% | 0 | 0 |
| throughput 输出 | 230.4 tok/s | 223.3 tok/s | −3.1% | | | | | |

注：prefill 点的 ITL 是 prefill 之后 8 个输出 token 的 ITL；longdecode
为 131k 输入、64 输出、4 条请求（conc 1/2/4）；throughput 为 4k 输入、
128 输出、conc 32、64 条请求。

## 观察

1. **TTFT 由 P 的 prefill 主导**：冷 prefill 约 9.2k tok/s（256k ≈ 28.4s）。
   ON 的 TTFT 增量 ≤1.3%（≤256k prefill）和 1.4–2.6%（128k longdecode），
   主要来自 D 侧拉取和 host 写入，量级远小于 prefill 本身。
2. **decode ITL 有稳定的 ~4% ON 固定开销**：短 decode 且 miss=0 的点
   （8k/32k/64k/128k prefill）ON 仍高 3.5–4.9%，说明即使没有 host 读，
   HiSparse 的 hot-cache/簿记路径也有固定成本。**注意：这里的 D 是 eager**
   （`--enforce-eager`），eager 的 ITL 约 90% 是 host 启动开销；开 CUDA
   graphs 后 ITL 从 ~87ms 降到 ~9ms，ON/OFF 差缩到 ~0.6ms（见“图模式”）。
3. **长上下文 miss 再叠加 ~1–2%**：128k longdecode 的 miss rate
   18.5–22.5%，ITL 额外增加；H2D 流量每个点 0.14–0.55GB。
4. **吞吐**：conc 32 时 ON −3.1% 输出 token 吞吐，与 ITL 比例一致。
5. 256k prefill 点 miss 25.7% 但 H2D 仅 0.09GB：该点的 host 读主要发生在
   后续 8 个 decode token，样本少，miss 率的绝对值不宜过度解读。

## 图模式 decode（2026-11-09）

eager 的 ITL 不是 GPU 计算下限：torch profiler 下 OFF eager 每步 GPU
kernel 时间总和仅 ~8.8ms，而 ITL 为 83ms；ON 的 ~4% 差也以 host 侧开销
为主。对 D 去掉 `--enforce-eager`（`FULL_AND_PIECEWISE`，decode batch
1–32；图捕获 0.62 GiB）后：

| 点位 | OFF eager | ON eager | OFF 图 | ON 图 | ON 图 / OFF 图 |
|---|---:|---:|---:|---:|---:|
| 128k c1 ITL (ms) | 83.07 | 87.22 | 8.71 | 9.30 | +6.8% |
| 128k c4 ITL (ms) | 82.83 | 87.84 | 8.71 | 9.31 | +6.9% |
| 4k/128 c32 ITL (ms) | 83.97 | 88.20 | 10.18 | 10.80 | +6.1% |
| 4k/128 c32 吞吐 (tok/s) | 230.4 | 223.3 | 331.9 | 327.3 | −1.4% |
| 128k c1 TTFT (ms) | 13621.6 | 13813.5 | 13892.9 | 13776.7 | P 侧 prefill，不变 |

- 正确性：ON 图模式 GSM8K-200 **0.890**（eager ON 0.895/0.905），全量
  GSM8K-1319 **0.9121 / 152.2s**（eager ON 0.9098/763.7s，eager OFF
  0.9113/713.1s）。精度在噪声内，端到端约 5×。
- 128k c4 ON 图模式期间 hot-buffer miss +4.47e6、host→device +2.61GB，
  ITL 仍为 9.31ms：host miss 路径在图模式下工作正常，且不再主导 ITL。
- 容量：GPU+host 8,621,930 tokens（8.22x；eager 9,266,616 / 8.84x），
  HiSparse steady-state 8.36x；图捕获占 0.62 GiB，容量代价约 7%。
- 复现：`EAGER=0 bash harness/start_pd_asym.sh prefill|decode`（默认图
  模式，P 保持 NIXL-only eager，D 为 NIXL+HiSparse 图模式）；
  `EAGER=1` 回到本文件上面的 eager 数字。

### profiler 归因（eager）

- HiSparse 自己的 `hisparse_resolve_residency` / `gather_compact` 合计仅
  ~0.1–0.15ms/step，不是 ~4% 的来源。
- 打开 HiSparse 会把 FlashInfer all-reduce 从 `mnnvl` 强制退回 `trtllm`
  （vllm#65201，规避 vllm#64327），用 `VLLM_FLASHINFER_ALLREDUCE_BACKEND=mnnvl`
  覆盖可回收 ~0.6–0.9ms/step；FlashInfer 修复
  flashinfer-ai/flashinfer#7782 已于 11-02 合入，但本机 0.7.2.post1 未包含，
  暂不能用该 env 长期开启（#64327 的竞态在 host spill 下仍可能触发）。
- 剩余为 host 侧：每步约 800 个额外 aten op、~80 个额外 kernel launch，
  来自每个 KV cache group 的 metadata build（`sparse_swa.build`、block
  table 切片）与 HiSparse runtime 的 view/slice；eager 下落在关键路径，
  图模式后基本消失。

## 下一步

- 以图模式为 D 的默认 baseline 重新积累数据（长上下文 c1/c4、c32 吞吐、
  容量、GSM8K）；eager 数字仅用于对照 host 开销。
- HiSparse 在图模式下的剩余 ~0.6ms/step（约 6–7%）优先看 host miss 与
  每步簿记；miss rate/H2D 必须匹配后再比较代码改动。
- 等 FlashInfer 发布包含 #7782 的版本后，移除 vllm#65201 的 HiSparse
  回退限制，可再回收 ~0.6–0.9ms/step。
- P 侧 MBT 已试：32768 起不来（KV 被激活挤到 0.08 GiB），16384 无实质
  收益，维持 8192。要压 longdecode TTFT 需要算法/并行度层面的改动，
  而不是简单调大 MBT。

## 复现

```bash
# 图模式 ON（P NIXL-only，D NIXL+HiSparse；默认 CUDA graphs）
EAGER=0 bash benchmarks/dsv41_hisparse/harness/start_pd_asym.sh prefill  # :8100
EAGER=0 bash benchmarks/dsv41_hisparse/harness/start_pd_asym.sh decode   # :8200
# eager 对照
EAGER=1 bash benchmarks/dsv41_hisparse/harness/start_pd_asym.sh decode

# ON
SEED_BASE=10000 bash benchmarks/dsv41_hisparse/harness/bench_pd_dsv41.sh pd_on_cold all
# OFF（先重启 D 为 NixlConnector-only）
SEED_BASE=20000 bash benchmarks/dsv41_hisparse/harness/bench_pd_dsv41.sh pd_off_cold all
# 汇总
python benchmarks/dsv41_hisparse/harness/compare_pd.py \
  /root/bench_results/pd_off_cold /root/bench_results/pd_on_cold OFF ON
```
