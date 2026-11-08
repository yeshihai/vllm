# DeepSeek-V4.1 HiSparse Stage A/B evaluation

Date: 2026-11-07
Code: `opt/review-fixes` @ `2025870b59`
Hardware: 8x H20 (SM90), TP8/EP8, 1M context, `fp8_ds_mla`, `BLHNC`
Purpose: paired HiSparse ON/OFF reference for the prefill gather-worker and
`max_num_batched_tokens` work.

Raw results and logs are on the test box under
`/dockerdata/yeshihai_dsv41_buffer/{results,logs}/stage_{a,b}_20261107_r1`.
They are intentionally not committed; the harness scripts in `harness/` assume
the same test-box paths.

## Protocol

- Same checkout for ON and OFF; the only launch difference is the HiSparse
  `attention_config` and `kv_transfer_config`. Both use prefix caching,
  `scheduler_reserve_full_isl`, `max_num_seqs=32`, `gpu_memory_utilization=0.92`.
- ON uses `host_pool_gib=128`, `device_buffer_size=4096`.
- Each mode/config is a fresh server. Cold-prefill points use exact random ISL
  and seed; decode points prewarm one 1M-token shared prefix before the
  concurrency sweep.
- ON snapshots `hisparse_cache_hits_total`, `misses_total`, and
  `host_to_device_bytes_total` before and after every point. All prefill points
  report zero misses and zero host-to-device bytes.

## Stage A: `max_num_batched_tokens` knee

Cold 1M c1 TTFT and the generic startup capacity bound:

| MBT | ON 1M TTFT | OFF 1M TTFT | ON capacity | OFF capacity | ON/OFF capacity |
|---:|---:|---:|---:|---:|---:|
| 4096 | 128.01 s | 116.68 s | 44.34x | 21.74x | 2.04x |
| 8192 | 112.37 s | 106.94 s | 30.82x | 19.17x | 1.61x |
| 16384 | 105.01 s | 102.29 s | 18.28x | 15.18x | 1.20x |
| 32768 | 102.37 s | 99.80 s | 8.84x | 9.91x | 0.89x |

131k cold prefill showed the same direction:

| MBT | ON c16 | OFF c16 | ON c32 | OFF c32 |
|---:|---:|---:|---:|---:|
| 4096 | 81.56 s | 79.79 s | 157.80 s | 154.47 s |
| 8192 | 74.05 s | 73.03 s | 143.28 s | 141.40 s |
| 16384 | 70.12 s | 69.61 s | 135.71 s | 134.54 s |
| 32768 | 68.09 s | 67.69 s | 131.58 s | 130.97 s |

**4096 is the capacity-preserving operating point.** 8192 gains ~12% TTFT at
1M but loses ~30% ON capacity and falls below 32x; 32768 loses ~80% capacity
and ON capacity is below OFF. 4096 is the only point that supports 32
concurrent distinct 1M requests.

### Why the capacity ratio falls

The capacity calculation is not a fixed shared workspace. It charges each
request its admission footprint, including the in-flight window of
recycling-aware groups:

- `VllmConfig.max_in_flight_tokens = max_concurrent_batches * max_num_batched_tokens`
  (`vllm/config/vllm.py:629`).
- `HiSparseResidentSpec.max_admission_blocks_per_request()` and the
  sliding-window/chunked-local specs size per-request pages from
  `max_in_flight_tokens` (`vllm/v1/kv_cache_interface.py:529`).
- `get_max_concurrency_for_kv_cache_config()` divides the pool by the summed
  per-request blocks (`vllm/v1/core/kv_cache_utils.py:1081`).

Raising MBT therefore raises ON's per-request resident/SWA reservation. OFF
has no HiSparse resident spec and loses much less; the ON/OFF ratio falls.
The observed GPU block count at 8192 is only 3.5% lower than at 4096, while
capacity is 30% lower, confirming the denominator is the dominant term.

## Stage B: full ON/OFF matrix at 4096

Prefill:

| Workload | ON | OFF | ON/OFF |
|---|---:|---:|---:|
| 131k c1 TTFT | 10.114 s | 9.866 s | +2.51% |
| 262k c1 TTFT | 21.586 s | 20.777 s | +3.89% |
| 524k c1 TTFT | 49.301 s | 46.430 s | +6.18% |
| 1M c1 TTFT | 127.600 s | 116.451 s | +9.57% |
| 131k c8 TTFT | 43.389 s | 42.426 s | +2.27% |
| 131k c16 TTFT | 81.554 s | 79.766 s | +2.24% |
| 131k c32 TTFT | 157.829 s | 154.410 s | +2.21% |
| 4k/128 c32 throughput | 338.8 tok/s | 349.6 tok/s | -3.07% |
| Capacity | 44.34x | 21.74x | 2.04x |

Decode, 1M shared prefix, 256 output tokens:

| Conc | ON ITL | OFF ITL | ON/OFF | ON miss | ON H2D |
|---:|---:|---:|---:|---:|---:|
| 1 | 8.606 ms | 8.012 ms | +7.41% | 0 | 0 |
| 4 | 8.605 ms | 8.001 ms | +7.55% | 3.98% | 33.7 MB |
| 8 | 8.603 ms | 8.002 ms | +7.50% | 4.25% | 70.9 MB |
| 16 | 8.603 ms | 8.004 ms | +7.48% | 4.70% | 123.3 MB |
| 32 | 8.602 ms | 8.004 ms | +7.47% | 4.82% | 243.8 MB |

ITL p99: ON/OFF c1 9.00/8.35 ms, c8 9.29/9.47 ms (-1.9%), c16 10.31/8.39 ms
(+22.9%), c32 11.31/9.55 ms (+18.4%). Decode c1 has zero host misses, so its
+7.4% is clean ON-path overhead; c16/c32 p99 must be read with the miss rate.

Accuracy:

- GSM8K-1319 8-shot: ON **91.43%**, OFF **91.05%** (paired McNemar p=0.635).

## Conclusions

- Keep 4096 as the single-instance protocol: it preserves the 2.04x capacity
  advantage and the 44.34x capacity bound.
- The mxfp8 gather-worker optimization is independent of this: at 4096 the ON
  1M cold prefill is 127.60 s vs the pre-optimization 153.78 s.
- Do not adopt 8192+/32K for a single instance without first addressing the
  per-request admission scaling. A code change that decouples resident/SWA
  admission from MBT is required to get the TTFT gain without the capacity
  cliff.
- For PD, a decode-only instance can use 4096 (or `max_num_seqs *
  max_decode_query_len` rounded up) and retain the 2.04x capacity advantage.
  A prefill-only instance can use 8192/32K if its concurrency fits the lower
  capacity. `HiSparseConnector` is local host-cache offload
  (`requires_kv_delivery=False`); P/D needs `MultiConnector` with a P/D
  connector such as NixlConnector. That path still needs startup, transfer
  ordering, accuracy, and PD-mixed validation.

## Harness

- `harness/start_on_mbt.sh <MBT>` / `harness/start_off_mbt.sh <MBT>`:
  parameterized launch scripts.
- `harness/stage_a.sh`: 4 MBT values x ON/OFF x 3 representative points.
- `harness/stage_b.sh`: full ON/OFF matrix at 4096, including decode and
  GSM8K-200.
- `harness/accuracy_full.sh`: GSM8K-1319 ON/OFF at 4096.
- `harness/summarize_stage_a.py`, `harness/summarize_stage_b.py`,
  `harness/delta_stage_b.py`: result aggregation.
- `harness/start_pd.sh` / `harness/start_pd_off.sh`: TP4 P/D launch scripts;
  see `PD_FEASIBILITY.md` for the current HiSparse + NIXL blockers.
- `harness/bench_pd_dsv41.sh`: cold-prompt P/D serving matrix (TTFT, ITL,
  throughput) with HiSparse counter snapshots; `analyze_pd.py` /
  `compare_pd.py` aggregate the results. See `PD_PERF_BASELINE.md` for the
  ON/OFF cold baseline and the `--num-warmups` caveat.
