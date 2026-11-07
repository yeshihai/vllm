# DeepSeek-V4.1 HiSparse PD feasibility

Date: 2026-11-07
Code: `opt/review-fixes` @ `b5e04dde53`
Test box: 8x H20, TP4 prefill on GPUs 0-3, TP4 decode on GPUs 4-7
Controller: `toy_proxy_server.py` on `127.0.0.1:8192`

## What works

The dependency-free baseline works:

- Both sides use `NixlConnector` (no HiSparse), TP4, `max_num_batched_tokens`
  8192 prefill / 4096 decode, `BLHNC`, eager mode.
- Prefill on 8100, decode on 8200, proxy 8192.
- 131k cold c1 through the proxy: **TTFT 13.03s**, 8 output tokens, no error.
- A 5-token request also returns normally.

This validates the TP4 split, NIXL installation (`nixl==1.5.2` from the internal
mirror), side-channel ports, and proxy routing.

## HiSparse blockers

### 1. Host-resident source caches fail NIXL registration

With `MultiConnector` = `{NixlConnector, HiSparseConnector}` on both sides,
startup fails in `NixlBaseConnectorWorker.register_kv_caches`:

```text
AssertionError: KV cache view cannot be partitioned into NIXL regions:
layer=language_model.model.layers.2.attn,
cache_nbytes=26127711488, num_blocks=1398101,
block_stride=19008, physical_page_size=19008,
cache_shape=(1398101, 1, 32, 584),
cache_stride=(19008, 18688, 584, 1)
```

The host source cache is a strided view over the shared pinned pool: the
logical view is payload-sized while the block stride includes page padding.

A test patch that registered host-resident groups as one block-strided span
(`registration_len = num_blocks * block_stride`, host region specs) lets both
sides start. It is saved on the test box at
`backups/pd_nixl_20261107/host_registration.patch`; it is not committed because
it does not solve the transfer-mode issue below.

### 2. Symmetric HiSparse: NIXL pull rejects mixed-memory producers

With the registration patch, the first request fails during handshake:

```text
NotImplementedError: NIXL pull does not yet support a mixed-memory producer
NIXL transfer failure: handshake_failed ...
remote_engine_id=..., num_remote_blocks=45
```

The prefill producer has both DRAM (host-resident source) and VRAM (device
groups). `NixlPushConnector` is not an alternative: it explicitly rejects
mixed-memory KV caches. So symmetric PD + HiSparse is unsupported by the
current NIXL connector in both transfer modes.

### 3. Asymmetric P OFF / D ON: cache-group layouts must match

If prefill is GPU-only and decode is HiSparse, prefill starts and decode
starts, but the first request fails on the decode side:

```text
RuntimeError: Different NIXL cache-group layouts need the same KV regions on
both sides
```

Turning HiSparse off on only one side changes the derived resident/hot group
layout, so NIXL cannot map the two sides.

## Conclusion

**PD without HiSparse is currently supported; PD with HiSparse is not a
configuration-only change.** The current NIXL connector cannot move the
HiSparse host-resident source cache together with the device caches.

To support PD + HiSparse, one of these is required:

1. NIXL mixed-memory producer support for pull, and mixed-memory cache support
   for push (upstream NIXL/vLLM work).
2. A HiSparse-specific P/D path: transfer host-resident source KV over a
   separate host-to-host channel, while NIXL handles only device groups.
3. A layout design where both sides expose identical transfer groups, with
   host-backed source rows added after the transfer on the decode side.

Until then, the practical options are:

- PD for TTFT/ITL isolation **without** the HiSparse capacity benefit.
- Single-instance HiSparse at 4096 for the 2.04x capacity benefit.

The NIXL host-registration patch is a useful prerequisite for option 1/2, but
it is not sufficient and should not be merged as-is.

## Repro

- `harness/start_pd.sh <prefill|decode>`: HiSparse + MultiConnector side.
- `harness/start_pd_off.sh <prefill|decode>`: NIXL-only baseline.
- `harness/stage_b.sh` and `README.md`: single-instance reference.
- Logs/results on the test box:
  `logs/pd_{off,on,asym}_*`, `results/pd_off_smoke`.
