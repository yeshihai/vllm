# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ShardQ support for DeepSeek-V4.1 prefill.

ShardQ (``vllm.v1.attention.ops.shardq``) gives every TP rank a zigzag slice of
a prefill step's tokens.  V4.1's attention runs its prefill as one chunk per
request, so this module re-expresses the slice as a ``CommonAttentionMetadata``
whose "requests" are the rank's contiguous token runs: a rank owns at most two
runs per request (the early and late zigzag chunk), and each run is a chunk
whose context ends at its last token.  The existing V4.1 SWA / FlashMLA
metadata builders then build the sharded metadata unchanged.

The compact top-k layout is the one the sparse indexer already writes under
ShardQ: the step's decode tokens in place, then this rank's tokens ascending.
The runs are emitted in that same ascending token order, so the attention reads
the indexer's output without a remap.
"""

from dataclasses import dataclass, field

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backend import CommonAttentionMetadata, max_decode_query_len
from vllm.v1.attention.ops.shardq import (
    ShardQMetadata,
    ShardQSplit,
    build_shardq_split,
)

logger = init_logger(__name__)


@dataclass
class DSV41ShardQ:
    """One step's ShardQ view for V4.1.

    ``sharded_cm`` describes this rank's prefill runs as requests; the builders
    turn it into the SWA and FlashMLA metadata ``forward_mqa`` reads.  The
    exchange tensors are shared by every builder of the step; each builder
    makes its own ``ShardQMetadata`` so the per-builder sharded metadata does
    not overwrite another builder's.
    """

    split: ShardQSplit
    sharded_cm: CommonAttentionMetadata
    run_req: torch.Tensor
    token_ids: torch.Tensor
    rank_offsets: torch.Tensor
    exchange: ShardQMetadata = field(init=False)

    def __post_init__(self) -> None:
        self.exchange = ShardQMetadata(
            num_decode_tokens=self.split.num_decode_tokens,
            num_tokens_per_rank=self.split.num_tokens_per_rank.tolist(),
            token_ids=self.token_ids,
            rank_offsets=self.rank_offsets,
        )

    def new_exchange(self) -> ShardQMetadata:
        """A fresh exchange metadata, for a builder to attach its metadata to."""
        return ShardQMetadata(
            num_decode_tokens=self.exchange.num_decode_tokens,
            num_tokens_per_rank=self.exchange.num_tokens_per_rank,
            token_ids=self.exchange.token_ids,
            rank_offsets=self.exchange.rank_offsets,
        )

    def shard_replay_start(
        self, replay_start: torch.Tensor | None
    ) -> torch.Tensor | None:
        if replay_start is None:
            return None
        return replay_start.index_select(0, self.run_req)


def _split_runs(
    token_ids: np.ndarray,
    num_decode_tokens: int,
    query_start_loc: np.ndarray,
    seq_lens: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Materialize one rank's contiguous request runs from a complete split."""
    tokens = token_ids + num_decode_tokens
    req = np.searchsorted(query_start_loc[1:], tokens, side="right")
    boundary = np.ones(tokens.size, dtype=bool)
    boundary[1:] = (tokens[1:] != tokens[:-1] + 1) | (req[1:] != req[:-1])
    starts = np.flatnonzero(boundary)
    ends = np.r_[starts[1:], tokens.size]
    run_qlen = (ends - starts).astype(np.int32)
    run_req = req[starts]
    run_last = tokens[ends - 1]
    run_seq = (seq_lens[run_req] - query_start_loc[run_req + 1] + run_last + 1).astype(
        np.int32
    )
    return tokens, starts, run_qlen, run_req, run_seq


def dsv41_shardq_eligible(
    cm: CommonAttentionMetadata,
    vllm_config: "VllmConfig",  # noqa: F821
    split: ShardQSplit,
) -> bool:
    """Whether every TP rank's V4.1 runs remain prefill requests.

    The decision scans the complete split so every rank takes the same path,
    including when uneven zigzag chunks give only one rank a short run.
    """
    query_start_loc = cm.query_start_loc_cpu.numpy()
    seq_lens = cm.seq_lens_cpu_upper_bound
    assert seq_lens is not None
    seq_lens_np = seq_lens.numpy()
    decode_threshold = max_decode_query_len(vllm_config)
    rank_offsets = np.r_[0, np.cumsum(split.num_tokens_per_rank)]
    for start, end in zip(rank_offsets[:-1], rank_offsets[1:]):
        _, _, run_qlen, _, run_seq = _split_runs(
            split.token_ids[start:end],
            split.num_decode_tokens,
            query_start_loc,
            seq_lens_np,
        )
        if int(run_qlen.min()) <= decode_threshold or int(run_seq.min()) <= 0:
            return False
    return True


def build_dsv41_shardq(
    cm: CommonAttentionMetadata,
    vllm_config: "VllmConfig",  # noqa: F821
    device: torch.device,
) -> DSV41ShardQ | None:
    """This step's ShardQ, or None when the step stays head-sharded.

    The result is memoized on ``cm``: every layer group's builder asks for the
    same step's split, and the CPU scan and H2D copies are not cheap enough to
    repeat per layer.
    """
    cached = getattr(cm, "_dsv41_shardq", False)
    if cached is not False:
        return cached
    result = _build_dsv41_shardq(cm, vllm_config, device)
    cm._dsv41_shardq = result  # type: ignore[attr-defined]
    return result


def _build_dsv41_shardq(
    cm: CommonAttentionMetadata,
    vllm_config: "VllmConfig",  # noqa: F821
    device: torch.device,
) -> DSV41ShardQ | None:
    split = build_shardq_split(cm, vllm_config, eligibility=dsv41_shardq_eligible)
    if split is None:
        return None
    num_decode_tokens = split.num_decode_tokens
    query_start_loc = cm.query_start_loc_cpu.numpy()
    seq_lens = cm.seq_lens_cpu_upper_bound
    assert seq_lens is not None
    local, starts, run_qlen, run_req, run_seq = _split_runs(
        split.local_token_ids,
        num_decode_tokens,
        query_start_loc,
        seq_lens.numpy(),
    )

    # Eligibility was decided from every rank above.  From here on the local
    # token ids only materialize this rank's metadata.
    run_qsl_cpu = np.zeros(starts.size + 1, dtype=np.int32)
    np.cumsum(run_qlen, out=run_qsl_cpu[1:])
    run_req_gpu = async_tensor_h2d(run_req.astype(np.int32), device=device)
    token_gpu = async_tensor_h2d(local.astype(np.int64), device=device)

    sharded_cm = CommonAttentionMetadata(
        query_start_loc=async_tensor_h2d(run_qsl_cpu, device=device),
        query_start_loc_cpu=torch.from_numpy(run_qsl_cpu),
        seq_lens=async_tensor_h2d(run_seq, device=device),
        seq_lens_cpu_upper_bound=torch.from_numpy(run_seq),
        num_reqs=int(starts.size),
        num_actual_tokens=int(local.size),
        max_query_len=int(run_qlen.max()),
        max_seq_len=int(run_seq.max()),
        block_table_tensor=cm.block_table_tensor.index_select(0, run_req_gpu),
        slot_mapping=cm.slot_mapping.index_select(0, token_gpu),
        positions=(
            cm.positions.index_select(0, token_gpu)
            if cm.positions is not None
            else None
        ),
        is_prefilling=torch.ones(starts.size, dtype=torch.bool, device=device),
        req_idx=run_req.astype(np.int32),
    )
    logger.info_once(
        "DeepSeek-V4.1 ShardQ prefill: %d requests, %d local tokens on this rank.",
        cm.num_reqs,
        local.size,
    )
    return DSV41ShardQ(
        split=split,
        sharded_cm=sharded_cm,
        run_req=run_req_gpu,
        token_ids=async_tensor_h2d(split.token_ids.astype(np.int32), device=device),
        rank_offsets=async_tensor_h2d(
            np.r_[0, np.cumsum(split.num_tokens_per_rank)].astype(np.int32),
            device=device,
        ),
    )
