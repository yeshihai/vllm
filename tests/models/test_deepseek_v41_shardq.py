# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 ShardQ run planning (vllm/models/deepseek_v41/shardq.py)."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
from vllm.config import CacheConfig, VllmConfig, set_current_vllm_config
from vllm.models.deepseek_v41 import shardq as dsv41_shardq
from vllm.models.deepseek_v41.attention import DeepseekV4IndexerCache
from vllm.v1.attention.backends.mla.indexer import _resolve_shardq_eligibility
from vllm.v1.attention.ops import shardq

WORLD = 4
CPU = torch.device("cpu")


def _config(
    min_tokens: int = 1024, window: int = 128, topk: int = 512, num_spec: int = 0
):
    speculative_config = (
        SimpleNamespace(num_speculative_tokens=num_spec, parallel_drafting=False)
        if num_spec
        else None
    )
    return SimpleNamespace(
        speculative_config=speculative_config,
        attention_config=SimpleNamespace(sparse_mla_shard_q_min_tokens=min_tokens),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(index_topk=topk),
            hf_config=SimpleNamespace(sliding_window=window),
        ),
    )


@pytest.fixture(autouse=True)
def fake_tp(monkeypatch):
    monkeypatch.setattr(
        shardq,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=WORLD, rank_in_group=1),
    )


def _metadata(seq_lens, query_lens):
    return create_common_attn_metadata(
        BatchSpec(seq_lens=seq_lens, query_lens=query_lens),
        block_size=16,
        device=CPU,
    )


def test_runs_follow_zigzag_chunks():
    """A rank's zigzag chunks become two prefill runs ending at their tokens."""
    seq_len = 8192
    cm = _metadata([seq_len], [seq_len])
    plan = dsv41_shardq.build_dsv41_shardq(cm, _config(), CPU)
    assert plan is not None
    sharded = plan.sharded_cm

    # Eight 1024-token chunks: rank 1 owns chunks 1 ([1024, 2048)) and 6
    # ([6144, 7168)).
    assert sharded.num_reqs == 2
    assert sharded.num_actual_tokens == 2048
    assert sharded.max_query_len == 1024
    assert sharded.seq_lens.tolist() == [2048, 7168]
    assert sharded.query_start_loc.tolist() == [0, 1024, 2048]

    expected = np.r_[np.arange(1024, 2048), np.arange(6144, 7168)]
    np.testing.assert_array_equal(
        sharded.slot_mapping.numpy(), cm.slot_mapping.numpy()[expected]
    )
    # The exchange metadata carries the whole step's tokens grouped by rank,
    # and the plan is reusable by every builder of the step.
    assert plan.exchange.num_decode_tokens == 0
    assert sum(plan.exchange.num_tokens_per_rank) == seq_len
    assert plan.exchange.token_ids.numel() == seq_len
    assert dsv41_shardq.build_dsv41_shardq(cm, _config(), CPU) is plan


def test_decode_prefix_stays_head_sharded():
    """Decode tokens stay in place; only the post-decode tokens are split."""
    cm = _metadata([1, 4096], [1, 4096])
    plan = dsv41_shardq.build_dsv41_shardq(cm, _config(), CPU)
    assert plan is not None
    assert plan.exchange.num_decode_tokens == 1
    # Rank 1 owns chunks 1 and 6 of the second request's eight chunks.
    assert plan.sharded_cm.seq_lens.tolist() == [1024, 3584]
    assert plan.sharded_cm.query_start_loc.tolist() == [0, 512, 1024]


def test_window_sized_prefill_chunks_can_shard():
    """Query length alone cannot distinguish ordinary chunks from replay."""
    cm = _metadata([2048] * 8, [128] * 8)
    plan = dsv41_shardq.build_dsv41_shardq(cm, _config(), CPU)
    assert plan is not None
    assert plan.sharded_cm.num_reqs == 16
    assert plan.sharded_cm.num_actual_tokens == 256


def test_short_batch_stays_head_sharded():
    cm = _metadata([512], [512])
    assert dsv41_shardq.build_dsv41_shardq(cm, _config(), CPU) is None


def test_single_token_run_stays_head_sharded():
    """A one-token run would be reclassified as a decode by the builders."""
    # Eight tokens over eight zigzag chunks are one-token runs; a top-k of one
    # lets the split itself through so the run guard is what rejects it.
    cm = _metadata([8], [8])
    plan = dsv41_shardq.build_dsv41_shardq(cm, _config(min_tokens=8, topk=1), CPU)
    assert plan is None


def test_spec_decode_sized_run_stays_head_sharded():
    """A run no wider than a spec-decode request is read as a decode."""
    # 16 tokens over eight zigzag chunks are runs of two tokens; with one
    # speculative token anything up to two tokens is a decode.
    cm = _metadata([16], [16])
    config = _config(min_tokens=16, topk=1, num_spec=1)
    assert dsv41_shardq.build_dsv41_shardq(cm, config, CPU) is None


def test_uneven_runs_are_rejected_consistently_on_every_rank(monkeypatch):
    """One rank's wide middle run cannot hide another rank's short runs."""
    config = _config(min_tokens=16, topk=1, num_spec=2)
    results = []
    for rank in range(WORLD):
        monkeypatch.setattr(
            shardq,
            "get_tp_group",
            lambda rank=rank: SimpleNamespace(world_size=WORLD, rank_in_group=rank),
        )
        # Rank 3 owns the adjacent middle chunks as one four-token run, but
        # ranks 0-2 each have two-token runs, below the decode threshold 3.
        results.append(
            dsv41_shardq.build_dsv41_shardq(_metadata([16], [16]), config, CPU)
        )
    assert results == [None] * WORLD


def test_indexer_resolves_dsv41_shardq_hook_from_static_layers():
    vllm_config = VllmConfig(cache_config=CacheConfig())
    with set_current_vllm_config(vllm_config):
        first = DeepseekV4IndexerCache(
            head_dim=132,
            dtype=torch.uint8,
            prefix="first.indexer.k_cache",
            cache_config=vllm_config.cache_config,
        )
        second = DeepseekV4IndexerCache(
            head_dim=132,
            dtype=torch.uint8,
            prefix="second.indexer.k_cache",
            cache_config=vllm_config.cache_config,
        )
    first.use_shardq = True
    second.use_shardq = True

    hook = _resolve_shardq_eligibility(
        {"second": second, "unrelated": torch.nn.Identity(), "first": first},
        ["second", "first"],
    )
    assert hook is dsv41_shardq.dsv41_shardq_eligible
