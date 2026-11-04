# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Env-gated lab hooks for MoE router drift experiments (rank 0 only)."""

import os
import re

import torch

_STATE: dict[int, tuple[torch.Tensor, torch.Tensor, int, int]] = {}
_COUNTS: dict[str, int] = {}


def _enabled() -> bool:
    return os.environ.get("FP8LAB_DUMP_MOE", "0") == "1"


def _dump_dir() -> str | None:
    return os.environ.get("FP8LAB_DUMP_DIR")


def _rank_zero() -> bool:
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
    except Exception:
        pass
    return True


def set_row_sample_from_name(
    layer_name: str,
    ridx: torch.Tensor,
    positions: torch.Tensor,
    start: int,
    rows: int,
) -> None:
    """Record the sampled token rows for the MoE gate of the same layer."""
    if not _enabled():
        return
    m = re.search(r"layers\.(\d+)", layer_name)
    if m is None:
        return
    _STATE[int(m.group(1))] = (
        ridx.detach().to("cpu").long(),
        positions.detach().to("cpu").long(),
        int(start),
        int(rows),
    )


def maybe_register_gate_hook(gate: torch.nn.Module, prefix: str) -> None:
    if not _enabled():
        return
    m = re.search(r"layers\.(\d+)", prefix)
    if m is None:
        return
    layer_idx = int(m.group(1))
    bias = getattr(gate, "e_score_correction_bias", None)

    def _hook(module, inputs, output):
        _maybe_dump_gate(layer_idx, prefix, output, bias)

    gate.register_forward_hook(_hook)


def _maybe_dump_gate(layer_idx, layer_name, output, bias) -> None:
    if not _enabled():
        return
    dump_dir = _dump_dir()
    if not dump_dir or not _rank_zero():
        return
    if os.environ.get("FP8LAB_DUMP_REQUIRE_MARKER", "0") == "1" and not os.path.exists(
        os.path.join(dump_dir, "ENABLE")
    ):
        return
    sample = _STATE.pop(layer_idx, None)
    if sample is None:
        return
    ridx, positions, start, attn_rows = sample
    logits = output[0] if isinstance(output, (tuple, list)) else output
    if not torch.is_tensor(logits) or logits.dim() != 2:
        return
    if logits.shape[0] != attn_rows:
        return
    ridx = ridx.to(logits.device)
    if ridx.numel() == 0 or int(ridx.max()) >= logits.shape[0]:
        return
    sel = logits.index_select(0, ridx).detach().to(torch.float32).cpu()
    tag = _COUNTS.get(layer_name, 0) + 1
    payload = {
        "logits": sel,
        "positions": positions,
        "rows": attn_rows,
        "start": start,
        "layer": layer_name,
        "tag": str(tag),
    }
    if bias is not None:
        payload["bias"] = bias.detach().to(torch.float32).cpu()
    _COUNTS[layer_name] = tag
    os.makedirs(dump_dir, exist_ok=True)
    path = os.path.join(
        dump_dir, "moe_%s_%s.pt" % (tag, layer_name.replace("/", "_"))
    )
    torch.save(payload, path)
