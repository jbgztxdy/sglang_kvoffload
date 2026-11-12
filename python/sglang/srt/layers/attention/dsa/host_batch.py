"""Decode through the fused host-batch MLA kernel.

One launch per layer performs top-k dedup, LRU slot selection, remote page fetch
and attention. Pages at or above ``host_page_start`` live outside HBM (pinned
host DRAM or a peer GPU mapped into the same address range); the kernel reads
them where they are and keeps a device copy in a per-request cache slot.

The kernel is only defined for one query token per row, page size one, FA2
non-causal decode and a multiple of 64 heads, so this module refuses anything
else instead of silently changing the math.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

_KV_LORA_RANK = 512
_QK_ROPE_DIM = 64


def _ensure_source_jit_paths() -> None:
    """Point the JIT at the source tree when flashinfer runs from a checkout.

    A wheel ships ``flashinfer/data/{include,csrc}``; a git checkout keeps them at
    the repository root instead, and the kernel header only exists in the latter.
    """
    from pathlib import Path

    from flashinfer.jit import env as jit_env

    wanted = Path(jit_env.FLASHINFER_INCLUDE_DIR) / "flashinfer/attention/mla_host_batch.cuh"
    if wanted.exists():
        return

    import flashinfer

    root = Path(flashinfer.__file__).resolve().parents[1]
    include = root / "include"
    if not (include / "flashinfer/attention/mla_host_batch.cuh").exists():
        raise RuntimeError(
            "flashinfer JIT cannot locate mla_host_batch.cuh; set FLASHINFER_INCLUDE_DIR "
            f"to the checkout's include dir (looked under {include})"
        )
    jit_env.FLASHINFER_INCLUDE_DIR = include
    jit_env.FLASHINFER_CSRC_DIR = root / "csrc"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw in (None, "") else int(raw)


class HostBatchMLADecode:
    """One wrapper per (layer, rows, topk); the kernel keeps LRU state across calls."""

    def __init__(self, workspace_bytes: int = 256 << 20) -> None:
        self.fetch_ctas = _env_int("SGLANG_HOST_BATCH_FETCH_CTAS", 24)
        self.cache_slots = _env_int("SGLANG_HOST_BATCH_CACHE_SLOTS", 64)
        self.item_entries = _env_int("SGLANG_HOST_BATCH_ITEM_ENTRIES", 64)
        self._workspace_bytes = workspace_bytes
        self._workspace: Optional[torch.Tensor] = None
        # (layer_id, rows, topk, sm_scale) -> (wrapper, index buffer)
        self._entries: Dict[Tuple[int, int, int, float], Tuple[object, torch.Tensor]] = {}

    def _ws(self, device: torch.device) -> torch.Tensor:
        if self._workspace is None or self._workspace.device != device:
            self._workspace = torch.empty(
                self._workspace_bytes, dtype=torch.uint8, device=device
            )
        return self._workspace

    def _build(
        self,
        layer_id: int,
        rows: int,
        topk: int,
        sm_scale: float,
        device: torch.device,
    ) -> Tuple[object, torch.Tensor]:
        _ensure_source_jit_paths()
        from flashinfer.mla import BatchMLAPagedAttentionWrapper

        index_buf = torch.empty(rows * topk, dtype=torch.int32, device=device)
        wrapper = BatchMLAPagedAttentionWrapper(self._ws(device), backend="fa2")
        wrapper.plan(
            torch.arange(rows + 1, dtype=torch.int32),
            torch.arange(rows + 1, dtype=torch.int32) * topk,
            index_buf,
            torch.full((rows,), topk, dtype=torch.int32),
            64,
            _KV_LORA_RANK,
            _QK_ROPE_DIM,
            1,
            False,
            sm_scale,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
        )
        wrapper.host_fetch_ctas = self.fetch_ctas
        wrapper.host_item_entries = self.item_entries
        wrapper.host_split = False
        wrapper.host_stage = True
        wrapper.host_staging_rows = rows * topk
        wrapper.set_host_cache(
            torch.arange(rows + 1, dtype=torch.int32), self.cache_slots
        )
        logger.info(
            "host-batch MLA: layer %d rows %d topk %d, fetch_ctas %d, cache slots %d",
            layer_id,
            rows,
            topk,
            self.fetch_ctas,
            self.cache_slots,
        )
        return wrapper, index_buf

    def forward(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        indices: torch.Tensor,
        layer_id: int,
        sm_scale: float,
        host_page_start: int,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """q_nope [rows, heads, 512], q_pe [rows, heads, 64], kv_cache [pages, 1, 576]."""
        rows, heads = q_nope.shape[0], q_nope.shape[1]
        topk = indices.shape[1]
        if heads % 64:
            raise ValueError(f"host-batch MLA needs a multiple of 64 heads, got {heads}")
        if q_nope.shape[2] != _KV_LORA_RANK or q_pe.shape[2] != _QK_ROPE_DIM:
            raise ValueError(
                "host-batch MLA needs the absorbed query "
                f"({_KV_LORA_RANK} nope + {_QK_ROPE_DIM} rope), got "
                f"{tuple(q_nope.shape)} / {tuple(q_pe.shape)}"
            )
        key = (layer_id, rows, topk, round(float(sm_scale), 8))
        entry = self._entries.get(key)
        if entry is None:
            entry = self._build(layer_id, rows, topk, sm_scale, q_nope.device)
            self._entries[key] = entry
        wrapper, index_buf = entry

        index_buf.copy_(indices.reshape(-1))
        if out is None:
            out = torch.empty_like(q_nope)
        wrapper.run(
            q_nope,
            q_pe,
            kv_cache[..., :_KV_LORA_RANK],
            kv_cache[..., _KV_LORA_RANK:],
            out=out,
            host_page_start=host_page_start,
        )
        return out
