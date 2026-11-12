"""KV pool for the flashinfer_host_batch decode path.

Every KV row of every layer lives in pinned host DRAM, exposed as a CUDA tensor
through UVA. Token ids therefore need no translation: the kernel sees one cache
spanning only the host tier (``host_page_start = 0``) and keeps its own device
copy of the pages it is asked for; the LRU slots are the whole device tier.

This trades HBM capacity for PCIe traffic, which is exactly the configuration
the fused kernel is built for. It is opt-in: ``SGLANG_HOST_BATCH_HOST_KV=0``
keeps the normal HBM pool with our kernel reading it in place.
"""

from __future__ import annotations

import logging
import os

import torch

from sglang.srt.mem_cache.host_batch_arena import host_kv_tensor
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool

logger = logging.getLogger(__name__)


def host_kv_enabled() -> bool:
    return os.environ.get("SGLANG_HOST_BATCH_HOST_KV", "1") not in ("0", "false", "False")


class HostBatchDSATokenToKVPool(DSATokenToKVPool):
    """DSA KV pool whose rows are all in host DRAM (one UVA tensor per layer)."""

    # The kernel treats every page as remote; its cache slots are the device tier.
    host_page_start = 0

    def _create_buffers(self):
        if self.store_dtype != self.dtype:
            raise RuntimeError(
                "HostBatchDSATokenToKVPool supports only the unquantized KV dtype "
                f"({self.dtype}), got store dtype {self.store_dtype}"
            )
        rows = self.size + self.page_size
        # ``device`` may be a torch.device or a "cuda:N" string, depending on the caller.
        device = self.device if isinstance(self.device, torch.device) else torch.device(self.device)
        index = 0 if device.index is None else int(device.index)
        self.kv_buffer = [
            host_kv_tensor(rows, self.kv_cache_dim, self.store_dtype, index)
            for _ in range(self.layer_num)
        ]
        logger.info(
            "host-batch KV pool: %d tokens x %d layers x %d dims in pinned host DRAM "
            "(%.1f MiB)",
            rows,
            self.layer_num,
            self.kv_cache_dim,
            rows * self.layer_num * self.kv_cache_dim * 2 / (1 << 20),
        )

    def _clear_buffers(self):
        self.kv_buffer = []
