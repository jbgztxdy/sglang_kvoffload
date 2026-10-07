"""Peer-GPU tier for HiCache: use spare HBM on other GPUs as the "host" tier.

Experiment: all requests are routed to DP rank 0 (cuda:0). Ranks 1..N-1 do no
attention (only EP/MoE work), so their HBM simulates NVLink-attached CPU DRAM.
KV evicted from rank 0's device pool is written to peer GPUs and loaded back
on a radix prefix hit. Everything else (radix tree, LRU eviction, write_back
policy, per-layer load/compute overlap, cached_tokens_details and Prometheus
hit metrics) is stock HiCache; only where the "host" buffers live changes.

* Main MLA KV (kv_lora_rank + rope, bf16): per-layer buffers on peer GPUs,
  layer i on peer i % num_peers, so a load-back reads all NVLink links at once.
  Copies use the stock JIT HiCache kernels launched on cuda:0; they take raw
  pointers, which reach peer memory through UVA once peer access is enabled.
* DSA indexer K cache (fp8 + scale): stays on rank 0's own GPU (a "host" pool
  allocated in local HBM), so the indexer never waits on NVLink.

Requires --enable-hierarchical-cache --hicache-io-backend kernel
--hicache-mem-layout layer_first, and env PEER_GPU_HICACHE=1, which
hybrid_pool_assembler checks before building host pools.

Env:
  PEER_GPU_HICACHE=1                  enable
  PEER_GPU_HICACHE_RANK=0             attention-DP rank that owns the peer tier
  PEER_GPU_HICACHE_IDLE_HOST_GB=0.2   CPU host tier size on the other ranks
                                      (they get no requests; keep it tiny)
  PEER_GPU_HICACHE_BLOCK_QUOTA=<int>  copy-kernel block quota (default: stock)
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional, Sequence

import torch

from sglang.kernels.ops.kvcache.hicache import can_use_hicache_jit_kernel
from sglang.kernels.ops.kvcache.hicache import (
    transfer_hicache_all_layer_mla as jit_transfer_hicache_all_layer_mla,
)
from sglang.kernels.ops.kvcache.hicache import (
    transfer_hicache_one_layer_mla as jit_transfer_hicache_one_layer_mla,
)
from sglang.srt.layers.dcp.layout import maybe_dcp_kernel_indices
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.mem_cache.pool_host.base import sync_fixed_hicache_size
from sglang.srt.mem_cache.pool_host.common import (
    get_allocator_from_storage,
    make_kernel_ptr_table,
)
from sglang.srt.mem_cache.pool_host.dsa import DSAIndexerPoolHost
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost

logger = logging.getLogger(__name__)


def peer_hicache_enabled() -> bool:
    return os.environ.get("PEER_GPU_HICACHE", "0") == "1"


def is_peer_owner_rank() -> bool:
    from sglang.srt.runtime_context import get_parallel

    return get_parallel().attn_dp_rank == int(os.environ.get("PEER_GPU_HICACHE_RANK", "0"))


def idle_rank_host_gb() -> float:
    return float(os.environ.get("PEER_GPU_HICACHE_IDLE_HOST_GB", "0.2"))


def _block_quota() -> Optional[int]:
    value = os.environ.get("PEER_GPU_HICACHE_BLOCK_QUOTA")
    return int(value) if value else None


def _enable_peer_access(local: int, peers: list[int]) -> None:
    """Enable P2P from the local GPU's context to every peer; fail loudly otherwise."""
    from cuda.bindings import runtime as cudart

    prev = torch.cuda.current_device()
    torch.cuda.set_device(local)
    try:
        for peer in peers:
            err, can = cudart.cudaDeviceCanAccessPeer(local, peer)
            if err != cudart.cudaError_t.cudaSuccess or not can:
                raise RuntimeError(f"Peer-GPU HiCache: cuda:{local} cannot access cuda:{peer}")
            (err,) = cudart.cudaDeviceEnablePeerAccess(peer, 0)
            if err not in (
                cudart.cudaError_t.cudaSuccess,
                cudart.cudaError_t.cudaErrorPeerAccessAlreadyEnabled,
            ):
                raise RuntimeError(
                    f"Peer-GPU HiCache: cudaDeviceEnablePeerAccess(cuda:{local} -> cuda:{peer}) failed: {err}"
                )
    finally:
        torch.cuda.set_device(prev)
    logger.info("Peer-GPU HiCache: peer access enabled cuda:%d -> %s", local, peers)


class PeerGPUMLAHostPool(MLATokenToKVPoolHost):
    """MLA host pool whose per-layer buffers live on peer GPUs (layer_first, kernel only)."""

    def __init__(
        self,
        device_pool: MLATokenToKVPool,
        host_to_device_ratio: float,
        host_size: float,
        page_size: int,
        override_kv_cache_dim: Optional[int] = None,
        mtp_draft_device_pools: Sequence[MLATokenToKVPool] = (),
        dcp_size: int = 1,
        dcp_rank: int = 0,
        *,
        allocator_type: str = "default",
        pool_label: str = "kv",
    ):
        # HostKVCache.__init__ sizes the pool against psutil host RAM and
        # allocates via the CPU-only allocator, so set the same fields here.
        self.override_kv_cache_dim = override_kv_cache_dim
        self.mtp_draft_device_pools = tuple(mtp_draft_device_pools)
        self._is_dummy = False
        self.device_pool = device_pool
        self.pool_label = pool_label
        self.dcp_size = dcp_size
        self.dcp_rank = dcp_rank
        assert page_size % dcp_size == 0
        self.page_size = page_size // dcp_size
        self.layout = "layer_first"
        self.pin_memory = False
        # Slot bookkeeping (mem_state, free_slots) stays on CPU like stock HiCache.
        self.device = "cpu"
        self.allocator = get_allocator_from_storage(allocator_type)
        self.can_use_write_back_jit = False

        self.dtype = device_pool.store_dtype
        self.size_per_token = self.get_size_per_token()
        if host_size > 0:
            self.size = sync_fixed_hicache_size(
                int(host_size * 1e9 // self.size_per_token), host_size
            )
        else:
            self.size = int(device_pool.size * host_to_device_ratio)
        self.page_num = self.size // self.page_size + 1
        self.size = self.page_num * self.page_size
        self.start_layer = device_pool.start_layer
        self.end_layer = device_pool.end_layer

        self.kv_buffer = self.init_kv_buffer()
        self.fd = None
        self.lock = threading.RLock()
        self.clear()

        self.can_use_jit = can_use_hicache_jit_kernel(
            element_size=self.kv_cache_dim * self.dtype.itemsize
        )
        assert self.can_use_jit, "Peer-GPU HiCache needs the JIT HiCache kernels"
        self.data_refs = list(self.kv_buffer)
        # Raw data_ptr()s of peer tensors; valid on cuda:0 through UVA + P2P.
        self.data_ptrs = make_kernel_ptr_table(
            self.data_refs, self.device_pool.device, host_memory_registered=False
        )
        if self.mtp_draft_device_pools:
            device_pools = (self.device_pool, *self.mtp_draft_device_pools)
            self.packed_device_data_ptrs = torch.cat([p.data_ptrs for p in device_pools])
            self.packed_device_kv_buffers = [b for p in device_pools for b in p.kv_buffer]
        self.staging_page_capacity = 0
        self.staging_token_capacity = 0
        self.staging_buffer = None
        self.block_quota = _block_quota()

    def init_kv_buffer(self) -> list[torch.Tensor]:
        local = torch.device(self.device_pool.device).index
        peers = [i for i in range(torch.cuda.device_count()) if i != local]
        if not peers:
            raise RuntimeError("Peer-GPU HiCache: no peer GPU visible to this process")
        _enable_peer_access(local, peers)

        self.token_stride_size = self.kv_cache_dim * self.dtype.itemsize
        self.layout_dim = self.token_stride_size * self.layer_num
        buffers = [
            torch.empty(
                (self.size, 1, self.kv_cache_dim),
                dtype=self.dtype,
                device=f"cuda:{peers[layer % len(peers)]}",
            )
            for layer in range(self.layer_num)
        ]
        per_peer = {p: 0 for p in peers}
        for layer in range(self.layer_num):
            per_peer[peers[layer % len(peers)]] += self.size * self.token_stride_size
        logger.info(
            "Peer-GPU HiCache: %s pool %d tokens x %d layers on peers, GB per peer: %s",
            self.pool_label,
            self.size,
            self.layer_num,
            {f"cuda:{p}": round(b / 1e9, 2) for p, b in per_peer.items()},
        )
        return buffers

    def destroy(self) -> None:
        self._destroyed = True
        self.kv_buffer = None

    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend, *, is_draft=False
    ) -> None:
        assert io_backend == "kernel", f"Peer-GPU HiCache needs io_backend=kernel, got {io_backend}"
        if not is_draft and not self._is_device_layer_owned(device_pool, layer_id):
            return
        host_indices = maybe_dcp_kernel_indices(host_indices, self.dcp_size, self.dcp_rank)
        device_indices = maybe_dcp_kernel_indices(device_indices, self.dcp_size, self.dcp_rank)
        host_layer_id = layer_id if is_draft else self._host_layer_index(layer_id)
        device_layer_id = 0 if is_draft else layer_id
        jit_transfer_hicache_one_layer_mla(
            cache_dst=device_pool.kv_buffer[device_layer_id],
            cache_src=self.kv_buffer[host_layer_id],
            indices_dst=device_indices,
            indices_src=host_indices,
            element_dim=self.kv_cache_dim,
            block_quota=self.block_quota,
        )

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ) -> None:
        assert io_backend == "kernel", f"Peer-GPU HiCache needs io_backend=kernel, got {io_backend}"
        assert not self._is_device_layer_sharded(device_pool)
        host_indices = maybe_dcp_kernel_indices(host_indices, self.dcp_size, self.dcp_rank)
        device_indices = maybe_dcp_kernel_indices(device_indices, self.dcp_size, self.dcp_rank)
        device_data_ptrs, _ = self._resolve_device_transfer_buffers(device_pool)
        jit_transfer_hicache_all_layer_mla(
            ptr_dst=self.data_ptrs,
            indices_dst=host_indices,
            ptr_src=device_data_ptrs,
            indices_src=device_indices,
            cache_dst_stride_bytes=self.token_stride_size,
            cache_src_stride_bytes=self.token_stride_size,
            element_size=self.kv_cache_dim * self.dtype.itemsize,
            block_quota=self.block_quota,
        )

    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        raise NotImplementedError("Peer-GPU HiCache does not support L3 storage")

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        raise NotImplementedError("Peer-GPU HiCache does not support L3 storage")

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        raise NotImplementedError("Peer-GPU HiCache does not support L3 storage")

    def is_stride_page_aligned(self, page_size_bytes: int = 4096) -> bool:
        return False


class LocalDeviceIndexerPoolHost(DSAIndexerPoolHost):
    """DSA indexer "host" pool kept in the owner rank's own HBM (layer_first)."""

    def __init__(self, device_pool, anchor_host, layout: str, allocator_type: str = "default"):
        assert layout == "layer_first", "Peer-GPU HiCache needs --hicache-mem-layout layer_first"
        super().__init__(
            device_pool, anchor_host, layout, pin_memory=False, allocator_type=allocator_type
        )

    def init_kv_buffer(self):
        device_pools = (self.device_pool, *self.mtp_draft_device_pools)
        self.packed_device_index_buffers = [
            b for p in device_pools for b in p.index_k_with_scale_buffer
        ]
        self.index_k_device_ptrs = torch.tensor(
            [x.data_ptr() for x in self.packed_device_index_buffers],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )
        self.index_k_with_scale_buffer = torch.empty(
            (self.layer_num, self.indexer_page_num, self.indexer_page_stride_size),
            dtype=self.indexer_dtype,
            device=self.device_pool.device,
        )
        self.index_k_data_refs = [self.index_k_with_scale_buffer[i] for i in range(self.layer_num)]
        self.index_k_data_ptrs = make_kernel_ptr_table(
            self.index_k_data_refs, self.device_pool.device, host_memory_registered=False
        )
        logger.info(
            "Peer-GPU HiCache: DSA indexer tier %.2f GB on local %s",
            self.index_k_with_scale_buffer.nbytes / 1e9,
            self.device_pool.device,
        )

    def destroy(self) -> None:
        self._destroyed = True
        self.kv_buffer = None


def build_kv_host_pool_hook(
    *,
    kv_pool,
    page_size,
    use_mla,
    override_kv_cache_dim,
    host_size,
    mtp_draft_device_pools,
    pool_label,
    allocator_type,
    default_builder,
):
    """Called by hybrid_pool_assembler.build_kv_host_pool when PEER_GPU_HICACHE=1."""
    from sglang.srt.runtime_context import get_memory, get_parallel

    if not is_peer_owner_rank():
        # Idle ranks get no requests; keep their (CPU) tier tiny.
        return default_builder(host_size=idle_rank_host_gb())
    assert use_mla, "Peer-GPU HiCache is only implemented for MLA KV pools"
    memory = get_memory()
    assert memory.hicache_io_backend == "kernel" and memory.hicache_mem_layout == "layer_first", (
        "Peer-GPU HiCache needs --hicache-io-backend kernel --hicache-mem-layout layer_first"
    )
    kwargs = {}
    parallel = get_parallel()
    if parallel.dcp_enabled:
        kwargs.update(dcp_size=parallel.attn_dcp_size, dcp_rank=parallel.attn_dcp_rank)
    return PeerGPUMLAHostPool(
        kv_pool,
        memory.hicache_ratio,
        memory.hicache_size if host_size is None else host_size,
        page_size,
        override_kv_cache_dim=override_kv_cache_dim,
        mtp_draft_device_pools=mtp_draft_device_pools,
        allocator_type=allocator_type,
        pool_label=pool_label,
        **kwargs,
    )


def build_dsa_indexer_host_pool(kv_pool, kv_host_pool, layout, allocator_type):
    """Indexer sidecar: local HBM on the owner rank, stock CPU pool elsewhere."""
    if peer_hicache_enabled() and is_peer_owner_rank():
        return LocalDeviceIndexerPoolHost(kv_pool, kv_host_pool, layout, allocator_type)
    return DSAIndexerPoolHost(kv_pool, kv_host_pool, layout, allocator_type=allocator_type)
