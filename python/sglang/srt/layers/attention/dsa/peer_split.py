"""Split sparse attention into local and remote top-k entries (peer-GPU HiCache).

While a layer's remote KV (slots being loaded back from peer GPUs) is still in
flight, attention runs over the top-k entries whose KV is already local; after
the load completes it runs over the remote entries, and the two partial
outputs are merged with their log-sum-exp (base 2, as FlashMLA returns it).

``split_topk``: per row of ``indices`` [s_q, k] (KV slot ids, -1 = invalid),
compact the local and the remote entries to the front of two [s_q, k] tensors
(-1 padded) and return their per-row counts. Only the first ``row_len[i]``
columns of row i are considered when ``row_len`` is given.

``merge_partial``: out = (o1 * 2^l1 + o2 * 2^l2) / (2^l1 + 2^l2) per (row, head);
a partial with count 0 contributes nothing (its o / lse may be garbage).
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _split_topk_kernel(
    indices_ptr,
    indices_stride,
    row_len_ptr,
    remote_mask_ptr,
    local_out_ptr,
    remote_out_ptr,
    local_len_ptr,
    remote_len_ptr,
    K,
    HAS_ROW_LEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    in_k = cols < K
    idx = tl.load(indices_ptr + row * indices_stride + cols, mask=in_k, other=-1)
    valid = in_k & (idx >= 0)
    if HAS_ROW_LEN:
        valid = valid & (cols < tl.load(row_len_ptr + row))
    is_remote = (tl.load(remote_mask_ptr + idx, mask=valid, other=0) != 0) & valid
    is_local = valid & (~is_remote)
    local_pos = tl.cumsum(is_local.to(tl.int32), axis=0) - 1
    remote_pos = tl.cumsum(is_remote.to(tl.int32), axis=0) - 1
    tl.store(local_out_ptr + row * K + local_pos, idx, mask=is_local)
    tl.store(remote_out_ptr + row * K + remote_pos, idx, mask=is_remote)
    tl.store(local_len_ptr + row, tl.sum(is_local.to(tl.int32), axis=0))
    tl.store(remote_len_ptr + row, tl.sum(is_remote.to(tl.int32), axis=0))


def split_topk(
    indices: torch.Tensor,
    remote_mask: torch.Tensor,
    row_len: Optional[torch.Tensor] = None,
):
    """Returns (local_idx, local_len, remote_idx, remote_len); int32."""
    assert indices.dim() == 2 and indices.dtype == torch.int32
    assert remote_mask.dtype == torch.uint8
    s_q, k = indices.shape
    local_idx = torch.full((s_q, k), -1, dtype=torch.int32, device=indices.device)
    remote_idx = torch.full_like(local_idx, -1)
    local_len = torch.empty(s_q, dtype=torch.int32, device=indices.device)
    remote_len = torch.empty_like(local_len)
    if s_q == 0:
        return local_idx, local_len, remote_idx, remote_len
    has_len = row_len is not None
    _split_topk_kernel[(s_q,)](
        indices,
        indices.stride(0),
        row_len if has_len else local_len,
        remote_mask,
        local_idx,
        remote_idx,
        local_len,
        remote_len,
        k,
        HAS_ROW_LEN=has_len,
        BLOCK=triton.next_power_of_2(k),
    )
    return local_idx, local_len, remote_idx, remote_len


def split_topk_reference(indices, remote_mask, row_len=None):
    s_q, k = indices.shape
    cols = torch.arange(k, device=indices.device).unsqueeze(0)
    valid = indices >= 0
    if row_len is not None:
        valid &= cols < row_len.unsqueeze(1)
    is_remote = valid & (remote_mask[indices.clamp_min(0).long()] != 0)
    is_local = valid & ~is_remote
    out = []
    for sel in (is_local, is_remote):
        compact = torch.full((s_q, k + 1), -1, dtype=torch.int32, device=indices.device)
        pos = torch.where(sel, sel.to(torch.int32).cumsum(1) - 1, k)
        compact.scatter_(1, pos.long(), torch.where(sel, indices, -1))
        out += [compact[:, :k].contiguous(), sel.sum(1).to(torch.int32)]
    return out[0], out[1], out[2], out[3]


@triton.jit
def _merge_partial_kernel(
    o1_ptr,
    o1_row_stride,
    o1_head_stride,
    lse1_ptr,
    lse1_row_stride,
    len1_ptr,
    o2_ptr,
    o2_row_stride,
    o2_head_stride,
    lse2_ptr,
    lse2_row_stride,
    len2_ptr,
    out_ptr,
    out_row_stride,
    out_head_stride,
    D,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    has1 = tl.load(len1_ptr + row) > 0
    has2 = tl.load(len2_ptr + row) > 0
    l1 = tl.load(lse1_ptr + row * lse1_row_stride + head)
    l2 = tl.load(lse2_ptr + row * lse2_row_stride + head)
    l1 = tl.where(has1, l1, float("-inf"))
    l2 = tl.where(has2, l2, float("-inf"))
    m = tl.maximum(l1, l2)
    m = tl.where(m == float("-inf"), 0.0, m)
    w1 = tl.where(has1, tl.exp2(l1 - m), 0.0)
    w2 = tl.where(has2, tl.exp2(l2 - m), 0.0)
    denom = w1 + w2
    denom = tl.where(denom > 0, denom, 1.0)
    d = tl.arange(0, BLOCK_D)
    in_d = d < D
    a = tl.load(
        o1_ptr + row * o1_row_stride + head * o1_head_stride + d, mask=in_d & has1, other=0.0
    ).to(tl.float32)
    b = tl.load(
        o2_ptr + row * o2_row_stride + head * o2_head_stride + d, mask=in_d & has2, other=0.0
    ).to(tl.float32)
    out = (a * w1 + b * w2) / denom
    tl.store(
        out_ptr + row * out_row_stride + head * out_head_stride + d,
        out.to(out_ptr.dtype.element_ty),
        mask=in_d,
    )


def merge_partial(o1, lse1, len1, o2, lse2, len2):
    """o*: [s_q, h, d]; lse*: [s_q, h] float32 (base 2); len*: [s_q] int32."""
    s_q, h, d = o1.shape
    assert o1.stride(-1) == 1 and o2.stride(-1) == 1 and lse1.stride(-1) == 1 and lse2.stride(-1) == 1
    out = torch.empty((s_q, h, d), dtype=o1.dtype, device=o1.device)
    if s_q == 0:
        return out
    _merge_partial_kernel[(s_q, h)](
        o1, o1.stride(0), o1.stride(1), lse1, lse1.stride(0), len1,
        o2, o2.stride(0), o2.stride(1), lse2, lse2.stride(0), len2,
        out, out.stride(0), out.stride(1),
        d,
        BLOCK_D=triton.next_power_of_2(d),
    )
    return out


def merge_partial_reference(o1, lse1, len1, o2, lse2, len2):
    has1 = (len1 > 0).view(-1, 1)
    has2 = (len2 > 0).view(-1, 1)
    l1 = torch.where(has1, lse1.float(), float("-inf"))
    l2 = torch.where(has2, lse2.float(), float("-inf"))
    m = torch.maximum(l1, l2)
    m = torch.where(torch.isinf(m), torch.zeros_like(m), m)
    w1 = torch.where(has1, torch.exp2(l1 - m), torch.zeros_like(m))
    w2 = torch.where(has2, torch.exp2(l2 - m), torch.zeros_like(m))
    denom = (w1 + w2).clamp_min(1e-30)
    a = torch.where(has1.unsqueeze(-1), o1.float(), torch.zeros((), device=o1.device))
    b = torch.where(has2.unsqueeze(-1), o2.float(), torch.zeros((), device=o2.device))
    return ((a * w1.unsqueeze(-1) + b * w2.unsqueeze(-1)) / denom.unsqueeze(-1)).to(o1.dtype)
