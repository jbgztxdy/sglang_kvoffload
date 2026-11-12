"""CUDA-visible views over pinned host memory.

The host-batch MLA kernel takes one cache tensor that spans both tiers, so the
host tier has to be addressable as a CUDA tensor. Pinned memory is UVA-visible
from the device, which makes a DLPack view over its pointer a valid (if slow)
CUDA tensor: kernels read it over PCIe.
"""

from __future__ import annotations

import ctypes
from typing import Dict, Tuple

import torch

_DL_DEVICE_CUDA = 2
_DTYPE_CODES: Dict[torch.dtype, Tuple[int, int]] = {
    torch.bfloat16: (4, 16),  # kDLBfloat
    torch.float16: (2, 16),   # kDLFloat
    torch.float32: (2, 32),
    torch.int32: (0, 32),     # kDLInt
}


class _DLDevice(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _DLDataType(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManagedTensor(ctypes.Structure):
    pass


_DLDeleter = ctypes.CFUNCTYPE(None, ctypes.POINTER(_DLManagedTensor))
_DLManagedTensor._fields_ = [
    ("tensor", _DLTensor),
    ("context", ctypes.c_void_p),
    ("deleter", _DLDeleter),
]

_owners: Dict[int, tuple] = {}


@_DLDeleter
def _release(pointer):
    _owners.pop(ctypes.addressof(pointer.contents), None)


_capsule_new = ctypes.pythonapi.PyCapsule_New
_capsule_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
_capsule_new.restype = ctypes.py_object


def host_kv_tensor(
    rows: int,
    dim: int,
    dtype: torch.dtype = torch.bfloat16,
    device_index: int = 0,
) -> torch.Tensor:
    """A [rows, 1, dim] CUDA tensor whose pages are pinned host DRAM."""
    if dtype not in _DTYPE_CODES:
        raise TypeError(f"host_kv_tensor does not support {dtype}")
    owner = torch.empty((rows, 1, dim), dtype=dtype, pin_memory=True)
    shape = (ctypes.c_int64 * 3)(rows, 1, dim)
    code, bits = _DTYPE_CODES[dtype]
    managed = _DLManagedTensor()
    managed.tensor = _DLTensor(
        owner.data_ptr(),
        _DLDevice(_DL_DEVICE_CUDA, device_index),
        3,
        _DLDataType(code, bits, 1),
        shape,
        None,
        0,
    )
    managed.deleter = _release
    address = ctypes.addressof(managed)
    _owners[address] = (managed, shape, owner)
    return torch.utils.dlpack.from_dlpack(_capsule_new(address, b"dltensor", None))
