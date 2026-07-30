from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB_PATH = os.path.join(ROOT, "dist", "libmojo-mpmath.so")
I = ctypes.c_int64

_SIGNATURES = {
    "mmp_compare_abs": ([I, I, I, I], I),
    "mmp_add_abs": ([I, I, I, I, I, I], I),
    "mmp_sub_abs": ([I, I, I, I, I], I),
    "mmp_mul_abs": ([I, I, I, I, I, I, I, I], I),
    "mmp_power": ([I, I, I, I, I, I, I, I, I], I),
    "mmp_factorial": ([I, I, I], I),
    "mmp_stride_product": ([I, I, I, I, I], I),
    "mmp_binomial": ([I, I, I, I], I),
    "mmp_fibonacci": ([I, I, I, I, I, I, I, I, I, I, I], I),
}

_library: ctypes.CDLL | None = None


def build() -> str:
    source = os.path.join(ROOT, "src", "kernels.mojo")
    if (
        not os.path.exists(LIB_PATH)
        or os.path.getmtime(LIB_PATH) < os.path.getmtime(source)
    ):
        subprocess.run(
            ["bash", os.path.join(ROOT, "build", "build.sh")],
            cwd=ROOT,
            check=True,
        )
    return LIB_PATH


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (arguments, result) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = arguments
            function.restype = result
    return _library


def addr(value: np.ndarray) -> int:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.dtype("=u4")
        or not value.flags.c_contiguous
        or not value.flags.aligned
        or value.ndim != 1
        or value.size == 0
        or value.ctypes.data == 0
    ):
        raise TypeError("FFI limbs must be a nonempty contiguous uint32 array")
    return int(value.ctypes.data)
