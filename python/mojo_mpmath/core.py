from __future__ import annotations

import math
import operator
from typing import Any

import numpy as np

from ._lib import addr, lib

_LOG2_PHI = math.log2((1.0 + math.sqrt(5.0)) / 2.0)
_LOG2_SQRT5 = 0.5 * math.log2(5.0)
_MAX_KERNEL_INT = 2**63 - 1


def _integer(value: Any, name: str = "value") -> int:
    try:
        return operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer") from None


def _limbs(value: int) -> np.ndarray:
    magnitude = abs(value)
    size = max(1, (magnitude.bit_length() + 31) // 32)
    raw = magnitude.to_bytes(size * 4, "little")
    # Decode the Python byte representation explicitly, then expose native-endian
    # UInt32 values to Mojo. This stays zero-copy on the supported little-endian
    # build platform and remains numerically correct elsewhere.
    return np.frombuffer(raw, dtype="<u4").astype(np.uint32, copy=False)


def _value(limbs: np.ndarray, length: int, sign: int = 1) -> int:
    if length < 1 or length > len(limbs):
        if length < 0:
            raise RuntimeError(f"Mojo kernel rejected the FFI call (status {length})")
        raise RuntimeError(
            f"Mojo kernel returned invalid length {length} for capacity {len(limbs)}"
        )
    raw = limbs[:length].astype("<u4", copy=False).tobytes()
    magnitude = int.from_bytes(raw, "little")
    return -magnitude if sign < 0 and magnitude else magnitude


def _empty(capacity: int) -> np.ndarray:
    return np.empty(max(1, capacity), dtype=np.uint32)


def _scratch(capacity: int, copies: int = 1) -> tuple[np.ndarray, int]:
    stride = 4 * capacity + 64
    return _empty(copies * stride), stride


def _kernel_int(value: int, name: str) -> int:
    if not 0 <= value <= _MAX_KERNEL_INT:
        raise OverflowError(f"{name} does not fit the Mojo kernel index")
    return value


def _kernel_signed(value: int, name: str) -> int:
    # Exclude INT64_MIN because taking its magnitude is not representable.
    if not -_MAX_KERNEL_INT <= value <= _MAX_KERNEL_INT:
        raise OverflowError(f"{name} does not fit the Mojo kernel integer")
    return value


def fadd(x, y, **kwargs):
    """Exact addition for integer operands; accepts mpmath's keyword shape."""
    a = _integer(x, "x")
    b = _integer(y, "y")
    aa = _limbs(a)
    bb = _limbs(b)
    result = _empty(max(len(aa), len(bb)) + 1)
    if (a < 0) == (b < 0):
        length = lib().mmp_add_abs(
            addr(aa), len(aa), addr(bb), len(bb), addr(result), len(result)
        )
        return _value(result, length, -1 if a < 0 else 1)
    comparison = lib().mmp_compare_abs(addr(aa), len(aa), addr(bb), len(bb))
    if comparison == 0:
        return 0
    if comparison > 0:
        length = lib().mmp_sub_abs(
            addr(aa), len(aa), addr(bb), len(bb), addr(result)
        )
        return _value(result, length, -1 if a < 0 else 1)
    length = lib().mmp_sub_abs(
        addr(bb), len(bb), addr(aa), len(aa), addr(result)
    )
    return _value(result, length, -1 if b < 0 else 1)


def fsub(x, y, **kwargs):
    """Exact subtraction for integer operands."""
    return fadd(_integer(x, "x"), -_integer(y, "y"), **kwargs)


def fmul(x, y, **kwargs):
    """Exact multiplication for integer operands."""
    a = _integer(x, "x")
    b = _integer(y, "y")
    aa = _limbs(a)
    bb = _limbs(b)
    result = _empty(len(aa) + len(bb))
    scratch, _ = _scratch(len(result))
    length = lib().mmp_mul_abs(
        addr(aa),
        len(aa),
        addr(bb),
        len(bb),
        addr(result),
        len(result),
        addr(scratch),
        len(scratch),
    )
    return _value(result, length, -1 if (a < 0) != (b < 0) else 1)


def power(x, y):
    """Exact nonnegative integral power for integer operands."""
    base_value = _integer(x, "x")
    exponent = _integer(y, "y")
    if exponent < 0:
        raise ValueError("negative exponents are outside the exact integer subset")
    _kernel_int(exponent, "exponent")
    if base_value == 0 and exponent == 0:
        return 1
    bits = max(1, abs(base_value).bit_length() * exponent + 1)
    capacity = (bits + 31) // 32 + 1
    base = _limbs(base_value)
    result = _empty(capacity)
    current = _empty(capacity)
    temporary = _empty(capacity)
    scratch, _ = _scratch(capacity)
    length = lib().mmp_power(
        addr(base),
        len(base),
        exponent,
        addr(result),
        addr(current),
        addr(temporary),
        capacity,
        addr(scratch),
        len(scratch),
    )
    sign = -1 if base_value < 0 and exponent & 1 else 1
    return _value(result, length, sign)


def fac(x, **kwargs):
    """Return n! exactly for a nonnegative integer n."""
    n = _integer(x, "x")
    if n < 0:
        raise ValueError("gamma function pole")
    _kernel_int(n, "x")
    bits = 1 if n < 2 else math.ceil(math.lgamma(n + 1) / math.log(2.0)) + 2
    result = _empty((bits + 31) // 32 + 1)
    length = lib().mmp_factorial(n, addr(result), len(result))
    return _value(result, length)


factorial = fac


def fac2(x):
    """Return the double factorial n!! exactly for integer n >= -1."""
    n = _integer(x, "x")
    if n in (-1, 0):
        return 1
    if n < -1:
        raise ValueError("gamma function pole")
    count = (n + 1) // 2
    first = 1 if n & 1 else 2
    _kernel_int(count, "factor count")
    bits = count * max(1, n.bit_length()) + 2
    result = _empty((bits + 31) // 32 + 1)
    length = lib().mmp_stride_product(
        first, count, 2, addr(result), len(result)
    )
    return _value(result, length)


def fib(x, **kwargs):
    """Return the exact Fibonacci number for an integer index."""
    n = _integer(x, "x")
    magnitude = abs(n)
    _kernel_int(magnitude, "x")
    bits = (
        1
        if magnitude < 2
        else math.ceil(magnitude * _LOG2_PHI - _LOG2_SQRT5) + 2
    )
    capacity = (bits + 31) // 32 + 2
    buffers = [_empty(capacity) for _ in range(6)]
    scratch, scratch_stride = _scratch(capacity, copies=3)
    length = lib().mmp_fibonacci(
        magnitude,
        *(addr(buffer) for buffer in buffers),
        capacity,
        addr(scratch),
        scratch_stride,
        len(scratch),
    )
    sign = -1 if n < 0 and magnitude % 2 == 0 else 1
    return _value(buffers[0], length, sign)


fibonacci = fib


def binomial(n, k):
    """Return the exact generalized binomial coefficient for integer inputs."""
    top = _integer(n, "n")
    bottom = _integer(k, "k")
    if bottom < 0:
        return 0
    sign = 1
    if top < 0:
        sign = -1 if bottom & 1 else 1
        top = bottom - top - 1
    if bottom > top:
        return 0
    bottom = min(bottom, top - bottom)
    _kernel_int(top, "n")
    _kernel_int(bottom, "k")
    if bottom == 0:
        return 1
    log_bits = (
        math.lgamma(top + 1)
        - math.lgamma(bottom + 1)
        - math.lgamma(top - bottom + 1)
    ) / math.log(2.0)
    result = _empty((math.ceil(log_bits) + 63) // 32 + 1)
    length = lib().mmp_binomial(
        top, bottom, addr(result), len(result)
    )
    return _value(result, length, sign)


def _stride_value(first: int, count: int, stride: int) -> int:
    if count < 0:
        raise ValueError("negative lengths are outside the exact integer subset")
    if count == 0:
        return 1
    last = first + (count - 1) * stride
    if first <= 0 <= last or last <= 0 <= first:
        return 0
    negative_count = count if max(first, last) < 0 else 0
    _kernel_int(count, "length")
    _kernel_signed(first, "first factor")
    _kernel_signed(last, "last factor")
    largest = max(abs(first), abs(last))
    bits = count * max(1, largest.bit_length()) + 2
    result = _empty((bits + 31) // 32 + 1)
    length = lib().mmp_stride_product(
        first, count, stride, addr(result), len(result)
    )
    sign = -1 if negative_count & 1 else 1
    return _value(result, length, sign)


def rf(x, n):
    """Rising factorial for integer x and nonnegative integer n."""
    start = _integer(x, "x")
    count = _integer(n, "n")
    return _stride_value(start, count, 1)


def ff(x, n):
    """Falling factorial for integer x and nonnegative integer n."""
    start = _integer(x, "x")
    count = _integer(n, "n")
    return _stride_value(start - count + 1, count, 1)


class MPContext:
    """Small mpmath-shaped context for the exact integer kernel subset."""

    def __init__(self) -> None:
        self.dps = 15
        self.prec = 53

    fadd = staticmethod(fadd)
    fsub = staticmethod(fsub)
    fmul = staticmethod(fmul)
    power = staticmethod(power)
    fac = staticmethod(fac)
    factorial = staticmethod(factorial)
    fac2 = staticmethod(fac2)
    fib = staticmethod(fib)
    fibonacci = staticmethod(fibonacci)
    binomial = staticmethod(binomial)
    rf = staticmethod(rf)
    ff = staticmethod(ff)


mp = MPContext()
