from __future__ import annotations

import math
import random

import mpmath
import numpy as np
import pytest

import mojo_mpmath as mm
from mojo_mpmath._lib import addr, lib


def exact_mpmath(function, *args, bits: int):
    with mpmath.workprec(bits + 20):
        return int(function(*args))


@pytest.mark.parametrize("bits", [1, 31, 32, 33, 63, 64, 127, 256, 1024])
def test_exact_arithmetic_matches_python_and_mpmath(bits):
    rng = random.Random(bits)
    for _ in range(8):
        a = rng.getrandbits(bits) * rng.choice((-1, 1))
        b = rng.getrandbits(bits) * rng.choice((-1, 1))
        assert mm.fadd(a, b, exact=True) == a + b
        assert mm.fsub(a, b, exact=True) == a - b
        product = mm.fmul(a, b, exact=True)
        assert product == a * b
        assert product == int(mpmath.fmul(a, b, exact=True))


@pytest.mark.parametrize(
    ("a_limbs", "b_limbs"),
    [(3, 5), (15, 17), (16, 16), (17, 17), (33, 65)],
)
def test_multiplication_simd_tail_and_karatsuba_boundaries(a_limbs, b_limbs):
    a = (1 << (32 * a_limbs - 1)) + (1 << (17 * a_limbs)) + 12345
    b = (1 << (32 * b_limbs - 1)) + (1 << (13 * b_limbs)) + 67890
    assert mm.fmul(a, b, exact=True) == a * b


@pytest.mark.parametrize(
    ("base", "exponent"),
    [(0, 0), (0, 9), (1, 1000), (-2, 127), (3, 333), (2**80 + 7, 19)],
)
def test_power_parity(base, exponent):
    expected = base**exponent
    assert mm.power(base, exponent) == expected
    assert mm.mp.power(base, exponent) == expected


@pytest.mark.parametrize("n", [0, 1, 2, 10, 100, 1000, 5000])
def test_factorial_parity(n):
    result = mm.fac(n)
    assert result == math.factorial(n)
    assert result == exact_mpmath(mpmath.fac, n, bits=result.bit_length())
    assert mm.factorial(n) == result
    assert mm.mp.fac(n) == result


@pytest.mark.parametrize("n", [-1, 0, 1, 2, 9, 100, 999])
def test_double_factorial_parity(n):
    result = mm.fac2(n)
    expected = math.prod(range(n if n > 0 else 1, 0, -2))
    assert result == expected
    assert result == exact_mpmath(mpmath.fac2, n, bits=result.bit_length())


@pytest.mark.parametrize("n", [-10000, -100, -2, -1, 0, 1, 2, 100, 10000])
def test_fibonacci_parity(n):
    result = mm.fib(n)
    expected = exact_mpmath(
        mpmath.fib, n, bits=max(1, int(abs(n) * 0.7) + 20)
    )
    assert result == expected
    assert mm.fibonacci(n) == result
    assert mm.mp.fib(n) == result


def test_fibonacci_parallel_threshold():
    n = 400_000
    a, b = 0, 1
    for bit in bin(n)[2:]:
        c = a * ((b << 1) - a)
        d = a * a + b * b
        a, b = (d, c + d) if bit == "1" else (c, d)
    assert mm.fib(n) == a


@pytest.mark.parametrize(
    ("n", "k"),
    [
        (0, 0),
        (10, -1),
        (10, 0),
        (10, 3),
        (100, 50),
        (1000, 500),
        (100000, 100),
        (-10, 3),
        (-1000, 40),
    ],
)
def test_binomial_parity(n, k):
    result = mm.binomial(n, k)
    expected = exact_mpmath(
        mpmath.binomial, n, k, bits=max(1, abs(result).bit_length())
    )
    assert result == expected
    assert mm.mp.binomial(n, k) == result


@pytest.mark.parametrize(
    ("x", "n"),
    [(3, 0), (3, 4), (-3, 5), (-10, 3), (100, 100), (10**12 + 39, 80)],
)
def test_rising_and_falling_factorial_parity(x, n):
    rising = mm.rf(x, n)
    falling = mm.ff(x, n)
    assert rising == exact_mpmath(
        mpmath.rf, x, n, bits=max(1, abs(rising).bit_length())
    )
    assert falling == exact_mpmath(
        mpmath.ff, x, n, bits=max(1, abs(falling).bit_length())
    )


def test_aliases_and_context_surface():
    assert mm.fac is mm.factorial
    assert mm.fib is mm.fibonacci
    assert mm.mp.factorial(20) == mm.fac(20)
    assert mm.mp.fibonacci(100) == mm.fib(100)
    assert mm.mp.fadd(2**200, 1) == 2**200 + 1
    assert mm.mp.fsub(1, 2) == -1
    assert mm.mp.fmul(2**100, 3) == 3 * 2**100
    assert mm.mp.power(7, 8) == 7**8
    assert mm.mp.fac2(15) == mm.fac2(15)
    assert mm.mp.rf(-8, 4) == mm.rf(-8, 4)
    assert mm.mp.ff(8, 4) == mm.ff(8, 4)


def test_documented_compatibility_keywords_remain_exact():
    assert mm.fadd(5, 7, exact=True) == 12
    assert mm.fsub(5, 7, prec=2) == -2
    assert mm.fmul(5, 7, dps=1) == 35
    assert mm.fac(20, prec=2) == math.factorial(20)
    assert mm.fib(100, dps=1) == exact_mpmath(mpmath.fib, 100, bits=100)


@pytest.mark.parametrize(
    "call",
    [
        lambda: mm.fac(-1),
        lambda: mm.power(2, -1),
        lambda: mm.rf(3, -1),
        lambda: mm.ff(3, -1),
        lambda: mm.fib(1.5),
        lambda: mm.binomial(4.5, 2),
    ],
)
def test_domain_errors(call):
    with pytest.raises((TypeError, ValueError)):
        call()


@pytest.mark.parametrize(
    "call",
    [
        lambda: mm.fac(2**63),
        lambda: mm.fib(-(2**63)),
        lambda: mm.power(2, 2**63),
        lambda: mm.rf(2**63, 2),
        lambda: mm.ff(-(2**63), 2),
        lambda: mm.binomial(-(2**63), 1),
    ],
)
def test_kernel_integer_narrowing_is_rejected(call):
    with pytest.raises(OverflowError):
        call()


def test_ffi_array_contract_rejects_wrong_layout_and_dtype():
    with pytest.raises(TypeError):
        addr(np.ones(2, dtype=np.uint64))
    with pytest.raises(TypeError):
        addr(np.ones(4, dtype=np.uint32)[::2])
    with pytest.raises(TypeError):
        addr(np.ones((1, 2), dtype=np.uint32))
    with pytest.raises(TypeError):
        addr(np.empty(0, dtype=np.uint32))


def test_exported_ffi_rejects_null_and_invalid_capacities():
    limbs = np.ones(2, dtype=np.uint32)
    scratch = np.empty(80, dtype=np.uint32)
    assert lib().mmp_add_abs(addr(limbs), 2, addr(limbs), 2, 0, 3) < 0
    assert (
        lib().mmp_mul_abs(
            addr(limbs), 2, addr(limbs), 2, addr(limbs), 2, addr(scratch), 80
        )
        < 0
    )
