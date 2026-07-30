"""Benchmark exact Mojo integer kernels against upstream mpmath."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import mpmath

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"
    ),
)

import mojo_mpmath as mm  # noqa: E402


def best_time(function, repeat: int = 5) -> float:
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
            for line in cpuinfo:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def checked(ours, upstream):
    expected = int(upstream())
    actual = ours()
    if actual != expected:
        raise AssertionError("benchmark implementations returned different results")
    return ours, upstream


def uncached_factorial(n: int, bits: int):
    mpmath.libmp.ifac.cache_clear()
    return mpmath.fac(n, prec=bits)


def uncached_fibonacci(n: int, bits: int):
    mpmath.libmp.ifib.cache_clear()
    return mpmath.fib(n, prec=bits)


def uncached_binomial(n: int, k: int):
    mpmath.libmp.ifac.cache_clear()
    return mpmath.binomial(n, k)


def cases():
    a = (1 << 8191) + (1 << 4097) + 0x123456789ABCDEF
    b = (1 << 8167) + (1 << 2029) + 0xFEDCBA987654321
    yield (
        "fmul (8192-bit x 8192-bit)",
        *checked(
            lambda: mm.fmul(a, b, exact=True),
            lambda: mpmath.fmul(a, b, exact=True),
        ),
    )

    base = (1 << 255) + 0x123456789ABCDEF
    exponent = 100
    power_bits = base.bit_length() * exponent + 8
    old_precision = mpmath.mp.prec
    mpmath.mp.prec = power_bits
    try:
        yield (
            "power (256-bit base, exponent 100)",
            *checked(
                lambda: mm.power(base, exponent),
                lambda: mpmath.power(base, exponent),
            ),
        )
    finally:
        mpmath.mp.prec = old_precision

    n = 10_000
    factorial_bits = math.factorial(n).bit_length()
    yield (
        "fac (10000!, exact precision)",
        *checked(
            lambda: mm.fac(n),
            lambda: uncached_factorial(n, factorial_bits + 8),
        ),
    )

    n = 100_000
    fibonacci_bits = int(n * 0.7) + 32
    yield (
        "fib (index 100000, exact precision)",
        *checked(
            lambda: mm.fib(n),
            lambda: uncached_fibonacci(n, fibonacci_bits),
        ),
    )

    n = 400_000
    fibonacci_bits = int(n * 0.7) + 32
    yield (
        "fib (index 400000, parallel threshold)",
        *checked(
            lambda: mm.fib(n),
            lambda: uncached_fibonacci(n, fibonacci_bits),
        ),
    )

    n, k = 100_000, 500
    binomial_bits = math.comb(n, k).bit_length()
    old_precision = mpmath.mp.prec
    mpmath.mp.prec = binomial_bits + 8
    try:
        yield (
            "binomial (100000 choose 500)",
            *checked(
                lambda: mm.binomial(n, k),
                lambda: uncached_binomial(n, k),
            ),
        )
    finally:
        mpmath.mp.prec = old_precision

    x, count = 10**12 + 39, 1_000
    rising_bits = sum((x + i).bit_length() for i in range(count)) + 8
    old_precision = mpmath.mp.prec
    mpmath.mp.prec = rising_bits
    try:
        yield (
            "rf (10^12+39, 1000)",
            *checked(
                lambda: mm.rf(x, count),
                lambda: mpmath.rf(x, count),
            ),
        )
    finally:
        mpmath.mp.prec = old_precision


def main() -> None:
    print(f"Machine: {cpu_name()} ({platform.system()} {platform.machine()})")
    print(f"Comparator: mpmath {mpmath.__version__} pure-Python backend")
    print()
    print("| case | mojo-mpmath | mpmath | relative |")
    print("| --- | ---: | ---: | ---: |")
    for name, ours, upstream in cases():
        ours()
        upstream()
        mojo_seconds = best_time(ours)
        upstream_seconds = best_time(upstream)
        relative = upstream_seconds / mojo_seconds
        label = "faster" if relative >= 1.0 else "slower"
        ratio = f"{relative:.3f}x" if relative < 0.01 else f"{relative:.2f}x"
        print(
            f"| {name} | {mojo_seconds * 1e3:.3f} ms | "
            f"{upstream_seconds * 1e3:.3f} ms | {ratio} {label} |"
        )


if __name__ == "__main__":
    main()
