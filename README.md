# mojo-mpmath

`mojo-mpmath` is a standalone Mojo implementation of the exact-integer,
arbitrary-precision core used by a focused subset of
[mpmath](https://mpmath.org/). Python owns the numbers and the allocation;
compiled Mojo kernels perform the limb arithmetic.

The package is useful when a calculation stays in the integer-valued part of
mpmath's API, particularly generalized binomial coefficients and long rising
or falling factorials. It returns ordinary Python `int` values, so results are
exact rather than rounded to the active `mp.dps`.

## Covered API

The Python module is `mojo_mpmath`. It provides these mpmath-compatible names
at module level and on `mojo_mpmath.mp`:

- exact integer `fadd`, `fsub`, and `fmul`;
- nonnegative integral `power`;
- `fac` / `factorial` and `fac2`;
- `fib` / `fibonacci`, including negative integer indices;
- generalized integer `binomial`, including negative upper indices;
- integer `rf` and `ff` for nonnegative lengths.

The common integer call signatures and aliases mirror mpmath. Optional keyword
arguments accepted by `fadd`, `fsub`, `fmul`, `fac`, and `fib` are accepted for
source compatibility, but the result remains exact.

This is not yet a general replacement for mpmath. It does not implement `mpf`
or `mpc`, division, roots, arbitrary real/complex arguments, transcendental
functions, matrices, interval arithmetic, or mpmath's precision-dependent
rounding modes. Passing a non-integer into the covered functions raises
`TypeError`; negative exponents and negative `rf`/`ff` lengths are outside the
covered exact-integer domain.

## Install

```bash
pixi install
pixi run build
pixi run test
```

The build produces `dist/libmojo-mpmath.so`. The Python binding also rebuilds
it on first use when the Mojo source is newer.

## Usage

```python
import mojo_mpmath as mm

assert mm.fac(100) == 100 * mm.fac(99)
assert mm.fib(1000) == mm.mp.fibonacci(1000)
assert mm.binomial(100_000, 500) == mm.binomial(100_000, 99_500)

x = 2**500 + 1
y = 2**400 - 1
assert mm.fmul(x, y, exact=True) == x * y
```

Run it inside the environment with:

```bash
pixi run python -c "import mojo_mpmath as mm; print(mm.binomial(1000, 500))"
```

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux x86-64. These are best-of-five wall times from the same process using
mpmath 1.4.1's pure-Python backend. Both implementations return the same exact
integer. The benchmark clears mpmath's factorial/Fibonacci result cache before
each corresponding run so it measures computation rather than a memoized
lookup.

| case | mojo-mpmath | mpmath | relative |
| --- | ---: | ---: | ---: |
| `fmul` (8192-bit x 8192-bit) | 0.059 ms | 0.043 ms | 0.73x slower |
| `power` (256-bit base, exponent 100) | 0.179 ms | 0.203 ms | 1.14x faster |
| `fac` (10000!, exact precision) | 7.588 ms | 8.838 ms | 1.16x faster |
| `fib` (index 100000, exact precision) | 1.717 ms | 3.736 ms | 2.18x faster |
| `fib` (index 400000, parallel threshold) | 17.894 ms | 37.100 ms | 2.07x faster |
| `binomial` (100000 choose 500) | 0.692 ms | 17.088 ms | 24.70x faster |
| `rf` (10^12+39, 1000) | 1.190 ms | 1380.880 ms | 1160.14x faster |

Large balanced products use thresholded Karatsuba multiplication, while small
and unbalanced products stay on the lower-overhead schoolbook kernel. Power
and Fibonacci reuse caller-owned scratch space. Factorial batches adjacent
factors into exact 64-bit products, reducing full accumulator passes.
Fibonacci evaluates its three independent doubling products in parallel only
once operands reach 4096 limbs; smaller inputs stay serial.

There is intentionally no GPU path. The hot limb loops are carry-dependent and
have low arithmetic intensity, so device transfer and launch overhead are not
justified. CPU remains the only execution device, and no GPU runtime dependency
is required.

Benchmark numbers are machine-specific. Run `pixi run bench` to measure the
current checkout and machine; the Pixi task holds a machine-wide lock.

## How it works

Magnitudes cross the C ABI as contiguous little-endian NumPy `uint32` arrays.
Input arrays are zero-copy views over Python's little-endian byte buffers. The
Python layer estimates the required output capacity, allocates each result and
reusable scratch buffer, and passes their addresses as 64-bit integers through
`ctypes`. Mojo reconstructs
`UnsafePointer[UInt32, AnyOrigin[mut=True]]` values inside non-parametric
`@export` functions.

The kernels implement carry-safe addition and subtraction, SIMD copy and clear
loops with scalar tails, schoolbook and Karatsuba multiplication,
exponentiation by squaring, fast-doubling Fibonacci, exact small-divisor
binomial recurrence, and 64-bit-factor accumulator products. The 64-bit-factor
path splits each factor into high and low 32-bit halves so a mathematical
96-bit intermediate never overflows a Mojo `UInt64`. Mojo never allocates or
retains memory across a call.

Parity tests compare every covered function with real upstream mpmath at
enough precision to preserve the integer, plus Python's exact integer
arithmetic for randomized limb-boundary cases.

MIT licensed.
