"""Unsigned base-2**32 arithmetic and exact integer special functions."""

from std.sys import simd_width_of

comptime LimbPtr = Pointer[UInt32, AnyOrigin[mut=True]]
comptime MASK32: UInt64 = 0xFFFFFFFF
comptime MAX_U64: UInt64 = 0xFFFFFFFFFFFFFFFF
comptime SIMD_WIDTH = simd_width_of[DType.float64]()
comptime KARATSUBA_GENERAL_LIMBS = 16
comptime KARATSUBA_REPEATED_LIMBS = 32
comptime PARALLEL_FIBONACCI_LIMBS = 4096


def limbs(addr: Int) -> LimbPtr:
    return LimbPtr(unsafe_from_address=addr)


def normalized_length(value: LimbPtr, length: Int) -> Int:
    var n = max(length, 1)
    while n > 1 and value[unsafe_offset=n - 1] == 0:
        n -= 1
    return n


def copy_limbs(source: LimbPtr, destination: LimbPtr, length: Int):
    var i = 0
    while i + SIMD_WIDTH <= length:
        destination.unsafe_store(i, source.unsafe_load[width=SIMD_WIDTH](i))
        i += SIMD_WIDTH
    while i < length:
        destination[unsafe_offset=i] = source[unsafe_offset=i]
        i += 1


def clear_limbs(destination: LimbPtr, length: Int):
    var zeros = SIMD[DType.uint32, SIMD_WIDTH](0)
    var i = 0
    while i + SIMD_WIDTH <= length:
        destination.unsafe_store(i, zeros)
        i += SIMD_WIDTH
    while i < length:
        destination[unsafe_offset=i] = 0
        i += 1


def compare_abs(a: LimbPtr, an_in: Int, b: LimbPtr, bn_in: Int) -> Int:
    var an = normalized_length(a, an_in)
    var bn = normalized_length(b, bn_in)
    if an < bn:
        return -1
    if an > bn:
        return 1
    for offset in range(an):
        var i = an - 1 - offset
        if a[unsafe_offset=i] < b[unsafe_offset=i]:
            return -1
        if a[unsafe_offset=i] > b[unsafe_offset=i]:
            return 1
    return 0


def add_abs(
    a: LimbPtr,
    an: Int,
    b: LimbPtr,
    bn: Int,
    destination: LimbPtr,
    capacity: Int,
) -> Int:
    var count = max(an, bn)
    var carry = UInt64(0)
    for i in range(count):
        var av = UInt64(a[unsafe_offset=i]) if i < an else UInt64(0)
        var bv = UInt64(b[unsafe_offset=i]) if i < bn else UInt64(0)
        var total = av + bv + carry
        destination[unsafe_offset=i] = UInt32(total & MASK32)
        carry = total >> 32
    if carry != 0 and count < capacity:
        destination[unsafe_offset=count] = UInt32(carry)
        count += 1
    return normalized_length(destination, count)


def sub_abs(
    a: LimbPtr, an: Int, b: LimbPtr, bn: Int, destination: LimbPtr
) -> Int:
    var borrow = UInt64(0)
    for i in range(an):
        var av = UInt64(a[unsafe_offset=i])
        var bv = (UInt64(b[unsafe_offset=i]) if i < bn else UInt64(0)) + borrow
        if av >= bv:
            destination[unsafe_offset=i] = UInt32(av - bv)
            borrow = 0
        else:
            destination[unsafe_offset=i] = UInt32((UInt64(1) << 32) + av - bv)
            borrow = 1
    return normalized_length(destination, an)


def multiply_abs(
    a: LimbPtr,
    an: Int,
    b: LimbPtr,
    bn: Int,
    destination: LimbPtr,
) -> Int:
    var count = an + bn
    clear_limbs(destination, count)
    for i in range(an):
        var carry = UInt64(0)
        var limb = UInt64(a[unsafe_offset=i])
        var j = 0
        while j + 1 < bn:
            var total = (
                UInt64(destination[unsafe_offset=i + j])
                + limb * UInt64(b[unsafe_offset=j])
                + carry
            )
            destination[unsafe_offset=i + j] = UInt32(total & MASK32)
            carry = total >> 32
            total = (
                UInt64(destination[unsafe_offset=i + j + 1])
                + limb * UInt64(b[unsafe_offset=j + 1])
                + carry
            )
            destination[unsafe_offset=i + j + 1] = UInt32(total & MASK32)
            carry = total >> 32
            j += 2
        if j < bn:
            var total = (
                UInt64(destination[unsafe_offset=i + j])
                + limb * UInt64(b[unsafe_offset=j])
                + carry
            )
            destination[unsafe_offset=i + j] = UInt32(total & MASK32)
            carry = total >> 32
        destination[unsafe_offset=i + bn] = UInt32(carry)
    return normalized_length(destination, count)


def add_shifted_in_place(
    destination: LimbPtr,
    destination_length: Int,
    value: LimbPtr,
    value_length: Int,
    shift: Int,
):
    var carry = UInt64(0)
    for i in range(value_length):
        var total = (
            UInt64(destination[unsafe_offset=shift + i])
            + UInt64(value[unsafe_offset=i])
            + carry
        )
        destination[unsafe_offset=shift + i] = UInt32(total & MASK32)
        carry = total >> 32
    var i = shift + value_length
    while carry != 0 and i < destination_length:
        var total = UInt64(destination[unsafe_offset=i]) + carry
        destination[unsafe_offset=i] = UInt32(total & MASK32)
        carry = total >> 32
        i += 1


def multiply_fast[
    karatsuba_limbs: Int
](
    a: LimbPtr,
    an: Int,
    b: LimbPtr,
    bn: Int,
    destination: LimbPtr,
    scratch: LimbPtr,
) -> Int:
    var largest = max(an, bn)
    var smallest = min(an, bn)
    if smallest < karatsuba_limbs or smallest + smallest < largest:
        return multiply_abs(a, an, b, bn, destination)

    var half = (largest + 1) // 2
    var a_low_length = min(an, half)
    var b_low_length = min(bn, half)
    var a_high_length = an - a_low_length
    var b_high_length = bn - b_low_length
    if a_high_length == 0 or b_high_length == 0:
        return multiply_abs(a, an, b, bn, destination)

    var sum_capacity = half + 1
    var sum_a = scratch
    var sum_b = scratch.unsafe_offset(sum_capacity)
    var middle = scratch.unsafe_offset(sum_capacity + sum_capacity)
    var middle_capacity = sum_capacity + sum_capacity
    var child_scratch = middle.unsafe_offset(middle_capacity)

    var count = an + bn
    clear_limbs(destination, count)
    var low_length = multiply_fast[karatsuba_limbs](
        a,
        a_low_length,
        b,
        b_low_length,
        destination,
        child_scratch,
    )
    var high_length = multiply_fast[karatsuba_limbs](
        a.unsafe_offset(a_low_length),
        a_high_length,
        b.unsafe_offset(b_low_length),
        b_high_length,
        destination.unsafe_offset(half + half),
        child_scratch,
    )
    var sum_a_length = add_abs(
        a,
        a_low_length,
        a.unsafe_offset(a_low_length),
        a_high_length,
        sum_a,
        sum_capacity,
    )
    var sum_b_length = add_abs(
        b,
        b_low_length,
        b.unsafe_offset(b_low_length),
        b_high_length,
        sum_b,
        sum_capacity,
    )
    var middle_length = multiply_fast[karatsuba_limbs](
        sum_a,
        sum_a_length,
        sum_b,
        sum_b_length,
        middle,
        child_scratch,
    )
    middle_length = sub_abs(
        middle, middle_length, destination, low_length, middle
    )
    middle_length = sub_abs(
        middle,
        middle_length,
        destination.unsafe_offset(half + half),
        high_length,
        middle,
    )
    add_shifted_in_place(destination, count, middle, middle_length, half)
    return normalized_length(destination, count)


def multiply_u64_in_place(
    value: LimbPtr, length: Int, factor: UInt64, capacity: Int
) -> Int:
    if factor == 0:
        value[unsafe_offset=0] = 0
        return 1
    var carry = UInt64(0)
    var factor_low = factor & MASK32
    var factor_high = factor >> 32
    for i in range(length):
        var limb = UInt64(value[unsafe_offset=i])
        var product_low = limb * factor_low
        var low_sum = (product_low & MASK32) + (carry & MASK32)
        value[unsafe_offset=i] = UInt32(low_sum & MASK32)
        carry = (
            (product_low >> 32)
            + (carry >> 32)
            + (low_sum >> 32)
            + limb * factor_high
        )
    var count = length
    while carry != 0 and count < capacity:
        value[unsafe_offset=count] = UInt32(carry & MASK32)
        carry >>= 32
        count += 1
    if carry != 0:
        return -1
    return count


def divide_small_in_place(value: LimbPtr, length: Int, divisor: UInt64) -> Int:
    var remainder = UInt64(0)
    for offset in range(length):
        var i = length - 1 - offset
        var current = (remainder << 32) | UInt64(value[unsafe_offset=i])
        value[unsafe_offset=i] = UInt32(current // divisor)
        remainder = current % divisor
    return normalized_length(value, length)


@export("mmp_compare_abs")
def mmp_compare_abs(a_addr: Int, an: Int, b_addr: Int, bn: Int) abi("C") -> Int:
    if a_addr == 0 or b_addr == 0 or an <= 0 or bn <= 0:
        return -1
    return compare_abs(limbs(a_addr), an, limbs(b_addr), bn)


@export("mmp_add_abs")
def mmp_add_abs(
    a_addr: Int,
    an: Int,
    b_addr: Int,
    bn: Int,
    destination_addr: Int,
    capacity: Int,
) abi("C") -> Int:
    if (
        a_addr == 0
        or b_addr == 0
        or destination_addr == 0
        or an <= 0
        or bn <= 0
        or capacity < max(an, bn) + 1
    ):
        return -1
    return add_abs(
        limbs(a_addr),
        an,
        limbs(b_addr),
        bn,
        limbs(destination_addr),
        capacity,
    )


@export("mmp_sub_abs")
def mmp_sub_abs(
    a_addr: Int, an: Int, b_addr: Int, bn: Int, destination_addr: Int
) abi("C") -> Int:
    if (
        a_addr == 0
        or b_addr == 0
        or destination_addr == 0
        or an <= 0
        or bn <= 0
        or bn > an
    ):
        return -1
    return sub_abs(
        limbs(a_addr), an, limbs(b_addr), bn, limbs(destination_addr)
    )


@export("mmp_mul_abs")
def mmp_mul_abs(
    a_addr: Int,
    an: Int,
    b_addr: Int,
    bn: Int,
    destination_addr: Int,
    capacity: Int,
    scratch_addr: Int,
    scratch_capacity: Int,
) abi("C") -> Int:
    if (
        a_addr == 0
        or b_addr == 0
        or destination_addr == 0
        or scratch_addr == 0
        or an <= 0
        or bn <= 0
        or capacity < an + bn
        or scratch_capacity < 4 * capacity + 64
    ):
        return -1
    return multiply_fast[KARATSUBA_GENERAL_LIMBS](
        limbs(a_addr),
        an,
        limbs(b_addr),
        bn,
        limbs(destination_addr),
        limbs(scratch_addr),
    )


@export("mmp_power")
def mmp_power(
    base_addr: Int,
    base_length: Int,
    exponent: Int,
    result_addr: Int,
    current_addr: Int,
    temporary_addr: Int,
    capacity: Int,
    scratch_addr: Int,
    scratch_capacity: Int,
) abi("C") -> Int:
    if (
        base_addr == 0
        or result_addr == 0
        or current_addr == 0
        or temporary_addr == 0
        or scratch_addr == 0
        or base_length <= 0
        or exponent < 0
        or capacity < base_length
        or scratch_capacity < 4 * capacity + 64
    ):
        return -1
    var base = limbs(base_addr)
    var result = limbs(result_addr)
    var current = limbs(current_addr)
    var temporary = limbs(temporary_addr)
    var scratch = limbs(scratch_addr)
    result[unsafe_offset=0] = 1
    var result_length = 1
    copy_limbs(base, current, base_length)
    var current_length = base_length
    var remaining = exponent
    while remaining > 0:
        if remaining & 1:
            var next_length = multiply_fast[KARATSUBA_REPEATED_LIMBS](
                result,
                result_length,
                current,
                current_length,
                temporary,
                scratch,
            )
            copy_limbs(temporary, result, next_length)
            result_length = next_length
        remaining >>= 1
        if remaining > 0:
            var next_length = multiply_fast[KARATSUBA_REPEATED_LIMBS](
                current,
                current_length,
                current,
                current_length,
                temporary,
                scratch,
            )
            copy_limbs(temporary, current, next_length)
            current_length = next_length
    return result_length


@export("mmp_factorial")
def mmp_factorial(n: Int, destination_addr: Int, capacity: Int) abi("C") -> Int:
    if n < 0 or destination_addr == 0 or capacity <= 0:
        return -1
    var destination = limbs(destination_addr)
    destination[unsafe_offset=0] = 1
    var length = 1
    var factor = 2
    while factor <= n:
        var batch = UInt64(1)
        while factor <= n and batch <= MAX_U64 // UInt64(factor):
            batch *= UInt64(factor)
            factor += 1
        length = multiply_u64_in_place(destination, length, batch, capacity)
        if length < 0:
            return length
    return length


@export("mmp_stride_product")
def mmp_stride_product(
    first: Int,
    count: Int,
    stride: Int,
    destination_addr: Int,
    capacity: Int,
) abi("C") -> Int:
    if (
        count < 0
        or destination_addr == 0
        or capacity <= 0
        or (
            count > 0
            and stride != 0
            and (first + (count - 1) * stride - first) // stride != count - 1
        )
    ):
        return -1
    var destination = limbs(destination_addr)
    destination[unsafe_offset=0] = 1
    var length = 1
    for i in range(count):
        var factor = first + i * stride
        if factor == 0:
            destination[unsafe_offset=0] = 0
            return 1
        var magnitude = -factor if factor < 0 else factor
        length = multiply_u64_in_place(
            destination, length, UInt64(magnitude), capacity
        )
        if length < 0:
            return length
    return length


@export("mmp_binomial")
def mmp_binomial(
    n: Int, k: Int, destination_addr: Int, capacity: Int
) abi("C") -> Int:
    if n < 0 or k < 0 or k > n or destination_addr == 0 or capacity <= 0:
        return -1
    var destination = limbs(destination_addr)
    destination[unsafe_offset=0] = 1
    var length = 1
    var reduced_k = min(k, n - k)
    for i in range(1, reduced_k + 1):
        length = multiply_u64_in_place(
            destination, length, UInt64(n - reduced_k + i), capacity
        )
        if length < 0:
            return length
        length = divide_small_in_place(destination, length, UInt64(i))
    return length


@export("mmp_fibonacci")
def mmp_fibonacci(
    n: Int,
    destination_addr: Int,
    next_addr: Int,
    c_addr: Int,
    d_addr: Int,
    work_addr: Int,
    work2_addr: Int,
    capacity: Int,
    scratch_addr: Int,
    scratch_stride: Int,
    scratch_capacity: Int,
) abi("C") -> Int:
    if (
        n < 0
        or destination_addr == 0
        or next_addr == 0
        or c_addr == 0
        or d_addr == 0
        or work_addr == 0
        or work2_addr == 0
        or scratch_addr == 0
        or capacity <= 0
        or scratch_stride < 4 * capacity + 64
        or scratch_capacity < 3 * scratch_stride
    ):
        return -1
    var a = limbs(destination_addr)
    var b = limbs(next_addr)
    var c = limbs(c_addr)
    var d = limbs(d_addr)
    var work = limbs(work_addr)
    var work2 = limbs(work2_addr)
    var scratch = limbs(scratch_addr)
    a[unsafe_offset=0] = 0
    b[unsafe_offset=0] = 1
    var an = 1
    var bn = 1
    var started = False
    for offset in range(63):
        var bit_index = 62 - offset
        var bit = (n >> bit_index) & 1
        if not started and bit == 0:
            continue
        started = True

        var work_length = add_abs(b, bn, b, bn, work, capacity)
        var work2_length = sub_abs(work, work_length, a, an, work2)
        var cn: Int
        var dn: Int
        if max(an, bn) >= PARALLEL_FIBONACCI_LIMBS:
            _ = multiply_fast[KARATSUBA_REPEATED_LIMBS](
                a, an, work2, work2_length, c, scratch
            )
            _ = multiply_fast[KARATSUBA_REPEATED_LIMBS](
                a, an, a, an, d, scratch.unsafe_offset(scratch_stride)
            )
            _ = multiply_fast[KARATSUBA_REPEATED_LIMBS](
                b,
                bn,
                b,
                bn,
                work,
                scratch.unsafe_offset(scratch_stride + scratch_stride),
            )
            cn = normalized_length(c, an + work2_length)
            dn = normalized_length(d, an + an)
            work_length = normalized_length(work, bn + bn)
        else:
            cn = multiply_fast[KARATSUBA_REPEATED_LIMBS](
                a, an, work2, work2_length, c, scratch
            )
            dn = multiply_fast[KARATSUBA_REPEATED_LIMBS](
                a, an, a, an, d, scratch
            )
            work_length = multiply_fast[KARATSUBA_REPEATED_LIMBS](
                b, bn, b, bn, work, scratch
            )
        dn = add_abs(d, dn, work, work_length, work2, capacity)
        copy_limbs(work2, d, dn)

        if bit == 0:
            copy_limbs(c, a, cn)
            copy_limbs(d, b, dn)
            an = cn
            bn = dn
        else:
            work_length = add_abs(c, cn, d, dn, work, capacity)
            copy_limbs(d, a, dn)
            copy_limbs(work, b, work_length)
            an = dn
            bn = work_length
    return an
